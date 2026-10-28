use crate::storage::SqliteDatabase;
use ml_exp_core::operations;
use ml_exp_io::directory::{Directory, effective_uid};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde_json::{Value, json};
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Action storage I/O failed")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Storage(#[from] crate::storage::Error),
    #[error("Action database transaction failed")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Action JSON record is invalid")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Decision(#[from] operations::Error),
    #[error("authoritative Action database is missing")]
    MissingAuthoritative,
    #[error("unsupported Action database schema")]
    UnsupportedSchema,
    #[error("legacy Action requires explicit offline migration")]
    LegacyMigration,
    #[error("Action execution revision does not match durable state")]
    InvalidRevision,
    #[error("Action not found")]
    NotFound,
    #[error("idempotency key is already bound to a different operation intent")]
    IdempotencyConflict,
    #[error("Action command is already recorded with a different payload")]
    CommandConflict,
    #[error("invalid Action storage record: {0}")]
    Invalid(&'static str),
}

impl Error {
    /// A failed persistence response may follow a durable execution claim.
    /// Even false is never positive proof that a scheduler submit was absent;
    /// only the core's exact reconciliation evidence can authorize a retry.
    pub fn effects_may_have_applied(&self) -> bool {
        matches!(self, Self::Storage(error) if error.effects_may_have_applied())
    }
}

// Byte/schema-compatible with actions/database.py user_version=1. Plans remain
// immutable external files; executions and audit transitions commit together.
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS executions (
    action_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    payload TEXT NOT NULL CHECK (json_valid(payload) AND json_type(payload)='object')
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    action_id TEXT NOT NULL REFERENCES executions(action_id),
    event_key TEXT NOT NULL,
    revision INTEGER,
    payload TEXT NOT NULL CHECK (json_valid(payload) AND json_type(payload)='object'),
    UNIQUE (action_id, event_key),
    UNIQUE (action_id, revision)
);
CREATE INDEX IF NOT EXISTS events_action ON events(action_id, seq);
PRAGMA user_version=1;
"#;

#[derive(Clone, Debug)]
pub struct Database {
    directory: Directory,
    sqlite: SqliteDatabase,
    initialized: Arc<OnceLock<()>>,
}
impl Database {
    pub fn open(root: &Path) -> Result<Self> {
        Self::in_directory(Directory::open(root, true)?)
    }
    pub(crate) fn in_directory(directory: Directory) -> Result<Self> {
        directory.verify_private_parent()?;
        let sqlite =
            SqliteDatabase::in_directory(directory.clone(), OsStr::new("actions.sqlite3"))?
                .with_wal();
        Ok(Self {
            directory,
            sqlite,
            initialized: Arc::new(OnceLock::new()),
        })
    }
    fn root_directory(&self) -> Result<&Directory> {
        self.directory.verify_private_parent()?;
        Ok(&self.directory)
    }
    pub fn path(&self) -> PathBuf {
        self.sqlite.path().to_owned()
    }
    fn marker_exists(&self) -> Result<bool> {
        let directory = self.root_directory()?;
        match directory.regular_identity(OsStr::new(".sqlite-authoritative"))? {
            None => return Ok(false),
            Some(identity) if identity.is_private_for(effective_uid()) => {}
            Some(_) => return Err(Error::Invalid("single-link authoritative marker required")),
        }
        match directory.read_regular(OsStr::new(".sqlite-authoritative"), 4096) {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }
    fn initialize(&self, validate: &dyn Fn() -> Result<()>) -> Result<()> {
        validate()?;
        let directory = self.root_directory()?;
        let marker = self.marker_exists()?;
        if marker
            && directory
                .regular_identity(OsStr::new("actions.sqlite3"))?
                .is_none()
        {
            return Err(Error::MissingAuthoritative);
        }
        if self.initialized.get().is_some() {
            if !marker {
                return Err(Error::Invalid("authoritative marker disappeared"));
            }
            return Ok(());
        }
        self.sqlite.initialize(
            || {
                validate()?;
                directory.verify_attached().map_err(Error::Io)
            },
            |transaction| {
                let version: i64 =
                    transaction.query_row("PRAGMA user_version", [], |row| row.get(0))?;
                if version == 0 && !marker {
                    transaction.execute_batch(SCHEMA)?;
                } else if version != 1 {
                    return Err(Error::UnsupportedSchema);
                }
                Ok::<_, Error>(())
            },
        )?;
        if !self.marker_exists()? {
            match directory.publish_new(
                OsStr::new(".sqlite-authoritative"),
                b"actions.sqlite3 is authoritative; see docs/action-storage.md\n",
            ) {
                Ok(()) => {}
                Err(error)
                    if error.kind() == std::io::ErrorKind::AlreadyExists
                        && self.marker_exists()? => {}
                Err(error) => return Err(error.into()),
            }
        }
        directory.verify_attached()?;
        let _ = self.initialized.set(());
        Ok(())
    }
    fn transaction<T>(
        &self,
        validate: &dyn Fn() -> Result<()>,
        operation: impl FnOnce(&Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        self.initialize(validate)?;
        let directory = self.root_directory()?;
        self.sqlite.transaction(
            || {
                validate()?;
                directory.verify_attached()?;
                if !self.marker_exists()? {
                    return Err(Error::Invalid("authoritative marker disappeared"));
                }
                Ok(())
            },
            |transaction| {
                let version: i64 =
                    transaction.query_row("PRAGMA user_version", [], |row| row.get(0))?;
                if version != 1 {
                    return Err(Error::UnsupportedSchema);
                }
                operation(transaction)
            },
        )
    }

    fn read_transaction(&self, connection: &Connection, action_id: &str) -> Result<Value> {
        let row = connection
            .query_row(
                "SELECT revision,payload FROM executions WHERE action_id=?",
                [action_id],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        let Some((revision, payload)) = row else {
            let root = self.root_directory()?;
            let directory = match root.child(OsStr::new(action_id), false) {
                Ok(directory) => Some(directory),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.into()),
            };
            if let Some(directory) = directory {
                if directory.entry_exists(OsStr::new("execution.json"))?
                    || directory.entry_exists(OsStr::new("journal.jsonl"))?
                {
                    return Err(Error::LegacyMigration);
                }
            }
            return Ok(json!({}));
        };
        let value: Value = serde_json::from_str(&payload)?;
        if !value.is_object() || operations::revision(&value)? != revision {
            return Err(Error::InvalidRevision);
        }
        Ok(value)
    }

    pub fn read(&self, validate: &dyn Fn() -> Result<()>, action_id: &str) -> Result<Value> {
        operations::validate_action_id(action_id)?;
        self.transaction(validate, |connection| {
            self.read_transaction(connection, action_id)
        })
    }

    /// Explicit admission check, not a GET. Bootstrap an empty store if needed
    /// and retain every committed Action identity, including orphaned plans.
    pub(super) fn admission_action_ids(
        &self,
        validate: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<String>> {
        self.transaction(validate, |connection| {
            let mut statement = connection
                .prepare("SELECT action_id FROM executions ORDER BY action_id LIMIT 4097")?;
            let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
            let mut ids = Vec::new();
            for row in rows {
                let id = row?;
                operations::validate_action_id(&id)?;
                ids.push(id);
            }
            if ids.len() > 4096 {
                return Err(Error::Invalid("Action admission inventory exceeds limit"));
            }
            Ok(ids)
        })
    }

    /// Observe an existing execution without initialization, recovery, schema
    /// changes or a write transaction. A plan with no committed execution is
    /// incomplete evidence; a GET must never turn it into PREPARED.
    pub fn observe(&self, validate: &dyn Fn() -> Result<()>, action_id: &str) -> Result<Value> {
        operations::validate_action_id(action_id)?;
        validate()?;
        if !self.marker_exists()? {
            return Err(Error::MissingAuthoritative);
        }
        let value = self.sqlite.read_only(|connection| {
            let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
            if version != 1 {
                return Err(Error::UnsupportedSchema);
            }
            let value = self.read_transaction(connection, action_id)?;
            if value.as_object().is_none_or(|map| map.is_empty()) {
                return Err(Error::Invalid("execution has not been committed"));
            }
            Ok(value)
        })?;
        validate()?;
        if !self.marker_exists()? {
            return Err(Error::MissingAuthoritative);
        }
        Ok(value)
    }

    pub fn commit(
        &self,
        validate: &dyn Fn() -> Result<()>,
        action_id: &str,
        proposed: &Value,
        event: &Value,
        expected_status: Option<&str>,
        transition_id: &str,
    ) -> Result<Value> {
        operations::validate_action_id(action_id)?;
        self.transaction(validate, |connection| {
            let current = self.read_transaction(connection,action_id)?;
            let (execution,event) = operations::commit_transition(&current,proposed,event,expected_status,transition_id)?;
            let revision = operations::revision(&execution)?;
            connection.execute("INSERT INTO executions VALUES (?,?,?) ON CONFLICT(action_id) DO UPDATE SET revision=excluded.revision,payload=excluded.payload",params![action_id,revision,serde_json::to_string(&execution)?])?;
            connection.execute("INSERT INTO events(action_id,event_key,revision,payload) VALUES (?,?,?,?)",params![action_id,format!("transition:{transition_id}"),revision,serde_json::to_string(&event)?])?;
            Ok(execution)
        })
    }

    pub fn append_event(
        &self,
        validate: &dyn Fn() -> Result<()>,
        action_id: &str,
        event: &Value,
        event_id: &str,
    ) -> Result<()> {
        operations::validate_action_id(action_id)?;
        if event_id.is_empty() || event_id.len() > 4096 || event_id.chars().any(char::is_control) {
            return Err(Error::Invalid("event_id"));
        }
        self.transaction(validate, |connection| {
            let current = self.read_transaction(connection, action_id)?;
            let mut record = event
                .as_object()
                .cloned()
                .ok_or(Error::Invalid("journal event"))?;
            record.insert("journal_event_id".into(), json!(event_id));
            record.insert(
                "state_revision".into(),
                json!(operations::revision(&current)?),
            );
            connection.execute(
                "INSERT OR IGNORE INTO executions VALUES (?,0,'{}')",
                [action_id],
            )?;
            connection.execute(
                "INSERT OR IGNORE INTO events(action_id,event_key,payload) VALUES (?,?,?)",
                params![
                    action_id,
                    format!("event:{event_id}"),
                    serde_json::to_string(&record)?
                ],
            )?;
            Ok(())
        })
    }

    pub fn journal(
        &self,
        validate: &dyn Fn() -> Result<()>,
        action_id: &str,
        limit: usize,
    ) -> Result<Vec<Value>> {
        operations::validate_action_id(action_id)?;
        let limit: i64 = limit
            .try_into()
            .map_err(|_| Error::Invalid("journal limit"))?;
        self.transaction(validate, |connection| {
            self.read_transaction(connection, action_id)?;
            let mut statement = connection.prepare(
                "SELECT payload FROM events WHERE action_id=? ORDER BY seq DESC LIMIT ?",
            )?;
            let rows =
                statement.query_map(params![action_id, limit], |row| row.get::<_, String>(0))?;
            let mut events = Vec::new();
            for row in rows {
                events.push(serde_json::from_str(&row?)?);
            }
            events.reverse();
            Ok(events)
        })
    }
}
