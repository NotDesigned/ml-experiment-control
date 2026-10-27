use ml_exp_core::operations;
use ml_exp_io::write_atomic;
use rusqlite::{
    Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior, params,
};
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
    time::Duration,
};

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Action storage I/O failed")]
    Io(#[from] std::io::Error),
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
    root: PathBuf,
}
impl Database {
    pub fn new(root: &Path) -> Self {
        Self { root: root.into() }
    }
    pub fn path(&self) -> PathBuf {
        self.root.join("actions.sqlite3")
    }

    fn marker_exists(&self) -> Result<bool> {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        match options.open(self.root.join(".sqlite-authoritative")) {
            Ok(file) if file.metadata()?.is_file() => Ok(true),
            Ok(_) => Err(Error::Invalid(
                "authoritative marker must be a regular file",
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    fn connection(&self) -> Result<Connection> {
        let path = self.path();
        let marker = self.root.join(".sqlite-authoritative");
        let marker_exists = self.marker_exists()?;
        if marker_exists && !path.is_file() {
            return Err(Error::MissingAuthoritative);
        }
        if std::fs::symlink_metadata(&path)
            .is_ok_and(|m| m.file_type().is_symlink() || !m.is_file())
            || std::fs::symlink_metadata(&marker).is_ok_and(|m| !m.is_file())
        {
            return Err(Error::Invalid("database or marker must be a regular file"));
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        // Validate the opened descriptor, not only an earlier lstat. Keeping
        // it open also prevents the checked inode being immediately reused.
        let descriptor = options.open(&path)?;
        if !descriptor.metadata()?.is_file() {
            return Err(Error::Invalid("Action database must be a regular file"));
        }
        let mut connection = Connection::open_with_flags(
            &path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        connection.busy_timeout(Duration::from_secs(30))?;
        connection.execute_batch(
            "PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;",
        )?;
        let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if version == 0 && !marker_exists {
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            transaction.execute_batch(SCHEMA)?;
            transaction.commit()?;
        } else if version != 1 {
            return Err(Error::UnsupportedSchema);
        }
        if !self.marker_exists()? {
            match write_atomic(
                &marker,
                b"actions.sqlite3 is authoritative; see docs/action-storage.md\n",
                true,
            ) {
                Ok(()) => {}
                Err(error)
                    if error.kind() == std::io::ErrorKind::AlreadyExists
                        && self.marker_exists()? => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(connection)
    }

    fn transaction<T>(&self, operation: impl FnOnce(&Transaction<'_>) -> Result<T>) -> Result<T> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let result = operation(&transaction)?;
        transaction.commit()?;
        Ok(result)
    }

    fn read_transaction(&self, connection: &Transaction<'_>, action_id: &str) -> Result<Value> {
        let row = connection
            .query_row(
                "SELECT revision,payload FROM executions WHERE action_id=?",
                [action_id],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        let Some((revision, payload)) = row else {
            let directory = self.root.join(action_id);
            if ["execution.json", "journal.jsonl"]
                .iter()
                .any(|name| directory.join(name).exists())
            {
                return Err(Error::LegacyMigration);
            }
            return Ok(json!({}));
        };
        let value: Value = serde_json::from_str(&payload)?;
        if !value.is_object() || operations::revision(&value)? != revision {
            return Err(Error::InvalidRevision);
        }
        Ok(value)
    }

    pub fn read(&self, action_id: &str) -> Result<Value> {
        operations::validate_action_id(action_id)?;
        self.transaction(|connection| self.read_transaction(connection, action_id))
    }

    pub fn commit(
        &self,
        action_id: &str,
        proposed: &Value,
        event: &Value,
        expected_status: Option<&str>,
        transition_id: &str,
    ) -> Result<Value> {
        operations::validate_action_id(action_id)?;
        self.transaction(|connection| {
            let current = self.read_transaction(connection,action_id)?;
            let (execution,event) = operations::commit_transition(&current,proposed,event,expected_status,transition_id)?;
            let revision = operations::revision(&execution)?;
            connection.execute("INSERT INTO executions VALUES (?,?,?) ON CONFLICT(action_id) DO UPDATE SET revision=excluded.revision,payload=excluded.payload",params![action_id,revision,serde_json::to_string(&execution)?])?;
            connection.execute("INSERT INTO events(action_id,event_key,revision,payload) VALUES (?,?,?,?)",params![action_id,format!("transition:{transition_id}"),revision,serde_json::to_string(&event)?])?;
            Ok(execution)
        })
    }

    pub fn append_event(&self, action_id: &str, event: &Value, event_id: &str) -> Result<()> {
        operations::validate_action_id(action_id)?;
        if event_id.is_empty() || event_id.len() > 4096 || event_id.chars().any(char::is_control) {
            return Err(Error::Invalid("event_id"));
        }
        self.transaction(|connection| {
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

    pub fn journal(&self, action_id: &str, limit: usize) -> Result<Vec<Value>> {
        operations::validate_action_id(action_id)?;
        let limit: i64 = limit
            .try_into()
            .map_err(|_| Error::Invalid("journal limit"))?;
        self.transaction(|connection| {
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
