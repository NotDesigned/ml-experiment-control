use crate::deletion::{self, RunFence};
use ml_exp_core::index::{self, Key};
use ml_exp_io::create_private_dir;
use rusqlite::{
    Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior, params,
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs::OpenOptions, path::PathBuf, time::Duration};

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Run index I/O failed")]
    Io(#[from] std::io::Error),
    #[error("Run index SQLite operation failed")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Run index JSON is invalid")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Row(#[from] index::Error),
    #[error(transparent)]
    Fence(#[from] deletion::Error),
    #[error("Run index file must be a regular file")]
    InvalidFile,
}
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS runs (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    campaign TEXT,
    role TEXT,
    scheduler_state TEXT,
    run_dir TEXT NOT NULL,
    row_json TEXT NOT NULL,
    scanned_at REAL,
    PRIMARY KEY (project,run_id)
);
CREATE TABLE IF NOT EXISTS collector_status (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    last_poll_at REAL,
    last_verb TEXT,
    last_error TEXT,
    verb_results TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (project,run_id)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT);
"#;

#[derive(Clone, Debug)]
pub struct RunIndex {
    path: PathBuf,
    fence: RunFence,
}
impl RunIndex {
    pub fn open(path: impl Into<PathBuf>, fence: RunFence) -> Result<Self> {
        let index = Self {
            path: path.into(),
            fence,
        };
        if let Some(parent) = index
            .path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            create_private_dir(parent)?;
        }
        let guard = index.fence.shared()?;
        let mut connection = index.connection(true)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        guard.validate()?;
        transaction.execute_batch(SCHEMA)?;
        let missing = {
            let mut statement = transaction.prepare("PRAGMA table_info(collector_status)")?;
            let columns = statement
                .query_map([], |row| row.get::<_, String>(1))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            !columns.iter().any(|name| name == "verb_results")
        };
        if missing {
            transaction.execute_batch(
                "ALTER TABLE collector_status ADD COLUMN verb_results TEXT NOT NULL DEFAULT '{}';",
            )?;
        }
        guard.validate()?;
        transaction.commit()?;
        guard.validate()?;
        drop(guard);
        Ok(index)
    }
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
    fn connection(&self, create: bool) -> Result<Connection> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create(create)
            .truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let descriptor = options.open(&self.path)?;
        if !descriptor.metadata()?.is_file() {
            return Err(Error::InvalidFile);
        }
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW;
        if create {
            flags |= OpenFlags::SQLITE_OPEN_CREATE;
        }
        let connection = Connection::open_with_flags(&self.path, flags)?;
        connection.busy_timeout(Duration::from_secs(30))?;
        connection.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")?;
        Ok(connection)
    }
    fn transaction<T>(
        &self,
        validate: impl Fn() -> Result<()>,
        operation: impl FnOnce(&Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut connection = self.connection(false)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // BEGIN IMMEDIATE may wait behind another process. Its earlier fence
        // check is no longer sufficient if an ancestor or lock was replaced.
        validate()?;
        let result = operation(&transaction)?;
        validate()?;
        transaction.commit()?;
        validate()?;
        Ok(result)
    }
    /// Caller publishes any update event only after this returns true. A scan
    /// timestamp can advance without creating an update notification.
    pub fn upsert_run(&self, row: &Value) -> Result<bool> {
        let key = index::validate_row(row, None)?;
        let guard = self.fence.shared_existing()?;
        if guard.is_deleted(&key.project, &key.run_id)? {
            guard.validate()?;
            return Ok(false);
        }
        self.transaction(|| guard.require_observed_live(&key.project, &key.run_id).map_err(Error::Fence), |connection| {
            let previous=connection.query_row("SELECT row_json FROM runs WHERE project=? AND run_id=?",params![key.project,key.run_id],|record|record.get::<_,String>(0)).optional()?.map(|raw|serde_json::from_str::<Value>(&raw)).transpose()?;
            let changed=index::changed(previous.as_ref(),row)?;
            connection.execute("INSERT INTO runs(project,run_id,campaign,role,scheduler_state,run_dir,row_json,scanned_at) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(project,run_id) DO UPDATE SET campaign=excluded.campaign,role=excluded.role,scheduler_state=excluded.scheduler_state,run_dir=excluded.run_dir,row_json=excluded.row_json,scanned_at=excluded.scanned_at",params![key.project,key.run_id,row.get("campaign").and_then(Value::as_str),row.get("role").and_then(Value::as_str),row.get("scheduler_state").and_then(Value::as_str),row["run_dir"].as_str(),serde_json::to_string(row)?,row.get("scanned_at").and_then(Value::as_f64)])?;
            Ok(changed)
        })
    }
    pub fn get_run(&self, project: &str, run_id: &str) -> Result<Option<Value>> {
        let guard = self.fence.shared_existing()?;
        if guard.is_deleted(project, run_id)? {
            guard.validate()?;
            return Ok(None);
        }
        let connection = self.connection(false)?;
        let raw = connection
            .query_row(
                "SELECT row_json FROM runs WHERE project=? AND run_id=?",
                params![project, run_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let Some(raw) = raw else {
            guard.require_observed_live(project, run_id)?;
            return Ok(None);
        };
        let value: Value = serde_json::from_str(&raw)?;
        index::validate_row(
            &value,
            Some(&Key {
                project: project.into(),
                run_id: run_id.into(),
            }),
        )?;
        guard.require_observed_live(project, run_id)?;
        Ok(Some(value))
    }
    pub fn list_runs(&self, project: Option<&str>, campaign: Option<&str>) -> Result<Vec<Value>> {
        let guard = self.fence.shared_existing()?;
        let connection = self.connection(false)?;
        let query = if project.is_some() {
            "SELECT project,run_id,row_json FROM runs WHERE project=? ORDER BY run_id"
        } else {
            "SELECT project,run_id,row_json FROM runs ORDER BY run_id"
        };
        let mut statement = connection.prepare(query)?;
        let mut rows = if let Some(project) = project {
            statement.query([project])?
        } else {
            statement.query([])?
        };
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            let project: String = row.get(0)?;
            let run_id: String = row.get(1)?;
            if guard.is_deleted(&project, &run_id)? {
                continue;
            }
            let value: Value = serde_json::from_str(&row.get::<_, String>(2)?)?;
            index::validate_row(&value, Some(&Key { project, run_id }))?;
            if campaign.is_none_or(|campaign| index::matches_campaign(&value, campaign)) {
                result.push(value);
            }
        }
        guard.validate()?;
        Ok(result)
    }
    pub fn record_poll(
        &self,
        project: &str,
        run_id: &str,
        verb: &str,
        error: Option<&str>,
        now: f64,
        outcome: Option<&str>,
    ) -> Result<bool> {
        let guard = self.fence.shared_existing()?;
        if guard.is_deleted(project, run_id)? {
            guard.validate()?;
            return Ok(false);
        }
        self.transaction(|| guard.require_observed_live(project, run_id).map_err(Error::Fence), |connection| {
            let previous=connection.query_row("SELECT verb_results FROM collector_status WHERE project=? AND run_id=?",params![project,run_id],|row|row.get::<_,String>(0)).optional()?;
            // Invalid legacy poll JSON is recoverable, matching the old index;
            // canonical Run files and scientific identity are never rewritten.
            let previous=previous.and_then(|raw|serde_json::from_str::<Value>(&raw).ok()).unwrap_or_else(||json!({}));
            let status=index::poll(&previous,verb,error,now,outcome)?;
            let results=ml_exp_core::canonical::encode(&status["verb_results"],ml_exp_core::canonical::JsonOptions{compact:false,..Default::default()}).map_err(|_|index::Error::Invalid("poll encoding"))?;
            let results=String::from_utf8(results).map_err(|_|index::Error::Invalid("poll encoding"))?;
            connection.execute("INSERT INTO collector_status(project,run_id,last_poll_at,last_verb,last_error,verb_results) VALUES (?,?,?,?,?,?) ON CONFLICT(project,run_id) DO UPDATE SET last_poll_at=excluded.last_poll_at,last_verb=excluded.last_verb,last_error=excluded.last_error,verb_results=excluded.verb_results",params![project,run_id,now,verb,status["last_error"].as_str(),results])?;
            Ok(true)
        })
    }
    pub fn collector_statuses(&self, project: Option<&str>) -> Result<Vec<Value>> {
        let guard = self.fence.shared_existing()?;
        let connection = self.connection(false)?;
        let query = if project.is_some() {
            "SELECT project,run_id,last_poll_at,last_verb,last_error,verb_results FROM collector_status WHERE project=?"
        } else {
            "SELECT project,run_id,last_poll_at,last_verb,last_error,verb_results FROM collector_status"
        };
        let mut statement = connection.prepare(query)?;
        let mut rows = if let Some(project) = project {
            statement.query([project])?
        } else {
            statement.query([])?
        };
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            let project: String = row.get(0)?;
            let run_id: String = row.get(1)?;
            if guard.is_deleted(&project, &run_id)? {
                continue;
            }
            let raw: String = row.get(5)?;
            let results: Value = if raw.is_empty() {
                json!({})
            } else {
                serde_json::from_str(&raw)?
            };
            if !results.is_object() {
                return Err(index::Error::Invalid("collector verb_results").into());
            }
            result.push(json!({"project":project,"run_id":run_id,"last_poll_at":row.get::<_,Option<f64>>(2)?,"last_verb":row.get::<_,Option<String>>(3)?,"last_error":row.get::<_,Option<String>>(4)?,"verb_results":results}));
        }
        guard.validate()?;
        Ok(result)
    }
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        let guard = self.fence.shared_existing()?;
        self.transaction(|| guard.validate().map_err(Error::Fence), |connection| {connection.execute("INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",params![key,value])?;Ok(())})
    }
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let guard = self.fence.shared_existing()?;
        let value: Option<Option<String>> = self
            .connection(false)?
            .query_row("SELECT value FROM meta WHERE key=?", [key], |row| {
                row.get(0)
            })
            .optional()?;
        guard.validate()?;
        Ok(value.flatten())
    }
    /// Removes only disposable query projections. Tombstones and all canonical
    /// files/results/checkpoints stay intact; no provider is contacted.
    pub fn delete_run(&self, project: &str, run_id: &str) -> Result<bool> {
        let guard = self.fence.shared_existing()?;
        self.transaction(
            || guard.validate().map_err(Error::Fence),
            |connection| {
                let polls = connection.execute(
                    "DELETE FROM collector_status WHERE project=? AND run_id=?",
                    params![project, run_id],
                )?;
                let runs = connection.execute(
                    "DELETE FROM runs WHERE project=? AND run_id=?",
                    params![project, run_id],
                )?;
                Ok(polls > 0 || runs > 0)
            },
        )
    }
    /// A partial or failed canonical scan can never prune cached Run metadata.
    pub fn prune_project_runs(
        &self,
        project: &str,
        keep: &BTreeSet<String>,
        complete_scan: bool,
    ) -> Result<Vec<String>> {
        let guard = self.fence.shared_existing()?;
        if !complete_scan {
            guard.validate()?;
            return Ok(Vec::new());
        }
        self.transaction(
            || guard.validate().map_err(Error::Fence),
            |connection| {
                let removed = {
                    let mut statement = connection
                        .prepare("SELECT run_id FROM runs WHERE project=? ORDER BY run_id")?;
                    let existing = statement
                        .query_map([project], |row| row.get::<_, String>(0))?
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    existing
                        .into_iter()
                        .filter(|run_id| !keep.contains(run_id))
                        .collect::<Vec<_>>()
                };
                for run_id in &removed {
                    connection.execute(
                        "DELETE FROM collector_status WHERE project=? AND run_id=?",
                        params![project, run_id],
                    )?;
                    connection.execute(
                        "DELETE FROM runs WHERE project=? AND run_id=?",
                        params![project, run_id],
                    )?;
                }
                Ok(removed)
            },
        )
    }
}
