use std::path::Path;

use ml_exp_core::records::{self, Event, ImportEmission, Scope};
pub use ml_exp_core::records::{RemoteHistoryEvidence, RemoteObservation};
use rusqlite::{OptionalExtension, params};
use serde_json::{Map, Value, json};

use super::{Error, Result};
use crate::{
    deletion::{Guard, RunFence},
    storage::SqliteDatabase,
};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (project TEXT, run TEXT, route TEXT NOT NULL, config TEXT NOT NULL, PRIMARY KEY(project,run));
CREATE TABLE IF NOT EXISTS scopes (
 id TEXT PRIMARY KEY, project TEXT, run TEXT, attempt TEXT, wandb_id TEXT NOT NULL,
 confirmed INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'PENDING', error TEXT,
 last_attempt_at TEXT,last_confirmed_at TEXT,failures INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS events (scope TEXT,seq INTEGER,event_id TEXT,payload TEXT NOT NULL,dedup TEXT,
 PRIMARY KEY(scope,seq),UNIQUE(scope,event_id),UNIQUE(scope,dedup));
CREATE TABLE IF NOT EXISTS event_ids (scope TEXT,event_id TEXT,digest TEXT NOT NULL,PRIMARY KEY(scope,event_id));
CREATE TABLE IF NOT EXISTS display_metrics (destination TEXT,signature TEXT,public_name TEXT,context TEXT,
 PRIMARY KEY(destination,signature),UNIQUE(destination,public_name));
"#;

pub struct Store {
    database: SqliteDatabase,
    read_only: bool,
    fence: RunFence,
}

/// All normal journal reads and writes require this same-Store guard.
/// Its shared deletion lock remains held through each SQLite commit.
pub struct RecordGuard<'a> {
    store: &'a Store,
    held: Guard<'a>,
    project: String,
    run: String,
    // Frozen binding-derived parents, proven while the global shared fence
    // is held. No normal writer can delete a parent until this guard drops.
    names: Vec<String>,
}
impl RecordGuard<'_> {
    pub fn project(&self) -> &str {
        &self.project
    }
    pub fn run(&self) -> &str {
        &self.run
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventRow {
    pub seq: i64,
    pub event_id: String,
    pub payload: Value,
}
impl EventRow {
    pub fn document(&self) -> Value {
        json!({"seq":self.seq,"event_id":self.event_id,"payload":self.payload})
    }
}

/// Cumulative parsed input counts, independent of journal metric deduplication.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportSummary {
    pub records: u64,
    pub invalid_lines: u64,
}
impl ImportSummary {
    pub fn document(&self) -> Value {
        json!({"status":"COMPLETE","records":self.records,"invalid_lines":self.invalid_lines})
    }
}

impl Store {
    /// New workspaces receive an explicit seed from the outer bootstrap shell.
    /// Existing workspace bytes, table paths and row sequences never change.
    pub fn open(registry_root: &Path, workspace_seed: &str) -> Result<Self> {
        let store = Self::read_only(registry_root);
        let store = Self {
            read_only: false,
            ..store
        };
        let bootstrap = store.fence.shared()?;
        store.database.initialize(
            || bootstrap.validate().map_err(Error::from),
            |transaction| initialize_schema(transaction, workspace_seed),
        )?;
        bootstrap.validate()?;
        drop(bootstrap);
        Ok(store)
    }
    pub fn read_only(registry_root: &Path) -> Self {
        let database = SqliteDatabase::new(registry_root.join("tracking/outbox.sqlite"));
        Self {
            database,
            read_only: true,
            fence: RunFence::new(registry_root),
        }
    }
    pub fn path(&self) -> &Path {
        self.database.path()
    }
    fn workspace(&self) -> Result<String> {
        self.database.read_only(|connection| -> Result<_> {
            Ok(
                connection.query_row(
                    "SELECT value FROM meta WHERE key='workspace'",
                    [],
                    |row| row.get(0),
                )?,
            )
        })
    }

    pub fn workspace_id(&self, guard: &RecordGuard<'_>) -> Result<String> {
        self.require_guard(guard)?;
        let value = self.workspace()?;
        self.require_guard(guard)?;
        Ok(value)
    }

    fn binding_raw(&self, project: &str, run: &str) -> Result<Option<Value>> {
        self.database.read_only(|connection| -> Result<_> {
            let row:Option<(String,String)>=connection.query_row("SELECT route,config FROM runs WHERE project=? AND run=?",params![project,run],|row|Ok((row.get(0)?,row.get(1)?))).optional()?;
            row.map(|(route,config)|Ok(json!({"route":serde_json::from_str::<Value>(&route)?,"config":serde_json::from_str::<Value>(&config)?}))).transpose()
        })
    }

    pub fn guard(&self, project: &str, run: &str) -> Result<RecordGuard<'_>> {
        let held = self.fence.existing_observation_guard(project, run)?;
        let names = match self.binding_raw(project, run)? {
            Some(binding) => records::fence_names(run, &binding["config"])?,
            None => vec![run.to_owned()],
        };
        for name in &names {
            held.require_observed_live(project, name)?;
        }
        held.validate()?;
        Ok(RecordGuard {
            store: self,
            held,
            project: project.into(),
            run: run.into(),
            names,
        })
    }
    fn require_guard(&self, guard: &RecordGuard<'_>) -> Result<()> {
        guard.held.validate()?;
        if !std::ptr::eq(self, guard.store) {
            return Err(Error::Guard);
        }
        for name in &guard.names {
            guard.held.require_observed_live(&guard.project, name)?;
        }
        guard.held.validate()?;
        Ok(())
    }
    fn require_scope(&self, guard: &RecordGuard<'_>, scope: &str) -> Result<()> {
        self.require_guard(guard)?;
        let parsed = Scope::parse(scope)?;
        if parsed.project != guard.project || parsed.run != guard.run {
            return Err(Error::Scope);
        }
        let bound:bool=self.database.read_only(|connection| -> Result<_> {Ok(connection.query_row("SELECT EXISTS(SELECT 1 FROM scopes s JOIN runs r ON r.project=s.project AND r.run=s.run WHERE s.id=? AND s.project=? AND s.run=?)",params![scope,guard.project,guard.run],|row|row.get(0))?)})?;
        if !bound {
            return Err(Error::Scope);
        }
        guard.held.validate()?;
        Ok(())
    }
    pub fn bind(&self, guard: &RecordGuard<'_>, route: &Value, config: &Value) -> Result<()> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_guard(guard)?;
        let bind_fence_names = records::fence_names(&guard.run, config)?;
        for name in &bind_fence_names {
            guard.held.require_observed_live(&guard.project, name)?;
        }
        let route = records::encoded(route)?;
        let config = records::encoded(config)?;
        self.database.transaction(
            || self.require_guard(guard),
            |tx| {
                self.require_guard(guard)?;
                tx.execute(
                    "INSERT OR IGNORE INTO runs VALUES (?,?,?,?)",
                    params![guard.project, guard.run, route, config],
                )?;
                let frozen: (String, String) = tx.query_row(
                    "SELECT route,config FROM runs WHERE project=? AND run=?",
                    params![guard.project, guard.run],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )?;
                if frozen != (route, config) {
                    return Err(Error::FrozenIdentity);
                }
                for name in &bind_fence_names {
                    guard.held.require_observed_live(&guard.project, name)?;
                }
                self.require_guard(guard)?;
                Ok(())
            },
        )
    }

    pub fn scope(&self, guard: &RecordGuard<'_>, attempt: &str) -> Result<String> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_guard(guard)?;
        let binding = self
            .binding_raw(&guard.project, &guard.run)?
            .ok_or(Error::Unbound)?;
        let route = binding["route"]
            .as_object()
            .ok_or(Error::Invalid("frozen route object"))?;
        let scope = Scope::new(&guard.project, &guard.run, attempt).encoded()?;
        let wandb_id = records::wandb_id(&self.workspace()?, &scope, &binding["route"])?;
        self.database.transaction(
            || self.require_guard(guard),
            |tx| {
                self.require_guard(guard)?;
                tx.execute(
            "INSERT OR IGNORE INTO scopes(id,project,run,attempt,wandb_id) VALUES (?,?,?,?,?)",
            params![scope, guard.project, guard.run, attempt, wandb_id],
        )?;
                let frozen: (String, String, String) = tx.query_row(
                    "SELECT project,run,wandb_id FROM scopes WHERE id=?",
                    [&scope],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )?;
                if frozen != (guard.project.clone(), guard.run.clone(), wandb_id) {
                    return Err(Error::Scope);
                }
                if !route.get("enabled").is_some_and(truthy) {
                    tx.execute(
                        "UPDATE scopes SET status='LOCAL_ONLY' WHERE id=? AND status!='LOCAL_ONLY'",
                        [&scope],
                    )?;
                }
                self.require_guard(guard)?;
                Ok(scope)
            },
        )
    }

    pub fn append(&self, guard: &RecordGuard<'_>, scope: &str, events: &[Event]) -> Result<i64> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_scope(guard, scope)?;
        let events = events
            .iter()
            .map(records::prepare_event)
            .collect::<records::Result<Vec<_>>>()?;
        self.database.transaction(||self.require_guard(guard), |tx| {
        self.require_guard(guard)?;
        let bound:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM scopes s JOIN runs r ON r.project=s.project AND r.run=s.run WHERE s.id=? AND s.project=? AND s.run=?)",params![scope,guard.project,guard.run],|row|row.get(0))?;
        if !bound {
            return Err(Error::Scope);
        }
        let mut seq: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=?",
            [scope],
            |row| row.get(0),
        )?;
        for event in events {
            let old: Option<String> = tx
                .query_row(
                    "SELECT digest FROM event_ids WHERE scope=? AND event_id=?",
                    params![scope, event.event_id],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(old) = old {
                if old != event.checksum {
                    return Err(Error::EventConflict);
                }
                continue;
            }
            tx.execute(
                "INSERT INTO event_ids VALUES (?,?,?)",
                params![scope, event.event_id, event.checksum],
            )?;
            if let Some(dedup) = &event.dedup {
                let duplicate: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM events WHERE scope=? AND dedup=?)",
                    params![scope, dedup],
                    |row| row.get(0),
                )?;
                if duplicate {
                    continue;
                }
            }
            seq = seq
                .checked_add(1)
                .ok_or(Error::Invalid("record sequence overflow"))?;
            tx.execute(
                "INSERT INTO events VALUES (?,?,?,?,?)",
                params![scope, seq, event.event_id, event.body, event.dedup],
            )?;
        }
        tx.execute("UPDATE scopes SET status='PENDING' WHERE confirmed<? AND error IS NULL AND status NOT IN ('LOCAL_ONLY','PENDING') AND id=?",params![seq,scope])?;
        self.require_guard(guard)?;
            Ok(seq)
        })
    }

    /// Bind an historical archive to its declared schema without enabling a
    /// publisher. Existing frozen routes and schemas always take precedence.
    pub fn prepare_import(
        &self,
        guard: &RecordGuard<'_>,
        attempt: &str,
        manifest: &Value,
    ) -> Result<(String, Value)> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_guard(guard)?;
        let binding = match self.binding_raw(&guard.project, &guard.run)? {
            Some(binding) => binding,
            None => {
                let binding = records::historical_binding(manifest)?;
                self.bind(guard, &binding["route"], &binding["config"])?;
                self.binding_raw(&guard.project, &guard.run)?
                    .ok_or(Error::Unbound)?
            }
        };
        let scope = self.scope(guard, attempt)?;
        self.require_scope(guard, &scope)?;
        Ok((scope, binding["config"].clone()))
    }
    pub fn append_import(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        emission: &ImportEmission,
    ) -> Result<ImportSummary> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_scope(guard, scope)?;
        for batch in emission.events.chunks(64) {
            self.append(guard, scope, batch)?;
        }
        self.require_scope(guard, scope)?;
        Ok(ImportSummary {
            records: emission.counts.records,
            invalid_lines: emission.counts.invalid_lines,
        })
    }
    pub fn events(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        after: i64,
        limit: i64,
    ) -> Result<Vec<EventRow>> {
        self.require_scope(guard, scope)?;
        let rows = self.database.read_only(|connection| -> Result<_> {
            let mut statement = connection.prepare(
            "SELECT seq,event_id,payload FROM events WHERE scope=? AND seq>? ORDER BY seq LIMIT ?",
        )?;
            let rows = statement
                .query_map(params![scope, after, limit], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        let result = rows
            .into_iter()
            .map(|(seq, event_id, payload)| {
                Ok(EventRow {
                    seq,
                    event_id,
                    payload: serde_json::from_str(&payload)?,
                })
            })
            .collect();
        self.require_scope(guard, scope)?;
        result
    }
    pub fn total(&self, guard: &RecordGuard<'_>, scope: &str) -> Result<i64> {
        self.require_scope(guard, scope)?;
        let value = self.database.read_only(|connection| -> Result<_> {
            Ok(connection.query_row(
                "SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=?",
                [scope],
                |row| row.get(0),
            )?)
        })?;
        self.require_scope(guard, scope)?;
        Ok(value)
    }
    pub fn event_id(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        sequence: i64,
    ) -> Result<Option<String>> {
        self.require_scope(guard, scope)?;
        let value = self.database.read_only(|connection| -> Result<_> {
            Ok(connection
                .query_row(
                    "SELECT event_id FROM events WHERE scope=? AND seq=?",
                    params![scope, sequence],
                    |row| row.get(0),
                )
                .optional()?)
        })?;
        self.require_scope(guard, scope)?;
        Ok(value)
    }
    pub fn metric_records(&self, guard: &RecordGuard<'_>, attempt: &str) -> Result<Vec<Value>> {
        self.require_guard(guard)?;
        let scope = Scope::new(&guard.project, &guard.run, attempt).encoded()?;
        let rows = self.database.read_only(|connection| -> Result<_> {
        let mut statement=connection.prepare("SELECT payload FROM events WHERE scope=? AND json_extract(payload,'$.kind')='metrics' ORDER BY seq")?;
        let rows = statement
            .query_map([scope], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
        })?;
        let mut items = Vec::new();
        for row in rows {
            let row: Value = serde_json::from_str(&row)?;
            items.extend(
                row["observations"]
                    .as_array()
                    .ok_or(Error::Invalid("metric observations array"))?
                    .iter()
                    .cloned(),
            );
        }
        self.require_guard(guard)?;
        Ok(items)
    }

    /// Read a complete history only if it fits explicit HTTP-boundary budgets.
    /// Hitting a limit is an error, never a silently incomplete science result.
    /// SQL checks each encoded row length before allocating its payload string.
    pub fn metric_records_bounded(
        &self,
        guard: &RecordGuard<'_>,
        attempt: &str,
        maximum_bytes: usize,
        maximum_records: usize,
    ) -> Result<Vec<Value>> {
        self.require_guard(guard)?;
        let scope = Scope::new(&guard.project, &guard.run, attempt).encoded()?;
        let result = self.database.read_only(|connection| -> Result<_> {
            let mut statement = connection.prepare("SELECT length(CAST(payload AS BLOB)),payload FROM events WHERE scope=? AND json_extract(payload,'$.kind')='metrics' ORDER BY seq")?;
            let mut rows = statement.query([scope])?;
            let mut bytes = 0usize;
            let mut items = Vec::new();
            while let Some(row) = rows.next()? {
                let size: i64 = row.get(0)?;
                let size = usize::try_from(size).map_err(|_| Error::Invalid("metric history exceeds read budget"))?;
                bytes = bytes.checked_add(size).filter(|size| *size <= maximum_bytes).ok_or(Error::Invalid("metric history exceeds read budget"))?;
                let raw: String = row.get(1)?;
                let value: Value = serde_json::from_str(&raw)?;
                let observations = value["observations"].as_array().ok_or(Error::Invalid("metric observations array"))?;
                if items.len().checked_add(observations.len()).is_none_or(|size| size > maximum_records) {
                    return Err(Error::Invalid("metric history exceeds read budget"));
                }
                items.extend(observations.iter().cloned());
            }
            Ok(items)
        })?;
        self.require_guard(guard)?;
        Ok(result)
    }
    pub fn latest_lifecycle(&self, guard: &RecordGuard<'_>, scope: &str) -> Result<Value> {
        self.require_scope(guard, scope)?;
        let payload:Option<String>=self.database.read_only(|connection| -> Result<_> {Ok(connection.query_row("SELECT payload FROM events WHERE scope=? AND json_extract(payload,'$.kind')='lifecycle' ORDER BY seq DESC LIMIT 1",[scope],|row|row.get(0)).optional()?)})?;
        let result = match payload {
            Some(payload) => serde_json::from_str::<Value>(&payload)?
                .get("data")
                .cloned()
                .ok_or(Error::Invalid("lifecycle data"))?,
            None => json!({}),
        };
        self.require_scope(guard, scope)?;
        Ok(result)
    }
    pub fn latest_worker_lifecycle(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
    ) -> Result<Option<Value>> {
        self.require_scope(guard, scope)?;
        let row:Option<(i64,String,String)>=self.database.read_only(|connection| -> Result<_> {Ok(connection.query_row("SELECT seq,event_id,payload FROM events WHERE scope=? AND event_id GLOB 'worker.*' AND json_extract(payload,'$.kind')='lifecycle' AND json_extract(payload,'$.data.phase') IN ('PREPARING_DATA','DATA_FAILED','TRAINING','TRAINING_COMPLETED','UPLOADING_RESULTS','RESULTS_UPLOADED','RESULTS_UPLOAD_FAILED') ORDER BY seq DESC LIMIT 1",[scope],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?))).optional()?)})?;
        self.require_scope(guard, scope)?;
        row.map(|(seq,event_id,payload)|{let payload:Value=serde_json::from_str(&payload)?;let data=&payload["data"];Ok(json!({"phase":data["phase"],"timestamp":data["timestamp"],"event_id":event_id,"sequence":seq}))}).transpose()
    }
    pub fn latest_metric_observation(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
    ) -> Result<Option<Value>> {
        self.require_scope(guard, scope)?;
        let row:Option<(i64,String,String)>=self.database.read_only(|connection| -> Result<_> {Ok(connection.query_row("SELECT seq,event_id,json_extract(payload,'$.observations[#-1]') FROM events WHERE scope=? AND json_extract(payload,'$.kind')='metrics' AND json_type(payload,'$.observations[#-1]')='object' ORDER BY seq DESC LIMIT 1",[scope],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?))).optional()?)})?;
        self.require_scope(guard, scope)?;
        row.map(|(seq,event_id,payload)|{let mut observation:Value=serde_json::from_str(&payload)?;if observation.get("timestamp").is_none(){observation["timestamp"]=Value::Null;}Ok(json!({"observation":observation,"event_id":event_id,"sequence":seq,"source":"record-journal"}))}).transpose()
    }
    pub fn delete_run(&self, project: &str, run: &str, at: &str) -> Result<Value> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.fence.path(project, run)?;
        let exclusive = self.fence.exclusive_existing(false)?;
        if !exclusive.is_deleted(project, run)? {
            return Err(Error::DeletionNotStarted);
        }
        let candidates = records::run_identities(run);
        let mut names = vec![run.to_owned()];
        for candidate in &candidates[1..] {
            if let Some(binding) = self.binding_raw(project, candidate)? {
                if records::fence_names(candidate, &binding["config"])?
                    .iter()
                    .skip(1)
                    .any(|parent| parent == run)
                {
                    names.push(candidate.clone());
                }
            }
        }
        for name in &names[1..] {
            exclusive.mark(
                project,
                name,
                json!({"publication_parent":run}).as_object().unwrap(),
                at,
            )?;
        }
        exclusive.validate()?;
        self.database.transaction(
            || exclusive.validate().map_err(Error::from),
            |tx| {
                exclusive.validate()?;
                let mut scopes = Vec::new();
                for name in &names {
                    let mut statement =
                        tx.prepare("SELECT id FROM scopes WHERE project=? AND run=?")?;
                    scopes.extend(
                        statement
                            .query_map(params![project, name], |row| row.get::<_, String>(0))?
                            .collect::<std::result::Result<Vec<_>, _>>()?,
                    );
                }
                for scope in &scopes {
                    tx.execute("DELETE FROM events WHERE scope=?", [scope])?;
                    tx.execute("DELETE FROM event_ids WHERE scope=?", [scope])?;
                }
                for name in &names {
                    tx.execute(
                        "DELETE FROM scopes WHERE project=? AND run=?",
                        params![project, name],
                    )?;
                    tx.execute(
                        "DELETE FROM runs WHERE project=? AND run=?",
                        params![project, name],
                    )?;
                }
                exclusive.validate()?;
                if !exclusive.is_deleted(project, run)? {
                    return Err(Error::DeletionNotStarted);
                }
                Ok(json!({"status":"DELETED","scopes_removed":scopes.len()}))
            },
        )
    }
}

fn initialize_schema(transaction: &rusqlite::Transaction<'_>, workspace_seed: &str) -> Result<()> {
    transaction.execute_batch(SCHEMA)?;
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("PRAGMA table_info(scopes)")?;
        statement
            .query_map([], |row| row.get(1))?
            .collect::<std::result::Result<_, _>>()?
    };
    for column in ["display_version", "display_confirmed"] {
        if !columns.iter().any(|existing| existing == column) {
            transaction.execute_batch(&format!(
                "ALTER TABLE scopes ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
            ))?;
        }
    }
    let workspace: Option<String> = transaction
        .query_row("SELECT value FROM meta WHERE key='workspace'", [], |row| {
            row.get(0)
        })
        .optional()?;
    if workspace.is_none() {
        if workspace_seed.len() != 32
            || !workspace_seed
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(Error::Invalid("explicit workspace seed"));
        }
        let count:i64=transaction.query_row("SELECT (SELECT COUNT(*) FROM runs)+(SELECT COUNT(*) FROM scopes)+(SELECT COUNT(*) FROM events)",[],|row|row.get(0))?;
        if count != 0 {
            return Err(Error::Invalid("missing existing workspace identity"));
        }
        transaction.execute("INSERT INTO meta VALUES ('workspace',?)", [workspace_seed])?;
    }
    Ok(())
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn row_document(row: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    use rusqlite::types::ValueRef;
    let mut result = Map::new();
    for index in 0..row.as_ref().column_count() {
        let value = match row.get_ref(index)? {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(value) => json!(value),
            ValueRef::Real(value) => serde_json::Number::from_f64(value)
                .map(Value::Number)
                .ok_or_else(|| {
                    rusqlite::Error::InvalidColumnType(
                        index,
                        row.as_ref().column_name(index).unwrap_or("").into(),
                        rusqlite::types::Type::Real,
                    )
                })?,
            ValueRef::Text(value) => json!(std::str::from_utf8(value).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    index,
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?),
            ValueRef::Blob(_) => {
                return Err(rusqlite::Error::InvalidColumnType(
                    index,
                    row.as_ref().column_name(index)?.into(),
                    rusqlite::types::Type::Blob,
                ));
            }
        };
        result.insert(row.as_ref().column_name(index)?.into(), value);
    }
    Ok(Value::Object(result))
}

impl Store {
    pub fn scope_status(&self, guard: &RecordGuard<'_>, scope: &str) -> Result<Option<Value>> {
        self.require_scope(guard, scope)?;
        let value=self.database.read_only(|connection| -> Result<_> {Ok(connection.query_row("SELECT s.*,(SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=s.id) AS total FROM scopes s WHERE id=?",[scope],row_document).optional()?)})?;
        self.require_scope(guard, scope)?;
        Ok(value)
    }
    pub fn run_scopes(&self, guard: &RecordGuard<'_>) -> Result<Vec<Value>> {
        self.require_guard(guard)?;
        let value=self.database.read_only(|connection| -> Result<_> {
        let mut statement=connection.prepare("SELECT s.*,(SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=s.id) AS total FROM scopes s WHERE project=? AND run=?")?;
        let value = statement
            .query_map(params![guard.project, guard.run], row_document)?
            .collect::<std::result::Result<_, _>>()?;
        Ok(value)
        })?;
        self.require_guard(guard)?;
        Ok(value)
    }
    pub fn pending(&self, guard: &RecordGuard<'_>) -> Result<Vec<Value>> {
        self.require_guard(guard)?;
        let rows=self.database.read_only(|connection| -> Result<_> {
        let mut statement=connection.prepare("SELECT s.* FROM scopes s JOIN runs r ON r.project=s.project AND r.run=s.run WHERE s.project=? AND s.run=? AND json_extract(r.route,'$.enabled')=1 AND (confirmed < (SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=s.id) OR (confirmed>0 AND (display_version<2 OR display_confirmed<confirmed))) ORDER BY COALESCE(last_attempt_at,'')")?;
        let rows = statement
            .query_map(params![guard.project, guard.run], row_document)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
        })?;
        self.require_guard(guard)?;
        Ok(rows)
    }
    pub fn publication_failure(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        error: &str,
        at: &str,
    ) -> Result<()> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_scope(guard, scope)?;
        self.database.transaction(||self.require_guard(guard), |tx| {
        self.require_guard(guard)?;
        let changed=tx.execute("UPDATE scopes SET status=CASE WHEN status='LOCAL_ONLY' THEN 'LOCAL_ONLY' ELSE 'RETRY_PENDING' END,error=?,last_attempt_at=?,failures=failures+1 WHERE id=? AND project=? AND run=?",params![error,at,scope,guard.project,guard.run])?;
        if changed != 1 {
            return Err(Error::Scope);
        }
        self.require_guard(guard)?;
            Ok(())
        })
    }

    pub fn confirm_publication(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        proof: &RemoteHistoryEvidence,
        error: Option<&str>,
        at: &str,
    ) -> Result<()> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_scope(guard, scope)?;
        self.database.transaction(||self.require_guard(guard), |tx| {
        self.require_guard(guard)?;
        let state=tx.query_row("SELECT s.*,(SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=s.id) AS total FROM scopes s WHERE id=? AND project=? AND run=?",params![scope,guard.project,guard.run],row_document).optional()?.ok_or(Error::Scope)?;
        let route: String = tx.query_row(
            "SELECT route FROM runs WHERE project=? AND run=?",
            params![guard.project, guard.run],
            |row| row.get(0),
        )?;
        let route: Value = serde_json::from_str(&route)?;
        let events: Vec<(i64, String)> = {
            let mut statement = tx.prepare(
                "SELECT seq,event_id FROM events WHERE scope=? AND seq>=? AND seq<=? ORDER BY seq",
            )?;
            statement
                .query_map(
                    params![
                        scope,
                        state["confirmed"]
                            .as_i64()
                            .ok_or(Error::Invalid("confirmed cursor"))?
                            .max(1),
                        proof.through
                    ],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )?
                .collect::<std::result::Result<_, _>>()?
        };
        let plan = records::confirmation_plan(&state, &route, &events, proof, error)
            .map_err(|_| Error::Confirmation)?;
        tx.execute(
            "UPDATE scopes SET confirmed=MAX(confirmed,?),last_confirmed_at=? WHERE id=?",
            params![plan.confirmed, at, scope],
        )?;
        if let Some(version) = plan.display_version {
            tx.execute("UPDATE scopes SET display_version=MAX(display_version,?),display_confirmed=MAX(display_confirmed,confirmed) WHERE id=?",params![version,scope])?;
        }
        let total: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=?",
            [scope],
            |row| row.get(0),
        )?;
        tx.execute("UPDATE scopes SET status=CASE WHEN ? IS NOT NULL THEN 'RETRY_PENDING' WHEN confirmed>=? THEN 'SYNCED' ELSE 'PENDING' END,error=?,last_attempt_at=?,failures=CASE WHEN ? IS NULL THEN 0 ELSE failures+1 END WHERE id=?",params![error,total,error,at,error,scope])?;
        self.require_guard(guard)?;
            Ok(())
        })
    }

    pub fn binding(&self, guard: &RecordGuard<'_>) -> Result<Option<Value>> {
        self.require_guard(guard)?;
        let value = self.binding_raw(&guard.project, &guard.run)?;
        self.require_guard(guard)?;
        Ok(value)
    }
}
