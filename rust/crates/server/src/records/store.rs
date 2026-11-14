use std::path::Path;

use ml_exp_core::records::{self, Event, ImportEmission, Scope};
use rusqlite::{OptionalExtension, params};
use serde_json::{Map, Value, json};

use super::{Error, Result};
use crate::{
    deletion::{Guard, RunFence},
    storage::SqliteDatabase,
};

use super::layout::{self, Layout, ScopeRow};

/// Training-time records and post-hoc derived batches share one sequence, so the
/// publisher proves both the same way; only publication reads them together.
/// Derived rows are always stored as full text.
const PUBLISHED: &str = "(SELECT scope_key,seq,event_id,form,payload FROM events UNION ALL SELECT scope_key,seq,event_id,NULL,payload FROM derived_events)";
const TRAINING_TOTAL: &str = "(SELECT COALESCE(MAX(seq),0) FROM events WHERE scope_key=KEY)";
const PUBLISHED_TOTAL: &str = "MAX((SELECT COALESCE(MAX(seq),0) FROM events WHERE scope_key=KEY),(SELECT COALESCE(MAX(seq),0) FROM derived_events WHERE scope_key=KEY))";
/// The key of the scope `SCOPE`, inside a total.
const KEY: &str = "(SELECT scope_key FROM scope_keys WHERE scope=SCOPE)";
const METRIC_ROWS: &str = "SELECT form,payload FROM events WHERE scope_key=? AND json_extract(payload,'$.kind')='metrics' ORDER BY seq";
const METRIC_SIZES: &str = "SELECT length(CAST(payload AS BLOB)),form,payload FROM events WHERE scope_key=? AND json_extract(payload,'$.kind')='metrics' ORDER BY seq";
const LATEST_LIFECYCLE: &str = "SELECT form,payload FROM events WHERE scope_key=? AND json_extract(payload,'$.kind')='lifecycle' ORDER BY seq DESC LIMIT 1";

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

/// One post-hoc file accepted for an Attempt, identified by its SHA256.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivedBatch {
    pub sha256: String,
    pub submitted_at: String,
    pub first: i64,
    pub last: i64,
    pub records: i64,
}
impl DerivedBatch {
    pub fn document(&self) -> Value {
        json!({"sha256":self.sha256,"submitted_at":self.submitted_at,"records":self.records,
            "first_sequence":self.first,"last_sequence":self.last,"source":"derived"})
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
            || Ok(()),
            |transaction| initialize_schema(transaction, workspace_seed),
        )?;
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
        Ok(RecordGuard {
            store: self,
            held,
            project: project.into(),
            run: run.into(),
            names,
        })
    }
    fn require_guard(&self, guard: &RecordGuard<'_>) -> Result<()> {
        if !std::ptr::eq(self, guard.store) {
            return Err(Error::Guard);
        }
        for name in &guard.names {
            guard.held.require_observed_live(&guard.project, name)?;
        }
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
        Ok(())
    }
    /// Every writable open creates the derived tables. A read-only Store may
    /// predate them, and sees only the training journal, as an older server did.
    fn published(&self) -> &'static str {
        if self.read_only { "events" } else { PUBLISHED }
    }
    /// The highest sequence of `scope`, an SQL term such as `?1` or `s.id`.
    fn total_sql(&self, scope: &str) -> String {
        let total = if self.read_only {
            TRAINING_TOTAL
        } else {
            PUBLISHED_TOTAL
        };
        total.replace("KEY", KEY).replace("SCOPE", scope)
    }
    /// Rows of a scope for publication, as stored.
    fn rows_sql(&self) -> String {
        format!(
            "SELECT seq,event_id,form,payload FROM {} WHERE scope_key=? AND seq>? ORDER BY seq LIMIT ?",
            self.published()
        )
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
                // A W&B rebuild moved the scope to the id it published again.
                let rebuilt: Option<String> = tx
                    .query_row(rebuild::CURRENT_ID, [&scope], |row| row.get(0))
                    .optional()?;
                let wandb_id = rebuilt.unwrap_or(wandb_id);
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
        let prepared = events
            .iter()
            .map(records::prepare_event)
            .collect::<records::Result<Vec<_>>>()?;
        self.database.transaction(||self.require_guard(guard), |tx| {
        self.require_guard(guard)?;
        let bound:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM scopes s JOIN runs r ON r.project=s.project AND r.run=s.run WHERE s.id=? AND s.project=? AND s.run=?)",params![scope,guard.project,guard.run],|row|row.get(0))?;
        if !bound {
            return Err(Error::Scope);
        }
        let mut keyed = ScopeRow::claim(tx, scope)?;
        let mut seq: i64 =
            tx.query_row(&format!("SELECT {}", self.total_sql("?1")), [scope], |row| {
                row.get(0)
            })?;
        // Identity, idempotency and dedup are those of the full canonical
        // payload, whatever form the row is stored in.
        for (event, prepared) in events.iter().zip(prepared) {
            let checksum = layout::digest_bytes(&prepared.checksum)?;
            let old: Option<Vec<u8>> = tx
                .query_row(
                    "SELECT digest FROM event_ids WHERE scope_key=? AND event_id=?",
                    params![keyed.key, prepared.event_id],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(old) = old {
                if old != checksum {
                    return Err(Error::EventConflict);
                }
                continue;
            }
            tx.execute(
                "INSERT INTO event_ids VALUES (?,?,?)",
                params![keyed.key, prepared.event_id, checksum],
            )?;
            let dedup = prepared.dedup.as_deref().map(layout::digest_bytes).transpose()?;
            if let Some(dedup) = &dedup {
                let duplicate: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM events WHERE scope_key=? AND dedup=?)",
                    params![keyed.key, dedup],
                    |row| row.get(0),
                )?;
                if duplicate {
                    continue;
                }
            }
            seq = seq
                .checked_add(1)
                .ok_or(Error::Invalid("record sequence overflow"))?;
            let (form, body) = keyed.stored(tx, &event.payload, prepared.body)?;
            tx.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?)",
                params![keyed.key, seq, prepared.event_id, form, body, dedup],
            )?;
        }
        tx.execute("UPDATE scopes SET status='PENDING' WHERE confirmed<? AND error IS NULL AND status NOT IN ('LOCAL_ONLY','PENDING') AND id=?",params![seq,scope])?;
        self.require_guard(guard)?;
            Ok(seq)
        })
    }

    /// The scope and frozen config an import writes under. The Run must already
    /// be bound, so an import can neither choose nor change a route or schema.
    pub fn import_scope(&self, guard: &RecordGuard<'_>, attempt: &str) -> Result<(String, Value)> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_guard(guard)?;
        let binding = self
            .binding_raw(&guard.project, &guard.run)?
            .ok_or(Error::Unbound)?;
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
    /// The rows as W&B shows them, in sequence: the training journal and derived
    /// batches, without any observation a W&B rebuild of this scope excludes.
    pub fn events(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        after: i64,
        limit: i64,
    ) -> Result<Vec<EventRow>> {
        self.require_scope(guard, scope)?;
        let sql = self.rows_sql();
        let mut result = self.database.read_only(|connection| -> Result<_> {
            let keyed = ScopeRow::find(connection, scope)?;
            let mut statement = connection.prepare(&sql)?;
            let mut query = statement.query(params![keyed.key, after, limit])?;
            let mut rows = Vec::new();
            while let Some(row) = query.next()? {
                let payload = keyed.text(row.get(3)?, row.get(2)?)?;
                rows.push(EventRow {
                    seq: row.get(0)?,
                    event_id: row.get(1)?,
                    payload: serde_json::from_str(&payload)?,
                });
            }
            Ok(rows)
        })?;
        self.as_published(scope, &mut result)?;
        self.require_scope(guard, scope)?;
        Ok(result)
    }
    /// Like `events`, but stops once `max_bytes` of payload are loaded, always
    /// returning at least one row so one large record cannot block a scope.
    /// A compact row counts the bytes of the text it restores.
    pub fn events_bounded(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        after: i64,
        limit: i64,
        max_bytes: usize,
    ) -> Result<Vec<EventRow>> {
        self.require_scope(guard, scope)?;
        let sql = self.rows_sql();
        let mut result = self.database.read_only(|connection| -> Result<_> {
            let keyed = ScopeRow::find(connection, scope)?;
            let mut statement = connection.prepare(&sql)?;
            let mut query = statement.query(params![keyed.key, after, limit])?;
            let mut rows = Vec::new();
            let mut bytes = 0;
            while let Some(row) = query.next()? {
                let payload = keyed.text(row.get(3)?, row.get(2)?)?;
                bytes += payload.len();
                if bytes > max_bytes && !rows.is_empty() {
                    break;
                }
                rows.push(EventRow {
                    seq: row.get(0)?,
                    event_id: row.get(1)?,
                    payload: serde_json::from_str(&payload)?,
                });
            }
            Ok(rows)
        })?;
        self.as_published(scope, &mut result)?;
        self.require_scope(guard, scope)?;
        Ok(result)
    }
    pub fn total(&self, guard: &RecordGuard<'_>, scope: &str) -> Result<i64> {
        self.require_scope(guard, scope)?;
        let value = self.database.read_only(|connection| -> Result<_> {
            Ok(connection.query_row(
                &format!("SELECT {}", self.total_sql("?1")),
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
        let sql = format!(
            "SELECT event_id FROM {} WHERE scope_key=? AND seq=?",
            self.published()
        );
        let value = self.database.read_only(|connection| -> Result<_> {
            let keyed = ScopeRow::find(connection, scope)?;
            let mut statement = connection.prepare(&sql)?;
            let value = statement
                .query_row(params![keyed.key, sequence], |row| row.get(0))
                .optional()?;
            Ok(value)
        })?;
        self.require_scope(guard, scope)?;
        Ok(value)
    }
    pub fn metric_records(&self, guard: &RecordGuard<'_>, attempt: &str) -> Result<Vec<Value>> {
        self.require_guard(guard)?;
        let scope = Scope::new(&guard.project, &guard.run, attempt).encoded()?;
        let rows = self.database.read_only(|connection| -> Result<_> {
            let keyed = ScopeRow::find(connection, &scope)?;
            let mut statement = connection.prepare(METRIC_ROWS)?;
            let mut query = statement.query([keyed.key])?;
            let mut rows = Vec::new();
            while let Some(row) = query.next()? {
                rows.push(keyed.text(row.get(1)?, row.get(0)?)?);
            }
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
    /// SQL checks each stored row length before allocating its payload string;
    /// a compact row is never longer than the text it restores, which counts.
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
            let keyed = ScopeRow::find(connection, &scope)?;
            let mut statement = connection.prepare(METRIC_SIZES)?;
            let mut rows = statement.query([keyed.key])?;
            let mut bytes = 0usize;
            let mut items = Vec::new();
            while let Some(row) = rows.next()? {
                let size: i64 = row.get(0)?;
                let size = usize::try_from(size)
                    .map_err(why!(Error::Invalid("metric history exceeds read budget")))?;
                if bytes
                    .checked_add(size)
                    .is_none_or(|size| size > maximum_bytes)
                {
                    return Err(Error::Invalid("metric history exceeds read budget"));
                }
                let raw = keyed.text(row.get(2)?, row.get(1)?)?;
                bytes = bytes
                    .checked_add(raw.len())
                    .filter(|size| *size <= maximum_bytes)
                    .ok_or(Error::Invalid("metric history exceeds read budget"))?;
                let value: Value = serde_json::from_str(&raw)?;
                let observations = value["observations"]
                    .as_array()
                    .ok_or(Error::Invalid("metric observations array"))?;
                if items
                    .len()
                    .checked_add(observations.len())
                    .is_none_or(|size| size > maximum_records)
                {
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
        let payload = self.database.read_only(|connection| -> Result<_> {
            let payload = lifecycle_text(connection, scope)?;
            Ok(payload)
        })?;
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
        let row:Option<(i64,String,String)>=self.database.read_only(|connection| -> Result<_> {let keyed=ScopeRow::find(connection,scope)?;let row:Option<(i64,String,Option<i64>,String)>=connection.query_row("SELECT seq,event_id,form,payload FROM events WHERE scope_key=? AND event_id GLOB 'worker.*' AND json_extract(payload,'$.kind')='lifecycle' AND json_extract(payload,'$.data.phase') IN ('PREPARING_DATA','DATA_FAILED','TRAINING','TRAINING_COMPLETED','UPLOADING_RESULTS','RESULTS_UPLOADED','RESULTS_UPLOAD_FAILED') ORDER BY seq DESC LIMIT 1",[keyed.key],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?))).optional()?;row.map(|(seq,event_id,form,payload)|Ok((seq,event_id,keyed.text(payload,form)?))).transpose()})?;
        self.require_scope(guard, scope)?;
        row.map(|(seq,event_id,payload)|{let payload:Value=serde_json::from_str(&payload)?;let data=&payload["data"];Ok(json!({"phase":data["phase"],"timestamp":data["timestamp"],"event_id":event_id,"sequence":seq}))}).transpose()
    }
    pub fn latest_metric_observation(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
    ) -> Result<Option<Value>> {
        self.require_scope(guard, scope)?;
        let row:Option<(i64,String,Value)>=self.database.read_only(|connection| -> Result<_> {let keyed=ScopeRow::find(connection,scope)?;let row:Option<(i64,String,Option<i64>,String,String)>=connection.query_row("SELECT seq,event_id,form,payload,json_extract(payload,'$.observations[#-1]') FROM events WHERE scope_key=? AND json_extract(payload,'$.kind')='metrics' AND json_type(payload,'$.observations[#-1]')='object' ORDER BY seq DESC LIMIT 1",[keyed.key],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?))).optional()?;row.map(|(seq,event_id,form,payload,last)|Ok((seq,event_id,last_observation(&keyed,form,payload,&last)?))).transpose()})?;
        self.require_scope(guard, scope)?;
        row.map(|(seq,event_id,mut observation)|{if observation.get("timestamp").is_none(){observation["timestamp"]=Value::Null;}Ok(json!({"observation":observation,"event_id":event_id,"sequence":seq,"source":"record-journal"}))}).transpose()
    }
}

/// The latest lifecycle row of `scope`, as it was appended.
fn lifecycle_text(connection: &rusqlite::Connection, scope: &str) -> Result<Option<String>> {
    let keyed = ScopeRow::find(connection, scope)?;
    let mut statement = connection.prepare(LATEST_LIFECYCLE)?;
    let row: Option<(Option<i64>, String)> = statement
        .query_row([keyed.key], |row| Ok((row.get(0)?, row.get(1)?)))
        .optional()?;
    match row {
        Some((form, payload)) => Ok(Some(keyed.text(payload, form)?)),
        None => Ok(None),
    }
}

/// The last observation of a metric row. A full row keeps SQL's own text of
/// it, as before; a compact row takes it from the text it restores.
fn last_observation(
    keyed: &ScopeRow,
    form: Option<i64>,
    payload: String,
    last: &str,
) -> Result<Value> {
    if form.is_none() {
        return Ok(serde_json::from_str(last)?);
    }
    let row: Value = serde_json::from_str(&keyed.text(payload, form)?)?;
    let observations = row["observations"].as_array();
    let last = observations.and_then(|items| items.last());
    last.cloned().ok_or(Error::Invalid("metric observation"))
}

#[path = "derived.rs"]
mod derived;
#[path = "rebuild.rs"]
mod rebuild;
pub use rebuild::WandbRebuild;
#[path = "mlflow.rs"]
mod mlflow;
#[path = "publication.rs"]
mod publication;

fn initialize_schema(transaction: &rusqlite::Transaction<'_>, workspace_seed: &str) -> Result<()> {
    // An old-layout journal is refused, never migrated or mixed in place.
    if layout::layout(transaction)? == Layout::Legacy {
        return Err(Error::LegacyLayout);
    }
    transaction.execute_batch(layout::TABLES)?;
    transaction.execute_batch(layout::INDEXES)?;
    transaction.execute(layout::MARK_LAYOUT, [])?;
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
        let query = format!(
            "SELECT s.*,{} AS total FROM scopes s WHERE id=?",
            self.total_sql("s.id")
        );
        let value = self.database.read_only(|connection| -> Result<_> {
            Ok(connection
                .query_row(&query, [scope], row_document)
                .optional()?)
        })?;
        self.require_scope(guard, scope)?;
        Ok(value)
    }
    pub fn run_scopes(&self, guard: &RecordGuard<'_>) -> Result<Vec<Value>> {
        self.require_guard(guard)?;
        let mut value: Vec<Value> = self.database.read_only(|connection| -> Result<_> {
            let mut statement = connection.prepare(&format!(
                "SELECT s.*,{} AS total FROM scopes s WHERE project=? AND run=?",
                self.total_sql("s.id")
            ))?;
            let value = statement
                .query_map(params![guard.project, guard.run], row_document)?
                .collect::<std::result::Result<_, _>>()?;
            Ok(value)
        })?;
        self.attach_mlflow(guard, &mut value)?;
        self.require_guard(guard)?;
        Ok(value)
    }
    pub fn pending(&self, guard: &RecordGuard<'_>) -> Result<Vec<Value>> {
        self.require_guard(guard)?;
        let rows=self.database.read_only(|connection| -> Result<_> {
        let mut statement=connection.prepare(&format!("SELECT s.* FROM scopes s JOIN runs r ON r.project=s.project AND r.run=s.run WHERE s.project=? AND s.run=? AND json_extract(r.route,'$.enabled')=1 AND (confirmed < {} OR (confirmed>0 AND (display_version<4 OR display_confirmed<confirmed))) ORDER BY COALESCE(last_attempt_at,'')",self.total_sql("s.id")))?;
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

    /// Move the W&B cursor to `through` once the SDK accepted every row up to it,
    /// and with `display` record that the summary for it was sent too. W&B is a
    /// best-effort mirror, so no remote read-back is required.
    pub fn advance_publication(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        wandb_id: &str,
        through: i64,
        display: bool,
        at: &str,
    ) -> Result<()> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_scope(guard, scope)?;
        self.database.transaction(||self.require_guard(guard), |tx| {
        self.require_guard(guard)?;
        let state=tx.query_row(&format!("SELECT s.*,{} AS total FROM scopes s WHERE id=? AND project=? AND run=?",self.total_sql("s.id")),params![scope,guard.project,guard.run],row_document).optional()?.ok_or(Error::Scope)?;
        let route: String = tx.query_row(
            "SELECT route FROM runs WHERE project=? AND run=?",
            params![guard.project, guard.run],
            |row| row.get(0),
        )?;
        let route: Value = serde_json::from_str(&route)?;
        let plan = records::cursor_advance(&state, &route, wandb_id, through, display)
            .map_err(why!(Error::Confirmation))?;
        tx.execute(
            "UPDATE scopes SET confirmed=MAX(confirmed,?),last_confirmed_at=CASE WHEN ?>confirmed THEN ? ELSE last_confirmed_at END WHERE id=?",
            params![plan.confirmed, plan.confirmed, at, scope],
        )?;
        if let Some(version) = plan.display_version {
            tx.execute("UPDATE scopes SET display_version=?,display_confirmed=? WHERE id=?",params![version,plan.confirmed,scope])?;
        }
        let total = state["total"].as_i64().ok_or(Error::Invalid("journal total"))?;
        tx.execute("UPDATE scopes SET status=CASE WHEN confirmed>=? THEN 'SYNCED' ELSE 'PENDING' END,error=NULL,last_attempt_at=?,failures=0 WHERE id=?",params![total,at,scope])?;
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
