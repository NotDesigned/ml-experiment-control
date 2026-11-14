//! The keyed journal layout. Each scope has one integer key, which `events`,
//! `event_ids` and `derived_events` store instead of the scope text. A metric
//! row is stored without the observation fields that equal its scope's
//! template, and readers restore the exact text that was appended. Event IDs,
//! checksums and dedup keys are always those of the full canonical payload.
use std::{fmt::Write as _, path::Path};

use ml_exp_core::records;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Map, Value};

use super::{Error, Result};
use crate::storage::SqliteDatabase;

/// Tables of the keyed layout. Unique secondary indexes are in `INDEXES`, so an
/// offline migration can build them after loading rows, in sorted order.
pub(crate) const TABLES: &str = r#"
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (project TEXT, run TEXT, route TEXT NOT NULL, config TEXT NOT NULL, PRIMARY KEY(project,run));
CREATE TABLE IF NOT EXISTS scopes (
 id TEXT PRIMARY KEY, project TEXT, run TEXT, attempt TEXT, wandb_id TEXT NOT NULL,
 confirmed INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'PENDING', error TEXT,
 last_attempt_at TEXT,last_confirmed_at TEXT,failures INTEGER NOT NULL DEFAULT 0,
 display_version INTEGER NOT NULL DEFAULT 0,display_confirmed INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS scope_keys (scope_key INTEGER PRIMARY KEY,scope TEXT NOT NULL UNIQUE,template TEXT);
CREATE TABLE IF NOT EXISTS events (scope_key INTEGER NOT NULL,seq INTEGER NOT NULL,event_id TEXT NOT NULL,
 form INTEGER,payload TEXT NOT NULL,dedup BLOB,PRIMARY KEY(scope_key,seq));
CREATE TABLE IF NOT EXISTS event_ids (scope_key INTEGER NOT NULL,event_id TEXT NOT NULL,digest BLOB NOT NULL,
 PRIMARY KEY(scope_key,event_id)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS native_publications (scope TEXT PRIMARY KEY,state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS derived_events (scope_key INTEGER NOT NULL,seq INTEGER NOT NULL,event_id TEXT NOT NULL,
 payload TEXT NOT NULL,batch TEXT NOT NULL,PRIMARY KEY(scope_key,seq));
CREATE TABLE IF NOT EXISTS derived_batches (scope TEXT,sha256 TEXT,submitted_at TEXT NOT NULL,
 first INTEGER NOT NULL,last INTEGER NOT NULL,records INTEGER NOT NULL,PRIMARY KEY(scope,sha256));
CREATE TABLE IF NOT EXISTS wandb_rebuilds (scope TEXT,ordinal INTEGER,sha256 TEXT NOT NULL,names TEXT NOT NULL,
 audit TEXT NOT NULL,old_wandb_id TEXT NOT NULL,new_wandb_id TEXT NOT NULL UNIQUE,requested_at TEXT NOT NULL,
 finished_at TEXT,previous_state TEXT,PRIMARY KEY(scope,ordinal),UNIQUE(scope,old_wandb_id,sha256));
CREATE TABLE IF NOT EXISTS mlflow_publications (scope TEXT PRIMARY KEY,state TEXT NOT NULL);
"#;
pub(crate) const INDEXES: &str = r#"
CREATE UNIQUE INDEX IF NOT EXISTS events_event_id ON events(scope_key,event_id);
CREATE UNIQUE INDEX IF NOT EXISTS events_dedup ON events(scope_key,dedup);
CREATE UNIQUE INDEX IF NOT EXISTS derived_events_event_id ON derived_events(scope_key,event_id);
"#;
/// The `meta` row that names the keyed layout.
pub(crate) const MARK_LAYOUT: &str = "INSERT OR IGNORE INTO meta VALUES ('journal_layout','2')";
const MARKER: &str = "SELECT value FROM meta WHERE key='journal_layout'";
const FIND: &str = "SELECT scope_key,template FROM scope_keys WHERE scope=?";
const CLAIM: &str = "INSERT OR IGNORE INTO scope_keys(scope) VALUES (?)";
const SET_TEMPLATE: &str =
    "UPDATE scope_keys SET template=? WHERE scope_key=? AND template IS NULL";
/// `events.form` of a row stored against its scope's template; NULL is full text.
pub(crate) const COMPACT: i64 = 1;
/// Fields that usually change from one observation to the next. `name` must
/// stay in every stored observation, because SQL reads it.
const VARYING: [&str; 4] = ["name", "step", "timestamp", "value"];

/// The refusal for a journal written before the keyed layout.
pub const PREDATES_COMPACT_LAYOUT: &str = "journal_predates_compact_layout: tracking/outbox.sqlite uses the old layout; stop the service, run `ml-expd-rs migrate-journal --registry REGISTRY_ROOT`, then start it. The journal was not changed";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    /// No journal tables yet.
    Empty,
    /// Scope text in every row: only `migrate-journal` reads it.
    Legacy,
    Keyed,
}

fn columns(connection: &Connection, table: &str) -> Result<Vec<String>> {
    let sql = format!("SELECT name FROM pragma_table_info('{table}')");
    let mut statement = connection.prepare(&sql)?;
    let names = statement
        .query_map([], |row| row.get(0))?
        .collect::<std::result::Result<_, _>>()?;
    Ok(names)
}

pub(crate) fn layout(connection: &Connection) -> Result<Layout> {
    let events = columns(connection, "events")?;
    if events.is_empty() {
        return Ok(Layout::Empty);
    }
    let mut marker: Option<String> = None;
    if !columns(connection, "meta")?.is_empty() {
        let mut statement = connection.prepare(MARKER)?;
        marker = statement.query_row([], |row| row.get(0)).optional()?;
    }
    let keyed = events.iter().any(|name| name == "scope_key");
    let legacy = events.iter().any(|name| name == "scope");
    match (keyed, legacy, marker.as_deref()) {
        (true, false, Some("2")) => Ok(Layout::Keyed),
        (false, true, None) => Ok(Layout::Legacy),
        _ => Err(Error::Invalid("unknown record journal layout")),
    }
}

fn legacy(connection: &Connection) -> Result<bool> {
    Ok(layout(connection)? == Layout::Legacy)
}

/// Whether the registry's journal has the old layout and needs `migrate-journal`.
/// An absent or unreadable journal is not one; later checks report it.
pub fn predates_compact_layout(registry_root: &Path) -> bool {
    let database = SqliteDatabase::new(registry_root.join("tracking/outbox.sqlite"));
    database.read_only(legacy).unwrap_or(false)
}

/// One scope's key and template. A scope without a key has no rows; queries
/// bind its absent key as NULL, which matches nothing.
#[derive(Clone, Debug, Default)]
pub(crate) struct ScopeRow {
    pub(crate) key: Option<i64>,
    template: Option<Map<String, Value>>,
}

fn parse_template(text: &str) -> Result<Map<String, Value>> {
    match serde_json::from_str::<Value>(text)? {
        Value::Object(fields) => Ok(fields),
        _ => Err(Error::Invalid("scope template")),
    }
}

impl ScopeRow {
    pub(crate) fn find(connection: &Connection, scope: &str) -> Result<Self> {
        let mut statement = connection.prepare(FIND)?;
        let row: Option<(i64, Option<String>)> = statement
            .query_row([scope], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()?;
        let Some((key, template)) = row else {
            return Ok(Self::default());
        };
        Ok(Self {
            key: Some(key),
            template: template.as_deref().map(parse_template).transpose()?,
        })
    }
    /// The scope's key, assigned on its first row.
    pub(crate) fn claim(connection: &Connection, scope: &str) -> Result<Self> {
        connection.execute(CLAIM, [scope])?;
        let row = Self::find(connection, scope)?;
        if row.key.is_none() {
            return Err(Error::Invalid("scope key"));
        }
        Ok(row)
    }
    fn fields(&self) -> Result<&Map<String, Value>> {
        self.template
            .as_ref()
            .ok_or(Error::Invalid("compact row template"))
    }
    /// The text of one stored row exactly as it was appended.
    pub(crate) fn text(&self, stored: String, form: Option<i64>) -> Result<String> {
        match form {
            None => Ok(stored),
            Some(COMPACT) => expand(&stored, self.fields()?),
            Some(_) => Err(Error::Invalid("journal row form")),
        }
    }
    /// How one new row is stored: compact when its metric observations restore
    /// to exactly `body`, else `body` itself. The scope's first metric row sets
    /// its template, once.
    pub(crate) fn stored(
        &mut self,
        connection: &Connection,
        payload: &Value,
        body: String,
    ) -> Result<(Option<i64>, String)> {
        let fresh = match &self.template {
            Some(_) => None,
            None => first_template(payload),
        };
        if let Some(fresh) = fresh {
            let text = records::encoded(&Value::Object(fresh.clone()))?;
            connection.execute(SET_TEMPLATE, params![text, self.key])?;
            self.template = Some(fresh);
        }
        let compacted = match &self.template {
            Some(template) => compact(payload, &body, template),
            None => None,
        };
        Ok(match compacted {
            Some(text) => (Some(COMPACT), text),
            None => (None, body),
        })
    }
}

/// The fields of a metric row's first observation that a template keeps.
fn first_template(payload: &Value) -> Option<Map<String, Value>> {
    if payload.get("kind")?.as_str()? != "metrics" {
        return None;
    }
    let first = payload.get("observations")?.as_array()?.first()?;
    let fields: Map<String, Value> = first
        .as_object()?
        .iter()
        .filter(|(key, _)| !VARYING.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    (!fields.is_empty()).then_some(fields)
}

/// `payload` without the observation fields equal to `template`, when that is
/// shorter and restores to exactly `body`. An observation must hold every
/// template field, so a field it lacks is never restored.
pub(crate) fn compact(
    payload: &Value,
    body: &str,
    template: &Map<String, Value>,
) -> Option<String> {
    let mut row = payload.as_object()?.clone();
    if row.get("kind")?.as_str()? != "metrics" {
        return None;
    }
    let mut kept = Vec::new();
    for observation in row.get("observations")?.as_array()? {
        let observation = observation.as_object()?;
        if !template.keys().all(|key| observation.contains_key(key)) {
            return None;
        }
        let fields: Map<String, Value> = observation
            .iter()
            .filter(|(key, value)| template.get(key.as_str()) != Some(*value))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        kept.push(Value::Object(fields));
    }
    row.insert("observations".into(), Value::Array(kept));
    let text = records::encoded(&Value::Object(row)).ok()?;
    let restored = expand(&text, template).ok()?;
    (text.len() < body.len() && restored == body).then_some(text)
}

/// The canonical text of a compact row with its template fields restored.
pub(crate) fn expand(stored: &str, template: &Map<String, Value>) -> Result<String> {
    let mut row: Value = serde_json::from_str(stored)?;
    let observations = row
        .get_mut("observations")
        .and_then(Value::as_array_mut)
        .ok_or(Error::Invalid("compact observations"))?;
    for observation in observations {
        let fields = observation
            .as_object_mut()
            .ok_or(Error::Invalid("compact observation"))?;
        for (key, value) in template {
            fields.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    Ok(records::encoded(&row)?)
}

/// A SHA256 hex digest as its 32 stored bytes.
pub(crate) fn digest_bytes(hex: &str) -> Result<Vec<u8>> {
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::Invalid("SHA256 digest"));
    }
    (0..32)
        .map(|index| {
            u8::from_str_radix(&hex[2 * index..2 * index + 2], 16)
                .map_err(|_| Error::Invalid("SHA256 digest"))
        })
        .collect()
}

pub(crate) fn digest_hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}
