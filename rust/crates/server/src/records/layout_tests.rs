//! The keyed, compact journal layout and the offline migration to it.
use std::{
    fs,
    io::Write as _,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

use ml_exp_core::records::{self, Event, Scope, normalized_metric};
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{DerivedBatch, Error, EventRow, Store, WandbRebuild, layout, migrate};
use crate::deletion::RunFence;

const SEED: &str = "0123456789abcdef0123456789abcdef";
const PROJECT: &str = "slowrun-language-models";
const AT: &str = "2026-11-14T00:00:00Z";
/// The journal schema before the keyed layout, as the old server created it.
const LEGACY_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (project TEXT, run TEXT, route TEXT NOT NULL, config TEXT NOT NULL, PRIMARY KEY(project,run));
CREATE TABLE IF NOT EXISTS scopes (
 id TEXT PRIMARY KEY, project TEXT, run TEXT, attempt TEXT, wandb_id TEXT NOT NULL,
 confirmed INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'PENDING', error TEXT,
 last_attempt_at TEXT,last_confirmed_at TEXT,failures INTEGER NOT NULL DEFAULT 0,
 display_version INTEGER NOT NULL DEFAULT 0,display_confirmed INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS events (scope TEXT,seq INTEGER,event_id TEXT,payload TEXT NOT NULL,dedup TEXT,
 PRIMARY KEY(scope,seq),UNIQUE(scope,event_id),UNIQUE(scope,dedup));
CREATE TABLE IF NOT EXISTS event_ids (scope TEXT,event_id TEXT,digest TEXT NOT NULL,PRIMARY KEY(scope,event_id));
CREATE TABLE IF NOT EXISTS native_publications (scope TEXT PRIMARY KEY,state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS derived_events (scope TEXT,seq INTEGER,event_id TEXT,payload TEXT NOT NULL,batch TEXT NOT NULL,
 PRIMARY KEY(scope,seq),UNIQUE(scope,event_id));
CREATE TABLE IF NOT EXISTS derived_batches (scope TEXT,sha256 TEXT,submitted_at TEXT NOT NULL,
 first INTEGER NOT NULL,last INTEGER NOT NULL,records INTEGER NOT NULL,PRIMARY KEY(scope,sha256));
CREATE TABLE IF NOT EXISTS wandb_rebuilds (scope TEXT,ordinal INTEGER,sha256 TEXT NOT NULL,names TEXT NOT NULL,
 audit TEXT NOT NULL,old_wandb_id TEXT NOT NULL,new_wandb_id TEXT NOT NULL UNIQUE,requested_at TEXT NOT NULL,
 finished_at TEXT,previous_state TEXT,PRIMARY KEY(scope,ordinal),UNIQUE(scope,old_wandb_id,sha256));
CREATE TABLE IF NOT EXISTS mlflow_publications (scope TEXT PRIMARY KEY,state TEXT NOT NULL);
"#;
/// Every table that keeps its layout, copied verbatim into an old-layout file.
const PLAIN: [&str; 7] = [
    "meta",
    "runs",
    "scopes",
    "native_publications",
    "derived_batches",
    "wandb_rebuilds",
    "mlflow_publications",
];
const KEYED_ROWS: &str = "SELECT k.scope,e.seq,e.event_id,e.form,e.payload,CASE WHEN e.dedup IS NULL THEN NULL ELSE lower(hex(e.dedup)) END FROM events e JOIN scope_keys k USING(scope_key) ORDER BY k.scope,e.seq";
const KEYED_IDS: &str = "INSERT INTO event_ids SELECT k.scope,i.event_id,lower(hex(i.digest)) FROM keyed.event_ids i JOIN keyed.scope_keys k USING(scope_key)";
const KEYED_DERIVED: &str = "INSERT INTO derived_events SELECT k.scope,d.seq,d.event_id,d.payload,d.batch FROM keyed.derived_events d JOIN keyed.scope_keys k USING(scope_key)";

/// What production configs freeze: a protocol identity, beside the worker source.
fn config() -> Value {
    let protocol = format!("sha256:{}", "ab".repeat(32));
    json!({"ml_expd":{"evaluation":{"protocol_id":protocol}}})
}
/// One worker metric row, its observation normalized as the worker does.
fn metric(step: u64, loss: f64) -> Value {
    let record = json!({"name":"train/loss","unit":"nats/token","value":loss,"step":step,
        "timestamp":AT});
    normalized_metric(&record, &config()).unwrap()
}
/// A row with one observation per metric field.
fn metrics(step: u64, loss: f64, accuracy: f64) -> Value {
    let record = json!({"train/loss":loss,"eval/accuracy":accuracy,"step":step,"timestamp":AT});
    normalized_metric(&record, &config()).unwrap()
}
fn lifecycle(phase: &str, state: &str) -> Value {
    json!({"kind":"lifecycle","data":{"phase":phase,"state":state,"timestamp":AT}})
}
fn event(id: &str, payload: Value) -> Event {
    Event {
        event_id: id.into(),
        payload,
    }
}
fn route() -> Value {
    json!({"enabled":true,"entity":"fixture-team","project":"fixture-science",
        "mlflow":{"enabled":true,"experiment":"fixture-experiment"}})
}
fn bound(store: &Store, run: &str, route: &Value, attempt: &str) -> String {
    let guard = store.guard(PROJECT, run).unwrap();
    store.bind(&guard, route, &config()).unwrap();
    store.scope(&guard, attempt).unwrap()
}
/// What a reader got from a full row before: its canonical text, parsed.
fn as_read(payload: &Value) -> String {
    let text = records::encoded(payload).unwrap();
    serde_json::to_string(&serde_json::from_str::<Value>(&text).unwrap()).unwrap()
}
fn text(value: &Value) -> String {
    serde_json::to_string(value).unwrap()
}
fn forms(path: &Path) -> Vec<(i64, Option<i64>)> {
    let connection = Connection::open(path).unwrap();
    let mut statement = connection
        .prepare("SELECT seq,form FROM events ORDER BY scope_key,seq")
        .unwrap();
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn compact_and_full_rows_read_back_exactly_and_keep_their_identities() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let scope = bound(&store, "run-a", &route(), "attempt-001");
    let guard = store.guard(PROJECT, "run-a").unwrap();
    let mut written = vec![
        event("worker.metrics.1", metric(1, 2.5)),
        event("worker.metrics.2", metrics(2, 2.25, 0.5)),
        event("worker.lifecycle.1", lifecycle("TRAINING", "RUNNING")),
    ];
    // An observation without a template field is stored as full text.
    let mut partial = metric(3, 2.0);
    partial["observations"][0]
        .as_object_mut()
        .unwrap()
        .remove("unit");
    written.push(event("worker.metrics.3", partial));
    // A field that differs from the template is kept in the compact row.
    let mut other = metric(4, 1.75);
    other["observations"][0]["protocol_id"] = json!("protocol.other");
    written.push(event("worker.metrics.4", other));
    assert_eq!(store.append(&guard, &scope, &written).unwrap(), 5);

    // A full row written before compaction existed, as a migration copies one.
    let old = event("worker.metrics.5", metric(5, 1.5));
    let prepared = records::prepare_event(&old).unwrap();
    let digest = layout::digest_bytes(&prepared.checksum).unwrap();
    let connection = Connection::open(store.path()).unwrap();
    let key: i64 = connection
        .query_row(
            "SELECT scope_key FROM scope_keys WHERE scope=?",
            [&scope],
            |row| row.get(0),
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO event_ids VALUES (?,?,?)",
            params![key, old.event_id, digest],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO events VALUES (?,6,?,NULL,?,?)",
            params![key, old.event_id, prepared.body, digest],
        )
        .unwrap();
    drop(connection);
    written.push(old.clone());
    // A compact row after the full one.
    written.push(event("worker.metrics.6", metrics(6, 1.25, 0.75)));
    assert_eq!(store.append(&guard, &scope, &written[6..]).unwrap(), 7);

    let compact = Some(layout::COMPACT);
    assert_eq!(
        forms(store.path()),
        vec![
            (1, compact),
            (2, compact),
            (3, None),
            (4, None),
            (5, compact),
            (6, None),
            (7, compact),
        ]
    );
    // Every reader gets exactly what it got from the full text.
    let rows = store.events(&guard, &scope, 0, 100).unwrap();
    assert_eq!(rows.len(), written.len());
    for (row, event) in rows.iter().zip(&written) {
        assert_eq!(row.event_id, event.event_id);
        assert_eq!(text(&row.payload), as_read(&event.payload));
    }
    let bounded = store.events_bounded(&guard, &scope, 0, 100, 1).unwrap();
    assert_eq!(bounded.len(), 1);
    let full = records::encoded(&written[0].payload).unwrap().len();
    let two = full + records::encoded(&written[1].payload).unwrap().len();
    let bounded = store.events_bounded(&guard, &scope, 0, 100, two).unwrap();
    assert_eq!(bounded.len(), 2);
    assert_eq!(
        store.event_id(&guard, &scope, 6).unwrap().unwrap(),
        old.event_id
    );
    let observations: Vec<Value> = written
        .iter()
        .filter(|event| event.payload["kind"] == "metrics")
        .flat_map(|event| {
            let read: Value = serde_json::from_str(&as_read(&event.payload)).unwrap();
            read["observations"].as_array().unwrap().clone()
        })
        .collect();
    let history = store.metric_records(&guard, "attempt-001").unwrap();
    assert_eq!(text(&json!(history)), text(&json!(observations)));
    let budget: usize = written
        .iter()
        .filter(|event| event.payload["kind"] == "metrics")
        .map(|event| records::encoded(&event.payload).unwrap().len())
        .sum();
    let count = observations.len();
    let bounded = store
        .metric_records_bounded(&guard, "attempt-001", budget, count)
        .unwrap();
    assert_eq!(text(&json!(bounded)), text(&json!(observations)));
    // Restored bytes count, as the full text counted before.
    let over = store.metric_records_bounded(&guard, "attempt-001", budget - 1, count);
    assert!(over.is_err());
    let latest = store
        .latest_metric_observation(&guard, &scope)
        .unwrap()
        .unwrap();
    assert_eq!(
        text(&latest["observation"]),
        text(observations.last().unwrap())
    );
    assert_eq!(latest["sequence"], 7);
    assert_eq!(
        store.latest_lifecycle(&guard, &scope).unwrap()["phase"],
        "TRAINING"
    );

    // Idempotent re-append and content dedup, on both sides of compaction.
    assert_eq!(store.append(&guard, &scope, &written).unwrap(), 7);
    let copies: Vec<Event> = [0, 1, 5, 6]
        .iter()
        .map(|index| event(&format!("copy.{index}"), written[*index].payload.clone()))
        .collect();
    assert_eq!(store.append(&guard, &scope, &copies).unwrap(), 7);
    for index in [0, 5] {
        let changed = event(&written[index].event_id, metric(99, 9.0));
        assert!(matches!(
            store.append(&guard, &scope, &[changed]),
            Err(Error::EventConflict)
        ));
    }
    assert_eq!(store.total(&guard, &scope).unwrap(), 7);
    let connection = Connection::open(store.path()).unwrap();
    let ids: i64 = connection
        .query_row("SELECT COUNT(*) FROM event_ids", [], |row| row.get(0))
        .unwrap();
    assert_eq!(ids, 11, "skipped content duplicates still own their IDs");
}

/// Write `events` as the old server's `append` did, in the old layout.
fn legacy_append(connection: &Connection, scope: &str, events: &[Event]) {
    let mut seq: i64 = connection
        .query_row(
            "SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=?",
            [scope],
            |row| row.get(0),
        )
        .unwrap();
    for event in events {
        let prepared = records::prepare_event(event).unwrap();
        let known: bool = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM event_ids WHERE scope=? AND event_id=?)",
                params![scope, prepared.event_id],
                |row| row.get(0),
            )
            .unwrap();
        if known {
            continue;
        }
        connection
            .execute(
                "INSERT INTO event_ids VALUES (?,?,?)",
                params![scope, prepared.event_id, prepared.checksum],
            )
            .unwrap();
        let duplicate: bool = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM events WHERE scope=? AND dedup=?)",
                params![scope, prepared.dedup],
                |row| row.get(0),
            )
            .unwrap();
        if duplicate {
            continue;
        }
        seq += 1;
        connection
            .execute(
                "INSERT INTO events VALUES (?,?,?,?,?)",
                params![scope, seq, prepared.event_id, prepared.body, prepared.dedup],
            )
            .unwrap();
    }
}

/// The same journal in the old layout: every row with its scope text and full
/// payload text, hex digests, and the other tables as they are.
fn downgrade(keyed: &Path, legacy: &Path) {
    let source = Connection::open(keyed).unwrap();
    let target = Connection::open(legacy).unwrap();
    target.execute_batch(LEGACY_SCHEMA).unwrap();
    let mut statement = source.prepare(KEYED_ROWS).unwrap();
    let mut rows = statement.query([]).unwrap();
    while let Some(row) = rows.next().unwrap() {
        let scope: String = row.get(0).unwrap();
        let seq: i64 = row.get(1).unwrap();
        let event_id: String = row.get(2).unwrap();
        let form: Option<i64> = row.get(3).unwrap();
        let stored: String = row.get(4).unwrap();
        let dedup: Option<String> = row.get(5).unwrap();
        let keyed = layout::ScopeRow::find(&source, &scope).unwrap();
        let payload = keyed.text(stored, form).unwrap();
        target
            .execute(
                "INSERT INTO events VALUES (?,?,?,?,?)",
                params![scope, seq, event_id, payload, dedup],
            )
            .unwrap();
    }
    let path = keyed.to_str().unwrap();
    target.execute("ATTACH DATABASE ? AS keyed", [path]).unwrap();
    for table in PLAIN {
        let filter = match table {
            "meta" => " WHERE key<>'journal_layout'",
            _ => "",
        };
        let sql = format!("INSERT INTO {table} SELECT * FROM keyed.{table}{filter}");
        target.execute(&sql, []).unwrap();
    }
    target.execute(KEYED_IDS, []).unwrap();
    target.execute(KEYED_DERIVED, []).unwrap();
    target.execute("DETACH DATABASE keyed", []).unwrap();
    fs::set_permissions(legacy, fs::Permissions::from_mode(0o600)).unwrap();
}

/// Training records, lifecycle, a derived batch, a finished W&B publication
/// with a rebuild, MLflow state, a failed publication and a local-only Run.
fn realistic(store: &Store) -> Vec<(String, Vec<Event>)> {
    let first = bound(store, "gpt2-small-lr3e-4-seed17", &route(), "attempt-001");
    let second = bound(store, "gpt2-small-lr3e-4-seed17", &route(), "attempt-002");
    let local = json!({"enabled":false,"reason":"NATIVE_LOCAL_ONLY"});
    let third = bound(store, "gpt2-small-lr1e-3-seed17", &local, "attempt-001");
    let mut journal = Vec::new();
    for (scope, steps) in [(&first, 60u32), (&second, 25), (&third, 30)] {
        let training = lifecycle("TRAINING", "RUNNING");
        let mut events = vec![event("worker.lifecycle.training", training)];
        for step in 1..=steps {
            let loss = 3.0 - f64::from(step) / 100.0;
            let payload = match step % 3 {
                0 => metrics(u64::from(step), loss, f64::from(step) / 100.0),
                _ => metric(u64::from(step), loss),
            };
            events.push(event(&format!("worker.metrics.{step:08}"), payload));
        }
        // An archive import of a record the worker already sent.
        let copy = events[1].payload.clone();
        events.push(event("archive.metrics.00000001", copy));
        let drained = lifecycle("METRICS_DRAINED", "SUCCEEDED");
        events.push(event("worker.lifecycle.drained", drained));
        journal.push((scope.clone(), events));
    }
    for (scope, events) in &journal {
        let run = Scope::parse(scope).unwrap().run;
        let guard = store.guard(PROJECT, &run).unwrap();
        for batch in events.chunks(16) {
            store.append(&guard, scope, batch).unwrap();
        }
    }
    let guard = store.guard(PROJECT, "gpt2-small-lr3e-4-seed17").unwrap();
    let derived = "{\"name\":\"eval/fid\",\"value\":3.0,\"unit\":\"fid\",\"step\":1}\n";
    let observations = records::derived_observations(derived.as_bytes(), &json!({})).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(derived.as_bytes()));
    store
        .append_derived(&guard, &first, &sha256, AT, &observations)
        .unwrap();
    let state = json!({"state":{"cursor":3},"remote_finished":true,"ever_opened":true});
    store
        .save_publication_state(&guard, &first, &state, "FINISHED", None, AT)
        .unwrap();
    let status = store.scope_status(&guard, &first).unwrap().unwrap();
    let wandb_id = status["wandb_id"].as_str().unwrap().to_owned();
    let total = store.total(&guard, &first).unwrap();
    store
        .advance_publication(&guard, &first, &wandb_id, total, true, AT)
        .unwrap();
    let names = "eval/accuracy\n";
    let sha256 = format!("{:x}", Sha256::digest(names.as_bytes()));
    store
        .rebuild_wandb(&guard, &first, &sha256, names, AT, false)
        .unwrap();
    let mlflow = json!({"confirmed":12,"status":"MLFLOW_PENDING","run_id":"mlflow-run"});
    store.save_mlflow_state(&guard, &first, &mlflow).unwrap();
    store
        .publication_failure(&guard, &second, "W&B timed out", AT)
        .unwrap();
    journal
}

fn sorted(mut rows: Vec<Value>) -> Value {
    rows.sort_by_key(|row| row["id"].as_str().unwrap_or_default().to_owned());
    json!(rows)
}

/// Every public read of every scope, as text.
fn snapshot(store: &Store, scopes: &[String]) -> Vec<(String, String)> {
    let runs = json!({"wandb":store.publication_runs().unwrap(),"mlflow":store.mlflow_runs().unwrap()});
    let mut reads = vec![("runs".to_owned(), text(&runs))];
    for scope in scopes {
        let parsed = Scope::parse(scope).unwrap();
        let attempt = parsed.attempt.as_str().unwrap();
        let guard = store.guard(PROJECT, &parsed.run).unwrap();
        let total = store.total(&guard, scope).unwrap();
        let events = store.events(&guard, scope, 0, 10_000).unwrap();
        let events: Vec<Value> = events.iter().map(EventRow::document).collect();
        let bounded = store.events_bounded(&guard, scope, 3, 10_000, 4096).unwrap();
        let bounded: Vec<Value> = bounded.iter().map(EventRow::document).collect();
        let derived = store.derived_batches(&guard, scope).unwrap();
        let derived: Vec<Value> = derived.iter().map(DerivedBatch::document).collect();
        let rebuilds = store.wandb_rebuilds(&guard, scope).unwrap();
        let rebuilds: Vec<Value> = rebuilds.iter().map(WandbRebuild::document).collect();
        let view = json!({
            "workspace": store.workspace_id(&guard).unwrap(),
            "binding": store.binding(&guard).unwrap(),
            "run_scopes": sorted(store.run_scopes(&guard).unwrap()),
            "pending": sorted(store.pending(&guard).unwrap()),
            "total": total,
            "events": events,
            "bounded": bounded,
            "first_id": store.event_id(&guard, scope, 1).unwrap(),
            "last_id": store.event_id(&guard, scope, total).unwrap(),
            "metrics": store.metric_records(&guard, attempt).unwrap(),
            "metrics_bounded": store.metric_records_bounded(&guard, attempt, 1 << 24, 1 << 20).unwrap(),
            "lifecycle": store.latest_lifecycle(&guard, scope).unwrap(),
            "worker": store.latest_worker_lifecycle(&guard, scope).unwrap(),
            "latest": store.latest_metric_observation(&guard, scope).unwrap(),
            "status": store.scope_status(&guard, scope).unwrap(),
            "publication": store.publication_state(&guard, scope).unwrap(),
            "mlflow": store.mlflow_state(&guard, scope).unwrap(),
            "derived": derived,
            "derived_after": store.derived_after(&guard, scope, 0).unwrap(),
            "rebuilds": rebuilds,
            "display": store.display(&guard, scope, total).unwrap(),
        });
        reads.push((scope.clone(), text(&view)));
    }
    reads
}

fn names(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn options(root: &Path, dry_run: bool) -> migrate::Options {
    migrate::Options {
        registry_root: root.to_path_buf(),
        dry_run,
    }
}

#[test]
fn migration_keeps_every_read_resumes_after_a_crash_and_dry_runs_change_nothing() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let store = Store::open(root, SEED).unwrap();
    let journal = realistic(&store);
    let scopes: Vec<String> = journal.iter().map(|(scope, _)| scope.clone()).collect();
    let before = snapshot(&store, &scopes);
    // Every event but the archive copy, which content dedup skipped.
    let events: usize = journal.iter().map(|(_, rows)| rows.len() - 1).sum();
    let tracking = root.join("tracking");
    let path = tracking.join("outbox.sqlite");
    let legacy = root.join("legacy.sqlite");
    downgrade(&path, &legacy);
    drop(store);
    fs::rename(&legacy, &path).unwrap();
    // The old file holds the full canonical text of every appended payload.
    let connection = Connection::open(&path).unwrap();
    for (scope, rows) in &journal {
        for event in rows {
            if !event.event_id.starts_with("worker.") {
                continue;
            }
            let stored: String = connection
                .query_row(
                    "SELECT payload FROM events WHERE scope=? AND event_id=?",
                    params![scope, event.event_id],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(stored, records::encoded(&event.payload).unwrap());
        }
    }
    drop(connection);
    let original = fs::read(&path).unwrap();

    // An old-layout journal is refused, never opened or changed in place.
    assert!(super::predates_compact_layout(root));
    assert!(matches!(Store::open(root, SEED), Err(Error::LegacyLayout)));
    // A running service holds the fence, and the migration refuses.
    let fence = RunFence::new(root);
    let held = fence.shared().unwrap();
    assert!(matches!(
        migrate::run(&options(root, false), &mut |_| {}),
        Err(Error::Refused(_))
    ));
    drop(held);
    assert_eq!(fs::read(&path).unwrap(), original);

    // A dry run discards an unfinished build, builds, verifies and swaps nothing.
    let listing = names(&tracking);
    fs::write(tracking.join("outbox.sqlite.compact-building"), b"crashed").unwrap();
    let mut lines = Vec::new();
    let mut record = |line: &str| lines.push(line.to_owned());
    let report = migrate::run(&options(root, true), &mut record).unwrap();
    assert!(lines.iter().any(|line| line.starts_with("events: ")));
    assert!(report.new_bytes < report.old_bytes, "{report:?}");
    assert_eq!(report.events, events as u64);
    assert!(report.backup.is_none());
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_eq!(names(&tracking), listing);

    // A crash after verification, then another after the backup name.
    let verified = migrate::run_until(&options(root, false), &mut |_| {}, true).unwrap();
    assert!(verified.backup.is_none());
    assert!(tracking.join("outbox.sqlite.compact-verified").exists());
    assert_eq!(fs::read(&path).unwrap(), original);
    let backup = tracking.join("outbox.sqlite.pre-compact-19700101T000000Z");
    fs::hard_link(&path, &backup).unwrap();
    let report = migrate::run(&options(root, false), &mut |_| {}).unwrap();
    assert_eq!(report.backup.as_deref(), Some(backup.as_path()));
    assert_eq!(fs::read(&backup).unwrap(), original);
    let backups = names(&tracking)
        .into_iter()
        .filter(|name| name.contains("pre-compact"))
        .count();
    assert_eq!(backups, 1);
    assert!(!tracking.join("outbox.sqlite.compact-verified").exists());
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let connection = Connection::open(&path).unwrap();
    let free: i64 = connection
        .query_row("PRAGMA freelist_count", [], |row| row.get(0))
        .unwrap();
    assert_eq!(free, 0, "the migrated file needs no VACUUM");
    drop(connection);
    assert!(!super::predates_compact_layout(root));
    let again = migrate::run(&options(root, false), &mut |_| {}).unwrap();
    assert!(again.already_migrated);

    // Every public read is unchanged, and so are idempotency and dedup.
    let store = Store::open(root, SEED).unwrap();
    assert_eq!(snapshot(&store, &scopes), before);
    for (scope, rows) in &journal {
        let run = Scope::parse(scope).unwrap().run;
        let guard = store.guard(PROJECT, &run).unwrap();
        let total = store.total(&guard, scope).unwrap();
        assert_eq!(store.append(&guard, scope, rows).unwrap(), total);
        let copy = event("copy.metrics.1", rows[1].payload.clone());
        assert_eq!(store.append(&guard, scope, &[copy]).unwrap(), total);
        let changed = event(&rows[1].event_id, metric(999, 0.1));
        assert!(matches!(
            store.append(&guard, scope, &[changed]),
            Err(Error::EventConflict)
        ));
    }
}

/// A registry whose journal is written in the old layout, as the old server
/// wrote it, one live batch at a time.
fn legacy_registry(root: &Path, journal: &[(String, Vec<Event>)]) -> PathBuf {
    let tracking = root.join("tracking");
    fs::create_dir_all(&tracking).unwrap();
    fs::set_permissions(&tracking, fs::Permissions::from_mode(0o700)).unwrap();
    let path = tracking.join("outbox.sqlite");
    let connection = Connection::open(&path).unwrap();
    connection.execute_batch(LEGACY_SCHEMA).unwrap();
    connection
        .execute("INSERT INTO meta VALUES ('workspace',?)", [SEED])
        .unwrap();
    interleaved(journal, |scope, batch| {
        connection.execute_batch("BEGIN").unwrap();
        legacy_append(&connection, scope, batch);
        connection.execute_batch("COMMIT").unwrap();
    });
    drop(connection);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    path
}

/// Batches of every scope in turn, as concurrent Attempts stream them.
fn interleaved(journal: &[(String, Vec<Event>)], mut write: impl FnMut(&str, &[Event])) {
    let longest = journal.iter().map(|(_, rows)| rows.len()).max().unwrap();
    for start in (0..longest).step_by(20) {
        for (scope, events) in journal {
            if start < events.len() {
                let end = (start + 20).min(events.len());
                write(scope, &events[start..end]);
            }
        }
    }
}

/// Worker metric rows with production-sized IDs, and a lifecycle row now and then.
fn production_like(scopes: &[String], per_scope: u64) -> Vec<(String, Vec<Event>)> {
    let mut journal = Vec::new();
    for scope in scopes {
        let mut events = Vec::new();
        for step in 1..=per_scope {
            let id = format!("worker.metrics.{step:016}.{}", "c".repeat(40));
            let loss = 3.0 / (1.0 + step as f64);
            let payload = match step % 4 {
                0 => metrics(step, loss, 1.0 - loss / 3.0),
                _ => metric(step, loss),
            };
            events.push(event(&id, payload));
            if step % 50 == 0 {
                let id = format!("worker.lifecycle.{step:016}.{}", "d".repeat(40));
                events.push(event(&id, lifecycle("TRAINING", "RUNNING")));
            }
        }
        journal.push((scope.clone(), events));
    }
    journal
}

#[test]
fn keyed_compact_layout_stores_fewer_bytes_per_event() {
    let runs = [
        "gpt2-small-lr3e-4-warmup2000-seed17",
        "gpt2-small-lr1e-3-warmup2000-s7",
    ];
    let temp = tempfile::tempdir().unwrap();
    let keyed_root = temp.path().join("keyed");
    fs::create_dir(&keyed_root).unwrap();
    let store = Store::open(&keyed_root, SEED).unwrap();
    let mut scopes = Vec::new();
    for run in runs {
        for attempt in ["attempt-001", "attempt-002"] {
            scopes.push(bound(&store, run, &route(), attempt));
        }
    }
    let journal = production_like(&scopes, 1500);
    let events: usize = journal.iter().map(|(_, rows)| rows.len()).sum();
    interleaved(&journal, |scope, batch| {
        let run = Scope::parse(scope).unwrap().run;
        let guard = store.guard(PROJECT, &run).unwrap();
        store.append(&guard, scope, batch).unwrap();
    });
    let keyed = fs::metadata(store.path()).unwrap().len();
    drop(store);

    let legacy_root = temp.path().join("legacy");
    let legacy = legacy_registry(&legacy_root, &journal);
    let old = fs::metadata(&legacy).unwrap().len();
    let report = migrate::run(&options(&legacy_root, false), &mut |_| {}).unwrap();
    assert_eq!(report.events, events as u64);
    let migrated = fs::metadata(&legacy).unwrap().len();

    let each = |bytes: u64| bytes as f64 / events as f64;
    let (old_each, keyed_each, migrated_each) = (each(old), each(keyed), each(migrated));
    let line = format!(
        "journal bytes per event over {events} events: old layout {old_each:.0}, \
         keyed compact {keyed_each:.0}, migrated {migrated_each:.0}\n"
    );
    // Written past the test harness capture, so CI logs carry the measurement.
    std::io::stderr().write_all(line.as_bytes()).unwrap();
    assert!(keyed * 10 < old * 7, "{line}");
    assert!(migrated <= keyed, "{line}");
}
