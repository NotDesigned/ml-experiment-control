use std::{
    fs,
    path::Path,
    sync::{Arc, mpsc},
    time::Duration,
};

use ml_exp_core::records::{Event, JsonlImporter, Scope, normalized_metric};
use rusqlite::{Connection, params};
use serde_json::{Value, json};

use super::{Error, RemoteHistoryEvidence, RemoteObservation, Store};
use crate::deletion::RunFence;

const SEED: &str = "0123456789abcdef0123456789abcdef";
fn metric(value: f64, step: u64) -> Value {
    normalized_metric(
        &json!({"name":"custom/loss","unit":"nats/token","value":value,"step":step}),
        &json!({}),
    )
    .unwrap()
}
fn event(id: &str, payload: Value) -> Event {
    Event {
        event_id: id.into(),
        payload,
    }
}
fn bound(store: &Store, run: &str, route: Value, config: Value) -> String {
    let guard = store.guard("demo", run).unwrap();
    store.bind(&guard, &route, &config).unwrap();
    store.scope(&guard, "attempt-001").unwrap()
}
fn proof(
    store: &Store,
    guard: &super::RecordGuard<'_>,
    scope: &str,
    through: i64,
    display: bool,
) -> RemoteHistoryEvidence {
    let state = store.scope_status(guard, scope).unwrap().unwrap();
    let first = state["confirmed"].as_i64().unwrap().max(1);
    let observed = (first..=through)
        .map(|sequence| RemoteObservation {
            sequence,
            event_id: store.event_id(guard, scope, sequence).unwrap().unwrap(),
        })
        .collect();
    RemoteHistoryEvidence {
        route: store.binding(guard).unwrap().unwrap()["route"].clone(),
        wandb_id: state["wandb_id"].as_str().unwrap().into(),
        through,
        observed,
        display_version: display.then_some(2),
        display_through: display.then_some(through),
    }
}
fn enabled() -> Value {
    json!({"enabled":true,"entity":"fixture-team","project":"fixture-science"})
}

#[test]
fn frozen_binding_batch_atomicity_and_metric_dedup_preserve_event_identities() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let scope = bound(&store, "run-a", enabled(), json!({}));
    let guard = store.guard("demo", "run-a").unwrap();
    let payload = metric(2.5, 1);
    assert_eq!(
        store
            .append(&guard, &scope, &[event("m1", payload.clone())])
            .unwrap(),
        1
    );
    assert_eq!(
        store
            .append(
                &guard,
                &scope,
                &[event("m1", payload.clone()), event("archive-copy", payload)]
            )
            .unwrap(),
        1
    );
    assert_eq!(
        store.events(&guard, &scope, 0, 128).unwrap()[0].event_id,
        "m1"
    );
    assert_eq!(
        store.metric_records(&guard, "attempt-001").unwrap()[0]["value"],
        2.5
    );
    assert!(matches!(
        store.append(
            &guard,
            &scope,
            &[
                event("new", json!({"kind":"lifecycle","data":{}})),
                event("m1", metric(9.0, 1))
            ]
        ),
        Err(Error::EventConflict)
    ));
    assert_eq!(store.total(&guard, &scope).unwrap(), 1);
    assert!(matches!(
        store.append(&guard, &scope, &[event("archive-copy", metric(9.0, 1))]),
        Err(Error::EventConflict)
    ));
    // The skipped content duplicate still permanently owns its event ID.
    let connection = Connection::open(store.path()).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM event_ids", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert!(matches!(
        store.bind(&guard, &json!({"enabled":false}), &json!({})),
        Err(Error::FrozenIdentity)
    ));
    let unknown = Scope::new("demo", "run-a", "attempt-999")
        .encoded()
        .unwrap();
    assert!(matches!(
        store.append(&guard, &unknown, &[event("orphan", metric(1.0, 1))]),
        Err(Error::Scope)
    ));
    let second = Store::open(temp.path(), "ffffffffffffffffffffffffffffffff").unwrap();
    assert!(matches!(
        second.append(&guard, &scope, &[event("wrong-store", metric(1.0, 1))]),
        Err(Error::Guard)
    ));
}

#[test]
fn publication_requires_exact_remote_prefix_and_ack_never_deletes_canonical_records() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let scope = bound(&store, "run-a", enabled(), json!({}));
    let guard = store.guard("demo", "run-a").unwrap();
    store
        .append(
            &guard,
            &scope,
            &[
                event("m1", metric(2.5, 1)),
                event(
                    "state",
                    json!({"kind":"lifecycle","data":{"state":"RUNNING"}}),
                ),
            ],
        )
        .unwrap();
    assert_eq!(store.pending(&guard).unwrap().len(), 1);
    let valid = proof(&store, &guard, &scope, 1, false);
    let mut invalid = Vec::new();
    let mut p = valid.clone();
    p.route["project"] = json!("other");
    invalid.push(p);
    let mut p = valid.clone();
    p.wandb_id = "foreign".into();
    invalid.push(p);
    let mut p = valid.clone();
    p.through = 3;
    invalid.push(p);
    let mut p = valid.clone();
    p.observed.clear();
    invalid.push(p);
    let mut p = valid.clone();
    p.observed[0].event_id = "foreign".into();
    invalid.push(p);
    let mut p = valid.clone();
    p.observed.push(p.observed[0].clone());
    invalid.push(p);
    let mut p = valid.clone();
    p.display_version = Some(999);
    p.display_through = Some(1);
    invalid.push(p);
    for p in invalid {
        assert!(matches!(
            store.confirm_publication(&guard, &scope, &p, None, "bad"),
            Err(Error::Confirmation)
        ));
        assert_eq!(
            store.scope_status(&guard, &scope).unwrap().unwrap()["confirmed"],
            0
        );
    }
    store
        .confirm_publication(&guard, &scope, &valid, Some("REMOTE_ACK_PENDING"), "t1")
        .unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["confirmed"],
        1
    );
    let complete = proof(&store, &guard, &scope, 2, false);
    store
        .confirm_publication(&guard, &scope, &complete, None, "t2")
        .unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["status"],
        "SYNCED"
    );
    assert_eq!(store.pending(&guard).unwrap().len(), 1); // Display repair remains pending.
    let display = proof(&store, &guard, &scope, 2, true);
    store
        .confirm_publication(&guard, &scope, &display, None, "t3")
        .unwrap();
    assert!(store.pending(&guard).unwrap().is_empty());
    let before = store.events(&guard, &scope, 0, 128).unwrap();
    assert!(matches!(
        store.confirm_publication(&guard, &scope, &valid, None, "stale"),
        Err(Error::Confirmation)
    ));
    assert_eq!(store.events(&guard, &scope, 0, 128).unwrap(), before);
    store
        .append(&guard, &scope, &[event("m2", metric(1.5, 2))])
        .unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["status"],
        "PENDING"
    );
    store
        .publication_failure(&guard, &scope, "PUBLISHER_UNAVAILABLE", "t4")
        .unwrap();
    let state = store.scope_status(&guard, &scope).unwrap().unwrap();
    assert_eq!(state["confirmed"], 2);
    assert_eq!(state["failures"], 1);
    assert_eq!(store.total(&guard, &scope).unwrap(), 3);
    assert_eq!(
        store.metric_records(&guard, "attempt-001").unwrap().len(),
        2
    );
}

#[test]
fn deletion_fences_proven_companions_and_retains_destination_schema_and_independent_suffix_runs() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    for run in ["run-a", "run-a--preparation", "run-a--backfill"] {
        let config = if run == "run-a" {
            json!({})
        } else {
            json!({"ml_expd":{"run_id":"run-a","record_type":run.rsplit("--").next().unwrap()}})
        };
        let scope = bound(&store, run, enabled(), config);
        let guard = store.guard("demo", run).unwrap();
        store
            .append(&guard, &scope, &[event("m1", metric(2.5, 1))])
            .unwrap();
    }
    let connection = Connection::open(store.path()).unwrap();
    connection
        .execute(
            "INSERT INTO display_metrics VALUES (?,?,?,?)",
            params!["destination", "signature", "loss", "{}"],
        )
        .unwrap();
    assert!(matches!(
        store.delete_run("demo", "run-a", "t0"),
        Err(Error::DeletionNotStarted)
    ));
    RunFence::new(temp.path())
        .mark("demo", "run-a", &serde_json::Map::new(), "t1")
        .unwrap();
    assert_eq!(
        store.delete_run("demo", "run-a", "t2").unwrap()["scopes_removed"],
        3
    );
    assert_eq!(
        store.delete_run("demo", "run-a", "t3").unwrap()["scopes_removed"],
        0
    );
    for run in ["run-a", "run-a--preparation", "run-a--backfill"] {
        assert!(store.guard("demo", run).is_err());
    }
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM display_metrics", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    for run in ["trial", "trial--preparation", "trial--backfill"] {
        let scope = bound(
            &store,
            run,
            json!({"enabled":false}),
            json!({"ml_expd":{"run_id":run}}),
        );
        let guard = store.guard("demo", run).unwrap();
        store
            .append(
                &guard,
                &scope,
                &[event("real", json!({"kind":"lifecycle","data":{}}))],
            )
            .unwrap();
    }
    RunFence::new(temp.path())
        .mark("demo", "trial", &serde_json::Map::new(), "t4")
        .unwrap();
    assert_eq!(
        store.delete_run("demo", "trial", "t5").unwrap()["scopes_removed"],
        1
    );
    for run in ["trial--preparation", "trial--backfill"] {
        let guard = store.guard("demo", run).unwrap();
        let scope = store.scope(&guard, "attempt-001").unwrap();
        store
            .append(
                &guard,
                &scope,
                &[event("late", json!({"kind":"lifecycle","data":{}}))],
            )
            .unwrap();
        assert_eq!(store.total(&guard, &scope).unwrap(), 2);
    }
    let guard = store.guard("demo", "trial--newalias").unwrap();
    assert!(matches!(
        store.bind(
            &guard,
            &enabled(),
            &json!({"ml_expd":{"run_id":"trial","record_type":"backfill"}})
        ),
        Ok(())
    )); // No suffix proof.
}

#[test]
fn new_companion_bind_cannot_bypass_an_already_fenced_parent() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    RunFence::new(temp.path())
        .mark("demo", "parent", &serde_json::Map::new(), "t1")
        .unwrap();
    let guard = store.guard("demo", "parent--preparation").unwrap();
    assert!(
        store
            .bind(
                &guard,
                &enabled(),
                &json!({"ml_expd":{"run_id":"parent","record_type":"preparation"}})
            )
            .is_err()
    );
    assert!(store.binding(&guard).unwrap().is_none());
}

#[test]
fn readonly_constructor_and_reads_do_not_create_or_migrate_journal_state() {
    let temp = tempfile::tempdir().unwrap();
    let absent = temp.path().join("absent");
    let missing = Store::read_only(&absent);
    assert!(missing.guard("demo", "run-a").is_err());
    assert!(!absent.exists());
    let store = Store::open(temp.path(), SEED).unwrap();
    let scope = bound(&store, "run-a", json!({"enabled":false}), json!({}));
    {
        let guard = store.guard("demo", "run-a").unwrap();
        store
            .append(&guard, &scope, &[event("m1", metric(1.0, 1))])
            .unwrap();
    }
    let before = fs::read(store.path()).unwrap();
    let readonly = Store::read_only(temp.path());
    let guard = readonly.guard("demo", "run-a").unwrap();
    assert_eq!(readonly.events(&guard, &scope, 0, 128).unwrap().len(), 1);
    assert!(matches!(
        readonly.scope(&guard, "attempt-002"),
        Err(Error::ReadOnly)
    ));
    assert!(matches!(
        readonly.bind(&guard, &enabled(), &json!({})),
        Err(Error::ReadOnly)
    ));
    assert_eq!(before, fs::read(store.path()).unwrap());
    drop(guard);
    fs::remove_file(temp.path().join(".run-deletion.lock")).unwrap();
    assert!(readonly.guard("demo", "run-a").is_err());
    assert!(store.guard("demo", "run-a").is_err());
    assert!(!temp.path().join(".run-deletion.lock").exists());
}

#[test]
fn last_received_metrics_and_worker_stage_do_not_invent_freshness_or_highest_step() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let scope = bound(&store, "run-a", enabled(), json!({}));
    let guard = store.guard("demo", "run-a").unwrap();
    store
        .append(
            &guard,
            &scope,
            &[
                event(
                    "worker.start",
                    json!({"kind":"lifecycle","data":{"phase":"TRAINING"}}),
                ),
                event("m100", metric(1.0, 100)),
                event("m2", metric(2.0, 2)),
                event(
                    "scheduler.ready",
                    json!({"kind":"lifecycle","data":{"phase":"PREPARING_DATA","state":"RUNNING"}}),
                ),
            ],
        )
        .unwrap();
    let worker = store
        .latest_worker_lifecycle(&guard, &scope)
        .unwrap()
        .unwrap();
    assert_eq!(
        worker,
        json!({"phase":"TRAINING","timestamp":null,"event_id":"worker.start","sequence":1})
    );
    let metric = store
        .latest_metric_observation(&guard, &scope)
        .unwrap()
        .unwrap();
    assert_eq!(metric["observation"]["step"], 2);
    assert_eq!(metric["observation"]["timestamp"], Value::Null);
    assert_eq!(metric["sequence"], 3);
    assert_eq!(
        store.latest_lifecycle(&guard, &scope).unwrap()["state"],
        "RUNNING"
    );
}

fn ingest(store: &Store, root_run: &str, scope: &str, bytes: &[u8]) -> Value {
    let mut importer = JsonlImporter::new("metrics.jsonl", json!({})).unwrap();
    for chunk in bytes.chunks(JsonlImporter::INPUT_BOUND) {
        let (next, emission) = importer.feed(chunk).unwrap();
        let guard = store.guard("demo", root_run).unwrap();
        store.append_import(&guard, scope, &emission).unwrap();
        importer = next;
    }
    let emission = importer.finish().unwrap();
    let guard = store.guard("demo", root_run).unwrap();
    store
        .append_import(&guard, scope, &emission)
        .unwrap()
        .document()
}
#[test]
fn archive_import_batches_long_histories_and_replay_never_duplicates_records() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let scope = bound(&store, "run-a", json!({"enabled":false}), json!({}));
    let mut body = String::new();
    for step in 0..65 {
        body.push_str(
            &json!({"name":"loss","unit":"nats/token","step":step,"value":1}).to_string(),
        );
        body.push('\n');
    }

    assert_eq!(
        ingest(&store, "run-a", &scope, body.as_bytes()),
        json!({"status":"COMPLETE","records":65,"invalid_lines":0})
    );
    let guard = store.guard("demo", "run-a").unwrap();
    assert_eq!(store.total(&guard, &scope).unwrap(), 65);
    drop(guard);
    assert_eq!(
        ingest(&store, "run-a", &scope, body.as_bytes())["records"],
        65
    );
    let guard = store.guard("demo", "run-a").unwrap();
    assert_eq!(
        store.metric_records(&guard, "attempt-001").unwrap().len(),
        65
    );
}

#[cfg(unix)]
fn replace_lock(root: &Path) {
    use std::os::unix::fs::PermissionsExt;
    fs::rename(
        root.join(".run-deletion.lock"),
        root.join("old-run-deletion.lock"),
    )
    .unwrap();
    fs::write(root.join(".run-deletion.lock"), b"").unwrap();
    fs::set_permissions(
        root.join(".run-deletion.lock"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
}
#[cfg(unix)]
#[test]
fn replaced_fence_lock_blocks_both_reads_and_queued_sqlite_writes() {
    let temp = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::open(temp.path(), SEED).unwrap());
    let scope = bound(&store, "run-a", enabled(), json!({}));
    let blocker = Connection::open(store.path()).unwrap();
    blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
    let (send, receive) = mpsc::channel();
    let cloned = Arc::clone(&store);
    let worker_scope = scope.clone();
    let worker = std::thread::spawn(move || {
        let guard = cloned.guard("demo", "run-a").unwrap();
        send.send(()).unwrap();
        cloned.append(&guard, &worker_scope, &[event("late", metric(1.0, 1))])
    });
    receive.recv_timeout(Duration::from_secs(5)).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    assert!(!worker.is_finished());
    replace_lock(temp.path());
    RunFence::new(temp.path())
        .mark(
            "demo",
            "run-a",
            &serde_json::Map::new(),
            "fenced-on-new-lock",
        )
        .unwrap();
    blocker.execute_batch("ROLLBACK").unwrap();
    assert!(worker.join().unwrap().is_err());
    assert_eq!(
        blocker
            .query_row("SELECT COUNT(*) FROM events", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    let other = Store::open(temp.path().join("other").as_path(), SEED).unwrap();
    let other_scope = bound(&other, "live", enabled(), json!({}));
    let guard = other.guard("demo", "live").unwrap();
    replace_lock(&temp.path().join("other"));
    assert!(other.events(&guard, &other_scope, 0, 128).is_err());
}

fn reference() -> Value {
    serde_json::from_str(include_str!("../../../../compatibility/records-store.json")).unwrap()
}
fn unhex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn verified_archive_import_matches_original_jsonl_framing_and_nonfinite_diagnoses() {
    let fixture = reference();
    let mut count = 0;
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["operation"] == "import")
    {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path(), SEED).unwrap();
        let scope = bound(&store, "run-a", json!({"enabled":false}), json!({}));
        let mut counts = super::ImportSummary::default();
        for name in ["metrics.jsonl", "train_metrics.jsonl"] {
            let file = case["input"]["files"]
                .as_array()
                .unwrap()
                .iter()
                .find(|file| file["name"] == name);
            let Some(file) = file else {
                continue;
            };
            let mut importer = JsonlImporter::new(name, case["input"]["manifest"].clone()).unwrap();
            let bytes = unhex(file["bytes_hex"].as_str().unwrap());
            for chunk in bytes.chunks(JsonlImporter::INPUT_BOUND) {
                let (next, emission) = importer.feed(chunk).unwrap();
                let guard = store.guard("demo", "run-a").unwrap();
                store.append_import(&guard, &scope, &emission).unwrap();
                importer = next;
            }
            let emission = importer.finish().unwrap();
            let guard = store.guard("demo", "run-a").unwrap();
            let summary = store.append_import(&guard, &scope, &emission).unwrap();
            counts.records += summary.records;
            counts.invalid_lines += summary.invalid_lines;
        }
        let guard = store.guard("demo", "run-a").unwrap();
        let events = store
            .events(&guard, &scope, 0, 10000)
            .unwrap()
            .iter()
            .map(super::EventRow::document)
            .collect::<Vec<_>>();
        let actual = json!({"result":counts.document(),"scope":scope,"events":events,"metric_records":store.metric_records(&guard,"attempt-001").unwrap()});
        assert_eq!(actual, case["expected"], "{}", case["id"]);
        count += 1;
    }
    assert_eq!(count, 35);
}

#[test]
fn original_sqlite_schema_is_read_without_mutation_and_additively_migrated() {
    let fixture = reference();
    let legacy = &fixture["legacy_database"];
    let temp = tempfile::tempdir().unwrap();
    fs::create_dir(temp.path().join("tracking")).unwrap();
    let path = temp.path().join("tracking/outbox.sqlite");
    let connection = Connection::open(&path).unwrap();
    for sql in legacy["schema"].as_array().unwrap() {
        connection.execute_batch(sql.as_str().unwrap()).unwrap();
    }
    for (table, rows) in legacy["tables"].as_object().unwrap() {
        for row in rows.as_array().unwrap() {
            let row = row.as_object().unwrap();
            let keys = row.keys().map(|s| s.as_str()).collect::<Vec<_>>();
            let placeholders = vec!["?"; keys.len()].join(",");
            let values = row
                .values()
                .map(|value| match value {
                    Value::Null => rusqlite::types::Value::Null,
                    Value::String(s) => rusqlite::types::Value::Text(s.clone()),
                    Value::Number(n) => rusqlite::types::Value::Integer(n.as_i64().unwrap()),
                    _ => panic!("unexpected SQLite reference field"),
                })
                .collect::<Vec<_>>();
            connection
                .execute(
                    &format!(
                        "INSERT INTO {table} ({}) VALUES ({placeholders})",
                        keys.join(",")
                    ),
                    rusqlite::params_from_iter(values),
                )
                .unwrap();
        }
    }
    drop(connection);
    // Existing compatible lock is required; readonly storage never creates it.
    {
        let fence = RunFence::new(temp.path());
        let _guard = fence.shared().unwrap();
    }
    let before = fs::read(&path).unwrap();
    let readonly = Store::read_only(temp.path());
    let scope = legacy["scope"].as_str().unwrap();
    {
        let guard = readonly.guard("demo", "run-a").unwrap();
        assert_eq!(
            readonly.events(&guard, scope, 0, 128).unwrap()[0].event_id,
            "original"
        );
    }
    assert_eq!(fs::read(&path).unwrap(), before);
    let store = Store::open(temp.path(), "ffffffffffffffffffffffffffffffff").unwrap();
    let guard = store.guard("demo", "run-a").unwrap();
    let state = store.scope_status(&guard, scope).unwrap().unwrap();
    assert_eq!(state["display_version"], 0);
    assert_eq!(state["display_confirmed"], 0);
    let connection = Connection::open(store.path()).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT value FROM meta WHERE key='workspace'", [], |r| {
                r.get::<_, String>(0)
            })
            .unwrap(),
        SEED
    );
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    for (table, rows) in legacy["tables"].as_object().unwrap() {
        let mut statement = connection
            .prepare(&format!("SELECT * FROM {table}"))
            .unwrap();
        let keys = statement
            .column_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let actual = statement
            .query_map([], |row| {
                let mut result = serde_json::Map::new();
                for (i, key) in keys.iter().enumerate() {
                    if ["display_version", "display_confirmed"].contains(&key.as_str()) {
                        continue;
                    }
                    let value = match row.get_ref(i)? {
                        rusqlite::types::ValueRef::Null => Value::Null,
                        rusqlite::types::ValueRef::Integer(v) => json!(v),
                        rusqlite::types::ValueRef::Text(s) => {
                            json!(std::str::from_utf8(s).unwrap())
                        }
                        _ => panic!("unexpected type"),
                    };
                    result.insert(key.clone(), value);
                }
                Ok(Value::Object(result))
            })
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual, *rows.as_array().unwrap(), "{table}");
    }
}

#[test]
fn historical_import_freezes_schema_and_never_enables_publication() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let guard = store.guard("demo", "history").unwrap();
    let manifest = json!({"resolved_config":{"evaluation":{"protocol_id":"protocol-historical","metrics_schema":{"definitions":{"loss":{"unit":"u"}}}}}});
    let (scope, config) = store
        .prepare_import(&guard, "attempt-001", &manifest)
        .unwrap();
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["status"],
        "LOCAL_ONLY"
    );
    let metric = normalized_metric(&json!({"loss":1}), &config).unwrap();
    assert_eq!(
        metric["observations"][0]["protocol_id"],
        "protocol-historical"
    );
    assert_eq!(metric["observations"][0]["unit"], "u");
    let original = store.binding(&guard).unwrap();
    let second = store
        .prepare_import(
            &guard,
            "attempt-001",
            &json!({"evaluation":{"protocol_id":"other"}}),
        )
        .unwrap();
    assert_eq!(second.1, config);
    assert_eq!(store.binding(&guard).unwrap(), original);
}

#[cfg(unix)]
#[test]
fn journal_namespace_replacement_is_not_treated_as_a_fresh_empty_store() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let scope = bound(&store, "run-a", enabled(), json!({}));
    let guard = store.guard("demo", "run-a").unwrap();
    store
        .append(&guard, &scope, &[event("m1", metric(1.0, 1))])
        .unwrap();
    let old = temp.path().join("detached.sqlite");
    fs::rename(store.path(), &old).unwrap();
    fs::copy(&old, store.path()).unwrap();
    assert!(store.events(&guard, &scope, 0, 128).is_err());
    assert!(
        store
            .append(&guard, &scope, &[event("m2", metric(2.0, 2))])
            .is_err()
    );
    let connection = Connection::open(store.path()).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM events", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[cfg(unix)]
#[test]
fn new_journal_is_private_and_retains_original_delete_journal_mode() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    assert_eq!(
        fs::metadata(store.path()).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(
        fs::metadata(store.path().parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    let connection = Connection::open(store.path()).unwrap();
    assert_eq!(
        connection
            .query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
            .unwrap(),
        "delete"
    );
}

#[test]
fn missing_fence_after_open_is_never_repaired_by_normal_reads_writes_or_deletion() {
    for leaf in [".run-deletion.lock", "deleted-runs"] {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path(), SEED).unwrap();
        let scope = bound(&store, "run-a", enabled(), json!({}));
        {
            let guard = store.guard("demo", "run-a").unwrap();
            store
                .append(&guard, &scope, &[event("m1", metric(1.0, 1))])
                .unwrap();
        }
        RunFence::new(temp.path())
            .mark("demo", "run-a", &serde_json::Map::new(), "deleted")
            .unwrap();
        let path = temp.path().join(leaf);
        fs::rename(&path, temp.path().join("removed-fence-evidence")).unwrap();
        assert!(store.guard("demo", "run-a").is_err());
        assert!(store.delete_run("demo", "run-a", "cleanup").is_err());
        assert!(!path.exists(), "{leaf} was silently rebuilt");
        let connection = Connection::open(store.path()).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT COUNT(*) FROM events", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }
}
