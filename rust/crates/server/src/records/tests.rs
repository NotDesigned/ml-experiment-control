use std::fs;

use ml_exp_core::records::{Event, JsonlImporter, Scope, normalized_metric};
use rusqlite::Connection;
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
        display_version: display.then_some(4),
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
    let mut p = valid.clone();
    p.display_version = Some(3);
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
fn deletion_fences_proven_companions_and_keeps_independent_suffix_runs() {
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

const PLAIN: &str =
    r#"{"name": "loss", "value": 2.5, "unit": "nats/token", "step": 1, "variant_id": "a"}"#;
fn plain_line() -> Vec<u8> {
    format!("{PLAIN}\n").into_bytes()
}
fn long_history() -> Vec<u8> {
    (0..65)
        .map(|step| {
            format!(
                "{{\"name\": \"loss\", \"value\": 2.5, \"unit\": \"nats/token\", \"step\": {step}, \"variant_id\": \"a\"}}\n"
            )
        })
        .collect::<String>()
        .into_bytes()
}
fn padded_line(total: usize) -> Vec<u8> {
    let mut line = br#"{"name":"loss","value":1,"unit":"u","padding":""#.to_vec();
    line.resize(total - 3, b'x');
    line.extend(b"\"}\n");
    line
}
fn oversized_then_empty() -> Vec<u8> {
    let mut bytes = vec![b'x'; 140000];
    bytes.extend(b"\n{}\n");
    bytes
}
#[derive(Clone, Copy)]
enum Encoding {
    Utf8Bom,
    Utf16LeBom,
    Utf16Be,
    Utf32LeBom,
    Utf32Be,
}
fn encoded_as(text: &str, encoding: Encoding) -> Vec<u8> {
    match encoding {
        Encoding::Utf8Bom => [&[0xefu8, 0xbb, 0xbf][..], text.as_bytes()].concat(),
        Encoding::Utf16LeBom => [0xff, 0xfe]
            .into_iter()
            .chain(text.encode_utf16().flat_map(u16::to_le_bytes))
            .collect(),
        Encoding::Utf16Be => text.encode_utf16().flat_map(u16::to_be_bytes).collect(),
        Encoding::Utf32LeBom => [0xff, 0xfe, 0, 0]
            .into_iter()
            .chain(text.chars().map(u32::from).flat_map(u32::to_le_bytes))
            .collect(),
        Encoding::Utf32Be => text
            .chars()
            .map(u32::from)
            .flat_map(u32::to_be_bytes)
            .collect(),
    }
}
fn encoded(encoding: Encoding) -> Vec<u8> {
    encoded_as(PLAIN, encoding)
}

/// One expected journal row. A literal digest only hashes ASCII, float-free
/// content; `None` marks content holding a float or a container's text form.
struct Row {
    prefix: &'static str,
    digest: Option<&'static str>,
    payload: Value,
    unpinned: Option<&'static str>,
}
fn diagnostic_row(digest: &'static str) -> Row {
    Row {
        prefix: "invalid",
        digest: Some(digest),
        payload: json!({"kind":"diagnostic","data":{"diagnostic":"INVALID_METRIC_JSONL"}}),
        unpinned: None,
    }
}
fn metric_row(digest: Option<&'static str>, observations: Vec<Value>) -> Row {
    Row {
        prefix: "archive",
        digest,
        payload: json!({"kind":"metrics","observations":observations}),
        unpinned: None,
    }
}
/// A container's text form under `key` is proven to exist but not asserted.
fn unpinned_row(key: &'static str, mut observation: Value) -> Row {
    observation.as_object_mut().unwrap().remove(key);
    Row {
        unpinned: Some(key),
        ..metric_row(None, vec![observation])
    }
}
fn plain_observation(step: i64) -> Value {
    json!({"error":null,"errors":[],"invalid_value":null,"name":"loss","protocol_id":null,
        "source":"exact-attempt-worker","status":"VALID","step":step,"unit":"nats/token",
        "value":2.5,"variant_id":"a"})
}
fn small_observation(value: Value, status: &str, errors: &[&str], invalid: Value) -> Value {
    json!({"error":errors.first(),"errors":errors,"invalid_value":invalid,"name":"loss",
        "protocol_id":null,"source":"exact-attempt-worker","status":status,"unit":"u",
        "value":value,"variant_id":null})
}
fn failed_observation(text: &str) -> Value {
    small_observation(
        json!(null),
        "FAILED",
        &["NONFINITE_OR_MISSING_VALUE"],
        json!(text),
    )
}
struct ImportCase {
    label: &'static str,
    files: Vec<(&'static str, Vec<u8>)>,
    records: u64,
    invalid_lines: u64,
    rows: Vec<Row>,
}
fn import_cases() -> Vec<ImportCase> {
    vec![
        ImportCase {
            label: "no files at all",
            files: vec![],
            records: 0,
            invalid_lines: 0,
            rows: vec![],
        },
        ImportCase {
            label: "an empty file",
            files: vec![("metrics.jsonl", b"".to_vec())],
            records: 0,
            invalid_lines: 0,
            rows: vec![],
        },
        ImportCase {
            label: "an empty object is a record with no observations",
            files: vec![("metrics.jsonl", b"{}\n".to_vec())],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(
                Some("28076d7d10989d09910638a60ec1cf5c3dffd59ace86ff83fae5228946afde28"),
                vec![],
            )],
        },
        ImportCase {
            label: "text, a list and invalid UTF-8 are each one invalid line",
            files: vec![("metrics.jsonl", b"bad\n[]\n\xff\n".to_vec())],
            records: 0,
            invalid_lines: 3,
            rows: vec![
                diagnostic_row("dda4a6b4a446b8f22fc9c0f13a99d9e8b5b5daf0add4f4393b04496cbaa300f9"),
                diagnostic_row("d6a7b4b683ed7e5be2ebb5a2c35c8e6ef33b95edb1e49c4d49c45670bea1c8fc"),
                diagnostic_row("18c638a363f17d57a294c2650b9f114045a296924f8125680d023fe5f537bb9b"),
            ],
        },
        ImportCase {
            label: "an oversized line does not swallow the next line",
            files: vec![("train_metrics.jsonl", oversized_then_empty())],
            records: 1,
            invalid_lines: 1,
            rows: vec![
                diagnostic_row("e5f5b5bdd8524f7129b9ed16eedf1303cc5871b6d6b065e042ed0bf29515998b"),
                metric_row(
                    Some("28076d7d10989d09910638a60ec1cf5c3dffd59ace86ff83fae5228946afde28"),
                    vec![],
                ),
            ],
        },
        ImportCase {
            label: "an oversized unterminated tail is one invalid line",
            files: vec![("train_metrics.jsonl", vec![b'x'; 140000])],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "6a15996a646e59df06e5789c09d57776e8dee5abbd98e36b6f0071b0f92a192e",
            )],
        },
        ImportCase {
            label: "the same record in both files is journaled once",
            files: vec![
                ("metrics.jsonl", plain_line()),
                ("train_metrics.jsonl", plain_line()),
            ],
            records: 2,
            invalid_lines: 0,
            rows: vec![metric_row(None, vec![plain_observation(1)])],
        },
        ImportCase {
            label: "one byte under the line limit",
            files: vec![("metrics.jsonl", padded_line(65535))],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(
                Some("1e933e4e8f270ce9e430975c8d3ba4d5bfcc90fccb96a2aede35bf134da4f654"),
                vec![small_observation(json!(1), "VALID", &[], json!(null))],
            )],
        },
        ImportCase {
            label: "exactly the line limit",
            files: vec![("metrics.jsonl", padded_line(65536))],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(
                Some("1e933e4e8f270ce9e430975c8d3ba4d5bfcc90fccb96a2aede35bf134da4f654"),
                vec![small_observation(json!(1), "VALID", &[], json!(null))],
            )],
        },
        ImportCase {
            label: "one byte over the line limit",
            files: vec![("metrics.jsonl", padded_line(65537))],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "7bfb3de768f6ffdc0a85cc2eee16d80d2bea7bf1a67f4a76a0737063623479b9",
            )],
        },
        ImportCase {
            label: "twice the line limit",
            files: vec![("metrics.jsonl", padded_line(131074))],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "316a6577c96a2c3908fa25ac378960c54291341829a5374ae907562108025968",
            )],
        },
        ImportCase {
            label: "NaN",
            files: vec![(
                "metrics.jsonl",
                b"{\"name\":\"loss\",\"value\":NaN,\"unit\":\"u\"}\n".to_vec(),
            )],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(
                Some("904d3162e090bffe484c8a01f8b583355c724899f1f8840010834f5f2a23ca75"),
                vec![failed_observation("nan")],
            )],
        },
        ImportCase {
            label: "Infinity",
            files: vec![(
                "metrics.jsonl",
                b"{\"name\":\"loss\",\"value\":Infinity,\"unit\":\"u\"}\n".to_vec(),
            )],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(
                Some("665ba035990b35b8c0d6ccb2a4e5c321101094b5c5ec6f66df1226cea748079f"),
                vec![failed_observation("inf")],
            )],
        },
        ImportCase {
            label: "negative Infinity",
            files: vec![(
                "metrics.jsonl",
                b"{\"name\":\"loss\",\"value\":-Infinity,\"unit\":\"u\"}\n".to_vec(),
            )],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(
                Some("545af82388b4b19587af16d306c03c2b285c0100026fb2af2aabb485ae9f23a1"),
                vec![failed_observation("-inf")],
            )],
        },
        ImportCase {
            label: "a nonfinite value inside a list",
            files: vec![(
                "metrics.jsonl",
                b"{\"name\":\"loss\",\"value\":[NaN],\"unit\":\"u\"}\n".to_vec(),
            )],
            records: 1,
            invalid_lines: 0,
            rows: vec![unpinned_row(
                "invalid_value",
                small_observation(
                    json!(null),
                    "FAILED",
                    &["NONFINITE_OR_MISSING_VALUE"],
                    json!(null),
                ),
            )],
        },
        ImportCase {
            label: "a nonfinite value inside the error field",
            files: vec![(
                "metrics.jsonl",
                b"{\"name\":\"loss\",\"value\":1,\"unit\":\"u\",\"error\":{\"x\":Infinity}}\n"
                    .to_vec(),
            )],
            records: 1,
            invalid_lines: 0,
            rows: vec![unpinned_row(
                "error",
                small_observation(json!(1), "VALID", &[], json!(null)),
            )],
        },
        ImportCase {
            label: "a literal that overflows to infinity",
            files: vec![(
                "metrics.jsonl",
                b"{\"name\":\"loss\",\"value\":1e400,\"unit\":\"u\"}\n".to_vec(),
            )],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(
                Some("665ba035990b35b8c0d6ccb2a4e5c321101094b5c5ec6f66df1226cea748079f"),
                vec![failed_observation("inf")],
            )],
        },
        ImportCase {
            label: "UTF-8 with a BOM and no newline",
            files: vec![("metrics.jsonl", encoded(Encoding::Utf8Bom))],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(None, vec![plain_observation(1)])],
        },
        ImportCase {
            label: "UTF-16 little endian with a BOM",
            files: vec![("metrics.jsonl", encoded(Encoding::Utf16LeBom))],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(None, vec![plain_observation(1)])],
        },
        ImportCase {
            label: "UTF-16 big endian without a BOM",
            files: vec![("metrics.jsonl", encoded(Encoding::Utf16Be))],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(None, vec![plain_observation(1)])],
        },
        ImportCase {
            label: "UTF-32 little endian with a BOM",
            files: vec![("metrics.jsonl", encoded(Encoding::Utf32LeBom))],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(None, vec![plain_observation(1)])],
        },
        ImportCase {
            label: "UTF-32 big endian without a BOM",
            files: vec![("metrics.jsonl", encoded(Encoding::Utf32Be))],
            records: 1,
            invalid_lines: 0,
            rows: vec![metric_row(None, vec![plain_observation(1)])],
        },
        ImportCase {
            label: "65 records cross the 64-event batch",
            files: vec![("metrics.jsonl", long_history())],
            records: 65,
            invalid_lines: 0,
            rows: (0..65)
                .map(|step| metric_row(None, vec![plain_observation(step)]))
                .collect(),
        },
        ImportCase {
            label: "a nonfinite token as a key: {NaN:1}\\n",
            files: vec![("metrics.jsonl", b"{NaN:1}\n".to_vec())],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "1cb1d41c6f61843fcceb47c7bf612126d69e4060ed1ecfad4a8e4a7b0a4eaef5",
            )],
        },
        ImportCase {
            label: "a nonfinite token as a key: {NaN : 1}\\n",
            files: vec![("metrics.jsonl", b"{NaN : 1}\n".to_vec())],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "d47a627de094093213b347a6738d50de50acbaf72359e996193ae62568d4faa1",
            )],
        },
        ImportCase {
            label: "a nonfinite token as a key: {NaN\\n: 1}\\n",
            files: vec![("metrics.jsonl", b"{NaN\n: 1}\n".to_vec())],
            records: 0,
            invalid_lines: 2,
            rows: vec![
                diagnostic_row("3ea355289919c5f19f0ea633bb687f615d9349af0d108ae6ac1c1855f90f6847"),
                diagnostic_row("8fadee21d5fa45e2e7c6e0947a12a8520351fc36e0d18ae387796ebb282c806c"),
            ],
        },
        ImportCase {
            label: "a nonfinite token as a key: {\"name\":\"loss\",\"unit\":\"u\",\"value\":{NaN: 1}}\\n",
            files: vec![(
                "metrics.jsonl",
                b"{\"name\":\"loss\",\"unit\":\"u\",\"value\":{NaN: 1}}\n".to_vec(),
            )],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "cce200c611b7848b0fe3fb256d2fff7f70cb7bf3430c14249e3b40cb98c6eec7",
            )],
        },
        ImportCase {
            label: "a nonfinite token as a key: {Infinity:1}\\n",
            files: vec![("metrics.jsonl", b"{Infinity:1}\n".to_vec())],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "135d191158cec7e091269924f782069effbc14a303d79684b380e54f34fed139",
            )],
        },
        ImportCase {
            label: "a nonfinite token as a key: {Infinity : 1}\\n",
            files: vec![("metrics.jsonl", b"{Infinity : 1}\n".to_vec())],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "da7260781dca72fc68652c48f9e823541430f6aefb6f79363bdb1ef07c3dbd6d",
            )],
        },
        ImportCase {
            label: "a nonfinite token as a key: {Infinity\\n: 1}\\n",
            files: vec![("metrics.jsonl", b"{Infinity\n: 1}\n".to_vec())],
            records: 0,
            invalid_lines: 2,
            rows: vec![
                diagnostic_row("f89fb5b06e02be22ba0cf01fb76995cc50b3dfd1d7623ae56d4d79a1f85fdbb2"),
                diagnostic_row("8fadee21d5fa45e2e7c6e0947a12a8520351fc36e0d18ae387796ebb282c806c"),
            ],
        },
        ImportCase {
            label: "a nonfinite token as a key: {\"name\":\"loss\",\"unit\":\"u\",\"value\":{Infinity: 1}}\\n",
            files: vec![(
                "metrics.jsonl",
                b"{\"name\":\"loss\",\"unit\":\"u\",\"value\":{Infinity: 1}}\n".to_vec(),
            )],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "8a5edf4d75e67a9a5bdb21de490c69e00e963b047520ed5368b6db2673f416d6",
            )],
        },
        ImportCase {
            label: "a nonfinite token as a key: {-Infinity:1}\\n",
            files: vec![("metrics.jsonl", b"{-Infinity:1}\n".to_vec())],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "ecef0b35f021525d159ac697fb9023174cc3a0e0ddec07d41ded115ace149397",
            )],
        },
        ImportCase {
            label: "a nonfinite token as a key: {-Infinity : 1}\\n",
            files: vec![("metrics.jsonl", b"{-Infinity : 1}\n".to_vec())],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "79a9f59a3ca391309bb7d1321eb096addf5d17d787fd2ce811081109e8a189aa",
            )],
        },
        ImportCase {
            label: "a nonfinite token as a key: {-Infinity\\n: 1}\\n",
            files: vec![("metrics.jsonl", b"{-Infinity\n: 1}\n".to_vec())],
            records: 0,
            invalid_lines: 2,
            rows: vec![
                diagnostic_row("9b41d07c51ce8a1aca06dcbd4f3f783ed2b28090cb85aee860c83685bc845dd1"),
                diagnostic_row("8fadee21d5fa45e2e7c6e0947a12a8520351fc36e0d18ae387796ebb282c806c"),
            ],
        },
        ImportCase {
            label: "a nonfinite token as a key: {\"name\":\"loss\",\"unit\":\"u\",\"value\":{-Infinity: 1}}\\n",
            files: vec![(
                "metrics.jsonl",
                b"{\"name\":\"loss\",\"unit\":\"u\",\"value\":{-Infinity: 1}}\n".to_vec(),
            )],
            records: 0,
            invalid_lines: 1,
            rows: vec![diagnostic_row(
                "8cf555fa82775ea16da5865185063b94eb01499e13d3309babf5eae1963f4d20",
            )],
        },
    ]
}
fn import_files(store: &Store, scope: &str, files: &[(&str, Vec<u8>)]) -> super::ImportSummary {
    let mut counts = super::ImportSummary::default();
    for (name, bytes) in files {
        let mut importer = JsonlImporter::new(name, json!({})).unwrap();
        for chunk in bytes.chunks(JsonlImporter::INPUT_BOUND) {
            let (next, emission) = importer.feed(chunk).unwrap();
            let guard = store.guard("demo", "run-a").unwrap();
            store.append_import(&guard, scope, &emission).unwrap();
            importer = next;
        }
        let emission = importer.finish().unwrap();
        let guard = store.guard("demo", "run-a").unwrap();
        let summary = store.append_import(&guard, scope, &emission).unwrap();
        counts.records += summary.records;
        counts.invalid_lines += summary.invalid_lines;
    }
    counts
}
#[test]
fn archive_import_frames_jsonl_diagnoses_nonfinite_values_and_replays_idempotently() {
    let cases = import_cases();
    assert_eq!(cases.len(), 35);
    for case in &cases {
        let label = case.label;
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path(), SEED).unwrap();
        let scope = bound(&store, "run-a", json!({"enabled":false}), json!({}));
        assert_eq!(scope, r#"["demo","run-a","attempt-001"]"#);
        let counts = import_files(&store, &scope, &case.files);
        assert_eq!(
            counts.document(),
            json!({"status":"COMPLETE","records":case.records,"invalid_lines":case.invalid_lines}),
            "{label}"
        );
        let guard = store.guard("demo", "run-a").unwrap();
        let events = store.events(&guard, &scope, 0, 10000).unwrap();
        assert_eq!(events.len(), case.rows.len(), "{label}");
        for (index, (actual, row)) in events.iter().zip(&case.rows).enumerate() {
            assert_eq!(actual.seq, index as i64 + 1, "{label}");
            match row.digest {
                Some(digest) => {
                    assert_eq!(
                        actual.event_id,
                        format!("{}.{digest}", row.prefix),
                        "{label}"
                    );
                }
                None => assert!(
                    actual.event_id.starts_with(&format!("{}.", row.prefix)),
                    "{label}"
                ),
            }
            let mut payload = actual.payload.clone();
            if let Some(key) = row.unpinned {
                let observation = &mut payload["observations"][0];
                assert!(observation[key].is_string(), "{label}");
                observation.as_object_mut().unwrap().remove(key);
            }
            assert_eq!(payload, row.payload, "{label}");
        }
        let observations: Vec<Value> = events
            .iter()
            .flat_map(|actual| {
                actual.payload["observations"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(
            store.metric_records(&guard, "attempt-001").unwrap(),
            observations,
            "{label}"
        );
        drop(guard);
        // A replayed window deduplicates, whatever the framing decided.
        assert_eq!(
            import_files(&store, &scope, &case.files).document(),
            counts.document(),
            "{label}"
        );
        let guard = store.guard("demo", "run-a").unwrap();
        assert_eq!(
            store.events(&guard, &scope, 0, 10000).unwrap(),
            events,
            "{label}"
        );
        // The same identity with a different payload is refused.
        if let Some(first) = events.first() {
            let changed = event(
                &first.event_id,
                json!({"kind":"lifecycle","data":{"changed":true}}),
            );
            assert!(
                matches!(
                    store.append(&guard, &scope, &[changed]),
                    Err(Error::EventConflict)
                ),
                "{label}"
            );
            assert_eq!(
                store.events(&guard, &scope, 0, 10000).unwrap(),
                events,
                "{label}"
            );
        }
    }
}

#[test]
fn one_record_has_one_event_id_in_every_text_encoding_and_a_changed_record_has_another() {
    let event_id = |bytes: Vec<u8>| {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path(), SEED).unwrap();
        let scope = bound(&store, "run-a", json!({"enabled":false}), json!({}));
        import_files(&store, &scope, &[("metrics.jsonl", bytes)]);
        let guard = store.guard("demo", "run-a").unwrap();
        let events = store.events(&guard, &scope, 0, 1).unwrap();
        events[0].event_id.clone()
    };
    let encodings = [
        Encoding::Utf8Bom,
        Encoding::Utf16LeBom,
        Encoding::Utf16Be,
        Encoding::Utf32LeBom,
        Encoding::Utf32Be,
    ];
    let plain = event_id(plain_line());
    assert!(plain.starts_with("archive."));
    for encoding in encodings {
        assert_eq!(event_id(encoded(encoding)), plain);
    }
    assert_ne!(event_id(PLAIN.replace("2.5", "2.75").into_bytes()), plain);
    let unicode = PLAIN.replace("loss", "训练/loss");
    let first = event_id(unicode.clone().into_bytes());
    assert_ne!(first, plain);
    for encoding in encodings {
        assert_eq!(event_id(encoded_as(&unicode, encoding)), first);
    }
}

#[test]
fn new_journal_starts_with_unacknowledged_display_columns() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let scope = bound(&store, "run-a", enabled(), json!({}));
    let guard = store.guard("demo", "run-a").unwrap();
    let state = store.scope_status(&guard, &scope).unwrap().unwrap();
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
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='display_metrics'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
}

#[test]
fn import_needs_a_bound_run_and_never_changes_its_binding() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let guard = store.guard("demo", "train").unwrap();
    assert!(matches!(
        store.import_scope(&guard, "attempt-001"),
        Err(Error::Unbound)
    ));
    assert!(store.binding(&guard).unwrap().is_none());
    let frozen = json!({"ml_expd":{"evaluation":{"protocol_id":"protocol-frozen","metrics_schema":{"definitions":{"loss":{"unit":"u"}}}}}});
    store
        .bind(
            &guard,
            &json!({"enabled":false,"reason":"NATIVE_LOCAL_ONLY"}),
            &frozen,
        )
        .unwrap();
    let original = store.binding(&guard).unwrap();
    let (scope, config) = store.import_scope(&guard, "attempt-001").unwrap();
    assert_eq!(config, frozen);
    assert_eq!(
        store.scope_status(&guard, &scope).unwrap().unwrap()["status"],
        "LOCAL_ONLY"
    );
    let metric = normalized_metric(&json!({"loss":1}), &config).unwrap();
    assert_eq!(metric["observations"][0]["protocol_id"], "protocol-frozen");
    assert_eq!(metric["observations"][0]["unit"], "u");
    assert_eq!(store.import_scope(&guard, "attempt-001").unwrap().1, frozen);
    assert_eq!(store.binding(&guard).unwrap(), original);
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
fn missing_fence_lock_after_open_is_never_repaired_by_normal_reads_writes_or_deletion() {
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
    let path = temp.path().join(".run-deletion.lock");
    fs::rename(&path, temp.path().join("removed-fence-evidence")).unwrap();
    assert!(store.guard("demo", "run-a").is_err());
    assert!(store.delete_run("demo", "run-a", "cleanup").is_err());
    assert!(!path.exists(), "the lock was silently rebuilt");
    let connection = Connection::open(store.path()).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM events", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[test]
fn bounded_metric_reads_are_complete_or_fail_without_truncating_science() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path(), SEED).unwrap();
    let scope = bound(&store, "bounded", enabled(), json!({}));
    let guard = store.guard("demo", "bounded").unwrap();
    let payload = normalized_metric(
        &json!({"name":"训练/loss","unit":"nats/token","value":2.5,"step":1}),
        &json!({}),
    )
    .unwrap();
    store
        .append(&guard, &scope, &[event("metric.1", payload.clone())])
        .unwrap();
    let bytes = ml_exp_core::records::encoded(&payload).unwrap().len();
    assert!(
        store
            .metric_records_bounded(&guard, "attempt-001", bytes - 1, 1)
            .is_err()
    );
    assert!(
        store
            .metric_records_bounded(&guard, "attempt-001", bytes, 0)
            .is_err()
    );
    assert_eq!(
        store
            .metric_records_bounded(&guard, "attempt-001", bytes, 1)
            .unwrap(),
        store.metric_records(&guard, "attempt-001").unwrap()
    );
    assert!(
        store
            .metric_records_bounded(&guard, "attempt-002", 0, 0)
            .unwrap()
            .is_empty()
    );
    assert_eq!(store.workspace_id(&guard).unwrap(), SEED);
}
