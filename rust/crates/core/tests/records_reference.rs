//! The journal's record contract: metric normalization, scope identity, companion
//! fences and event checksums. A literal digest here always hashes ASCII,
//! float-free input; non-ASCII and float cases assert the structure instead.
use ml_exp_core::records::{self, Event, Scope};
use serde_json::{Value, json};
use std::collections::BTreeSet;

const ZERO: &str = "0123456789abcdef0123456789abcdef";

fn observation(
    step: i64,
    value: Value,
    unit: Value,
    variant: Value,
    protocol: Value,
    status: &str,
    errors: &[&str],
) -> Value {
    json!({"step":step,"name":"loss","value":value,"unit":unit,"invalid_value":null,
        "variant_id":variant,"protocol_id":protocol,"status":status,"error":errors.first(),
        "source":"exact-attempt-worker","errors":errors})
}
fn named(unit: &str, variant: &str, value: Value) -> Value {
    json!({"name":"loss","value":value,"unit":unit,"step":1,"variant_id":variant})
}
fn normalized(record: &Value, config: &Value, expected: &[Value]) {
    assert_eq!(
        records::normalized_metric(record, config).unwrap(),
        json!({"kind":"metrics","observations":expected}),
        "{record} under {config}"
    );
}

#[test]
fn normalization_without_a_schema_keeps_the_declared_unit_and_diagnoses_gaps() {
    let none = || json!(null);
    let nats = || json!("nats/token");
    for config in [
        json!({}),
        json!({"ml_expd":{}}),
        json!({"ml_expd":{"evaluation":{}}}),
    ] {
        for (record, expected) in [
            (
                named("nats/token", "a", json!(2.5)),
                observation(1, json!(2.5), nats(), json!("a"), none(), "VALID", &[]),
            ),
            (
                json!({"step":2,"loss":3}),
                observation(
                    2,
                    json!(3),
                    none(),
                    none(),
                    none(),
                    "UNDECLARED",
                    &["UNIT_NOT_DECLARED"],
                ),
            ),
            (
                named("nats/token", "a", json!(null)),
                observation(
                    1,
                    none(),
                    nats(),
                    json!("a"),
                    none(),
                    "FAILED",
                    &["NONFINITE_OR_MISSING_VALUE"],
                ),
            ),
            (
                named("bad", "a", json!(2.5)),
                observation(
                    1,
                    json!(2.5),
                    json!("bad"),
                    json!("a"),
                    none(),
                    "VALID",
                    &[],
                ),
            ),
            (
                named("nats/token", "b", json!(2.5)),
                observation(1, json!(2.5), nats(), json!("b"), none(), "VALID", &[]),
            ),
        ] {
            normalized(&record, &config, &[expected]);
        }
        normalized(&json!({}), &config, &[]);
    }
}

#[test]
fn normalization_with_a_schema_applies_its_protocol_and_refuses_another_unit() {
    let config = json!({"ml_expd":{"evaluation":{"protocol_id":"p1",
        "metrics_schema":{"definitions":{"loss":{"unit":"nats/token","required":true}}}}}});
    let none = || json!(null);
    let nats = || json!("nats/token");
    let p1 = || json!("p1");
    for (record, expected) in [
        (
            named("nats/token", "a", json!(2.5)),
            observation(1, json!(2.5), nats(), json!("a"), p1(), "VALID", &[]),
        ),
        (
            json!({"step":2,"loss":3}),
            observation(2, json!(3), nats(), none(), p1(), "VALID", &[]),
        ),
        (
            named("nats/token", "a", json!(null)),
            observation(
                1,
                none(),
                nats(),
                json!("a"),
                p1(),
                "FAILED",
                &["NONFINITE_OR_MISSING_VALUE"],
            ),
        ),
        (
            named("bad", "a", json!(2.5)),
            observation(
                1,
                json!(2.5),
                json!("bad"),
                json!("a"),
                p1(),
                "FAILED",
                &["UNIT_MISMATCH"],
            ),
        ),
        (
            named("nats/token", "b", json!(2.5)),
            observation(1, json!(2.5), nats(), json!("b"), p1(), "VALID", &[]),
        ),
    ] {
        normalized(&record, &config, &[expected]);
    }
    normalized(&json!({}), &config, &[]);
}

#[test]
fn ascii_scopes_and_wandb_ids_are_pinned() {
    let off = json!({"enabled":false});
    let on = json!({"enabled":true,"entity":"fixture","project":"science"});
    let long = format!("{}--preparation", "r".repeat(128));
    for (workspace, project, run, attempt, route, wandb_id) in [
        (
            ZERO,
            "p",
            "r",
            "attempt-001",
            &off,
            "d1011ffa6617bb0d250372c10ce40345",
        ),
        (
            ZERO,
            "p",
            "r",
            "attempt-001",
            &on,
            "615cc13fc3b21ef3a5bac1bdae20ffed",
        ),
        (
            ZERO,
            "demo",
            long.as_str(),
            "preparation",
            &off,
            "bf533dbc917539a8ca3eb3ccf1908bd8",
        ),
        (
            ZERO,
            "demo",
            long.as_str(),
            "preparation",
            &on,
            "3bdc5f3ceae38bb3a3f365928687a5d8",
        ),
        (
            ZERO,
            "p",
            "r",
            "attempt-999",
            &off,
            "9db1ce06befd157011def181cb22e16c",
        ),
        (
            ZERO,
            "p",
            "r",
            "attempt-999",
            &on,
            "2661b0aef49dd638281969d89f4ecf24",
        ),
        (
            "workspace-b",
            "p",
            "r",
            "attempt-001",
            &off,
            "aa28d06f1ed6325ae228d7d5dd9a87fb",
        ),
        (
            "workspace-b",
            "p",
            "r",
            "attempt-001",
            &on,
            "7d16c8cf75898baa050170e187653eb9",
        ),
        (
            "workspace-b",
            "demo",
            long.as_str(),
            "preparation",
            &off,
            "b0540ba7f5cd3d0e52462aa17faf85d6",
        ),
        (
            "workspace-b",
            "demo",
            long.as_str(),
            "preparation",
            &on,
            "a025b4b38e84bffaabe0d1a228bc3615",
        ),
        (
            "workspace-b",
            "p",
            "r",
            "attempt-999",
            &off,
            "f5aea9cf4cfc136a393d31f0bb4e693c",
        ),
        (
            "workspace-b",
            "p",
            "r",
            "attempt-999",
            &on,
            "79a7b570e1d086912300d195ea4c6ede",
        ),
    ] {
        let scope = Scope::new(project, run, attempt).encoded().unwrap();
        assert_eq!(scope, format!(r#"["{project}","{run}","{attempt}"]"#));
        assert_eq!(
            records::wandb_id(workspace, &scope, route).unwrap(),
            wandb_id
        );
    }
}

#[test]
fn wandb_ids_are_deterministic_and_separate_every_workspace_scope_and_route() {
    let off = json!({"enabled":false});
    let on = json!({"enabled":true,"entity":"fixture","project":"science"});
    let long = format!("{}--preparation", "r".repeat(128));
    let mut ids = BTreeSet::new();
    for workspace in [ZERO, "workspace-b", "中文"] {
        for (project, run, attempt) in [
            ("p", "r", "attempt-001"),
            ("demo", long.as_str(), "preparation"),
            ("项目", "旧/观察", "other"),
            ("p", "r", "attempt-999"),
        ] {
            let parts = Scope::new(project, run, attempt);
            let scope = parts.encoded().unwrap();
            assert_eq!(Scope::parse(&scope).unwrap(), parts);
            assert_eq!(parts.encoded().unwrap(), scope);
            for route in [&off, &on] {
                let id = records::wandb_id(workspace, &scope, route).unwrap();
                assert_eq!(records::wandb_id(workspace, &scope, route).unwrap(), id);
                assert_eq!(id.len(), 32);
                assert!(id.bytes().all(|b| b.is_ascii_hexdigit()));
                assert!(ids.insert(id));
            }
        }
    }
    assert_eq!(ids.len(), 24);
}

#[test]
fn only_a_config_proving_the_suffix_names_its_parent_in_the_fence() {
    for run in [
        "r".to_owned(),
        "r--preparation".to_owned(),
        "r--backfill".to_owned(),
        "independent--backfill".to_owned(),
        "旧/观察--backfill".to_owned(),
        format!("{}--preparation", "x".repeat(128)),
    ] {
        for (index, config) in [
            json!({}),
            json!({"ml_expd":{}}),
            json!({"ml_expd":{"run_id":"r","record_type":"preparation"}}),
            json!({"ml_expd":{"run_id":"r","record_type":"backfill"}}),
            json!({"ml_expd":{"run_id":run,"record_type":"backfill"}}),
        ]
        .iter()
        .enumerate()
        {
            let parent = matches!(
                (run.as_str(), index),
                ("r--preparation", 2) | ("r--backfill", 3)
            );
            let mut expected = vec![run.clone()];
            if parent {
                expected.push("r".to_owned());
            }
            assert_eq!(
                records::fence_names(&run, config).unwrap(),
                expected,
                "{run} under {config}"
            );
        }
    }
    assert!(records::fence_names("r", &json!([])).is_err());
    assert!(records::fence_names("r", &json!({"ml_expd":1})).is_err());
}

fn prepared(payload: Value) -> records::Result<records::PreparedEvent> {
    records::prepare_event(&Event {
        event_id: "fixture".into(),
        payload,
    })
}

#[test]
fn ascii_float_free_events_have_pinned_bodies_and_checksums_and_no_dedup() {
    for (payload, body, checksum) in [
        (
            json!({"kind":"lifecycle","data":{"state":"RUNNING"}}),
            r#"{"data":{"state":"RUNNING"},"kind":"lifecycle"}"#,
            "710f652d3964cd095daca0c0af516afbce005cc8923bf07b15ab4a17bf2f1ba8",
        ),
        (
            json!({"kind":"checkpoints","data":{"step":1}}),
            r#"{"data":{"step":1},"kind":"checkpoints"}"#,
            "5210364235f524511798ca0f4e5e05d27091b516226648006f005ca40d8020ae",
        ),
        (
            json!({"kind":123,"data":{}}),
            r#"{"data":{},"kind":123}"#,
            "11d9dbf2371c921a506806b4a1edbd220c3f08cae2e2e02c6da8a1d58165637f",
        ),
    ] {
        let event = prepared(payload).unwrap();
        assert_eq!(event.event_id, "fixture");
        assert_eq!(event.body, body);
        assert_eq!(event.checksum, checksum);
        assert_eq!(event.dedup, None);
    }
    for payload in [json!({}), json!([]), json!("text")] {
        assert!(prepared(payload).is_err());
    }
}

#[test]
fn a_metric_event_deduplicates_on_the_checksum_of_its_content() {
    let metric = |name: &str, value: Value| {
        records::normalized_metric(
            &json!({"name":name,"value":value,"unit":"nats/token","step":1,"variant_id":"a"}),
            &json!({}),
        )
        .unwrap()
    };
    for name in ["loss", "训练/loss"] {
        let payload = metric(name, json!(2.5));
        let first = prepared(payload.clone()).unwrap();
        let again = prepared(metric(name, json!(2.5))).unwrap();
        assert_eq!(first.checksum.len(), 64);
        assert_eq!(first.dedup.as_deref(), Some(first.checksum.as_str()));
        assert_eq!(first.checksum, again.checksum);
        assert_eq!(first.body, again.body);
        assert_eq!(serde_json::from_str::<Value>(&first.body).unwrap(), payload);
        assert_ne!(
            first.checksum,
            prepared(metric(name, json!(3.5))).unwrap().checksum
        );
    }
    assert_ne!(
        prepared(metric("loss", json!(2.5))).unwrap().checksum,
        prepared(metric("训练/loss", json!(2.5))).unwrap().checksum
    );
}
