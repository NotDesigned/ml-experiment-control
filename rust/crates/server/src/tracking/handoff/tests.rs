use super::*;
use ml_exp_core::records::Event;
use serde_json::json;
const WORKSPACE: &str = "0123456789abcdef0123456789abcdef";
fn fixture() -> (tempfile::TempDir, Store, Config) {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path(), WORKSPACE).unwrap();
    let guard = store.guard("demo", "old-train").unwrap();
    store
        .bind(
            &guard,
            &json!({"enabled":true,"entity":"fixture","project":"science"}),
            &json!({"ml_expd":{"run_id":"old-train","seed":7}}),
        )
        .unwrap();
    let scope = store.scope(&guard, "attempt-001").unwrap();
    store
        .append(
            &guard,
            &scope,
            &[
                Event {
                    event_id: "started".into(),
                    payload: json!({"kind":"lifecycle","data":{"state":"RUNNING"}}),
                },
                Event {
                    event_id: "finished".into(),
                    payload: json!({"kind":"lifecycle","data":{"state":"SUCCEEDED"}}),
                },
            ],
        )
        .unwrap();
    let binding = store.binding(&guard).unwrap().unwrap();
    let status = store.scope_status(&guard, &scope).unwrap().unwrap();
    let pin = ScopePin {
        scope: scope.clone(),
        wandb_id: status["wandb_id"].as_str().unwrap().into(),
        binding_sha256: canonical::protocol_identity(&binding).unwrap(),
        events_sha256: event_digest(&store, &guard, &scope, 2).unwrap(),
        confirmed: 0,
        total: 2,
        reserved_through: 0,
    };
    let config = Config {
        schema_version: 1,
        registry_root: root.path().to_owned(),
        workspace_id: WORKSPACE.into(),
        scopes: vec![pin],
    };
    drop(guard);
    (root, store, config)
}
#[test]
fn handoff_preflight_preserves_original_rows_and_refuses_changed_event_or_scope_identity() {
    let (_root, store, config) = fixture();
    let before = std::fs::read(store.path()).unwrap();
    config.check().unwrap();
    assert_eq!(before, std::fs::read(store.path()).unwrap());
    for field in [
        "wandb_id",
        "binding_sha256",
        "events_sha256",
        "confirmed",
        "total",
    ] {
        let mut value = json!(config);
        value["scopes"][0][field] = match field {
            "confirmed" => json!(1),
            "total" => json!(3),
            "wandb_id" => json!("foreign"),
            _ => json!(format!("sha256:{}", "a".repeat(64))),
        };
        let invalid: Config = serde_json::from_value(value).unwrap();
        assert!(invalid.check().is_err(), "{field}");
    }
    let mut duplicated = config.clone();
    duplicated.scopes.push(duplicated.scopes[0].clone());
    assert!(duplicated.check().is_err());
    let mut native = config.clone();
    native.registry_root = PathBuf::from("/tmp/../private");
    assert!(native.check().is_err());
    let mut completed = config.clone();
    completed.scopes[0].confirmed = 2;
    assert!(completed.check().is_err());
    let mut other_workspace = config.clone();
    other_workspace.workspace_id = "a".repeat(32);
    assert!(other_workspace.check().is_err());
    assert_eq!(before, std::fs::read(store.path()).unwrap());
}
#[test]
fn opted_out_scope_cannot_be_granted_publication_by_a_matching_pin() {
    let (_root, store, mut config) = fixture();
    let connection = rusqlite::Connection::open(store.path()).unwrap();
    connection
        .execute(
            "UPDATE runs SET route=?",
            [json!({"enabled":false,"reason":"REQUESTED_OFF"}).to_string()],
        )
        .unwrap();
    let guard = store.guard("demo", "old-train").unwrap();
    config.scopes[0].binding_sha256 =
        canonical::protocol_identity(&store.binding(&guard).unwrap().unwrap()).unwrap();
    assert!(config.check().is_err());
}
#[test]
fn migrated_failure_counter_is_not_truncated_and_confirmations_remain_monotonic() {
    let (_root, store, config) = fixture();
    let pin = &config.scopes[0];
    let connection = rusqlite::Connection::open(store.path()).unwrap();
    connection
        .execute("UPDATE scopes SET failures=463,confirmed=1", [])
        .unwrap();
    config.check().unwrap();
    let guard = store.guard("demo", "old-train").unwrap();
    store
        .save_publication_state(
            &guard,
            &pin.scope,
            &json!({}),
            "ACK_PENDING",
            Some("REMOTE_ACK_PENDING"),
            "2026-10-30T00:00:00Z",
        )
        .unwrap();
    let status = store.scope_status(&guard, &pin.scope).unwrap().unwrap();
    assert_eq!(status["confirmed"], 1);
    assert_eq!(status["failures"], 464);
    let saved = store.publication_state(&guard, &pin.scope).unwrap();
    store
        .initialize_publication_state(
            &guard,
            &pin.scope,
            &json!({"foreign":"must not replace original reservation"}),
        )
        .unwrap();
    assert_eq!(saved, store.publication_state(&guard, &pin.scope).unwrap());
    assert_eq!(
        status,
        store.scope_status(&guard, &pin.scope).unwrap().unwrap()
    );
    connection
        .execute("UPDATE scopes SET confirmed=0", [])
        .unwrap();
    let mut advanced_pin = config.clone();
    advanced_pin.scopes[0].confirmed = 1;
    advanced_pin.scopes[0].reserved_through = 1;
    assert!(advanced_pin.check().is_err());
}
