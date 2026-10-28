use ml_exp_core::operations::submission_view;
use serde_json::{Value, json};

const NOW: &str = "2026-10-28T00:00:00Z";

#[test]
fn submission_views_match_the_python_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/submission-view.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        assert_eq!(
            submission_view(&case["action"], fixture["now"].as_str().unwrap()).unwrap(),
            case["expected"]
        );
    }
}
fn action() -> Value {
    json!({"action_id":"action-0123456789abcdef", "scope":{"project":"demo","scope_type":"run","object_id":"train"},
        "run_id":"train","attempt_id":"attempt-001","operation":"SUBMIT_RUN","ready":true,
        "gate_expires_at":"2026-10-28T00:00:01Z", "execution":{"status":"PREPARED","revision":1},
        "preflight_summary":{"git_commit":"frozen-commit"},"command_preview":["private internal command"]})
}

#[test]
fn submission_view_preserves_exact_binding_and_clock_gate_without_effects() {
    for (status, expected) in [
        ("PREPARED", "AUTHORIZE"),
        ("AUTHORIZED", "EXECUTE"),
        ("EXECUTING", "RECONCILE"),
        ("RECONCILE_REQUIRED", "RECONCILE"),
        ("BLOCKED", "REPREPARE"),
        ("VERIFIED", "NONE"),
        ("FAILED", "NONE"),
    ] {
        let mut value = action();
        value["execution"]["status"] = json!(status);
        let original = value.clone();
        let view = submission_view(&value, NOW).unwrap();
        assert_eq!(view["next_action"], expected);
        assert_eq!(view["project"], "demo");
        assert_eq!(view["run_id"], "train");
        assert_eq!(view["attempt_id"], "attempt-001");
        assert_eq!(view["execution"], value["execution"]);
        assert_eq!(view["git_commit"], "frozen-commit");
        assert!(view.get("command_preview").is_none());
        assert_eq!(original, value);
    }
    for expiry in [
        Value::Null,
        json!("invalid"),
        json!(NOW),
        json!("2026-10-27T00:00:00Z"),
    ] {
        let mut value = action();
        value["gate_expires_at"] = expiry;
        assert_eq!(
            submission_view(&value, NOW).unwrap()["next_action"],
            "REPREPARE"
        );
    }
    let mut value = action();
    value["preflight_summary"] = Value::Null;
    value["semantic_changes"] = json!([{"path":"git_commit","after":"old-commit"}]);
    assert_eq!(
        submission_view(&value, NOW).unwrap()["git_commit"],
        "old-commit"
    );
    for (name, bad) in [
        ("action_id", json!("action-../foreign")),
        ("attempt_id", json!("attempt-1")),
        ("run_id", json!("../train")),
        ("operation", json!("CANCEL_RUN")),
        ("execution", json!({"status":"EXECUTING","revision":-1})),
    ] {
        let mut value = action();
        value[name] = bad;
        assert!(submission_view(&value, NOW).is_err(), "{name}");
    }
    assert!(submission_view(&action(), "invalid").is_err());
}
