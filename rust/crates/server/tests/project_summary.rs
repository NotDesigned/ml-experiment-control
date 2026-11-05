//! Project-list aggregation with explicit synthetic observations.
use ml_exp_server::read_model::{self, Error};
use serde_json::{Value, json};

#[test]
fn run_counts_and_attention_follow_state_and_staleness() {
    for (rows, expected) in [
        (
            r#"[]"#,
            r#"{"project":"demo","title":"Synthetic 科学","run_counts":{},"attention_count":0}"#,
        ),
        (
            r#"[{"project":"demo","run_id":"run-0","scheduler_state":"","evidence_stale":true}]"#,
            r#"{"project":"demo","title":"Synthetic 科学","run_counts":{"UNKNOWN":1},"attention_count":1}"#,
        ),
        (
            r#"[{"project":"demo","run_id":"run-0","scheduler_state":"RUNNING","evidence_stale":false},{"project":"demo","run_id":"run-1","scheduler_state":"FAILED","evidence_stale":false},{"project":"demo","run_id":"run-2","scheduler_state":"PREEMPTED","evidence_stale":false}]"#,
            r#"{"project":"demo","title":"Synthetic 科学","run_counts":{"RUNNING":1,"FAILED":1,"PREEMPTED":1},"attention_count":2}"#,
        ),
        (
            r#"[{"project":"demo","run_id":"run-0","scheduler_state":"RUNNING","evidence_stale":true},{"project":"demo","run_id":"run-1","scheduler_state":"FAILED","evidence_stale":true},{"project":"demo","run_id":"run-2","scheduler_state":"PREEMPTED","evidence_stale":true}]"#,
            r#"{"project":"demo","title":"Synthetic 科学","run_counts":{"RUNNING":1,"FAILED":1,"PREEMPTED":1},"attention_count":5}"#,
        ),
        (
            r#"[{"project":"demo","run_id":"run-0","scheduler_state":"failed","evidence_stale":false},{"project":"demo","run_id":"run-1","scheduler_state":"SUCCEEDED","evidence_stale":false},{"project":"demo","run_id":"run-2","scheduler_state":"UNKNOWN","evidence_stale":false}]"#,
            r#"{"project":"demo","title":"Synthetic 科学","run_counts":{"failed":1,"SUCCEEDED":1,"UNKNOWN":1},"attention_count":1}"#,
        ),
    ] {
        let rows: Vec<Value> = serde_json::from_str(rows).unwrap();
        let expected: Value = serde_json::from_str(expected).unwrap();
        assert_eq!(
            read_model::project_summary("demo", "Synthetic 科学", &rows).unwrap(),
            expected
        );
    }
}

#[test]
fn aggregation_refuses_foreign_duplicates_invalid_evidence_and_unbounded_output() {
    let row = json!({"project":"demo","run_id":"trial","scheduler_state":"RUNNING","evidence_stale":false});
    assert!(matches!(
        read_model::project_summary("other", "Title", std::slice::from_ref(&row)),
        Err(Error::Identity)
    ));
    assert!(matches!(
        read_model::project_summary("demo", "Title", &[row.clone(), row.clone()]),
        Err(Error::Identity)
    ));
    for (key, value) in [
        ("scheduler_state", json!(1)),
        ("evidence_stale", Value::Null),
    ] {
        let mut invalid = row.clone();
        invalid[key] = value;
        assert!(matches!(
            read_model::project_summary("demo", "Title", &[invalid]),
            Err(Error::Evidence)
        ));
    }
    assert!(matches!(
        read_model::project_summary("demo", &"x".repeat(2001), &[]),
        Err(Error::OutputLimit)
    ));
    let mut oversized = row;
    oversized["scheduler_state"] = json!("x".repeat(read_model::MAX_RESPONSE_BYTES));
    assert!(matches!(
        read_model::project_summary("demo", "Title", &[oversized]),
        Err(Error::OutputLimit)
    ));
}
