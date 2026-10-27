//! Existing Python project-list DTO with explicit synthetic observations.
use ml_exp_server::read_model::{self, Error};
use serde_json::{Value, json};

#[test]
fn metadata_aggregation_matches_actual_python_project_list() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/project-summary.json")).unwrap();
    assert_eq!(
        fixture["baseline_commit"],
        "e725faec689fb8395e692e27d00a8896fa7fe5a5"
    );
    for case in fixture["cases"].as_array().unwrap() {
        assert_eq!(
            read_model::project_summary("demo", "Synthetic 科学", case["rows"].as_array().unwrap())
                .unwrap(),
            case["expected"]
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
