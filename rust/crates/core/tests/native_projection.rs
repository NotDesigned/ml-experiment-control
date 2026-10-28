use ml_exp_core::{first_submission::native, slurm::helper::Scope};
use serde_json::{Value, json};
fn scope(attempt: &str) -> Scope {
    Scope {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: attempt.into(),
    }
}
fn submission() -> Value {
    json!({"submission_id":"action-0123456789abcdef","project":"demo","run_id":"train","attempt_id":"attempt-001",
        "operation":"SUBMIT_RUN","status":"RECONCILE_REQUIRED","execution":{"status":"RECONCILE_REQUIRED"}})
}
#[test]
fn exact_attempt_views_project_only_their_own_committed_scheduler_evidence() {
    let collection = json!({"project":"demo","run_id":"train","attempt_id":"attempt-001",
        "evidence":{"scheduler":{"state":"NOT_SUBMITTED"},"process":{"state":null}},
        "training":{"status":"UNKNOWN","exit_code":null},"download_available":false});
    let projected =
        native::project_attempt(collection.clone(), &submission(), &scope("attempt-001")).unwrap();
    assert_eq!(projected["evidence"]["scheduler"]["state"], "UNKNOWN");
    assert_eq!(projected["training"], collection["training"]);
    assert_eq!(projected["download_available"], false);
    assert!(
        native::project_attempt(collection.clone(), &submission(), &scope("attempt-002")).is_err()
    );
    let mut other = collection;
    other["attempt_id"] = json!("attempt-002");
    assert!(native::project_attempt(other, &submission(), &scope("attempt-002")).is_err());
}
#[test]
fn attempt_summary_keeps_initial_and_known_terminal_evidence_without_inventing_training() {
    let shown = json!({"scope":{"project":"demo","scope_type":"attempt","object_id":"train::attempt-001"},
        "run_id":"train","attempt":{"attempt_id":"attempt-001","state":"NOT_SUBMITTED"},
        "evidence":{"scheduler":{"state":"NOT_SUBMITTED"}}});
    let mut prepared = submission();
    prepared["status"] = json!("PREPARED");
    prepared["execution"]["status"] = json!("PREPARED");
    let initial = native::project_attempt(shown.clone(), &prepared, &scope("attempt-001")).unwrap();
    assert_eq!(initial["attempt"]["state"], "NOT_SUBMITTED");
    let uncertain =
        native::project_attempt(shown.clone(), &submission(), &scope("attempt-001")).unwrap();
    assert_eq!(uncertain["attempt"]["state"], "UNKNOWN");
    let mut known = shown;
    known["evidence"]["scheduler"] = json!({"state":"SUCCEEDED","detail":{"backend_job_id":"42"}});
    let retained =
        native::project_attempt(known.clone(), &submission(), &scope("attempt-001")).unwrap();
    assert_eq!(retained["evidence"], known["evidence"]);
}
