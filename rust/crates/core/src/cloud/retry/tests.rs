use super::*;
use crate::{first_submission::native::retry::StartFrom, state::RunState};
const ACTION: &str = "action-0123456789abcdef";
fn fixture() -> (Value, Receipt) {
    let sha = format!("sha256:{}", "a".repeat(64));
    let binding = Binding {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        project_sha256: sha.clone(),
        run_sha256: sha.clone(),
        attempt_sha256: sha.clone(),
        source_receipt_sha256: sha.clone(),
        backend_config_sha256: sha.clone(),
    };
    let request = Request {
        confirmation: format!("RETRY {ACTION}"),
        resume_from: None,
        restart_from: Some(StartFrom {
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
        }),
        max_gpu_hours: 2.,
        reason: "cache repaired before training".into(),
    };
    let receipt = Receipt {
        contract: CONTRACT.into(),
        parent_action: ACTION.into(),
        parent_binding: binding,
        parent_job_sha256: sha,
        termination: JobObservation {
            job_id: "dlc-original".into(),
            provider_state: "Failed".into(),
            state: RunState::Failed,
            queue: None,
        },
        request_sha256: request.identity(ACTION).unwrap(),
        request,
        reserved_gpu_seconds: 7200,
        created_at: "2026-10-31T00:00:03Z".into(),
    };
    (
        json!({"project":"demo","run_id":"train","resources":{"gpus":1},"native_cloud":{"walltime_seconds":3600},
        "source_id":format!("source.{}","d".repeat(64)),"image_id":format!("registry.example/image@sha256:{}","e".repeat(64)),"seed":42,"parameters":{"lr":0.1}}),
        receipt,
    )
}
#[test]
fn full_restart_preserves_science_and_budget_and_is_deterministic() {
    let (run, receipt) = fixture();
    let attempt = plan(&run, receipt.clone(), &Value::Null).unwrap();
    assert_eq!(attempt, plan(&run, receipt.clone(), &Value::Null).unwrap());
    assert_eq!(attempt["attempt_id"], "attempt-002");
    assert!(attempt["resume_from"].is_null());
    assert!(attempt["native_checkpoint"].is_null());
    let mut binding = receipt.parent_binding.clone();
    binding.attempt_id = "attempt-002".into();
    validate_attempt(&run, &attempt, &binding).unwrap();
    let mut changed = attempt.clone();
    changed["parameters"]["lr"] = json!(0.2);
    assert!(validate_attempt(&run, &changed, &binding).is_err());
    let mut insufficient = receipt.clone();
    insufficient.request.max_gpu_hours = 1.;
    insufficient.request_sha256 = insufficient.request.identity(ACTION).unwrap();
    assert!(plan(&run, insufficient, &Value::Null).is_ok());
    let mut resumed = run.clone();
    resumed["resume_from"] = json!({"run_id":"other"});
    assert!(
        plan(&resumed, receipt, &Value::Null).is_err(),
        "a claim with nothing to restore must not train from step 0"
    );
}
#[test]
fn explicit_restart_needs_a_terminal_parent_not_a_special_failure_code() {
    let (run, mut receipt) = fixture();
    receipt.request.max_gpu_hours = 0.;
    receipt.request_sha256 = receipt.request.identity(ACTION).unwrap();
    assert!(plan(&run, receipt.clone(), &Value::Null).is_ok());
    receipt.termination.state = RunState::Unknown;
    assert!(plan(&run, receipt, &Value::Null).is_err());
}
#[test]
fn retry_request_json_and_hash_are_stable_and_sources_are_exclusive() {
    let original = json!({"confirmation":format!("RETRY {ACTION}"),"resume_from":{"run_id":"train","attempt_id":"attempt-001","checkpoint_id":format!("checkpoint.{}","a".repeat(64))},"max_gpu_hours":2.0,"reason":"restore"});
    let request: Request = serde_json::from_value(original.clone()).unwrap();
    assert_eq!(json!(request), original);
    let expected = crate::canonical::protocol_identity(
        &json!({"contract":retry::CONTRACT,"parent_action":ACTION,"request":original}),
    )
    .unwrap();
    assert_eq!(request.identity(ACTION).unwrap(), expected);
    let (_, startup) = fixture();
    let mut both = request.clone();
    both.restart_from = startup.request.restart_from;
    assert!(both.validate(ACTION).is_err());
    let mut neither = request;
    neither.resume_from = None;
    assert!(neither.validate(ACTION).is_err());
}
#[test]
fn a_retry_without_a_checkpoint_restores_what_the_run_was_frozen_with() {
    let frozen = json!({"checkpoint_id":"frozen"});
    let reference = json!({"run_id":"earlier","attempt_id":"attempt-001","checkpoint_id":"frozen"});
    let (mut run, receipt) = fixture();
    run["resume_from"] = reference.clone();
    run["native_checkpoint"] = frozen.clone();
    let attempt = plan(&run, receipt.clone(), &Value::Null).unwrap();
    assert_eq!(attempt["resume_from"], reference);
    assert_eq!(attempt["native_checkpoint"], frozen);
    let mut binding = receipt.parent_binding.clone();
    binding.attempt_id = "attempt-002".into();
    validate_attempt(&run, &attempt, &binding).unwrap();
    // Only the frozen one: a saved Attempt cannot swap in another under this request.
    let mut swapped = attempt.clone();
    swapped["native_checkpoint"] = json!({"checkpoint_id":"other"});
    assert!(validate_attempt(&run, &swapped, &binding).is_err());
    let mut dropped = attempt;
    dropped["native_checkpoint"] = Value::Null;
    assert!(validate_attempt(&run, &dropped, &binding).is_err());
}
#[test]
fn every_kind_of_retry_keeps_the_code_it_started_with() {
    let code = format!("source.{}", "c".repeat(64));
    let (mut run, receipt) = fixture();
    run["code_source_id"] = json!(code);
    let attempt = plan(&run, receipt, &Value::Null).unwrap();
    assert_eq!(attempt["code_source_id"], code);
    // An Attempt whose code differs from its Run is not the plan of that Run.
    let (run, receipt) = fixture();
    let attempt = plan(&run, receipt, &Value::Null).unwrap();
    assert!(attempt.get("code_source_id").is_none());
}
