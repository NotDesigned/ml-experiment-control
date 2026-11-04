use super::*;
use crate::{
    first_submission::native::retry::StartFrom,
    managed::evidence::{Observation, Phase},
    state::RunState,
};
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
            sync_request_id: "b".repeat(32),
        }),
        max_gpu_hours: 2.,
        reason: "cache repaired before training".into(),
    };
    let worker = Observation {
        contract: crate::managed::evidence::CONTRACT.into(),
        identity: crate::launch::Identity {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
        },
        manifest_sha256: sha.clone(),
        phase: Phase::Prepared,
        safe_to_retry: false,
        receipt: None,
        runtime_environment_sha256: None,
    };
    let input = serde_json::from_value(json!({"at":"2026-10-31T00:00:01Z","phase":"FAILED","asset_id":format!("asset.{}","c".repeat(64)),"code":"DATA_CACHE_UNAVAILABLE"})).unwrap();
    let proof = StartupProof {
        sync_request_id: "b".repeat(32),
        baseline_sequence: 2,
        sequence: 4,
        fenced_at: "2026-10-31T00:00:00Z".into(),
        observed_at: "2026-10-31T00:00:02Z".into(),
        packet_sha256: sha.clone(),
        worker,
        inputs: vec![input],
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
        startup_failure: Some(proof),
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
    assert!(plan(&resumed, receipt, &Value::Null).unwrap()["resume_from"].is_null());
}
#[test]
fn explicit_restart_needs_a_terminal_parent_not_a_special_failure_code() {
    let (run, mut receipt) = fixture();
    receipt.startup_failure = None;
    receipt.request.max_gpu_hours = 0.;
    receipt.request_sha256 = receipt.request.identity(ACTION).unwrap();
    assert!(plan(&run, receipt.clone(), &Value::Null).is_ok());
    receipt.termination.state = RunState::Unknown;
    assert!(plan(&run, receipt, &Value::Null).is_err());
}
#[test]
fn legacy_retry_request_json_and_hash_are_unchanged_and_sources_are_exclusive() {
    let legacy = json!({"confirmation":format!("RETRY {ACTION}"),"resume_from":{"run_id":"train","attempt_id":"attempt-001","checkpoint_id":format!("checkpoint.{}","a".repeat(64))},"max_gpu_hours":2.0,"reason":"restore"});
    let request: Request = serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(json!(request), legacy);
    let expected = crate::canonical::protocol_identity(
        &json!({"contract":retry::CONTRACT,"parent_action":ACTION,"request":legacy}),
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
