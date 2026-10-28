use ml_exp_core::{
    operations,
    slurm::{self, action, helper::*},
};
use serde_json::{Value, json};
const NOW: &str = "2026-10-28T00:00:00Z";
fn fixture() -> (Value, Request) {
    let manifest = "run_id: train\n";
    let script = format!(
        "#!/bin/sh\n#SBATCH --job-name=train--attempt-001\n#SBATCH --comment=ml-exp-{}\n\nexit 0\n",
        "a".repeat(32)
    );
    let request = Request::prepare(
        FrozenIntent {
            target: Target {
                backend_id: "slurm-test".into(),
                profile_sha256: sha256_bytes(b"profile"),
                storage_scope: sha256_bytes(b"storage"),
            },
            scope: Scope {
                project: "demo".into(),
                run_id: "train".into(),
                attempt_id: "attempt-001".into(),
            },
            submission_token: "a".repeat(32),
            scheduler_name: "train--attempt-001".into(),
            manifest_sha256: sha256_bytes(manifest.as_bytes()),
            script_sha256: sha256_bytes(script.as_bytes()),
        },
        manifest.into(),
        script,
    )
    .unwrap();
    let plan = json!({"operation":"SUBMIT_RUN","scope":{"project":"demo","scope_type":"run","object_id":"train"},
        "run_id":"train","attempt_id":"attempt-001","execution_manifest_sha256":request.intent().manifest_sha256,
        "execution_campaign_sha256":sha256_bytes(b"campaign"),"ready":true,"gates":[],
        "created_at":NOW,"gate_expires_at":"2027-10-28T01:00:00Z"});
    (action::prepare(plan, &request).unwrap(), request)
}
#[test]
fn native_plan_pins_all_execution_fields_and_does_not_reinterpret_legacy() {
    let (plan, request) = fixture();
    action::validate(&plan, &request).unwrap();
    assert_eq!(
        action::prepare(
            {
                let mut p = plan.clone();
                p.as_object_mut().unwrap().remove("native_slurm");
                p
            },
            &request
        )
        .unwrap(),
        plan
    );
    for (key, value) in [
        ("run_id", json!("foreign")),
        ("gate_expires_at", json!("2030-01-01T00:00:00Z")),
        ("ready", json!(false)),
        ("gates", json!(["changed"])),
        ("verification_command_preview", json!(["foreign-command"])),
        ("request_digest", json!(sha256_bytes(b"foreign"))),
    ] {
        let mut changed = plan.clone();
        changed[key] = value;
        assert!(action::validate(&changed, &request).is_err());
    }
    let mut legacy = plan;
    legacy.as_object_mut().unwrap().remove("native_slurm");
    assert!(action::validate(&legacy, &request).is_err());
    legacy["execution"] = json!({});
    assert!(action::prepare(legacy, &request).is_err());
    let (_, mut intent) = fixture();
    // A second preparation with a different token changes its stable Action ID.
    let mut frozen = intent.intent().clone();
    frozen.submission_token = "b".repeat(32);
    let (manifest, script) = intent.prepared_content().unwrap();
    let script = script.replace(&"a".repeat(32), &"b".repeat(32));
    frozen.script_sha256 = sha256_bytes(script.as_bytes());
    intent = Request::prepare(frozen, manifest.into(), script).unwrap();
    assert!(action::validate(&fixture().0, &intent).is_err());
}
#[test]
fn owned_submission_is_verified_separately_from_training_state_and_retry() {
    let (mut plan, request) = fixture();
    plan["execution"] = operations::initial_execution(&plan);
    plan["execution"]["status"] = json!("EXECUTING");
    plan["execution"]["revision"] = json!(3);
    let scope = slurm::JobScope {
        ssh_alias: "slurm-test".into(),
        job_id: slurm::JobId::parse("123").unwrap(),
        scheduler_name: request.intent().scheduler_name.clone(),
        partition: "h100".into(),
    };
    let row = format!(
        "123|train--attempt-001|ml-exp-{}|h100|RUNNING|0:01|0:0",
        "a".repeat(32)
    );
    let observation = slurm::observe_owned_status(
        &scope,
        &"a".repeat(32),
        None,
        slurm::Query::Success(&row),
        slurm::Query::NotRequested,
        Some(NOW),
    )
    .unwrap();
    let reply = Reply::for_request(
        &Request::observe_job(request.intent().clone(), "123".into()).unwrap(),
        ReplyOutcome::JobObserved {
            observation: Box::new(observation),
        },
    )
    .unwrap();
    let transition = action::finish(
        &plan,
        &request,
        action::Outcome::Owned {
            job_id: "123".into(),
            reply: reply.clone(),
        },
        NOW,
    )
    .unwrap();
    assert_eq!(transition.execution["status"], "VERIFIED");
    assert_eq!(
        transition.execution["result"]["observation"]["status"]["state"],
        "RUNNING"
    );
    assert_eq!(transition.execution["safe_to_retry"], false);
    assert_eq!(transition.execution["revision"], 3);
    let mut other = plan.clone();
    other["execution"]["result"] = json!({"submission":{"backend_job_id":"456"}});
    assert!(
        action::finish(
            &other,
            &request,
            action::Outcome::Owned {
                job_id: "123".into(),
                reply
            },
            NOW
        )
        .is_err()
    );
    let pending = action::finish(
        &plan,
        &request,
        action::Outcome::Uncertain {
            job_id: Some("123".into()),
            reason: action::Uncertain::Observation,
        },
        NOW,
    )
    .unwrap();
    assert_eq!(pending.execution["status"], "RECONCILE_REQUIRED");
    assert_eq!(pending.execution["safe_to_retry"], false);
    plan["execution"] = pending.execution;
    let next = action::finish(
        &plan,
        &request,
        action::Outcome::Uncertain {
            job_id: None,
            reason: action::Uncertain::Ledger,
        },
        NOW,
    )
    .unwrap();
    assert_eq!(
        next.execution["result"]["submission"]["backend_job_id"],
        "123"
    );
}
