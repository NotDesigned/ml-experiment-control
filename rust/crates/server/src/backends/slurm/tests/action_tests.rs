//! Action SQLite + real native helper processes through the bounded fake SSH.
use super::super::action::{Execute, Runner};
use super::*;
use crate::operations::{ClaimContext, Store};
use ml_exp_core::{
    operations::{FrozenFiles, Policy},
    slurm::action as decision,
};
use serde_json::{Value, json};
const NOW: &str = "2026-10-28T00:00:00Z";
fn id(n: u64) -> String {
    format!("{n:032x}")
}
fn plan(request: &Request) -> Value {
    decision::prepare(json!({"operation":"SUBMIT_RUN","scope":{"project":"demo","scope_type":"run","object_id":"train"},
        "run_id":"train","attempt_id":"attempt-001","execution_manifest_sha256":request.intent().manifest_sha256,
        "execution_campaign_sha256":helper::sha256_bytes(b"campaign"),"ready":true,"gates":[],
        "created_at":NOW,"gate_expires_at":"2027-10-28T01:00:00Z"}),request).unwrap()
}
fn fixture(mode: &str) -> (tempfile::TempDir, SshConfig, Store, Request, Value) {
    let root = tempfile::tempdir().unwrap();
    let config = config(root.path(), mode);
    for (name, body) in [
        (
            "sbatch",
            "cat >/dev/null; printf 'call\\n' >> \"$COUNTER\"; printf '24827\\n'",
        ),
        ("sacct", "cat \"$EVIDENCE\""),
        ("squeue", "exit 0"),
        ("scancel", "exit 99"),
    ] {
        let path = root.path().join(name);
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    fs::write(
        root.path().join("evidence"),
        format!(
            "24827|train--attempt-001|ml-exp-{}|h100|RUNNING|0:01|0:0",
            "a".repeat(32)
        ),
    )
    .unwrap();
    ml_exp_worker::slurm::Helper::initialize(ledger_config(root.path())).unwrap();
    let prepare = request();
    let plan = plan(&prepare);
    let store = Store::open(root.path().join("actions")).unwrap();
    store.save_plan(&plan, NOW, &id(1)).unwrap();
    store
        .authorize(
            plan["action_id"].as_str().unwrap(),
            "actor:test",
            "reviewed original native plan",
            NOW,
            &id(2),
        )
        .unwrap();
    (root, config, store, prepare, plan)
}
fn files(request: &Request) -> FrozenFiles {
    FrozenFiles {
        execution_campaign_sha256: Some(helper::sha256_bytes(b"campaign")),
        execution_manifest_sha256: Some(request.intent().manifest_sha256.clone()),
        ..Default::default()
    }
}
fn policy() -> Policy {
    Policy {
        allow_scheduler_mutations: true,
        ..Default::default()
    }
}
fn command<'a>(
    action_id: &'a str,
    confirmation: &'a str,
    policy: &'a Policy,
    files: &'a FrozenFiles,
) -> Execute<'a> {
    Execute {
        action_id,
        confirmation,
        policy,
        files,
        claim: ClaimContext {
            now: NOW,
            transition_id: "00000000000000000000000000000003",
            pid: 123,
        },
    }
}
#[test]
fn claimed_action_runs_through_helper_and_verifies_owned_submission_once() {
    let (root, ssh, store, prepare, plan) = fixture("ledger");
    let runner = Runner {
        actions: &store,
        ssh: &ssh,
    };
    let cancelled = AtomicBool::new(false);
    let action = plan["action_id"].as_str().unwrap();
    let confirmation = format!("EXECUTE {action}");
    let policy = policy();
    let files = files(&prepare);
    let result = runner
        .execute(
            &prepare,
            command(action, &confirmation, &policy, &files),
            &|| Ok(()),
            &cancelled,
        )
        .unwrap();
    assert_eq!(result["execution"]["status"], "VERIFIED");
    assert_eq!(
        result["execution"]["result"]["observation"]["status"]["state"],
        "RUNNING"
    );
    assert_eq!(result["execution"]["safe_to_retry"], false);
    assert_eq!(calls(root.path()), 3);
    assert_eq!(
        runner
            .execute(
                &prepare,
                command(action, &confirmation, &policy, &files),
                &|| Ok(()),
                &cancelled
            )
            .unwrap()["execution"],
        result["execution"]
    );
    assert_eq!(calls(root.path()), 3);
    assert_eq!(
        fs::read_to_string(root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
    assert!(
        !serde_json::to_string(&result)
            .unwrap()
            .contains(&prepare.intent().submission_token)
    );
}
#[test]
fn lost_submit_reply_and_process_restart_inspect_original_receipt_without_replay() {
    let (root, ssh, store, prepare, plan) = fixture("ledger-lost-submit");
    let action = plan["action_id"].as_str().unwrap();
    let confirmation = format!("EXECUTE {action}");
    let policy = policy();
    let files = files(&prepare);
    let cancelled = AtomicBool::new(false);
    let result = Runner {
        actions: &store,
        ssh: &ssh,
    }
    .execute(
        &prepare,
        command(action, &confirmation, &policy, &files),
        &|| Ok(()),
        &cancelled,
    )
    .unwrap();
    assert_eq!(result["execution"]["status"], "RECONCILE_REQUIRED");
    assert_eq!(result["execution"]["safe_to_retry"], false);
    assert!(
        Runner {
            actions: &store,
            ssh: &ssh
        }
        .execute(
            &prepare,
            command(action, &confirmation, &policy, &files),
            &|| Ok(()),
            &cancelled
        )
        .is_err()
    );
    drop(store);
    drop(prepare); // Reload the original private frame after daemon memory is gone.
    let reopened = Store::open_existing(root.path().join("actions")).unwrap();
    fs::write(root.path().join("mode"), "ledger").unwrap();
    let result = Runner {
        actions: &reopened,
        ssh: &ssh,
    }
    .reconcile(action, &|| Ok(()), &cancelled)
    .unwrap();
    assert_eq!(result["execution"]["status"], "VERIFIED");
    assert_eq!(
        result["execution"]["result"]["submission"]["backend_job_id"],
        "24827"
    );
    assert_eq!(
        fs::read_to_string(root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
}
#[test]
fn restart_before_submit_does_not_treat_prepared_ledger_as_absence() {
    let (root, ssh, store, prepare, plan) = fixture("ledger");
    let action = plan["action_id"].as_str().unwrap();
    let confirmation = format!("EXECUTE {action}");
    let policy = policy();
    let files = files(&prepare);
    let command = command(action, &confirmation, &policy, &files);
    store.bind_slurm_request(action, &prepare).unwrap();
    store
        .claim(
            action,
            command.confirmation,
            command.policy,
            command.files,
            command.claim,
        )
        .unwrap();
    let helper = ml_exp_worker::slurm::Helper::open(ledger_config(root.path())).unwrap();
    helper.handle(&prepare, &AtomicBool::new(false)).unwrap();
    store.recover_interrupted(NOW, &id(4)).unwrap();
    let runner = Runner {
        actions: &store,
        ssh: &ssh,
    };
    let result = runner
        .reconcile(action, &|| Ok(()), &AtomicBool::new(false))
        .unwrap();
    assert_eq!(result["execution"]["status"], "RECONCILE_REQUIRED");
    assert_eq!(result["execution"]["safe_to_retry"], false);
    assert!(!root.path().join("sbatch-calls").exists());
    // An old already-sent request arrives later; reconciliation still only reads.
    helper
        .handle(
            &Request::submit(prepare.intent().clone()).unwrap(),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(
        runner
            .reconcile(action, &|| Ok(()), &AtomicBool::new(false))
            .unwrap()["execution"]["status"],
        "VERIFIED"
    );
    assert_eq!(
        fs::read_to_string(root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
}
#[test]
fn frozen_mismatch_disabled_policy_or_lost_guard_cannot_dispatch() {
    let (root, ssh, store, prepare, plan) = fixture("ledger");
    let action = plan["action_id"].as_str().unwrap();
    let confirmation = format!("EXECUTE {action}");
    let runner = Runner {
        actions: &store,
        ssh: &ssh,
    };
    let cancelled = AtomicBool::new(false);
    let policy = policy();
    let mut wrong = files(&prepare);
    wrong.execution_manifest_sha256 = Some(helper::sha256_bytes(b"foreign"));
    let path = store.directory(action).unwrap().join("plan.json");
    let original = fs::read(&path).unwrap();
    let mut changed: Value = serde_json::from_slice(&original).unwrap();
    changed["native_slurm"]["intent_sha256"] = json!(helper::sha256_bytes(b"different-target"));
    fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(
        runner
            .execute(
                &prepare,
                command(action, &confirmation, &policy, &files(&prepare)),
                &|| Ok(()),
                &cancelled
            )
            .is_err()
    );
    assert_eq!(
        store.observe(action).unwrap()["execution"]["status"],
        "AUTHORIZED"
    );
    fs::write(&path, original).unwrap();
    assert!(
        runner
            .execute(
                &prepare,
                command(action, &confirmation, &policy, &wrong),
                &|| Ok(()),
                &cancelled
            )
            .is_err()
    );
    assert!(
        runner
            .execute(
                &prepare,
                command(action, &confirmation, &Policy::default(), &files(&prepare)),
                &|| Ok(()),
                &cancelled
            )
            .is_err()
    );
    let guard = || {
        if store.observe(action).unwrap()["execution"]["status"] == "EXECUTING" {
            Err(super::super::action::Error::Guard)
        } else {
            Ok(())
        }
    };
    let result = runner
        .execute(
            &prepare,
            command(action, &confirmation, &policy, &files(&prepare)),
            &guard,
            &cancelled,
        )
        .unwrap();
    assert_eq!(result["execution"]["status"], "RECONCILE_REQUIRED");
    assert!(!root.path().join("calls").exists());
    assert!(!root.path().join("sbatch-calls").exists());
}
#[test]
fn cancellation_before_claim_and_timeout_after_claim_have_distinct_durable_states() {
    let (root, mut ssh, store, prepare, plan) = fixture("hang");
    let action = plan["action_id"].as_str().unwrap();
    let confirmation = format!("EXECUTE {action}");
    let policy = policy();
    let files = files(&prepare);
    assert!(
        Runner {
            actions: &store,
            ssh: &ssh
        }
        .execute(
            &prepare,
            command(action, &confirmation, &policy, &files),
            &|| Ok(()),
            &AtomicBool::new(true)
        )
        .is_err()
    );
    assert_eq!(
        store.observe(action).unwrap()["execution"]["status"],
        "AUTHORIZED"
    );
    assert!(!root.path().join("calls").exists());
    ssh.budget.timeout = Duration::from_millis(100);
    let start = Instant::now();
    let result = Runner {
        actions: &store,
        ssh: &ssh,
    }
    .execute(
        &prepare,
        command(action, &confirmation, &policy, &files),
        &|| Ok(()),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(start.elapsed() < Duration::from_secs(2));
    assert_eq!(result["execution"]["status"], "RECONCILE_REQUIRED");
    assert_eq!(result["execution"]["safe_to_retry"], false);
    assert!(!root.path().join("sbatch-calls").exists());
}
#[test]
fn a_durable_job_receipt_with_foreign_scheduler_state_is_not_verified() {
    let (root, ssh, store, prepare, plan) = fixture("ledger");
    fs::write(
        root.path().join("evidence"),
        "24827|foreign|wrong|h100|RUNNING|0:01|0:0",
    )
    .unwrap();
    let action = plan["action_id"].as_str().unwrap();
    let confirmation = format!("EXECUTE {action}");
    let runner = Runner {
        actions: &store,
        ssh: &ssh,
    };
    let result = runner
        .execute(
            &prepare,
            command(action, &confirmation, &policy(), &files(&prepare)),
            &|| Ok(()),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(result["execution"]["status"], "RECONCILE_REQUIRED");
    assert_eq!(
        result["execution"]["result"]["submission"]["backend_job_id"],
        "24827"
    );
    assert_eq!(
        runner
            .reconcile(action, &|| Ok(()), &AtomicBool::new(false))
            .unwrap()["execution"]["status"],
        "RECONCILE_REQUIRED"
    );
    assert_eq!(
        fs::read_to_string(root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
}
#[test]
fn missing_corrupt_or_unsafe_private_frame_cannot_be_repaired_after_claim() {
    for mutation in ["missing", "corrupt", "readable", "symlink", "hardlink"] {
        let (root, ssh, store, prepare, plan) = fixture("ledger-lost-submit");
        let action = plan["action_id"].as_str().unwrap();
        let confirmation = format!("EXECUTE {action}");
        let runner = Runner {
            actions: &store,
            ssh: &ssh,
        };
        runner
            .execute(
                &prepare,
                command(action, &confirmation, &policy(), &files(&prepare)),
                &|| Ok(()),
                &AtomicBool::new(false),
            )
            .unwrap();
        let frame = store
            .directory(action)
            .unwrap()
            .join("native-slurm-request.frame");
        let saved = fs::read(&frame).unwrap();
        assert_eq!(
            fs::metadata(&frame).unwrap().permissions().mode() & 0o777,
            0o600
        );
        match mutation {
            "missing" => fs::remove_file(&frame).unwrap(),
            "corrupt" => fs::write(&frame, b"invalid frame").unwrap(),
            "readable" => fs::set_permissions(&frame, fs::Permissions::from_mode(0o644)).unwrap(),
            "symlink" => {
                let foreign = root.path().join("other-frame");
                fs::write(&foreign, &saved).unwrap();
                fs::remove_file(&frame).unwrap();
                std::os::unix::fs::symlink(foreign, &frame).unwrap();
            }
            "hardlink" => fs::hard_link(&frame, root.path().join("other-frame")).unwrap(),
            _ => unreachable!(),
        }
        let before = calls(root.path());
        assert!(
            runner
                .reconcile(action, &|| Ok(()), &AtomicBool::new(false))
                .is_err()
        );
        assert!(store.bind_slurm_request(action, &prepare).is_err());
        assert_eq!(calls(root.path()), before);
        assert_eq!(
            store.observe(action).unwrap()["execution"]["status"],
            "RECONCILE_REQUIRED"
        );
        assert_eq!(
            fs::read_to_string(root.path().join("sbatch-calls")).unwrap(),
            "call\n"
        );
    }
}
