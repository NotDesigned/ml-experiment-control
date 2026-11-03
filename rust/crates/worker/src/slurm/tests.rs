use super::{Helper, HelperConfig};
use ml_exp_core::slurm::helper::{
    self, FrozenIntent, LedgerPhase, LedgerRecord, ReplyOutcome, Request, Scope, Target,
    UnknownReason,
};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};
#[path = "staging_tests.rs"]
mod staging_tests;

fn target() -> Target {
    Target {
        backend_id: "slurm-test".into(),
        profile_sha256: helper::sha256_bytes(b"profile"),
        storage_scope: helper::sha256_bytes(b"storage"),
    }
}
fn request() -> Request {
    let manifest = "run_id: train\n".to_owned();
    let script = format!(
        "#!/usr/bin/env bash\n#SBATCH --job-name=train--attempt-001\n#SBATCH --comment=ml-exp-{}\n\nexit 0\n",
        "a".repeat(32)
    );
    Request::prepare(
        FrozenIntent {
            staging: None,
            target: target(),
            scope: Scope {
                project: "demo".into(),
                run_id: "train".into(),
                attempt_id: "attempt-001".into(),
            },
            submission_token: "a".repeat(32),
            scheduler_name: "train--attempt-001".into(),
            manifest_sha256: helper::sha256_bytes(manifest.as_bytes()),
            script_sha256: helper::sha256_bytes(script.as_bytes()),
        },
        manifest,
        script,
    )
    .unwrap()
}
fn config(root: &Path) -> HelperConfig {
    HelperConfig {
        root: root.join("helper"),
        shared_run_root: None,
        target: target(),
        sbatch_program: root.join("sbatch"),
        environment: BTreeMap::from([
            ("PATH".into(), "/usr/bin:/bin".into()),
            ("COUNTER".into(), root.join("calls").display().to_string()),
        ]),
        lock_timeout: Duration::from_secs(2),
        execution_timeout: Duration::from_secs(1),
        scheduler: None,
        data_preparer: None,
    }
}
fn fixture(body: &str) -> (tempfile::TempDir, Helper) {
    let root = tempfile::tempdir().unwrap();
    fake_program(root.path(), body);
    let helper = Helper::initialize(config(root.path())).unwrap();
    (root, helper)
}
fn fake_program(root: &Path, body: &str) {
    fs::write(root.join("sbatch"), format!("#!/bin/sh\n[ \"$#\" -eq 3 ] && [ \"$1\" = --parsable ] && [ \"$2\" = --job-name=train--attempt-001 ] && [ \"$3\" = --comment=ml-exp-{} ] || exit 99\ncat >/dev/null\nprintf 'call\\n' >> \"$COUNTER\"\n{body}\n", "a".repeat(32))).unwrap();
    fs::set_permissions(root.join("sbatch"), fs::Permissions::from_mode(0o700)).unwrap();
}
fn claim(root: &Path) -> PathBuf {
    root.join("helper/demo/runs/train/.submission-attempt-001")
}
fn calls(root: &Path) -> usize {
    fs::read_to_string(root.join("calls"))
        .unwrap_or_default()
        .lines()
        .count()
}
fn handle(helper: &Helper, request: &Request) -> helper::Reply {
    helper.handle(request, &AtomicBool::new(false)).unwrap()
}

#[test]
fn prepares_frozen_bytes_once_and_retrieves_a_lost_reply_without_resubmitting() {
    let (root, helper) = fixture("printf '12345;test-cluster\\n'");
    let prepare = request();
    let first = handle(&helper, &prepare);
    assert_eq!(first.outcome, ReplyOutcome::Prepared);
    assert!(!first.safe_to_retry());
    let (_, script) = prepare.prepared_content().unwrap();
    assert_eq!(
        fs::read(claim(root.path()).join("job.sbatch")).unwrap(),
        script.as_bytes()
    );
    assert_eq!(handle(&helper, &prepare).outcome, ReplyOutcome::Prepared);
    let submit = Request::submit(prepare.intent().clone()).unwrap();
    drop(handle(&helper, &submit)); // Simulated lost transport response.
    assert_eq!(calls(root.path()), 1);
    let expected = ReplyOutcome::Submitted {
        job_id: "12345".into(),
    };
    assert_eq!(
        handle(
            &helper,
            &Request::inspect(prepare.intent().clone()).unwrap()
        )
        .outcome,
        expected
    );
    assert_eq!(handle(&helper, &submit).outcome, expected);
    assert_eq!(handle(&helper, &prepare).outcome, expected);
    assert_eq!(calls(root.path()), 1);
}

#[test]
fn scheduler_identity_is_fixed_by_cli_even_if_the_script_contains_a_short_alias() {
    let (root, helper) = fixture("printf '12345;test-cluster\\n'");
    let original = request();
    let (manifest, script) = original.prepared_content().unwrap();
    let script = script.replacen("\n\nexit 0", "\n#SBATCH -J another-job\n\nexit 0", 1);
    let mut intent = original.intent().clone();
    intent.script_sha256 = helper::sha256_bytes(script.as_bytes());
    let prepare = Request::prepare(intent, manifest.into(), script).unwrap();
    assert_eq!(handle(&helper, &prepare).outcome, ReplyOutcome::Prepared);
    // The fake executable validates all three fixed options before incrementing
    // the dispatch count; it cannot accept only a matching script directive.
    assert_eq!(
        handle(&helper, &Request::submit(prepare.intent().clone()).unwrap()).outcome,
        ReplyOutcome::Submitted {
            job_id: "12345".into()
        }
    );
    assert_eq!(calls(root.path()), 1);
}

#[test]
fn a_scheduler_effect_cannot_publish_success_after_authority_or_claim_replacement() {
    for body in [
        "rm registry.json; printf '12345\\n'",
        "printf '{\"version\":1,\"claims\":{}}' > registry.json; printf '12345\\n'",
        "rm root.json; printf '12345\\n'",
        "mv demo/runs/train/.submission-attempt-001 demo/runs/train/retired-claim; cp -R demo/runs/train/retired-claim demo/runs/train/.submission-attempt-001; printf '12345\\n'",
    ] {
        let (root, helper) = fixture(body);
        let prepare = request();
        assert_eq!(handle(&helper, &prepare).outcome, ReplyOutcome::Prepared);
        let submit = Request::submit(prepare.intent().clone()).unwrap();
        assert!(helper.handle(&submit, &AtomicBool::new(false)).is_err());
        assert_eq!(calls(root.path()), 1);
        // A reply was lost after a real scheduler effect. Keep the started
        // evidence and never manufacture success or dispatch it for a retry.
        let state: LedgerRecord =
            serde_json::from_slice(&fs::read(claim(root.path()).join("state.json")).unwrap())
                .unwrap();
        assert_eq!(state.phase, LedgerPhase::SubmitStarted);
        assert!(claim(root.path()).join("submit.started.json").is_file());
        assert!(!claim(root.path()).join("receipt.json").exists());
        let replay = helper.handle(&submit, &AtomicBool::new(false));
        assert!(matches!(
            replay,
            Err(_)
                | Ok(helper::Reply {
                    outcome: ReplyOutcome::Unknown { .. },
                    ..
                })
        ));
        assert_eq!(calls(root.path()), 1);
    }
}

#[test]
fn concurrent_same_token_calls_dispatch_sbatch_exactly_once() {
    let (root, helper) = fixture("sleep 0.05; printf '24680\\n'");
    let prepare = request();
    handle(&helper, &prepare);
    let helper = Arc::new(helper);
    let submit = Request::submit(prepare.intent().clone()).unwrap();
    let mut threads = Vec::new();
    for _ in 0..6 {
        let helper = Arc::clone(&helper);
        let request = submit.clone();
        threads.push(std::thread::spawn(move || {
            handle(&helper, &request).outcome
        }));
    }
    for thread in threads {
        assert_eq!(
            thread.join().unwrap(),
            ReplyOutcome::Submitted {
                job_id: "24680".into()
            }
        );
    }
    assert_eq!(calls(root.path()), 1);
}

#[test]
fn uncertain_process_outcomes_and_corruption_never_replay_started_effects() {
    for (label, body) in [
        ("invalid", "printf 'not-a-job-id\\n'"),
        ("nonzero", "exit 7"),
        ("timeout", "trap '' TERM; sleep 10"),
    ] {
        let (root, helper) = fixture(body);
        let prepare = request();
        handle(&helper, &prepare);
        let submit = Request::submit(prepare.intent().clone()).unwrap();
        let result = handle(&helper, &submit);
        assert_eq!(
            result.outcome,
            ReplyOutcome::Unknown {
                reason: UnknownReason::UncertainProcessResult
            },
            "{label}"
        );
        assert_eq!(calls(root.path()), 1);
        assert_eq!(
            handle(&helper, &submit).outcome,
            ReplyOutcome::Unknown {
                reason: UnknownReason::SubmitStarted
            }
        );
        assert!(
            !handle(
                &helper,
                &Request::inspect(prepare.intent().clone()).unwrap()
            )
            .safe_to_retry()
        );
        assert_eq!(calls(root.path()), 1);
        // Removing the immutable start cannot reset the durable started head.
        fs::remove_file(claim(root.path()).join("submit.started.json")).unwrap();
        assert_eq!(
            handle(&helper, &submit).outcome,
            ReplyOutcome::Unknown {
                reason: UnknownReason::SubmitStarted
            }
        );
        assert_eq!(calls(root.path()), 1);
    }
}

#[test]
fn foreign_targets_tokens_and_frozen_content_are_rejected_without_dispatch() {
    let (root, helper) = fixture("printf '123\\n'");
    let prepare = request();
    handle(&helper, &prepare);
    let mut foreign = prepare.intent().clone();
    foreign.target.profile_sha256 = helper::sha256_bytes(b"foreign");
    assert_eq!(
        handle(&helper, &Request::submit(foreign).unwrap()).outcome,
        ReplyOutcome::Rejected {
            code: helper::Rejection::IdentityConflict
        }
    );
    let mut foreign = prepare.intent().clone();
    foreign.submission_token = "b".repeat(32);
    assert_eq!(
        handle(&helper, &Request::submit(foreign).unwrap()).outcome,
        ReplyOutcome::Rejected {
            code: helper::Rejection::IdentityConflict
        }
    );
    fs::write(claim(root.path()).join("job.sbatch"), b"tamper").unwrap();
    assert_eq!(
        handle(&helper, &Request::submit(prepare.intent().clone()).unwrap()).outcome,
        ReplyOutcome::Unknown {
            reason: UnknownReason::MalformedLedger
        }
    );
    assert_eq!(calls(root.path()), 0);
}

#[test]
fn missing_heads_or_locks_are_never_recreated_and_legacy_claims_remain_unknown() {
    let (root, helper) = fixture("printf '123\\n'");
    let prepare = request();
    let path = claim(root.path());
    fs::create_dir_all(&path).unwrap();
    assert_eq!(
        handle(&helper, &prepare).outcome,
        ReplyOutcome::Unknown {
            reason: UnknownReason::LegacyClaim
        }
    );
    assert_eq!(fs::read_dir(&path).unwrap().count(), 0);
    fs::remove_dir(&path).unwrap();
    handle(&helper, &prepare);
    fs::remove_file(path.join("state.json")).unwrap();
    assert_eq!(
        handle(&helper, &prepare).outcome,
        ReplyOutcome::Unknown {
            reason: UnknownReason::MalformedLedger
        }
    );
    assert!(!path.join("state.json").exists());
    fs::remove_file(root.path().join("helper/.slurm-helper.lock")).unwrap();
    assert!(helper.handle(&prepare, &AtomicBool::new(false)).is_err());
    assert!(Helper::open(config(root.path())).is_err());
    assert!(Helper::initialize(config(root.path())).is_err());
    assert!(!root.path().join("helper/.slurm-helper.lock").exists());
    assert_eq!(calls(root.path()), 0);
}

#[test]
fn lost_claim_or_registry_never_becomes_a_fresh_submission() {
    let (root, helper) = fixture("printf '123\\n'");
    let prepare = request();
    handle(&helper, &prepare);
    let submit = Request::submit(prepare.intent().clone()).unwrap();
    handle(&helper, &submit);
    assert_eq!(calls(root.path()), 1);
    fs::remove_dir_all(claim(root.path())).unwrap();
    for request in [&prepare, &submit] {
        assert_eq!(
            handle(&helper, request).outcome,
            ReplyOutcome::Unknown {
                reason: UnknownReason::MissingLedger
            }
        );
    }
    assert!(!claim(root.path()).exists());
    assert_eq!(calls(root.path()), 1);
    fs::remove_file(root.path().join("helper/registry.json")).unwrap();
    assert!(helper.handle(&prepare, &AtomicBool::new(false)).is_err());
    assert!(Helper::open(config(root.path())).is_err());
    assert!(Helper::initialize(config(root.path())).is_err());
    assert!(!root.path().join("helper/registry.json").exists());
    assert_eq!(calls(root.path()), 1);
}

#[test]
fn immutable_start_and_receipt_disagreeing_with_the_head_fail_closed() {
    let (root, helper) = fixture("printf '123\\n'");
    let prepare = request();
    handle(&helper, &prepare);
    fs::write(claim(root.path()).join("submit.started.json"), b"broken").unwrap();
    let submit = Request::submit(prepare.intent().clone()).unwrap();
    assert_eq!(
        handle(&helper, &submit).outcome,
        ReplyOutcome::Unknown {
            reason: UnknownReason::MalformedLedger
        }
    );
    fs::remove_file(claim(root.path()).join("submit.started.json")).unwrap();
    handle(&helper, &submit);
    fs::write(claim(root.path()).join("receipt.json"), b"broken").unwrap();
    assert_eq!(
        handle(&helper, &submit).outcome,
        ReplyOutcome::Unknown {
            reason: UnknownReason::MalformedLedger
        }
    );
    assert_eq!(calls(root.path()), 1);
}

#[test]
fn process_crash_after_started_is_durable_and_never_replays() {
    const CHILD: &str = "ML_EXP_SLURM_CRASH_TEST_ROOT";
    if let Some(root) = std::env::var_os(CHILD) {
        let root = PathBuf::from(root);
        let helper = Helper::open(config(&root)).unwrap();
        let submit = Request::submit(request().intent().clone()).unwrap();
        let _ = handle(&helper, &submit);
        panic!("fixture sbatch should kill only its owning helper subprocess");
    }
    let (root, helper) = fixture("kill -KILL \"$PPID\"; exit 0");
    let prepare = request();
    handle(&helper, &prepare);
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "slurm::tests::process_crash_after_started_is_durable_and_never_replays",
            "--nocapture",
        ])
        .env(CHILD, root.path())
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(calls(root.path()), 1);
    let head: LedgerRecord =
        serde_json::from_slice(&fs::read(claim(root.path()).join("state.json")).unwrap()).unwrap();
    assert_eq!(head.phase, LedgerPhase::SubmitStarted);
    let submit = Request::submit(prepare.intent().clone()).unwrap();
    assert_eq!(
        handle(&helper, &submit).outcome,
        ReplyOutcome::Unknown {
            reason: UnknownReason::SubmitStarted
        }
    );
    assert_eq!(calls(root.path()), 1);
}

#[test]
fn metadata_stdio_framing_is_exact_bounded_and_cannot_accept_trailing_payload() {
    let request = request();
    let frame = helper::encode_request(&request).unwrap();
    assert!(super::read_request(&mut frame.as_slice()).unwrap() == request);
    assert!(super::read_request(&mut [frame.clone(), vec![1]].concat().as_slice()).is_err());
    assert!(super::read_request(&mut [0, 128, 0, 1].as_slice()).is_err());
    let reply = helper::Reply::for_request(&request, ReplyOutcome::Prepared).unwrap();
    let mut bytes = Vec::new();
    super::write_reply(&mut bytes, &reply).unwrap();
    assert_eq!(helper::decode_reply(&bytes, &request).unwrap(), reply);
}

#[test]
fn inspect_is_readonly_and_recovers_a_receipt_before_head_replacement() {
    let (root, helper) = fixture("printf '123\\n'");
    let prepare = request();
    let inspect = Request::inspect(prepare.intent().clone()).unwrap();
    assert_eq!(
        handle(&helper, &inspect).outcome,
        ReplyOutcome::Unknown {
            reason: UnknownReason::MissingLedger
        }
    );
    assert!(!root.path().join("helper/demo").exists());
    handle(&helper, &prepare);
    let submit = Request::submit(prepare.intent().clone()).unwrap();
    handle(&helper, &submit);
    // Exact durable crash shape: receipt was fsynced, but the final mutable
    // head publication did not complete. Observation must not repair the head.
    let mut head: LedgerRecord =
        serde_json::from_slice(&fs::read(claim(root.path()).join("state.json")).unwrap()).unwrap();
    head.phase = LedgerPhase::SubmitStarted;
    let before = serde_json::to_vec(&head).unwrap();
    fs::write(claim(root.path()).join("state.json"), &before).unwrap();
    assert_eq!(
        handle(&helper, &inspect).outcome,
        ReplyOutcome::Submitted {
            job_id: "123".into()
        }
    );
    assert_eq!(
        fs::read(claim(root.path()).join("state.json")).unwrap(),
        before
    );
    assert_eq!(
        handle(&helper, &submit).outcome,
        ReplyOutcome::Submitted {
            job_id: "123".into()
        }
    );
    assert_eq!(calls(root.path()), 1);
}

#[test]
fn bootstrap_rejects_unknown_content_and_invalid_trusted_environment() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("helper")).unwrap();
    fs::write(root.path().join("helper/unknown"), b"keep").unwrap();
    assert!(Helper::initialize(config(root.path())).is_err());
    assert_eq!(
        fs::read(root.path().join("helper/unknown")).unwrap(),
        b"keep"
    );
    let other = tempfile::tempdir().unwrap();
    let mut invalid = config(other.path());
    invalid
        .environment
        .insert("BAD=KEY".into(), "not-printed".into());
    assert!(Helper::initialize(invalid).is_err());
    assert!(!other.path().join("helper").exists());
    for key in [
        "SBATCH_TIMELIMIT",
        "SBATCH_JOB_NAME",
        "SQUEUE_FORMAT",
        "SACCT_FORMAT",
        "SCANCEL_INTERACTIVE",
        "SLURM_CLUSTERS",
        "SLURM_CONF",
        "SLURM_CONF_SERVER",
    ] {
        let other = tempfile::tempdir().unwrap();
        let mut invalid = config(other.path());
        invalid.environment.insert(key.into(), "not-printed".into());
        assert!(Helper::initialize(invalid).is_err());
        assert!(!other.path().join("helper").exists());
    }
    let bounded = tempfile::tempdir().unwrap();
    let mut limit = config(bounded.path());
    limit.environment = BTreeMap::from([("A".into(), "x".repeat(65533))]);
    limit.validate().unwrap(); // key + value + two separators == 64 KiB.
    limit.environment.get_mut("A").unwrap().push('x');
    assert!(limit.validate().is_err());
    limit.environment = (0..128)
        .map(|i| (format!("VAR{i}"), String::new()))
        .collect();
    limit.validate().unwrap();
    limit.environment.insert("EXTRA".into(), String::new());
    assert!(limit.validate().is_err());
    assert!(!bounded.path().join("helper").exists());
}

#[test]
fn cancelling_before_dispatch_creates_no_attempt_and_lock_wait_is_bounded() {
    use fs2::FileExt;
    let (root, helper) = fixture("printf '123\\n'");
    let result = helper.handle(&request(), &AtomicBool::new(true)).unwrap();
    assert_eq!(
        result.outcome,
        ReplyOutcome::Unknown {
            reason: UnknownReason::Cancelled
        }
    );
    assert!(!claim(root.path()).exists());
    let file = File::open(root.path().join("helper/.slurm-helper.lock")).unwrap();
    FileExt::lock_exclusive(&file).unwrap();
    let started = Instant::now();
    let result = helper.handle(&request(), &AtomicBool::new(false));
    assert!(result.is_err());
    assert!(started.elapsed() < Duration::from_secs(3));
    FileExt::unlock(&file).unwrap();
    assert_eq!(calls(root.path()), 0);
}

fn job_fixture(cancel_body: &str) -> (tempfile::TempDir, Helper) {
    job_fixture_configured(cancel_body, |_, _| {})
}
fn job_fixture_configured(
    cancel_body: &str,
    configure: impl FnOnce(&Path, &mut HelperConfig),
) -> (tempfile::TempDir, Helper) {
    let root = tempfile::tempdir().unwrap();
    fake_program(root.path(), "printf '12345\\n'");
    for (name, body) in [
        (
            "sacct",
            "[ \"$1\" = -j ] && [ \"$2\" = 12345 ] && [ \"$3\" = -X ] || exit 99; cat \"$EVIDENCE\"",
        ),
        (
            "squeue",
            "[ \"$1\" = -j ] && [ \"$2\" = 12345 ] || exit 99; cat \"$QUEUE\"",
        ),
        ("scancel", cancel_body),
    ] {
        fs::write(root.path().join(name), format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(root.path().join(name), fs::Permissions::from_mode(0o700)).unwrap();
    }
    let mut cfg = config(root.path());
    cfg.scheduler = Some(super::SchedulerTools {
        sacct_program: root.path().join("sacct"),
        squeue_program: root.path().join("squeue"),
        scancel_program: root.path().join("scancel"),
        scontrol_program: None,
        partition: "h100".into(),
    });
    for (key, file) in [
        ("EVIDENCE", "evidence"),
        ("QUEUE", "queue"),
        ("CANCEL_CALLS", "cancel-calls"),
    ] {
        cfg.environment
            .insert(key.into(), root.path().join(file).display().to_string());
    }
    configure(root.path(), &mut cfg);
    let helper = Helper::initialize(cfg).unwrap();
    let prepare = request();
    handle(&helper, &prepare);
    handle(&helper, &Request::submit(prepare.intent().clone()).unwrap());
    (root, helper)
}
fn evidence(root: &Path, state: &str, marker: &str) {
    fs::write(
        root.join("evidence"),
        format!("12345|train--attempt-001|ml-exp-{marker}|h100|{state}|00:01|0:0\n"),
    )
    .unwrap();
}
fn job_request(cancel: bool) -> Request {
    if cancel {
        Request::cancel_job(request().intent().clone(), "12345".into()).unwrap()
    } else {
        Request::observe_job(request().intent().clone(), "12345".into()).unwrap()
    }
}
const CANCEL: &str =
    "[ \"$#\" -eq 1 ] && [ \"$1\" = 12345 ] || exit 99; printf 'call\\n' >> \"$CANCEL_CALLS\"";
#[test]
fn owned_job_progress_and_durable_cancel_never_reissue_after_lost_response() {
    let (root, helper) = job_fixture(CANCEL);
    evidence(root.path(), "PENDING", &"a".repeat(32));
    fs::write(
        root.path().join("queue"),
        format!(
            "12345|train--attempt-001|ml-exp-{}|h100|PENDING|0:01|0:0|Resources\n",
            "a".repeat(32)
        ),
    )
    .unwrap();
    let ReplyOutcome::JobObserved { observation } = handle(&helper, &job_request(false)).outcome
    else {
        panic!("missing owned status");
    };
    assert_eq!(observation.status.reason.as_deref(), Some("Resources"));
    assert_eq!(
        observation.status.state,
        ml_exp_core::state::RunState::Queued
    );
    let cancel = job_request(true);
    assert_eq!(
        handle(&helper, &cancel).outcome,
        ReplyOutcome::CancelPending {
            job_id: "12345".into()
        }
    );
    drop(helper);
    // Reopen trusted config without bootstrap: durable REQUESTED prevents replay.
    let mut cfg = config(root.path());
    cfg.scheduler = Some(super::SchedulerTools {
        sacct_program: root.path().join("sacct"),
        squeue_program: root.path().join("squeue"),
        scancel_program: root.path().join("scancel"),
        scontrol_program: None,
        partition: "h100".into(),
    });
    for (key, file) in [
        ("EVIDENCE", "evidence"),
        ("QUEUE", "queue"),
        ("CANCEL_CALLS", "cancel-calls"),
    ] {
        cfg.environment
            .insert(key.into(), root.path().join(file).display().to_string());
    }
    let reopened = Helper::open(cfg).unwrap();
    evidence(root.path(), "RUNNING", &"a".repeat(32));
    assert!(matches!(
        handle(&reopened, &cancel).outcome,
        ReplyOutcome::CancelPending { .. }
    ));
    evidence(root.path(), "CANCELLED", &"a".repeat(32));
    let terminal = handle(&reopened, &cancel);
    assert!(
        matches!(&terminal.outcome, ReplyOutcome::JobObserved {observation} if observation.status.state == ml_exp_core::state::RunState::Cancelled)
    );
    fs::write(root.path().join("evidence"), "invalid").unwrap();
    assert_eq!(handle(&reopened, &cancel).outcome, terminal.outcome);
    assert_eq!(
        fs::read_to_string(root.path().join("cancel-calls")).unwrap(),
        "call\n"
    );
    assert!(!terminal.safe_to_retry());
}
#[test]
fn foreign_missing_duplicate_and_failed_evidence_never_cancel() {
    for text in [
        format!(
            "12345|train--attempt-001|ml-exp-{}|h100|RUNNING|0:01|0:0\n",
            "b".repeat(32)
        ),
        "12345|foreign|wrong|h100|RUNNING|0:01|0:0\n".into(),
        "bad\nbad\n".into(),
        String::new(),
    ] {
        let (root, helper) = job_fixture(CANCEL);
        fs::write(root.path().join("evidence"), text).unwrap();
        fs::write(root.path().join("queue"), "").unwrap();
        assert!(matches!(
            handle(&helper, &job_request(true)).outcome,
            ReplyOutcome::Unknown { .. }
        ));
        assert!(!root.path().join("cancel-calls").exists());
        assert!(!claim(root.path()).join("cancel.started.json").exists());
    }
}
#[test]
fn cancel_timeout_leaves_outbox_and_missing_or_corrupt_head_cannot_replay() {
    let (root, helper) = job_fixture(&format!("{CANCEL}; sleep 2"));
    evidence(root.path(), "RUNNING", &"a".repeat(32));
    let cancel = job_request(true);
    assert!(matches!(
        handle(&helper, &cancel).outcome,
        ReplyOutcome::CancelPending { .. }
    ));
    assert!(matches!(
        handle(&helper, &cancel).outcome,
        ReplyOutcome::CancelPending { .. }
    ));
    let head = claim(root.path()).join("cancel-intent.json");
    let original = fs::read(&head).unwrap();
    fs::remove_file(&head).unwrap();
    assert!(helper.handle(&cancel, &AtomicBool::new(false)).is_err());
    fs::write(&head, "{}").unwrap();
    assert!(helper.handle(&cancel, &AtomicBool::new(false)).is_err());
    fs::write(&head, original).unwrap();
    assert!(matches!(
        handle(&helper, &cancel).outcome,
        ReplyOutcome::CancelPending { .. }
    ));
    assert_eq!(
        fs::read_to_string(root.path().join("cancel-calls")).unwrap(),
        "call\n"
    );
}
#[test]
fn unavailable_tools_wrong_receipt_and_known_token_echo_fail_closed() {
    let (root, helper) = fixture("printf '12345\\n'");
    let prepare = request();
    handle(&helper, &prepare);
    handle(&helper, &Request::submit(prepare.intent().clone()).unwrap());
    assert!(matches!(
        handle(&helper, &job_request(false)).outcome,
        ReplyOutcome::Rejected {
            code: helper::Rejection::UnsupportedOperation
        }
    ));
    let wrong = Request::cancel_job(prepare.intent().clone(), "67890".into()).unwrap();
    assert!(matches!(
        handle(&helper, &wrong).outcome,
        ReplyOutcome::Rejected {
            code: helper::Rejection::IdentityConflict
        }
    ));
    assert_eq!(calls(root.path()), 1);
    let (root, helper) = job_fixture(CANCEL);
    evidence(root.path(), "PENDING", &"a".repeat(32));
    fs::write(
        root.path().join("queue"),
        format!(
            "12345|train--attempt-001|ml-exp-{}|h100|PENDING|0:01|0:0|secret {}\n",
            "a".repeat(32),
            "a".repeat(32)
        ),
    )
    .unwrap();
    assert!(
        helper
            .handle(&job_request(true), &AtomicBool::new(false))
            .is_err()
    );
    assert!(!root.path().join("cancel-calls").exists());
}

#[test]
fn immediate_cancel_success_requires_terminal_observation_and_stores_it() {
    let body = format!(
        "{CANCEL}; printf '12345|train--attempt-001|ml-exp-{}|h100|CANCELLED|0:02|0:15\\n' > \"$EVIDENCE\"",
        "a".repeat(32)
    );
    let (root, helper) = job_fixture(&body);
    evidence(root.path(), "RUNNING", &"a".repeat(32));
    let cancel = job_request(true);
    let result = handle(&helper, &cancel);
    assert!(
        matches!(&result.outcome,ReplyOutcome::JobObserved {observation} if observation.status.state==ml_exp_core::state::RunState::Cancelled)
    );
    assert_eq!(handle(&helper, &cancel).outcome, result.outcome);
    assert_eq!(
        fs::read_to_string(root.path().join("cancel-calls")).unwrap(),
        "call\n"
    );
}

#[test]
fn commentless_accounting_uses_owned_queue_then_controller_for_cancel_confirmation() {
    let controller = |state| {
        format!(
            "JobId=12345 JobName=train--attempt-001 Comment=ml-exp-{} Partition=h100 JobState={state} RunTime=00:00:01 ExitCode=0:15 Reason=None",
            "a".repeat(32)
        )
    };
    let cancel = format!(
        "{CANCEL}; printf '%s' '{}' > \"$CONTROL\"",
        controller("CANCELLED")
    );
    let (root, helper) = job_fixture_configured(&cancel, |root, cfg| {
        let program = root.join("scontrol");
        fs::write(&program, "#!/bin/sh\n[ \"$#\" -eq 4 ] && [ \"$1\" = show ] && [ \"$2\" = job ] && [ \"$3\" = 12345 ] && [ \"$4\" = --oneliner ] || exit 99\ncat \"$CONTROL\"\n").unwrap();
        fs::set_permissions(&program, fs::Permissions::from_mode(0o700)).unwrap();
        cfg.scheduler.as_mut().unwrap().scontrol_program = Some(program);
        cfg.environment.insert(
            "CONTROL".into(),
            root.join("controller").display().to_string(),
        );
    });
    fs::write(
        root.path().join("evidence"),
        "12345|train--attempt-001||h100|PENDING|00:00|0:0\n",
    )
    .unwrap();
    fs::write(
        root.path().join("queue"),
        format!(
            "12345|train--attempt-001|ml-exp-{}|h100|PENDING|0:00|0:0|Priority\n",
            "a".repeat(32)
        ),
    )
    .unwrap();
    let ReplyOutcome::JobObserved { observation } = handle(&helper, &job_request(false)).outcome
    else {
        panic!("missing exact queue proof");
    };
    assert_eq!(observation.status.reason.as_deref(), Some("Priority"));
    assert_eq!(
        observation.status.observation_source,
        Some(ml_exp_core::slurm::StatusSource::Squeue)
    );
    fs::write(root.path().join("queue"), "").unwrap();
    for bad in [
        controller("RUNNING").replace(&"a".repeat(32), &"b".repeat(32)),
        controller("RUNNING").replace("12345", "54321"),
        "".into(),
    ] {
        fs::write(root.path().join("controller"), bad).unwrap();
        assert!(matches!(
            handle(&helper, &job_request(true)).outcome,
            ReplyOutcome::Unknown { .. }
        ));
        assert!(!root.path().join("cancel-calls").exists());
    }
    fs::write(root.path().join("controller"), controller("RUNNING")).unwrap();
    let result = handle(&helper, &job_request(true));
    assert!(
        matches!(&result.outcome, ReplyOutcome::JobObserved { observation } if observation.status.state==ml_exp_core::state::RunState::Cancelled && observation.status.observation_source==Some(ml_exp_core::slurm::StatusSource::Scontrol))
    );
    assert_eq!(handle(&helper, &job_request(true)).outcome, result.outcome);
    assert_eq!(
        fs::read_to_string(root.path().join("cancel-calls")).unwrap(),
        "call\n"
    );
}
