use super::*;
use serde_json::{Value, json};
fn staged_request(root: &Path) -> Request {
    let mut intent = request().intent().clone();
    intent.staging = Some(ml_exp_core::slurm::staging::Requirement {
        runs_root: root.join("shared-runs").display().to_string(),
    });
    let doc = json!({"schema_version":1,"identity_version":2,"project":"demo","run_id":"train","attempt_id":"attempt-001",
        "image_id":format!("sha256:{}","1".repeat(64)),
        "backend":{"kind":"slurm","ssh_alias":"test","partition":"h100","account":"science","qos":"normal",
            "gres":"gpu:h100:1","time":"00:05:00","source_dir":root.join("source"),"sif_path":root.join("image.sif"),"mount_root":root},
        "storage":{"run_dir":root.join("shared-runs/demo/train"),"project_data_root":root.join("data/demo")},
        "resources":{"gpus":1,"cpus":2,"memory_gb":4},
        "execution":{"source_mount":"/workspace","workdir":"/workspace","managed_io":true},
        "command":["python","train.py"]});
    let manifest = serde_json::to_string(&doc).unwrap();
    intent.manifest_sha256 = helper::sha256_bytes(manifest.as_bytes());
    Request::prepare(
        intent,
        manifest,
        request().prepared_content().unwrap().1.into(),
    )
    .unwrap()
}
fn staged_fixture() -> (tempfile::TempDir, Helper, Request) {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("shared-runs")).unwrap();
    fake_program(root.path(), "printf '12345\\n'");
    let mut config = config(root.path());
    config.shared_run_root = Some(root.path().join("shared-runs"));
    let helper = Helper::initialize(config).unwrap();
    let request = staged_request(root.path());
    (root, helper, request)
}
fn attempt(root: &Path) -> PathBuf {
    root.join("shared-runs/demo/train/attempts/attempt-001")
}
#[test]
fn staged_attempt_exists_before_submission_and_reuses_original_files_after_restart() {
    let (root, helper, request) = staged_fixture();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    let path = attempt(root.path());
    let bytes = fs::read(path.join("attempt.json")).unwrap();
    assert_eq!(bytes, request.prepared_content().unwrap().0.as_bytes());
    assert!(path.join("inputs").is_dir() && path.join("outputs").is_dir());
    assert_eq!(calls(root.path()), 0);
    drop(helper);
    let mut config = config(root.path());
    config.shared_run_root = Some(root.path().join("shared-runs"));
    let helper = Helper::open(config).unwrap();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    assert_eq!(fs::read(path.join("attempt.json")).unwrap(), bytes);
    assert_eq!(
        handle(&helper, &Request::submit(request.intent().clone()).unwrap()).outcome,
        ReplyOutcome::Submitted {
            job_id: "12345".into()
        }
    );
    assert_eq!(calls(root.path()), 1);
}
#[test]
fn missing_operator_storage_permission_rejects_without_creating_a_claim() {
    let (root, helper) = fixture("printf '12345\\n'");
    let request = staged_request(root.path());
    assert_eq!(
        handle(&helper, &request).outcome,
        ReplyOutcome::Rejected {
            code: helper::Rejection::UnsupportedOperation
        }
    );
    assert!(!claim(root.path()).exists());
    assert!(!root.path().join("shared-runs").exists());
    assert_eq!(calls(root.path()), 0);
}
#[test]
fn preexisting_attempt_is_never_adopted_or_repaired() {
    let (root, helper, request) = staged_fixture();
    let path = attempt(root.path());
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join("keep"), b"existing research output").unwrap();
    assert!(helper.handle(&request, &AtomicBool::new(false)).is_err());
    assert!(matches!(
        handle(&helper, &request).outcome,
        ReplyOutcome::Unknown { .. }
    ));
    assert_eq!(
        fs::read(path.join("keep")).unwrap(),
        b"existing research output"
    );
    assert!(!path.join("attempt.json").exists());
    assert_eq!(calls(root.path()), 0);
}
#[test]
fn damaged_or_replaced_shared_evidence_never_starts_a_scheduler_submission() {
    for change in [
        "manifest",
        "marker",
        "missing-outputs",
        "symlink-inputs",
        "replaced-attempt",
        "replaced-outputs",
    ] {
        let (root, helper, request) = staged_fixture();
        assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
        let path = attempt(root.path());
        match change {
            "manifest" => fs::write(path.join("attempt.json"), b"{}").unwrap(),
            "marker" => fs::write(path.join(".native-staging.json"), b"{}").unwrap(),
            "missing-outputs" => fs::remove_dir(path.join("outputs")).unwrap(),
            "symlink-inputs" => {
                fs::remove_dir(path.join("inputs")).unwrap();
                std::os::unix::fs::symlink(root.path(), path.join("inputs")).unwrap();
            }
            "replaced-attempt" => {
                let saved = root.path().join("saved-attempt");
                fs::rename(&path, &saved).unwrap();
                fs::create_dir(&path).unwrap();
                fs::create_dir(path.join("inputs")).unwrap();
                fs::create_dir(path.join("outputs")).unwrap();
                for name in ["attempt.json", ".native-staging.json"] {
                    fs::copy(saved.join(name), path.join(name)).unwrap();
                }
            }
            "replaced-outputs" => {
                fs::rename(path.join("outputs"), root.path().join("saved-outputs")).unwrap();
                fs::create_dir(path.join("outputs")).unwrap();
            }
            _ => unreachable!(),
        }
        assert_eq!(
            handle(&helper, &Request::submit(request.intent().clone()).unwrap()).outcome,
            ReplyOutcome::Unknown {
                reason: UnknownReason::StorageUnavailable
            },
            "{change}"
        );
        assert_eq!(calls(root.path()), 0, "{change}");
        assert!(
            !claim(root.path()).join("submit.started.json").exists(),
            "{change}"
        );
    }
}
#[test]
fn replaced_shared_root_or_changed_configuration_cannot_be_opened_after_restart() {
    let (root, helper, request) = staged_fixture();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    let shared = root.path().join("shared-runs");
    fs::rename(&shared, root.path().join("original-shared")).unwrap();
    fs::create_dir(&shared).unwrap();
    assert!(helper.handle(&request, &AtomicBool::new(false)).is_err());
    let mut config = config(root.path());
    config.shared_run_root = Some(shared.clone());
    assert!(Helper::open(config).is_err());
    assert!(fs::read_dir(shared).unwrap().next().is_none());
    assert_eq!(calls(root.path()), 0);
}
#[test]
fn staged_request_cannot_use_a_foreign_scope_or_directory_even_with_matching_bytes() {
    let root = tempfile::tempdir().unwrap();
    let request = staged_request(root.path());
    let (manifest, script) = request.prepared_content().unwrap();
    for change in ["project", "run_id", "attempt_id", "storage", "duplicate"] {
        let mut doc: Value = serde_json::from_str(manifest).unwrap();
        if change == "storage" {
            doc["storage"]["run_dir"] = json!(root.path().join("foreign"));
        } else if change != "duplicate" {
            doc[change] = json!("other");
        }
        let mut changed = doc.to_string();
        if change == "duplicate" {
            changed = changed.replacen('{', "{\"project\":\"demo\",", 1);
        }
        let mut intent = request.intent().clone();
        intent.manifest_sha256 = helper::sha256_bytes(changed.as_bytes());
        assert!(
            Request::prepare(intent, changed, script.into()).is_err(),
            "{change}"
        );
    }
}

#[test]
fn changed_output_directory_after_dispatch_does_not_erase_the_known_submission() {
    let (root, helper, request) = staged_fixture();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    let submit = Request::submit(request.intent().clone()).unwrap();
    assert_eq!(
        handle(&helper, &submit).outcome,
        ReplyOutcome::Submitted {
            job_id: "12345".into()
        }
    );
    fs::remove_dir(attempt(root.path()).join("outputs")).unwrap();
    for operation in [submit, Request::inspect(request.intent().clone()).unwrap()] {
        assert_eq!(
            handle(&helper, &operation).outcome,
            ReplyOutcome::Submitted {
                job_id: "12345".into()
            }
        );
    }
    assert_eq!(calls(root.path()), 1);
}

#[test]
fn missing_outputs_after_dispatch_still_allow_owned_observation_and_single_cancel() {
    let (root, helper, request) = staged_fixture();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    handle(&helper, &Request::submit(request.intent().clone()).unwrap());
    fs::remove_dir(attempt(root.path()).join("outputs")).unwrap();
    drop(helper);
    for (name, body) in [
        ("sacct", "cat \"$EVIDENCE\""),
        ("squeue", "exit 0"),
        ("scancel", CANCEL),
    ] {
        let path = root.path().join(name);
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let mut config = config(root.path());
    config.shared_run_root = Some(root.path().join("shared-runs"));
    config.scheduler = Some(super::super::SchedulerTools {
        sacct_program: root.path().join("sacct"),
        squeue_program: root.path().join("squeue"),
        scancel_program: root.path().join("scancel"),
        partition: "h100".into(),
    });
    config.environment.insert(
        "EVIDENCE".into(),
        root.path().join("evidence").display().to_string(),
    );
    config.environment.insert(
        "CANCEL_CALLS".into(),
        root.path().join("cancel-calls").display().to_string(),
    );
    let helper = Helper::open(config).unwrap();
    evidence(root.path(), "RUNNING", &"a".repeat(32));
    let observe = Request::observe_job(request.intent().clone(), "12345".into()).unwrap();
    assert!(matches!(
        handle(&helper, &observe).outcome,
        ReplyOutcome::JobObserved { .. }
    ));
    let cancel = Request::cancel_job(request.intent().clone(), "12345".into()).unwrap();
    for _ in 0..2 {
        assert_eq!(
            handle(&helper, &cancel).outcome,
            ReplyOutcome::CancelPending {
                job_id: "12345".into()
            }
        );
    }
    assert_eq!(
        fs::read_to_string(root.path().join("cancel-calls")).unwrap(),
        "call\n"
    );
    evidence(root.path(), "CANCELLED", &"a".repeat(32));
    assert!(
        matches!(handle(&helper, &cancel).outcome, ReplyOutcome::JobObserved { observation }
        if observation.status.state == ml_exp_core::state::RunState::Cancelled)
    );
    assert_eq!(calls(root.path()), 1);
}

#[test]
fn competing_shared_attempt_preparations_keep_one_immutable_manifest() {
    let (root, helper, first) = staged_fixture();
    let (manifest, script) = first.prepared_content().unwrap();
    let mut changed: Value = serde_json::from_str(manifest).unwrap();
    changed["command"] = json!(["other-training-program"]);
    let changed = changed.to_string();
    let mut intent = first.intent().clone();
    intent.manifest_sha256 = helper::sha256_bytes(changed.as_bytes());
    let second = Request::prepare(intent, changed, script.into()).unwrap();
    let mut config = config(root.path());
    config.shared_run_root = Some(root.path().join("shared-runs"));
    let other = Helper::open(config).unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let jobs = [(helper, first), (other, second)]
        .into_iter()
        .map(|(helper, request)| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                (
                    helper
                        .handle(&request, &AtomicBool::new(false))
                        .unwrap()
                        .outcome,
                    request,
                )
            })
        })
        .collect::<Vec<_>>();
    let mut winners = 0;
    for job in jobs {
        let (outcome, request) = job.join().unwrap();
        match outcome {
            ReplyOutcome::Prepared => {
                winners += 1;
                assert_eq!(
                    fs::read(attempt(root.path()).join("attempt.json")).unwrap(),
                    request.prepared_content().unwrap().0.as_bytes()
                );
            }
            ReplyOutcome::Rejected {
                code: helper::Rejection::IdentityConflict,
            } => {}
            _ => panic!("unexpected competing preparation outcome"),
        }
    }
    assert_eq!(winners, 1);
    assert_eq!(calls(root.path()), 0);
}
