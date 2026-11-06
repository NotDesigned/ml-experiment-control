use super::*;
use serde_json::{Value, json};
fn staged_request(root: &Path) -> Request {
    request_at(root)
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
    assert!(path.join("state").is_dir());
    assert!(!path.join("inputs").exists() && !path.join("outputs").exists());
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
    let root = tempfile::tempdir().unwrap();
    fake_program(root.path(), "printf '12345\\n'");
    let helper = Helper::initialize(unshared_config(root.path())).unwrap();
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
fn damaged_shared_evidence_never_starts_a_scheduler_submission() {
    for change in ["manifest", "marker", "missing-state", "symlink-state"] {
        let (root, helper, request) = staged_fixture();
        assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
        let path = attempt(root.path());
        match change {
            "manifest" => fs::write(path.join("attempt.json"), b"{}").unwrap(),
            "marker" => fs::write(path.join(".native-staging.json"), b"{}").unwrap(),
            "missing-state" => fs::remove_dir(path.join("state")).unwrap(),
            "symlink-state" => {
                fs::remove_dir(path.join("state")).unwrap();
                std::os::unix::fs::symlink(root.path(), path.join("state")).unwrap();
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
fn shared_damage_found_just_before_start_leaves_the_attempt_able_to_submit() {
    let (root, helper, request) = staged_fixture();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    let submit = Request::submit(request.intent().clone()).unwrap();
    let worker = attempt(root.path()).join("worker");
    assert_eq!(
        helper
            .submit_after(&submit, || fs::create_dir(&worker).unwrap())
            .unwrap(),
        ReplyOutcome::Unknown {
            reason: UnknownReason::StorageUnavailable
        }
    );
    assert_eq!(calls(root.path()), 0);
    assert!(!claim(root.path()).join("submit.started.json").exists());
    fs::remove_dir(&worker).unwrap();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    assert_eq!(
        handle(&helper, &submit).outcome,
        ReplyOutcome::Submitted {
            job_id: "12345".into()
        }
    );
    assert_eq!(calls(root.path()), 1);
}
/// A Run whose one dataset is a sealed tree holding `tokens.bin`, which contains "abcd".
fn request_with_dataset(root: &Path) -> (Request, PathBuf) {
    let base = request_at(root);
    let (manifest, script) = base.prepared_content().unwrap();
    let sha256 = "b".repeat(64);
    let tree = root
        .join("data/demo")
        .join(format!("asset.{sha256}"))
        .join("tree");
    fs::create_dir_all(&tree).unwrap();
    let file = tree.join("tokens.bin");
    fs::write(&file, b"abcd").unwrap();
    fs::set_permissions(&file, fs::Permissions::from_mode(0o400)).unwrap();
    fs::set_permissions(&tree, fs::Permissions::from_mode(0o500)).unwrap();
    let mut document: Value = serde_json::from_str(manifest).unwrap();
    document["native_datasets"] = json!([{
        "name": "tokens",
        "receipt": {
            "contract": ml_exp_core::datasets::CONTRACT,
            "project": "demo",
            "target": target(),
            "asset_id": format!("asset.{sha256}"),
            "definition": {
                "sha256": sha256,
                "archive_bytes": 4096,
                "files": [{
                    "path": "tokens.bin",
                    "bytes": 4,
                    "sha256": "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589",
                }],
            },
            "root": tree.to_str().unwrap(),
        },
    }]);
    let manifest = document.to_string();
    let mut intent = base.intent().clone();
    intent.manifest_sha256 = helper::sha256_bytes(manifest.as_bytes());
    (
        Request::prepare(intent, manifest, script.into()).unwrap(),
        tree,
    )
}
#[test]
fn submission_checks_dataset_files_and_sizes_and_leaves_their_content_to_the_job() {
    let (root, helper, _) = staged_fixture();
    let (request, tree) = request_with_dataset(root.path());
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    let file = tree.join("tokens.bin");
    let rewrite = |bytes: &[u8]| {
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&file, bytes).unwrap();
        fs::set_permissions(&file, fs::Permissions::from_mode(0o400)).unwrap();
    };
    let submit = Request::submit(request.intent().clone()).unwrap();
    rewrite(b"abc");
    assert_eq!(
        handle(&helper, &submit).outcome,
        ReplyOutcome::Unknown {
            reason: UnknownReason::StorageUnavailable
        }
    );
    assert_eq!(calls(root.path()), 0);
    assert!(!claim(root.path()).join("submit.started.json").exists());
    // The same size with other content is the job's own hash to refuse.
    rewrite(b"abcx");
    assert_eq!(
        handle(&helper, &submit).outcome,
        ReplyOutcome::Submitted {
            job_id: "12345".into()
        }
    );
    assert_eq!(calls(root.path()), 1);
    fs::set_permissions(&tree, fs::Permissions::from_mode(0o700)).unwrap();
}
#[test]
fn replaced_shared_root_or_changed_configuration_cannot_be_opened_after_restart() {
    let (root, helper, request) = staged_fixture();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    let shared = root.path().join("shared-runs");
    fs::rename(&shared, root.path().join("original-shared")).unwrap();
    fs::create_dir(&shared).unwrap();
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
fn changed_state_directory_after_dispatch_does_not_erase_the_known_submission() {
    let (root, helper, request) = staged_fixture();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    let submit = Request::submit(request.intent().clone()).unwrap();
    assert_eq!(
        handle(&helper, &submit).outcome,
        ReplyOutcome::Submitted {
            job_id: "12345".into()
        }
    );
    fs::remove_dir(attempt(root.path()).join("state")).unwrap();
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
fn missing_state_after_dispatch_still_allows_owned_observation_and_single_cancel() {
    let (root, helper, request) = staged_fixture();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    handle(&helper, &Request::submit(request.intent().clone()).unwrap());
    fs::remove_dir(attempt(root.path()).join("state")).unwrap();
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
        scontrol_program: None,
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

#[test]
fn long_preparation_releases_metadata_lock_and_honors_cancellation_without_dispatch() {
    let (root, helper, request) = staged_fixture();
    drop(helper);
    let mut doc: Value = serde_json::from_str(request.prepared_content().unwrap().0).unwrap();
    doc["source_id"] = json!(format!("source.{}", "2".repeat(64)));
    doc["image_id"] = json!(format!("sha256:{}", "3".repeat(64)));
    doc["backend"]["oci_image"] =
        json!(format!("registry.example/image@sha256:{}", "3".repeat(64)));
    doc["data_preparation"] = json!({"script":"fetch.py","timeout_seconds":5});
    let manifest = doc.to_string();
    let mut intent = request.intent().clone();
    intent.manifest_sha256 = helper::sha256_bytes(manifest.as_bytes());
    let request = Request::prepare(
        intent,
        manifest,
        request.prepared_content().unwrap().1.into(),
    )
    .unwrap();
    let program = root.path().join("apptainer");
    fs::write(
        &program,
        "#!/bin/sh\nprintf started > \"$PROBE_STARTED\"\nwhile :; do sleep .02; done\n",
    )
    .unwrap();
    fs::set_permissions(&program, fs::Permissions::from_mode(0o700)).unwrap();
    let marker = root.path().join("download-started");
    let mut config = config(root.path());
    config.shared_run_root = Some(root.path().join("shared-runs"));
    config.data_preparer = Some(program);
    config
        .environment
        .insert("PROBE_STARTED".into(), marker.display().to_string());
    let helper = Helper::open(config).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let job = std::thread::spawn(move || helper.handle(&request, &flag));
    let deadline = Instant::now() + Duration::from_secs(3);
    while !marker.exists() && !job.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    let started = marker.exists();
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.path().join("helper/.slurm-helper.lock"))
        .unwrap();
    let available = fs2::FileExt::try_lock_exclusive(&lock).is_ok();
    if available {
        fs2::FileExt::unlock(&lock).unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(job.join().unwrap().is_err());
    assert!(started, "download was not started");
    assert!(available, "a download held the global helper metadata lock");
    let diagnostic: Value = serde_json::from_slice(
        &fs::read(claim(root.path()).join("data-preparation.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(diagnostic["cancelled"], true);
    assert_eq!(diagnostic["scheduler_submitted"], false);
    assert!(!claim(root.path()).join("state.json").exists());
    assert!(!claim(root.path()).join("submit.started.json").exists());
    assert_eq!(calls(root.path()), 0);
}
#[test]
fn managed_launch_is_staged_with_exact_local_binding_and_keeps_control_outputs_fresh() {
    let (root, helper, request) = staged_fixture();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    let plan = ml_exp_core::slurm::staging::plan(
        request.intent(),
        request.prepared_content().unwrap().0.as_bytes(),
    )
    .unwrap()
    .unwrap();
    let package = plan.managed.unwrap();
    let config: ml_exp_core::managed::Config = serde_json::from_slice(
        &fs::read(attempt(root.path()).join("managed-config.json")).unwrap(),
    )
    .unwrap();
    assert!(config.endpoint.is_empty());
    let binding = config.local_binding.as_ref().unwrap();
    assert_eq!(binding.identity.project, "demo");
    assert_eq!(binding.identity.run_id, "train");
    assert_eq!(binding.identity.attempt_id, "attempt-001");
    assert_eq!(
        binding.sha256,
        helper::sha256_bytes(&package.manifest_bytes().unwrap())[7..]
    );
    assert_eq!(package.manifest.argv, vec!["python", "train.py"]);
    assert_eq!(config.runtime_environment, vec!["CUDA_VISIBLE_DEVICES"]);
    assert!(!attempt(root.path()).join("worker").exists());
    assert!(!attempt(root.path()).join("outputs").exists());
    assert_eq!(
        handle(&helper, &Request::submit(request.intent().clone()).unwrap()).outcome,
        ReplyOutcome::Submitted {
            job_id: "12345".into()
        }
    );
    assert_eq!(calls(root.path()), 1);
}
#[test]
fn changed_launch_package_or_preexisting_worker_never_dispatches() {
    for change in ["config", "launch", "control", "outputs", "state"] {
        let (root, helper, request) = staged_fixture();
        assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
        let path = attempt(root.path());
        match change {
            "config" => fs::write(path.join("managed-config.json"), b"{}").unwrap(),
            "launch" => fs::write(path.join("launch-manifest.json"), b"{}").unwrap(),
            "control" => fs::create_dir(path.join("worker")).unwrap(),
            "outputs" => fs::create_dir(path.join("outputs")).unwrap(),
            "state" => fs::write(path.join("state/stale"), b"").unwrap(),
            _ => unreachable!(),
        }
        let submit = Request::submit(request.intent().clone()).unwrap();
        assert_eq!(
            handle(&helper, &submit).outcome,
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
fn managed_environment_admission_and_large_sealed_configuration_are_checked_before_dispatch() {
    let (root, helper, mut request) = staged_fixture();
    let mut intent = request.intent().clone();
    intent.staging.worker.base_environment.clear();
    assert!(
        ml_exp_core::slurm::staging::plan(
            &intent,
            request.prepared_content().unwrap().0.as_bytes()
        )
        .is_err()
    );
    let mut intent = request.intent().clone();
    intent
        .staging
        .worker
        .base_environment
        .insert("SYNTHETIC_LARGE_VALUE".into(), "x".repeat(40 * 1024));
    let (manifest, script) = request.prepared_content().unwrap();
    request = Request::prepare(intent, manifest.into(), script.into()).unwrap();
    assert_eq!(handle(&helper, &request).outcome, ReplyOutcome::Prepared);
    assert_eq!(
        handle(&helper, &Request::submit(request.intent().clone()).unwrap()).outcome,
        ReplyOutcome::Submitted {
            job_id: "12345".into()
        }
    );
    assert_eq!(calls(root.path()), 1);
}
