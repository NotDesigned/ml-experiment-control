use super::*;
use crate::backends::cloud::adapter::Transport;
use ml_exp_core::cloud::{Request, pai};
use std::sync::{Arc, Mutex};
#[derive(Default)]
struct State {
    creates: u32,
    stops: u32,
    visible: bool,
    stopped: bool,
    foreign: bool,
}
struct Fake {
    state: Arc<Mutex<State>>,
    row: Value,
}
impl Transport for Fake {
    fn request(&self, request: &Request, _: &AtomicBool) -> Result<Value> {
        request.validate().map_err(|_| Error::configuration())?;
        let mut state = self.state.lock().unwrap();
        let mut row = self.row.clone();
        if state.stopped {
            row["Status"] = json!("Stopped");
        }
        if state.foreign {
            row["Settings"]["Tags"]["ml-expd.attempt"] = json!("attempt-999");
        }
        match request.action.as_str() {
            "CreateJob" => {
                state.creates += 1;
                Err(Error {
                    code: "REPLY_LOST".into(),
                    http_status: None,
                    uncertain: true,
                })
            }
            "ListJobs" => Ok(if state.visible {
                json!({"Jobs":[row],"TotalCount":1})
            } else {
                json!({"Jobs":[],"TotalCount":0})
            }),
            "GetJob" => Ok(row),
            "ListQuotaWorkloads" => Ok(json!({"TotalCount":0,"Workloads":[]})),
            "StopJob" => {
                state.stops += 1;
                state.stopped = true;
                Err(Error {
                    code: "STOP_REPLY_LOST".into(),
                    http_status: None,
                    uncertain: true,
                })
            }
            _ => panic!("unexpected operation"),
        }
    }
}
fn setup() -> (Job, Profile, Value) {
    let job = Job {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        submission_token: "a".repeat(32),
        source_id: format!("source.{}", "b".repeat(64)),
        image: format!("r.example/team/image@sha256:{}", "c".repeat(64)),
        command: vec!["train".into()],
        gpus: 1,
        cpus: 8,
        memory_gb: 64,
        max_time_seconds: 300,
        priority: 5,
        read_only_storage: false,
    };
    let profile = pai::Profile {
        region: "us-southeast-1".into(),
        workspace_id: "253".into(),
        quota_id: "quota1".into(),
        max_gpus: 8,
        max_cpus: 200,
        max_memory_gb: 1600,
        gpu_type: None,
        storage: None,
        vpc: None,
    };
    let row = json!({"JobId":"dlc1","WorkspaceId":"253","ResourceId":"quota1","DisplayName":job.name().unwrap(),"Settings":{"Tags":job.tags(&profile.scope().unwrap()).unwrap()},"Status":"Running"});
    (job, Profile::Pai(profile), row)
}
#[test]
fn sync_recovery_lost_reply_keeps_failed_parent_and_cannot_create_a_second_child() {
    let temp = tempfile::tempdir().unwrap();
    Outbox::initialize(temp.path()).unwrap();
    let outbox = Outbox::open(temp.path()).unwrap();
    let (parent, profile, mut row) = setup();
    let allocation = Allocation::Guaranteed;
    let flag = AtomicBool::new(false);
    let request_id = "a".repeat(32);
    let action = "action-0123456789abcdef";
    let original = ml_exp_core::cloud::exchange::Sync {
        request_id: request_id.clone(),
        confirmation: format!("SYNC {action}"),
    }
    .job(&parent)
    .unwrap();
    outbox
        .prepare(&profile, &original, allocation, &|| Ok(()))
        .unwrap();
    let lock = outbox.lock("test").unwrap();
    let mut entry = outbox.load(&Outbox::slot(&original).unwrap()).unwrap();
    entry.started = true;
    entry.handle = Some(Handle {
        profile_sha256: profile.digest().unwrap(),
        request_sha256: job_digest(&original).unwrap(),
        job_id: "dlc-failed".into(),
        provider_uid: None,
    });
    entry.observation = Some(JobObservation {
        job_id: "dlc-failed".into(),
        provider_state: "Failed".into(),
        state: ml_exp_core::state::RunState::Failed,
        queue: None,
    });
    outbox.save(&entry, &lock).unwrap();
    drop(lock);
    let before = outbox
        .directory
        .read_regular(
            OsStr::new(&Outbox::slot(&original).unwrap()),
            2 * 1024 * 1024,
        )
        .unwrap();
    let retry = ml_exp_core::cloud::exchange::SyncRetry {
        expected_failed_job_id: "dlc-failed".into(),
        confirmation: format!("RETRY SYNC {action} {request_id} dlc-failed"),
    };
    let child = retry.job(action, &request_id, &parent).unwrap();
    let Profile::Pai(p) = &profile;
    row["DisplayName"] = json!(child.name().unwrap());
    row["Settings"]["Tags"] = json!(child.tags(&p.scope().unwrap()).unwrap());
    let state = Arc::new(Mutex::new(State::default()));
    let adapter = Adapter::new(
        profile.clone(),
        Fake {
            state: state.clone(),
            row,
        },
    )
    .unwrap();
    for _ in 0..2 {
        let prepared = outbox
            .prepare(&profile, &child, allocation, &|| Ok(()))
            .unwrap();
        outbox
            .execute(
                &adapter,
                &child,
                allocation,
                prepared["confirmation"].as_str().unwrap(),
                &|| Ok(()),
                &flag,
            )
            .unwrap();
    }
    assert_eq!(state.lock().unwrap().creates, 1);
    drop(outbox);
    let outbox = Outbox::open(temp.path()).unwrap();
    assert_eq!(
        outbox
            .reconcile(&adapter, &child, allocation, &|| Ok(()), &flag)
            .unwrap()["phase"],
        "RECONCILE_REQUIRED"
    );
    state.lock().unwrap().visible = true;
    assert_eq!(
        outbox
            .reconcile(&adapter, &child, allocation, &|| Ok(()), &flag)
            .unwrap()["phase"],
        "OWNED"
    );
    assert_eq!(state.lock().unwrap().creates, 1);
    assert_eq!(
        outbox
            .directory
            .read_regular(
                OsStr::new(&Outbox::slot(&original).unwrap()),
                2 * 1024 * 1024
            )
            .unwrap(),
        before
    );
}
#[test]
fn lost_create_and_stop_replies_restart_absence_and_foreign_identity_never_replay() {
    let temp = tempfile::tempdir().unwrap();
    Outbox::initialize(temp.path()).unwrap();
    assert!(Outbox::initialize(temp.path()).is_err());
    let outbox = Outbox::open(temp.path()).unwrap();
    let (job, profile, row) = setup();
    let state = Arc::new(Mutex::new(State::default()));
    let adapter = Adapter::new(
        profile.clone(),
        Fake {
            state: state.clone(),
            row,
        },
    )
    .unwrap();
    let flag = AtomicBool::new(false);
    let guard = || Ok(());
    let allocation = Allocation::Guaranteed;
    let prepared = outbox.prepare(&profile, &job, allocation, &guard).unwrap();
    let confirmation = prepared["confirmation"].as_str().unwrap();
    let result = outbox
        .execute(&adapter, &job, allocation, confirmation, &guard, &flag)
        .unwrap();
    assert_eq!(result["phase"], "RECONCILE_REQUIRED");
    assert_eq!(state.lock().unwrap().creates, 1);
    drop(outbox);
    let outbox = Outbox::open(temp.path()).unwrap();
    outbox
        .execute(&adapter, &job, allocation, confirmation, &guard, &flag)
        .unwrap();
    assert_eq!(state.lock().unwrap().creates, 1);
    let absent = outbox
        .reconcile(&adapter, &job, allocation, &guard, &flag)
        .unwrap();
    assert_eq!(absent["retry_safe"], false);
    assert_eq!(absent["phase"], "RECONCILE_REQUIRED");
    {
        let mut s = state.lock().unwrap();
        s.visible = true;
        s.foreign = true;
    }
    let rejected = outbox
        .reconcile(&adapter, &job, allocation, &guard, &flag)
        .unwrap();
    assert_eq!(rejected["phase"], "RECONCILE_REQUIRED");
    assert_eq!(rejected["diagnostic"]["code"], "CLOUD_EVIDENCE_INVALID");
    state.lock().unwrap().foreign = false;
    let owned = outbox
        .reconcile(&adapter, &job, allocation, &guard, &flag)
        .unwrap();
    assert_eq!(owned["handle"]["job_id"], "dlc1");
    assert_eq!(owned["observation"]["state"], "RUNNING");
    let confirmation = format!("STOP {}", Outbox::slot(&job).unwrap());
    let stopped = outbox
        .stop(&adapter, &job, allocation, &confirmation, &guard, &flag)
        .unwrap();
    assert_eq!(stopped["termination_verified"], false);
    assert_eq!(state.lock().unwrap().stops, 1);
    outbox
        .stop(&adapter, &job, allocation, &confirmation, &guard, &flag)
        .unwrap();
    assert_eq!(state.lock().unwrap().stops, 1);
    let verified = outbox
        .reconcile(&adapter, &job, allocation, &guard, &flag)
        .unwrap();
    assert_eq!(verified["termination_verified"], true);
    assert_eq!(verified["observation"]["state"], "CANCELLED");
    let Profile::Pai(mut changed) = profile;
    changed.quota_id = "different-quota".into();
    assert!(
        outbox
            .prepare(&Profile::Pai(changed), &job, allocation, &guard)
            .is_err()
    );
    assert_eq!(state.lock().unwrap().creates, 1);
    assert_eq!(state.lock().unwrap().stops, 1);
}
#[test]
fn cloud_runner_uses_the_action_claim_private_request_and_restart_reconciliation() {
    use crate::{
        backends::cloud::action::Runner,
        operations::{ClaimContext, Execute, Store},
    };
    use ml_exp_core::{
        cloud::action,
        first_submission::native::{Approval, Binding},
        operations::FrozenFiles,
    };
    let temp = tempfile::tempdir().unwrap();
    let effects = temp.path().join("effects");
    std::fs::create_dir(&effects).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&effects, std::fs::Permissions::from_mode(0o700)).unwrap();
    Outbox::initialize(&effects).unwrap();
    let outbox = Outbox::open(&effects).unwrap();
    let actions = Store::open(temp.path().join("actions")).unwrap();
    let (mut job, profile, row) = setup();
    let digest = format!("sha256:{}", "d".repeat(64));
    let package = ml_exp_core::cloud::launch::compile(ml_exp_core::cloud::launch::Input {
        root: "/mnt/workspace/runs",
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        workdir: "/workspace",
        argv: &["train".into()],
        timeout_seconds: 300,
        worker_sha256: &format!("sha256:{}", "e".repeat(64)),
        storage: ml_exp_core::cloud::launch::StorageLayout {
            mount: "/mnt/workspace".into(),
            runs: "/mnt/workspace/runs".into(),
            datasets: "/mnt/workspace/datasets".into(),
        },
        channel: ml_exp_core::cloud::exchange::Channel {
            url: "https://api.example/api/worker-exchange/demo/train/attempt-001".into(),
            token: "f".repeat(64),
        },
    })
    .unwrap();
    job.command = package.download_command("/worker").unwrap();
    let request = action::Execution {
        backend_id: "pai".into(),
        profile_sha256: profile.digest().unwrap(),
        allocation: Allocation::Guaranteed,
        binding: Binding {
            project: job.project.clone(),
            run_id: job.run_id.clone(),
            attempt_id: job.attempt_id.clone(),
            project_sha256: digest.clone(),
            run_sha256: digest.clone(),
            attempt_sha256: digest.clone(),
            source_receipt_sha256: digest.clone(),
            backend_config_sha256: profile.digest().unwrap(),
        },
        job: job.clone(),
        launch_package: Box::new(package),
    };
    let now = "2026-10-29T00:00:00Z";
    let expiry = "2026-10-29T01:00:00Z";
    let plan = action::prepare(
        &request,
        &Approval {
            max_gpu_hours: 1.,
            reason: "native adapter integration".into(),
        },
        now,
        expiry,
    )
    .unwrap();
    let id = plan["action_id"].as_str().unwrap();
    actions.save_plan(&plan, now, &"1".repeat(32)).unwrap();
    actions.bind_cloud_request(id, &request).unwrap();
    assert!(actions.cloud_request(id).unwrap() == request);
    let state = Arc::new(Mutex::new(State::default()));
    let adapter = Adapter::new(
        profile,
        Fake {
            state: state.clone(),
            row,
        },
    )
    .unwrap();
    let flag = AtomicBool::new(false);
    let files = FrozenFiles {
        execution_campaign_sha256: Some(digest.clone()),
        execution_manifest_sha256: Some(digest),
    };
    let confirmation = format!("EXECUTE {id}");
    let guard = || Ok(());
    let run = |runner: &Runner<'_, Fake>| {
        runner.execute(
            &request,
            Execute {
                action_id: id,
                confirmation: &confirmation,
                files: &files,
                claim: ClaimContext {
                    now,
                    transition_id: "22222222222222222222222222222222",
                    pid: std::process::id(),
                    reason: "operator approved",
                },
            },
            &guard,
            &flag,
        )
    };
    let runner = Runner {
        actions: &actions,
        adapter: &adapter,
        outbox: &outbox,
    };
    let refused = runner.execute(
        &request,
        Execute {
            action_id: id,
            confirmation: "EXECUTE action-0000000000000000",
            files: &files,
            claim: ClaimContext {
                now,
                transition_id: "33333333333333333333333333333333",
                pid: std::process::id(),
                reason: "",
            },
        },
        &guard,
        &flag,
    );
    assert!(refused.is_err());
    assert_eq!(state.lock().unwrap().creates, 0);
    let unknown = run(&runner).unwrap();
    assert_eq!(unknown["execution"]["status"], "RECONCILE_REQUIRED");
    assert_eq!(
        unknown["execution"]["execution_reason"],
        "operator approved"
    );
    assert_eq!(state.lock().unwrap().creates, 1);
    assert!(run(&runner).is_err());
    assert_eq!(state.lock().unwrap().creates, 1);
    state.lock().unwrap().visible = true;
    let verified = runner.reconcile(id, &guard, &flag).unwrap();
    assert_eq!(verified["execution"]["status"], "VERIFIED");
    assert_eq!(
        verified["execution"]["result"]["submission"]["backend_job_id"],
        "dlc1"
    );
    let repeat = run(&runner).unwrap();
    assert_eq!(repeat["execution"]["status"], "VERIFIED");
    assert_eq!(state.lock().unwrap().creates, 1);
    let private = actions.cloud_request(id).unwrap();
    assert!(private == request);
    let public = serde_json::to_string(&verified).unwrap();
    assert!(!public.contains(&job.submission_token));
}
/// Calls `hook` once, on the first GetJob after it is armed, then answers as `Fake`.
struct Hooked<F: Fn() + Send + Sync> {
    inner: Fake,
    armed: Arc<AtomicBool>,
    hook: F,
}
impl<F: Fn() + Send + Sync> Transport for Hooked<F> {
    fn request(&self, request: &Request, cancelled: &AtomicBool) -> Result<Value> {
        if request.action == "GetJob" && self.armed.swap(false, Ordering::SeqCst) {
            (self.hook)();
        }
        self.inner.request(request, cancelled)
    }
}
/// A job whose reply was lost and was then found: the entry holds a handle.
fn owned(outbox: &Outbox, adapter: &Adapter<impl Transport>, state: &Mutex<State>) -> Job {
    let (job, profile, _) = setup();
    let allocation = Allocation::Guaranteed;
    let flag = AtomicBool::new(false);
    let prepared = outbox
        .prepare(&profile, &job, allocation, &|| Ok(()))
        .unwrap();
    outbox
        .execute(
            adapter,
            &job,
            allocation,
            prepared["confirmation"].as_str().unwrap(),
            &|| Ok(()),
            &flag,
        )
        .unwrap();
    state.lock().unwrap().visible = true;
    let found = outbox
        .reconcile(adapter, &job, allocation, &|| Ok(()), &flag)
        .unwrap();
    assert_eq!(found["phase"], "OWNED");
    job
}
#[test]
fn a_status_read_waits_for_a_brief_holder() {
    let temp = tempfile::tempdir().unwrap();
    Outbox::initialize(temp.path()).unwrap();
    let outbox = Outbox::open(temp.path()).unwrap();
    let (job, profile, _) = setup();
    outbox
        .prepare(&profile, &job, Allocation::Guaranteed, &|| Ok(()))
        .unwrap();
    let (held, wait) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let other = Outbox::open(temp.path()).unwrap();
            let _lock = other.lock("test.holder").unwrap();
            held.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(500));
        });
        wait.recv().unwrap();
        let start = Instant::now();
        assert_eq!(outbox.status(&job).unwrap()["phase"], "PREPARED");
        assert!(outbox.optional_status(&job).unwrap().is_some());
        assert!(start.elapsed() >= Duration::from_millis(200));
    });
}
#[test]
fn an_observation_is_read_unlocked_and_discarded_when_the_entry_changed() {
    let temp = tempfile::tempdir().unwrap();
    Outbox::initialize(temp.path()).unwrap();
    let outbox = Outbox::open(temp.path()).unwrap();
    let (_, profile, row) = setup();
    let state = Arc::new(Mutex::new(State::default()));
    let path = temp.path().to_owned();
    let armed = Arc::new(AtomicBool::new(false));
    let adapter = Adapter::new(
        profile,
        Hooked {
            inner: Fake {
                state: state.clone(),
                row,
            },
            armed: armed.clone(),
            // Another writer, in the middle of the provider GET. Taking the lock at
            // all shows the reconcile does not hold it across that call.
            hook: move || {
                let other = Outbox::open(&path).unwrap();
                let lock = other.lock("test.writer").unwrap();
                let (job, _, _) = setup();
                let slot = Outbox::slot(&job).unwrap();
                let mut entry = other.load(&slot).unwrap();
                entry.observation = None;
                entry.diagnostic = Some(ErrorDocument {
                    code: "CONCURRENT_WRITER".into(),
                    http_status: None,
                    uncertain: false,
                });
                other.save(&entry, &lock).unwrap();
            },
        },
    )
    .unwrap();
    let job = owned(&outbox, &adapter, &state);
    let flag = AtomicBool::new(false);
    armed.store(true, Ordering::SeqCst);
    let raced = outbox
        .reconcile(&adapter, &job, Allocation::Guaranteed, &|| Ok(()), &flag)
        .unwrap();
    assert!(!armed.load(Ordering::SeqCst));
    // The concurrent record stands; the stale RUNNING reply was not written.
    assert_eq!(raced["diagnostic"]["code"], "CONCURRENT_WRITER");
    assert!(raced["observation"].is_null());
    assert_eq!(
        outbox.status(&job).unwrap()["diagnostic"]["code"],
        "CONCURRENT_WRITER"
    );
    // Unchanged entry: the next observation is committed as before.
    let observed = outbox
        .reconcile(&adapter, &job, Allocation::Guaranteed, &|| Ok(()), &flag)
        .unwrap();
    assert!(observed["diagnostic"].is_null());
    assert_eq!(observed["observation"]["state"], "RUNNING");
    assert_eq!(state.lock().unwrap().creates, 1);
}
