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
    let Profile::Pai(mut changed) = profile else {
        unreachable!()
    };
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
fn cloud_runner_uses_existing_action_authorization_private_request_and_restart_reconciliation() {
    use crate::{
        backends::cloud::action::Runner,
        operations::{ClaimContext, Execute, Store},
    };
    use ml_exp_core::{
        cloud::action,
        first_submission::native::{Approval, Binding},
        operations::{FrozenFiles, Policy},
    };
    let temp = tempfile::tempdir().unwrap();
    let effects = temp.path().join("effects");
    std::fs::create_dir(&effects).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&effects, std::fs::Permissions::from_mode(0o700)).unwrap();
    Outbox::initialize(&effects).unwrap();
    let outbox = Outbox::open(&effects).unwrap();
    let actions = Store::open(temp.path().join("actions")).unwrap();
    let (job, profile, row) = setup();
    let digest = format!("sha256:{}", "d".repeat(64));
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
    let policy = Policy {
        allow_scheduler_mutations: true,
        ..Default::default()
    };
    let files = FrozenFiles {
        execution_campaign_sha256: Some(digest.clone()),
        execution_manifest_sha256: Some(digest),
        ..Default::default()
    };
    let confirmation = format!("EXECUTE {id}");
    let guard = || Ok(());
    let run = |runner: &Runner<'_, Fake>| {
        runner.execute(
            &request,
            Execute {
                action_id: id,
                confirmation: &confirmation,
                policy: &policy,
                files: &files,
                claim: ClaimContext {
                    now,
                    transition_id: "22222222222222222222222222222222",
                    pid: std::process::id(),
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
    assert!(run(&runner).is_err());
    assert_eq!(state.lock().unwrap().creates, 0);
    actions
        .authorize(id, "operator", "approved", now, &"3".repeat(32))
        .unwrap();
    let unknown = run(&runner).unwrap();
    assert_eq!(unknown["execution"]["status"], "RECONCILE_REQUIRED");
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
