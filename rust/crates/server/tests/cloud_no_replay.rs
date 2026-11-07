//! Service-level no-replay proof for PAI and SenseCore. Real Run stores, Actions
//! and outboxes run against a fake provider that counts creates. Every scenario
//! ends by asserting how many creates the provider saw: an uncertain submission
//! must never be sent twice, whatever the service was asked to do next.
use ml_exp_core::{
    artifacts::FileReceipt,
    canonical,
    cloud::{Job, Request, action::Execution, data::Image, pai, run::Runtime, sensecore},
    datasets::{Definition, State},
    first_submission::native::{self, Approval, creation::Request as CreateRun},
    multipart::PartRecord,
    operations::FrozenFiles,
    slurm::{ExecutionConfig, JobSpec, Resources, SlurmConfig, Storage, helper},
    sources::Policy,
};
use ml_exp_io::directory::{Directory, FileLock, LockOptions};
use ml_exp_server::{
    administration,
    api::Api,
    backends::{
        cloud::{
            self,
            adapter::{Adapter, Profile, Transport},
            outbox::Outbox,
            service::{Continue, Service, Settings},
        },
        slurm::{Budget, SshConfig, service as slurm_service},
    },
    background::{Cycle, Polled},
    deletion::RunFence,
    error,
    operations::{ClaimContext, Store},
    read_config,
    runs::{self, Operation, Owner},
    sources,
};
use serde_json::{Value, json};
use std::{
    cell::Cell,
    collections::BTreeMap,
    fs,
    io::Cursor,
    ops::Range,
    os::unix::fs::PermissionsExt,
    path::Path,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::Duration,
};

const T_CREATE: &str = "2026-10-29T00:00:00Z";
const T_PREPARE: &str = "2026-10-29T00:01:00Z";
const T_EXECUTE: &str = "2026-10-29T00:03:00Z";
const T_RECONCILE: &str = "2026-10-29T00:04:00Z";
const GATE_EXPIRES: &str = "2026-10-29T01:01:00Z";
const T_GATE_EXPIRED: &str = "2026-10-29T02:00:00Z";
const RENEWED_GATE_EXPIRES: &str = "2026-10-29T03:00:00Z";
const PAI_JOB_ID: &str = "dlc-owned";
const SENSECORE_USER: &str = "12345678-1234-1234-1234-123456789abc";
const REQUEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const REQUEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const REQUEST_C: &str = "cccccccccccccccccccccccccccccccc";

#[derive(Clone, Copy)]
enum Kind {
    Pai,
    Sensecore,
}
const KINDS: [Kind; 2] = [Kind::Pai, Kind::Sensecore];

/// What the provider holds and how it answers. The row is the one job a test
/// has bound; `exists` says whether that job reached the provider.
#[derive(Default)]
struct World {
    row: Value,
    calls: Vec<String>,
    lose_create_reply: bool,
    exists: bool,
    foreign: bool,
    down: bool,
    /// Job reads that still succeed before the read path goes down.
    gets_left: Option<u32>,
    terminal: bool,
}
struct FakeProvider {
    profile: Profile,
    world: Mutex<World>,
}
impl FakeProvider {
    fn world(&self) -> MutexGuard<'_, World> {
        self.world.lock().unwrap()
    }
    fn calls(&self) -> Vec<String> {
        self.world().calls.clone()
    }
    fn creates(&self) -> usize {
        self.world()
            .calls
            .iter()
            .filter(|action| matches!(action.as_str(), "CreateJob" | "CreateTrainingJob"))
            .count()
    }
    fn reads(&self) -> usize {
        self.world()
            .calls
            .iter()
            .filter(|action| matches!(action.as_str(), "GetJob" | "GetTrainingJob"))
            .count()
    }
    fn bind(&self, job: &Job) {
        let name = job.name().unwrap();
        let row = match &self.profile {
            Profile::Pai(p) => {
                json!({"JobId":PAI_JOB_ID,"WorkspaceId":p.workspace_id,"ResourceId":p.quota_id,
                "DisplayName":name,"Settings":{"Tags":job.tags(&p.scope().unwrap()).unwrap()},"Status":"Running"})
            }
            Profile::Sensecore(p) => json!({"name":name,"uid":"owned-job-uid",
                "ownership":{"user_id":p.scope.user_id},"state":"RUNNING"}),
        };
        self.world().row = row;
    }
    fn observed_row(&self, world: &World) -> Value {
        let mut row = world.row.clone();
        match &self.profile {
            Profile::Pai(_) => {
                if world.terminal {
                    row["Status"] = json!("Failed");
                }
                if world.foreign {
                    row["Settings"]["Tags"]["ml-expd.attempt"] = json!("attempt-999");
                }
            }
            Profile::Sensecore(_) => {
                if world.terminal {
                    row["state"] = json!("FAILED");
                }
                if world.foreign {
                    row["ownership"]["user_id"] = json!("00000000-0000-0000-0000-000000000000");
                }
            }
        }
        row
    }
}
impl Transport for FakeProvider {
    fn request(&self, request: &Request, _: &AtomicBool) -> cloud::Result<Value> {
        request
            .validate()
            .map_err(|_| cloud::Error::configuration())?;
        let mut world = self.world();
        world.calls.push(request.action.clone());
        let action = request.action.as_str();
        if matches!(action, "CreateJob" | "CreateTrainingJob") {
            if world.lose_create_reply {
                return Err(cloud::Error {
                    code: "REPLY_LOST".into(),
                    http_status: None,
                    uncertain: true,
                });
            }
            world.exists = true;
            return Ok(match &self.profile {
                Profile::Pai(_) => json!({"JobId":PAI_JOB_ID}),
                Profile::Sensecore(_) => json!({"name":world.row["name"]}),
            });
        }
        let is_get = matches!(action, "GetJob" | "GetTrainingJob");
        if world.down || is_get && world.gets_left == Some(0) {
            return Err(cloud::Error {
                code: "PROVIDER_DOWN".into(),
                http_status: Some(503),
                uncertain: false,
            });
        }
        if is_get {
            world.gets_left = world.gets_left.map(|left| left - 1);
        }
        let row = self.observed_row(&world);
        match action {
            "ListJobs" => Ok(if world.exists {
                json!({"Jobs":[row],"TotalCount":1})
            } else {
                json!({"Jobs":[],"TotalCount":0})
            }),
            "ListTrainingJobs" => Ok(if world.exists {
                json!({"training_jobs":[row],"next_page_token":""})
            } else {
                json!({"training_jobs":[],"next_page_token":""})
            }),
            "GetJob" | "GetTrainingJob" if world.exists => Ok(row),
            "GetJob" | "GetTrainingJob" => Err(cloud::Error {
                code: "PROVIDER_NOT_FOUND".into(),
                http_status: Some(404),
                uncertain: false,
            }),
            "GetJobEvents" => Ok(json!({"Events":["scheduled"]})),
            "ListWorkers" => Ok(json!({"workers":[]})),
            other => panic!("unexpected provider operation {other}"),
        }
    }
}

fn private(path: &Path, value: &Value) {
    ml_exp_io::write_atomic(path, &serde_json::to_vec(value).unwrap(), true).unwrap();
}
fn archive() -> Vec<u8> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    let mut tar = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    let body = b"print('cloud fixture')\n";
    header.set_size(body.len() as u64);
    header.set_mode(0o500);
    header.set_cksum();
    tar.append_data(&mut header, "train.py", &body[..]).unwrap();
    tar.into_inner().unwrap().finish().unwrap()
}
fn provider_profile(kind: Kind) -> (Profile, &'static str, &'static str) {
    match kind {
        Kind::Pai => (
            Profile::Pai(pai::Profile {
                region: "us-southeast-1".into(),
                workspace_id: "253".into(),
                quota_id: "quota1".into(),
                max_gpus: 8,
                max_cpus: 200,
                max_memory_gb: 1600,
                gpu_type: None,
                storage: Some(pai::Storage {
                    cpfs_uri: "bmcpfs://cpfs-example.us-southeast-1.cpfs.aliyuncs.com/".into(),
                    workspace_user: "researcher".into(),
                    role_arn: "acs:ram::12345:role/pai-data-researcher".into(),
                }),
                vpc: None,
            }),
            "pai-h200",
            "/mnt/workspace/ml-expd/runs",
        ),
        Kind::Sensecore => (
            Profile::Sensecore(sensecore::Profile {
                scope: sensecore::Scope {
                    subscription: "sub".into(),
                    resource_group: "group".into(),
                    region: "cn-east-1".into(),
                    zone: "cn-east-1a".into(),
                    workspace_zone: Some("cn-east-1z".into()),
                    workspace: "work".into(),
                    user_id: SENSECORE_USER.into(),
                },
                cluster: "training".into(),
                vpc_id: "vpc1".into(),
                worker_spec: "gpu.8c64g".into(),
                worker_gpus: 1,
                worker_cpus: 8,
                worker_memory_gb: 64,
                worker_nodes: 1,
                storage_volume: "nas".into(),
                storage_subdir: "/personal/researcher".into(),
                mount_path: "/data".into(),
                purpose: sensecore::Purpose::Training,
                quota: sensecore::Quota::Spot,
            }),
            "acp-h200",
            "/data/ml-expd/runs",
        ),
    }
}

struct Fixture {
    temp: tempfile::TempDir,
    settings: read_config::Settings,
    sources: Arc<sources::Service>,
    config: Vec<u8>,
    executor: &'static str,
    runtime: Runtime,
    provider: Arc<FakeProvider>,
    transitions: AtomicU32,
}
impl Fixture {
    fn new(kind: Kind) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        read_config::initialize(&registry).unwrap();
        let project = temp.path().join("project.json");
        let runs = temp.path().join("runs");
        fs::create_dir(&runs).unwrap();
        private(
            &project,
            &json!({"schema_version":1,"project":"demo","title":"No-replay fixture","run_roots":["runs"]}),
        );
        let settings = read_config::Settings {
            schema_version: 1,
            registry_root: registry,
            projects: vec![read_config::Project {
                project: "demo".into(),
                project_file: project,
                run_roots: vec![runs],
                runs: Vec::new(),
                discover_runs: None,
            }],
            source_policy: Some(Policy::default()),
        };
        sources::Service::initialize(&settings).unwrap();
        let sources = Arc::new(sources::Service::open(&settings, None).unwrap());
        let bytes = archive();
        let source = sources
            .publish(
                "demo",
                &mut Cursor::new(&bytes),
                &helper::sha256_bytes(&bytes)[7..],
                T_CREATE,
            )
            .unwrap();
        let runtime = Runtime {
            runtime_id: format!("runtime.{}", "1".repeat(64)),
            source_id: source["source_id"].as_str().unwrap().into(),
            image: format!("registry.example/team/code@sha256:{}", "2".repeat(64)),
            source_dir: "/workspace".into(),
            worker_path: "/opt/ml-expd/worker".into(),
            worker_sha256: format!("sha256:{}", "3".repeat(64)),
        };
        let signing_key = temp.path().join("worker-signing-key");
        ml_exp_io::write_atomic(
            &signing_key,
            b"synthetic-exchange-key-0000000000000000000000",
            true,
        )
        .unwrap();
        let (profile, executor, run_root) = provider_profile(kind);
        // No credential file exists, so a real client could not even be built.
        let config = serde_json::to_vec(&json!({"schema_version":1,"enabled":true,"allow_run_creation":true,
            "exchange":{"public_url":"https://api.example/ml-expd","signing_key_file":signing_key},
            "backends":[{"executor":executor,"profile":profile,"run_root":run_root,
                "credentials_file":temp.path().join("no-credentials-exist"),"registry_auth_file":null,
                "runtimes":{runtime.runtime_id.clone():runtime}}]}))
        .unwrap();
        Service::initialize(&settings).unwrap();
        Self {
            temp,
            settings,
            sources,
            config,
            executor,
            runtime,
            provider: Arc::new(FakeProvider {
                profile,
                world: Mutex::default(),
            }),
            transitions: AtomicU32::new(1),
        }
    }
    /// Opening again is how each test models a daemon restart.
    fn open(&self) -> Service {
        Service::open_with_transport(
            self.settings.clone(),
            Settings::from_slice(&self.config).unwrap(),
            self.sources.clone(),
            None,
            Default::default(),
            self.provider.clone(),
        )
        .unwrap()
    }
    fn transition(&self) -> String {
        format!("{:032x}", self.transitions.fetch_add(1, Ordering::Relaxed))
    }
    fn run_directory(&self, run: &str) -> std::path::PathBuf {
        self.temp.path().join("runs").join(run)
    }
    /// A Run with a prepared first Action, ready to spend. The provider row
    /// is bound to this Run's job, though no provider request has been sent.
    fn ready(&self, service: &Service, run: &str) -> (String, Execution) {
        let request: CreateRun = serde_json::from_value(json!({"run_id":run,
            "runtime_id":self.runtime.runtime_id,"executor":self.executor,"command":["python","train.py"],
            "resources":{"gpus":1,"cpus":8,"memory_gb":64},"walltime_seconds":300,"seed":42}))
        .unwrap();
        service.create_run("demo", request, T_CREATE).unwrap();
        let prepared = self.prepare(service, run);
        assert_eq!(prepared["status"], "PREPARED");
        let action = prepared["submission_id"].as_str().unwrap().to_owned();
        let slot = self
            .settings
            .registry_root
            .join("cloud-submissions/slots")
            .join(format!(
                "{}.json",
                native::attempt_slot("demo", run, "attempt-001").unwrap()
            ));
        let saved: Value = serde_json::from_slice(&fs::read(slot).unwrap()).unwrap();
        let execution: Execution = serde_json::from_value(saved["execution"].clone()).unwrap();
        self.provider.bind(&execution.job);
        (action, execution)
    }
    fn prepare(&self, service: &Service, run: &str) -> Value {
        service
            .prepare(
                "demo",
                run,
                Approval {
                    max_gpu_hours: 1.,
                    reason: String::new(),
                },
                T_PREPARE,
                GATE_EXPIRES,
                &self.transition(),
            )
            .unwrap()
    }
    fn execute(&self, service: &Service, action: &str) -> cloud::Result<Value> {
        service.execute(
            action,
            &format!("EXECUTE {action}"),
            "owned fixture",
            T_EXECUTE,
            &self.transition(),
            &AtomicBool::new(false),
        )
    }
    fn get(&self, service: &Service, action: &str) -> Value {
        service.get(action, T_RECONCILE).unwrap()
    }
    fn reconcile(&self, service: &Service, action: &str) -> Value {
        service
            .reconcile(
                action,
                T_RECONCILE,
                &self.transition(),
                &AtomicBool::new(false),
            )
            .unwrap()
    }
    /// Everything `Runner::execute` does up to its claim, then the process dies.
    /// The Action stays EXECUTING and only the outbox handle survives the crash.
    fn die_after_claim(&self, action: &str, execution: &Execution) -> Outbox {
        let registry = &self.settings.registry_root;
        let outbox = Outbox::open(&registry.join("cloud-submissions/effects")).unwrap();
        outbox
            .prepare(
                &self.provider.profile,
                &execution.job,
                execution.allocation,
                &|| Ok(()),
            )
            .unwrap();
        let claimed = Store::open_existing(registry.join("actions"))
            .unwrap()
            .claim(
                action,
                &format!("EXECUTE {action}"),
                &FrozenFiles {
                    execution_campaign_sha256: Some(execution.binding.run_sha256.clone()),
                    execution_manifest_sha256: Some(execution.binding.attempt_sha256.clone()),
                },
                ClaimContext {
                    now: T_EXECUTE,
                    transition_id: &self.transition(),
                    pid: std::process::id(),
                    reason: "owned fixture",
                },
            )
            .unwrap();
        assert!(claimed.execute);
        outbox
    }
}
fn retry_request(run: &str) -> native::retry::Request {
    native::retry::Request {
        confirmation: String::new(),
        resume_from: None,
        restart_from: Some(native::retry::StartFrom {
            run_id: run.into(),
            attempt_id: "attempt-001".into(),
        }),
        max_gpu_hours: 1.,
        reason: String::new(),
    }
}
fn renewal(action: &str) -> ml_exp_core::renewal::Request {
    ml_exp_core::renewal::Request {
        request_id: "c".repeat(32),
        confirmation: format!("RENEW {action}"),
    }
}
fn expired_gate(transition_id: &str) -> slurm_service::Context<'_> {
    slurm_service::Context {
        now: T_GATE_EXPIRED,
        expires: RENEWED_GATE_EXPIRES,
        transition_id,
    }
}
fn open_gate(transition_id: &str) -> slurm_service::Context<'_> {
    slurm_service::Context {
        now: T_PREPARE,
        expires: GATE_EXPIRES,
        transition_id,
    }
}

#[test]
fn execute_creates_once_and_a_second_execute_makes_no_second_create() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        let (action, _) = f.ready(&service, "train");
        assert!(f.provider.calls().is_empty());
        let view = f.execute(&service, &action).unwrap();
        assert_eq!(view["status"], "VERIFIED");
        assert_eq!(view["cloud_observation"]["phase"], "OWNED");
        assert_eq!(view["cloud_observation"]["observation"]["state"], "RUNNING");
        assert_eq!(f.provider.creates(), 1);
        assert_eq!(f.execute(&service, &action).unwrap()["status"], "VERIFIED");
        assert_eq!(f.provider.creates(), 1);
        drop(service);
        let service = f.open();
        assert_eq!(f.execute(&service, &action).unwrap()["status"], "VERIFIED");
        assert_eq!(f.reconcile(&service, &action)["status"], "VERIFIED");
        assert_eq!(f.provider.creates(), 1);
    }
}

#[test]
fn a_lost_reply_is_reconciled_after_reopen_with_one_create() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        f.provider.world().lose_create_reply = true;
        let service = f.open();
        let (action, _) = f.ready(&service, "train");
        let view = f.execute(&service, &action).unwrap();
        assert_eq!(view["status"], "RECONCILE_REQUIRED");
        assert_eq!(view["next_action"], "RECONCILE");
        assert_eq!(view["cloud_observation"]["phase"], "RECONCILE_REQUIRED");
        assert_eq!(view["cloud_observation"]["retry_safe"], false);
        assert_eq!(f.provider.creates(), 1);
        assert_eq!(
            f.execute(&service, &action).unwrap_err().code,
            "CLOUD_ACTION_UNAVAILABLE"
        );
        assert_eq!(f.provider.creates(), 1);
        drop(service);
        let service = f.open();
        // A failed read and a foreign row both leave the evidence uncertain.
        f.provider.world().down = true;
        let view = f.reconcile(&service, &action);
        assert_eq!(view["status"], "RECONCILE_REQUIRED");
        assert_eq!(
            view["cloud_observation"]["diagnostic"]["code"],
            "PROVIDER_DOWN"
        );
        {
            let mut world = f.provider.world();
            world.down = false;
            // The original create did reach the provider, under some other owner's tags.
            world.exists = true;
            world.foreign = true;
        }
        let view = f.reconcile(&service, &action);
        assert_eq!(view["status"], "RECONCILE_REQUIRED");
        assert_eq!(
            view["cloud_observation"]["diagnostic"]["code"],
            "CLOUD_EVIDENCE_INVALID"
        );
        assert_eq!(f.provider.creates(), 1);
        f.provider.world().foreign = false;
        let view = f.reconcile(&service, &action);
        assert_eq!(view["status"], "VERIFIED");
        assert_eq!(view["cloud_observation"]["phase"], "OWNED");
        assert!(view["cloud_observation"]["handle"]["job_id"].is_string());
        assert_eq!(f.execute(&service, &action).unwrap()["status"], "VERIFIED");
        assert_eq!(f.provider.creates(), 1);
    }
}

/// The create was confirmed and its handle saved, then the first observation
/// failed. On SenseCore the recorded job name contains the submission token.
#[test]
fn a_failed_read_after_a_confirmed_create_recovers_with_one_create() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        // The create's own ownership read succeeds; the next read does not.
        f.provider.world().gets_left = Some(1);
        let service = f.open();
        let (action, _) = f.ready(&service, "train");
        let view = f.execute(&service, &action).unwrap();
        assert_eq!(view["status"], "RECONCILE_REQUIRED");
        assert_eq!(view["cloud_observation"]["phase"], "OWNED");
        assert_eq!(
            view["cloud_observation"]["diagnostic"]["code"],
            "PROVIDER_DOWN"
        );
        assert_eq!(f.provider.creates(), 1);
        drop(service);
        let service = f.open();
        let view = f.reconcile(&service, &action);
        assert_eq!(view["status"], "RECONCILE_REQUIRED");
        assert_eq!(f.provider.creates(), 1);
        f.provider.world().gets_left = None;
        let view = f.reconcile(&service, &action);
        assert_eq!(view["status"], "VERIFIED");
        assert_eq!(view["cloud_observation"]["observation"]["state"], "RUNNING");
        assert_eq!(f.execute(&service, &action).unwrap()["status"], "VERIFIED");
        assert_eq!(f.provider.creates(), 1);
    }
}

#[test]
fn absence_after_started_stays_reconcile_only() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        f.provider.world().lose_create_reply = true;
        let service = f.open();
        let (action, _) = f.ready(&service, "train");
        assert_eq!(
            f.execute(&service, &action).unwrap()["status"],
            "RECONCILE_REQUIRED"
        );
        drop(service);
        // Complete, repeated, provider-confirmed absence after a restart grants
        // nothing: the in-flight create cannot be fenced by a list.
        for _ in 0..3 {
            let service = f.open();
            let view = f.reconcile(&service, &action);
            assert_eq!(view["status"], "RECONCILE_REQUIRED");
            assert_eq!(view["next_action"], "RECONCILE");
            assert_eq!(view["cloud_observation"]["retry_safe"], false);
            assert!(view["cloud_observation"]["handle"].is_null());
            assert_eq!(
                f.execute(&service, &action).unwrap_err().code,
                "CLOUD_ACTION_UNAVAILABLE"
            );
            // Preparing again opens no way back to create either.
            let prepared = f.prepare(&service, "train");
            assert_eq!(prepared["reused"], true);
            assert_eq!(prepared["submission_id"], action.as_str());
            assert_eq!(prepared["status"], "RECONCILE_REQUIRED");
            assert_eq!(f.provider.creates(), 1);
        }
    }
}

#[test]
fn a_claim_that_died_before_its_durable_start_never_creates() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        let (action, execution) = f.ready(&service, "train");
        drop(service);
        f.die_after_claim(&action, &execution);
        let service = f.open();
        assert_eq!(
            service.get(&action, T_RECONCILE).unwrap()["status"],
            "EXECUTING"
        );
        let view = f.reconcile(&service, &action);
        assert_eq!(view["status"], "RECONCILE_REQUIRED");
        assert_eq!(view["cloud_observation"]["phase"], "PREPARED");
        // The claim is the authority here, even though the outbox never started.
        assert_eq!(
            f.execute(&service, &action).unwrap_err().code,
            "CLOUD_ACTION_UNAVAILABLE"
        );
        assert_eq!(
            f.reconcile(&service, &action)["status"],
            "RECONCILE_REQUIRED"
        );
        assert!(f.provider.calls().is_empty());
    }
}

#[test]
fn a_claim_that_died_after_its_create_recovers_with_one_create() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        f.provider.world().lose_create_reply = true;
        let service = f.open();
        let (action, execution) = f.ready(&service, "train");
        drop(service);
        let outbox = f.die_after_claim(&action, &execution);
        let slot = Outbox::slot(&execution.job).unwrap();
        let adapter = Adapter::new(f.provider.profile.clone(), &*f.provider).unwrap();
        let sent = outbox
            .execute(
                &adapter,
                &execution.job,
                execution.allocation,
                &format!("SUBMIT {slot}"),
                &|| Ok(()),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(sent["phase"], "RECONCILE_REQUIRED");
        assert_eq!(f.provider.creates(), 1);
        let service = f.open();
        assert_eq!(
            service.get(&action, T_RECONCILE).unwrap()["status"],
            "EXECUTING"
        );
        assert_eq!(
            f.reconcile(&service, &action)["status"],
            "RECONCILE_REQUIRED"
        );
        assert_eq!(
            f.execute(&service, &action).unwrap_err().code,
            "CLOUD_ACTION_UNAVAILABLE"
        );
        f.provider.world().exists = true;
        assert_eq!(f.reconcile(&service, &action)["status"], "VERIFIED");
        assert_eq!(f.execute(&service, &action).unwrap()["status"], "VERIFIED");
        assert_eq!(f.provider.creates(), 1);
    }
}

#[test]
fn retry_is_refused_on_absent_or_non_terminal_evidence() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        let flag = AtomicBool::new(false);
        // Never spent: there is no effect record, and no provider request is made.
        let (never, _) = f.ready(&service, "never");
        assert_eq!(
            service
                .retry(&never, retry_request("never"), T_RECONCILE, &flag)
                .unwrap_err()
                .code,
            "CLOUD_EVIDENCE_INVALID"
        );
        assert!(f.provider.calls().is_empty());

        f.provider.world().lose_create_reply = true;
        let (action, _) = f.ready(&service, "train");
        assert_eq!(
            f.execute(&service, &action).unwrap()["status"],
            "RECONCILE_REQUIRED"
        );
        let next = f.run_directory("train").join("attempts/attempt-002");
        // Absent after started.
        assert_eq!(
            service
                .retry(&action, retry_request("train"), T_RECONCILE, &flag)
                .unwrap_err()
                .code,
            "CLOUD_EVIDENCE_INVALID"
        );
        assert!(!next.exists());
        // Present and still running.
        f.provider.world().exists = true;
        assert_eq!(
            service
                .retry(&action, retry_request("train"), T_RECONCILE, &flag)
                .unwrap_err()
                .code,
            "CLOUD_EVIDENCE_INVALID"
        );
        assert!(!next.exists());
        // A provider failure is not evidence of termination.
        f.provider.world().down = true;
        assert!(
            service
                .retry(&action, retry_request("train"), T_RECONCILE, &flag)
                .is_err()
        );
        assert!(!next.exists());
        assert_eq!(f.provider.creates(), 1);
        // Exact terminal evidence is the only way through, and it only publishes
        // a new Attempt: the provider still sees one create.
        {
            let mut world = f.provider.world();
            world.down = false;
            world.terminal = true;
        }
        let retried = service
            .retry(&action, retry_request("train"), T_RECONCILE, &flag)
            .unwrap();
        assert_eq!(retried["retry_state"], "CREATED");
        assert_eq!(retried["attempt_id"], "attempt-002");
        assert_eq!(retried["scheduler_mutation_performed"], false);
        assert!(next.exists());
        assert_eq!(
            service
                .retry(&action, retry_request("train"), T_RECONCILE, &flag)
                .unwrap()["reused"],
            true
        );
        assert_eq!(f.provider.creates(), 1);
    }
}

/// SenseCore reports CANCELLED for these too, but neither shows the job is over, so a
/// new Attempt could train beside the old job.
#[test]
fn a_sensecore_job_that_may_still_run_cannot_be_retried() {
    for raw in ["DELETING", "SUSPENDED"] {
        let f = Fixture::new(Kind::Sensecore);
        let service = f.open();
        let flag = AtomicBool::new(false);
        let (action, _) = f.ready(&service, "train");
        f.execute(&service, &action).unwrap();
        let next = f.run_directory("train").join("attempts/attempt-002");
        f.provider.world().row["state"] = json!(raw);
        assert_eq!(
            service
                .retry(&action, retry_request("train"), T_RECONCILE, &flag)
                .unwrap_err()
                .code,
            "CLOUD_EVIDENCE_INVALID",
            "{raw}"
        );
        assert!(!next.exists(), "{raw}");

        f.provider.world().row["state"] = json!("DELETED");
        let retried = service
            .retry(&action, retry_request("train"), T_RECONCILE, &flag)
            .unwrap();
        assert_eq!(retried["retry_state"], "CREATED", "{raw}");
        assert!(next.exists(), "{raw}");
        assert_eq!(f.provider.creates(), 1, "{raw}");
    }
}

#[test]
fn renew_is_refused_once_started() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        // The control: an expired gate that never started does renew.
        let (control, _) = f.ready(&service, "control");
        let renewed = service
            .renew(&control, renewal(&control), expired_gate(&f.transition()))
            .unwrap();
        assert_eq!(renewed["status"], "PREPARED");
        assert_eq!(renewed["renewed_from"], control.as_str());
        assert_ne!(renewed["submission_id"], control.as_str());
        assert!(f.provider.calls().is_empty());

        f.provider.world().lose_create_reply = true;
        let (uncertain, _) = f.ready(&service, "uncertain");
        assert_eq!(
            f.execute(&service, &uncertain).unwrap()["status"],
            "RECONCILE_REQUIRED"
        );
        f.provider.world().lose_create_reply = false;
        let (verified, _) = f.ready(&service, "verified");
        assert_eq!(
            f.execute(&service, &verified).unwrap()["status"],
            "VERIFIED"
        );
        assert_eq!(f.provider.creates(), 2);
        for (run, action) in [("uncertain", &uncertain), ("verified", &verified)] {
            assert_eq!(
                service
                    .renew(action, renewal(action), expired_gate(&f.transition()))
                    .unwrap_err()
                    .code,
                "CLOUD_EVIDENCE_INVALID"
            );
            assert_eq!(
                service.for_run("demo", run, T_GATE_EXPIRED).unwrap()["submission_id"],
                action.as_str()
            );
        }
        assert_eq!(f.provider.creates(), 2);
    }
}

/// The cloud Runner writes its outbox entry before its claim, and renew refuses once an
/// entry exists. An expired execute has to stop ahead of the Runner, or it would make
/// the very command that replaces its gate refuse.
#[test]
fn an_expired_execute_makes_no_provider_call_and_leaves_renew_working() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        let (action, execution) = f.ready(&service, "train");
        let refused = service
            .execute(
                &action,
                &format!("EXECUTE {action}"),
                "too late",
                T_GATE_EXPIRED,
                &f.transition(),
                &AtomicBool::new(false),
            )
            .unwrap_err();
        assert_eq!(refused.code, "CLOUD_ACTION_UNAVAILABLE");
        assert!(!refused.uncertain);
        assert!(f.provider.calls().is_empty());
        assert_eq!(f.provider.creates(), 0);
        let view = service.get(&action, T_GATE_EXPIRED).unwrap();
        assert_eq!(view["status"], "PREPARED");
        assert_eq!(view["next_action"], "REPREPARE");
        assert!(view["execution"]["started_at"].is_null());
        assert!(view.get("cloud_observation").is_none(), "no outbox entry");
        let outbox =
            Outbox::open(&f.settings.registry_root.join("cloud-submissions/effects")).unwrap();
        assert!(outbox.optional_status(&execution.job).unwrap().is_none());

        let renewed = service
            .renew(&action, renewal(&action), expired_gate(&f.transition()))
            .unwrap();
        assert_eq!(renewed["status"], "PREPARED");
        assert_eq!(renewed["renewed_from"], action.as_str());
        assert_eq!(renewed["next_action"], "EXECUTE");
        assert!(f.provider.calls().is_empty());
    }
}

/// A request the claim would refuse leaves no outbox entry either, whichever part of it
/// is wrong. These cannot arrive over HTTP, which refuses them first.
#[test]
fn a_bad_confirmation_or_reason_leaves_no_outbox_entry() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        let (action, execution) = f.ready(&service, "train");
        let good = format!("EXECUTE {action}");
        for (confirmation, reason) in [("wrong", ""), (good.as_str(), "bad\nreason")] {
            assert!(
                service
                    .execute(
                        &action,
                        confirmation,
                        reason,
                        T_EXECUTE,
                        &f.transition(),
                        &AtomicBool::new(false),
                    )
                    .is_err()
            );
        }
        let outbox =
            Outbox::open(&f.settings.registry_root.join("cloud-submissions/effects")).unwrap();
        assert!(outbox.optional_status(&execution.job).unwrap().is_none());
        assert!(f.provider.calls().is_empty());
    }
}

/// Execute stays safe to repeat once its gate has passed: a verified Action answers from
/// its record, and an uncertain one is still sent to reconcile. Neither creates again.
#[test]
fn repeating_execute_after_the_gate_passes_never_creates_again() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        let again = |action: &str| {
            service.execute(
                action,
                &format!("EXECUTE {action}"),
                "a different reason",
                T_GATE_EXPIRED,
                &f.transition(),
                &AtomicBool::new(false),
            )
        };
        f.provider.world().lose_create_reply = true;
        let (uncertain, _) = f.ready(&service, "uncertain");
        assert_eq!(
            f.execute(&service, &uncertain).unwrap()["status"],
            "RECONCILE_REQUIRED"
        );
        assert_eq!(
            again(&uncertain).unwrap_err().code,
            "CLOUD_ACTION_UNAVAILABLE"
        );
        assert_eq!(f.provider.creates(), 1);

        f.provider.world().lose_create_reply = false;
        let (verified, _) = f.ready(&service, "verified");
        let first = f.execute(&service, &verified).unwrap();
        assert_eq!(first["status"], "VERIFIED");
        assert_eq!(first["execution"]["execution_reason"], "owned fixture");
        let repeated = again(&verified).unwrap();
        assert_eq!(repeated["status"], "VERIFIED");
        assert_eq!(repeated["execution"]["execution_reason"], "owned fixture");
        assert_eq!(f.provider.creates(), 2);
    }
}

/// The pointer is the last write of a renewal. Rewinding it and refusing new files
/// in the slot directory stops a repeat exactly there, because every earlier record
/// already matches and writes nothing. CI runs this as an ordinary user: root ignores
/// the directory mode, so under root the first `unwrap_err` panics instead of passing.
#[test]
fn a_renewal_whose_pointer_write_fails_is_uncertain_and_a_repeat_finishes_it() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        let (old, _) = f.ready(&service, "train");
        let slots = f.settings.registry_root.join("cloud-submissions/slots");
        let pointer = slots.join(format!(
            "{}.json",
            native::attempt_slot("demo", "train", "attempt-001").unwrap()
        ));
        let before = fs::read(&pointer).unwrap();
        let transition = f.transition();
        let renewed = service
            .renew(&old, renewal(&old), expired_gate(&transition))
            .unwrap();
        let new = renewed["submission_id"].as_str().unwrap().to_owned();
        assert_ne!(new, old);

        fs::write(&pointer, &before).unwrap();
        let writable = fs::metadata(&slots).unwrap().permissions();
        fs::set_permissions(&slots, fs::Permissions::from_mode(0o500)).unwrap();
        let failed = service
            .renew(&old, renewal(&old), expired_gate(&transition))
            .unwrap_err();
        fs::set_permissions(&slots, writable).unwrap();
        assert!(failed.uncertain);
        assert_eq!(failed.code, "CLOUD_RENEWAL_UNCERTAIN");
        let public = error::Error::from(failed);
        assert_eq!(public.status, 500);
        assert_eq!(public.code, "CLOUD_OPERATION_UNCERTAIN");
        assert_eq!(fs::read(&pointer).unwrap(), before);
        assert_eq!(
            service.for_run("demo", "train", T_GATE_EXPIRED).unwrap()["submission_id"],
            old.as_str()
        );

        let repeated = service
            .renew(&old, renewal(&old), expired_gate(&transition))
            .unwrap();
        assert_eq!(repeated["submission_id"], new.as_str());
        assert_eq!(repeated["status"], "PREPARED");
        assert_eq!(
            service.for_run("demo", "train", T_GATE_EXPIRED).unwrap()["submission_id"],
            new.as_str()
        );
        assert!(f.provider.calls().is_empty());
    }
}

/// The real cloud observer, one second per step, with the default 10 s interval.
struct Observer<P: Polled> {
    polled: P,
    cycle: Cycle,
}
impl<P: Polled> Observer<P> {
    fn walk(&mut self, seconds: Range<u64>) {
        let clock = Cell::new(0);
        for second in seconds {
            clock.set(second);
            self.cycle
                .step(&self.polled, &|| clock.get(), &AtomicBool::new(false));
        }
    }
}
fn observer(service: Arc<Service>) -> Observer<impl Polled> {
    Observer {
        polled: service.observer_polling(),
        cycle: Cycle::new(ml_exp_core::polling::Policy {
            enabled: true,
            ..Default::default()
        }),
    }
}

#[test]
fn an_observed_job_costs_one_provider_read_per_cycle() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = Arc::new(f.open());
        let (action, _) = f.ready(&service, "train");
        assert_eq!(f.execute(&service, &action).unwrap()["status"], "VERIFIED");
        let (calls, reads) = (f.provider.calls().len(), f.provider.reads());
        let mut observer = observer(service.clone());
        // Scheduler at 0, then Worker at 1 and History at 2 with no provider call.
        observer.walk(0..3);
        assert_eq!(f.provider.reads(), reads + 1);
        observer.walk(3..12);
        assert_eq!(f.provider.reads(), reads + 1);
        observer.walk(12..15);
        assert_eq!(f.provider.reads(), reads + 2);
        observer.walk(15..24);
        assert_eq!(f.provider.reads(), reads + 2);
        observer.walk(24..27);
        assert_eq!(f.provider.reads(), reads + 3);
        assert_eq!(f.provider.calls().len(), calls + 3);
        assert_eq!(f.provider.creates(), 1);
    }
}

#[test]
fn a_provider_outage_is_read_at_the_same_rate_and_never_creates() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = Arc::new(f.open());
        let (action, _) = f.ready(&service, "train");
        f.execute(&service, &action).unwrap();
        let reads = f.provider.reads();
        f.provider.world().down = true;
        let mut observer = observer(service.clone());
        // The outbox keeps a failed read as a diagnostic, so it counts as a sample.
        observer.walk(0..27);
        assert_eq!(f.provider.reads(), reads + 3);
        let view = f.get(&service, &action);
        assert_eq!(
            view["cloud_observation"]["diagnostic"]["code"],
            "PROVIDER_DOWN"
        );
        assert_eq!(f.provider.creates(), 1);
    }
}

/// The outbox refuses this read after the provider call, so each one costs a GET. The
/// phases with no work must not reset the failure count between them.
#[test]
fn a_read_refused_after_the_provider_call_backs_off_to_the_cap() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = Arc::new(f.open());
        let (action, _) = f.ready(&service, "train");
        f.execute(&service, &action).unwrap();
        service
            .refresh(&action, T_RECONCILE, &AtomicBool::new(false))
            .unwrap();
        // RUNNING does not allow STARTING.
        match kind {
            Kind::Pai => f.provider.world().row["Status"] = json!("EnvPreparing"),
            Kind::Sensecore => f.provider.world().row["state"] = json!("STARTING"),
        }
        assert!(
            service
                .refresh(&action, T_RECONCILE, &AtomicBool::new(false))
                .is_err()
        );
        let reads = f.provider.reads();
        // Reads at 0, 22, 64, 146, 308, 610, 912 and 1214, then 300 s apart.
        observer(service.clone()).walk(0..1300);
        assert_eq!(f.provider.reads(), reads + 8);
        assert_eq!(f.provider.creates(), 1);
    }
}

#[test]
fn a_job_that_vanishes_under_the_observer_is_never_created_again() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = Arc::new(f.open());
        let (action, _) = f.ready(&service, "train");
        f.execute(&service, &action).unwrap();
        let reads = f.provider.reads();
        f.provider.world().exists = false;
        observer(service.clone()).walk(0..120);
        assert!(f.provider.reads() > reads);
        assert_eq!(f.provider.creates(), 1);
    }
}

#[test]
fn an_action_awaiting_reconcile_is_not_sampled_at_all() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        f.provider.world().lose_create_reply = true;
        let service = Arc::new(f.open());
        let (action, _) = f.ready(&service, "train");
        let view = f.execute(&service, &action).unwrap();
        assert_eq!(view["status"], "RECONCILE_REQUIRED");
        let calls = f.provider.calls().len();
        observer(service.clone()).walk(0..120);
        assert_eq!(f.provider.calls().len(), calls);
        assert_eq!(f.provider.creates(), 1);
    }
}

#[test]
fn a_terminal_job_leaves_the_observer_at_the_next_discovery() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = Arc::new(f.open());
        let (action, _) = f.ready(&service, "train");
        f.execute(&service, &action).unwrap();
        let reads = f.provider.reads();
        f.provider.world().terminal = true;
        let mut observer = observer(service.clone());
        observer.walk(0..30);
        assert_eq!(f.provider.reads(), reads + 3);
        observer.walk(30..120);
        assert_eq!(f.provider.reads(), reads + 3);
        assert_eq!(f.provider.creates(), 1);
    }
}

/// One Slurm backend's configuration over the fixture's registry and source.
/// Copied from the fixture in `backends/slurm/tests/submission_tests.rs`, which is
/// private to its crate.
fn slurm_parts(f: &Fixture) -> (helper::Target, native::Runtime, native::Profile) {
    let root = f.temp.path();
    slurm_service::Service::initialize(&f.settings).unwrap();
    let source_id = f.runtime.source_id.clone();
    let image = format!("sha256:{}", "1".repeat(64));
    let mount_path = root.join("shared");
    fs::create_dir_all(mount_path.join("runs")).unwrap();
    let mount = mount_path.to_str().unwrap();
    let target = helper::Target {
        backend_id: "slurm-test".into(),
        profile_sha256: helper::sha256_bytes(b"profile"),
        storage_scope: helper::sha256_bytes(b"storage"),
    };
    let runtime = native::Runtime {
        worker: ml_exp_core::slurm::managed::Runtime {
            program: "/opt/ml-expd/bin/ml-exp-worker-rs".into(),
            base_environment: BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
            runtime_environment: Vec::new(),
        },
        source_dir: "/workspace".into(),
        runtime_id: format!("runtime.{}", "2".repeat(64)),
        source_id: source_id.clone(),
        image_id: image.clone(),
        sif_path: format!("{mount}/images/train.sif"),
        sif_sha256: helper::sha256_bytes(b"synthetic materialized SIF"),
        oci_image: format!("registry.invalid/demo@{image}"),
    };
    let profile = native::Profile {
        target: target.clone(),
        ssh_alias: "fake-gateway".into(),
        partition: "h100".into(),
        account: "science".into(),
        qos: "normal".into(),
        gpu_type: "h100".into(),
        mount_root: mount.into(),
        run_root: format!("{mount}/runs"),
        data_root: format!("{mount}/data"),
        max_gpus: 8,
        max_cpus: 64,
        max_memory_gb: 512,
        max_walltime_seconds: 86400,
    };
    (target, runtime, profile)
}
/// The Slurm service over the same registry with one real prepared Action.
fn slurm_side(f: &Fixture) -> (slurm_service::Service, String, std::path::PathBuf) {
    let run = f.run_directory("train");
    fs::create_dir_all(run.join("attempts/attempt-001")).unwrap();
    let (target, runtime, profile) = slurm_parts(f);
    let source_id = runtime.source_id.clone();
    let image = runtime.image_id.clone();
    let mount = profile.mount_root.as_str();
    let spec = JobSpec {
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        image_id: image.clone(),
        backend: SlurmConfig {
            kind: "slurm".into(),
            ssh_alias: profile.ssh_alias.clone(),
            partition: profile.partition.clone(),
            account: profile.account.clone(),
            qos: profile.qos.clone(),
            gres: "gpu:h100:1".into(),
            time: "00:05:00".into(),
            source_dir: format!("{mount}/sources/{source_id}"),
            sif_path: runtime.sif_path.clone(),
            mount_root: mount.into(),
            oci_image: Some(runtime.oci_image.clone()),
            apptainer_cache_dir: None,
            apptainer_tmp_dir: None,
            apptainer_unsquash: false,
            image_stage_timeout_seconds: 1080,
        },
        storage: Storage {
            run_dir: format!("{mount}/runs/demo/train"),
            project_data_root: format!("{mount}/data/demo"),
        },
        resources: Resources {
            gpus: 1,
            cpus: 8,
            nodes: 1,
            memory_gb: Some(16),
        },
        execution: ExecutionConfig {
            source_mount: "/workspace".into(),
            workdir: "/workspace".into(),
            managed_io: true,
            env: Default::default(),
        },
        command: vec!["python".into(), "-u".into(), "/workspace/train.py".into()],
    };
    let manifest = json!({"schema_version":1,"project":"demo","run_id":"train","source_id":source_id,"image_id":image,"config_path":"config.json","seed":42,"campaign":"fixture",
        "backend":spec.backend,"storage":spec.storage,"resources":spec.resources,"execution":spec.execution,"command":spec.command,"container":{"runtime_id":runtime.runtime_id},"evaluation":{"protocol_id":"fixture"},"parameters":{},"inputs":[],"native_creation":{"format":"native-run-creation.v1"}});
    private(&run.join("manifest.json"), &manifest);
    let mut attempt = manifest;
    attempt["attempt_id"] = json!("attempt-001");
    private(&run.join("attempts/attempt-001/attempt.json"), &attempt);
    let status = json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","state":"NOT_SUBMITTED"});
    private(&run.join("status.json"), &status);
    private(&run.join("attempts/attempt-001/status.json"), &status);
    let (service, calls) = slurm_open(f, target, runtime, profile);
    let prepared = service
        .prepare(
            "demo",
            "train",
            Approval {
                max_gpu_hours: 1.,
                reason: "owned test".into(),
            },
            open_gate(&f.transition()),
        )
        .unwrap();
    assert_eq!(prepared["status"], "PREPARED");
    let action = prepared["submission_id"].as_str().unwrap().to_owned();
    (service, action, calls)
}
/// The service, and an SSH program that only records that it was called.
fn slurm_open(
    f: &Fixture,
    target: helper::Target,
    runtime: native::Runtime,
    profile: native::Profile,
) -> (slurm_service::Service, std::path::PathBuf) {
    let root = f.temp.path();
    for name in ["ssh-config", "known-hosts"] {
        // A test may open this service more than once, and these refuse to be replaced.
        if !root.join(name).exists() {
            ml_exp_io::write_atomic(&root.join(name), b"synthetic test fixture\n", true).unwrap();
        }
    }
    let calls = root.join("ssh-calls");
    let script = root.join("fake-ssh");
    fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf 'call\\n' >> '{}'\nexit 1\n",
            calls.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
    let service = slurm_service::Service::open(
        f.settings.clone(),
        f.sources.clone(),
        None,
        vec![slurm_service::Backend {
            profile,
            ssh: SshConfig {
                target,
                ssh_binary: script,
                config_file: root.join("ssh-config"),
                known_hosts_file: root.join("known-hosts"),
                target_alias: "fake-gateway".into(),
                helper_path: "/opt/ml-expd/bin/ml-exp-worker-rs".into(),
                helper_config_path: "/etc/ml-expd/slurm-helper.json".into(),
                local_cwd: root.into(),
                agent_socket: Some(root.join("agent.sock")),
                budget: Budget {
                    timeout: Duration::from_secs(5),
                    connect_timeout_seconds: 1,
                    termination_grace: Duration::from_millis(20),
                    drain_grace: Duration::from_millis(100),
                },
            },
            runtimes: BTreeMap::from([(runtime.runtime_id.clone(), runtime)]),
            data_helper_config: None,
            script_preparation: false,
        }],
        true,
    )
    .unwrap();
    (service, calls)
}
fn stops_at_slurm_load(result: Result<Value, slurm_service::Error>) {
    match result {
        Err(slurm_service::Error::Authority) => {}
        other => panic!("a cloud Action must stop at the Slurm load, got {other:?}"),
    }
}
fn stops_at_cloud_load(result: cloud::Result<Value>) {
    match result {
        Err(error) if error.code == "CLOUD_SUBMISSION_NOT_FOUND" => {
            assert_eq!(error.http_status, Some(404));
        }
        other => panic!("a Slurm Action must stop at the cloud load, got {other:?}"),
    }
}

#[test]
fn an_action_given_to_the_other_backend_is_refused_with_no_provider_call() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let cloud_service = f.open();
        let (cloud_action, _) = f.ready(&cloud_service, "cloud-run");
        let (slurm, slurm_action, ssh_calls) = slurm_side(&f);
        let flag = AtomicBool::new(false);
        assert!(cloud_service.handles_action(&cloud_action).unwrap());
        assert!(!cloud_service.handles_action(&slurm_action).unwrap());

        let a = cloud_action.as_str();
        stops_at_slurm_load(slurm.get(a, T_EXECUTE));
        stops_at_slurm_load(slurm.execute(
            a,
            &format!("EXECUTE {a}"),
            "tester",
            open_gate(&f.transition()),
            &flag,
        ));
        stops_at_slurm_load(slurm.reconcile(a, T_EXECUTE, &flag));
        stops_at_slurm_load(slurm.refresh_scheduler(a, T_EXECUTE, &flag));
        stops_at_slurm_load(slurm.cancel(a, &format!("CANCEL {a}"), T_EXECUTE, &flag));
        stops_at_slurm_load(slurm.retry(a, retry_request("cloud-run"), T_EXECUTE, &flag));
        stops_at_slurm_load(slurm.renew(a, renewal(a), expired_gate(&f.transition())));
        assert_eq!(
            cloud_service.get(a, T_EXECUTE).unwrap()["status"],
            "PREPARED"
        );

        let a = slurm_action.as_str();
        stops_at_cloud_load(cloud_service.get(a, T_EXECUTE));
        stops_at_cloud_load(f.execute(&cloud_service, a));
        stops_at_cloud_load(cloud_service.reconcile(a, T_EXECUTE, &f.transition(), &flag));
        stops_at_cloud_load(cloud_service.refresh(a, T_EXECUTE, &flag));
        stops_at_cloud_load(cloud_service.cancel(a, &format!("CANCEL {a}"), T_EXECUTE, &flag));
        stops_at_cloud_load(cloud_service.retry(a, retry_request("train"), T_EXECUTE, &flag));
        stops_at_cloud_load(cloud_service.renew(a, renewal(a), expired_gate(&f.transition())));
        assert_eq!(slurm.get(a, T_EXECUTE).unwrap()["status"], "PREPARED");

        assert!(f.provider.calls().is_empty());
        assert!(!ssh_calls.exists());
    }
}

/// A provider that holds many jobs, as data copies need. Each create adds a row
/// built from the request it received, so every later read must find its own job.
#[derive(Default)]
struct CopyWorld {
    rows: Vec<Value>,
    calls: Vec<String>,
    lose_create_reply: bool,
    reads_down: bool,
    /// What a pod's log read returns, by pod.
    pod_logs: BTreeMap<String, Value>,
}
struct CopyProvider {
    kind: Kind,
    world: Mutex<CopyWorld>,
}
impl CopyProvider {
    fn world(&self) -> MutexGuard<'_, CopyWorld> {
        self.world.lock().unwrap()
    }
    fn creates(&self) -> usize {
        self.world()
            .calls
            .iter()
            .filter(|action| matches!(action.as_str(), "CreateJob" | "CreateTrainingJob"))
            .count()
    }
    /// Set the newest job's state, named as SenseCore names it.
    fn set_state(&self, state: &str) {
        let (field, text) = match (self.kind, state) {
            (Kind::Sensecore, _) => ("state", state),
            (Kind::Pai, "FAILED") => ("Status", "Failed"),
            (Kind::Pai, "SUCCEEDED") => ("Status", "Succeeded"),
            (Kind::Pai, "CANCELLED") => ("Status", "Stopped"),
            (Kind::Pai, other) => panic!("no PAI state for {other}"),
        };
        self.world().rows.last_mut().unwrap()[field] = json!(text);
    }
    fn row_for_create(&self, body: &Value, number: usize) -> (Value, Value) {
        match self.kind {
            Kind::Pai => {
                let id = format!("dlc-copy-{number}");
                (
                    json!({"JobId":id,"WorkspaceId":body["WorkspaceId"],"ResourceId":body["ResourceId"],
                        "DisplayName":body["DisplayName"],"Settings":{"Tags":body["Settings"]["Tags"]},"Status":"Running",
                        "Pods":[{"PodId":"pod-a"},{"PodId":"pod-b"}]}),
                    json!({"JobId":id}),
                )
            }
            Kind::Sensecore => (
                json!({"name":body["name"],"uid":format!("copy-uid-{number}"),
                    "ownership":{"user_id":SENSECORE_USER},"state":"RUNNING"}),
                json!({"name":body["name"]}),
            ),
        }
    }
}
impl Transport for CopyProvider {
    fn request(&self, request: &Request, _: &AtomicBool) -> cloud::Result<Value> {
        request
            .validate()
            .map_err(|_| cloud::Error::configuration())?;
        let mut world = self.world();
        world.calls.push(request.action.clone());
        let named = |row: &&Value, field: &str, name: &str| row[field] == name;
        if world.reads_down && !request.action.starts_with("Create") {
            return Err(cloud::Error {
                code: "PROVIDER_DOWN".into(),
                http_status: Some(503),
                uncertain: false,
            });
        }
        match request.action.as_str() {
            "CreateJob" | "CreateTrainingJob" => {
                if world.lose_create_reply {
                    return Err(cloud::Error {
                        code: "REPLY_LOST".into(),
                        http_status: None,
                        uncertain: true,
                    });
                }
                let body = request.body.as_ref().unwrap();
                let (row, reply) = self.row_for_create(body, world.rows.len());
                world.rows.push(row);
                Ok(reply)
            }
            "GetJob" | "GetTrainingJob" => {
                let id = request.path.rsplit('/').next().unwrap();
                let field = match self.kind {
                    Kind::Pai => "JobId",
                    Kind::Sensecore => "name",
                };
                world
                    .rows
                    .iter()
                    .find(|row| named(row, field, id))
                    .cloned()
                    .ok_or_else(|| cloud::Error {
                        code: "PROVIDER_NOT_FOUND".into(),
                        http_status: Some(404),
                        uncertain: false,
                    })
            }
            "ListJobs" => {
                let name = &request.query["DisplayName"];
                let rows: Vec<_> = world
                    .rows
                    .iter()
                    .filter(|row| named(row, "DisplayName", name))
                    .collect();
                Ok(json!({"Jobs":rows,"TotalCount":rows.len()}))
            }
            "ListTrainingJobs" => {
                let name = &request.query["name"];
                let rows: Vec<_> = world
                    .rows
                    .iter()
                    .filter(|row| named(row, "name", name))
                    .collect();
                Ok(json!({"training_jobs":rows,"next_page_token":""}))
            }
            "GetPodLogs" => {
                let pod = request.path.rsplit('/').nth(1).unwrap();
                let logs = world.pod_logs.get(pod).cloned().unwrap_or(json!([]));
                Ok(json!({"Logs":logs}))
            }
            "GetJobEvents" => Ok(json!({"Events":["scheduled"]})),
            "ListWorkers" => Ok(json!({"workers":[]})),
            other => panic!("unexpected provider operation {other}"),
        }
    }
}

/// One verified dataset archive, waiting to be copied on the backend.
struct CopyFixture {
    f: Fixture,
    config: Vec<u8>,
    provider: Arc<CopyProvider>,
    kind: Kind,
    asset: String,
}
impl CopyFixture {
    fn new(kind: Kind, files: usize) -> Self {
        let f = Fixture::new(kind);
        let mut config: Value = serde_json::from_slice(&f.config).unwrap();
        match &f.provider.profile {
            Profile::Pai(_) => {
                config["backends"][0]["collector"] = json!(f.provider.profile);
            }
            Profile::Sensecore(training) => {
                let mut copier = training.clone();
                copier.purpose = sensecore::Purpose::DataCopy;
                copier.cluster = "debug-cpu".into();
                copier.worker_spec = "cpu.2c4g".into();
                copier.worker_gpus = 0;
                copier.worker_cpus = 2;
                copier.worker_memory_gb = 4;
                copier.quota = sensecore::Quota::Reserved;
                config["backends"][0]["data_copier"] = json!(Profile::Sensecore(copier));
            }
        }
        let config = serde_json::to_vec(&config).unwrap();
        let settings = Settings::from_slice(&config).unwrap();
        let definition = Definition {
            sha256: "e".repeat(64),
            archive_bytes: 1024 * 1024,
            files: (0..files)
                .map(|n| FileReceipt {
                    path: format!("dataset/tasks/sample-{n:05}.bin"),
                    bytes: 128,
                    sha256: "f".repeat(64),
                })
                .collect(),
        };
        let mut state = State::new(
            "demo".into(),
            settings.backends[0].target().unwrap(),
            definition.clone(),
        )
        .unwrap();
        state.parts.insert(
            0,
            PartRecord {
                bytes: definition.archive_bytes,
                sha256: "d".repeat(64),
            },
        );
        let asset = state.asset_id.clone();
        let saved = match kind {
            Kind::Pai => {
                json!({"state":state,"archive_verified":true,"image_recipe":null,"image_started":false,
                    "image":null,"copy_request_id":null,"copy_status":null,"diagnostic":null})
            }
            Kind::Sensecore => {
                let recipe = Image {
                    project: "demo".into(),
                    executor: f.executor.into(),
                    definition,
                    base_image: format!("registry.example/team/base@sha256:{}", "4".repeat(64)),
                    worker_sha256: format!("sha256:{}", "5".repeat(64)),
                };
                json!({"state":state,"archive_verified":true,"image_recipe":recipe,"image_started":true,
                    "image":format!("registry.sensecore.cn/team/data@sha256:{}", "6".repeat(64)),
                    "copy_request_id":null,"copy_storage":settings.backends[0].storage_layout().unwrap(),
                    "copy_status":null,"diagnostic":null})
            }
        };
        let name = format!(
            "dataset-{}.json",
            canonical::digest(&json!(["demo", f.executor, asset])).unwrap()
        );
        private(
            &f.settings
                .registry_root
                .join("cloud-submissions/slots")
                .join(name),
            &saved,
        );
        Self {
            f,
            config,
            provider: Arc::new(CopyProvider {
                kind,
                world: Mutex::default(),
            }),
            kind,
            asset,
        }
    }
    fn open(&self) -> Arc<Service> {
        Arc::new(
            Service::open_with_transport(
                self.f.settings.clone(),
                Settings::from_slice(&self.config).unwrap(),
                self.f.sources.clone(),
                None,
                Default::default(),
                self.provider.clone(),
            )
            .unwrap(),
        )
    }
    fn sync_effects(&self) -> std::path::PathBuf {
        self.f
            .settings
            .registry_root
            .join("cloud-submissions/sync-effects")
    }
    fn sync_entry_paths(&self) -> Vec<std::path::PathBuf> {
        fs::read_dir(self.sync_effects())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| !path.file_name().unwrap().to_str().unwrap().starts_with('.'))
            .collect()
    }
    /// Every entry in the synchronous outbox, exactly as it is on disk.
    fn sync_entries(&self) -> Vec<Value> {
        self.sync_entry_paths()
            .into_iter()
            .map(|path| serde_json::from_slice(&fs::read(path).unwrap()).unwrap())
            .collect()
    }
    /// While this is held the outbox refuses `prepare`, which leaves the state a crash
    /// leaves between the durable copy request and `Outbox::prepare`.
    fn hold_sync_outbox(&self) -> FileLock {
        Directory::open(&self.sync_effects(), false)
            .unwrap()
            .lock(
                std::ffi::OsStr::new(".cloud-effects.lock"),
                LockOptions::default(),
            )
            .unwrap()
    }
    /// Put the one entry back to prepared, as if the service stopped before `started`.
    fn rewind_sync_entry_to_prepared(&self) {
        let paths = self.sync_entry_paths();
        let [path] = paths.as_slice() else {
            panic!("expected one entry");
        };
        let mut entry: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        entry["started"] = json!(false);
        entry["handle"] = Value::Null;
        entry["observation"] = Value::Null;
        fs::write(path, serde_json::to_vec(&entry).unwrap()).unwrap();
        let mut world = self.provider.world();
        world.rows.clear();
        world.calls.clear();
    }
    /// The dataset record, exactly as it is on disk.
    fn saved_dataset(&self) -> Value {
        let digest = canonical::digest(&json!(["demo", self.f.executor, self.asset])).unwrap();
        let path = self
            .f
            .settings
            .registry_root
            .join("cloud-submissions/slots")
            .join(format!("dataset-{digest}.json"));
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }
    fn superseded_copy(&self, request: &str) -> Option<Value> {
        let digest =
            canonical::digest(&json!(["demo", self.f.executor, self.asset, request])).unwrap();
        fs::read(
            self.f
                .settings
                .registry_root
                .join("cloud-submissions/slots")
                .join(format!("superseded-copy-{digest}.json")),
        )
        .ok()
        .map(|bytes| serde_json::from_slice(&bytes).unwrap())
    }
    fn assert_new_request_refused(&self, service: &Arc<Service>) {
        let creates = self.provider.creates();
        let entries = self.sync_entries().len();
        let error = self.publish(service, REQUEST_B).unwrap_err();
        assert_eq!(error.code, "DATA_COPY_IDENTITY_CONFLICT");
        assert_eq!(self.provider.creates(), creates);
        assert_eq!(self.sync_entries().len(), entries);
        assert!(self.superseded_copy(REQUEST_A).is_none());
    }
    fn publish(&self, service: &Arc<Service>, request: &str) -> cloud::Result<Value> {
        let runtime = match self.kind {
            Kind::Pai => Some(self.f.runtime.runtime_id.as_str()),
            Kind::Sensecore => None,
        };
        service.publish_dataset("demo", self.f.executor, &self.asset, request, runtime)
    }
}

#[test]
fn a_sync_job_driven_twice_creates_once() {
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        let service = c.open();
        let first = c.publish(&service, REQUEST_A).unwrap();
        assert_eq!(first["copy_job"]["phase"], "OWNED");
        assert_eq!(c.provider.creates(), 1);
        assert_eq!(
            c.publish(&service, REQUEST_A).unwrap()["copy_job"]["slot"],
            first["copy_job"]["slot"]
        );
        drop(service);
        let service = c.open();
        c.publish(&service, REQUEST_A).unwrap();
        assert_eq!(c.provider.creates(), 1);
    }
}

fn with_result_job(config: &[u8], limits: Value) -> Vec<u8> {
    let mut config: Value = serde_json::from_slice(config).unwrap();
    config["backends"][0]["result_job"] = limits;
    serde_json::to_vec(&config).unwrap()
}

#[test]
fn a_changed_result_job_limit_keeps_each_recorded_job_and_reaches_only_new_ones() {
    let mut c = CopyFixture::new(Kind::Pai, 1);
    c.config = with_result_job(&c.config, json!({"max_time_seconds":3600}));
    let service = c.open();
    let (action, _) = c.f.ready(&service, "train");
    assert_eq!(
        c.f.execute(&service, &action).unwrap()["status"],
        "VERIFIED"
    );
    assert_eq!(c.provider.creates(), 1);
    let sync = |service: &Service, request: &str| {
        service
            .synchronize(
                &action,
                &ml_exp_core::cloud::exchange::Sync {
                    request_id: request.into(),
                    confirmation: format!("SYNC {action}"),
                },
                T_RECONCILE,
                &AtomicBool::new(false),
            )
            .unwrap()["synchronization"]
            .clone()
    };
    let first = sync(&service, REQUEST_A);
    assert_eq!(first["max_time_seconds"], 3600);
    assert_eq!(first["resources"], json!({"gpus":0,"cpus":2,"memory_gb":4}));
    assert_eq!(c.provider.creates(), 2);

    // A daemon restarted with a higher limit still drives the job it recorded.
    c.config = with_result_job(
        &c.config,
        json!({"cpus":4,"memory_gb":8,"max_time_seconds":7200}),
    );
    drop(service);
    let service = c.open();
    assert_eq!(sync(&service, REQUEST_A)["max_time_seconds"], 3600);
    assert_eq!(
        service
            .sync_status(&action, REQUEST_A, T_RECONCILE)
            .unwrap()["synchronization"]["max_time_seconds"],
        3600
    );
    assert_eq!(c.provider.creates(), 2);

    // A new request and an explicit retry are new jobs, so they take the new limit.
    let second = sync(&service, REQUEST_B);
    assert_eq!(second["max_time_seconds"], 7200);
    assert_eq!(
        second["resources"],
        json!({"gpus":0,"cpus":4,"memory_gb":8})
    );
    assert_eq!(c.provider.creates(), 3);
    let job_id = |row: usize| c.provider.world().rows[row]["JobId"].clone();
    let failed = job_id(1).as_str().unwrap().to_owned();
    for row in [0, 1] {
        c.provider.world().rows[row]["Status"] = json!("Failed");
    }
    let retry = ml_exp_core::cloud::exchange::SyncRetry {
        expected_failed_job_id: failed.clone(),
        confirmation: format!("RETRY SYNC {action} {REQUEST_A} {failed}"),
    };
    let retried = |service: &Service| {
        service
            .retry_sync(
                &action,
                REQUEST_A,
                &retry,
                T_RECONCILE,
                &AtomicBool::new(false),
            )
            .unwrap()["synchronization"]
            .clone()
    };
    let child = retried(&service);
    assert_eq!(child["max_time_seconds"], 7200);
    assert_eq!(c.provider.creates(), 4);

    // Neither the failed original nor the retried child is orphaned by a later change.
    c.config = with_result_job(&c.config, json!({"max_time_seconds":10800}));
    drop(service);
    let service = c.open();
    assert_eq!(retried(&service)["max_time_seconds"], 7200);
    assert_eq!(sync(&service, REQUEST_A)["max_time_seconds"], 7200);
    assert_eq!(c.provider.creates(), 4);
    let recorded: Vec<u64> = c
        .sync_entries()
        .iter()
        .map(|entry| entry["job"]["max_time_seconds"].as_u64().unwrap())
        .collect();
    assert_eq!(
        recorded
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from([3600, 7200])
    );
    assert_eq!(recorded.len(), 3);
}

#[test]
fn a_maintenance_job_takes_the_configured_limit_and_keeps_it_once_recorded() {
    let mut c = CopyFixture::new(Kind::Pai, 1);
    c.config = with_result_job(&c.config, json!({"max_time_seconds":3600}));
    let service = c.open();
    let (action, _) = c.f.ready(&service, "train");
    c.f.execute(&service, &action).unwrap();
    c.provider.world().rows[0]["Status"] = json!("Failed");
    let plan = service
        .maintenance_plan(
            &action,
            ml_exp_core::maintenance::Selection::Outputs,
            REQUEST_A.into(),
            T_RECONCILE,
        )
        .unwrap();
    service.maintenance_execute(&action, &plan).unwrap();
    assert_eq!(c.provider.creates(), 2);
    c.config = with_result_job(&c.config, json!({"max_time_seconds":7200}));
    drop(service);
    let service = c.open();
    service.maintenance_execute(&action, &plan).unwrap();
    assert_eq!(c.provider.creates(), 2);
    let entries = c.sync_entries();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["job"]["max_time_seconds"], 3600);
    assert_eq!(entries[0]["job"]["gpus"], 0);
}

/// Each plan is one job of its own, so removing outputs leaves state and a generation
/// removable. Driving a plan again observes its job and creates nothing.
#[test]
fn every_maintenance_plan_of_an_attempt_runs_as_its_own_job_once() {
    for kind in KINDS {
        let mut c = CopyFixture::new(kind, 1);
        if let Profile::Sensecore(training) = &c.f.provider.profile {
            let mut collector = training.clone();
            collector.purpose = sensecore::Purpose::ResultCollection;
            collector.cluster = "debug-cpu".into();
            collector.worker_spec = "cpu.2c4g".into();
            collector.worker_gpus = 0;
            collector.worker_cpus = 2;
            collector.worker_memory_gb = 4;
            collector.quota = sensecore::Quota::Reserved;
            let mut config: Value = serde_json::from_slice(&c.config).unwrap();
            config["backends"][0]["collector"] = json!(Profile::Sensecore(collector));
            c.config = serde_json::to_vec(&config).unwrap();
        }
        let service = c.open();
        let (action, _) = c.f.ready(&service, "train");
        c.f.execute(&service, &action).unwrap();
        c.provider.set_state("FAILED");
        let plan = |selection, request: &str| {
            service
                .maintenance_plan(&action, selection, request.into(), T_RECONCILE)
                .unwrap()
        };
        let outputs = plan(ml_exp_core::maintenance::Selection::Outputs, REQUEST_A);
        let state = plan(ml_exp_core::maintenance::Selection::State, REQUEST_B);
        service.maintenance_execute(&action, &outputs).unwrap();
        assert_eq!(c.provider.creates(), 2);
        service.maintenance_execute(&action, &state).unwrap();
        assert_eq!(c.provider.creates(), 3);
        for repeated in [&outputs, &state] {
            service.maintenance_execute(&action, repeated).unwrap();
        }
        assert_eq!(c.provider.creates(), 3);
        let entries = c.sync_entries();
        assert_eq!(entries.len(), 2);
        assert_ne!(
            entries[0]["job"]["submission_token"],
            entries[1]["job"]["submission_token"]
        );
        for entry in &entries {
            assert_eq!(entry["job"]["project"], "demo");
            assert_eq!(entry["job"]["attempt_id"], "attempt-001");
            assert_eq!(entry["job"]["gpus"], 0);
            assert_ne!(entry["job"]["run_id"], "train");
        }
    }
}

#[test]
fn result_job_limits_default_and_refuse_what_the_collector_cannot_run() {
    let c = CopyFixture::new(Kind::Pai, 1);
    let load = |limits: Value| Settings::from_slice(&with_result_job(&c.config, limits));
    let default = Settings::from_slice(&c.config).unwrap();
    let limits = default.backends[0].result_job;
    assert_eq!(
        (limits.cpus, limits.memory_gb, limits.max_time_seconds),
        (2, 4, 21600)
    );
    let chosen = load(json!({"max_time_seconds":86400})).ok().unwrap();
    assert_eq!(chosen.backends[0].result_job.max_time_seconds, 86400);
    assert_eq!(chosen.backends[0].result_job.cpus, 2);
    for refused in [
        json!({"max_time_seconds":0}),
        json!({"max_time_seconds":86401}),
        json!({"cpus":0}),
        json!({"cpus":201}),
        json!({"memory_gb":1601}),
        json!({"unknown":1}),
    ] {
        assert!(load(refused.clone()).is_err(), "{refused}");
    }
    // SenseCore's collector has one fixed shape, and only the time limit can move.
    let f = Fixture::new(Kind::Sensecore);
    let Profile::Sensecore(training) = &f.provider.profile else {
        panic!("expected SenseCore");
    };
    let mut collector = training.clone();
    collector.purpose = sensecore::Purpose::ResultCollection;
    collector.cluster = "debug-cpu".into();
    collector.worker_spec = "cpu.2c4g".into();
    collector.worker_gpus = 0;
    collector.worker_cpus = 2;
    collector.worker_memory_gb = 4;
    collector.quota = sensecore::Quota::Reserved;
    let mut config: Value = serde_json::from_slice(&f.config).unwrap();
    config["backends"][0]["collector"] = json!(Profile::Sensecore(collector));
    let config = serde_json::to_vec(&config).unwrap();
    let load = |limits: Value| Settings::from_slice(&with_result_job(&config, limits));
    assert!(load(json!({"max_time_seconds":86400})).is_ok());
    assert!(load(json!({"cpus":4})).is_err());
    assert!(load(json!({"memory_gb":8})).is_err());
}

#[test]
fn a_started_sync_job_makes_no_create() {
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        c.provider.world().lose_create_reply = true;
        let service = c.open();
        let first = c.publish(&service, REQUEST_A).unwrap();
        assert_eq!(first["copy_job"]["phase"], "RECONCILE_REQUIRED");
        assert_eq!(first["copy_job"]["retry_safe"], false);
        assert_eq!(c.provider.creates(), 1);
        c.provider.world().calls.clear();
        // The provider would now accept a create, and still nothing is sent.
        c.provider.world().lose_create_reply = false;
        drop(service);
        let service = c.open();
        let again = c.publish(&service, REQUEST_A).unwrap();
        assert_eq!(again["copy_job"]["phase"], "RECONCILE_REQUIRED");
        assert_eq!(c.provider.creates(), 0);
    }
}

#[test]
fn a_long_file_inventory_reaches_the_provider_in_bounded_arguments() {
    for kind in KINDS {
        for files in [1_900, 5_000] {
            let c = CopyFixture::new(kind, files);
            let service = c.open();
            let view = c.publish(&service, REQUEST_A).unwrap();
            assert_eq!(view["copy_job"]["phase"], "OWNED");
            c.publish(&service, REQUEST_A).unwrap();
            assert_eq!(c.provider.creates(), 1);
            let entries = c.sync_entries();
            assert_eq!(entries.len(), 1);
            let command = entries[0]["job"]["command"].as_array().unwrap();
            let lengths = command.iter().map(|s| s.as_str().unwrap().len());
            assert!(lengths.clone().all(|length| length <= 64 * 1024));
            assert!(lengths.sum::<usize>() > 128 * 1024);
        }
    }
}

#[test]
fn an_unprepared_copy_refuses_a_new_request_and_can_still_be_driven() {
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        let service = c.open();
        {
            let _effects = c.hold_sync_outbox();
            let error = c.publish(&service, REQUEST_A).unwrap_err();
            assert_eq!(error.code, "CLOUD_EFFECT_BUSY");
        }
        assert!(c.sync_entries().is_empty());
        c.assert_new_request_refused(&service);
        assert_eq!(c.provider.creates(), 0);
        assert_eq!(
            c.publish(&service, REQUEST_A).unwrap()["copy_job"]["phase"],
            "OWNED"
        );
        assert_eq!(c.provider.creates(), 1);
    }
}

#[test]
fn a_prepared_copy_that_never_started_refuses_a_new_request() {
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        let service = c.open();
        c.publish(&service, REQUEST_A).unwrap();
        c.rewind_sync_entry_to_prepared();
        c.assert_new_request_refused(&service);
        assert_eq!(c.provider.creates(), 0);
        assert_eq!(c.sync_entries()[0]["started"], false);
    }
}

#[test]
fn a_started_copy_with_no_handle_refuses_a_new_request() {
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        c.provider.world().lose_create_reply = true;
        let service = c.open();
        let first = c.publish(&service, REQUEST_A).unwrap();
        assert_eq!(first["copy_job"]["phase"], "RECONCILE_REQUIRED");
        assert_eq!(first["new_request_allowed"], false);
        c.provider.world().lose_create_reply = false;
        c.provider.world().reads_down = true;
        c.assert_new_request_refused(&service);
        c.provider.world().reads_down = false;
        // The provider has no record of the job. That cannot fence a create in flight.
        c.assert_new_request_refused(&service);
        assert_eq!(c.provider.creates(), 1);
        assert!(c.sync_entries()[0]["handle"].is_null());
    }
}

#[test]
fn a_running_or_succeeded_copy_refuses_a_new_request() {
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        let service = c.open();
        c.publish(&service, REQUEST_A).unwrap();
        c.assert_new_request_refused(&service);
        // SUCCEEDED without READY contradicts the worker's exit rule.
        c.provider.set_state("SUCCEEDED");
        let view = c.publish(&service, REQUEST_A).unwrap();
        assert_eq!(view["new_request_allowed"], false);
        c.assert_new_request_refused(&service);
        assert_eq!(c.provider.creates(), 1);
    }
}

/// A Complete verifies with the lock released. It used to write back the record it
/// started from, which erased a copy job started meanwhile, and the next request then
/// reached a create with no proof that the first job had ended.
#[test]
fn a_late_archive_verification_keeps_the_copy_that_started_meanwhile() {
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        let service = c.open();
        let verifying: State = serde_json::from_value(c.saved_dataset()["state"].clone()).unwrap();
        let first = c.publish(&service, REQUEST_A).unwrap();
        assert_eq!(c.provider.creates(), 1);
        let before = c.saved_dataset();
        assert_eq!(before["copy_request_id"], REQUEST_A);

        let view = service
            .dataset_archive_verified("demo", c.f.executor, &c.asset, &verifying)
            .unwrap();
        assert_eq!(view["archive_verified"], true);
        assert_eq!(c.saved_dataset(), before);
        assert_eq!(view["copy_job"]["slot"], first["copy_job"]["slot"]);

        c.assert_new_request_refused(&service);
        assert_eq!(
            c.publish(&service, REQUEST_A).unwrap()["copy_job"]["slot"],
            first["copy_job"]["slot"]
        );
        assert_eq!(c.provider.creates(), 1);
    }
}

#[test]
fn a_late_archive_verification_of_another_state_changes_nothing() {
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        let service = c.open();
        let mut stale = c.saved_dataset()["state"].clone();
        stale["parts"] = json!({});
        let stale: State = serde_json::from_value(stale).unwrap();
        let before = c.saved_dataset();
        assert!(
            service
                .dataset_archive_verified("demo", c.f.executor, &c.asset, &stale)
                .is_err()
        );
        assert_eq!(c.saved_dataset(), before);
    }
}

/// SenseCore reports CANCELLED for these too, but neither shows the job is over.
#[test]
fn a_sensecore_copy_that_may_still_run_refuses_a_new_request() {
    for raw in ["DELETING", "SUSPENDED"] {
        let c = CopyFixture::new(Kind::Sensecore, 3);
        let service = c.open();
        c.publish(&service, REQUEST_A).unwrap();
        c.provider.set_state(raw);
        let view = c.publish(&service, REQUEST_A).unwrap();
        assert_eq!(view["new_request_allowed"], false, "{raw}");
        c.assert_new_request_refused(&service);
        assert_eq!(c.provider.creates(), 1, "{raw}");

        c.provider.set_state("DELETED");
        let view = c.publish(&service, REQUEST_A).unwrap();
        assert_eq!(view["new_request_allowed"], true, "{raw}");
        c.publish(&service, REQUEST_B).unwrap();
        assert_eq!(c.provider.creates(), 2, "{raw}");
    }
}

#[test]
fn a_failure_seen_earlier_needs_a_fresh_read_before_a_new_request() {
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        let service = c.open();
        c.publish(&service, REQUEST_A).unwrap();
        c.provider.set_state("FAILED");
        c.publish(&service, REQUEST_A).unwrap();
        c.provider.world().reads_down = true;
        c.assert_new_request_refused(&service);
        c.provider.world().reads_down = false;
        c.publish(&service, REQUEST_B).unwrap();
        assert_eq!(c.provider.creates(), 2);
    }
}

#[test]
fn a_proven_failure_permits_one_new_job_and_retires_the_old_request() {
    for kind in KINDS {
        for ended in ["FAILED", "CANCELLED"] {
            let c = CopyFixture::new(kind, 3);
            let service = c.open();
            c.publish(&service, REQUEST_A).unwrap();
            c.provider.set_state(ended);
            let failed = c.publish(&service, REQUEST_A).unwrap();
            assert_eq!(failed["new_request_allowed"], true);
            assert_eq!(c.provider.creates(), 1);
            let old_job = c.sync_entries()[0]["job"].clone();

            let replaced = c.publish(&service, REQUEST_B).unwrap();
            assert_eq!(c.provider.creates(), 2);
            assert_eq!(replaced["copy_job"]["phase"], "OWNED");
            assert_eq!(replaced["new_request_allowed"], false);
            assert_ne!(replaced["copy_job"]["slot"], failed["copy_job"]["slot"]);
            assert_eq!(c.sync_entries().len(), 2);
            assert_eq!(c.superseded_copy(REQUEST_A).unwrap()["job"], old_job);

            // Repeating either id creates nothing, also after a restart.
            c.publish(&service, REQUEST_B).unwrap();
            drop(service);
            let service = c.open();
            c.publish(&service, REQUEST_B).unwrap();
            let error = c.publish(&service, REQUEST_A).unwrap_err();
            assert_eq!(error.code, "DATA_COPY_REQUEST_SUPERSEDED");
            assert_eq!(c.provider.creates(), 2);

            // The new job is the current one, so it decides the next request.
            let error = c.publish(&service, REQUEST_C).unwrap_err();
            assert_eq!(error.code, "DATA_COPY_IDENTITY_CONFLICT");
            assert_eq!(c.provider.creates(), 2);
        }
    }
}

/// A preparation whose Run reads the fixture's dataset.
fn dataset_setup(c: &CopyFixture) -> ml_exp_core::preparation::Request {
    serde_json::from_value(
        json!({"request_id":"a".repeat(32),"run":{"run_id":"large-input",
        "runtime_id":c.f.runtime.runtime_id,"executor":c.f.executor,"command":["python","train.py"],
        "resources":{"gpus":1,"cpus":8,"memory_gb":64},"walltime_seconds":300,"seed":42,
        "data":[{"name":"train","asset_id":c.asset}]},"approval":{"max_gpu_hours":1.0}}),
    )
    .unwrap()
}

#[test]
fn dataset_manifests_over_their_budget_are_refused_before_any_copy() {
    for kind in KINDS {
        let small = CopyFixture::new(kind, 3);
        let view = small
            .open()
            .experiment_prepare("demo", dataset_setup(&small), None)
            .unwrap();
        assert_eq!(view["status"], "PREPARED");

        let c = CopyFixture::new(kind, 5000);
        let service = c.open();
        let setup = dataset_setup(&c);
        let id = setup.id("demo").unwrap();
        let refused = service
            .experiment_prepare("demo", setup.clone(), None)
            .unwrap_err();
        assert_eq!(refused.code, "DATASET_MANIFESTS_TOO_LARGE");
        assert_eq!(refused.http_status, Some(422));
        assert!(service.experiment_status("demo", &id).is_err());

        // Continuing checks again, because the dataset copy comes straight after.
        private(
            &c.f.settings
                .registry_root
                .join("cloud-submissions/slots")
                .join(format!("{id}.setup.json")),
            &json!({"project":"demo","request":setup,"worker_sha256":format!("sha256:{}", "0".repeat(64)),
                "started":false,"submission":null}),
        );
        let refused = service
            .experiment_continue(
                "demo",
                &id,
                &format!("PREPARE {id}"),
                None,
                open_gate(&c.f.transition()),
            )
            .err()
            .unwrap();
        assert_eq!(refused.code, "DATASET_MANIFESTS_TOO_LARGE");
        assert_eq!(c.provider.creates(), 0);
        assert!(!c.f.run_directory("large-input").exists());
    }
}

#[test]
fn a_dataset_no_run_could_mount_is_refused_at_begin_on_every_backend() {
    let begin = |files: usize| ml_exp_core::datasets::Command::Begin {
        definition: Definition {
            sha256: "9".repeat(64),
            archive_bytes: 1024 * 1024,
            files: (0..files)
                .map(|n| FileReceipt {
                    path: format!("dataset/tasks/sample-{n:05}.bin"),
                    bytes: 128,
                    sha256: "f".repeat(64),
                })
                .collect(),
        },
    };
    for kind in KINDS {
        let c = CopyFixture::new(kind, 3);
        let service = c.open();
        let refused = service
            .dataset_transfer("demo", c.f.executor, begin(5000), &[])
            .unwrap_err();
        assert_eq!(refused.code, "DATASET_MANIFESTS_TOO_LARGE");
        assert_eq!(refused.http_status, Some(422));
        assert!(!refused.uncertain);
        // A definition that fits goes on to this fixture's refusal: it has no dataset
        // storage configured.
        let other = service
            .dataset_transfer("demo", c.f.executor, begin(3), &[])
            .unwrap_err();
        assert_eq!(other.code, cloud::Error::configuration().code);
        assert_eq!(c.provider.creates(), 0);
    }
    // This Slurm fixture has no data helper, so the refusal cannot have used SSH.
    let f = Fixture::new(Kind::Pai);
    let (slurm, _, _) = slurm_side(&f);
    let transfer =
        |files| slurm.dataset_transfer("demo", "slurm", begin(files), &[], &AtomicBool::new(false));
    match transfer(5000) {
        Err(slurm_service::Error::Admission(cause)) => {
            assert_eq!(cause.0, ml_exp_core::datasets::ManifestsTooLarge::CAUSE);
        }
        other => panic!("expected the manifest refusal, got {other:?}"),
    }
    assert!(matches!(
        transfer(3),
        Err(slurm_service::Error::DatasetUnavailable)
    ));
}

#[test]
fn a_pai_run_over_the_manifest_budget_is_refused_before_it_exists() {
    let c = CopyFixture::new(Kind::Pai, 5000);
    let request = dataset_setup(&c)
        .resolve("demo", &format!("sha256:{}", "0".repeat(64)))
        .unwrap();
    let refused = c.open().create_run("demo", request, T_CREATE).unwrap_err();
    assert_eq!(refused.code, "DATASET_MANIFESTS_TOO_LARGE");
    assert_eq!(refused.http_status, Some(422));
    assert_eq!(c.provider.creates(), 0);
    assert!(!c.f.run_directory("large-input").exists());
}

#[test]
fn a_superseded_pai_request_stays_readable() {
    let c = CopyFixture::new(Kind::Pai, 3);
    let service = c.open();
    c.publish(&service, REQUEST_A).unwrap();
    c.provider.set_state("FAILED");
    c.publish(&service, REQUEST_B).unwrap();
    let diagnostic = |request| {
        service
            .data_diagnostic("demo", c.f.executor, &c.asset, request, false, T_RECONCILE)
            .unwrap_err()
            .code
    };
    assert_eq!(diagnostic(REQUEST_A), "CLOUD_SUBMISSION_NOT_FOUND");
    assert_eq!(diagnostic(REQUEST_C), "DATA_COPY_IDENTITY_CONFLICT");
}

#[test]
fn sensecore_data_diagnostics_are_typed_refusals_with_no_provider_call() {
    let c = CopyFixture::new(Kind::Sensecore, 3);
    let service = c.open();
    // Before any copy exists, and again once a copy request is recorded.
    for published in [false, true] {
        if published {
            c.publish(&service, REQUEST_A).unwrap();
        }
        let calls = c.provider.world().calls.len();
        for execute in [false, true] {
            let refused = service
                .data_diagnostic(
                    "demo",
                    c.f.executor,
                    &c.asset,
                    REQUEST_A,
                    execute,
                    T_RECONCILE,
                )
                .unwrap_err();
            assert_eq!(refused.code, "READONLY_DIAGNOSTIC_UNSUPPORTED");
        }
        let view = service
            .data_provider_diagnostics("demo", c.f.executor, &c.asset, REQUEST_A, 50)
            .unwrap();
        assert_eq!(view["diagnostic"], "PROVIDER_DIAGNOSTICS_UNSUPPORTED");
        assert_eq!(view["provider_read"], false);
        assert_eq!(c.provider.world().calls.len(), calls);
    }
}

#[test]
fn a_pai_copy_with_no_recorded_job_says_so_without_reading_the_provider() {
    let c = CopyFixture::new(Kind::Pai, 3);
    let service = c.open();
    c.provider.world().lose_create_reply = true;
    c.publish(&service, REQUEST_A).unwrap();
    c.provider.world().lose_create_reply = false;
    let calls = c.provider.world().calls.len();
    let view = service
        .data_provider_diagnostics("demo", c.f.executor, &c.asset, REQUEST_A, 50)
        .unwrap();
    assert_eq!(view["copy"]["diagnostic"], "NO_PROVIDER_JOB_RECORDED");
    assert_eq!(view["provider_read"], true);
    assert!(view.get("read_only_diagnostic").is_none());
    assert_eq!(c.provider.world().calls.len(), calls);
}

#[test]
fn pai_data_diagnostics_read_each_job_through_the_sync_outbox() {
    let c = CopyFixture::new(Kind::Pai, 3);
    let service = c.open();
    let read = || {
        service
            .data_provider_diagnostics("demo", c.f.executor, &c.asset, REQUEST_A, 50)
            .unwrap()
    };
    c.publish(&service, REQUEST_A).unwrap();
    let copy = read();
    assert_eq!(copy["copy"]["job_id"], "dlc-copy-0");
    assert_eq!(copy["provider_read"], true);
    assert!(copy.get("read_only_diagnostic").is_none());
    c.provider.set_state("FAILED");
    service
        .data_diagnostic("demo", c.f.executor, &c.asset, REQUEST_A, true, T_RECONCILE)
        .unwrap();
    let both = read();
    assert_eq!(both["copy"]["job_id"], "dlc-copy-0");
    assert_eq!(both["read_only_diagnostic"]["job_id"], "dlc-copy-1");
    assert_eq!(c.provider.creates(), 2);
}

#[test]
fn attempt_diagnostics_read_through_the_run_outbox_and_keep_their_keys() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        let (action, _) = f.ready(&service, "train");
        let unread = service
            .provider_diagnostics("demo", "train", "attempt-001", 50)
            .unwrap();
        assert_eq!(unread["provider_read"], false);
        assert_eq!(
            unread["diagnostic"],
            match kind {
                Kind::Pai => "NO_PROVIDER_JOB_RECORDED",
                Kind::Sensecore => "PROVIDER_DIAGNOSTICS_UNSUPPORTED",
            }
        );
        f.execute(&service, &action).unwrap();
        let read = service
            .provider_diagnostics("demo", "train", "attempt-001", 50)
            .unwrap();
        match kind {
            Kind::Pai => {
                assert_eq!(read["provider_read"], true);
                assert_eq!(read["provider"]["job_id"], PAI_JOB_ID);
            }
            Kind::Sensecore => {
                assert_eq!(read["provider_read"], false);
                assert_eq!(read["diagnostic"], "PROVIDER_DIAGNOSTICS_UNSUPPORTED");
            }
        }
        assert_eq!(f.provider.creates(), 1);
    }
}

/// The log lines the diagnostic job prints for a report.
fn diagnostic_lines(report: &Value) -> Vec<String> {
    let text = report.to_string();
    let digest = helper::sha256_bytes(text.as_bytes());
    let chunks: Vec<_> = text.as_bytes().chunks(1800).collect();
    chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| {
            format!(
                "ML_EXPD_DIAGNOSTIC_PART={}:{index}:{}:{}",
                &digest[7..],
                chunks.len(),
                std::str::from_utf8(chunk).unwrap()
            )
        })
        .collect()
}

#[test]
fn a_report_split_across_pods_stays_pending_until_it_is_whole_with_one_diagnostic_job() {
    let c = CopyFixture::new(Kind::Pai, 3);
    let service = c.open();
    c.publish(&service, REQUEST_A).unwrap();
    c.provider.set_state("FAILED");
    let expected = json!({"contract":"data-diagnostic.v1","project":"demo","asset_id":c.asset,
        "request_id":REQUEST_A,"filler":"x".repeat(5000)});
    let lines = diagnostic_lines(&expected);
    assert!(lines.len() > 1);
    let diagnose = || {
        service
            .data_diagnostic("demo", c.f.executor, &c.asset, REQUEST_A, true, T_RECONCILE)
            .unwrap()
    };
    c.provider
        .world()
        .pod_logs
        .insert("pod-a".into(), json!(["starting", lines[1]]));
    let pending = diagnose();
    assert_eq!(pending["report_status"], "AWAITING_OFFLINE_LOGS");
    assert!(pending["report"].is_null());
    assert_eq!(c.provider.creates(), 2);
    c.provider
        .world()
        .pod_logs
        .insert("pod-b".into(), json!([lines.join("\n")]));
    let whole = diagnose();
    assert_eq!(whole["report_status"], "AVAILABLE");
    assert_eq!(whole["report"], expected);
    assert_eq!(c.provider.creates(), 2, "observing never submits again");
}

#[test]
fn a_crash_after_retaining_the_old_job_leaves_its_request_current() {
    let c = CopyFixture::new(Kind::Pai, 3);
    let service = c.open();
    c.publish(&service, REQUEST_A).unwrap();
    c.provider.set_state("FAILED");
    // Compiling the new job fails after the old one is retained and before the switch.
    let absent = format!("runtime.{}", "9".repeat(64));
    assert!(
        service
            .publish_dataset(
                "demo",
                c.f.executor,
                &c.asset,
                REQUEST_B,
                Some(absent.as_str())
            )
            .is_err()
    );
    assert!(c.superseded_copy(REQUEST_A).is_some());
    c.publish(&service, REQUEST_A).unwrap();
    assert_eq!(c.provider.creates(), 1);
    c.publish(&service, REQUEST_B).unwrap();
    assert_eq!(c.provider.creates(), 2);
    let error = c.publish(&service, REQUEST_A).unwrap_err();
    assert_eq!(error.code, "DATA_COPY_REQUEST_SUPERSEDED");
}

const ROUTING_TOKEN: &str = "synthetic-routing-http-token-0000000000";

fn routing_api(
    f: &Fixture,
    slurm: Option<Arc<slurm_service::Service>>,
    cloud: Option<Arc<Service>>,
) -> Arc<Api> {
    let mut api = Api::open(f.settings.clone())
        .unwrap()
        .with_sources(f.sources.clone())
        .unwrap();
    if let Some(slurm) = slurm {
        api = api.with_submissions(slurm).unwrap();
    }
    if let Some(cloud) = cloud {
        api = api.with_cloud(cloud).unwrap();
    }
    Arc::new(api)
}
fn routing_app(f: &Fixture, api: Arc<Api>) -> axum::Router {
    let token = f.temp.path().join("routing-token");
    ml_exp_io::write_atomic(&token, ROUTING_TOKEN.as_bytes(), true).unwrap();
    ml_exp_server::app(ml_exp_server::Auth::load(&token).unwrap(), Some(api))
}
async fn routed(app: &axum::Router, method: &str, path: &str, body: &str) -> (u16, Value) {
    use tower::ServiceExt;
    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(method)
                .uri(path)
                .header("authorization", format!("Bearer {ROUTING_TOKEN}"))
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_owned()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// Cloud authority needs one discovered root, but a Slurm Run can live elsewhere.
/// Only the Run's owner may decide whether its layout matters.
#[tokio::test]
async fn a_configured_cloud_service_can_read_a_slurm_run_in_an_explicit_run_project() {
    for kind in KINDS {
        let mut f = Fixture::new(kind);
        f.settings.projects[0].runs = vec![read_config::Run {
            run_id: "train".into(),
            root_index: 0,
            relative_components: vec!["train".into()],
        }];
        let cloud_service = Arc::new(f.open());
        let (slurm, action, ssh_calls) = slurm_side(&f);
        let app = routing_app(
            &f,
            routing_api(&f, Some(Arc::new(slurm)), Some(cloud_service)),
        );
        for path in [
            "/api/runs/demo/train",
            "/api/runs/demo/train/submission",
            "/api/runs/demo/train/preparation",
            "/api/runs/demo/train/tracking",
            "/api/experiments/demo/train/submissions",
        ] {
            let (status, body) = routed(&app, "GET", path, "").await;
            assert_eq!(status, 200, "{path}: {body}");
        }
        let (_, listed) = routed(&app, "GET", "/api/experiments/demo/train/submissions", "").await;
        assert_eq!(listed["submissions"], json!([]));
        let (_, run) = routed(&app, "GET", "/api/runs/demo/train", "").await;
        assert_eq!(run["native_submission"]["submission_id"], action.as_str());
        let (_, submission) = routed(&app, "GET", "/api/runs/demo/train/submission", "").await;
        assert_eq!(submission["submission_id"], action.as_str());
        assert!(f.provider.calls().is_empty());
        assert!(!ssh_calls.exists());
    }
}

fn refusal(result: Result<Owner, error::Error>) -> (u16, &'static str) {
    let error = result.err().unwrap();
    (error.status, error.code)
}

#[tokio::test]
async fn a_run_or_action_reaches_its_own_backend_through_the_app() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let cloud_service = Arc::new(f.open());
        let (cloud_action, _) = f.ready(&cloud_service, "cloud-run");
        let (slurm, slurm_action, ssh_calls) = slurm_side(&f);
        let app = routing_app(
            &f,
            routing_api(&f, Some(Arc::new(slurm)), Some(cloud_service)),
        );
        for (run, action) in [("cloud-run", &cloud_action), ("train", &slurm_action)] {
            let (status, body) =
                routed(&app, "GET", &format!("/api/runs/demo/{run}/submission"), "").await;
            assert_eq!(status, 200, "{body}");
            assert_eq!(body["submission_id"], action.as_str());
            let (status, body) =
                routed(&app, "GET", &format!("/api/submissions/{action}"), "").await;
            assert_eq!(status, 200, "{body}");
            assert_eq!(body["submission_id"], action.as_str());
        }
        let (status, listed) = routed(
            &app,
            "GET",
            "/api/experiments/demo/cloud-run/submissions",
            "",
        )
        .await;
        assert_eq!(status, 200, "{listed}");
        assert_eq!(
            listed["submissions"][0]["submission_id"],
            cloud_action.as_str()
        );
        let ticket = |action: &str| format!("/api/submissions/{action}/download-ticket");
        let (status, body) = routed(&app, "POST", &ticket(&slurm_action), r#"{"part":0}"#).await;
        assert_eq!(
            (status, body["code"].as_str()),
            (422, Some("DIRECT_DOWNLOAD_UNAVAILABLE"))
        );
        let (_, body) = routed(&app, "POST", &ticket(&cloud_action), r#"{"part":0}"#).await;
        assert_ne!(body["code"], "DIRECT_DOWNLOAD_UNAVAILABLE");
        assert!(f.provider.calls().is_empty());
        assert!(!ssh_calls.exists());
    }
}

#[test]
fn an_owner_that_is_not_configured_answers_501_and_the_other_backend_is_never_asked() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let cloud_service = Arc::new(f.open());
        let (cloud_action, _) = f.ready(&cloud_service, "cloud-run");
        let (slurm, slurm_action, ssh_calls) = slurm_side(&f);

        let slurm_only = routing_api(&f, Some(Arc::new(slurm)), None);
        let absent = (501, "CLOUD_NOT_CONFIGURED");
        assert_eq!(refusal(slurm_only.action_owner(&cloud_action)), absent);
        assert_eq!(refusal(slurm_only.run_owner("demo", "cloud-run")), absent);
        assert!(matches!(
            slurm_only.action_owner(&slurm_action),
            Ok(Owner::Slurm(_))
        ));
        assert!(matches!(
            slurm_only.run_owner("demo", "train"),
            Ok(Owner::Slurm(_))
        ));

        let cloud_only = routing_api(&f, None, Some(cloud_service));
        let absent = (501, "SUBMISSION_NOT_CONFIGURED");
        assert_eq!(refusal(cloud_only.action_owner(&slurm_action)), absent);
        assert_eq!(refusal(cloud_only.run_owner("demo", "train")), absent);
        assert!(matches!(
            cloud_only.action_owner(&cloud_action),
            Ok(Owner::Cloud(_))
        ));
        assert!(matches!(
            cloud_only.run_owner("demo", "cloud-run"),
            Ok(Owner::Cloud(_))
        ));
        assert!(f.provider.calls().is_empty());
        assert!(!ssh_calls.exists());
    }
}

/// Routing reads the Run's own manifest. A scan of the Project would stop at the first
/// bad entry and refuse every other Run there, W&B publication included.
#[tokio::test]
async fn a_bad_sibling_cannot_stop_a_healthy_run_in_its_project() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let cloud_service = Arc::new(f.open());
        let (cloud_action, _) = f.ready(&cloud_service, "cloud-run");
        let (slurm, _, ssh_calls) = slurm_side(&f);
        let api = routing_api(&f, Some(Arc::new(slurm)), Some(cloud_service));
        let app = routing_app(&f, api.clone());
        let sibling = f.run_directory("broken-sibling");
        fs::create_dir_all(&sibling).unwrap();
        private(&sibling.join("manifest.json"), &json!({}));
        fs::write(sibling.join("manifest.json"), b"{").unwrap();

        // Finding a Run that is not directly below the root still needs the scan.
        assert_eq!(
            refusal(api.run_owner("demo", "absent")),
            (409, "EXPERIMENT_DISCOVERY_UNAVAILABLE")
        );
        assert!(matches!(
            api.run_owner("demo", "cloud-run"),
            Ok(Owner::Cloud(_))
        ));
        assert!(matches!(
            api.run_owner("demo", "train"),
            Ok(Owner::Slurm(_))
        ));
        // The Slurm service reads its Runs through discovery itself, so routing is all
        // this can promise for one. The cloud service never scans.
        let (status, body) = routed(&app, "GET", "/api/runs/demo/cloud-run/submission", "").await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["submission_id"], cloud_action.as_str());
        let (status, body) = routed(&app, "GET", "/api/runs/demo/cloud-run/preparation", "").await;
        assert_eq!(status, 200, "{body}");
        let (status, body) = routed(&app, "GET", "/api/runs/demo/cloud-run/tracking", "").await;
        assert_eq!(status, 200, "{body}");
        assert!(
            ml_exp_server::tracking::Source::tracking_journal(&*api, "demo", "cloud-run").is_ok()
        );
        assert!(f.provider.calls().is_empty());
        assert!(!ssh_calls.exists());
    }
}

/// A Run route that cannot read the manifest stops there. It used to be possible
/// to reach the next backend by treating such a failure as "not mine".
#[tokio::test]
async fn a_run_that_cannot_be_read_or_is_deleted_has_no_owner() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let cloud_service = Arc::new(f.open());
        f.ready(&cloud_service, "cloud-run");
        let (slurm, _, ssh_calls) = slurm_side(&f);
        let api = routing_api(&f, Some(Arc::new(slurm)), Some(cloud_service));
        let app = routing_app(&f, api.clone());
        assert_eq!(
            refusal(api.action_owner("action-1111111111111111")).0,
            404,
            "an unknown Action keeps the reading service's own answer"
        );
        // Nobody created this Run, so Slurm's own admission gets to refuse it.
        assert!(matches!(
            api.run_owner("demo", "absent"),
            Ok(Owner::Slurm(_))
        ));

        let manifest = f.run_directory("cloud-run").join("manifest.json");
        let original = fs::read(&manifest).unwrap();
        fs::write(&manifest, b"{").unwrap();
        let unreadable = (409, "EXPERIMENT_DISCOVERY_UNAVAILABLE");
        assert_eq!(refusal(api.run_owner("demo", "cloud-run")), unreadable);
        let (status, body) = routed(
            &app,
            "POST",
            "/api/runs/demo/cloud-run/submissions",
            r#"{"max_gpu_hours":1}"#,
        )
        .await;
        assert_eq!((status, body["code"].as_str()), (409, Some(unreadable.1)));
        fs::write(&manifest, &original).unwrap();
        assert!(matches!(
            api.run_owner("demo", "cloud-run"),
            Ok(Owner::Cloud(_))
        ));

        let fence = RunFence::new(&f.settings.registry_root);
        for run in ["cloud-run", "train"] {
            fence
                .mark("demo", run, &serde_json::Map::new(), T_EXECUTE)
                .unwrap();
            assert_eq!(
                refusal(api.run_owner("demo", run)),
                (409, "EXPERIMENT_EVIDENCE_UNAVAILABLE")
            );
        }
        assert!(f.provider.calls().is_empty());
        assert!(!ssh_calls.exists());
    }
}

/// The order the docs give: bytes first, then the Run. Retiring removes nothing and
/// says so, and it is the end of the line for new plans.
#[test]
fn a_run_retires_after_its_bytes_and_takes_no_later_plan() {
    let c = CopyFixture::new(Kind::Pai, 1);
    let service = c.open();
    let (action, _) = c.f.ready(&service, "train");
    c.f.execute(&service, &action).unwrap();
    c.provider.world().rows[0]["Status"] = json!("Failed");
    let plan = |request: &str| {
        service.maintenance_plan(
            &action,
            ml_exp_core::maintenance::Selection::Outputs,
            request.into(),
            T_RECONCILE,
        )
    };
    let outputs = plan(REQUEST_A).unwrap();
    service.maintenance_execute(&action, &outputs).unwrap();

    administration::Service::initialize(&c.f.settings).unwrap();
    let admin =
        administration::Service::open(c.f.settings.clone(), None, Some(service.clone())).unwrap();
    assert!(
        admin
            .retire("demo", "train", "RETIRE demo/train", T_RECONCILE)
            .is_err()
    );
    let fence = RunFence::new(&c.f.settings.registry_root);
    assert!(!fence.is_deleted("demo", "train").unwrap());
    let retired = admin
        .retire("demo", "train", "DELETE demo/train", T_RECONCILE)
        .unwrap();
    assert_eq!(retired["status"], "RETIRED");
    assert_eq!(retired["payloads_removed"], false);
    assert!(fence.is_deleted("demo", "train").unwrap());
    assert!(plan(REQUEST_B).is_err());
}

#[test]
fn administration_stops_when_it_cannot_tell_who_owns_an_action() {
    let f = Fixture::new(Kind::Pai);
    let cloud_service = Arc::new(f.open());
    let (slurm, _, ssh_calls) = slurm_side(&f);
    administration::Service::initialize(&f.settings).unwrap();
    let service = administration::Service::open(
        f.settings.clone(),
        Some(Arc::new(slurm)),
        Some(cloud_service),
    )
    .unwrap();
    let error = service
        .prepare(
            "action-1111111111111111",
            administration::Request {
                request_id: REQUEST_A.into(),
                selection: administration::Choice::Outputs,
            },
            T_EXECUTE,
        )
        .unwrap_err();
    assert_eq!(error.code, "ADMINISTRATION_BLOCKED");
    assert!(f.provider.calls().is_empty());
    assert!(!ssh_calls.exists());
}

#[test]
fn a_lost_reply_keeps_one_create_through_the_one_dispatch() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        f.provider.world().lose_create_reply = true;
        let service = Arc::new(f.open());
        let (action, _) = f.ready(&service, "train");
        let apply = |operation| {
            let transition = f.transition();
            runs::apply(
                Owner::Cloud(service.clone()),
                &action,
                operation,
                slurm_service::Context {
                    now: T_EXECUTE,
                    expires: GATE_EXPIRES,
                    transition_id: &transition,
                },
            )
        };
        let execute = || Operation::Execute {
            confirmation: format!("EXECUTE {action}"),
            reason: "owned fixture".into(),
        };
        let view = apply(execute()).unwrap();
        assert_eq!(view["status"], "RECONCILE_REQUIRED");
        assert_eq!(f.provider.creates(), 1);
        assert!(apply(execute()).is_err());
        assert_eq!(
            apply(Operation::Reconcile).unwrap()["status"],
            "RECONCILE_REQUIRED"
        );
        assert_eq!(f.provider.creates(), 1);
        f.provider.world().exists = true;
        assert_eq!(apply(Operation::Reconcile).unwrap()["status"], "VERIFIED");
        assert_eq!(f.provider.creates(), 1);
    }
}

#[test]
fn sync_and_direct_download_stay_unsupported_in_the_slurm_arm() {
    let f = Fixture::new(Kind::Pai);
    let (slurm, action, ssh_calls) = slurm_side(&f);
    let slurm = Arc::new(slurm);
    let transition = f.transition();
    let answer = |operation| {
        let error = runs::apply(
            Owner::Slurm(slurm.clone()),
            &action,
            operation,
            open_gate(&transition),
        )
        .unwrap_err();
        (error.status, error.code)
    };
    assert_eq!(
        answer(Operation::DownloadTicket(0)),
        (422, "DIRECT_DOWNLOAD_UNAVAILABLE")
    );
    assert_eq!(
        answer(Operation::Sync(ml_exp_core::cloud::exchange::Sync {
            request_id: REQUEST_A.into(),
            confirmation: format!("SYNC {action}"),
        })),
        (501, "CLOUD_SYNC_NOT_CONFIGURED")
    );
    assert!(!ssh_calls.exists());
}

/// A preparation for a Slurm Run: the cloud service keeps its state and Slurm
/// creates and prepares the Run. Nothing here may reach SSH or the provider.
fn slurm_preparation(runtime_id: &str, request_id: &str, run: &str) -> Value {
    json!({"request_id":request_id,"runtime":null,"approval":{"max_gpu_hours":1.0,"reason":"owned test"},
        "run":{"run_id":run,"runtime_id":runtime_id,"executor":"slurm-test",
            "command":["python","-u","/workspace/train.py"],"resources":{"gpus":1,"cpus":8,"memory_gb":16},
            "walltime_seconds":300,"seed":42,"parameters":{"lr":0.001},
            "metrics_schema":{"definitions":{"val_loss":{"unit":"nats/token","required":true}}},
            "evaluation":{"tokenizer":"fixture-v1","variant_id":"v1","D_FT":0}}})
}
fn preparation_request(value: Value) -> ml_exp_core::preparation::Request {
    serde_json::from_value(value).unwrap()
}
struct SlurmPreparation {
    cloud: Arc<Service>,
    slurm: Arc<slurm_service::Service>,
    runtime_id: String,
    ssh_calls: std::path::PathBuf,
}
fn slurm_preparation_side(f: &Fixture, creation: bool) -> SlurmPreparation {
    let cloud = Arc::new(f.open());
    let (target, runtime, profile) = slurm_parts(f);
    let runtime_id = runtime.runtime_id.clone();
    let (slurm, ssh_calls) = slurm_open(f, target, runtime, profile);
    SlurmPreparation {
        cloud,
        slurm: Arc::new(slurm.with_run_creation(creation).unwrap()),
        runtime_id,
        ssh_calls,
    }
}
const PREPARATIONS: &str = "/api/projects/demo/experiment-preparations";
async fn prepared_over_http(app: &axum::Router, request: &Value) -> (String, String, String) {
    let (status, prepared) = routed(app, "POST", PREPARATIONS, &request.to_string()).await;
    assert_eq!(status, 200, "{prepared}");
    assert_eq!(prepared["status"], "PREPARED");
    let id = prepared["preparation_id"].as_str().unwrap().to_owned();
    let path = format!("{PREPARATIONS}/{id}");
    let body = json!({"confirmation":format!("PREPARE {id}")}).to_string();
    (path.clone(), format!("{path}/continue"), body)
}

#[tokio::test]
async fn a_slurm_preparation_submits_through_slurm_and_repeats_from_saved_state() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let side = slurm_preparation_side(&f, true);
        let app = routing_app(
            &f,
            routing_api(&f, Some(side.slurm.clone()), Some(side.cloud.clone())),
        );
        let request = slurm_preparation(&side.runtime_id, REQUEST_A, "prepared");
        let id = preparation_request(request.clone()).id("demo").unwrap();
        let (show, advance, body) = prepared_over_http(&app, &request).await;

        let (status, ready) = routed(&app, "POST", &advance, &body).await;
        assert_eq!(status, 200, "{ready}");
        assert_eq!(ready["status"], "READY");
        assert_eq!(ready["phase"], "SUBMISSION_PREPARED");
        assert_eq!(ready["attempt_id"], "attempt-001");
        assert_eq!(ready["training_authorized"], false);
        assert_eq!(ready["scheduler_submitted"], false);
        assert_eq!(ready["submission"]["status"], "PREPARED");
        assert_eq!(ready["submission"]["reused"], false);
        assert!(f.run_directory("prepared").join("manifest.json").exists());

        // Asking Slurm again would answer `reused: true`, so an identical view
        // means this one came from the saved state alone.
        let (status, repeated) = routed(&app, "POST", &advance, &body).await;
        assert_eq!(status, 200, "{repeated}");
        assert_eq!(repeated, ready);

        // The saved state never moves to a different Action, and saying so changes nothing.
        let refused = side
            .cloud
            .experiment_finish("demo", &id, json!({"submission_id":"another-action"}))
            .unwrap_err();
        assert_eq!(refused.code, "CLOUD_EVIDENCE_INVALID");
        let (status, shown) = routed(&app, "GET", &show, "").await;
        assert_eq!(status, 200, "{shown}");
        assert_eq!(shown, ready);

        assert!(f.provider.calls().is_empty());
        assert!(!side.ssh_calls.exists());
    }
}

#[test]
fn a_slurm_preparation_that_lost_its_finish_reaches_the_same_action() {
    let f = Fixture::new(Kind::Pai);
    let side = slurm_preparation_side(&f, true);
    let request = preparation_request(slurm_preparation(&side.runtime_id, REQUEST_A, "prepared"));
    let id = request.id("demo").unwrap();
    let confirmation = format!("PREPARE {id}");
    side.cloud
        .experiment_prepare("demo", request, Some(&side.slurm.executors()))
        .unwrap();
    let advance = || {
        side.cloud.experiment_continue(
            "demo",
            &id,
            &confirmation,
            Some(&side.slurm.executors()),
            open_gate(&f.transition()),
        )
    };
    let mut actions = Vec::new();
    for reused in [false, true] {
        let Continue::SubmitElsewhere { run, approval } = advance().unwrap() else {
            panic!("a Slurm Run has no submission saved yet, so Slurm is asked again");
        };
        let run_id = run.run_id.clone();
        assert_eq!(
            side.slurm.create_run("demo", *run, T_CREATE).unwrap()["reused"],
            reused
        );
        let prepared = side
            .slurm
            .prepare("demo", &run_id, approval, open_gate(&f.transition()))
            .unwrap();
        assert_eq!(prepared["reused"], reused);
        actions.push(prepared);
    }
    assert_eq!(actions[0]["submission_id"], actions[1]["submission_id"]);
    let finished = side
        .cloud
        .experiment_finish("demo", &id, actions[1].clone())
        .unwrap();
    assert_eq!(finished["status"], "READY");
    let Continue::View(repeated) = advance().unwrap() else {
        panic!("a saved submission is answered from the saved state");
    };
    assert_eq!(repeated, finished);
    assert!(f.provider.calls().is_empty());
    assert!(!side.ssh_calls.exists());
}

#[tokio::test]
async fn a_slurm_refusal_during_a_preparation_reaches_the_client_with_its_own_code() {
    // Run creation is off for this Project, which only Slurm knows.
    let f = Fixture::new(Kind::Pai);
    let side = slurm_preparation_side(&f, false);
    let app = routing_app(
        &f,
        routing_api(&f, Some(side.slurm.clone()), Some(side.cloud.clone())),
    );
    let request = slurm_preparation(&side.runtime_id, REQUEST_A, "disabled");
    let (show, advance, body) = prepared_over_http(&app, &request).await;
    let (status, refused) = routed(&app, "POST", &advance, &body).await;
    assert_eq!(
        (status, refused["code"].as_str()),
        (403, Some("RUN_CREATION_DISABLED"))
    );
    let (_, shown) = routed(&app, "GET", &show, "").await;
    assert_eq!(shown["submission"], Value::Null);

    // The request is valid until it meets this backend's limits, so admission
    // refuses it here, the same way direct Run creation does.
    let f = Fixture::new(Kind::Pai);
    let side = slurm_preparation_side(&f, true);
    let app = routing_app(
        &f,
        routing_api(&f, Some(side.slurm.clone()), Some(side.cloud.clone())),
    );
    let mut request = slurm_preparation(&side.runtime_id, REQUEST_A, "too-large");
    request["run"]["resources"]["memory_gb"] = json!(513);
    let (_, advance, body) = prepared_over_http(&app, &request).await;
    let (status, refused) = routed(&app, "POST", &advance, &body).await;
    assert_eq!(
        (status, refused["code"].as_str()),
        (422, Some("RUN_CREATION_REJECTED"))
    );
    assert!(!f.run_directory("too-large").exists());
    assert!(f.provider.calls().is_empty());
    assert!(!side.ssh_calls.exists());
}

#[test]
fn a_preparation_for_an_executor_nobody_runs_is_refused_before_any_effect() {
    let f = Fixture::new(Kind::Pai);
    let side = slurm_preparation_side(&f, true);
    let request = preparation_request(slurm_preparation(&side.runtime_id, REQUEST_A, "orphan"));
    let id = request.id("demo").unwrap();
    assert!(
        side.cloud
            .experiment_prepare("demo", request.clone(), None)
            .is_err()
    );
    assert!(side.cloud.experiment_status("demo", &id).is_err());

    // Saved while Slurm was configured, then continued without it.
    side.cloud
        .experiment_prepare("demo", request, Some(&side.slurm.executors()))
        .unwrap();
    assert!(
        side.cloud
            .experiment_continue(
                "demo",
                &id,
                &format!("PREPARE {id}"),
                None,
                open_gate(&f.transition()),
            )
            .is_err()
    );
    assert!(!f.run_directory("orphan").exists());

    // Slurm accepts the executor but not its script.
    let mut scripted = slurm_preparation(&side.runtime_id, REQUEST_B, "scripted");
    scripted["run"]["data_preparation"] = json!({"script":"prepare.py"});
    let refused = side
        .cloud
        .experiment_prepare(
            "demo",
            preparation_request(scripted),
            Some(&side.slurm.executors()),
        )
        .unwrap_err();
    assert_eq!(refused.code, "SCRIPT_PREPARATION_NOT_CONFIGURED");
}

/// Answers for the one Runtime it built, and refuses any other.
struct Built(native::Runtime);
impl slurm_service::RuntimeBuilder for Built {
    fn slurm_runtime(
        &self,
        project: &str,
        executor: &str,
        id: &str,
    ) -> Result<native::Runtime, slurm_service::Error> {
        assert_eq!((project, executor), ("demo", "slurm-test"));
        if id == self.0.runtime_id {
            Ok(self.0.clone())
        } else {
            Err(slurm_service::Error::Authority)
        }
    }
}

#[test]
fn a_slurm_run_finds_an_unlisted_runtime_only_through_its_builder() {
    let f = Fixture::new(Kind::Pai);
    let (target, runtime, profile) = slurm_parts(&f);
    let built = native::Runtime {
        runtime_id: format!("runtime.{}", "4".repeat(64)),
        ..runtime.clone()
    };
    let request = preparation_request(slurm_preparation(&built.runtime_id, REQUEST_A, "built"))
        .resolve("demo", &format!("sha256:{}", "0".repeat(64)))
        .unwrap();
    let open = |builder: Option<Arc<dyn slurm_service::RuntimeBuilder>>| {
        slurm_open(&f, target.clone(), runtime.clone(), profile.clone())
            .0
            .with_runtime_builder(builder)
            .with_run_creation(true)
            .unwrap()
    };
    let authority = |result: Result<Value, slurm_service::Error>| {
        assert!(matches!(result, Err(slurm_service::Error::Authority)));
    };

    authority(open(None).create_run("demo", request.clone(), T_CREATE));
    let cloud: Arc<dyn slurm_service::RuntimeBuilder> = Arc::new(f.open());
    authority(open(Some(cloud)).create_run("demo", request.clone(), T_CREATE));
    assert!(!f.run_directory("built").exists());

    let builder: Arc<dyn slurm_service::RuntimeBuilder> = Arc::new(Built(built));
    let created = open(Some(builder))
        .create_run("demo", request, T_CREATE)
        .unwrap();
    assert_eq!(created["creation_state"], "CREATED");
}

/// The object store behaviour direct upload relies on, over a real socket so that real
/// reqwest parses each reply. Real OSS also checks signatures and expiry; this cannot.
#[derive(Default)]
struct FakeOss {
    objects: Mutex<BTreeMap<String, Vec<u8>>>,
    /// While set, a GET waits to be let through: `reached` says one has arrived.
    hold: Mutex<Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>>,
}
impl FakeOss {
    fn start(self: &Arc<Self>) -> String {
        use std::io::{BufRead, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let oss = self.clone();
        std::thread::spawn(move || {
            for socket in listener.incoming() {
                let oss = oss.clone();
                std::thread::spawn(move || {
                    let mut socket = socket.unwrap();
                    let mut reader = std::io::BufReader::new(socket.try_clone().unwrap());
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let mut words = line.split_whitespace();
                    let method = words.next().unwrap_or_default().to_owned();
                    let key = words
                        .next()
                        .unwrap_or_default()
                        .split('?')
                        .next()
                        .unwrap()
                        .to_owned();
                    let (mut length, mut forbid) = (0, false);
                    loop {
                        let mut header = String::new();
                        reader.read_line(&mut header).unwrap();
                        let lower = header.trim().to_lowercase();
                        if lower.is_empty() {
                            break;
                        }
                        if let Some(value) = lower.strip_prefix("content-length: ") {
                            length = value.parse().unwrap();
                        }
                        forbid |= lower == "x-oss-forbid-overwrite: true";
                    }
                    let mut body = vec![0; length];
                    reader.read_exact(&mut body).unwrap();
                    let held = if method == "GET" {
                        oss.hold.lock().unwrap().take()
                    } else {
                        None
                    };
                    if let Some((reached, release)) = held {
                        reached.send(()).unwrap();
                        release.recv().unwrap();
                    }
                    let mut objects = oss.objects.lock().unwrap();
                    let (status, reply, size) = match method.as_str() {
                        "PUT" if forbid && objects.contains_key(&key) => {
                            ("409 Conflict", vec![], 0)
                        }
                        "PUT" => {
                            objects.insert(key, body);
                            ("200 OK", vec![], 0)
                        }
                        "DELETE" => {
                            objects.remove(&key);
                            ("204 No Content", vec![], 0)
                        }
                        _ => match objects.get(&key) {
                            None => ("404 Not Found", vec![], 0),
                            Some(found) if method == "HEAD" => ("200 OK", vec![], found.len()),
                            Some(found) => ("200 OK", found.clone(), found.len()),
                        },
                    };
                    drop(objects);
                    let _ = write!(
                        socket,
                        "HTTP/1.1 {status}\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(&reply);
                });
            }
        });
        origin
    }
}
/// A service with an object store, one real tar and its Begin already done.
struct DirectUpload {
    f: Fixture,
    oss: Arc<FakeOss>,
    service: Service,
    archive: Vec<u8>,
    definition: Definition,
    sha: String,
}
impl DirectUpload {
    fn new(kind: Kind) -> Self {
        use sha2::Digest;
        let f = Fixture::new(kind);
        let credentials = f.temp.path().join("oss-credentials");
        private(
            &credentials,
            &json!({"access_key_id":"id","access_key_secret":"secret"}),
        );
        let mut config: Value = serde_json::from_slice(&f.config).unwrap();
        config["artifacts"] = json!({"endpoint":"oss-cn-example.aliyuncs.com","bucket":"bucket",
            "prefix":"users/alice/ml-expd/","credentials_file":credentials});
        let oss = Arc::new(FakeOss::default());
        let mut service = Service::open_with_transport(
            f.settings.clone(),
            Settings::from_slice(&serde_json::to_vec(&config).unwrap()).unwrap(),
            f.sources.clone(),
            None,
            Default::default(),
            f.provider.clone(),
        )
        .unwrap();
        service.redirect_objects_for_tests(oss.start());
        let content = vec![7u8; 1024];
        let mut tar = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o600);
        header.set_cksum();
        tar.append_data(&mut header, "train.bin", content.as_slice())
            .unwrap();
        let archive = tar.into_inner().unwrap();
        let sha = format!("{:x}", sha2::Sha256::digest(&archive));
        let definition = Definition {
            sha256: sha.clone(),
            archive_bytes: archive.len() as u64,
            files: vec![FileReceipt {
                path: "train.bin".into(),
                sha256: format!("{:x}", sha2::Sha256::digest(&content)),
                bytes: content.len() as u64,
            }],
        };
        let upload = Self {
            f,
            oss,
            service,
            archive,
            definition: definition.clone(),
            sha,
        };
        upload
            .transfer(ml_exp_core::datasets::Command::Begin { definition })
            .unwrap();
        upload
    }
    fn asset(&self) -> String {
        self.definition.asset_id()
    }
    fn transfer(&self, command: ml_exp_core::datasets::Command) -> cloud::Result<Value> {
        self.service
            .dataset_transfer("demo", self.f.executor, command, &[])
    }
    fn ticket(&self) -> Value {
        self.service
            .dataset_upload_ticket("demo", self.f.executor, &self.asset(), 0, &self.sha)
            .unwrap()
    }
    /// The way the client sends a part: the ticket's URL and header, no credential.
    fn put(&self, bytes: &[u8]) -> u16 {
        let ticket = self.ticket();
        reqwest::blocking::Client::new()
            .put(ticket["url"].as_str().unwrap())
            .header("x-oss-forbid-overwrite", "true")
            .body(bytes.to_vec())
            .send()
            .unwrap()
            .status()
            .as_u16()
    }
    fn uploaded(&self) -> cloud::Result<Value> {
        self.service
            .dataset_part_uploaded("demo", self.f.executor, &self.asset(), 0, &self.sha)
    }
    fn complete(&self) -> cloud::Result<Value> {
        self.transfer(ml_exp_core::datasets::Command::Complete {
            asset_id: self.asset(),
        })
    }
    fn view(&self) -> Value {
        self.service
            .dataset_state("demo", self.f.executor, &self.asset())
            .unwrap()
    }
}
#[test]
fn a_ticket_needs_the_exact_project_asset_and_part() {
    for kind in KINDS {
        let c = DirectUpload::new(kind);
        let executor = c.f.executor;
        let asset = c.asset();
        let ticket = c.ticket();
        let url = ticket["url"].as_str().unwrap();
        assert!(url.contains(&format!("/datasets/demo/{executor}/{}/parts/00000?", c.sha)));
        assert_eq!(ticket["bytes"], c.archive.len() as u64);
        let ask = |project: &str, executor: &str, asset: &str, part: u64, sha: &str| {
            c.service
                .dataset_upload_ticket(project, executor, asset, part, sha)
                .is_err()
        };
        assert!(ask("other", executor, &asset, 0, &c.sha), "another Project");
        assert!(ask("demo", "other", &asset, 0, &c.sha), "another executor");
        let never_begun = format!("asset.{}", "9".repeat(64));
        assert!(ask("demo", executor, &never_begun, 0, &c.sha), "no Begin");
        assert!(ask("demo", executor, &asset, 1, &c.sha), "beyond the asset");
        assert!(ask("demo", executor, &asset, 0, "not-a-hash"), "no hash");
        assert_eq!(c.put(&c.archive), 200);
        c.uploaded().unwrap();
        let other = "8".repeat(64);
        assert!(ask("demo", executor, &asset, 0, &other), "another claim");
        let record = c
            .service
            .dataset_part_uploaded("demo", executor, &asset, 0, &other);
        assert!(record.is_err(), "a record cannot change");
    }
}
#[test]
fn a_wrong_part_is_refused_at_complete_and_the_repeat_can_replace_it() {
    for kind in KINDS {
        let c = DirectUpload::new(kind);
        let mut wrong = c.archive.clone();
        *wrong.last_mut().unwrap() ^= 1;
        assert_eq!(c.put(&wrong), 200);
        // The size is right, so it is recorded. That accepts nothing.
        let recorded = c.uploaded().unwrap();
        assert_eq!(recorded["parts"].as_object().unwrap().len(), 1);
        assert_eq!(recorded["archive_verified"], false);
        let refused = c.complete().unwrap_err();
        assert_eq!(refused.code, "DATASET_PART_REJECTED");
        assert_eq!(
            error::Error::from(refused).code,
            "DATASET_PART_REJECTED",
            "the code reaches the client"
        );
        let view = c.view();
        assert_eq!(view["archive_verified"], false);
        assert!(view["ready"].is_null());
        assert_eq!(view["parts"].as_object().unwrap().len(), 0);
        assert!(c.oss.objects.lock().unwrap().is_empty());
        // Nothing that reads an asset can get a ticket for it.
        assert!(
            c.service
                .dataset_download("demo", c.f.executor, &c.asset(), 0)
                .is_err()
        );
        // The same upload again.
        assert_eq!(c.put(&c.archive), 200);
        c.uploaded().unwrap();
        assert_eq!(c.complete().unwrap()["archive_verified"], true);
        // Accepted now: no more tickets, and no later observation can remove the part.
        assert!(
            c.service
                .dataset_upload_ticket("demo", c.f.executor, &c.asset(), 0, &c.sha)
                .is_err()
        );
        assert_eq!(c.uploaded().unwrap()["archive_verified"], true);
        assert_eq!(c.oss.objects.lock().unwrap().len(), 1);
        assert!(
            c.service
                .dataset_download("demo", c.f.executor, &c.asset(), 0)
                .is_ok()
        );
    }
}
#[test]
fn a_part_is_not_recorded_unless_it_exists_at_its_exact_size() {
    for kind in KINDS {
        let c = DirectUpload::new(kind);
        assert_eq!(c.uploaded().unwrap_err().code, "DATASET_PART_NOT_STORED");
        assert_eq!(c.view()["parts"].as_object().unwrap().len(), 0);
        assert_eq!(c.put(&c.archive[..10]), 200);
        assert_eq!(c.uploaded().unwrap_err().code, "DATASET_PART_REJECTED");
        assert_eq!(c.view()["parts"].as_object().unwrap().len(), 0);
        assert!(c.oss.objects.lock().unwrap().is_empty());
        // A lost reply: the object is there, and the repeat is told so and still records.
        assert_eq!(c.put(&c.archive), 200);
        assert_eq!(c.put(&c.archive), 409);
        c.uploaded().unwrap();
        c.uploaded().unwrap();
        assert_eq!(c.complete().unwrap()["archive_verified"], true);
    }
}
#[test]
fn through_the_server_recovers_from_a_wrong_object_no_record_accepted() {
    for kind in KINDS {
        let c = DirectUpload::new(kind);
        let mut wrong = c.archive.clone();
        wrong[0] ^= 1;
        assert_eq!(c.put(&wrong), 200);
        let value = c
            .service
            .dataset_transfer(
                "demo",
                c.f.executor,
                ml_exp_core::datasets::Command::Part {
                    asset_id: c.asset(),
                    number: 0,
                    sha256: c.sha.clone(),
                    bytes: c.archive.len() as u64,
                },
                &c.archive,
            )
            .unwrap();
        assert_eq!(value["parts"].as_object().unwrap().len(), 1);
        assert_eq!(c.complete().unwrap()["archive_verified"], true);
    }
}
/// A large archive takes minutes to read, and everything else needs this lock.
#[test]
fn complete_does_not_hold_the_service_lock_while_it_reads() {
    let c = DirectUpload::new(Kind::Pai);
    assert_eq!(c.put(&c.archive), 200);
    c.uploaded().unwrap();
    let (reached, arrived) = std::sync::mpsc::channel();
    let (release, wait) = std::sync::mpsc::channel();
    *c.oss.hold.lock().unwrap() = Some((reached, wait));
    std::thread::scope(|scope| {
        let completing = scope.spawn(|| c.complete());
        arrived.recv().unwrap();
        // A Begin takes the lock. It would wait for its deadline if complete held it.
        let started = std::time::Instant::now();
        c.transfer(ml_exp_core::datasets::Command::Begin {
            definition: c.definition.clone(),
        })
        .unwrap();
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        release.send(()).unwrap();
        assert_eq!(
            completing.join().unwrap().unwrap()["archive_verified"],
            true
        );
    });
}

/// Through the real service to what a job would be created with, for a Run stored with
/// no `scheduling` and for one that froze a choice, on the first Attempt and on a retry.
#[test]
fn a_retry_asks_for_the_allocation_its_run_froze() {
    for (scheduling, expected) in [
        (None, "guaranteed"),
        (Some(json!({"allocation":"allow_idle"})), "allow_idle"),
    ] {
        let f = Fixture::new(Kind::Pai);
        let service = f.open();
        let mut request = json!({"run_id":"train","runtime_id":f.runtime.runtime_id,
            "executor":f.executor,"command":["python","train.py"],
            "resources":{"gpus":1,"cpus":8,"memory_gb":64},"walltime_seconds":300,"seed":42});
        if let Some(scheduling) = scheduling {
            request["scheduling"] = scheduling;
        }
        service
            .create_run("demo", serde_json::from_value(request).unwrap(), T_CREATE)
            .unwrap();
        let slot = |attempt: &str| -> Value {
            let name = native::attempt_slot("demo", "train", attempt).unwrap();
            let path = f
                .settings
                .registry_root
                .join("cloud-submissions/slots")
                .join(format!("{name}.json"));
            serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
        };
        let prepared = f.prepare(&service, "train");
        let action = prepared["submission_id"].as_str().unwrap().to_owned();
        let first = slot("attempt-001");
        assert_eq!(first["execution"]["allocation"], expected);
        let execution: Execution = serde_json::from_value(first["execution"].clone()).unwrap();
        f.provider.bind(&execution.job);
        f.execute(&service, &action).unwrap();
        f.provider.world().terminal = true;
        let retried = service
            .retry(
                &action,
                retry_request("train"),
                T_RECONCILE,
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(retried["attempt_id"], "attempt-002");
        f.prepare(&service, "train");
        assert_eq!(slot("attempt-002")["execution"]["allocation"], expected);
    }
}
#[test]
fn a_preparation_shows_the_allocation_it_will_ask_for_and_sensecore_has_no_choice() {
    let c = CopyFixture::new(Kind::Pai, 3);
    let view = c
        .open()
        .experiment_prepare("demo", dataset_setup(&c), None)
        .unwrap();
    assert_eq!(
        view["frozen"]["allocation"], "guaranteed",
        "saved before there was a choice"
    );

    let c = CopyFixture::new(Kind::Pai, 3);
    let mut chosen = dataset_setup(&c);
    chosen
        .run
        .insert("scheduling".into(), json!({"allocation":"allow_idle"}));
    let view = c.open().experiment_prepare("demo", chosen, None).unwrap();
    assert_eq!(view["frozen"]["allocation"], "allow_idle");

    let c = CopyFixture::new(Kind::Sensecore, 3);
    let view = c
        .open()
        .experiment_prepare("demo", dataset_setup(&c), None)
        .unwrap();
    assert!(view["frozen"].get("allocation").is_none());
    let c = CopyFixture::new(Kind::Sensecore, 3);
    let mut chosen = dataset_setup(&c);
    chosen
        .run
        .insert("scheduling".into(), json!({"allocation":"allow_idle"}));
    let refused = c
        .open()
        .experiment_prepare("demo", chosen, None)
        .unwrap_err();
    assert_eq!(refused.code, "INVALID_CLOUD_REQUEST");
    assert_eq!(c.provider.creates(), 0);
}
