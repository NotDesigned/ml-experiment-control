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
    backends::{
        cloud::{
            self,
            adapter::{Adapter, Profile, Transport},
            outbox::Outbox,
        },
        slurm::{Budget, SshConfig},
    },
    cloud_runs::{Service, Settings},
    operations::{ClaimContext, Store},
    read_config, sources, submissions,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Cursor,
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
const T_AUTHORIZE: &str = "2026-10-29T00:02:00Z";
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
    /// A Run with an authorized first Action, ready to spend. The provider row
    /// is bound to this Run's job, though no provider request has been sent.
    fn authorized(&self, service: &Service, run: &str) -> (String, Execution) {
        let request: CreateRun = serde_json::from_value(json!({"run_id":run,
            "runtime_id":self.runtime.runtime_id,"executor":self.executor,"command":["python","train.py"],
            "resources":{"gpus":1,"cpus":8,"memory_gb":64},"walltime_seconds":300,"seed":42}))
        .unwrap();
        service.create_run("demo", request, T_CREATE).unwrap();
        let prepared = self.prepare(service, run);
        assert_eq!(prepared["status"], "PREPARED");
        let action = prepared["submission_id"].as_str().unwrap().to_owned();
        let authorized = service
            .authorize(&action, "owned fixture", T_AUTHORIZE, &self.transition())
            .unwrap();
        assert_eq!(authorized["status"], "AUTHORIZED");
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
            T_EXECUTE,
            &self.transition(),
            &AtomicBool::new(false),
        )
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
fn expired_gate(transition_id: &str) -> submissions::Context<'_> {
    submissions::Context {
        now: T_GATE_EXPIRED,
        expires: RENEWED_GATE_EXPIRES,
        transition_id,
    }
}
fn open_gate(transition_id: &str) -> submissions::Context<'_> {
    submissions::Context {
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
        let (action, _) = f.authorized(&service, "train");
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
        let (action, _) = f.authorized(&service, "train");
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
        let (action, _) = f.authorized(&service, "train");
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
        let (action, _) = f.authorized(&service, "train");
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
            // Neither preparing again nor authorizing again opens a way back to create.
            let prepared = f.prepare(&service, "train");
            assert_eq!(prepared["reused"], true);
            assert_eq!(prepared["submission_id"], action.as_str());
            assert_eq!(prepared["status"], "RECONCILE_REQUIRED");
            assert!(
                service
                    .authorize(&action, "again", T_AUTHORIZE, &f.transition())
                    .is_err()
            );
            assert_eq!(f.provider.creates(), 1);
        }
    }
}

#[test]
fn a_claim_that_died_before_its_durable_start_never_creates() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        let (action, execution) = f.authorized(&service, "train");
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
        let (action, execution) = f.authorized(&service, "train");
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
        let (never, _) = f.authorized(&service, "never");
        assert_eq!(
            service
                .retry(&never, retry_request("never"), T_RECONCILE, &flag)
                .unwrap_err()
                .code,
            "CLOUD_EVIDENCE_INVALID"
        );
        assert!(f.provider.calls().is_empty());

        f.provider.world().lose_create_reply = true;
        let (action, _) = f.authorized(&service, "train");
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

#[test]
fn renew_is_refused_once_started() {
    for kind in KINDS {
        let f = Fixture::new(kind);
        let service = f.open();
        // The control: an expired gate that never started does renew.
        let (control, _) = f.authorized(&service, "control");
        let renewed = service
            .renew(&control, renewal(&control), expired_gate(&f.transition()))
            .unwrap();
        assert_eq!(renewed["status"], "PREPARED");
        assert_eq!(renewed["renewed_from"], control.as_str());
        assert_ne!(renewed["submission_id"], control.as_str());
        assert!(f.provider.calls().is_empty());

        f.provider.world().lose_create_reply = true;
        let (uncertain, _) = f.authorized(&service, "uncertain");
        assert_eq!(
            f.execute(&service, &uncertain).unwrap()["status"],
            "RECONCILE_REQUIRED"
        );
        f.provider.world().lose_create_reply = false;
        let (verified, _) = f.authorized(&service, "verified");
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

/// The Slurm service over the same registry: one real authorized Action, and an
/// SSH program that only records that it was called. Copied from the fixture in
/// `backends/slurm/tests/submission_tests.rs`, which is private to its crate.
fn slurm_side(f: &Fixture) -> (submissions::Service, String, std::path::PathBuf) {
    let root = f.temp.path();
    submissions::Service::initialize(&f.settings).unwrap();
    let run = f.run_directory("train");
    fs::create_dir_all(run.join("attempts/attempt-001")).unwrap();
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
            additional_roots: BTreeMap::new(),
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
    for name in ["ssh-config", "known-hosts"] {
        ml_exp_io::write_atomic(&root.join(name), b"synthetic test fixture\n", true).unwrap();
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
    let service = submissions::Service::open(
        f.settings.clone(),
        f.sources.clone(),
        None,
        vec![submissions::Backend {
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
    let action = prepared["submission_id"].as_str().unwrap().to_owned();
    let authorized = service
        .authorize(&action, "tester", "owned", open_gate(&f.transition()))
        .unwrap();
    assert_eq!(authorized["status"], "AUTHORIZED");
    (service, action, calls)
}
fn stops_at_slurm_load(result: Result<Value, submissions::Error>) {
    match result {
        Err(submissions::Error::Authority) => {}
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
        let (cloud_action, _) = f.authorized(&cloud_service, "cloud-run");
        let (slurm, slurm_action, ssh_calls) = slurm_side(&f);
        let flag = AtomicBool::new(false);
        assert!(cloud_service.handles_action(&cloud_action).unwrap());
        assert!(!cloud_service.handles_action(&slurm_action).unwrap());

        let a = cloud_action.as_str();
        stops_at_slurm_load(slurm.get(a, T_EXECUTE));
        stops_at_slurm_load(slurm.authorize(a, "tester", "note", open_gate(&f.transition())));
        stops_at_slurm_load(slurm.execute(
            a,
            &format!("EXECUTE {a}"),
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
            "AUTHORIZED"
        );

        let a = slurm_action.as_str();
        stops_at_cloud_load(cloud_service.get(a, T_EXECUTE));
        stops_at_cloud_load(cloud_service.authorize(a, "note", T_AUTHORIZE, &f.transition()));
        stops_at_cloud_load(f.execute(&cloud_service, a));
        stops_at_cloud_load(cloud_service.reconcile(a, T_EXECUTE, &f.transition(), &flag));
        stops_at_cloud_load(cloud_service.refresh(a, T_EXECUTE, &flag));
        stops_at_cloud_load(cloud_service.cancel(a, &format!("CANCEL {a}"), T_EXECUTE, &flag));
        stops_at_cloud_load(cloud_service.retry(a, retry_request("train"), T_EXECUTE, &flag));
        stops_at_cloud_load(cloud_service.renew(a, renewal(a), expired_gate(&f.transition())));
        assert_eq!(slurm.get(a, T_EXECUTE).unwrap()["status"], "AUTHORIZED");

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
                        "DisplayName":body["DisplayName"],"Settings":{"Tags":body["Settings"]["Tags"]},"Status":"Running"}),
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
            .unwrap_err();
        assert_eq!(refused.code, "DATASET_MANIFESTS_TOO_LARGE");
        assert_eq!(c.provider.creates(), 0);
        assert!(!c.f.run_directory("large-input").exists());
    }
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
