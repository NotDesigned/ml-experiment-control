//! Real immutable sources, canonical Run/Attempt reads, SQLite Actions and
//! native helper processes. Scheduler binaries and SSH are owned fixtures.
use super::*;
use crate::{
    read_config, sources,
    submissions::{Backend, Context, Service},
};
use ml_exp_core::{
    first_submission::native::{self, Approval, Prepared, Profile, Runtime},
    operations::{FrozenFiles, Policy},
    slurm::{ExecutionConfig, JobSpec, Resources, SlurmConfig, Storage},
};
use serde_json::{Value, json};

const NOW: &str = "2026-10-28T00:00:00Z";
fn context(n: u8) -> Context<'static> {
    Context {
        now: NOW,
        expires: "2026-10-28T01:00:00Z",
        transition_id: match n {
            1 => "00000000000000000000000000000001",
            2 => "00000000000000000000000000000002",
            _ => "00000000000000000000000000000003",
        },
    }
}
fn write(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
struct Fixture {
    root: tempfile::TempDir,
    settings: read_config::Settings,
    source: Arc<sources::Service>,
    runtime: Runtime,
    profile: Profile,
}
impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        fs::create_dir_all(project.join("runs/train/attempts/attempt-001")).unwrap();
        write(
            &project.join("project.json"),
            &json!({"schema_version":1,"project":"demo","title":"Native submission fixture","run_roots":["runs"]}),
        );
        let registry = root.path().join("registry");
        read_config::initialize(&registry).unwrap();
        let settings=read_config::Settings::from_json(&serde_json::to_vec(&json!({"schema_version":1,"registry_root":registry,
            "projects":[{"project":"demo","project_file":project.join("project.json"),"run_roots":[project.join("runs")]}]})).unwrap()).unwrap();
        sources::Service::initialize(&settings).unwrap();
        let source = Arc::new(sources::Service::open(&settings, None).unwrap());
        let tree = root.path().join("code");
        fs::create_dir(&tree).unwrap();
        fs::write(
            tree.join("train.py"),
            "raise RuntimeError('source must not execute during admission')\n",
        )
        .unwrap();
        let mut archive = ml_exp_client::sources::pack_directory(&tree).unwrap();
        let receipt = source
            .publish("demo", &mut archive.file, &archive.sha256, NOW)
            .unwrap();
        let source_id = receipt["source_id"].as_str().unwrap().to_owned();
        let image = format!("sha256:{}", "1".repeat(64));
        let mount = "/datapool/ml-expd-synthetic";
        let runtime = Runtime {
            runtime_id: format!("runtime.{}", "2".repeat(64)),
            source_id: source_id.clone(),
            image_id: image.clone(),
            sif_path: format!("{mount}/images/train.sif"),
            sif_sha256: helper::sha256_bytes(b"synthetic materialized SIF"),
            oci_image: format!("registry.invalid/demo@{image}"),
        };
        let profile = Profile {
            target: target(),
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
            },
            command: vec!["python".into(), "-u".into(), "/workspace/train.py".into()],
        };
        let manifest = json!({"schema_version":1,"identity_version":2,"project":"demo","run_id":"train","source_id":source_id,"image_id":image,"config_path":"config.json","seed":42,"campaign":"fixture",
            "backend":spec.backend,"storage":spec.storage,"resources":spec.resources,"execution":spec.execution,"command":spec.command,"container":{"runtime_id":runtime.runtime_id},"evaluation":{"protocol_id":"fixture"},"parameters":{},"inputs":[]});
        write(&project.join("runs/train/manifest.yaml"), &manifest);
        let mut attempt = manifest;
        attempt["attempt_id"] = json!("attempt-001");
        write(
            &project.join("runs/train/attempts/attempt-001/attempt.yaml"),
            &attempt,
        );
        let status = json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","state":"NOT_SUBMITTED"});
        write(&project.join("runs/train/status.json"), &status);
        write(
            &project.join("runs/train/attempts/attempt-001/status.json"),
            &status,
        );
        Service::initialize(&settings).unwrap();
        Self {
            root,
            settings,
            source,
            runtime,
            profile,
        }
    }
    fn service(&self, mode: &str, enabled: bool) -> Service {
        let ssh = config(self.root.path(), mode);
        for name in ["ssh-config", "known-hosts"] {
            fs::set_permissions(
                self.root.path().join(name),
                fs::Permissions::from_mode(0o600),
            )
            .unwrap();
        }
        Service::open(
            self.settings.clone(),
            self.source.clone(),
            None,
            vec![Backend {
                profile: self.profile.clone(),
                ssh,
                runtimes: BTreeMap::from([(self.runtime.runtime_id.clone(), self.runtime.clone())]),
            }],
            enabled,
        )
        .unwrap()
    }
    fn prepared(&self) -> Prepared {
        serde_json::from_slice(
            &fs::read(
                self.settings
                    .registry_root
                    .join("submissions/slots")
                    .join(format!("{}.json", native::slot("demo", "train").unwrap())),
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn slot(&self) -> PathBuf {
        self.settings
            .registry_root
            .join("submissions/slots")
            .join(format!("{}.json", native::slot("demo", "train").unwrap()))
    }
    fn tools(&self, prepared: &Prepared) {
        for (name, body) in [
            (
                "sbatch",
                "cat >/dev/null; printf 'call\\n' >> \"$COUNTER\"; printf '24827\\n'",
            ),
            ("sacct", "cat \"$EVIDENCE\""),
            ("squeue", "exit 0"),
            ("scancel", "exit 99"),
        ] {
            let path = self.root.path().join(name);
            fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        }
        fs::write(
            self.root.path().join("evidence"),
            format!(
                "24827|train--attempt-001|{}|h100|RUNNING|0:01|0:0",
                ml_exp_core::identity::submission_marker(
                    &prepared.request.intent().submission_token
                )
                .unwrap()
            ),
        )
        .unwrap();
        ml_exp_worker::slurm::Helper::initialize(ledger_config(self.root.path())).unwrap();
    }
}
fn approval() -> Approval {
    Approval {
        max_gpu_hours: 1.0,
        reason: "owned test".into(),
    }
}

#[test]
fn first_preparation_reuses_original_private_intent_across_service_restart() {
    let f = Fixture::new();
    let service = f.service("ledger", true);
    let prepared = service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    assert_eq!(prepared["status"], "PREPARED");
    assert_eq!(prepared["attempt_id"], "attempt-001");
    let private = f.prepared();
    let (_, script) = private.request.prepared_content().unwrap();
    assert!(script.contains("sha256sum --check --status"));
    assert!(script.contains(&f.runtime.sif_sha256[7..]));
    let bytes = fs::read(f.slot()).unwrap();
    assert!(
        !prepared
            .to_string()
            .contains(&private.request.intent().submission_token)
    );
    assert_eq!(
        fs::metadata(f.slot()).unwrap().permissions().mode() & 0o777,
        0o600
    );
    drop(service);
    let reopened = f.service("ledger", true);
    assert_eq!(
        reopened
            .prepare("demo", "train", approval(), context(2))
            .unwrap()["submission_id"],
        prepared["submission_id"]
    );
    assert_eq!(fs::read(f.slot()).unwrap(), bytes);
    assert!(
        reopened
            .prepare(
                "demo",
                "train",
                Approval {
                    max_gpu_hours: 2.,
                    ..approval()
                },
                context(3)
            )
            .is_err()
    );
    assert!(!f.root.path().join("calls").exists());
}

#[test]
fn service_claims_only_reviewed_frozen_run_and_native_helper_submits_once() {
    let f = Fixture::new();
    let service = f.service("ledger", true);
    let p = service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    let id = p["submission_id"].as_str().unwrap();
    f.tools(&f.prepared());
    assert!(
        service
            .execute(
                id,
                &format!("EXECUTE {id}"),
                context(2),
                &AtomicBool::new(false)
            )
            .is_err()
    );
    service
        .authorize(id, "actor:fixture", "reviewed", context(2))
        .unwrap();
    assert!(
        service
            .execute(id, "wrong", context(3), &AtomicBool::new(false))
            .is_err()
    );
    let done = service
        .execute(
            id,
            &format!("EXECUTE {id}"),
            context(3),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(done["status"], "VERIFIED");
    service
        .execute(
            id,
            &format!("EXECUTE {id}"),
            context(3),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(
        fs::read_to_string(f.root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
    assert_eq!(
        service
            .prepare("demo", "train", approval(), context(3))
            .unwrap()["status"],
        "VERIFIED"
    );
}

#[test]
fn lost_response_reopens_same_slot_and_reconciles_without_second_sbatch() {
    let f = Fixture::new();
    let service = f.service("ledger-lost-submit", true);
    let p = service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    let id = p["submission_id"].as_str().unwrap();
    f.tools(&f.prepared());
    service
        .authorize(id, "actor:fixture", "reviewed", context(2))
        .unwrap();
    let result = service
        .execute(
            id,
            &format!("EXECUTE {id}"),
            context(3),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(result["status"], "RECONCILE_REQUIRED");
    assert_eq!(result["execution"]["safe_to_retry"], false);
    drop(service);
    let reopened = f.service("ledger", true);
    assert_eq!(
        reopened
            .prepare("demo", "train", approval(), context(3))
            .unwrap()["submission_id"],
        id
    );
    let done = reopened
        .reconcile(id, NOW, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(done["status"], "VERIFIED");
    assert_eq!(
        fs::read_to_string(f.root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
}

#[test]
fn missing_slot_after_an_action_exists_never_generates_a_replacement_intent() {
    for remove_action_directory in [false, true] {
        let f = Fixture::new();
        let service = f.service("ledger", true);
        let prepared = service
            .prepare("demo", "train", approval(), context(1))
            .unwrap();
        fs::remove_file(f.slot()).unwrap();
        if remove_action_directory {
            fs::remove_dir_all(
                f.settings
                    .registry_root
                    .join("actions")
                    .join(prepared["submission_id"].as_str().unwrap()),
            )
            .unwrap();
        }
        drop(service);
        let reopened = f.service("ledger", true);
        assert!(
            reopened
                .prepare("demo", "train", approval(), context(2))
                .is_err()
        );
        assert!(!f.slot().exists());
        assert!(!f.root.path().join("calls").exists());
    }
}

#[test]
fn unapproved_resources_runtime_inputs_and_attempt_mismatches_fail_before_preparation() {
    for change in [
        "budget",
        "nan",
        "cpus",
        "gpus",
        "memory",
        "walltime",
        "account",
        "runtime",
        "image",
        "source",
        "input",
        "resume",
        "run_directory",
        "cache_directory",
        "attempt_resources",
        "attempt_running",
        "legacy",
    ] {
        let f = Fixture::new();
        let service = f.service("ledger", true);
        let mut request = approval();
        if change == "budget" {
            request.max_gpu_hours = 0.001;
        }
        if change == "nan" {
            request.max_gpu_hours = f64::NAN;
        }
        for filename in ["manifest.yaml", "attempts/attempt-001/attempt.yaml"] {
            let path = f.root.path().join("project/runs/train").join(filename);
            let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            match change {
                "cpus" => value["resources"]["cpus"] = json!(65),
                "gpus" => {
                    value["resources"]["gpus"] = json!(9);
                    value["backend"]["gres"] = json!("gpu:h100:9");
                }
                "memory" => value["resources"]["memory_gb"] = json!(513),
                "walltime" => value["backend"]["time"] = json!("25:00:00"),
                "account" => value["backend"]["account"] = json!("other-account"),
                "runtime" => {
                    value["container"]["runtime_id"] = json!(format!("runtime.{}", "3".repeat(64)))
                }
                "image" => value["image_id"] = json!(format!("sha256:{}", "4".repeat(64))),
                "source" => value["source_id"] = json!(format!("source.{}", "5".repeat(64))),
                "input" => value["inputs"] = json!([{"asset_id":"unmaterialized"}]),
                "resume" => value["resume_from"] = json!({"checkpoint_id":"unverified"}),
                "run_directory" => {
                    value["storage"]["run_dir"] =
                        json!("/datapool/ml-expd-synthetic/runs/demo/another-run")
                }
                "cache_directory" => {
                    value["backend"]["apptainer_cache_dir"] = json!("/tmp/foreign-cache")
                }
                "attempt_resources" if filename.starts_with("attempts/") => {
                    value["resources"]["cpus"] = json!(16)
                }
                "legacy" => value["identity_version"] = json!(1),
                _ => {}
            }
            write(&path, &value);
        }
        if change == "attempt_running" {
            write(
                &f.root
                    .path()
                    .join("project/runs/train/attempts/attempt-001/status.json"),
                &json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","state":"RUNNING"}),
            );
        }
        assert!(
            service
                .prepare("demo", "train", request, context(1))
                .is_err(),
            "{change}"
        );
        assert!(!f.slot().exists(), "{change}");
        assert!(!f.root.path().join("calls").exists(), "{change}");
    }
}

#[test]
fn concurrent_preparation_keeps_a_single_slot_and_submission_identity() {
    let f = Fixture::new();
    let service = Arc::new(f.service("ledger", true));
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let service = service.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                service.prepare("demo", "train", approval(), context(1))
            })
        })
        .collect();
    let mut ids = std::collections::BTreeSet::new();
    for thread in threads {
        match thread.join().unwrap() {
            Ok(view) => {
                ids.insert(view["submission_id"].as_str().unwrap().to_owned());
            }
            Err(crate::submissions::Error::Busy) => {}
            Err(error) => panic!("unexpected prepare failure: {error}"),
        }
    }
    assert_eq!(ids.len(), 1);
    let repeated = service
        .prepare("demo", "train", approval(), context(2))
        .unwrap();
    assert_eq!(repeated["reused"], true);
    assert!(ids.contains(repeated["submission_id"].as_str().unwrap()));
    assert_eq!(
        fs::read_dir(f.settings.registry_root.join("submissions/slots"))
            .unwrap()
            .count(),
        1
    );
    assert!(!f.root.path().join("calls").exists());
}

#[test]
fn generated_native_script_checks_actual_sif_bytes_before_starting_payload() {
    let mut f = Fixture::new();
    let mount = f.root.path().join("job-storage");
    fs::create_dir(&mount).unwrap();
    let mount = mount.to_str().unwrap().to_owned();
    f.profile.mount_root = mount.clone();
    f.profile.run_root = format!("{mount}/runs");
    f.profile.data_root = format!("{mount}/data");
    f.runtime.sif_path = format!("{mount}/image.sif");
    let sif = PathBuf::from(&f.runtime.sif_path);
    for filename in ["manifest.yaml", "attempts/attempt-001/attempt.yaml"] {
        let path = f.root.path().join("project/runs/train").join(filename);
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["backend"]["mount_root"] = json!(mount);
        value["backend"]["source_dir"] = json!(format!("{mount}/source"));
        value["backend"]["sif_path"] = json!(f.runtime.sif_path);
        value["storage"]["run_dir"] = json!(format!("{mount}/runs/demo/train"));
        value["storage"]["project_data_root"] = json!(format!("{mount}/data/demo"));
        write(&path, &value);
    }
    let service = f.service("ledger", true);
    service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    let prepared = f.prepared();
    let (_, script) = prepared.request.prepared_content().unwrap();
    let script_path = f.root.path().join("job.sh");
    fs::write(&script_path, script).unwrap();
    let bin = f.root.path().join("bin");
    fs::create_dir(&bin).unwrap();
    fs::write(
        bin.join("srun"),
        "#!/bin/sh\nprintf invoked > \"$SYNTHETIC_PAYLOAD_MARKER\"\n",
    )
    .unwrap();
    fs::set_permissions(bin.join("srun"), fs::Permissions::from_mode(0o700)).unwrap();
    let marker = f.root.path().join("payload-started");
    for valid in [false, true] {
        fs::write(
            &sif,
            if valid {
                b"synthetic materialized SIF".as_slice()
            } else {
                b"corrupt".as_slice()
            },
        )
        .unwrap();
        let output = std::process::Command::new("/bin/bash")
            .arg(&script_path)
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .env("SLURM_JOB_ID", "24827")
            .env("SYNTHETIC_PAYLOAD_MARKER", &marker)
            .output()
            .unwrap();
        if valid {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(marker.exists());
            assert!(String::from_utf8_lossy(&output.stdout).contains("ML_EXPD_RUNTIME=VERIFIED"));
        } else {
            assert_eq!(output.status.code(), Some(78));
            assert!(!marker.exists());
            assert!(String::from_utf8_lossy(&output.stderr).contains("SIF_VERIFICATION_FAILED"));
            assert!(!String::from_utf8_lossy(&output.stdout).contains("ML_EXPD_RUNTIME=VERIFIED"));
        }
    }
    assert!(!f.root.path().join("calls").exists());
}

#[test]
fn storage_corruption_and_frozen_authority_changes_prevent_dispatch() {
    for change in [
        "run",
        "attempt",
        "project",
        "source",
        "ssh",
        "permissions",
        "symlink",
        "hardlink",
        "corrupt",
        "deleted",
        "disabled",
        "scheduler",
    ] {
        let f = Fixture::new();
        let service = f.service("ledger", change != "disabled");
        let p = service
            .prepare("demo", "train", approval(), context(1))
            .unwrap();
        let id = p["submission_id"].as_str().unwrap();
        service
            .authorize(id, "actor:fixture", "reviewed", context(2))
            .unwrap();
        let root = f.root.path();
        match change {
            "run" => {
                let path = root.join("project/runs/train/manifest.yaml");
                let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                v["command"] = json!(["altered"]);
                write(&path, &v);
            }
            "attempt" => {
                let path = root.join("project/runs/train/attempts/attempt-001/attempt.yaml");
                let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                v["parameters"]["changed"] = json!(true);
                write(&path, &v);
            }
            "project" => {
                let path = root.join("project/project.json");
                let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                v["title"] = json!("changed");
                write(&path, &v);
            }
            "source" => {
                let path = f
                    .settings
                    .registry_root
                    .join("source-revisions/sources/demo")
                    .join(&f.runtime.source_id)
                    .join("tree/train.py");
                fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
                fs::write(path, "modified source").unwrap();
            }
            "ssh" => fs::write(root.join("ssh-config"), "Host forged\n").unwrap(),
            "permissions" => {
                fs::set_permissions(f.slot(), fs::Permissions::from_mode(0o644)).unwrap()
            }
            "symlink" => {
                let saved = root.join("saved-slot");
                fs::rename(f.slot(), &saved).unwrap();
                std::os::unix::fs::symlink(saved, f.slot()).unwrap();
            }
            "hardlink" => fs::hard_link(f.slot(), root.join("slot-link")).unwrap(),
            "corrupt" => fs::write(f.slot(), b"{").unwrap(),
            "deleted" => {
                crate::deletion::RunFence::new(&f.settings.registry_root)
                    .mark("demo", "train", &serde_json::Map::new(), NOW)
                    .unwrap();
            }
            "disabled" => {}
            "scheduler" => write(
                &root.join("project/runs/train/status.json"),
                &json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","state":"RUNNING"}),
            ),
            _ => unreachable!(),
        }
        assert!(
            service
                .execute(
                    id,
                    &format!("EXECUTE {id}"),
                    context(3),
                    &AtomicBool::new(false)
                )
                .is_err(),
            "{change}"
        );
        assert!(!root.join("calls").exists(), "{change}");
    }
}

#[test]
fn interrupted_claim_recovery_is_explicit_and_never_submits_before_reconcile() {
    let f = Fixture::new();
    let service = f.service("ledger", true);
    let p = service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    let id = p["submission_id"].as_str().unwrap();
    service
        .authorize(id, "actor:fixture", "reviewed", context(2))
        .unwrap();
    let private = f.prepared();
    let actions =
        crate::operations::Store::open_existing(f.settings.registry_root.join("actions")).unwrap();
    actions
        .claim(
            id,
            &format!("EXECUTE {id}"),
            &Policy {
                allow_scheduler_mutations: true,
                ..Default::default()
            },
            &FrozenFiles {
                execution_campaign_sha256: Some(private.binding.run_sha256.clone()),
                execution_manifest_sha256: Some(private.request.intent().manifest_sha256.clone()),
                ..Default::default()
            },
            crate::operations::ClaimContext {
                now: NOW,
                transition_id: "00000000000000000000000000000004",
                pid: 123,
            },
        )
        .unwrap();
    drop(service);
    let reopened = f.service("ledger", true);
    assert_eq!(reopened.get(id, NOW).unwrap()["status"], "EXECUTING");
    let recovered = reopened
        .recover_interrupted(id, NOW, "00000000000000000000000000000005")
        .unwrap();
    assert_eq!(recovered["status"], "RECONCILE_REQUIRED");
    assert_eq!(recovered["execution"]["safe_to_retry"], false);
    assert!(!f.root.path().join("calls").exists());
}

fn http_app(f: &Fixture, enabled: bool) -> axum::Router {
    let service = Arc::new(f.service("ledger", enabled));
    let api = crate::read_api::Api::open(f.settings.clone())
        .unwrap()
        .with_sources(f.source.clone())
        .unwrap()
        .with_submissions(service)
        .unwrap();
    crate::app(
        crate::Auth {
            token: "synthetic-native-submission-http-token-00000000".into(),
        },
        Some(Arc::new(api)),
    )
}
async fn http_request(app: axum::Router, method: &str, path: &str, body: &str) -> (u16, Value) {
    use tower::ServiceExt;
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method(method)
                .uri(path)
                .header(
                    "authorization",
                    "Bearer synthetic-native-submission-http-token-00000000",
                )
                .header("x-ml-expd-client-protocol", "2")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_owned()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let body = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}
#[tokio::test]
async fn native_http_preparation_restart_execute_and_reconcile_keep_one_submission() {
    let f = Fixture::new();
    let app = http_app(&f, true);
    let (status, health) = http_request(app.clone(), "GET", "/api/health", "").await;
    assert_eq!(status, 200);
    assert_eq!(health["scheduler_mutations"], true);
    assert!(
        health["capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("native-slurm-submissions.v1"))
    );
    assert_eq!(
        http_request(app.clone(), "GET", "/api/runs/demo/train/submission", "")
            .await
            .0,
        404
    );
    let (status, prepared) = http_request(
        app.clone(),
        "POST",
        "/api/runs/demo/train/submissions",
        r#"{"max_gpu_hours":1,"reason":"owned test"}"#,
    )
    .await;
    assert_eq!(status, 200, "{prepared}");
    let id = prepared["submission_id"].as_str().unwrap();
    let original = fs::read(f.slot()).unwrap();
    f.tools(&f.prepared());
    drop(app);
    let app = http_app(&f, true);
    let (status, recovered) =
        http_request(app.clone(), "GET", "/api/runs/demo/train/submission", "").await;
    assert_eq!(status, 200);
    assert_eq!(recovered["submission_id"], id);
    assert_eq!(fs::read(f.slot()).unwrap(), original);
    let payload = json!({"confirmation":format!("EXECUTE {id}")}).to_string();
    assert_eq!(
        http_request(
            app.clone(),
            "POST",
            &format!("/api/submissions/{id}/execute"),
            &payload
        )
        .await
        .0,
        409
    );
    assert!(!f.root.path().join("sbatch-calls").exists());
    assert_eq!(
        http_request(
            app.clone(),
            "POST",
            &format!("/api/submissions/{id}/authorize"),
            r#"{"reason":"reviewed"}"#
        )
        .await
        .0,
        200
    );
    let (status, done) = http_request(
        app.clone(),
        "POST",
        &format!("/api/submissions/{id}/execute"),
        &payload,
    )
    .await;
    assert_eq!(status, 200, "{done}");
    assert_eq!(done["status"], "VERIFIED");
    // Deliberately discard this response and reconstruct the complete service.
    drop(app);
    let app = http_app(&f, true);
    for (method, path, body) in [
        ("GET", format!("/api/submissions/{id}"), "".to_string()),
        (
            "POST",
            format!("/api/submissions/{id}/reconcile"),
            "{}".to_string(),
        ),
        ("POST", format!("/api/submissions/{id}/execute"), payload),
    ] {
        let (status, value) = http_request(app.clone(), method, &path, &body).await;
        assert_eq!(status, 200, "{value}");
        assert_eq!(value["status"], "VERIFIED");
        assert_eq!(value["submission_id"], id);
    }
    let (status, view) = http_request(app.clone(), "GET", "/api/runs/demo/train", "").await;
    assert_eq!(status, 200, "{view}");
    assert_eq!(view["scheduler_state"], "RUNNING");
    assert_eq!(
        view["evidence"]["scheduler"]["detail"]["backend_job_id"],
        "24827"
    );
    assert_eq!(view["evidence"]["scheduler"]["stale"], true);
    let terminal =
        json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","state":"SUCCEEDED"});
    write(
        &f.root.path().join("project/runs/train/status.json"),
        &terminal,
    );
    write(
        &f.root
            .path()
            .join("project/runs/train/attempts/attempt-001/status.json"),
        &terminal,
    );
    let (status, view) = http_request(app, "GET", "/api/runs/demo/train", "").await;
    assert_eq!(status, 200, "{view}");
    assert_eq!(view["scheduler_state"], "SUCCEEDED");
    assert_eq!(
        fs::read_to_string(f.root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
}
#[tokio::test]
async fn native_http_rejects_untrusted_bodies_and_execution_disabled_by_operator() {
    let f = Fixture::new();
    let app = http_app(&f, false);
    for body in [
        r#"{"max_gpu_hours":1,"script":"unsafe"}"#,
        r#"{"max_gpu_hours":1,"max_gpu_hours":2}"#,
        r#"{"max_gpu_hours":0}"#,
        r#"{"max_gpu_hours":1,"reason":"bad\nreason"}"#,
    ] {
        let (status, value) = http_request(
            app.clone(),
            "POST",
            "/api/runs/demo/train/submissions",
            body,
        )
        .await;
        assert_eq!(status, 422, "{body} {value}");
    }
    assert!(!f.slot().exists());
    let (status, prepared) = http_request(
        app.clone(),
        "POST",
        "/api/runs/demo/train/submissions",
        r#"{"max_gpu_hours":1}"#,
    )
    .await;
    assert_eq!(status, 200, "{prepared}");
    let id = prepared["submission_id"].as_str().unwrap();
    for (operation, body) in [
        ("execute", r#"{"confirmation":"wrong"}"#),
        ("reconcile", r#"{"new_run":true}"#),
        ("authorize", r#"{"actor":"admin"}"#),
        ("cancel", "{}"),
    ] {
        assert_eq!(
            http_request(
                app.clone(),
                "POST",
                &format!("/api/submissions/{id}/{operation}"),
                body
            )
            .await
            .0,
            422
        );
    }
    assert_eq!(
        http_request(
            app.clone(),
            "POST",
            &format!("/api/submissions/{id}/execute"),
            &json!({"confirmation":format!("EXECUTE {id}")}).to_string()
        )
        .await
        .0,
        403
    );
    assert!(!f.root.path().join("calls").exists());
    let (_, readiness) = http_request(app.clone(), "GET", "/api/readiness", "").await;
    assert_eq!(readiness["scheduler_mutations"], false);
    assert_eq!(readiness["providers_configured"], true);
    let (_, schema) = http_request(app, "GET", "/api/v2/openapi.json", "").await;
    assert!(
        schema["paths"]
            .get("/api/submissions/{identity}/execute")
            .is_some()
    );
}
#[test]
fn api_refuses_submission_service_bound_to_another_source_instance_or_workspace() {
    let f = Fixture::new();
    let service = Arc::new(f.service("ledger", true));
    let other = Arc::new(sources::Service::open(&f.settings, None).unwrap());
    let api = crate::read_api::Api::open(f.settings.clone())
        .unwrap()
        .with_sources(other)
        .unwrap();
    assert!(api.with_submissions(service).is_err());
    assert!(!f.root.path().join("calls").exists());
}

#[test]
fn explicit_reconcile_recovers_a_dead_claim_without_replaying_submit() {
    let f = Fixture::new();
    let service = f.service("ledger", true);
    let prepared = service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    let id = prepared["submission_id"].as_str().unwrap();
    service
        .authorize(id, "actor:fixture", "reviewed", context(2))
        .unwrap();
    let saved = f.prepared();
    f.tools(&saved);
    let store =
        crate::operations::Store::open_existing(f.settings.registry_root.join("actions")).unwrap();
    store
        .claim(
            id,
            &format!("EXECUTE {id}"),
            &Policy {
                allow_scheduler_mutations: true,
                ..Default::default()
            },
            &FrozenFiles {
                execution_campaign_sha256: Some(saved.binding.run_sha256.clone()),
                execution_manifest_sha256: Some(saved.request.intent().manifest_sha256.clone()),
                ..Default::default()
            },
            crate::operations::ClaimContext {
                now: NOW,
                transition_id: "00000000000000000000000000000004",
                pid: 123,
            },
        )
        .unwrap();
    drop(service);
    let service = f.service("ledger", true);
    let result = service.reconcile(id, NOW, &AtomicBool::new(false)).unwrap();
    assert_eq!(result["status"], "RECONCILE_REQUIRED");
    assert_eq!(result["execution"]["safe_to_retry"], false);
    assert!(!f.root.path().join("sbatch-calls").exists());
}
