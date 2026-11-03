//! Real immutable sources, canonical Run/Attempt reads, SQLite Actions and
//! native helper processes. Scheduler binaries and SSH are owned fixtures.
use super::*;
use crate::{
    read_config, sources,
    submissions::{Backend, Context, Error as ServiceError, Service},
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
        let mount_path = root.path().join("shared");
        fs::create_dir_all(mount_path.join("runs")).unwrap();
        let mount = mount_path.to_str().unwrap();
        fs::write(
            root.path().join("native-shared-root"),
            format!("{mount}/runs"),
        )
        .unwrap();
        let runtime = Runtime {
            worker: None,
            source_dir: None,
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
                env: Default::default(),
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
        self.service_with_control(mode, enabled, None)
    }
    fn service_with_control(
        &self,
        mode: &str,
        enabled: bool,
        control_helper: Option<crate::submissions::config::ControlHelper>,
    ) -> Service {
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
                control_helper,
                runtimes: BTreeMap::from([(self.runtime.runtime_id.clone(), self.runtime.clone())]),
                data_helper_config: None,
                script_preparation: false,
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
    let service = f.service_with_control(
        "ledger-lost-submit",
        true,
        Some(crate::submissions::config::ControlHelper {
            helper_path: "/opt/ml-expd/bin/previous-control-helper".into(),
            helper_config_path: "/etc/ml-expd/previous-control.json".into(),
        }),
    );
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
    let frozen_slot = fs::read(f.slot()).unwrap();
    let original_argv = fs::read_to_string(f.root.path().join("argv")).unwrap();
    assert!(original_argv.contains("/opt/ml-expd/bin/ml-exp-worker-rs"));
    drop(service);
    let reopened = f.service_with_control(
        "ledger",
        true,
        Some(crate::submissions::config::ControlHelper {
            helper_path: "/opt/ml-expd/bin/upgraded-control-helper".into(),
            helper_config_path: "/etc/ml-expd/upgraded-control.json".into(),
        }),
    );
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
    assert_eq!(fs::read(f.slot()).unwrap(), frozen_slot);
    let control_argv = fs::read_to_string(f.root.path().join("argv")).unwrap();
    assert!(control_argv.contains("/opt/ml-expd/bin/upgraded-control-helper"));
    assert!(control_argv.contains("/etc/ml-expd/upgraded-control.json"));
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
    let (manifest, script) = prepared.request.prepared_content().unwrap();
    let attempt_directory = PathBuf::from(format!("{mount}/runs/demo/train/attempts/attempt-001"));
    fs::create_dir_all(&attempt_directory).unwrap();
    fs::write(attempt_directory.join("attempt.json"), manifest).unwrap();
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
    fs::remove_file(&marker).unwrap();
    fs::write(attempt_directory.join("attempt.json"), b"changed").unwrap();
    let output = std::process::Command::new("/bin/bash")
        .arg(&script_path)
        .env_clear()
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("SLURM_JOB_ID", "24827")
        .env("SYNTHETIC_PAYLOAD_MARKER", &marker)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(78));
    assert!(!marker.exists());
    assert!(String::from_utf8_lossy(&output.stderr).contains("ATTEMPT_VERIFICATION_FAILED"));
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
    for path in [
        "/api/runs/demo/train/attempts/attempt-001/collection",
        "/api/attempts/demo/train::attempt-001/show",
    ] {
        let (status, exact) = http_request(app.clone(), "GET", path, "").await;
        assert_eq!(status, 200, "{exact}");
        assert_eq!(
            exact["evidence"]["scheduler"]["state"], "RUNNING",
            "{exact}"
        );
        assert_eq!(
            exact["evidence"]["scheduler"]["detail"]["backend_job_id"],
            "24827"
        );
        assert_eq!(exact["evidence"]["scheduler"]["stale"], true);
        if path.ends_with("collection") {
            assert_eq!(exact["training"]["status"], "UNKNOWN");
            assert_eq!(exact["download_available"], false);
        } else {
            assert_eq!(exact["attempt"]["state"], "RUNNING");
        }
    }
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

fn creation_request(f: &Fixture) -> native::creation::Request {
    serde_json::from_value(json!({"run_id":"created","runtime_id":f.runtime.runtime_id,"executor":f.profile.target.backend_id,
        "command":["python","-u","/workspace/train.py"],"resources":{"gpus":1,"cpus":8,"memory_gb":16},
        "walltime_seconds":300,"seed":42,"parameters":{"lr":0.001},
        "metrics_schema":{"definitions":{"val_loss":{"unit":"nats/token","required":true}}},
        "evaluation":{"tokenizer":"fixture-v1","variant_id":"v1","D_FT":0}})).unwrap()
}
#[test]
fn immutable_run_creation_reuses_exact_metadata_and_admits_first_submission_after_restart() {
    let f = Fixture::new();
    let request = creation_request(&f);
    assert!(matches!(
        f.service("ledger", true)
            .create_run("demo", request.clone(), NOW),
        Err(ServiceError::RunCreationDisabled)
    ));
    let service = f.service("ledger", true).with_run_creation(true).unwrap();
    let first = service.create_run("demo", request.clone(), NOW).unwrap();
    assert_eq!(first["creation_state"], "CREATED");
    assert_eq!(first["scheduler_mutation_performed"], false);
    assert_eq!(first["reused"], false);
    assert!(!f.root.path().join("sbatch-calls").exists());
    let directory = f.root.path().join("project/runs/created");
    let manifest = fs::read(directory.join("manifest.json")).unwrap();
    let attempt = fs::read(directory.join("attempts/attempt-001/attempt.json")).unwrap();
    assert_eq!(first["run_sha256"], helper::sha256_bytes(&manifest));
    assert_eq!(first["attempt_sha256"], helper::sha256_bytes(&attempt));
    let run: Value = serde_json::from_slice(&manifest).unwrap();
    assert_eq!(run["evaluation"]["protocol"]["evaluation"]["D_FT"], 0);
    assert_eq!(
        run["evaluation"]["metrics_schema"]["definitions"]["val_loss"]["unit"],
        "nats/token"
    );
    assert!(
        fs::read_dir(f.settings.registry_root.join("submissions/slots"))
            .unwrap()
            .next()
            .is_none()
    );
    drop(service);
    let service = f.service("ledger", true).with_run_creation(true).unwrap();
    let second = service
        .create_run("demo", request.clone(), "2026-10-28T00:01:00Z")
        .unwrap();
    assert_eq!(second["reused"], true);
    assert_eq!(second["created_at"], first["created_at"]);
    assert_eq!(second["run_sha256"], first["run_sha256"]);
    assert_eq!(fs::read(directory.join("manifest.json")).unwrap(), manifest);
    assert_eq!(
        fs::read(directory.join("attempts/attempt-001/attempt.json")).unwrap(),
        attempt
    );
    let prepared = service
        .prepare("demo", "created", approval(), context(1))
        .unwrap();
    assert_eq!(prepared["status"], "PREPARED");
    assert_eq!(prepared["attempt_id"], "attempt-001");
    assert!(!f.root.path().join("sbatch-calls").exists());
    let mut changed = request.clone();
    changed.seed = 43;
    assert!(matches!(
        service.create_run("demo", changed, NOW),
        Err(ServiceError::Conflict)
    ));
    assert_eq!(fs::read(directory.join("manifest.json")).unwrap(), manifest);
}
#[test]
fn newly_materialized_image_keeps_its_source_directory_through_the_launch_package() {
    let mut f = Fixture::new();
    let original = serde_json::to_value(&f.runtime).unwrap();
    assert!(original.get("source_dir").is_none());
    assert_eq!(
        serde_json::to_value(serde_json::from_value::<Runtime>(original.clone()).unwrap()).unwrap(),
        original
    );
    f.runtime.source_dir = Some("/ml-expd/source".into());
    f.runtime.worker = Some(ml_exp_core::slurm::managed::Runtime {
        program: "/usr/local/bin/ml-exp-worker-rs".into(),
        base_environment: BTreeMap::new(),
        runtime_environment: Vec::new(),
    });
    let mut request = creation_request(&f);
    request.command = vec!["/usr/local/bin/python".into(), "train.py".into()];
    let service = f.service("ledger", true).with_run_creation(true).unwrap();
    service.create_run("demo", request, NOW).unwrap();
    let directory = f.root.path().join("project/runs/created");
    let manifest = fs::read(directory.join("manifest.json")).unwrap();
    let run: Value = serde_json::from_slice(&manifest).unwrap();
    assert_eq!(run["execution"]["workdir"], "/ml-expd/source");
    let prepared = service
        .prepare("demo", "created", approval(), context(1))
        .unwrap();
    let id = prepared["submission_id"].as_str().unwrap();
    let saved = service.get(id, NOW).unwrap();
    assert_eq!(saved["status"], "PREPARED");
    // The durable helper request must carry the same directory into the
    // managed worker; an image build alone cannot establish launch readiness.
    let path = f
        .settings
        .registry_root
        .join("submissions/slots")
        .join(format!("{}.json", native::slot("demo", "created").unwrap()));
    let value: Prepared = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let request = value.request;
    let (manifest, script) = request.prepared_content().unwrap();
    let package = ml_exp_core::slurm::staging::plan(request.intent(), manifest.as_bytes())
        .unwrap()
        .unwrap()
        .managed
        .unwrap();
    assert_eq!(package.manifest.workdir, "/ml-expd/source");
    assert_eq!(package.config.workdir, Path::new("/ml-expd/source"));
    assert!(script.contains("--pwd /ml-expd/source"));
    assert!(!f.root.path().join("sbatch-calls").exists());
}
#[test]
fn creation_refuses_partial_corrupt_and_replaced_authority_without_repair() {
    let f = Fixture::new();
    let service = f.service("ledger", true).with_run_creation(true).unwrap();
    let mut request = creation_request(&f);
    request.resources.gpus = 99;
    assert!(service.create_run("demo", request, NOW).is_err());
    let directory = f.root.path().join("project/runs/created");
    assert!(!directory.exists());
    fs::create_dir(&directory).unwrap();
    assert!(matches!(
        service.create_run("demo", creation_request(&f), NOW),
        Err(ServiceError::Conflict)
    ));
    assert!(fs::read_dir(&directory).unwrap().next().is_none());
    fs::remove_dir(&directory).unwrap();
    service
        .create_run("demo", creation_request(&f), NOW)
        .unwrap();
    fs::remove_file(directory.join("attempts/attempt-001/attempt.json")).unwrap();
    assert!(
        service
            .create_run("demo", creation_request(&f), NOW)
            .is_err()
    );
    assert!(!directory.join("attempts/attempt-001/attempt.json").exists());
    let root = f.root.path().join("project/runs");
    fs::rename(&root, f.root.path().join("old-runs")).unwrap();
    fs::create_dir(&root).unwrap();
    assert!(
        service
            .create_run("demo", creation_request(&f), NOW)
            .is_err()
    );
    assert!(fs::read_dir(root).unwrap().next().is_none());
}
#[test]
fn creation_refuses_reformatted_original_files_without_replacing_them() {
    for filename in [
        "manifest.json",
        "request.json",
        "attempts/attempt-001/attempt.json",
    ] {
        let f = Fixture::new();
        let service = f.service("ledger", true).with_run_creation(true).unwrap();
        service
            .create_run("demo", creation_request(&f), NOW)
            .unwrap();
        let path = f.root.path().join("project/runs/created").join(filename);
        let original = fs::read(&path).unwrap();
        let mut changed = original.clone();
        changed.push(b'\n');
        fs::write(&path, &changed).unwrap();
        assert!(
            service
                .create_run("demo", creation_request(&f), NOW)
                .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), changed);
        assert!(!f.root.path().join("sbatch-calls").exists());
        fs::write(&path, original).unwrap();
        assert_eq!(
            service
                .create_run("demo", creation_request(&f), NOW)
                .unwrap()["reused"],
            true
        );
    }
}

#[tokio::test]
async fn run_creation_http_and_native_status_work_without_authored_run_files() {
    let f = Fixture::new();
    let service = Arc::new(f.service("ledger", true).with_run_creation(true).unwrap());
    let api = crate::read_api::Api::open(f.settings.clone())
        .unwrap()
        .with_sources(f.source.clone())
        .unwrap()
        .with_submissions(service)
        .unwrap();
    let app = crate::app(
        crate::Auth {
            token: "synthetic-native-submission-http-token-00000000".into(),
        },
        Some(Arc::new(api)),
    );
    let (status, health) = http_request(app.clone(), "GET", "/api/health", "").await;
    assert_eq!(status, 200);
    assert!(
        health["capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("native-run-creation.v1"))
    );
    let body = serde_json::to_string(&creation_request(&f)).unwrap();
    let (status, receipt) =
        http_request(app.clone(), "POST", "/api/projects/demo/runs", &body).await;
    assert_eq!(status, 200, "{receipt}");
    assert_eq!(receipt["reused"], false);
    let (status, run) = http_request(app.clone(), "GET", "/api/runs/demo/created", "").await;
    assert_eq!(status, 200, "{run}");
    assert_eq!(run["scheduler_state"], "NOT_SUBMITTED");
    let (status, receipt) =
        http_request(app.clone(), "POST", "/api/projects/demo/runs", &body).await;
    assert_eq!(status, 200);
    assert_eq!(receipt["reused"], true);
    let bad = body.replacen("{", "{\"run_id\":\"foreign\",", 1);
    assert_eq!(
        http_request(app.clone(), "POST", "/api/projects/demo/runs", &bad)
            .await
            .0,
        422
    );
    assert!(!f.root.path().join("project/runs/foreign").exists());
    let (status, prepared) = http_request(
        app,
        "POST",
        "/api/runs/demo/created/submissions",
        r#"{"max_gpu_hours":1}"#,
    )
    .await;
    assert_eq!(status, 200, "{prepared}");
    assert_eq!(prepared["status"], "PREPARED");
    assert!(!f.root.path().join("sbatch-calls").exists());
}

#[test]
fn submission_composition_requires_the_same_complete_read_scope() {
    let f = Fixture::new();
    let service = Arc::new(f.service("ledger", true).with_run_creation(true).unwrap());
    let mut changed = f.settings.clone();
    changed.projects[0].discover_runs = Some(false);
    let api = crate::read_api::Api::open(changed)
        .unwrap()
        .with_sources(f.source.clone())
        .unwrap();
    assert!(api.with_submissions(service).is_err());
}

#[test]
fn competing_run_creators_and_deletion_fence_cannot_replace_an_identity() {
    let f = Fixture::new();
    let a = f.service("ledger", true).with_run_creation(true).unwrap();
    let b = f.service("ledger", true).with_run_creation(true).unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let jobs = [a, b]
        .into_iter()
        .enumerate()
        .map(|(i, service)| {
            let barrier = barrier.clone();
            let mut request = creation_request(&f);
            request.seed = 42 + i as i64;
            std::thread::spawn(move || {
                barrier.wait();
                service.create_run("demo", request, NOW)
            })
        })
        .collect::<Vec<_>>();
    let results = jobs
        .into_iter()
        .map(|job| job.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert!(
        results
            .iter()
            .any(|r| matches!(r, Err(ServiceError::Busy | ServiceError::Conflict)))
    );
    let directory = f.root.path().join("project/runs/created");
    let original = fs::read(directory.join("manifest.json")).unwrap();
    let fence = crate::deletion::RunFence::new(&f.settings.registry_root);
    let guard = fence.exclusive_existing(false).unwrap();
    guard
        .mark("demo", "created", &serde_json::Map::new(), NOW)
        .unwrap();
    drop(guard);
    let service = f.service("ledger", true).with_run_creation(true).unwrap();
    assert!(matches!(
        service.create_run("demo", creation_request(&f), NOW),
        Err(ServiceError::Authority)
    ));
    assert_eq!(fs::read(directory.join("manifest.json")).unwrap(), original);
    assert!(
        fs::read_dir(f.settings.registry_root.join("submissions/slots"))
            .unwrap()
            .next()
            .is_none()
    );
}

#[test]
fn trusted_image_worker_changes_only_launch_entry_and_keeps_scientific_command_frozen() {
    let mut f = Fixture::new();
    assert!(
        serde_json::to_value(&f.runtime)
            .unwrap()
            .get("worker")
            .is_none()
    );
    f.runtime.worker = Some(ml_exp_core::slurm::managed::Runtime {
        program: "/opt/ml-expd/bin/ml-exp-worker-rs".into(),
        base_environment: BTreeMap::from([("PATH".into(), "/opt/venv/bin:/usr/bin:/bin".into())]),
        runtime_environment: vec!["CUDA_VISIBLE_DEVICES".into()],
    });
    let service = f.service("ledger", true);
    service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    let prepared = f.prepared();
    let (manifest, script) = prepared.request.prepared_content().unwrap();
    let original: Value = serde_json::from_str(manifest).unwrap();
    assert_eq!(
        original["command"],
        json!(["python", "-u", "/workspace/train.py"])
    );
    let package = ml_exp_core::slurm::staging::plan(prepared.request.intent(), manifest.as_bytes())
        .unwrap()
        .unwrap()
        .managed
        .unwrap();
    assert_eq!(
        package.manifest.argv,
        vec!["python", "-u", "/workspace/train.py"]
    );
    assert!(script.contains("/opt/ml-expd/bin/ml-exp-worker-rs managed-launch --config"));
    assert!(script.contains("--initialize-and-run"));
    assert!(script.contains(&helper::sha256_bytes(&package.config_bytes().unwrap())[7..]));
    assert!(package.config.endpoint.is_empty());
    assert_eq!(
        package.config.local_binding.unwrap().identity.attempt_id,
        "attempt-001"
    );
    assert!(!f.root.path().join("sbatch-calls").exists());
}

#[test]
fn backend_checkpoint_registration_and_explicit_new_run_restore_are_scoped_and_immutable() {
    let mut f = Fixture::new();
    let workdir = f.root.path().join("code");
    f.runtime.worker = Some(ml_exp_core::slurm::managed::Runtime {
        program: "/opt/ml-expd/bin/ml-exp-worker-rs".into(),
        base_environment: BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
        runtime_environment: Vec::new(),
    });
    let ready = json!({"step":4,"files":[{"path":"checkpoints/step-4/state.bin","bytes":7,"sha256":"9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"}]});
    for filename in ["manifest.yaml", "attempts/attempt-001/attempt.yaml"] {
        let path = f.root.path().join("project/runs/train").join(filename);
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["execution"]["workdir"] = json!(workdir);
        value["command"] = json!([
            "/bin/sh",
            "-c",
            format!(
                "mkdir -p \"$OUTPUT_DIR/checkpoints/step-4\"; printf weights > \"$OUTPUT_DIR/checkpoints/step-4/state.bin\"; printf %s {} > \"$OUTPUT_DIR/checkpoint.ready.json\"",
                ml_exp_core::slurm::shell_quote(&ready.to_string()).unwrap()
            )
        ]);
        write(&path, &value);
    }
    let service = f.service("ledger", true).with_run_creation(true).unwrap();
    let prepared = service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    let id = prepared["submission_id"].as_str().unwrap();
    let frozen = f.prepared();
    f.tools(&frozen);
    assert!(
        service
            .refresh_checkpoint(id, NOW, &AtomicBool::new(false))
            .is_err()
    );
    service.authorize(id, "test", "", context(2)).unwrap();
    service
        .execute(
            id,
            &format!("EXECUTE {id}"),
            context(3),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(
        service
            .checkpoints("demo", "train", "attempt-001", None)
            .unwrap()["checkpoints"],
        json!([])
    );
    let plan = ml_exp_core::slurm::staging::plan(
        frozen.request.intent(),
        frozen.request.prepared_content().unwrap().0.as_bytes(),
    )
    .unwrap()
    .unwrap();
    let config = plan.managed.unwrap().config;
    ml_exp_worker::managed::initialize(&config).unwrap();
    ml_exp_worker::managed::run(&config, &AtomicBool::new(false)).unwrap();
    let registered = service
        .refresh_checkpoint(id, NOW, &AtomicBool::new(false))
        .unwrap();
    let document = registered["checkpoint"].clone();
    let checkpoint_id = document["checkpoint_id"].as_str().unwrap();
    assert_eq!(document["step"], 4);
    assert_eq!(registered["scheduler_mutation_performed"], false);
    assert_eq!(
        service
            .refresh_checkpoint(id, NOW, &AtomicBool::new(false))
            .unwrap(),
        registered
    );
    assert_eq!(
        service
            .checkpoints("demo", "train", "attempt-001", Some(checkpoint_id))
            .unwrap(),
        document
    );
    assert!(
        service
            .checkpoints("demo", "train", "attempt-002", Some(checkpoint_id))
            .is_err()
    );
    let original = fs::read(
        f.root
            .path()
            .join("project/runs/train/attempts/attempt-001/attempt.yaml"),
    )
    .unwrap();
    let mut request = creation_request(&f);
    request.command=vec!["/bin/sh".into(),"-c".into(),"test \"$(cat \"$RESUME_DIR/state.bin\")\" = weights; test \"$RESUME_STEP\" = 4; printf restored".into()];
    request.resume_from = Some(ml_exp_core::checkpoints::Reference {
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        checkpoint_id: checkpoint_id.into(),
    });
    let created = service.create_run("demo", request.clone(), NOW).unwrap();
    assert!(!created["reused"].as_bool().unwrap());
    assert_eq!(
        service.create_run("demo", request.clone(), NOW).unwrap()["reused"],
        true
    );
    let value: Value = serde_json::from_slice(
        &fs::read(f.root.path().join("project/runs/created/manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(value["native_checkpoint"], document);
    let spec = native::job(
        &value,
        &native::Binding {
            project: "demo".into(),
            run_id: "created".into(),
            attempt_id: "attempt-001".into(),
            project_sha256: "".into(),
            run_sha256: "".into(),
            attempt_sha256: "".into(),
            source_receipt_sha256: "".into(),
            backend_config_sha256: "".into(),
        },
    )
    .unwrap();
    let mut package = ml_exp_core::slurm::managed::plan_restoring(
        &helper::Scope {
            project: "demo".into(),
            run_id: "created".into(),
            attempt_id: "attempt-001".into(),
        },
        &spec,
        f.runtime.worker.as_ref().unwrap(),
        Some(&document),
    )
    .unwrap();
    let encoded = ml_exp_core::canonical::python_json_bytes(&document).unwrap();
    let reordered: Value = serde_json::from_slice(&encoded).unwrap();
    let other = ml_exp_core::slurm::managed::plan_restoring(
        &helper::Scope {
            project: "demo".into(),
            run_id: "created".into(),
            attempt_id: "attempt-001".into(),
        },
        &spec,
        f.runtime.worker.as_ref().unwrap(),
        Some(&reordered),
    )
    .unwrap();
    assert_eq!(
        package.config_bytes().unwrap(),
        other.config_bytes().unwrap()
    );
    // Local effect shell exercises the actual restore check without a second scheduler job.
    package.config.workdir = workdir.clone();
    package.manifest.workdir = workdir.to_str().unwrap().into();
    let restore = f.root.path().join("restore-local");
    fs::create_dir(&restore).unwrap();
    package.config.attempt_root = restore.join("worker");
    package.config.output_directory = Some(restore.join("outputs"));
    package.manifest.environment.insert(
        "OUTPUT_DIR".into(),
        restore.join("outputs").to_str().unwrap().into(),
    );
    package.config.manifest_path = restore.join("manifest.json");
    let bytes = package.manifest.encode().unwrap();
    fs::write(&package.config.manifest_path, &bytes).unwrap();
    fs::set_permissions(
        &package.config.manifest_path,
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    package.config.local_binding = Some(ml_exp_core::launch::Binding {
        identity: package.manifest.identity.clone(),
        sha256: helper::sha256_bytes(&bytes)[7..].into(),
    });
    ml_exp_worker::managed::initialize(&package.config).unwrap();
    let finished = ml_exp_worker::managed::run(&package.config, &AtomicBool::new(false)).unwrap();
    assert_eq!(
        finished.receipt.unwrap().training,
        ml_exp_worker::managed::Training::Completed
    );
    let checkpoint_file = std::path::Path::new(document["state_root"].as_str().unwrap())
        .join("checkpoints/step-4/state.bin");
    fs::set_permissions(&checkpoint_file, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&checkpoint_file, b"changed").unwrap();
    package.config.attempt_root = restore.join("rejected-worker");
    package.config.output_directory = Some(restore.join("rejected-outputs"));
    package.manifest.environment.insert(
        "OUTPUT_DIR".into(),
        restore.join("rejected-outputs").to_str().unwrap().into(),
    );
    let bytes = package.manifest.encode().unwrap();
    fs::write(&package.config.manifest_path, &bytes).unwrap();
    package.config.local_binding.as_mut().unwrap().sha256 =
        helper::sha256_bytes(&bytes)[7..].into();
    ml_exp_worker::managed::initialize(&package.config).unwrap();
    assert!(matches!(
        ml_exp_worker::managed::run(&package.config, &AtomicBool::new(false)),
        Err(ml_exp_worker::managed::Error::Binding)
    ));
    assert!(!restore.join("rejected-outputs").exists());
    assert!(matches!(
        service.refresh_checkpoint(id, NOW, &AtomicBool::new(false)),
        Err(ServiceError::CheckpointUnavailable)
    ));
    assert_eq!(
        service
            .checkpoints("demo", "train", "attempt-001", Some(checkpoint_id))
            .unwrap(),
        document
    );
    assert_eq!(
        fs::read_to_string(f.root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
    assert_eq!(
        fs::read(
            f.root
                .path()
                .join("project/runs/train/attempts/attempt-001/attempt.yaml")
        )
        .unwrap(),
        original
    );
}

#[test]
fn exact_worker_refresh_survives_restart_and_never_changes_submission_or_retrains() {
    let mut f = Fixture::new();
    let workdir = f.root.path().join("code");
    f.runtime.worker = Some(ml_exp_core::slurm::managed::Runtime {
        program: "/opt/ml-expd/bin/ml-exp-worker-rs".into(),
        base_environment: BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
        runtime_environment: Vec::new(),
    });
    for filename in ["manifest.yaml", "attempts/attempt-001/attempt.yaml"] {
        let path = f.root.path().join("project/runs/train").join(filename);
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["execution"]["workdir"] = json!(workdir);
        value["command"] = json!([
            "/bin/sh",
            "-c",
            "printf 'training started\\n'; printf '{\"step\":1,\"name\":\"custom/loss\",\"value\":2,\"unit\":\"nats/token\",\"variant_id\":\"a\"}\\n' > \"$OUTPUT_DIR/metrics.jsonl\""
        ]);
        write(&path, &value);
    }
    let service = f.service("ledger", true);
    let initial = service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    let id = initial["submission_id"].as_str().unwrap();
    assert!(service.refresh(id, NOW, &AtomicBool::new(false)).is_err());
    let prepared = f.prepared();
    f.tools(&prepared);
    service.authorize(id, "test", "", context(2)).unwrap();
    let submitted = service
        .execute(
            id,
            &format!("EXECUTE {id}"),
            context(3),
            &AtomicBool::new(false),
        )
        .unwrap();
    let bytes = fs::read(
        f.root
            .path()
            .join("project/runs/train/attempts/attempt-001/attempt.yaml"),
    )
    .unwrap();
    let plan = ml_exp_core::slurm::staging::plan(
        prepared.request.intent(),
        prepared.request.prepared_content().unwrap().0.as_bytes(),
    )
    .unwrap()
    .unwrap();
    let config = plan.managed.unwrap().config;
    ml_exp_worker::managed::initialize(&config).unwrap();
    ml_exp_worker::managed::run(&config, &AtomicBool::new(false)).unwrap();
    let refreshed = service.refresh(id, NOW, &AtomicBool::new(false)).unwrap();
    assert_eq!(refreshed["execution"], submitted["execution"]);
    assert_eq!(refreshed["worker_observation"]["phase"], "FINISHED");
    assert_eq!(
        fs::read_to_string(f.root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
    let evidence = crate::experiments::Store::open(
        f.settings.bindings(),
        crate::deletion::RunFence::new(&f.settings.registry_root),
    )
    .unwrap();
    let reference = evidence
        .discover("demo", &crate::experiments::DiscoveryLimits::default())
        .unwrap()
        .runs
        .into_iter()
        .find(|r| r.reference.run_id == "train")
        .unwrap()
        .reference;
    let mut snapshot = evidence.snapshot(&reference).unwrap();
    service.apply_observation(&mut snapshot).unwrap();
    let collection =
        crate::read_model::attempt_collection(&snapshot, "attempt-001", 1793145600.).unwrap();
    assert_eq!(collection["training"]["status"], "COMPLETED");
    assert_eq!(collection["training"]["exit_code"], 0);
    assert_eq!(collection["download_available"], false);
    let logs = crate::read_model::attempt_logs(
        &snapshot,
        "attempt-001",
        &crate::read_model::LogQuery::default(),
    )
    .unwrap();
    assert_eq!(logs["streams"]["stdout"]["lines"][0], "training started");
    assert_eq!(snapshot.attempts[0].metrics.records[0]["variant_id"], "a");
    assert!(!snapshot.attempts[0].metrics.truncated);
    drop(service);
    let service = f.service("ledger", false);
    service.refresh(id, NOW, &AtomicBool::new(false)).unwrap(); // reads work with submissions disabled
    let mut after = evidence.snapshot(&reference).unwrap();
    service.apply_observation(&mut after).unwrap();
    assert_eq!(
        after.attempts[0].metrics.records,
        snapshot.attempts[0].metrics.records
    );
    // Temporarily missing worker state preserves the last accepted sample.
    let state = config.attempt_root.join(".managed-launch.lock");
    fs::rename(&state, state.with_extension("saved")).unwrap();
    assert!(matches!(
        service.refresh(id, NOW, &AtomicBool::new(false)),
        Err(ServiceError::ObservationUnavailable)
    ));
    let mut retained = evidence.snapshot(&reference).unwrap();
    service.apply_observation(&mut retained).unwrap();
    assert_eq!(
        retained.attempts[0].metrics.records,
        snapshot.attempts[0].metrics.records
    );
    assert_eq!(
        fs::read(
            f.root
                .path()
                .join("project/runs/train/attempts/attempt-001/attempt.yaml")
        )
        .unwrap(),
        bytes
    );
    assert_eq!(
        fs::read_to_string(f.root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
}

#[test]
fn expired_unsubmitted_gate_renewal_preserves_identity_and_survives_response_loss() {
    let f = Fixture::new();
    let service = f.service("ledger", false);
    let first = service
        .prepare("demo", "train", approval(), context(1))
        .unwrap();
    let original = first["submission_id"].as_str().unwrap();
    service
        .authorize(original, "fixture", "owned", context(2))
        .unwrap();
    let request = ml_exp_core::renewal::Request {
        request_id: "a".repeat(32),
        confirmation: format!("RENEW {original}"),
    };
    assert!(
        service
            .renew(original, request.clone(), context(3))
            .is_err()
    );
    let later = || Context {
        now: "2026-10-28T02:00:00Z",
        expires: "2026-10-28T02:30:00Z",
        transition_id: "00000000000000000000000000000004",
    };
    let before = f.prepared();
    let renewed = service.renew(original, request.clone(), later()).unwrap();
    let id = renewed["submission_id"].as_str().unwrap().to_owned();
    assert_ne!(id, original);
    assert_eq!(renewed["status"], "PREPARED");
    assert_eq!(renewed["scheduler_mutation_performed"], false);
    let after = f.prepared();
    assert!(before.binding == after.binding);
    assert!(before.request == after.request);
    assert_eq!(before.definition_sha256, after.definition_sha256);
    let old = service.get(original, later().now).unwrap();
    assert_eq!(old["status"], "AUTHORIZED");
    assert_eq!(old["replacement_submission_id"], id);
    assert_eq!(
        service.get(&id, later().now).unwrap()["renewed_from"],
        original
    );
    assert!(
        service
            .authorize(original, "fixture", "expired", later())
            .is_err()
    );
    drop(service);
    let service = f.service("ledger", false);
    assert_eq!(
        service.renew(original, request.clone(), later()).unwrap()["submission_id"],
        id
    );
    assert_eq!(
        service.for_run("demo", "train", later().now).unwrap()["submission_id"],
        id
    );
    let mut another = request;
    another.request_id = "b".repeat(32);
    assert!(service.renew(original, another, later()).is_err());
    assert!(!f.root.path().join("calls").exists());
}
