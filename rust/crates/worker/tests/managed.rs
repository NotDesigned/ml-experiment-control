#![cfg(unix)]
use ml_exp_core::launch;
use ml_exp_worker::managed::{self, Config, Error, Phase, Training};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::AtomicBool,
    thread,
    time::{Duration, Instant},
};

fn private_directory(path: &Path) {
    fs::create_dir(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn private_file(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
struct Fixture {
    _root: tempfile::TempDir,
    config: Config,
    document: Value,
    config_file: PathBuf,
}
impl Fixture {
    fn new(mode: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        let workdir = root.path().join("code");
        private_directory(&workdir);
        let attempt = root.path().join("attempt-001");
        let manifest_path = root.path().join("source-manifest.json");
        let config_file = root.path().join("config.json");
        let document = json!({"contract":launch::CONTRACT,"identity":{"project":"demo","run_id":"train","attempt_id":"attempt-001"},
            "environment":{"OUTPUT_DIR":attempt.join("outputs"),"FIXTURE_MODE":mode,"FROZEN_PARAMETER":"scientific-value"},
            "argv":[std::env::current_exe().unwrap(),"--exact","managed_child","--ignored","--nocapture"],
            "workdir":workdir,"timeout_seconds":if mode=="timeout" {1}else{5}});
        let config = Config {
            manifest_path,
            endpoint: String::new(),
            attempt_root: attempt,
            workdir,
            maximum_timeout_seconds: 5,
            base_environment: BTreeMap::new(),
            local_binding: None,
            output_directory: None,
            runtime_environment: Vec::new(),
            checkpoint_restore: None,
            checkpoint_restore_root: None,
            datasets: Vec::new(),
            data_preparation: None,
        };
        let mut fixture = Self {
            _root: root,
            config,
            document,
            config_file,
        };
        fixture.seal();
        fixture
    }
    fn seal(&mut self) {
        let bytes = launch::encode(&self.document).unwrap();
        self.config.endpoint = format!(
            "https://example.test/ml-expd/api/launch-transfers/demo/train/attempt-001/{:x}",
            Sha256::digest(&bytes)
        );
        private_file(&self.config.manifest_path, &bytes);
        private_file(
            &self.config_file,
            &serde_json::to_vec(&self.config).unwrap(),
        );
    }
    fn initialize(&self) {
        let prepared = managed::initialize(&self.config).unwrap();
        assert_eq!(prepared.phase, Phase::Prepared);
        assert!(!prepared.safe_to_retry);
        assert!(prepared.receipt.is_none());
    }
    fn count(&self) -> usize {
        fs::read_to_string(self.config.workdir.join("executions"))
            .unwrap_or_default()
            .lines()
            .count()
    }
    fn cli(&self, action: Option<&str>) -> std::process::Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ml-exp-worker-rs"));
        command
            .env_clear()
            .args(["managed-launch", "--config"])
            .arg(&self.config_file);
        if let Some(action) = action {
            command.arg(action);
        }
        command.output().unwrap()
    }
}

/// A real native Rust training fixture, executed only as a sealed child.
#[test]
#[ignore = "subprocess fixture; run by the managed-launch acceptance tests"]
fn managed_child() {
    assert!(std::env::var_os("HOME").is_none());
    assert!(std::env::var_os("ML_EXPD_BOOTSTRAP_TOKEN").is_none());
    assert!(std::env::var_os("ML_EXPD_UPLOAD_TOKEN").is_none());
    assert_eq!(std::env::var("PROJECT_NAME").unwrap(), "demo");
    assert_eq!(std::env::var("RUN_ID").unwrap(), "train");
    assert_eq!(std::env::var("ATTEMPT_ID").unwrap(), "attempt-001");
    if let Ok(expected) = std::env::var("EXPECTED_CUDA") {
        assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").unwrap(), expected);
    }
    assert_eq!(
        std::env::var("FROZEN_PARAMETER").unwrap(),
        "scientific-value"
    );
    let output = PathBuf::from(std::env::var_os("OUTPUT_DIR").unwrap());
    let mut counter = OpenOptions::new()
        .create(true)
        .append(true)
        .open("executions")
        .unwrap();
    writeln!(counter, "one").unwrap();
    counter.sync_all().unwrap();
    fs::write(output.join("weights.bin"), b"synthetic model weights").unwrap();
    println!("SEALED_NATIVE_TRAINING=READY");
    match std::env::var("FIXTURE_MODE").unwrap().as_str() {
        "pause" => thread::sleep(Duration::from_millis(300)),
        "timeout" => thread::sleep(Duration::from_secs(30)),
        "fail" => std::process::exit(7),
        "block-receipt" => symlink(
            "outputs/weights.bin",
            output.parent().unwrap().join("managed-receipt.json"),
        )
        .unwrap(),
        "replace-lock" => {
            let lock = output.parent().unwrap().join(".managed-launch.lock");
            fs::remove_file(&lock).unwrap();
            private_file(&lock, b"replacement lock\n");
        }
        "complete" => {}
        mode @ ("flood" | "flood-and-kill-worker") => {
            let flood = vec![b'x'; 9 * 1024 * 1024];
            let mut out = std::io::stdout();
            out.write_all(&flood).unwrap();
            out.write_all(b"\nFLOOD_END\n").unwrap();
            out.flush().unwrap();
            let mut err = std::io::stderr();
            err.write_all(&flood).unwrap();
            err.write_all(b"\nFLOOD_END\n").unwrap();
            err.flush().unwrap();
            if mode == "flood-and-kill-worker" {
                let log = output.join("stderr.log");
                let started = Instant::now();
                while !fs::read(&log).unwrap().ends_with(b"FLOOD_END\n")
                    && started.elapsed() < Duration::from_secs(3)
                {
                    thread::sleep(Duration::from_millis(10));
                }
                // SAFETY: this sealed fixture's parent is the worker under test.
                unsafe { libc::kill(libc::getppid(), libc::SIGKILL) };
            }
        }
        mode => panic!("unknown synthetic fixture mode: {mode}"),
    }
}

#[test]
fn sealed_training_executes_once_with_explicit_environment_and_durable_receipt() {
    let fixture = Fixture::new("complete");
    assert!(managed::inspect(&fixture.config).is_err());
    assert!(!fixture.config.attempt_root.exists());
    fixture.initialize();
    assert!(managed::initialize(&fixture.config).is_err());
    let result = managed::run(&fixture.config, &AtomicBool::new(false)).unwrap();
    assert_eq!(result.phase, Phase::Finished);
    let receipt = result.receipt.unwrap();
    assert!(receipt.succeeded());
    assert_eq!(receipt.training, Training::Completed);
    assert!(
        serde_json::to_value(&receipt)
            .unwrap()
            .get("publication")
            .is_none()
    );
    assert!(!receipt.download_available);
    assert_eq!(receipt.identity.project, "demo");
    assert_eq!(receipt.identity.run_id, "train");
    assert_eq!(receipt.identity.attempt_id, "attempt-001");
    assert_eq!(fixture.count(), 1);
    for stream in ["stdout", "stderr"] {
        let actual = ml_exp_io::sha256_file(
            &fixture
                .config
                .attempt_root
                .join(format!("outputs/{stream}.log")),
        )
        .unwrap();
        let expected = if stream == "stdout" {
            receipt.process.stdout_sha256.as_ref()
        } else {
            receipt.process.stderr_sha256.as_ref()
        };
        assert_eq!(expected, Some(&actual));
    }
    // Retrieving after a lost successful response never starts training again.
    assert!(
        managed::run(&fixture.config, &AtomicBool::new(false))
            .unwrap()
            .receipt
            .unwrap()
            .succeeded()
    );
    assert_eq!(
        managed::inspect(&fixture.config).unwrap().phase,
        Phase::Finished
    );
    assert_eq!(fixture.count(), 1);
    let outward =
        fs::read_to_string(fixture.config.attempt_root.join("managed-receipt.json")).unwrap();
    assert!(!outward.contains("scientific-value"));
    assert!(!outward.contains(fixture.config.workdir.to_str().unwrap()));
}

/// Raw UTF-8 is the sealed form: only its exact bytes satisfy a binding.
#[test]
fn a_non_ascii_manifest_is_sealed_as_raw_utf8_and_only_those_bytes_bind() {
    let label = "基础\u{2028}模型😀";
    let argument = "--名称=模型😀";
    let root = "/srv/ml-expd-agreement";
    let document = json!({"contract":launch::CONTRACT,
        "identity":{"project":"demo","run_id":"train","attempt_id":"attempt-001"},
        "environment":{"OUTPUT_DIR":format!("{root}/outputs"),"LABEL":label},
        "argv":["python","训练.py",argument],"workdir":format!("{root}/代码"),"timeout_seconds":5});
    let bytes = launch::encode(&document).unwrap();
    let text = format!(
        r#"{{"argv":["python","训练.py","{argument}"],"contract":"launcher-manifest.v1","environment":{{"LABEL":"{label}","OUTPUT_DIR":"{root}/outputs"}},"identity":{{"attempt_id":"attempt-001","project":"demo","run_id":"train"}},"timeout_seconds":5,"workdir":"{root}/代码"}}"#
    );
    assert_eq!(std::str::from_utf8(&bytes).unwrap(), text);
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        sha256,
        "07fcd31bbe571ee13231732db9056fd547ed3c757cd182fab7e7285be844b6ae"
    );
    let binding = |sha256: String| launch::Binding {
        identity: launch::Identity {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
        },
        sha256,
    };
    assert_eq!(
        launch::verify_binding(&bytes, &binding(sha256))
            .unwrap()
            .environment["LABEL"],
        label
    );
    let mut escaped = String::new();
    for c in text.chars() {
        if c.is_ascii() {
            escaped.push(c);
        } else {
            for unit in c.encode_utf16(&mut [0; 2]).iter() {
                escaped.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    let escaped_sha256 = format!("{:x}", Sha256::digest(escaped.as_bytes()));
    assert!(launch::verify_binding(escaped.as_bytes(), &binding(escaped_sha256)).is_err());

    let mut fixture = Fixture::new("normal");
    fixture.document["environment"]["LABEL"] = json!(label);
    fixture.document["argv"]
        .as_array_mut()
        .unwrap()
        .push(json!(argument));
    fixture.seal();
    let sealed = fs::read(&fixture.config.manifest_path).unwrap();
    assert!(std::str::from_utf8(&sealed).unwrap().contains(label));
    fixture.initialize();
    assert_eq!(
        managed::inspect(&fixture.config).unwrap().manifest_sha256,
        format!("{:x}", Sha256::digest(&sealed))
    );
}

#[test]
fn real_process_exit_and_timeout_do_not_become_publication_success() {
    for (mode, training) in [("fail", Training::Failed), ("timeout", Training::TimedOut)] {
        let fixture = Fixture::new(mode);
        fixture.initialize();
        let started = Instant::now();
        let receipt = managed::run(&fixture.config, &AtomicBool::new(false))
            .unwrap()
            .receipt
            .unwrap();
        assert!(started.elapsed() < Duration::from_secs(6));
        assert_eq!(receipt.training, training);
        assert!(!receipt.succeeded());
        assert!(!receipt.download_available);
        if mode == "fail" {
            assert_eq!(receipt.process.exit_code, Some(7));
        } else {
            assert!(receipt.process.timed_out);
        }
        assert_eq!(fixture.count(), 1);
    }
}

#[test]
fn concurrent_launches_lost_responses_and_persisted_start_never_replay() {
    let fixture = Fixture::new("pause");
    fixture.initialize();
    thread::scope(|scope| {
        let first = scope.spawn(|| managed::run(&fixture.config, &AtomicBool::new(false)));
        let deadline = Instant::now() + Duration::from_secs(5);
        while fixture.count() == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(fixture.count(), 1);
        assert!(matches!(
            managed::run(&fixture.config, &AtomicBool::new(false)),
            Err(Error::Busy)
        ));
        assert!(first.join().unwrap().unwrap().receipt.unwrap().succeeded());
    });
    let unstarted = Fixture::new("complete");
    unstarted.initialize();
    let lock = unstarted.config.attempt_root.join(".managed-launch.lock");
    let mut record: Value = serde_json::from_slice(&fs::read(&lock).unwrap()).unwrap();
    record["phase"] = json!("STARTED");
    let mut file = OpenOptions::new().append(true).open(&lock).unwrap();
    file.write_all(&serde_json::to_vec(&record).unwrap())
        .unwrap();
    file.write_all(b"\n").unwrap();
    file.sync_all().unwrap();
    assert_eq!(
        managed::inspect(&unstarted.config).unwrap().phase,
        Phase::Started
    );
    assert!(matches!(
        managed::run(&unstarted.config, &AtomicBool::new(false)),
        Err(Error::Uncertain)
    ));
    assert_eq!(unstarted.count(), 0);
}

#[test]
fn cancellation_before_start_preserves_prepared_state_without_spawning() {
    let fixture = Fixture::new("complete");
    fixture.initialize();
    assert!(matches!(
        managed::run(&fixture.config, &AtomicBool::new(true)),
        Err(Error::Cancelled)
    ));
    assert_eq!(
        managed::inspect(&fixture.config).unwrap().phase,
        Phase::Prepared
    );
    assert!(!fixture.config.attempt_root.join("outputs").exists());
    assert_eq!(fixture.count(), 0);
}

#[test]
fn checksum_identity_path_budget_and_unsupported_features_are_checked_before_start() {
    for variant in [
        "checksum",
        "foreign-identity",
        "workdir",
        "output",
        "budget",
        "reserved-capability",
        "implicit-path",
    ] {
        let mut fixture = Fixture::new("complete");
        match variant {
            "checksum" => private_file(&fixture.config.manifest_path, b"changed"),
            "foreign-identity" => {
                fixture.document["identity"]["attempt_id"] = json!("attempt-002");
                fixture.seal();
            }
            "workdir" => {
                fixture.document["workdir"] = json!("/not-the-frozen-code");
                fixture.seal();
            }
            "output" => {
                fixture.document["environment"]["OUTPUT_DIR"] = json!("/not-the-frozen-output");
                fixture.seal();
            }
            "budget" => {
                fixture.document["timeout_seconds"] = json!(6);
                fixture.seal();
            }
            "reserved-capability" => {
                fixture.document["environment"]["ML_EXPD_INPUT_ASSETS"] = json!("[]");
                fixture.seal();
            }
            "implicit-path" => {
                fixture.document["argv"] = json!(["not-explicitly-resolved"]);
                fixture.seal();
            }
            _ => unreachable!(),
        }
        assert!(managed::initialize(&fixture.config).is_err(), "{variant}");
        assert!(!fixture.config.attempt_root.exists(), "{variant}");
        assert_eq!(fixture.count(), 0);
    }
}

#[test]
fn damaged_missing_foreign_and_aliased_state_are_not_repaired_or_replayed() {
    for variant in [
        "missing-lock",
        "partial-start",
        "foreign-config",
        "manifest-symlink",
        "workdir-replaced",
    ] {
        let mut fixture = Fixture::new("complete");
        fixture.initialize();
        let lock = fixture.config.attempt_root.join(".managed-launch.lock");
        match variant {
            "missing-lock" => fs::remove_file(&lock).unwrap(),
            "partial-start" => {
                OpenOptions::new()
                    .append(true)
                    .open(&lock)
                    .unwrap()
                    .write_all(b"{\"phase\":\"START")
                    .unwrap();
            }
            "foreign-config" => fixture.config.maximum_timeout_seconds = 4,
            "manifest-symlink" => {
                let manifest = fixture.config.attempt_root.join("launch-manifest.json");
                fs::remove_file(&manifest).unwrap();
                symlink(&fixture.config.manifest_path, &manifest).unwrap();
            }
            "workdir-replaced" => {
                fs::remove_dir(&fixture.config.workdir).unwrap();
                symlink(fixture._root.path(), &fixture.config.workdir).unwrap();
            }
            _ => unreachable!(),
        }
        assert!(
            managed::run(&fixture.config, &AtomicBool::new(false)).is_err(),
            "{variant}"
        );
        assert!(
            !fixture.config.attempt_root.join("outputs").exists(),
            "{variant}"
        );
        if variant == "missing-lock" {
            assert!(!lock.exists());
        }
    }
}

#[test]
fn post_execution_storage_failure_keeps_started_unknown_and_prevents_replay() {
    for mode in ["block-receipt", "replace-lock"] {
        let fixture = Fixture::new(mode);
        fixture.initialize();
        assert!(matches!(
            managed::run(&fixture.config, &AtomicBool::new(false)),
            Err(Error::Uncertain)
        ));
        assert_eq!(fixture.count(), 1);
        assert!(managed::run(&fixture.config, &AtomicBool::new(false)).is_err());
        assert_eq!(fixture.count(), 1);
        if mode == "block-receipt" {
            let result = managed::inspect(&fixture.config).unwrap();
            assert_eq!(result.phase, Phase::Started);
            assert!(result.receipt.is_none());
        }
    }
    let fixture = Fixture::new("complete");
    fixture.initialize();
    managed::run(&fixture.config, &AtomicBool::new(false)).unwrap();
    private_file(
        &fixture.config.attempt_root.join("managed-receipt.json"),
        b"corrupt",
    );
    assert!(managed::inspect(&fixture.config).is_err());
    assert!(managed::run(&fixture.config, &AtomicBool::new(false)).is_err());
    assert_eq!(fixture.count(), 1);
}

#[test]
fn a_flooded_log_exits_successfully_and_the_receipt_records_what_it_dropped() {
    let fixture = Fixture::new("flood");
    fixture.initialize();
    let executed = fixture.cli(None);
    assert!(
        executed.status.success(),
        "{}",
        String::from_utf8_lossy(&executed.stderr)
    );
    let reply: Value = serde_json::from_slice(&executed.stdout).unwrap();
    assert_eq!(reply["receipt"]["training"], "COMPLETED");
    let process = &reply["receipt"]["process"];
    assert!(process["error"].is_null());
    let seam = ml_exp_io::process::SEAM;
    for stream in ["stdout", "stderr"] {
        let log = fs::read(
            fixture
                .config
                .attempt_root
                .join(format!("outputs/{stream}.log")),
        )
        .unwrap();
        assert!(log.len() <= 8 * 1024 * 1024);
        assert_eq!(log[4 * 1024 * 1024..][..seam.len()], *seam);
        assert!(
            log[log.len() - 4096..]
                .windows(b"FLOOD_END".len())
                .any(|bytes| bytes == b"FLOOD_END")
        );
        assert_eq!(
            process[format!("{stream}_sha256").as_str()],
            format!("{:x}", Sha256::digest(&log))
        );
        assert!(process[format!("{stream}_dropped_bytes").as_str()].as_u64() > Some(0));
    }
    assert_eq!(fixture.count(), 1);
}

#[test]
fn killing_the_worker_leaves_the_end_of_the_log_on_disk() {
    let fixture = Fixture::new("flood-and-kill-worker");
    fixture.initialize();
    let executed = fixture.cli(None);
    assert!(!executed.status.success());
    let log = fs::read(fixture.config.attempt_root.join("outputs/stderr.log")).unwrap();
    assert!(log.len() <= 8 * 1024 * 1024);
    assert!(log.ends_with(b"FLOOD_END\n"));
    let seam = ml_exp_io::process::SEAM;
    assert_eq!(log[4 * 1024 * 1024..][..seam.len()], *seam);
    assert!(
        !fixture
            .config
            .attempt_root
            .join("managed-receipt.json")
            .exists()
    );
}

#[test]
fn operator_cli_only_loads_private_config_and_can_initialize_run_and_inspect() {
    let fixture = Fixture::new("complete");
    let initialized = fixture.cli(Some("--initialize"));
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&initialized.stdout).unwrap()["phase"],
        "PREPARED"
    );
    let executed = fixture.cli(None);
    assert!(
        executed.status.success(),
        "{}",
        String::from_utf8_lossy(&executed.stderr)
    );
    let reply: Value = serde_json::from_slice(&executed.stdout).unwrap();
    assert_eq!(reply["phase"], "FINISHED");
    assert_eq!(reply["receipt"]["training"], "COMPLETED");
    assert_eq!(reply["receipt"]["download_available"], false);
    assert!(fixture.cli(Some("--inspect")).status.success());
    assert!(fixture.cli(None).status.success());
    assert_eq!(fixture.count(), 1);
    let override_attempt = Command::new(env!("CARGO_BIN_EXE_ml-exp-worker-rs"))
        .env_clear()
        .args(["managed-launch", "--config"])
        .arg(&fixture.config_file)
        .args(["--command", "/bin/true"])
        .output()
        .unwrap();
    assert!(!override_attempt.status.success());
    fs::set_permissions(&fixture.config_file, fs::Permissions::from_mode(0o644)).unwrap();
    let exposed_config = fixture.cli(None);
    assert!(!exposed_config.status.success());
    assert!(!String::from_utf8_lossy(&exposed_config.stderr).contains("scientific-value"));
    assert_eq!(fixture.count(), 1);
}

#[test]
fn local_sealed_launch_bootstraps_once_and_freezes_only_approved_runtime_environment() {
    let mut f = Fixture::new("complete");
    let shared = f._root.path().join("shared-attempt");
    private_directory(&shared);
    let outputs = shared.join("outputs");
    f.config.output_directory = Some(outputs.clone());
    f.config.runtime_environment = vec!["CUDA_VISIBLE_DEVICES".into()];
    f.document["environment"]["OUTPUT_DIR"] = json!(f.config.output_path());
    f.document["environment"]["EXPECTED_CUDA"] = json!("3,5");
    f.seal();
    let target = launch::endpoint(&f.config.endpoint).unwrap();
    f.config.endpoint.clear();
    f.config.local_binding = Some(launch::Binding {
        identity: target.identity,
        sha256: target.sha256,
    });
    let bytes = serde_json::to_vec(&f.config).unwrap();
    private_file(&f.config_file, &bytes);
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let result = Command::new(env!("CARGO_BIN_EXE_ml-exp-worker-rs"))
        .env_clear()
        .env("CUDA_VISIBLE_DEVICES", "3,5")
        .env("ML_EXPD_BOOTSTRAP_TOKEN", "must-not-reach-child")
        .args(["managed-launch", "--config"])
        .arg(&f.config_file)
        .args(["--sha256", &hash, "--initialize-and-run"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let observation: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(observation["phase"], "FINISHED");
    assert_eq!(observation["receipt"]["training"], "COMPLETED");
    assert!(observation["receipt"]["runtime_environment_sha256"].is_string());
    assert_eq!(f.count(), 1);
    assert_eq!(
        fs::read(outputs.join("weights.bin")).unwrap(),
        b"synthetic model weights"
    );
    assert!(!f.config.attempt_root.join("outputs").exists());
    let context = f.config.attempt_root.join("runtime-environment.json");
    let saved: Value = serde_json::from_slice(&fs::read(&context).unwrap()).unwrap();
    assert_eq!(saved, json!({"CUDA_VISIBLE_DEVICES":"3,5"}));
    assert!(!f.cli(Some("--initialize-and-run")).status.success());
    assert_eq!(f.count(), 1);
    assert!(f.cli(Some("--inspect")).status.success());
    private_file(&context, b"{\"CUDA_VISIBLE_DEVICES\":\"99\"}");
    assert!(managed::inspect(&f.config).is_err());
    assert_eq!(f.count(), 1);
}

#[test]
fn staged_config_digest_and_existing_external_outputs_fail_before_payload_execution() {
    let f = Fixture::new("complete");
    let result = Command::new(env!("CARGO_BIN_EXE_ml-exp-worker-rs"))
        .env_clear()
        .args(["managed-launch", "--config"])
        .arg(&f.config_file)
        .args(["--sha256", &"0".repeat(64), "--initialize-and-run"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!f.config.attempt_root.exists());
    assert_eq!(f.count(), 0);
    let mut f = Fixture::new("complete");
    let outputs = f._root.path().join("preexisting-outputs");
    private_directory(&outputs);
    private_file(&outputs.join("keep"), b"existing result must survive");
    f.config.output_directory = Some(outputs.clone());
    f.document["environment"]["OUTPUT_DIR"] = json!(outputs);
    f.seal();
    f.initialize();
    assert!(managed::run(&f.config, &AtomicBool::new(false)).is_err());
    assert_eq!(
        fs::read(outputs.join("keep")).unwrap(),
        b"existing result must survive"
    );
    assert_eq!(f.count(), 0);
}

#[test]
fn local_and_endpoint_bindings_are_exclusive_and_unset_fields_stay_out_of_the_configuration_bytes()
{
    let f = Fixture::new("complete");
    let old = json!({"manifest_path":f.config.manifest_path,"endpoint":f.config.endpoint,"attempt_root":f.config.attempt_root,
        "workdir":f.config.workdir,"maximum_timeout_seconds":5,"base_environment":{}});
    assert_eq!(serde_json::to_value(&f.config).unwrap(), old);
    assert_eq!(
        f.config.digest().unwrap(),
        ml_exp_core::canonical::digest(&old).unwrap()
    );
    let mut config = f.config.clone();
    let target = launch::endpoint(&config.endpoint).unwrap();
    config.local_binding = Some(launch::Binding {
        identity: target.identity,
        sha256: target.sha256,
    });
    assert!(config.validate().is_err());
    config.endpoint.clear();
    assert!(config.validate().is_ok());
    config.runtime_environment = vec!["HOME".into()];
    assert!(config.validate().is_err());
    config.runtime_environment = vec!["CUDA_VISIBLE_DEVICES".into()];
    config
        .base_environment
        .insert("CUDA_VISIBLE_DEVICES".into(), "1".into());
    assert!(config.validate().is_err());
    config.base_environment.clear();
    assert!(
        config
            .validate_runtime(&BTreeMap::from([(
                "ML_EXPD_API_TOKEN".into(),
                "not-a-real-token".into()
            )]))
            .is_err()
    );
}

#[test]
fn readonly_sampling_during_execution_never_unlocks_or_replays_the_process() {
    let mut fixture = Fixture::new("success");
    fixture.document["argv"] = json!([
        "/bin/sh",
        "-c",
        "printf live; while ! test -f \"$OUTPUT_DIR/release\"; do sleep 0.01; done; printf done"
    ]);
    fixture.seal();
    fixture.initialize();
    let config = fixture.config.clone();
    let child = thread::spawn(move || managed::run(&config, &AtomicBool::new(false)).unwrap());
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if fixture.config.output_path().join("stdout.log").exists() {
            break;
        }
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    let sample = managed::sample(&fixture.config).unwrap();
    assert_eq!(sample.phase, Phase::Started);
    assert!(sample.receipt.is_none());
    assert!(!sample.safe_to_retry);
    assert!(matches!(
        managed::inspect(&fixture.config),
        Err(Error::Busy)
    ));
    assert!(matches!(
        managed::run(&fixture.config, &AtomicBool::new(false)),
        Err(Error::Busy)
    ));
    fs::write(fixture.config.output_path().join("release"), b"").unwrap();
    let completed = child.join().unwrap();
    assert_eq!(completed.phase, Phase::Finished);
    assert_eq!(managed::sample(&fixture.config).unwrap(), completed);
}
