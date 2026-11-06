//! Real executable deployment setup; all files and credentials are synthetic.
//! Local preflight must not initialize stores, execute an SDK or call a provider.
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

const SERVER: &str = env!("CARGO_BIN_EXE_ml-expd-rs");
const TOKEN: &str = "synthetic-native-deployment-token-00000000000000";
const RETIRED_MARKER: &str = ".rust-preview-workspace.json";
const RESTART_STAGE: &str = "registry_predates_clean_restart";
fn write(path: &Path, value: &Value) {
    ml_exp_io::write_atomic(path, &serde_json::to_vec(value).unwrap(), false).unwrap();
}
struct Fixture {
    temp: tempfile::TempDir,
    config: PathBuf,
    reads: PathBuf,
    registry: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::create_dir(root.join("runs")).unwrap();
        write(
            &root.join("project.json"),
            &json!({"schema_version":1,"project":"demo","title":"Owned fixture","run_roots":["runs"]}),
        );
        ml_exp_io::write_atomic(&root.join("token"), TOKEN.as_bytes(), true).unwrap();
        let reads = root.join("reads.json");
        let registry = root.join("registry");
        write(
            &reads,
            &json!({"schema_version":1,"registry_root":registry,"projects":[{"project":"demo","project_file":root.join("project.json"),"run_roots":[root.join("runs")]}]}),
        );
        let config = root.join("server.json");
        write(
            &config,
            &json!({"schema_version":1,"bind":"127.0.0.1:0","token_file":root.join("token"),"read_config":reads,"project_lifecycle":true,"source_archives":true,"administration":true}),
        );
        Self {
            temp,
            config,
            reads,
            registry,
        }
    }
    fn command(&self, operation: &str) -> Output {
        Command::new(SERVER)
            .arg("--config")
            .arg(&self.config)
            .arg(operation)
            .output()
            .unwrap()
    }
    fn startup(&self) -> Output {
        exited(Command::new(SERVER).arg("--config").arg(&self.config))
    }
    fn value(&self) -> Value {
        serde_json::from_slice(&fs::read(&self.config).unwrap()).unwrap()
    }
    fn old_registry(&self) {
        fs::create_dir(&self.registry).unwrap();
        fs::write(self.registry.join(RETIRED_MARKER), b"\xff\x00 not json").unwrap();
    }
}
/// Run a command that must refuse by itself, failing rather than hanging if it serves.
fn exited(command: &mut Command) -> Output {
    let mut child = command
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the command kept running instead of refusing");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    child.wait_with_output().unwrap()
}
fn refused(output: &Output, stage: &str) {
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{error}");
    assert!(error.contains(stage), "{error}");
}
fn tree(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn visit(root: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                visit(&entry.path(), out);
            } else {
                out.insert(entry.path(), fs::read(entry.path()).unwrap());
            }
        }
    }
    let mut result = BTreeMap::new();
    visit(root, &mut result);
    result
}
fn value(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
struct Running(Child);
impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn start(config: &Path) -> (Running, String) {
    let mut child = Running(
        Command::new(SERVER)
            .arg("--config")
            .arg(config)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let stdout = child.0.stdout.take().unwrap();
    let (send, receive) = mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = BufReader::new(stdout).read_line(&mut line);
        let _ = send.send(line);
    });
    let address = receive.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(!address.trim().is_empty(), "daemon exited before listening");
    (child, format!("http://{}", address.trim()))
}

#[test]
fn single_config_initializes_once_then_supports_restart_without_reinitializing() {
    let fixture = Fixture::new();
    let before = tree(fixture.temp.path());
    let report = value(fixture.command("--check"));
    assert_eq!(report["configuration_valid"], true);
    assert_eq!(report["provider_connectivity_checked"], false);
    assert_eq!(report["registry_initialized"], false);
    assert_eq!(before, tree(fixture.temp.path()));
    let initialized = value(fixture.command("--initialize"));
    assert_eq!(initialized["provider_jobs_created"], 0);
    assert_eq!(
        initialized["stores"],
        json!([
            "registry",
            "project_lifecycle",
            "source_archives",
            "administration"
        ])
    );
    let after = tree(fixture.temp.path());
    assert!(!fixture.command("--initialize").status.success());
    assert_eq!(after, tree(fixture.temp.path()));
    assert_eq!(
        value(fixture.command("--check"))["registry_initialized"],
        true
    );
    assert_eq!(after, tree(fixture.temp.path()));
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    for _ in 0..2 {
        let (_server, url) = start(&fixture.config);
        let result = client
            .get(format!("{url}/api/projects"))
            .bearer_auth(TOKEN)
            .send()
            .unwrap();
        assert!(result.status().is_success());
        let body: Value = serde_json::from_slice(&result.bytes().unwrap()).unwrap();
        assert_eq!(body[0]["project"], "demo");
    }
}

/// Every worker fetches its package through the exchange, so this refuses before
/// any Run or outbox entry exists. A server that creates nothing may still omit it.
#[test]
fn run_creation_without_an_exchange_refuses_at_check_and_read_only_stays_valid() {
    let fixture = Fixture::new();
    let root = fixture.temp.path();
    let mut cloud: Value =
        serde_json::from_str(include_str!("../../../deploy/pai.example.json")).unwrap();
    cloud["backends"][0]["credentials_file"] = json!(root.join("credentials.json"));
    cloud["backends"][0]["registry_auth_file"] = Value::Null;
    cloud.as_object_mut().unwrap().remove("exchange");
    cloud.as_object_mut().unwrap().remove("artifacts");
    cloud["allow_run_creation"] = json!(true);
    write(
        &root.join("credentials.json"),
        &json!({"provider":"pai","access_key_id":"owned-fixture","access_key_secret":"synthetic-sensitive-value"}),
    );
    write(&root.join("cloud.json"), &cloud);
    let mut config = fixture.value();
    config["cloud_config"] = json!(root.join("cloud.json"));
    write(&fixture.config, &config);
    let before = tree(root);
    for mode in ["--check", "--initialize"] {
        refused(
            &fixture.command(mode),
            "cloud_run_creation_requires_exchange",
        );
        assert!(!fixture.registry.exists());
        assert_eq!(before, tree(root));
    }
    cloud["allow_run_creation"] = json!(false);
    write(&root.join("cloud.json"), &cloud);
    assert_eq!(
        value(fixture.command("--check"))["configuration_valid"],
        true
    );
}

#[test]
fn invalid_private_dependencies_block_bootstrap_before_creating_any_namespace() {
    let fixture = Fixture::new();
    let root = fixture.temp.path();
    let mut cloud: Value =
        serde_json::from_str(include_str!("../../../deploy/pai.example.json")).unwrap();
    cloud["backends"][0]["credentials_file"] = json!(root.join("credentials.json"));
    write(
        &root.join("credentials.json"),
        &json!({"provider":"sensecore","access_key_id":"owned-fixture","access_key_secret":"synthetic-sensitive-value"}),
    );
    write(&root.join("cloud.json"), &cloud);
    let mut config = fixture.value();
    config["cloud_config"] = json!(root.join("cloud.json"));
    write(&fixture.config, &config);
    let before = tree(root);
    for mode in ["--check", "--initialize"] {
        let result = fixture.command(mode);
        assert!(!result.status.success());
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(error.contains("cloud_provider_credentials"));
        assert!(!error.contains("synthetic-sensitive-value"));
        assert!(!fixture.registry.exists());
        assert_eq!(before, tree(root));
    }
    // A valid cloud configuration constructs local transport clients, but must
    // never contact a provider or require the builder to be listening.
    write(
        &root.join("credentials.json"),
        &json!({"provider":"pai","access_key_id":"owned-fixture","access_key_secret":"synthetic-sensitive-value"}),
    );
    cloud["backends"][0]["registry_auth_file"] = Value::Null;
    cloud.as_object_mut().unwrap().remove("exchange");
    cloud.as_object_mut().unwrap().remove("artifacts");
    write(&root.join("cloud.json"), &cloud);
    let before = tree(root);
    assert_eq!(
        value(fixture.command("--check"))["provider_connectivity_checked"],
        false
    );
    assert_eq!(before, tree(root));
    let initialized = value(fixture.command("--initialize"));
    assert!(
        initialized["stores"]
            .as_array()
            .unwrap()
            .contains(&json!("cloud"))
    );
    let (_server, _) = start(&fixture.config);
    // A check must not execute the configured SDK, including entity lookup.
    config.as_object_mut().unwrap().remove("cloud_config");
    config["tracking_config"] = json!(root.join("tracking.json"));
    write(
        &root.join("tracking.json"),
        &json!({"schema_version":1,"python":"/bin/false","api_key_file":root.join("wandb-key"),"enabled":true}),
    );
    ml_exp_io::write_atomic(
        &root.join("wandb-key"),
        b"0123456789012345678901234567890123456789",
        true,
    )
    .unwrap();
    write(&fixture.config, &config);
    let before = tree(root);
    assert_eq!(
        value(fixture.command("--check"))["configuration_valid"],
        true
    );
    assert_eq!(before, tree(root));
}

#[test]
fn existing_foreign_registry_and_invalid_project_are_never_initialized_or_erased() {
    let fixture = Fixture::new();
    fs::create_dir(&fixture.registry).unwrap();
    fs::write(
        fixture.registry.join("original-state"),
        b"must remain untouched",
    )
    .unwrap();
    let before = tree(fixture.temp.path());
    for mode in ["--check", "--initialize"] {
        let result = fixture.command(mode);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("existing_non_native_registry"));
        assert_eq!(before, tree(fixture.temp.path()));
    }
    let mut reads: Value = serde_json::from_slice(&fs::read(&fixture.reads).unwrap()).unwrap();
    reads["registry_root"] = json!(fixture.temp.path().join("new-registry"));
    write(&fixture.reads, &reads);
    write(
        &fixture.temp.path().join("project.json"),
        &json!({"schema_version":1,"project":"another-project","title":"mismatch"}),
    );
    let result = fixture.command("--initialize");
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("project_definition"));
    assert!(!fixture.temp.path().join("new-registry").exists());
}

#[test]
fn retired_marker_refuses_every_entry_point_without_being_parsed_or_changed() {
    let fixture = Fixture::new();
    fixture.old_registry();
    let before = tree(fixture.temp.path());
    for result in [
        fixture.command("--check"),
        fixture.command("--initialize"),
        fixture.startup(),
        exited(
            Command::new(SERVER)
                .arg("--read-config")
                .arg(&fixture.reads)
                .arg("--initialize-source-archives"),
        ),
        exited(
            Command::new(SERVER)
                .args(["--bind", "127.0.0.1:0", "--token-file"])
                .arg(fixture.temp.path().join("token"))
                .arg("--read-config")
                .arg(&fixture.reads),
        ),
    ] {
        refused(&result, RESTART_STAGE);
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(error.contains("set registry_root to a new path and run --initialize"));
        assert!(error.contains("nothing was read or changed"));
        assert_eq!(before, tree(fixture.temp.path()));
    }
}

#[test]
fn retired_marker_wins_over_unrelated_configuration_errors() {
    let fixture = Fixture::new();
    let root = fixture.temp.path();
    fixture.old_registry();
    fs::remove_file(root.join("token")).unwrap();
    write(
        &root.join("project.json"),
        &json!({"schema_version":1,"project":"another-project","title":"mismatch"}),
    );
    let mut config = fixture.value();
    config["bind"] = json!("0.0.0.0:18765");
    config["cloud_config"] = json!(root.join("absent-cloud.json"));
    config["tracking_config"] = json!(root.join("absent-tracking.json"));
    write(&fixture.config, &config);
    let before = tree(root);
    for result in [
        fixture.command("--check"),
        fixture.command("--initialize"),
        fixture.startup(),
    ] {
        refused(&result, RESTART_STAGE);
        assert_eq!(before, tree(root));
    }
}

#[test]
fn absent_registry_is_created_with_the_new_marker_and_an_altered_marker_is_refused() {
    let fixture = Fixture::new();
    let root = fixture.temp.path();
    let check = value(fixture.command("--check"));
    assert_eq!(check["registry_initialized"], false);
    assert!(!fixture.registry.exists());
    let initialized = value(fixture.command("--initialize"));
    let id = initialized["workspace_id"].as_str().unwrap();
    assert_eq!(id.len(), 32);
    assert!(id.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')));
    let marker = fixture.registry.join(".ml-expd-workspace.json");
    let written: Value = serde_json::from_slice(&fs::read(&marker).unwrap()).unwrap();
    assert_eq!(
        written,
        json!({"contract":"ml-expd-registry.v1","workspace_id":id})
    );
    assert!(!fixture.registry.join(RETIRED_MARKER).exists());
    assert_eq!(
        value(fixture.command("--check"))["registry_initialized"],
        true
    );
    for altered in [
        json!({"contract":"ml-expd-rust-preview.v1","workspace_id":id}),
        json!({"contract":"ml-expd-registry.v1","workspace_id":format!("rust-preview-{id}")}),
    ] {
        write(&marker, &altered);
        let before = tree(root);
        for result in [
            fixture.command("--check"),
            fixture.command("--initialize"),
            fixture.startup(),
        ] {
            refused(&result, "existing_non_native_registry");
            assert_eq!(before, tree(root));
        }
    }
}

#[test]
fn initialize_refuses_an_occupied_run_root_and_leaves_no_registry() {
    let fixture = Fixture::new();
    let root = fixture.temp.path();
    fs::create_dir(root.join("runs/old-run")).unwrap();
    fs::write(root.join("runs/old-run/manifest.yaml"), b"\xff not yaml").unwrap();
    let before = tree(root);
    refused(
        &fixture.command("--initialize"),
        "project_run_root_not_empty",
    );
    assert!(!fixture.registry.exists());
    assert_eq!(before, tree(root));
}
