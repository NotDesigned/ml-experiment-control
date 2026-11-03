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
    time::Duration,
};

const SERVER: &str = env!("CARGO_BIN_EXE_ml-expd-rs");
const TOKEN: &str = "synthetic-native-deployment-token-00000000000000";
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
            &json!({"schema_version":1,"project":"demo","title":"Owned fixture","run_roots":["runs"],"campaigns":[]}),
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
    fn value(&self) -> Value {
        serde_json::from_slice(&fs::read(&self.config).unwrap()).unwrap()
    }
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
            .stderr(Stdio::piped())
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
            .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2")
            .send()
            .unwrap();
        assert!(result.status().is_success());
        let body: Value = serde_json::from_slice(&result.bytes().unwrap()).unwrap();
        assert_eq!(body[0]["project"], "demo");
    }
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
