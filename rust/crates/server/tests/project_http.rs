//! Native server CLI startup/restart plus official client lifecycle requests.
//! Every file, identity and credential is synthetic and local; no provider runs.
use ml_exp_client::{
    Client, Error as ClientError,
    projects::{self, Action, Command as ProjectCommand},
};
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::Duration,
};

const TOKEN: &str = "synthetic-native-lifecycle-token-00000000000000000";
const SERVER: &str = env!("CARGO_BIN_EXE_ml-expd-rs");
fn private_json(path: &Path, value: &Value) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    ml_exp_io::write_atomic(path, &serde_json::to_vec(value).unwrap(), true).unwrap();
}
struct Fixture {
    temp: tempfile::TempDir,
    config: PathBuf,
    token: PathBuf,
    project: PathBuf,
    source_files: Vec<PathBuf>,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        let initialized = Command::new(SERVER)
            .args(["--initialize-registry"])
            .arg(&registry)
            .output()
            .unwrap();
        assert!(
            initialized.status.success(),
            "{}",
            String::from_utf8_lossy(&initialized.stderr)
        );
        let project = temp.path().join("demo/research_project.yaml");
        let runs = temp.path().join("demo/runs");
        let run = runs.join("state/instance-1/campaign-1/trial");
        let manifest = run.join("manifest.json");
        let attempt = run.join("attempts/attempt-001/attempt.json");
        let status = run.join("status.json");
        private_json(
            &project,
            &json!({"schema_version":1,"project":"demo","title":"Native synthetic science","run_roots":["runs"],"campaigns":[]}),
        );
        private_json(
            &manifest,
            &json!({"schema_version":1,"project":"demo","run_id":"trial"}),
        );
        private_json(
            &attempt,
            &json!({"project":"demo","run_id":"trial","attempt_id":"attempt-001"}),
        );
        private_json(
            &status,
            &json!({"attempt_id":"attempt-001","state":"QUEUED"}),
        );
        private_json(
            &run.join("attempts/attempt-001/status.json"),
            &json!({"state":"QUEUED"}),
        );
        std::os::unix::fs::symlink(
            "/never-read-project-list-log",
            run.join("attempts/attempt-001/stdout.log"),
        )
        .unwrap();
        let config = temp.path().join("reads.json");
        // No per-Run locators: discovery must resolve the namespace layout.
        private_json(
            &config,
            &json!({"schema_version":1,"registry_root":registry,"projects":[{"project":"demo","project_file":project,"run_roots":[runs]}]}),
        );
        let token = temp.path().join("token");
        ml_exp_io::write_atomic(&token, TOKEN.as_bytes(), true).unwrap();
        Self {
            temp,
            config,
            token,
            project,
            source_files: vec![manifest, attempt, status],
        }
    }
    fn initialize_projects(&self) {
        let result = Command::new(SERVER)
            .arg("--read-config")
            .arg(&self.config)
            .arg("--initialize-project-lifecycle")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    fn science_bytes(&self) -> Vec<Vec<u8>> {
        std::iter::once(&self.project)
            .chain(self.source_files.iter())
            .map(|path| fs::read(path).unwrap())
            .collect()
    }
    fn ledger_bytes(&self) -> (Vec<u8>, Vec<u8>) {
        let root = self.temp.path().join("registry/projects");
        (
            fs::read(root.join("registry.json")).unwrap(),
            fs::read(root.join("events.jsonl")).unwrap(),
        )
    }
}
struct Server {
    child: Child,
    address: SocketAddr,
}
impl Server {
    fn start(fixture: &Fixture, lifecycle: bool) -> Self {
        let mut command = Command::new(SERVER);
        command
            .arg("--bind")
            .arg("127.0.0.1:0")
            .arg("--token-file")
            .arg(&fixture.token)
            .arg("--read-config")
            .arg(&fixture.config)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if lifecycle {
            command.arg("--project-lifecycle");
        }
        let child = command.spawn().unwrap();
        let mut server = Self {
            child,
            address: "127.0.0.1:0".parse().unwrap(),
        };
        let stdout = server.child.stdout.take().unwrap();
        let (sent, received) = mpsc::channel();
        std::thread::spawn(move || {
            let mut address = String::new();
            let result = BufReader::new(stdout)
                .take(1024)
                .read_line(&mut address)
                .map(|_| address);
            let _ = sent.send(result);
        });
        server.address = received
            .recv_timeout(Duration::from_secs(10))
            .expect("native preview startup is bounded")
            .unwrap()
            .trim()
            .parse()
            .expect("native server returned its listening address");
        server
    }
    fn client(&self) -> Client {
        Client::new(&format!("http://{}", self.address), TOKEN.into()).unwrap()
    }
    fn post(&self, path: &str, body: &[u8]) -> (u16, Value) {
        let mut stream = TcpStream::connect(self.address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(stream,"POST {path} HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {TOKEN}\r\nX-ML-Expd-Client-Protocol: 2\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",self.address,body.len()).unwrap();
        stream.write_all(body).unwrap();
        let mut bytes = Vec::new();
        stream
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .unwrap();
        assert!(bytes.len() <= 8 * 1024 * 1024);
        let split = bytes.windows(4).position(|b| b == b"\r\n\r\n").unwrap();
        let status = std::str::from_utf8(&bytes[..split])
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes[split + 4..]).unwrap();
        assert!(!value.to_string().contains(TOKEN));
        (status, value)
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn send(client: &Client, health: &Value, command: ProjectCommand<'_>) -> Value {
    projects::execute(client, &projects::plan(command).unwrap(), health).unwrap()
}
fn lifecycle(client: &Client, health: &Value) -> Value {
    send(client, health, ProjectCommand::List)
}
fn revision(client: &Client, health: &Value) -> u64 {
    lifecycle(client, health)["durability"]["revision"]
        .as_u64()
        .unwrap()
}

#[test]
fn native_cli_lifecycle_and_auto_discovered_reads_survive_restart_without_resurrection() {
    let fixture = Fixture::new();
    let science = fixture.science_bytes();
    fixture.initialize_projects();
    let server = Server::start(&fixture, true);
    let client = server.client();
    let health = client.negotiate().unwrap();
    assert!(
        health["capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("project-lifecycle.v1"))
    );
    assert_eq!(health["project_writes"], false);
    assert_eq!(health["scheduler_mutations"], false);
    let rows = client.get("/api/projects").unwrap();
    let rows = rows.as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["project"], "demo");
    assert_eq!(
        rows[0]["run_counts"]
            .as_object()
            .unwrap()
            .values()
            .map(|value| value.as_u64().unwrap())
            .sum::<u64>(),
        1
    );
    assert_eq!(
        client.get("/api/runs/demo/trial").unwrap()["run_id"],
        "trial"
    );
    let initial = fixture.ledger_bytes();
    let _ = lifecycle(&client, &health);
    assert_eq!(fixture.ledger_bytes(), initial);
    assert_eq!(
        server
            .post(
                "/api/project-lifecycle/demo/pause",
                &vec![b'x'; 64 * 1024 + 1]
            )
            .0,
        413
    );
    assert_eq!(
        server
            .post(
                "/api/project-lifecycle/demo/pause",
                b"{\"reason\":\"one\",\"reason\":\"two\"}"
            )
            .0,
        422
    );
    assert_eq!(fixture.ledger_bytes(), initial);
    for (action, reason, active) in [
        (Action::Pause, "", false),
        (Action::Resume, "", true),
        (Action::Archive, "finished", false),
        (Action::Restore, "", false),
        (Action::Resume, "", true),
    ] {
        let current = revision(&client, &health);
        let result = send(
            &client,
            &health,
            ProjectCommand::Change {
                project: "demo",
                action,
                reason,
                expected_revision: Some(current),
            },
        );
        assert_eq!(result["active"], active);
        assert_eq!(client.negotiate().unwrap()["projects"], usize::from(active));
    }
    let current = revision(&client, &health);
    let before = fixture.ledger_bytes();
    let stale = projects::plan(ProjectCommand::Change {
        project: "demo",
        action: Action::Pause,
        reason: "",
        expected_revision: Some(current - 1),
    })
    .unwrap();
    assert!(
        matches!(projects::execute(&client,&stale,&health),Err(ClientError::Http(failure)) if failure.status==409)
    );
    assert_eq!(fixture.ledger_bytes(), before);
    let _ = send(
        &client,
        &health,
        ProjectCommand::Change {
            project: "demo",
            action: Action::Unregister,
            reason: "stop tracking",
            expected_revision: Some(current),
        },
    );
    assert!(
        lifecycle(&client, &health)["projects"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    drop(server);
    let server = Server::start(&fixture, true);
    let client = server.client();
    let health = client.negotiate().unwrap();
    assert_eq!(health["projects"], 0);
    assert!(
        lifecycle(&client, &health)["projects"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(client.get("/api/projects").unwrap(), json!([]));
    let current = revision(&client, &health);
    let result = send(
        &client,
        &health,
        ProjectCommand::Register {
            manifest_path: fixture.project.to_str().unwrap(),
            expected_revision: Some(current),
        },
    );
    assert_eq!(result["project"]["state"], "ACTIVE");
    let current = revision(&client, &health);
    let _ = send(
        &client,
        &health,
        ProjectCommand::UnregisterAll {
            reason: "finish synthetic test",
            expected_revision: Some(current),
        },
    );
    assert!(
        lifecycle(&client, &health)["projects"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(fixture.science_bytes(), science);
    assert!(!fixture.temp.path().join("registry/tracking").exists());
}

#[test]
fn existing_lifecycle_ledger_is_not_implicitly_enabled_without_the_cli_flag() {
    let fixture = Fixture::new();
    fixture.initialize_projects();
    let before = fixture.ledger_bytes();
    let science = fixture.science_bytes();
    let server = Server::start(&fixture, false);
    let client = server.client();
    let health = client.negotiate().unwrap();
    assert!(
        !health["capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("project-lifecycle.v1"))
    );
    let plan = projects::plan(ProjectCommand::Change {
        project: "demo",
        action: Action::Pause,
        reason: "",
        expected_revision: None,
    })
    .unwrap();
    assert!(matches!(
        projects::execute(&client, &plan, &health),
        Err(ClientError::Observation(_))
    ));
    let (status, value) = server.post("/api/project-lifecycle/demo/pause", b"{}");
    assert_eq!(status, 501);
    assert_eq!(value["code"], "PROJECT_LIFECYCLE_NOT_CONFIGURED");
    assert_eq!(fixture.ledger_bytes(), before);
    assert_eq!(fixture.science_bytes(), science);
}

#[test]
fn real_post_response_failure_reports_uncertainty_after_its_durable_effect() {
    let fixture = Fixture::new();
    let science = fixture.science_bytes();
    fixture.initialize_projects();
    let server = Server::start(&fixture, true);
    let (status, response) = server.post(
        "/api/project-lifecycle/demo/pause",
        &serde_json::to_vec(&json!({"reason":TOKEN,"expected_revision":1})).unwrap(),
    );
    assert_eq!(status, 500);
    assert_eq!(response["code"], "PROJECT_OPERATION_UNCERTAIN");
    assert!(!response.to_string().contains(TOKEN));
    // The response guard cannot undo a committed transition and must never
    // describe this response failure as evidence that the operation was absent.
    let (head, journal) = fixture.ledger_bytes();
    let head: Value = serde_json::from_slice(&head).unwrap();
    assert_eq!(head["projects"][0]["state"], "PAUSED");
    assert_eq!(head["_durability"]["revision"], 2);
    assert_eq!(
        journal
            .split(|b| *b == b'\n')
            .filter(|line| !line.is_empty())
            .count(),
        2
    );
    assert_eq!(fixture.science_bytes(), science);
}
