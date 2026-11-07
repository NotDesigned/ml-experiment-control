//! A cloud worker that cannot start says why, through the real store and server
//! executable. Credentials and images are synthetic; no provider call is made.
use ml_exp_core::{
    cloud::run::Runtime,
    first_submission::native::{Approval, attempt_slot, creation::Request},
    sources::Policy,
};
use ml_exp_server::{
    cloud_runs::{Service, Settings},
    read_config, sources,
};
use serde_json::{Value, json};
use std::{fs, io::Cursor, os::unix::fs::PermissionsExt, sync::Arc};

const NOW: &str = "2026-10-29T00:00:00Z";

fn private(path: &std::path::Path, value: &Value) {
    ml_exp_io::write_atomic(path, &serde_json::to_vec(value).unwrap(), true).unwrap();
}
fn archive() -> Vec<u8> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    let mut tar = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    let body = b"FROM x\n";
    header.set_size(body.len() as u64);
    header.set_mode(0o400);
    header.set_cksum();
    tar.append_data(&mut header, "Dockerfile", &body[..])
        .unwrap();
    tar.into_inner().unwrap().finish().unwrap()
}
struct Server(std::process::Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn a_worker_that_cannot_start_says_why_until_it_first_reports() {
    let temp = tempfile::tempdir().unwrap();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let registry = temp.path().join("registry");
    read_config::initialize(&registry).unwrap();
    let project = temp.path().join("project.json");
    let runs = temp.path().join("runs");
    fs::create_dir(&runs).unwrap();
    private(
        &project,
        &json!({"schema_version":1,"project":"demo","title":"Fixture","run_roots":["runs"]}),
    );
    let settings = read_config::Settings {
        schema_version: 1,
        registry_root: registry.clone(),
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
    let environment = sources
        .publish(
            "demo",
            &mut Cursor::new(&bytes),
            &ml_exp_core::slurm::helper::sha256_bytes(&bytes)[7..],
            NOW,
        )
        .unwrap()["source_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let runtime = Runtime {
        runtime_id: format!("runtime.{}", "1".repeat(64)),
        source_id: environment,
        image: format!("registry.example/team/code@sha256:{}", "2".repeat(64)),
        source_dir: "/workspace".into(),
        worker_path: "/opt/ml-expd/worker".into(),
        worker_sha256: format!("sha256:{}", "3".repeat(64)),
    };
    let credentials = temp.path().join("credentials.json");
    private(
        &credentials,
        &json!({"provider":"pai","access_key_id":"synthetic","access_key_secret":"never-sent"}),
    );
    let signing_key = temp.path().join("worker-signing-key");
    ml_exp_io::write_atomic(
        &signing_key,
        b"synthetic-exchange-key-0000000000000000000000",
        true,
    )
    .unwrap();
    let config = json!({"schema_version":1,"enabled":false,"allow_run_creation":true,
        "exchange":{"public_url":"https://api.example/ml-expd","signing_key_file":signing_key},
        "backends":[{"executor":"pai-h200","profile":{"provider":"pai","region":"us-southeast-1",
            "workspace_id":"253","quota_id":"quota1","max_gpus":8,"max_cpus":200,"max_memory_gb":1600,
            "storage":{"cpfs_uri":"bmcpfs://cpfs-example.us-southeast-1.cpfs.aliyuncs.com/",
                "workspace_user":"researcher","role_arn":"acs:ram::12345:role/pai-data-researcher"}},
            "run_root":"/mnt/workspace/ml-expd/runs","credentials_file":credentials,
            "registry_auth_file":null,"runtimes":{runtime.runtime_id.clone():runtime}}]});
    Service::initialize(&settings).unwrap();
    let service = Service::open(
        settings.clone(),
        Settings::from_slice(&serde_json::to_vec(&config).unwrap()).unwrap(),
        sources.clone(),
        None,
        Default::default(),
    )
    .unwrap();
    let request: Request = serde_json::from_value(json!({"run_id":"train","runtime_id":runtime.runtime_id,
        "executor":"pai-h200","command":["python","train.py"],"resources":{"gpus":1,"cpus":8,"memory_gb":64},
        "walltime_seconds":300,"seed":42}))
    .unwrap();
    service.create_run("demo", request, NOW).unwrap();
    let prepared = service
        .prepare(
            "demo",
            "train",
            Approval {
                max_gpu_hours: 1.,
                reason: String::new(),
            },
            "2026-10-29T00:01:00Z",
            "2026-10-29T01:01:00Z",
            &"1".repeat(32),
        )
        .unwrap();
    let action = prepared["submission_id"].as_str().unwrap().to_owned();
    let actions =
        ml_exp_server::operations::Store::open_existing(registry.join("actions")).unwrap();
    let package = actions.cloud_request(&action).unwrap().package().unwrap();
    let channel = package.channel.clone();

    let read_file = temp.path().join("reads.json");
    private(&read_file, &serde_json::to_value(&settings).unwrap());
    let cloud_file = temp.path().join("cloud.json");
    private(&cloud_file, &config);
    let token = "native-cloud-http-synthetic-token-000000000000";
    let token_file = temp.path().join("token");
    ml_exp_io::write_atomic(&token_file, token.as_bytes(), true).unwrap();
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_ml-expd-rs"))
        .args(["--bind", "127.0.0.1:0", "--token-file"])
        .arg(&token_file)
        .arg("--read-config")
        .arg(&read_file)
        .arg("--source-archives")
        .arg("--cloud-config")
        .arg(&cloud_file)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    let mut server = Server(child);
    let stdout = server.0.stdout.take().unwrap();
    let (sent, received) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let line = std::io::BufReader::new(stdout)
            .lines()
            .next()
            .unwrap()
            .unwrap();
        let _ = sent.send(line);
    });
    let address = received
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("server startup");
    let client = ml_exp_client::Client::new(&format!("http://{address}"), token.into()).unwrap();
    let health = client.health().unwrap();
    let http = reqwest::blocking::Client::new();
    let post = |credential: Option<&str>, path: &str, body: Vec<u8>| {
        let request = http
            .post(format!("http://{address}{path}"))
            .header("content-type", "application/json")
            .body(body);
        match credential {
            Some(credential) => request.bearer_auth(credential),
            None => request,
        }
        .send()
        .unwrap()
    };
    let path = "/api/worker-exchange/demo/train/attempt-001/startup-failure";
    let report = |message: &str| serde_json::to_vec(&json!({ "message": message })).unwrap();
    let slot = registry.join("cloud-submissions/slots").join(format!(
        "{}.json.startup-failure.json",
        attempt_slot("demo", "train", "attempt-001").unwrap()
    ));

    // Only this Attempt's own capability is accepted, and never the API token.
    for (credential, path) in [
        (None, path),
        (Some("wrong-worker-token"), path),
        (Some(token), path),
        (
            Some(channel.token.as_str()),
            "/api/worker-exchange/demo/train/attempt-002/startup-failure",
        ),
    ] {
        assert_eq!(
            post(credential, path, report("x")).status().as_u16(),
            401,
            "{path} {credential:?}"
        );
    }
    // A body is bounded, exact and says something.
    let oversize = report(&"x".repeat(5000));
    assert_eq!(
        post(Some(&channel.token), path, oversize).status().as_u16(),
        413
    );
    for body in [
        br#"{"message":"x","extra":1}"#.to_vec(),
        br#"{"message":" \n "}"#.to_vec(),
        b"not json".to_vec(),
    ] {
        assert_eq!(
            post(Some(&channel.token), path, body).status().as_u16(),
            422
        );
    }
    assert!(!slot.exists());

    // Redacted, one line, clipped; only the latest is kept.
    let first = format!(
        "managed launch storage failed at crates/worker/src/cloud.rs:123: denied\npassword=hunter2 {} {}",
        channel.token,
        "y".repeat(3000)
    );
    let response = post(Some(&channel.token), path, report(&first));
    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(
        serde_json::from_slice::<Value>(&response.bytes().unwrap()).unwrap(),
        json!({"recorded":true})
    );
    let kept: Value = serde_json::from_slice(&fs::read(&slot).unwrap()).unwrap();
    let message = kept["message"].as_str().unwrap();
    assert!(message.starts_with(
        "managed launch storage failed at crates/worker/src/cloud.rs:123: denied password="
    ));
    assert!(!message.contains("hunter2") && !message.contains(&channel.token));
    assert_eq!(message.chars().count(), 1000);
    assert_eq!(kept.as_object().unwrap().len(), 2);
    assert!(kept["received_at"].is_string());
    let latest =
        "managed launch storage failed at crates/worker/src/cloud.rs:123: PermissionDenied";
    assert_eq!(
        post(Some(&channel.token), path, report(latest))
            .status()
            .as_u16(),
        200
    );

    // It is evidence only: the Submission is as it was, and status and diagnose show it.
    assert_eq!(
        client.get(&format!("/api/submissions/{action}")).unwrap()["status"],
        "PREPARED"
    );
    let progress = client
        .get(&format!("/api/submissions/{action}/progress"))
        .unwrap();
    assert_eq!(progress["availability"], "NOT_REPORTED");
    assert_eq!(progress["startup_failure"]["message"], latest);
    let selection = || ml_exp_client::observation::Selection {
        project: Some("demo"),
        run: Some("train"),
        attempt: Some("attempt-001"),
    };
    let status =
        ml_exp_client::status::observe(&client, &health, None, selection(), false).unwrap();
    assert_eq!(status["startup_failure"]["message"], latest, "{status}");
    let text = ml_exp_client::presentation::status_summary(&status).unwrap();
    assert!(
        text.lines()
            .any(|line| line.starts_with(&format!("Startup failure: {latest} (reported "))),
        "{text}"
    );
    let diagnosed =
        ml_exp_client::read::provider_diagnostics(&client, None, selection(), 10).unwrap();
    assert_eq!(diagnosed["startup_failure"]["message"], latest);
    assert!(
        ml_exp_client::presentation::provider_diagnostics(&diagnosed)
            .contains(&format!("Startup failure: {latest}"))
    );

    // Once the Attempt has reported, a report is ignored and none is shown.
    let mut current = actions.observe(&action).unwrap()["execution"].clone();
    current["status"] = json!("VERIFIED");
    actions
        .commit(
            &action,
            &ml_exp_core::operations::Transition {
                execution: current,
                event: json!({"event":"SYNTHETIC_TEST_RECEIPT"}),
                expected_status: Some("PREPARED".into()),
            },
            &"4".repeat(32),
            "2026-10-29T00:04:01Z",
        )
        .unwrap();
    let binding = package.config.binding().unwrap();
    let sha = ml_exp_core::slurm::helper::sha256_bytes;
    let config_sha256 = sha(&serde_json::to_vec(&package.config).unwrap());
    let packet: ml_exp_core::cloud::exchange::Packet = serde_json::from_value(json!({
        "contract":ml_exp_core::cloud::exchange::CONTRACT,"sequence":1,"previous_sha256":null,
        "package_sha256":sha(&serde_json::to_vec(&package).unwrap()),
        "evidence":{"contract":ml_exp_core::managed::remote::OBSERVATION_CONTRACT,"config_sha256":config_sha256,"job_id":binding.sha256,
            "worker":{"contract":ml_exp_core::managed::evidence::CONTRACT,"identity":binding.identity,"manifest_sha256":binding.sha256,
                "phase":"STARTED","safe_to_retry":false,"receipt":null,"runtime_environment_sha256":"a".repeat(64)},
            "stdout":{"text":"","offset_bytes":0,"total_bytes":0,"problem":null},
            "stderr":{"text":"","offset_bytes":0,"total_bytes":0,"problem":null},
            "metrics":{"text":"","offset_bytes":0,"total_bytes":0,"problem":null}},
        "history":null,"checkpoint":null}))
    .unwrap();
    let response = post(
        Some(&channel.token),
        "/api/worker-exchange/demo/train/attempt-001",
        serde_json::to_vec(&packet).unwrap(),
    );
    assert_eq!(response.status().as_u16(), 200, "{:?}", response.text());
    let response = post(Some(&channel.token), path, report("later"));
    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(
        serde_json::from_slice::<Value>(&response.bytes().unwrap()).unwrap(),
        json!({"recorded":false})
    );
    let progress = client
        .get(&format!("/api/submissions/{action}/progress"))
        .unwrap();
    assert_eq!(progress["availability"], "AVAILABLE");
    assert!(progress.get("startup_failure").is_none(), "{progress}");
    drop(server);
}
