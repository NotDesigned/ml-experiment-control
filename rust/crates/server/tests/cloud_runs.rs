//! Public cloud authorship is exercised with real native stores and HTTP
//! routing. Credentials and images are synthetic; no provider call is made.
use ml_exp_core::{
    cloud::{pai, run::Runtime},
    first_submission::native::{Approval, creation::Request},
    sources::Policy,
};
use ml_exp_server::{
    cloud_runs::{Service, Settings},
    read_config, sources,
};
use serde_json::{Value, json};
use std::{fs, io::Cursor, os::unix::fs::PermissionsExt, sync::Arc};
fn private(path: &std::path::Path, value: &Value) {
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
#[test]
fn create_reuse_authorize_restart_and_frozen_guard_without_provider_effects() {
    let temp = tempfile::tempdir().unwrap();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let registry = temp.path().join("registry");
    read_config::initialize(&registry).unwrap();
    let project = temp.path().join("project.json");
    let runs = temp.path().join("runs");
    fs::create_dir(&runs).unwrap();
    private(
        &project,
        &json!({"schema_version":1,"project":"demo","title":"Native cloud fixture","run_roots":["runs"]}),
    );
    let settings = read_config::Settings {
        schema_version: 1,
        registry_root: registry.clone(),
        projects: vec![read_config::Project {
            project: "demo".into(),
            project_file: project,
            run_roots: vec![runs.clone()],
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
            &ml_exp_core::slurm::helper::sha256_bytes(&bytes)[7..],
            "2026-10-29T00:00:00Z",
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
    let credentials = temp.path().join("credentials.json");
    private(
        &credentials,
        &json!({"provider":"pai","access_key_id":"synthetic","access_key_secret":"never-sent"}),
    );
    let profile = pai::Profile {
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
    };
    let config = json!({"schema_version":1,"enabled":false,"allow_run_creation":true,"backends":[{"executor":"pai-h200","profile":{"provider":"pai","region":profile.region,"workspace_id":profile.workspace_id,"quota_id":profile.quota_id,"max_gpus":8,"max_cpus":200,"max_memory_gb":1600,"storage":profile.storage},"run_root":"/mnt/workspace/ml-expd/runs","credentials_file":credentials,"registry_auth_file":null,"runtimes":{runtime.runtime_id.clone():runtime}}]});
    Service::initialize(&settings).unwrap();
    assert!(Service::initialize(&settings).is_err());
    let open = || {
        Service::open(
            settings.clone(),
            Settings::from_slice(&serde_json::to_vec(&config).unwrap()).unwrap(),
            sources.clone(),
            None,
            Default::default(),
        )
        .unwrap()
    };
    let service = open();
    let request:Request=serde_json::from_value(json!({"run_id":"train","runtime_id":runtime.runtime_id,"executor":"pai-h200","command":["python","train.py"],"resources":{"gpus":1,"cpus":8,"memory_gb":64},"walltime_seconds":300,"seed":42})).unwrap();
    let first = service
        .create_run("demo", request.clone(), "2026-10-29T00:00:00Z")
        .unwrap();
    assert_eq!(first["scheduler_mutation_performed"], false);
    assert_eq!(
        service
            .create_run("demo", request.clone(), "2026-10-29T00:01:00Z")
            .unwrap()["reused"],
        true
    );
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
    let action = prepared["submission_id"].as_str().unwrap();
    assert_eq!(prepared["status"], "PREPARED");
    assert!(service.handles_action(action).unwrap());
    let authorized = service
        .authorize(
            action,
            "owned fixture",
            "2026-10-29T00:02:00Z",
            &"2".repeat(32),
        )
        .unwrap();
    assert_eq!(authorized["status"], "AUTHORIZED");
    assert!(
        service
            .execute(
                action,
                &format!("EXECUTE {action}"),
                "2026-10-29T00:03:00Z",
                &"3".repeat(32),
                &std::sync::atomic::AtomicBool::new(false)
            )
            .is_err()
    );
    drop(service);
    let service = open();
    assert_eq!(
        service
            .for_run("demo", "train", "2026-10-29T00:04:00Z")
            .unwrap()["submission_id"],
        action
    );
    // Start the actual server executable with cloud-only configuration and use
    // the official client. This also checks blocking clients stay off Tokio.
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
    struct Server(std::process::Child);
    impl Drop for Server {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
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
        .expect("native cloud startup");
    let client = ml_exp_client::Client::new(&format!("http://{address}"), token.into()).unwrap();
    let health = client.negotiate().unwrap();
    assert!(
        health["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "native-cloud-submissions.v1")
    );
    let plan = ml_exp_client::submissions::plan(ml_exp_client::submissions::Command::Create {
        project: "demo",
        request: &request,
    })
    .unwrap();
    assert_eq!(
        ml_exp_client::submissions::execute(&client, &plan, &health).unwrap()["reused"],
        true
    );
    assert_eq!(
        client.get(&format!("/api/submissions/{action}")).unwrap()["status"],
        "AUTHORIZED"
    );
    assert_eq!(
        client.get("/api/runs/demo/train").unwrap()["native_submission"]["status"],
        "AUTHORIZED"
    );
    drop(server);
    let mut changed = request;
    changed.seed = 43;
    assert!(
        service
            .create_run("demo", changed, "2026-10-29T00:05:00Z")
            .is_err()
    );
    let path = runs.join("train/manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["seed"] = json!(123);
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(service.get(action, "2026-10-29T00:06:00Z").is_err());
}
