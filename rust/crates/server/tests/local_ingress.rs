//! Actual native TCP client/server, synthetic append-only metrics and isolated
//! journal. No process control, provider credentials or cloud calls.
use ml_exp_client::{Client, local};
use ml_exp_core::local::{Finish, INITIAL_CHAIN, MAX_BATCH, Packet, Registration};
use ml_exp_server::{Auth, api::Api, read_config};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    net::{SocketAddr, TcpListener},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};
const TOKEN: &str = "synthetic-local-ingress-auth-00000000000000000000";
fn put(path: &Path, value: &Value) {
    ml_exp_io::write_atomic(path, &serde_json::to_vec(value).unwrap(), true).unwrap();
}
struct Fixture {
    temp: tempfile::TempDir,
    settings: read_config::Settings,
    definition: PathBuf,
    metrics: PathBuf,
    state: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let runs = temp.path().join("runs");
        fs::create_dir(&runs).unwrap();
        fs::set_permissions(&runs, fs::Permissions::from_mode(0o700)).unwrap();
        let project = temp.path().join("project.json");
        put(
            &project,
            &json!({"schema_version":1,"project":"demo","title":"Local fixture","run_roots":["runs"]}),
        );
        let registry = temp.path().join("registry");
        read_config::initialize(&registry).unwrap();
        let settings = read_config::Settings {
            schema_version: 1,
            registry_root: registry,
            projects: vec![read_config::Project {
                project: "demo".into(),
                project_file: project,
                run_roots: vec![runs],
                runs: vec![],
                discover_runs: None,
            }],
            source_policy: None,
        };
        let definition = temp.path().join("local.json");
        put(
            &definition,
            &json!({"run_id":"train","execution_id":"a".repeat(32),"evaluation":{"protocol_id":"fixture-v1"},"parameters":{"lr":0.001},"wandb":{"enabled":false}}),
        );
        let metrics = temp.path().join("metrics.jsonl");
        fs::write(&metrics, b"").unwrap();
        let state = temp.path().join("upload.json");
        Self {
            temp,
            settings,
            definition,
            metrics,
            state,
        }
    }
    fn api(&self) -> Api {
        Api::open(self.settings.clone()).unwrap()
    }
    fn request(&self) -> Registration {
        serde_json::from_slice(&fs::read(&self.definition).unwrap()).unwrap()
    }
    fn attach(
        &self,
        client: &Client,
        finish: Option<Finish>,
    ) -> Result<Value, ml_exp_client::Error> {
        local::attach(
            client,
            "demo",
            &self.definition,
            &self.metrics,
            &self.state,
            finish.map(local::Mode::Finish).unwrap_or(local::Mode::Once),
            |_| {},
        )
    }
}
struct Server {
    address: SocketAddr,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn start(f: &Fixture, address: Option<SocketAddr>, drop_ack: bool) -> Self {
        let token = f.temp.path().join("token");
        fs::write(&token, TOKEN).unwrap();
        fs::set_permissions(&token, fs::Permissions::from_mode(0o600)).unwrap();
        let auth = Auth::load(&token).unwrap();
        let api = f.api();
        let listener =
            TcpListener::bind(address.unwrap_or_else(|| "127.0.0.1:0".parse().unwrap())).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let drop_ack = Arc::new(AtomicBool::new(drop_ack));
        let thread = thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async move {
                    let app =
                        ml_exp_server::app(auth, Some(Arc::new(api)))
                            .layer(axum::middleware::from_fn(
                            move |request: axum::extract::Request, next: axum::middleware::Next| {
                                let flag = drop_ack.clone();
                                async move {
                                    let metrics = request.uri().path().ends_with("/metrics")
                                        && request.method() == "POST";
                                    let response = next.run(request).await;
                                    if metrics
                                        && flag.swap(false, Ordering::SeqCst)
                                        && response.status().is_success()
                                    {
                                        use axum::response::IntoResponse;
                                        (
                                            axum::http::StatusCode::SERVICE_UNAVAILABLE,
                                            axum::Json(json!({"code":"FIXTURE_ACK_LOST"})),
                                        )
                                            .into_response()
                                    } else {
                                        response
                                    }
                                }
                            },
                        ));
                    let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                    let serving =
                        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
                    let _ = stopped.await;
                    serving.abort();
                    let _ = serving.await;
                });
        });
        Self {
            address,
            stop: Some(stop),
            thread: Some(thread),
        }
    }
    fn client(&self) -> Client {
        Client::new(&format!("http://{}", self.address), TOKEN.into()).unwrap()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.stop.take().unwrap().send(());
        self.thread.take().unwrap().join().unwrap();
    }
}
#[test]
fn other_runs_are_not_local_and_missing_local_manifest_is_not_a_fallback() {
    let f = Fixture::new();
    let api = f.api();
    let run = f.settings.projects[0].run_roots[0].join("remote");
    fs::create_dir(&run).unwrap();
    put(&run.join("manifest.json"), &json!({"run_id":"remote"}));
    assert!(
        api.local_tracking_status("demo", "remote")
            .unwrap()
            .is_none()
    );
    api.register_local("demo", f.request(), "initial").unwrap();
    fs::remove_file(f.settings.projects[0].run_roots[0].join("train/manifest.json")).unwrap();
    assert_eq!(
        api.local_tracking_status("demo", "train").unwrap_err().code,
        "LOCAL_EVIDENCE_INVALID"
    );
}
#[test]
fn multiple_packets_lost_ack_restart_exact_replay_finish_and_public_reads() {
    let f = Fixture::new();
    let mut bytes = Vec::new();
    for step in 1..=3000 {
        bytes.extend(serde_json::to_vec(&json!({"name":"train/loss","value":5.0-step as f64/1000.0,"unit":"nats/token","step":step,"epoch":step as f64/100.0,"variant_id":"a"})).unwrap());
        bytes.push(b'\n');
    }
    assert!(bytes.len() > MAX_BATCH * 3);
    fs::write(&f.metrics, &bytes).unwrap();
    let server = Server::start(&f, None, true);
    let address = server.address;
    assert!(f.attach(&server.client(), None).is_err());
    let pending: Value = serde_json::from_slice(&fs::read(&f.state).unwrap()).unwrap();
    assert_eq!(pending["offset"], 0);
    assert_eq!(
        pending["pending"]["packet"]["bytes"]
            .as_array()
            .unwrap()
            .len(),
        MAX_BATCH
    );
    let first = f.api().local_status("demo", "train").unwrap();
    assert_eq!(first["cursor"]["offset"], MAX_BATCH);
    drop(server);
    let server = Server::start(&f, Some(address), false);
    let client = server.client();
    let result = f.attach(&client, Some(Finish::Succeeded)).unwrap();
    assert_eq!(result["cursor"]["offset"], bytes.len());
    assert_eq!(result["cursor"]["records"], 3000);
    assert_eq!(result["cursor"]["invalid_lines"], 0);
    assert_eq!(result["cursor"]["finished"], "SUCCEEDED");
    assert!(result["provider_job_id"].is_null());
    assert!(result["submission_id"].is_null());
    assert_eq!(result["process_liveness"], "UNKNOWN");
    let total = result["scopes"][0]["total"].as_i64().unwrap();
    assert_eq!(total, 3002);
    assert_eq!(
        f.attach(&client, Some(Finish::Succeeded)).unwrap()["scopes"][0]["total"],
        total
    );
    let records = client
        .get("/api/attempts/demo/train::attempt-001/metrics?max_points=4000")
        .unwrap();
    assert!(
        serde_json::to_string(&records)
            .unwrap()
            .contains("train/loss")
    );
    let run = client.get("/api/runs/demo/train").unwrap();
    assert_eq!(run["current_attempt_id"], "attempt-001");
    assert_eq!(run["execution_kind"], "local");
    assert_eq!(run["local_execution"]["state"], "SUCCEEDED");
    let schema = client.get("/api/v2/openapi.json").unwrap();
    assert!(
        schema["paths"]["/api/local-runs/{project}/{run}/metrics"]
            .get("post")
            .is_some()
    );
    assert_eq!(
        client.get("/api/runs/demo/train/tracking").unwrap()["cursor"],
        result["cursor"]
    );
    assert_eq!(
        client.get("/api/health").unwrap()["local_run_creation"],
        true
    );
    fs::OpenOptions::new()
        .append(true)
        .open(&f.metrics)
        .unwrap()
        .write_all(b"\n")
        .unwrap();
    assert!(f.attach(&client, None).is_err());
}
#[test]
fn journal_receipt_cursor_crash_windows_and_pending_packet_conflict() {
    let f = Fixture::new();
    let api = f.api();
    api.register_local("demo", f.request(), "initial").unwrap();
    let ingress = f.settings.projects[0].run_roots[0].join("train/local-ingress");
    let original = fs::read(ingress.join("cursor.json")).unwrap();
    let packet = Packet {
        execution_id: "a".repeat(32),
        offset: 0,
        previous_chain_sha256: INITIAL_CHAIN.into(),
        bytes: b"{\"name\":\"loss\",\"value\":2.5,\"unit\":\"nats/token\",\"step\":7}\n".into(),
        finish: None,
    };
    let ack = api
        .ingest_local("demo", "train", packet.clone(), "first")
        .unwrap();
    ml_exp_io::write_atomic(&ingress.join("cursor.json"), &original, false).unwrap();
    assert_eq!(
        api.ingest_local("demo", "train", packet.clone(), "later")
            .unwrap(),
        ack
    );
    assert_eq!(
        api.local_status("demo", "train").unwrap()["scopes"][0]["total"],
        2
    );
    ml_exp_io::write_atomic(&ingress.join("cursor.json"), &original, false).unwrap();
    fs::remove_file(ingress.join("packets/0-data.json")).unwrap();
    let mut different = packet.clone();
    different.bytes = b"different".into();
    assert_eq!(
        api.ingest_local("demo", "train", different, "later")
            .unwrap_err()
            .code,
        "LOCAL_PACKET_CONFLICT"
    );
    api.ingest_local("demo", "train", packet.clone(), "later")
        .unwrap();
    assert_eq!(
        api.local_status("demo", "train").unwrap()["scopes"][0]["total"],
        2
    );
    let mut wrong = packet.clone();
    wrong.execution_id = "b".repeat(32);
    assert_eq!(
        api.ingest_local("demo", "train", wrong, "later")
            .unwrap_err()
            .code,
        "LOCAL_IDENTITY_CONFLICT"
    );
    let mut changed = f.request();
    changed.seed = 42;
    assert_eq!(
        api.register_local("demo", changed, "later")
            .unwrap_err()
            .code,
        "LOCAL_IDENTITY_CONFLICT"
    );
    let store = api.local_journal("demo", "train").unwrap().unwrap();
    let guard = store.guard("demo", "train").unwrap();
    let scope = store.scope(&guard, "attempt-001").unwrap();
    let events = store.events(&guard, &scope, 0, 10).unwrap();
    assert_eq!(events[1].payload["observations"][0]["step"], 7);
    assert_eq!(events[1].payload["observations"][0]["value"], 2.5);
}
#[test]
fn native_client_detects_prefix_mutation_truncation_replacement_and_state_lock() {
    let f = Fixture::new();
    let server = Server::start(&f, None, false);
    let client = server.client();
    let point = b"{\"name\":\"loss\",\"unit\":\"nats/token\",\"value\":4.2,\"step\":1}\n";
    fs::write(&f.metrics, point).unwrap();
    f.attach(&client, None).unwrap();
    let before = fs::read(&f.state).unwrap();
    let mut changed = point.to_vec();
    changed[1] = b' ';
    fs::write(&f.metrics, &changed).unwrap();
    assert!(f.attach(&client, None).is_err());
    assert_eq!(fs::read(&f.state).unwrap(), before);
    fs::write(&f.metrics, b"x").unwrap();
    assert!(f.attach(&client, None).is_err());
    fs::write(&f.metrics, point).unwrap();
    let mut append = fs::OpenOptions::new()
        .append(true)
        .open(&f.metrics)
        .unwrap();
    append.write_all(point).unwrap();
    drop(append);
    let replacement = f.temp.path().join("replacement");
    fs::write(&replacement, b"replacement").unwrap();
    let result = local::attach(
        &client,
        "demo",
        &f.definition,
        &f.metrics,
        &f.state,
        local::Mode::Once,
        |_| {
            assert!(
                f.attach(&client, None)
                    .unwrap_err()
                    .to_string()
                    .contains("another uploader")
            );
            fs::rename(&replacement, &f.metrics).unwrap();
        },
    );
    assert!(result.is_err());
}

#[test]
fn exponent_number_text_in_parameters_keeps_the_local_journal_usable() {
    // Python's json.dumps writes small floats as `2e-05`. The registry stores the
    // binding canonically (`0.00002`), so comparing it with the manifest by
    // number text refused every later read and packet of such a Run.
    let f = Fixture::new();
    fs::write(
        &f.definition,
        br#"{"run_id":"train","execution_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","evaluation":{"protocol_id":"fixture-v1"},"parameters":{"min_p0":2e-05,"lr":1e-07,"steps":80},"wandb":{"enabled":false}}"#,
    )
    .unwrap();
    fs::set_permissions(&f.definition, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(
        &f.metrics,
        b"{\"name\":\"train/loss\",\"value\":0.5,\"unit\":\"nats/token\",\"step\":1}\n",
    )
    .unwrap();
    let server = Server::start(&f, None, false);
    let client = server.client();
    let first = f.attach(&client, None).unwrap();
    assert_eq!(first["cursor"]["records"], 1);
    let status = f.api().local_status("demo", "train").unwrap();
    assert_eq!(status["cursor"]["records"], 1);
    // A second attach reads the same binding again and finishes the stream.
    let done = f.attach(&client, Some(Finish::Succeeded)).unwrap();
    assert_eq!(done["cursor"]["records"], 1);
}
