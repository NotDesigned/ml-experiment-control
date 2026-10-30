//! Real TCP acceptance of the isolated native read service and native client.
//! All manifests, journals and credentials are synthetic private temp fixtures.
use ml_exp_client::{Client, Error as ClientError, observation::Selection, read, status, watch};
use ml_exp_core::records::{Event, normalized_metric};
use ml_exp_server::{Auth, read_api::Api, read_config, records};
use serde_json::{Value, json};
use std::{
    cell::Cell,
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::Duration,
};

const TOKEN: &str = "synthetic-native-http-token-0000000000000000000000";

fn private_directory(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

struct Fixture {
    _temp: tempfile::TempDir,
    project: PathBuf,
    registry: PathBuf,
    token: PathBuf,
    workspace: String,
    manifest: Value,
    journal_config: Value,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let project = temp.path().join("demo");
        private_directory(&project);
        private_directory(&project.join("runs"));
        private_directory(&project.join("runs/train"));
        private_directory(&project.join("runs/train/attempts"));
        write_json(
            &project.join("research_project.yaml"),
            &json!({"schema_version":1,"project":"demo","title":"Synthetic science"}),
        );
        let evaluation = json!({"protocol_id":"synthetic-protocol-v1"});
        let manifest = json!({"schema_version":1,"identity_version":2,"project":"demo","run_id":"train",
            "source_id":"source.fixture","image_id":"sha256:fixture","config_path":"config.json","seed":42,"campaign":"fixture",
            "created_at":"2026-10-27T00:00:00Z","evaluation":evaluation,"parameters":{"lr":0.001},
            "resources":{"gpus":1},"container":{"runtime_id":"runtime.fixture"},"backend":{"kind":"slurm"},"inputs":[]});
        write_json(&project.join("runs/train/manifest.yaml"), &manifest);
        for (attempt, state) in [("attempt-001", "SUCCEEDED"), ("attempt-002", "RUNNING")] {
            let directory = project.join("runs/train/attempts").join(attempt);
            private_directory(&directory);
            let mut attempt_manifest = manifest.clone();
            attempt_manifest["attempt_id"] = json!(attempt);
            write_json(&directory.join("attempt.yaml"), &attempt_manifest);
            write_json(
                &directory.join("status.json"),
                &json!({"project":"demo","run_id":"train","attempt_id":attempt,"state":state,
                "updated_at":"2026-10-27T00:03:00Z"}),
            );
            write_json(
                &directory.join("collection.json"),
                &json!({"project":"demo","run_id":"train","attempt_id":attempt,
                "status":"AVAILABLE","verification":"SHA256_VERIFIED","download_available":true,
                "worker_state":"COMPLETED","process_state":if attempt=="attempt-001"{"COMPLETED"}else{"RUNNING"},
                "failure_class":"RESOURCE_EXHAUSTED",
                "observed_at":"2026-10-27T00:03:00Z","process_evidence":{"exit_code":if attempt=="attempt-001"{json!(0)}else{Value::Null}}}),
            );
            fs::write(
                directory.join("stdout.log"),
                format!("{attempt} exact output\n"),
            )
            .unwrap();
            fs::write(directory.join("stderr.log"), b"synthetic diagnostic\n").unwrap();
        }
        let registry = temp.path().join("registry");
        let workspace = read_config::initialize(&registry).unwrap();
        let token = temp.path().join("token");
        fs::write(&token, TOKEN).unwrap();
        fs::set_permissions(&token, fs::Permissions::from_mode(0o600)).unwrap();
        let journal_config = json!({"ml_expd":{"project":"demo","run_id":"train","source_id":manifest["source_id"],
            "image_id":manifest["image_id"],"resources":manifest["resources"],"evaluation":evaluation,
            "parameters":manifest["parameters"],"runtime_id":"runtime.fixture","backend":"slurm","inputs":[]}});
        let fixture = Self {
            _temp: temp,
            project,
            registry,
            token,
            workspace,
            manifest,
            journal_config,
        };
        fixture.current("attempt-001");
        fixture
    }
    fn run(&self) -> PathBuf {
        self.project.join("runs/train")
    }
    fn current(&self, attempt: &str) {
        write_json(
            &self.run().join("status.json"),
            &json!({"project":"demo","run_id":"train","attempt_id":attempt,
            "state":if attempt=="attempt-001"{"SUCCEEDED"}else{"RUNNING"},"updated_at":"2026-10-27T00:03:00Z"}),
        );
    }
    fn settings(&self) -> read_config::Settings {
        read_config::Settings::from_json(&serde_json::to_vec(&json!({"schema_version":1,"registry_root":self.registry,
            "projects":[{"project":"demo","project_file":self.project.join("research_project.yaml"),"run_roots":[self.project.join("runs")],
                "runs":[{"run_id":"train","root_index":0,"relative_components":["train"]}]}]})).unwrap()).unwrap()
    }
    fn journal(&self, seed: &str, config: &Value) {
        let store = records::Store::open(&self.registry, seed).unwrap();
        let guard = store.guard("demo", "train").unwrap();
        store
            .bind(&guard, &json!({"enabled":false}), config)
            .unwrap();
        for (attempt, value) in [("attempt-001", 2.5), ("attempt-002", 9.0)] {
            let scope = store.scope(&guard, attempt).unwrap();
            for (index, variant) in ["left", "right"].into_iter().enumerate() {
                let metric = json!({"name":"validation/loss","unit":"nats/token","value":value+index as f64,"step":7,
                    "variant_id":variant,"protocol_id":"synthetic-protocol-v1","dataset_id":"dataset.fixture","checkpoint_id":"checkpoint.fixture",
                    "numerator":250,"denominator":100});
                store
                    .append(
                        &guard,
                        &scope,
                        &[Event {
                            event_id: format!("{attempt}-{variant}"),
                            payload: normalized_metric(&metric, config).unwrap(),
                        }],
                    )
                    .unwrap();
            }
        }
    }
    fn journal_matching(&self) {
        self.journal(
            self.workspace.strip_prefix("rust-preview-").unwrap(),
            &self.journal_config,
        );
    }
    fn server(&self) -> Server {
        Server::start(
            Auth::load(&self.token).unwrap(),
            Api::open(self.settings()).unwrap(),
        )
    }
    fn historical_settings(&self, journal: bool) -> ml_exp_server::history::Settings {
        fs::remove_file(self.registry.join(".rust-preview-workspace.json")).unwrap();
        write_json(
            &self.registry.join("registry.json"),
            &json!({"schema_version":1,"projects":[{
                "project":"demo","project_file":self.project.join("research_project.yaml"),"state":"ACTIVE","source":"MANUAL",
                "registered_at":"2026-10-27T00:00:00Z","updated_at":"2026-10-27T00:00:00Z","state_reason":"original registration"
            }]}),
        );
        fs::write(self.registry.join(".registry.lock"), b"").unwrap();
        fs::write(self.registry.join("events.jsonl"), b"").unwrap();
        ml_exp_server::history::Settings {
            schema_version: 1,
            reads: self.settings(),
            journal_workspace_id: journal
                .then(|| self.workspace.strip_prefix("rust-preview-").unwrap().into()),
            artifact_store: None,
        }
    }
}

struct Server {
    address: SocketAddr,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn start(auth: Auth, api: Api) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let thread = thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                    let router = axum::Router::new()
                        .nest("/ml-expd", ml_exp_server::app(auth, Some(Arc::new(api))));
                    let serving =
                        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
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
        Client::new(&format!("http://{}/ml-expd", self.address), TOKEN.into()).unwrap()
    }
    fn raw(&self, path: &str, headers: &str) -> (u16, Value) {
        self.request("GET", path, headers, "")
    }
    fn request(&self, method: &str, path: &str, headers: &str, body: &str) -> (u16, Value) {
        let mut connection = TcpStream::connect(self.address).unwrap();
        connection
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        connection
            .set_write_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(
            connection,
            "{method} /ml-expd{path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Length: {}\r\n{headers}\r\n{body}",
            self.address,body.len()
        )
        .unwrap();
        let mut bytes = Vec::new();
        connection
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .unwrap();
        assert!(bytes.len() <= 8 * 1024 * 1024);
        let split = bytes.windows(4).position(|v| v == b"\r\n\r\n").unwrap();
        let head = std::str::from_utf8(&bytes[..split]).unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let body = &bytes[split + 4..];
        (
            status,
            if body.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(body).unwrap()
            },
        )
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}
fn selected(attempt: Option<&str>) -> Selection<'_> {
    Selection {
        project: Some("demo"),
        run: Some("train"),
        attempt,
    }
}
fn headers() -> String {
    format!("Authorization: Bearer {TOKEN}\r\nX-ML-Expd-Client-Protocol: 2\r\n")
}

#[test]
fn actual_http_reads_historical_checkpoints_without_a_native_submission_service() {
    for mode in ["native-reader", "historical-reader", "unified"] {
        use ml_exp_core::checkpoints::{self, Context};
        let f = Fixture::new();
        let source_id = format!("source.{}", "a".repeat(64));
        let mut manifest = f.manifest.clone();
        manifest["source_id"] = json!(source_id);
        write_json(&f.run().join("manifest.yaml"), &manifest);
        for attempt in ["attempt-001", "attempt-002"] {
            let mut value = manifest.clone();
            value["attempt_id"] = json!(attempt);
            write_json(
                &f.run().join("attempts").join(attempt).join("attempt.yaml"),
                &value,
            );
        }
        let context = Context {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
            source_id,
            image_id: "sha256:fixture".into(),
            storage_scope: "storage.original".into(),
            state_root: "/approved/original/train/state".into(),
        };
        let document = checkpoints::register(&checkpoints::document(&context,
        &json!({"step":100,"files":[{"path":"checkpoints/s100/model.bin","bytes":4,"sha256":"b".repeat(64)}]})
    ).unwrap(),"2026-10-29T00:00:00Z").unwrap();
        let id = document["checkpoint_id"].as_str().unwrap();
        let transfer = f.registry.join("artifact-transfers/demo/train");
        private_directory(&transfer);
        write_json(
            &transfer.join("attempt-001.json"),
            &json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","checkpoint_state":context,"token":"not-an-api-credential"}),
        );
        let checkpoints = f
            .registry
            .join("persistent-checkpoints/demo/train/attempt-001");
        private_directory(&checkpoints);
        write_json(&checkpoints.join(format!("{id}.json")), &document);
        // A different current Attempt must never redirect the explicit historical
        // query. No submission/source/backend service is configured in this app.
        f.current("attempt-002");
        let native = Fixture::new();
        let server = if mode == "historical-reader" {
            let settings = f.historical_settings(false);
            Server::start(
                Auth::load(&f.token).unwrap(),
                ml_exp_server::history::open(settings).unwrap(),
            )
        } else if mode == "unified" {
            let settings = f.historical_settings(false);
            let native_settings = rename_fixture_run(&native, "fresh-checkpoint-reader");
            let bridge = ml_exp_server::history::Bridge::open(settings).unwrap();
            let reference = ml_exp_core::checkpoints::Reference {
                run_id: "train".into(),
                attempt_id: "attempt-001".into(),
                checkpoint_id: id.into(),
            };
            assert!(bridge.claims("demo", &reference.run_id).unwrap());
            Server::start(
                Auth::load(&native.token).unwrap(),
                Api::open(native_settings)
                    .unwrap()
                    .with_history(bridge)
                    .unwrap(),
            )
        } else {
            f.server()
        };
        let path = "/api/runs/demo/train/attempts/attempt-001/checkpoints";
        let (status, listed) = server.raw(path, &headers());
        assert_eq!(status, 200);
        assert_eq!(listed["checkpoints"], json!([document]));
        let (status, found) = server.raw(&format!("{path}/{id}"), &headers());
        assert_eq!(status, 200);
        assert_eq!(found, document);
        assert!(
            !serde_json::to_string(&found)
                .unwrap()
                .contains("not-an-api-credential")
        );
        assert!(
            !f.registry
                .join("persistent-checkpoints/.checkpoints.lock")
                .exists()
        );
        assert_eq!(server.raw(path, "X-ML-Expd-Client-Protocol: 2\r\n").0, 401);
        let client = server.client();
        let health = client.negotiate().unwrap();
        let plan =
            ml_exp_client::submissions::plan(ml_exp_client::submissions::Command::Checkpoints {
                project: "demo",
                run: "train",
                attempt: "attempt-001",
                checkpoint: Some(id),
            })
            .unwrap();
        assert_eq!(
            ml_exp_client::submissions::execute(&client, &plan, &health).unwrap(),
            document
        );
        // Read capability does not permit provider checkpoint refresh/registration.
        let refresh =
            ml_exp_client::submissions::plan(ml_exp_client::submissions::Command::Checkpoint {
                submission: "action-0123456789abcdef",
            })
            .unwrap();
        assert!(ml_exp_client::submissions::execute(&client, &refresh, &health).is_err());
    }
}

#[test]
fn saved_submission_reads_committed_identity_without_bootstrap_or_latest_guess() {
    use ml_exp_server::operations;
    let fixture = Fixture::new();
    let actions = operations::Store::open(fixture.registry.join("actions")).unwrap();
    let id = "action-0123456789abcdef";
    let plan = json!({"action_id":id,"scope":{"project":"demo","scope_type":"run","object_id":"train"},
        "run_id":"train","attempt_id":"attempt-001","operation":"SUBMIT_RUN","ready":true,
        "gate_expires_at":"2100-01-01T00:00:00Z", "request_digest":format!("sha256:{}","a".repeat(64)),
        "intent_digest":format!("sha256:{}","b".repeat(64)),"command_preview":["do not expose"],
        "preflight_summary":{"git_commit":"frozen-commit"}});
    let prepared = actions
        .save_plan(&plan, "2026-10-28T00:00:00Z", &"1".repeat(32))
        .unwrap();
    let before_events = actions.database().journal(&|| Ok(()), id, 100).unwrap();
    fixture.current("attempt-002");
    for _restart in 0..2 {
        let server = fixture.server();
        let client = server.client();
        let health = client.negotiate().unwrap();
        let submission = client.get(&format!("/api/submissions/{id}")).unwrap();
        assert_eq!(submission["attempt_id"], "attempt-001");
        assert_eq!(submission["status"], "PREPARED");
        assert_eq!(submission["next_action"], "AUTHORIZE");
        assert!(submission.get("command_preview").is_none());
        let saved = json!({"server":format!("http://{}/ml-expd", server.address),"project":"demo","run_id":"train", "config_sha256":"synthetic",
            "submission":{"submission_id":id,"attempt_id":"attempt-001"}});
        let selected = Selection {
            project: None,
            run: None,
            attempt: None,
        };
        let view = status::observe(&client, &health, Some(&saved), selected, true).unwrap();
        assert_eq!(view["attempt_id"], "attempt-001");
        assert_eq!(view["preparation"]["status"], "PREPARED");
        let selected = Selection {
            project: None,
            run: None,
            attempt: None,
        };
        let logs = read::observe(
            &client,
            &health,
            Some(&saved),
            selected,
            read::View::Logs {
                stream: "both".into(),
                lines: 80,
            },
        )
        .unwrap();
        assert!(logs.to_string().contains("attempt-001 exact output"));
        assert!(!logs.to_string().contains("attempt-002 exact output"));
        assert_eq!(
            actions.observe(id).unwrap()["execution"],
            prepared["execution"]
        );
        assert_eq!(
            actions.database().journal(&|| Ok(()), id, 100).unwrap(),
            before_events
        );
        for bad in ["action-../bad", "action-a", "action-AAAAAAAAAAAAAAAA"] {
            let (_, body) = server.raw(&format!("/api/submissions/{bad}"), &headers());
            assert!(body.get("detail").is_some());
        }
        assert_eq!(
            server
                .raw(&format!("/api/submissions/{id}?repair=true"), &headers())
                .0,
            422
        );
        assert_eq!(
            server
                .raw(
                    &format!("/api/submissions/{id}"),
                    "X-ML-Expd-Client-Protocol: 2\r\n"
                )
                .0,
            401
        );
    }
    let mut foreign = plan.clone();
    foreign["action_id"] = json!("action-1111111111111111");
    foreign["scope"]["project"] = json!("foreign");
    actions
        .save_plan(&foreign, "2026-10-28T00:00:00Z", &"2".repeat(32))
        .unwrap();
    let server = fixture.server();
    assert_eq!(
        server
            .raw("/api/submissions/action-1111111111111111", &headers())
            .0,
        404
    );
    drop(server);
    let fence = ml_exp_server::deletion::RunFence::new(&fixture.registry);
    let guard = fence.exclusive(true).unwrap();
    guard
        .mark(
            "demo",
            "train",
            &serde_json::Map::new(),
            "2026-10-28T00:00:00Z",
        )
        .unwrap();
    drop(guard);
    let server = fixture.server();
    assert_eq!(
        server.raw(&format!("/api/submissions/{id}"), &headers()).0,
        409
    );
}
fn http_failure(error: ClientError, status: u16) -> Box<ml_exp_core::diagnostic::HttpFailure> {
    let ClientError::Http(error) = error else {
        panic!("expected HTTP failure: {error}")
    };
    assert_eq!(error.status, status);
    error
}

#[test]
fn native_clients_pin_exact_attempt_and_keep_scientific_metric_dimensions() {
    let fixture = Fixture::new();
    fixture.journal_matching();
    let server = fixture.server();
    let client = server.client();
    let health = client.negotiate().unwrap();
    assert_eq!(health["workspace_id"], fixture.workspace);
    assert_eq!(health["scheduler_mutations"], false);
    assert!(
        !health["capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("result-collection.v1"))
    );
    assert_eq!(client.get("/api/projects").unwrap()[0]["project"], "demo");
    let initial = status::observe(&client, &health, None, selected(None), true).unwrap();
    assert_eq!(initial["attempt_id"], "attempt-001");
    assert_eq!(initial["training"]["status"], "COMPLETED");
    assert_eq!(
        initial["model"]["metric_record"]["source"],
        "canonical_record_journal"
    );
    assert_eq!(initial["model"]["last_metric"]["value"], 3.5);
    assert_eq!(initial["publication"]["download_available"], false);
    assert_eq!(
        initial["publication"]["diagnostic"],
        "ARTIFACT_PUBLICATION_NOT_MIGRATED"
    );
    assert_eq!(initial["publication"]["status"], "NOT_OBSERVED");
    // A producer's coarse failure_class is not an assessed failure fact.
    assert!(
        initial["evidence"]["collection"]
            .get("failure_class")
            .is_none()
    );
    fixture.current("attempt-002");
    let historical =
        status::observe(&client, &health, None, selected(Some("attempt-001")), true).unwrap();
    assert_eq!(historical["attempt_id"], "attempt-001");
    assert_eq!(historical["scheduler"]["state"], "SUCCEEDED");
    assert_eq!(historical["training"]["status"], "COMPLETED");
    let compact_historical =
        status::observe(&client, &health, None, selected(Some("attempt-001")), false).unwrap();
    assert_eq!(compact_historical["scheduler"]["state"], "SUCCEEDED");
    assert_eq!(compact_historical["training"]["status"], "COMPLETED");
    assert!(compact_historical.get("evidence").is_none());
    assert_eq!(
        historical["evidence"]["run"]["current_attempt_id"],
        "attempt-002"
    );
    let logs = read::observe(
        &client,
        &health,
        None,
        selected(Some("attempt-001")),
        read::View::Logs {
            stream: "stdout".into(),
            lines: 80,
        },
    )
    .unwrap();
    assert_eq!(
        logs["streams"]["stdout"]["lines"],
        json!(["attempt-001 exact output"])
    );
    let metrics = read::observe(
        &client,
        &health,
        None,
        selected(Some("attempt-001")),
        read::View::Metrics {
            max_points: 100,
            keys: None,
            names: Some(vec!["validation/loss".into()]),
        },
    )
    .unwrap();
    let records = metrics["records"].as_array().unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(metrics["source_attempt_id"], "attempt-001");
    for (record, (variant, value)) in records.iter().zip([("left", 2.5), ("right", 3.5)]) {
        assert_eq!(record["unit"], "nats/token");
        assert_eq!(record["variant_id"], variant);
        assert_eq!(record["protocol_id"], "synthetic-protocol-v1");
        assert_eq!(record["dataset_id"], "dataset.fixture");
        assert_eq!(record["checkpoint_id"], "checkpoint.fixture");
        assert_eq!(record["numerator"], 250);
        assert_eq!(record["denominator"], 100);
        assert_eq!(record["value"], json!(value));
    }
}

#[test]
fn watch_resolves_current_once_then_keeps_attempt_across_real_http_rounds() {
    let fixture = Fixture::new();
    let server = fixture.server();
    let client = server.client();
    let health = client.negotiate().unwrap();
    let elapsed = Cell::new(Duration::ZERO);
    let rounds = Cell::new(0);
    let result = watch::observe_with(
        Duration::from_secs(10),
        selected(None),
        |selection| {
            if rounds.get() > 0 {
                assert_eq!(selection.attempt, Some("attempt-001"));
            }
            rounds.set(rounds.get() + 1);
            status::observe(&client, &health, None, selection, false)
        },
        || elapsed.get(),
        |delay| elapsed.set(elapsed.get() + delay),
        |value| {
            assert_eq!(value["attempt_id"], "attempt-001");
            fixture.current("attempt-002");
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(rounds.get(), 2);
    assert_eq!(result["attempt_id"], "attempt-001");
    assert_eq!(result["publication"]["status"], "NOT_OBSERVED");
    assert_eq!(elapsed.get(), Duration::from_secs(10));
    assert!(!fixture.registry.join("tracking").exists());
}

#[test]
fn authentication_protocol_and_query_errors_never_echo_request_credentials() {
    let fixture = Fixture::new();
    let server = fixture.server();
    assert_eq!(server.raw("/api/projects", "").0, 401);
    assert_eq!(
        server
            .raw(
                "/api/projects",
                &format!("Authorization: Bearer {TOKEN}\r\n")
            )
            .0,
        426
    );
    assert_eq!(
        server
            .raw("/api/health", &format!("Authorization: Bearer {TOKEN}\r\n"))
            .0,
        200
    );
    for added in [
        format!("Authorization: Bearer {TOKEN}\r\n"),
        "X-ML-Expd-Client-Protocol: 2\r\n".into(),
    ] {
        let response = server.raw("/api/projects", &(headers() + &added));
        assert!(matches!(response.0, 401 | 426));
        assert!(!response.1.to_string().contains(TOKEN));
    }
    for path in [
        "/api/projects?unknown=yes",
        "/api/attempts/demo/train::attempt-001/logs?lines=1&lines=2",
        "/api/attempts/demo/train::attempt-001/metrics?max_points=0",
        "/api/attempts/demo/train::attempt-001/metrics?name=%FF",
        "/api/attempts/demo/train::attempt-001/metrics?name=%",
        "/api/runs/demo/train?unused=1",
    ] {
        let response = server.raw(path, &headers());
        assert_eq!(response.0, 422, "{path}: {}", response.1);
        assert!(!response.1.to_string().contains(TOKEN));
    }
    let missing = http_failure(
        server
            .client()
            .get("/api/attempts/demo/train::attempt-999/logs")
            .unwrap_err(),
        404,
    );
    assert!(!missing.retryable);
    assert_eq!(missing.error_code, "EXPERIMENT_NOT_FOUND");
    // A bare token in ordinary metadata is refused, not leaked or replaced.
    write_json(
        &fixture.project.join("research_project.yaml"),
        &json!({"project":"demo","title":TOKEN}),
    );
    let response = server.raw("/api/projects", &headers());
    assert_eq!(response.0, 409);
    assert_eq!(response.1["code"], "OUTWARD_SECRET_DETECTED");
    assert!(!response.1.to_string().contains(TOKEN));
    assert_eq!(server.raw("/api/projects", &headers()), response); // no automatic mutation or repair
    write_json(
        &fixture.project.join("research_project.yaml"),
        &json!({"project":"demo","title":"Synthetic science"}),
    );
    let journal = records::Store::open(
        &fixture.registry,
        fixture.workspace.strip_prefix("rust-preview-").unwrap(),
    )
    .unwrap();
    let guard = journal.guard("demo", "train").unwrap();
    journal
        .bind(&guard, &json!({"enabled":false}), &fixture.journal_config)
        .unwrap();
    let scope = journal.scope(&guard, "attempt-001").unwrap();
    journal
        .append(
            &guard,
            &scope,
            &[Event {
                event_id: "synthetic-token-in-scientific-unit".into(),
                payload: normalized_metric(
                    &json!({"name":"custom/value","unit":TOKEN,"value":1.25}),
                    &fixture.journal_config,
                )
                .unwrap(),
            }],
        )
        .unwrap();
    let error = http_failure(
        server
            .client()
            .get("/api/attempts/demo/train::attempt-001/metrics?format=records")
            .unwrap_err(),
        409,
    );
    assert_eq!(error.error_code, "OUTWARD_SECRET_DETECTED");
    assert!(!error.response_body.unwrap().to_string().contains(TOKEN));
}

#[test]
fn missing_locks_corrupt_canonical_and_file_aliases_fail_without_repair() {
    for variant in [
        "missing-lock",
        "corrupt-canonical",
        "hardlink-canonical",
        "symlink-canonical",
    ] {
        let fixture = Fixture::new();
        let server = fixture.server();
        let manifest = fixture.run().join("manifest.yaml");
        let before = fs::read(&manifest).unwrap();
        match variant {
            "missing-lock" => fs::remove_file(fixture.registry.join(".run-deletion.lock")).unwrap(),
            "corrupt-canonical" => {
                write_json(
                    &fixture.run().join("control_manifest.yaml"),
                    &fixture.manifest,
                );
                fs::write(&manifest, b"broken: [").unwrap();
            }
            "hardlink-canonical" => {
                fs::hard_link(&manifest, fixture.run().join("copied-manifest")).unwrap();
            }
            "symlink-canonical" => {
                fs::rename(&manifest, fixture.run().join("original-manifest")).unwrap();
                symlink("original-manifest", &manifest).unwrap();
            }
            _ => unreachable!(),
        }
        let client = server.client();
        let error = http_failure(client.get("/api/runs/demo/train").unwrap_err(), 409);
        assert_eq!(
            error.error_code,
            if variant == "missing-lock" {
                "READ_WORKSPACE_UNAVAILABLE"
            } else {
                "EXPERIMENT_EVIDENCE_UNAVAILABLE"
            }
        );
        if variant == "missing-lock" {
            assert!(!fixture.registry.join(".run-deletion.lock").exists());
        }
        if variant == "corrupt-canonical" {
            assert_eq!(fs::read(&manifest).unwrap(), b"broken: [");
        }
        if variant == "hardlink-canonical" {
            assert_eq!(fs::read(&manifest).unwrap(), before);
        }
        if variant == "symlink-canonical" {
            assert!(
                fs::symlink_metadata(&manifest)
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
        }
    }
}

#[test]
fn copied_or_wrong_science_journals_never_supply_live_metrics() {
    for variant in [
        "wrong-workspace",
        "wrong-source",
        "wrong-protocol",
        "wrong-runtime",
        "wrong-backend",
        "wrong-resources",
        "wrong-parameters",
        "wrong-inputs",
        "corrupt-journal",
    ] {
        let fixture = Fixture::new();
        let mut config = fixture.journal_config.clone();
        let seed = if variant == "wrong-workspace" {
            "00000000000000000000000000000000"
        } else {
            fixture.workspace.strip_prefix("rust-preview-").unwrap()
        };
        if variant == "wrong-source" {
            config["ml_expd"]["source_id"] = json!("source.foreign");
        }
        if variant == "wrong-protocol" {
            config["ml_expd"]["evaluation"]["protocol_id"] = json!("different-protocol");
        }
        if variant == "wrong-runtime" {
            config["ml_expd"]["runtime_id"] = json!("runtime.foreign");
        }
        if variant == "wrong-backend" {
            config["ml_expd"]["backend"] = json!("pai-dlc");
        }
        if variant == "wrong-resources" {
            config["ml_expd"]["resources"]["gpus"] = json!(8);
        }
        if variant == "wrong-parameters" {
            config["ml_expd"]["parameters"]["lr"] = json!(0.3);
        }
        if variant == "wrong-inputs" {
            config["ml_expd"]["inputs"] = json!([{"asset_id":"asset.foreign"}]);
        }
        fixture.journal(seed, &config);
        if variant == "corrupt-journal" {
            fs::write(
                fixture.registry.join("tracking/outbox.sqlite"),
                b"not sqlite",
            )
            .unwrap();
        }
        let server = fixture.server();
        let client = server.client();
        let error = http_failure(
            client
                .get("/api/attempts/demo/train::attempt-001/metrics?format=records")
                .unwrap_err(),
            409,
        );
        assert_eq!(
            error.error_code,
            if variant == "corrupt-journal" {
                "JOURNAL_UNAVAILABLE"
            } else {
                "JOURNAL_IDENTITY_MISMATCH"
            }
        );
        assert!(!error.to_string().contains(TOKEN));
        assert!(fixture.registry.join("tracking/outbox.sqlite").exists());
    }
}

#[test]
fn malformed_internal_journal_observations_never_become_complete_science() {
    let valid = json!({"name":"validation/loss","unit":"nats/token","value":2.5,"status":"VALID","source":null,
        "protocol_id":"synthetic-protocol-v1","variant_id":null});
    for (field, value) in [
        ("unit", json!("")),
        ("value", Value::Null),
        ("protocol_id", json!(17)),
        ("variant_id", json!({"unexpected":"object"})),
    ] {
        let fixture = Fixture::new();
        fixture.journal_matching();
        let store = records::Store::open(
            &fixture.registry,
            fixture.workspace.strip_prefix("rust-preview-").unwrap(),
        )
        .unwrap();
        let guard = store.guard("demo", "train").unwrap();
        let scope = store.scope(&guard, "attempt-001").unwrap();
        let mut malformed = valid.clone();
        malformed[field] = value;
        store
            .append(
                &guard,
                &scope,
                &[Event {
                    event_id: "malformed-internal-record".into(),
                    payload: json!({"kind":"metrics","observations":[malformed]}),
                }],
            )
            .unwrap();
        drop(guard);
        let server = fixture.server();
        let client = server.client();
        for path in [
            "/api/runs/demo/train",
            "/api/attempts/demo/train::attempt-001/metrics?format=records",
        ] {
            let error = http_failure(client.get(path).unwrap_err(), 409);
            assert_eq!(
                error.error_code, "EXPERIMENT_EVIDENCE_UNAVAILABLE",
                "{field}: {path}"
            );
            assert!(!error.to_string().contains(TOKEN));
        }
        assert!(fixture.registry.join("tracking/outbox.sqlite").exists());
    }
}

fn file_inventory(path: &Path) -> Vec<(PathBuf, String)> {
    let mut values = Vec::new();
    for item in fs::read_dir(path).unwrap() {
        let path = item.unwrap().path();
        let metadata = fs::symlink_metadata(&path).unwrap();
        if metadata.is_dir() {
            values.extend(file_inventory(&path));
        } else if metadata.is_file() {
            values.push((
                path.clone(),
                ml_exp_core::slurm::helper::sha256_bytes(&fs::read(&path).unwrap()),
            ));
        }
    }
    values.sort();
    values
}

fn rename_fixture_run(fixture: &Fixture, name: &str) -> read_config::Settings {
    let mut settings = fixture.settings();
    let target = fixture.project.join("runs").join(name);
    fs::rename(fixture.run(), &target).unwrap();
    for relative in [
        "manifest.yaml",
        "status.json",
        "attempts/attempt-001/attempt.yaml",
        "attempts/attempt-001/status.json",
        "attempts/attempt-001/collection.json",
        "attempts/attempt-002/attempt.yaml",
        "attempts/attempt-002/status.json",
        "attempts/attempt-002/collection.json",
    ] {
        let path = target.join(relative);
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["run_id"] = json!(name);
        write_json(&path, &value);
    }
    settings.projects[0].runs[0].run_id = name.into();
    settings.projects[0].runs[0].relative_components = vec![name.into()];
    settings
}

#[test]
fn unified_api_reads_both_namespaces_and_reserves_historical_identities() {
    let old = Fixture::new();
    old.journal_matching();
    let history = ml_exp_server::history::Bridge::open(old.historical_settings(true)).unwrap();
    let before = file_inventory(old._temp.path());
    let native = Fixture::new();
    let settings = rename_fixture_run(&native, "new-trial");
    for _ in 0..2 {
        let api = Api::open(settings.clone())
            .unwrap()
            .with_history(history.clone())
            .unwrap();
        let server = Server::start(Auth::load(&native.token).unwrap(), api);
        let client = server.client();
        let health = client.negotiate().unwrap();
        assert_eq!(health["historical_read_only"], false);
        assert_eq!(health["historical_records_attached"], true);
        assert_eq!(
            client.get("/api/runs/demo/train").unwrap()["run_id"],
            "train"
        );
        assert_eq!(
            client.get("/api/runs/demo/new-trial").unwrap()["run_id"],
            "new-trial"
        );
        let metrics = client
            .get("/api/attempts/demo/train::attempt-001/metrics?format=records")
            .unwrap();
        assert_eq!(metrics["records"].as_array().unwrap().len(), 2);
        let tracking = client.get("/api/runs/demo/train/tracking").unwrap();
        assert_eq!(tracking["read_only"], true);
        assert_eq!(tracking["scopes"].as_array().unwrap().len(), 2);
        assert_eq!(tracking["scopes"][0]["pending"], 2);
        let plan =
            ml_exp_client::submissions::plan(ml_exp_client::submissions::Command::Tracking {
                project: "demo",
                run: "train",
            })
            .unwrap();
        assert_eq!(
            ml_exp_client::submissions::execute(&client, &plan, &health).unwrap(),
            tracking
        );
        let projects = client.get("/api/projects").unwrap();
        assert_eq!(projects.as_array().unwrap().len(), 1);
        assert_eq!(projects[0]["run_counts"]["SUCCEEDED"], 2);
        assert_eq!(server.raw("/api/runs/demo/missing", &headers()).0, 404);
        assert_eq!(
            history.require_available("demo", "train").unwrap_err().code,
            "RUN_ID_RESERVED_BY_HISTORY"
        );
        history.require_available("demo", "fresh").unwrap();
        assert_eq!(file_inventory(old._temp.path()), before);
    }
    // Deletion remains reserved after the original payload has disappeared.
    let fence = ml_exp_server::deletion::RunFence::new(&old.registry);
    fence
        .exclusive_existing(true)
        .unwrap()
        .mark(
            "demo",
            "train",
            &serde_json::Map::new(),
            "2026-10-29T00:00:00Z",
        )
        .unwrap();
    fs::remove_dir_all(old.run()).unwrap();
    assert!(history.require_available("demo", "train").is_err());
    let api = Api::open(settings).unwrap().with_history(history).unwrap();
    let server = Server::start(Auth::load(&native.token).unwrap(), api);
    assert_eq!(server.raw("/api/runs/demo/train", &headers()).0, 409);
}

#[test]
fn unified_api_refuses_duplicate_runs_and_overlapping_storage() {
    let old = Fixture::new();
    let history = ml_exp_server::history::Bridge::open(old.historical_settings(false)).unwrap();
    let native = Fixture::new();
    let mut overlapping = native.settings();
    overlapping.projects[0].run_roots = vec![old.project.join("runs")];
    assert!(
        Api::open(overlapping)
            .unwrap()
            .with_history(history.clone())
            .is_err()
    );
    let api = Api::open(native.settings())
        .unwrap()
        .with_history(history)
        .unwrap();
    let server = Server::start(Auth::load(&native.token).unwrap(), api);
    for path in [
        "/api/runs/demo/train",
        "/api/projects",
        "/api/runs/demo/train/attempts/attempt-001/checkpoints",
    ] {
        let response = server.raw(path, &headers());
        assert_eq!(response.0, 409, "{path}: {}", response.1);
        assert_eq!(response.1["code"], "RUN_ID_RESERVED_BY_HISTORY");
    }
}

#[test]
fn historical_namespace_reads_original_metrics_and_refuses_every_mutation_route() {
    let fixture = Fixture::new();
    fixture.journal_matching();
    let settings = fixture.historical_settings(true);
    let before = file_inventory(fixture._temp.path());
    for _restart in 0..2 {
        let api = ml_exp_server::history::open(settings.clone()).unwrap();
        assert!(api.is_historical());
        let server = Server::start(Auth::load(&fixture.token).unwrap(), api);
        let health = server.client().negotiate().unwrap();
        assert_eq!(health["historical_read_only"], true);
        assert_eq!(health["scheduler_mutations"], false);
        assert_eq!(health["run_writes"], false);
        assert_eq!(health["source_imports"], false);
        let policy = server.client().get("/api/actions/policy").unwrap();
        for key in [
            "allow_project_writes",
            "allow_source_imports",
            "allow_scheduler_mutations",
            "allow_local_evidence_rebuild",
            "allow_resource_deletions",
        ] {
            assert_eq!(policy[key], false, "historical permission: {key}");
        }
        assert_eq!(policy["historical_records_read_only"], true);
        assert_eq!(health["projects"], 1);
        assert!(
            health["workspace_id"]
                .as_str()
                .unwrap()
                .starts_with("historical-")
        );
        let client = server.client();
        assert_eq!(client.get("/api/projects").unwrap()[0]["project"], "demo");
        let lifecycle = client.get("/api/project-lifecycle").unwrap();
        assert_eq!(lifecycle["read_only"], true);
        assert_eq!(lifecycle["projects"][0]["state"], "ACTIVE");
        let list = ml_exp_client::projects::plan(ml_exp_client::projects::Command::List).unwrap();
        assert_eq!(
            ml_exp_client::projects::execute(&client, &list, &health).unwrap(),
            lifecycle
        );
        let pause = ml_exp_client::projects::plan(ml_exp_client::projects::Command::Change {
            project: "demo",
            action: ml_exp_client::projects::Action::Pause,
            reason: "not allowed in history",
            expected_revision: None,
        })
        .unwrap();
        assert!(ml_exp_client::projects::execute(&client, &pause, &health).is_err());
        let metrics = client
            .get("/api/attempts/demo/train::attempt-001/metrics?format=records")
            .unwrap();
        assert_eq!(metrics["records"].as_array().unwrap().len(), 2);
        assert_eq!(metrics["records"][0]["variant_id"], "left");
        assert_eq!(metrics["records"][0]["value"], 2.5);
        assert_eq!(metrics["records"][1]["variant_id"], "right");
        let logs = client
            .get("/api/attempts/demo/train::attempt-001/logs")
            .unwrap();
        assert!(
            serde_json::to_string(&logs)
                .unwrap()
                .contains("attempt-001 exact output")
        );
        for path in [
            "/api/project-lifecycle/demo/pause",
            "/api/projects/demo/runs",
            "/api/projects/demo/runtimes",
            "/api/submissions/action-0123456789abcdef/execute",
            "/api/worker-exchange/not-a-capability",
            "/api/dataset-exchange/not-a-capability",
        ] {
            let (status, value) = server.request("POST", path, &headers(), "{}");
            assert_eq!(status, 405, "{path}");
            assert_eq!(value["code"], "HISTORICAL_NAMESPACE_READ_ONLY");
            assert_eq!(
                server
                    .request("POST", path, "X-ML-Expd-Client-Protocol: 2\r\n", "{}")
                    .0,
                401
            );
        }
        assert_eq!(file_inventory(fixture._temp.path()), before);
    }
    assert_eq!(file_inventory(fixture._temp.path()), before);
}

#[test]
fn historical_reader_pins_journal_and_lifecycle_without_native_marker_fallback() {
    let fixture = Fixture::new();
    fixture.journal_matching();
    let settings = fixture.historical_settings(true);
    let mut wrong = settings.clone();
    wrong.journal_workspace_id = Some("0".repeat(32));
    assert!(ml_exp_server::history::open(wrong).is_err());
    let mut unexpected = settings.clone();
    unexpected.journal_workspace_id = None;
    assert!(ml_exp_server::history::open(unexpected).is_err());
    assert!(Api::open(settings.reads.clone()).is_err());
    let server = Server::start(
        Auth::load(&fixture.token).unwrap(),
        ml_exp_server::history::open(settings.clone()).unwrap(),
    );
    let registry = fixture.registry.join("registry.json");
    let original = fs::read(&registry).unwrap();
    let mut value: Value = serde_json::from_slice(&original).unwrap();
    value["projects"][0]["state"] = json!("ARCHIVED");
    write_json(&registry, &value);
    assert_eq!(server.client().get("/api/projects").unwrap(), json!([]));
    assert_eq!(server.client().negotiate().unwrap()["projects"], 0);
    assert_eq!(
        http_failure(
            server.client().get("/api/runs/demo/train").unwrap_err(),
            404
        )
        .error_code,
        "PROJECT_INACTIVE"
    );
    fs::write(&registry, &original).unwrap();
    value["projects"][0]["state"] = json!("ACTIVE");
    value["projects"][0]["project_file"] = json!("/foreign/unapproved/project.yaml");
    write_json(&registry, &value);
    assert_eq!(
        http_failure(server.client().get("/api/projects").unwrap_err(), 409).error_code,
        "PROJECT_SOURCE_BINDING_MISMATCH"
    );
    fs::write(&registry, original).unwrap();
    write_json(
        &fixture.registry.join(".rust-preview-workspace.json"),
        &json!({"contract":"wrong"}),
    );
    assert_eq!(
        http_failure(server.client().get("/api/projects").unwrap_err(), 409).error_code,
        "READ_WORKSPACE_CHANGED"
    );
    assert!(ml_exp_server::history::open(settings).is_err());
}

#[test]
fn historical_reader_without_a_journal_does_not_create_or_accept_a_later_one() {
    let fixture = Fixture::new();
    let settings = fixture.historical_settings(false);
    let before = file_inventory(fixture._temp.path());
    let server = Server::start(
        Auth::load(&fixture.token).unwrap(),
        ml_exp_server::history::open(settings).unwrap(),
    );
    assert_eq!(
        server
            .client()
            .get("/api/attempts/demo/train::attempt-001/metrics?format=records")
            .unwrap()["records"],
        json!([])
    );
    assert_eq!(file_inventory(fixture._temp.path()), before);
    fixture.journal_matching();
    assert_eq!(
        http_failure(
            server
                .client()
                .get("/api/attempts/demo/train::attempt-001/metrics?format=records")
                .unwrap_err(),
            409
        )
        .error_code,
        "READ_WORKSPACE_CHANGED"
    );
}

#[test]
fn historical_config_inspection_uses_actual_binary_without_creating_files_or_new_identity() {
    let fixture = Fixture::new();
    fixture.journal_matching();
    let settings = fixture.historical_settings(true);
    let config = fixture._temp.path().join("original-reads.json");
    write_json(&config, &serde_json::to_value(&settings.reads).unwrap());
    let before = file_inventory(fixture._temp.path());
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ml-expd-rs"))
        .arg("--inspect-historical-config")
        .arg(&config)
        .env_clear()
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap(),
        serde_json::to_value(&settings).unwrap()
    );
    assert_eq!(file_inventory(fixture._temp.path()), before);
    for incompatible in [
        "--source-archives",
        "--project-lifecycle",
        "--administration",
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_ml-expd-rs"))
            .arg("--historical-read-config")
            .arg(&config)
            .arg("--token-file")
            .arg(&fixture.token)
            .arg(incompatible)
            .env_clear()
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
    }
    assert_eq!(file_inventory(fixture._temp.path()), before);
}

#[test]
fn historical_result_inventory_is_read_only_exact_and_preserves_release_evidence() {
    let fixture = Fixture::new();
    let settings = fixture.historical_settings(false);
    let directory = fixture.registry.join("artifact-transfers/demo/train");
    private_directory(&directory);
    let path = directory.join("attempt-001.json");
    let sha = "a".repeat(64);
    let mut transfer = json!({"project":"demo","run_id":"train","attempt_id":"attempt-001",
        "token":"private-transfer-capability","run_dir":"/private/old-service",
        "receipt":{"sha256":sha,"bytes":10240,"object_key":format!("demo/train/attempt-001/{sha}.tar"),
            "files":[{"path":"final.pt","sha256":"b".repeat(64),"bytes":128}]},
        "results_ready":{"project":"demo","run_id":"train","attempt_id":"attempt-001","exit_code":0}});
    write_json(&path, &transfer);
    let before = file_inventory(fixture._temp.path());
    let server = Server::start(
        Auth::load(&fixture.token).unwrap(),
        ml_exp_server::history::open(settings).unwrap(),
    );
    let route = "/api/runs/demo/train/attempts/attempt-001/files";
    let result = server.client().get(route).unwrap();
    assert_eq!(result["files"][0]["path"], "outputs/final.pt");
    assert_eq!(result["available"], true);
    assert_eq!(result["training"]["exit_code"], 0);
    assert!(!result.to_string().contains("private-transfer-capability"));
    assert!(!result.to_string().contains("/private/old-service"));
    assert_eq!(
        server
            .client()
            .get(&format!("{route}?checksums=true"))
            .unwrap(),
        result
    );
    assert_eq!(file_inventory(fixture._temp.path()), before);
    assert_eq!(
        server
            .client()
            .get("/api/runs/demo/train/attempts/attempt-002/files")
            .unwrap()["collection_required"],
        true
    );
    assert_eq!(
        http_failure(
            server
                .client()
                .get("/api/runs/demo/train/attempts/attempt-003/files")
                .unwrap_err(),
            404
        )
        .error_code,
        "ATTEMPT_NOT_FOUND"
    );
    assert_eq!(
        http_failure(
            server
                .client()
                .get(&format!("{route}?checksums=yes"))
                .unwrap_err(),
            422
        )
        .error_code,
        "INVALID_READ_QUERY"
    );
    transfer["retention"] = json!({"status":"RELEASED","sha256":sha});
    write_json(&path, &transfer);
    let released_before = file_inventory(fixture._temp.path());
    let released = server.client().get(route).unwrap();
    assert_eq!(released["available"], false);
    assert_eq!(released["collection_required"], false);
    assert_eq!(released["storage_status"], "RELEASED");
    assert_eq!(released["files"], result["files"]);
    assert_eq!(file_inventory(fixture._temp.path()), released_before);
    transfer["attempt_id"] = json!("attempt-002");
    write_json(&path, &transfer);
    assert_eq!(
        http_failure(server.client().get(route).unwrap_err(), 409).error_code,
        "ARTIFACT_EVIDENCE_UNAVAILABLE"
    );
    transfer["attempt_id"] = json!("attempt-001");
    write_json(&path, &transfer);
    fs::remove_file(&path).unwrap();
    symlink("/private/forbidden", &path).unwrap();
    assert_eq!(
        http_failure(server.client().get(route).unwrap_err(), 409).error_code,
        "ARTIFACT_EVIDENCE_UNAVAILABLE"
    );
}

#[test]
fn historical_download_signs_only_exact_available_receipts_with_a_pinned_profile() {
    let fixture = Fixture::new();
    let mut settings = fixture.historical_settings(false);
    let profile = fixture._temp.path().join("original-s3.json");
    write_json(
        &profile,
        &json!({"public_endpoint":"https://example.com","bucket":"results","region":"garage","access_key":"fixture-key","secret_key":"fixture-secret"}),
    );
    settings.artifact_store = Some(ml_exp_server::history::ObjectConfig {
        profile_file: profile.clone(),
        profile_sha256: ml_exp_io::sha256_file(&profile).unwrap(),
    });
    let directory = fixture.registry.join("artifact-transfers/demo/train");
    private_directory(&directory);
    let path = directory.join("attempt-001.json");
    let sha = "a".repeat(64);
    let mut transfer = json!({"project":"demo","run_id":"train","attempt_id":"attempt-001",
        "receipt":{"sha256":sha,"bytes":10240,"object_key":format!("demo/train/attempt-001/{sha}.tar"),
            "files":[{"path":"final.pt","bytes":128}]}});
    write_json(&path, &transfer);
    let before = file_inventory(fixture._temp.path());
    let server = Server::start(
        Auth::load(&fixture.token).unwrap(),
        ml_exp_server::history::open(settings).unwrap(),
    );
    let route = "/api/runs/demo/train/attempts/attempt-001/artifacts/download";
    let result = server.client().get(route).unwrap();
    assert_eq!(result["transport"], "s3-presigned-get");
    assert_eq!(result["sha256"], sha);
    assert_eq!(result["files"][0]["sha256"], Value::Null);
    assert_eq!(result["acknowledgement_supported"], false);
    assert!(!result.to_string().contains("fixture-secret"));
    assert_eq!(file_inventory(fixture._temp.path()), before);
    assert_eq!(
        http_failure(
            server
                .client()
                .get("/api/runs/demo/train/attempts/attempt-002/artifacts/download")
                .unwrap_err(),
            404
        )
        .error_code,
        "ARTIFACT_UNAVAILABLE"
    );
    transfer["retention"] = json!({"status":"RELEASED","sha256":sha});
    write_json(&path, &transfer);
    assert_eq!(
        http_failure(server.client().get(route).unwrap_err(), 410).error_code,
        "ARTIFACT_RELEASED"
    );
    transfer.as_object_mut().unwrap().remove("retention");
    write_json(&path, &transfer);
    fs::write(&profile, b"{}").unwrap();
    assert_eq!(
        http_failure(server.client().get(route).unwrap_err(), 409).error_code,
        "ARTIFACT_EVIDENCE_UNAVAILABLE"
    );
}

#[test]
fn historical_cold_wal_is_refused_without_auxiliary_creation_or_weaker_fallback() {
    let fixture = Fixture::new();
    fixture.journal_matching();
    let settings = fixture.historical_settings(true);
    let connection =
        rusqlite::Connection::open(fixture.registry.join("tracking/outbox.sqlite")).unwrap();
    connection.execute_batch("PRAGMA journal_mode=WAL").unwrap();
    drop(connection);
    let before = file_inventory(fixture._temp.path());
    assert!(ml_exp_server::history::open(settings.clone()).is_err());
    assert!(ml_exp_server::history::inspect(settings.reads).is_err());
    assert_eq!(file_inventory(fixture._temp.path()), before);
}
