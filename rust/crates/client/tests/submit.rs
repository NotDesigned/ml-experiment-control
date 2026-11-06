//! `submit` against a real loopback socket; no scheduler used. What matters most is
//! which requests it sends, and how many of them can start a job.
use ml_exp_client::{
    Client, Error,
    submit::{self, Hooks, Spec},
};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

const ID: &str = "action-0123456789abcdef";
const SECOND: &str = "action-fedcba9876543210";
const RUNTIME: &str = "runtime.3333333333333333333333333333333333333333333333333333333333333333";
const IMAGE: &str = "docker.io/library/python@sha256:1111111111111111111111111111111111111111111111111111111111111111";

/// What the fake server knows. `None` as a reply closes the socket without answering.
struct World {
    prepared: bool,
    continues: u32,
    continues_until_ready: u32,
    status: &'static str,
    next_action: &'static str,
    build_failed: bool,
    lose_execute_reply: bool,
    /// The request is lost on the way, so the server never acts on it.
    execute_never_arrives: bool,
    /// The server answers and refuses, as it does after a project edit.
    execute_refused: bool,
    /// Cloud reports through its observation; Slurm through the Submission's result.
    cloud: bool,
    /// Whether the server's background polling reports a Slurm job's later states.
    polling: bool,
    /// A build left running by an API restart: no receipt yet, with the builder alive.
    restarted: bool,
    builder_gone: bool,
    no_executors: bool,
    cpu_limit: u32,
    /// The first prepared request, which any later one must equal.
    prepared_request: Option<Value>,
    /// A failed provider read keeps the old state beside this.
    provider_read_failing: bool,
    prepare_never_arrives: bool,
    /// The backend's states, oldest first; the last one stays.
    job_states: Vec<(&'static str, &'static str)>,
    reads_down: bool,
    progress_reads: usize,
    reads: usize,
    /// What a preparation names, which stays the first Submission after a retry.
    preparation_submission: Option<&'static str>,
    action: &'static str,
    attempt: &'static str,
    run_id: String,
}
impl World {
    fn new() -> Self {
        Self {
            prepared: false,
            continues: 0,
            continues_until_ready: 1,
            status: "PREPARED",
            next_action: "EXECUTE",
            build_failed: false,
            lose_execute_reply: false,
            execute_never_arrives: false,
            execute_refused: false,
            cloud: true,
            polling: true,
            restarted: false,
            builder_gone: false,
            no_executors: false,
            cpu_limit: 8,
            prepared_request: None,
            provider_read_failing: false,
            prepare_never_arrives: false,
            job_states: vec![("RUNNING", "Running"), ("SUCCEEDED", "Succeeded")],
            reads_down: false,
            progress_reads: 0,
            reads: 0,
            preparation_submission: None,
            action: ID,
            attempt: "attempt-001",
            run_id: "train".into(),
        }
    }
    fn submission(&self) -> Value {
        let (state, own) = self.job_states[0];
        let mut value = json!({"submission_id":self.action,"project":"demo","run_id":self.run_id,
            "attempt_id":self.attempt,
            "operation":"SUBMIT_RUN","status":self.status,"next_action":self.next_action,
            "confirmation":format!("EXECUTE {}",self.action),
            "preflight_summary":{"resources":{"gpus":1},"max_gpu_hours":0.5},
            "execution":{"status":self.status,"revision":0,
                "result":{"submission":{"backend_job_id":"123"}}}});
        if self.status == "VERIFIED" {
            if self.cloud {
                value["cloud_observation"] = json!({"observation":{"job_id":"job",
                    "provider_state":own,"state":state,"queue":null},
                    "diagnostic":if self.provider_read_failing {json!("read failed")} else {Value::Null}});
            } else {
                let shown = if self.polling { state } else { "QUEUED" };
                value["execution"]["result"]["observation"] = json!({"status":{"state":shown}});
                if self.polling {
                    value["native_scheduler_observation"] = json!(true);
                }
            }
        }
        if !self.cloud {
            value["cancellation"] = json!({"request_sha256":format!("sha256:{}","4".repeat(64)),
                "job_id":"123","requested_at":"2026-11-06T00:00:00Z",
                "termination_verified":false,"observation":null});
        }
        value
    }
    /// The backend moves on to its next state.
    fn advance_job(&mut self) {
        if self.job_states.len() > 1 {
            self.job_states.remove(0);
        }
    }
    fn view(&self, path: &str) -> Value {
        let id = path.rsplit('/').find(|p| p.starts_with("preparation-"));
        let ready = self.continues > self.continues_until_ready;
        json!({"project":"demo","preparation_id":id,"run_id":self.run_id,
            "runtime":{"runtime_id":RUNTIME,
                "status":if self.build_failed || (self.restarted && !ready) {"RECONCILE_REQUIRED"} else if ready {"READY"} else {"EXECUTING"},
                "build_error":if self.build_failed {
                    json!({"code":"BUILD_RECEIPT_MISSING","build_failure":{"code":"BAD_DOCKERFILE"}})
                } else if self.restarted && !ready {
                    json!({"code":"BUILD_RECEIPT_MISSING"})
                } else {Value::Null}},
            "phase":"ENVIRONMENT_PREPARATION","status":"PREPARING",
            "confirmation":format!("PREPARE {}",id.unwrap_or_default()),
            "submission":if ready {
                let mut value = self.submission();
                if let Some(id) = self.preparation_submission {
                    value["submission_id"] = json!(id);
                }
                value
            } else {Value::Null},
            "frozen":self.prepared_request.as_ref().map(|request| {
                let resources = &request["run"]["resources"];
                json!({"executor":request["run"]["executor"],"gpus":resources["gpus"],
                    "cpus":resources["cpus"],"memory_gb":resources["memory_gb"],
                    "max_gpu_hours":request["approval"]["max_gpu_hours"]})
            }),
            "training_authorized":false,"scheduler_submitted":false})
    }
    fn reply(&mut self, method: &str, path: &str, body: &[u8]) -> Option<(u16, Value)> {
        let route = path.split('?').next().unwrap_or(path);
        Some(match (method, route) {
            ("GET", "/api/health") => (
                200,
                json!({"capabilities":["native-submissions.v1","source-archive.v1",
                    "native-experiment-preparations.v1","native-runtime-builds.v1",
                    "native-scheduler-observation.v1","native-attempt-retry.v1",
                    "native-cancellation.v1"],
                    "scheduler_mutations":true}),
            ),
            ("GET", "/api/executors") if self.no_executors => (200, json!({"executors":[]})),
            ("GET", "/api/executors") => (
                200,
                json!({"executors":[{"executor_id":"pai","enabled":true,"run_creation":true,
                    "limits":{"gpus":1,"cpus":self.cpu_limit,"memory_gb":32,"walltime_seconds":3600}}]}),
            ),
            ("GET", r) if r.contains("/source-imports/") => (404, json!({"code":"NOT_FOUND"})),
            ("POST", "/api/source-imports/archive") => {
                let sha = path.rsplit("sha256=").next().unwrap_or_default();
                let digest = "2".repeat(64);
                (
                    200,
                    json!({"source_id":format!("source.{digest}"),"project":"demo",
                        "tree_digest":format!("sha256:{digest}"),
                        "observation":{"kind":"archive","sha256":sha},
                        "registration":{"existing":true}}),
                )
            }
            ("GET", r) if r.contains("/experiment-preparations/") => {
                if self.prepared {
                    (200, self.view(r))
                } else {
                    (404, json!({"code":"CLOUD_SUBMISSION_NOT_FOUND"}))
                }
            }
            ("POST", "/api/projects/demo/experiment-preparations") => {
                if self.prepare_never_arrives {
                    return None;
                }
                self.prepared = true;
                let request: Value = serde_json::from_slice(body).unwrap();
                match &self.prepared_request {
                    Some(first) if *first != request => {
                        return Some((409, json!({"code":"CLOUD_EVIDENCE_INVALID"})));
                    }
                    _ => self.prepared_request = Some(request.clone()),
                }
                let id = ml_exp_core::preparation::id("demo", request["request_id"].as_str()?);
                (200, json!({"project":"demo","preparation_id":id.unwrap()}))
            }
            ("POST", r) if r.ends_with("/continue") => {
                self.continues += 1;
                (200, self.view(r))
            }
            ("GET", r) if r.starts_with("/api/submissions/") && r.matches('/').count() == 3 => {
                let mut value = self.submission();
                let id = r.rsplit('/').next().unwrap_or_default();
                value["submission_id"] = json!(id);
                value["confirmation"] = json!(format!("EXECUTE {id}"));
                // Slurm's view has the latest state; a cloud view waits for a refresh.
                if !self.cloud && self.polling && self.status == "VERIFIED" {
                    self.advance_job();
                }
                (200, value)
            }
            ("GET", r) if r.starts_with("/api/runs/") && r.ends_with("/submission") => {
                (200, self.submission())
            }
            ("POST", r) if r.ends_with("/cancel") => (200, self.submission()),
            ("POST", r) if r.ends_with("/retry") => {
                let (state, _) = self.job_states[0];
                if !matches!(state, "SUCCEEDED" | "FAILED" | "CANCELLED") {
                    return Some((409, json!({"code":"CLOUD_EVIDENCE_INVALID"})));
                }
                let request: ml_exp_core::first_submission::native::retry::Request =
                    serde_json::from_slice(body).unwrap();
                let (run, _) = request.parent().unwrap();
                (
                    200,
                    json!({"parent_submission_id":self.action,"run_id":run,"attempt_id":"attempt-002",
                        "request_sha256":request.identity(self.action).unwrap(),
                        "resume_from":request.resume_from,"restart_from":request.restart_from,
                        "max_gpu_hours":request.max_gpu_hours,"retry_state":"CREATED",
                        "scheduler_mutation_performed":false,"next_action":"PREPARE_NEW_ACTION",
                        "reused":false,"project":"demo","reserved_gpu_seconds":0,
                        "run_sha256":format!("sha256:{}","5".repeat(64)),
                        "attempt_sha256":format!("sha256:{}","6".repeat(64))}),
                )
            }
            ("POST", r) if r.ends_with("/submissions") => {
                self.action = SECOND;
                self.attempt = "attempt-002";
                self.status = "PREPARED";
                self.next_action = "EXECUTE";
                (200, self.submission())
            }
            ("POST", r) if r.ends_with("/execute") => {
                if self.execute_never_arrives {
                    return None;
                }
                if self.execute_refused {
                    return Some((409, json!({"code":"SUBMISSION_EVIDENCE_UNAVAILABLE"})));
                }
                self.status = "EXECUTING";
                self.next_action = "RECONCILE";
                if self.lose_execute_reply {
                    return None;
                }
                self.status = "VERIFIED";
                self.next_action = "NONE";
                (200, self.submission())
            }
            ("POST", r) if r.ends_with("/reconcile") => {
                self.status = "VERIFIED";
                self.next_action = "NONE";
                (200, self.submission())
            }
            ("POST", r) if r.ends_with("/refresh") => {
                if self.reads_down {
                    return Some((503, json!({"code":"BUSY"})));
                }
                self.reads += 1;
                if self.reads > 1 {
                    self.advance_job();
                }
                (200, self.submission())
            }
            ("GET", r) if r.ends_with("/progress") => {
                self.progress_reads += 1;
                let lines = if self.progress_reads == 1 {
                    json!(["step 1"])
                } else {
                    json!(["step 1", "step 2"])
                };
                (
                    200,
                    json!({"runtime_id":RUNTIME,"lines":lines,
                        "progress":{"phase":"BUILDING_ON_DSW","message":"Docker build running",
                            "status":if self.builder_gone {"RECONCILE_REQUIRED"} else {"EXECUTING"}}}),
                )
            }
            _ => (404, json!({"code":"NOT_FOUND"})),
        })
    }
}

/// Method, path and body of each request, in order.
type Requests = Mutex<Vec<(String, String, Vec<u8>)>>;
struct Fake {
    world: Arc<Mutex<World>>,
    requests: Arc<Requests>,
    stop: Arc<AtomicBool>,
    address: std::net::SocketAddr,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl Fake {
    fn start(world: World) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let world = Arc::new(Mutex::new(world));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = {
            let (world, requests, stop) = (world.clone(), requests.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((socket, _)) => serve(socket, &world, &requests),
                        Err(_) => std::thread::sleep(Duration::from_millis(1)),
                    }
                }
            })
        };
        Self {
            world,
            requests,
            stop,
            address,
            worker: Some(worker),
        }
    }
    fn client(&self) -> Client {
        Client::new(
            &format!("http://{}", self.address),
            "synthetic-token".into(),
        )
        .unwrap()
    }
    fn count(&self, method: &str, suffix: &str) -> usize {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(m, path, _)| m == method && path.split('?').next().unwrap().ends_with(suffix))
            .count()
    }
    fn mutations(&self) -> usize {
        let requests = self.requests.lock().unwrap();
        requests.iter().filter(|(m, ..)| m == "POST").count()
    }
    fn prepared_request(&self) -> Value {
        let requests = self.requests.lock().unwrap();
        let (.., body) = requests
            .iter()
            .find(|(m, path, _)| m == "POST" && path.ends_with("/experiment-preparations"))
            .unwrap();
        serde_json::from_slice(body).unwrap()
    }
    fn execute_request(&self) -> Value {
        let requests = self.requests.lock().unwrap();
        let (.., body) = requests
            .iter()
            .find(|(m, path, _)| m == "POST" && path.ends_with("/execute"))
            .unwrap();
        serde_json::from_slice(body).unwrap()
    }
}
impl Drop for Fake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
    }
}
fn serve(mut socket: TcpStream, world: &Mutex<World>, requests: &Requests) {
    socket.set_nonblocking(false).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut head = Vec::new();
    while !head.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        if socket.read_exact(&mut byte).is_err() {
            return;
        }
        head.push(byte[0]);
    }
    let head = String::from_utf8(head).unwrap();
    let mut words = head.split_whitespace();
    let (method, path) = (words.next().unwrap(), words.next().unwrap());
    let size: usize = head
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .map(|size| size.parse().unwrap())
        })
        .unwrap_or(0);
    let mut body = vec![0; size];
    socket.read_exact(&mut body).unwrap();
    requests
        .lock()
        .unwrap()
        .push((method.into(), path.into(), body.clone()));
    let Some((status, value)) = world.lock().unwrap().reply(method, path, &body) else {
        return;
    };
    let text = value.to_string();
    let _ = write!(
        socket,
        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    );
}

struct Files {
    root: tempfile::TempDir,
}
impl Files {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        std::fs::create_dir(&source).unwrap();
        std::fs::write(source.join("Dockerfile"), format!("FROM {IMAGE}\n")).unwrap();
        std::fs::write(source.join("train.py"), "print('train')\n").unwrap();
        Self { root }
    }
    fn spec(&self) -> Spec {
        Spec {
            project: "demo".into(),
            executor: "pai".into(),
            directory: self.root.path().join("source"),
            dockerfile: "Dockerfile".into(),
            command: vec!["python3".into(), "train.py".into()],
            gpus: 1,
            cpus: None,
            memory_gb: None,
            walltime_seconds: 600,
            seed: 0,
            ..Spec::default()
        }
    }
    fn run(&self, fake: &Fake, answer: bool, asked: &mut Vec<String>) -> Result<Value, Error> {
        self.drive(fake, answer, asked, false, &mut Vec::new())
    }
    fn drive(
        &self,
        fake: &Fake,
        answer: bool,
        asked: &mut Vec<String>,
        follow: bool,
        shown: &mut Vec<String>,
    ) -> Result<Value, Error> {
        submit::run(
            &fake.client(),
            &self.spec(),
            &self.root.path().join("states"),
            Hooks {
                confirm: &mut |summary| {
                    asked.push(summary.to_owned());
                    Ok(answer)
                },
                progress: &mut |line| shown.push(line.to_owned()),
                poll: Duration::from_millis(1),
                deadline: Duration::from_secs(5),
                follow,
                silence: Duration::from_millis(30),
                follow_poll: Duration::from_millis(1),
            },
        )
    }
}

#[test]
fn one_command_confirms_once_and_sends_one_job() {
    let files = Files::new();
    let fake = Fake::start(World::new());
    let mut asked = Vec::new();
    let value = files.run(&fake, true, &mut asked).unwrap();
    assert_eq!(value["status"], "VERIFIED");
    assert_eq!(value["submission_id"], ID);
    assert_eq!(value["already_submitted"], false);
    assert_eq!(asked.len(), 1);
    for shown in [
        "pai",
        "python3 train.py",
        "1 GPU",
        "600 seconds",
        "0.17 GPU hours",
        IMAGE,
    ] {
        assert!(asked[0].contains(shown), "{shown}");
    }
    assert_eq!(fake.count("POST", "/execute"), 1);
    assert_eq!(
        fake.execute_request(),
        json!({"confirmation":format!("EXECUTE {ID}"),"reason":"ml-exp submit"})
    );
    let request = fake.prepared_request();
    assert_eq!(request["runtime"]["base_image"], IMAGE);
    assert_eq!(
        request["run"]["resources"],
        json!({"gpus":1,"cpus":8,"memory_gb":32})
    );
    assert_eq!(request["approval"]["max_gpu_hours"], 0.17);
}

#[test]
fn the_latest_submission_is_remembered_for_the_short_commands() {
    let files = Files::new();
    let states = files.root.path().join("states");
    assert!(submit::latest(&states).is_err());
    let fake = Fake::start(World::new());
    let value = files.run(&fake, true, &mut Vec::new()).unwrap();
    let latest = submit::latest(&states).unwrap();
    assert_eq!(latest.project, "demo");
    assert_eq!(value["run_id"], latest.run_id);
    assert_eq!(latest.submission_id.as_deref(), Some(ID));
    assert!(!latest.unsettled);
    assert_eq!(value["next"][0], "ml-exp status");
}

#[test]
fn repeating_the_command_after_it_was_sent_sends_nothing_new() {
    let files = Files::new();
    let fake = Fake::start(World::new());
    let mut asked = Vec::new();
    let first = files.run(&fake, true, &mut asked).unwrap();
    let mutations = fake.mutations();
    let again = files.run(&fake, true, &mut asked).unwrap();
    assert_eq!(asked.len(), 1, "no second question");
    assert_eq!(fake.mutations(), mutations, "no second request of any kind");
    assert_eq!(again["already_submitted"], true);
    assert_eq!(again["run_id"], first["run_id"]);
    assert_eq!(again["submission_id"], first["submission_id"]);
}

#[test]
fn declining_sends_nothing() {
    let files = Files::new();
    let fake = Fake::start(World::new());
    assert!(files.run(&fake, false, &mut Vec::new()).is_err());
    assert_eq!(fake.mutations(), 0);
}

#[test]
fn a_lost_execute_reply_is_observed_and_never_sent_again() {
    let files = Files::new();
    let mut world = World::new();
    world.lose_execute_reply = true;
    let fake = Fake::start(world);
    let value = files.run(&fake, true, &mut Vec::new()).unwrap();
    assert_eq!(fake.count("POST", "/execute"), 1);
    assert_eq!(fake.count("POST", "/reconcile"), 1);
    assert_eq!(value["status"], "VERIFIED");
}

#[test]
fn repeating_after_a_lost_reply_observes_the_original_job() {
    let files = Files::new();
    let mut world = World::new();
    world.status = "RECONCILE_REQUIRED";
    world.next_action = "RECONCILE";
    world.prepared = true;
    world.continues = 2;
    let fake = Fake::start(world);
    let mut asked = Vec::new();
    let value = files.run(&fake, true, &mut asked).unwrap();
    assert!(asked.is_empty());
    assert_eq!(fake.count("POST", "/execute"), 0);
    assert_eq!(fake.count("POST", "/reconcile"), 1);
    assert_eq!(value["already_submitted"], true);
}

#[test]
fn a_failed_build_stops_rather_than_building_again() {
    let files = Files::new();
    let mut world = World::new();
    world.build_failed = true;
    let fake = Fake::start(world);
    let error = files.run(&fake, true, &mut Vec::new()).unwrap_err();
    assert!(error.to_string().contains("BAD_DOCKERFILE"), "{error}");
    assert_eq!(fake.count("POST", "/continue"), 1);
    assert_eq!(fake.count("POST", "/execute"), 0);
}

#[test]
fn an_expired_submission_says_what_to_do_and_sends_no_job() {
    let files = Files::new();
    let mut world = World::new();
    world.next_action = "REPREPARE";
    let fake = Fake::start(world);
    let error = files.run(&fake, true, &mut Vec::new()).unwrap_err();
    assert!(error.to_string().contains("run renew"), "{error}");
    assert_eq!(fake.count("POST", "/execute"), 0);
}

#[test]
fn a_missing_dockerfile_is_explained_before_anything_is_sent() {
    let files = Files::new();
    std::fs::remove_file(files.root.path().join("source/Dockerfile")).unwrap();
    let fake = Fake::start(World::new());
    let mut asked = Vec::new();
    let error = files.run(&fake, true, &mut asked).unwrap_err();
    let text = error.to_string();
    assert!(
        text.contains("Dockerfile") && text.contains("/api/environments"),
        "{text}"
    );
    assert!(asked.is_empty());
    assert_eq!(fake.mutations(), 0);
}

#[test]
fn an_unknown_executor_or_too_many_gpus_is_refused_before_the_question() {
    let files = Files::new();
    let fake = Fake::start(World::new());
    let client = fake.client();
    let states = files.root.path().join("states");
    let mut bad_executor = files.spec();
    bad_executor.executor = "other".into();
    let mut too_many = files.spec();
    too_many.gpus = 2;
    for spec in [bad_executor, too_many] {
        let mut asked = 0;
        let result = submit::run(
            &client,
            &spec,
            &states,
            Hooks {
                confirm: &mut |_| {
                    asked += 1;
                    Ok(true)
                },
                progress: &mut |_| {},
                poll: Duration::from_millis(1),
                deadline: Duration::from_secs(5),
                follow: false,
                silence: Duration::from_millis(30),
                follow_poll: Duration::from_millis(1),
            },
        );
        assert!(result.is_err());
        assert_eq!(asked, 0);
    }
    assert_eq!(fake.mutations(), 0);
}

#[test]
fn the_build_shows_its_phase_and_each_log_line_once() {
    let files = Files::new();
    let fake = Fake::start(World::new());
    let mut shown = Vec::new();
    files
        .drive(&fake, true, &mut Vec::new(), false, &mut shown)
        .unwrap();
    let count = |text: &str| shown.iter().filter(|line| line.contains(text)).count();
    assert_eq!(count("BUILDING_ON_DSW"), 1, "{shown:?}");
    assert_eq!(count("step 1"), 1, "{shown:?}");
    assert_eq!(count("step 2"), 1, "{shown:?}");
}

#[test]
fn following_shows_the_backends_states_until_the_job_ends() {
    let files = Files::new();
    let mut world = World::new();
    world.job_states = vec![
        ("QUEUED", "EnvPreparing"),
        ("RUNNING", "Running"),
        ("SUCCEEDED", "Succeeded"),
    ];
    let fake = Fake::start(world);
    let mut shown = Vec::new();
    let value = files
        .drive(&fake, true, &mut Vec::new(), true, &mut shown)
        .unwrap();
    assert_eq!(value["job_state"], "SUCCEEDED");
    let jobs: Vec<&String> = shown
        .iter()
        .filter(|line| line.starts_with("[job]"))
        .collect();
    assert_eq!(
        jobs,
        [
            "[job] QUEUED (EnvPreparing)",
            "[job] RUNNING",
            "[job] SUCCEEDED"
        ]
    );
    assert_eq!(fake.count("POST", "/execute"), 1);
    assert_eq!(fake.count("POST", "/refresh"), 3);
}

#[test]
fn losing_the_status_stops_watching_and_touches_nothing() {
    let files = Files::new();
    let mut world = World::new();
    world.reads_down = true;
    let fake = Fake::start(world);
    let mut shown = Vec::new();
    let value = files
        .drive(&fake, true, &mut Vec::new(), true, &mut shown)
        .unwrap();
    assert_eq!(value["status"], "VERIFIED");
    assert!(
        shown.iter().any(|line| line.contains("keeps running")),
        "{shown:?}"
    );
    assert_eq!(fake.count("POST", "/execute"), 1);
    assert_eq!(fake.count("POST", "/cancel"), 0);
    assert!(fake.count("POST", "/refresh") > 1, "it kept trying");
}

#[test]
fn repeating_the_command_goes_back_to_watching_the_same_job() {
    let files = Files::new();
    let mut world = World::new();
    world.job_states = vec![("RUNNING", "Running"), ("FAILED", "Failed")];
    let fake = Fake::start(world);
    let mut shown = Vec::new();
    files
        .drive(&fake, true, &mut Vec::new(), false, &mut Vec::new())
        .unwrap();
    let mut asked = Vec::new();
    let value = files
        .drive(&fake, true, &mut asked, true, &mut shown)
        .unwrap();
    assert!(asked.is_empty());
    assert_eq!(value["already_submitted"], true);
    assert_eq!(value["job_state"], "FAILED");
    assert_eq!(fake.count("POST", "/execute"), 1);
}

fn ended() -> World {
    let mut world = World::new();
    world.job_states = vec![("FAILED", "Failed")];
    world
}
fn quiet(confirm: &mut dyn FnMut(&str) -> Result<bool, Error>) -> Hooks<'_> {
    Hooks {
        confirm,
        progress: Box::leak(Box::new(|_: &str| {})),
        poll: Duration::from_millis(1),
        deadline: Duration::from_secs(5),
        follow: false,
        silence: Duration::from_millis(30),
        follow_poll: Duration::from_millis(1),
    }
}
fn submitted_train(files: &Files, fake: &Fake) {
    let mut spec = files.spec();
    spec.run_id = Some("train".into());
    submit::run(
        &fake.client(),
        &spec,
        &files.root.path().join("states"),
        Hooks {
            confirm: &mut |_| Ok(true),
            progress: &mut |_| {},
            poll: Duration::from_millis(1),
            deadline: Duration::from_secs(5),
            follow: false,
            silence: Duration::from_millis(30),
            follow_poll: Duration::from_millis(1),
        },
    )
    .unwrap();
}

#[test]
fn cancel_says_what_it_will_stop_and_declining_sends_nothing() {
    let files = Files::new();
    let fake = Fake::start(World::new());
    submitted_train(&files, &fake);
    let states = files.root.path().join("states");
    let mut asked = Vec::new();
    let declined = submit::cancel(
        &fake.client(),
        &states,
        Hooks {
            confirm: &mut |question| {
                asked.push(question.to_owned());
                Ok(false)
            },
            progress: &mut |_| {},
            poll: Duration::from_millis(1),
            deadline: Duration::from_secs(5),
            follow: false,
            silence: Duration::from_millis(30),
            follow_poll: Duration::from_millis(1),
        },
    );
    assert!(declined.is_err());
    assert!(
        asked[0].contains("train") && asked[0].contains("attempt-001"),
        "{asked:?}"
    );
    assert_eq!(fake.count("POST", "/cancel"), 0);
    submit::cancel(
        &fake.client(),
        &states,
        Hooks {
            confirm: &mut |_| Ok(true),
            progress: &mut |_| {},
            poll: Duration::from_millis(1),
            deadline: Duration::from_secs(5),
            follow: false,
            silence: Duration::from_millis(30),
            follow_poll: Duration::from_millis(1),
        },
    )
    .unwrap();
    assert_eq!(fake.count("POST", "/cancel"), 1);
}

#[test]
fn retry_creates_the_next_attempt_and_sends_exactly_one_new_job() {
    let files = Files::new();
    let fake = Fake::start(ended());
    submitted_train(&files, &fake);
    let states = files.root.path().join("states");
    assert_eq!(fake.count("POST", "/execute"), 1);
    let mut asked = Vec::new();
    let value = submit::retry(
        &fake.client(),
        &states,
        None,
        Hooks {
            confirm: &mut |question| {
                asked.push(question.to_owned());
                Ok(true)
            },
            progress: &mut |_| {},
            poll: Duration::from_millis(1),
            deadline: Duration::from_secs(5),
            follow: false,
            silence: Duration::from_millis(30),
            follow_poll: Duration::from_millis(1),
        },
    )
    .unwrap();
    assert_eq!(asked.len(), 1);
    assert!(
        asked[0].contains("the start") && asked[0].contains("0.5"),
        "{asked:?}"
    );
    assert_eq!(value["submission_id"], SECOND);
    assert_eq!(fake.count("POST", "/retry"), 1);
    assert_eq!(fake.count("POST", "/submissions"), 1);
    assert_eq!(
        fake.count("POST", "/execute"),
        2,
        "one original and one new job"
    );
    assert_eq!(
        submit::latest(&states).unwrap().submission_id.as_deref(),
        Some(SECOND)
    );
}

#[test]
fn declining_a_retry_creates_nothing() {
    let files = Files::new();
    let fake = Fake::start(ended());
    submitted_train(&files, &fake);
    let declined = submit::retry(
        &fake.client(),
        &files.root.path().join("states"),
        None,
        Hooks {
            confirm: &mut |_| Ok(false),
            progress: &mut |_| {},
            poll: Duration::from_millis(1),
            deadline: Duration::from_secs(5),
            follow: false,
            silence: Duration::from_millis(30),
            follow_poll: Duration::from_millis(1),
        },
    );
    assert!(declined.is_err());
    // Asking the provider for the latest state is allowed; creating anything is not.
    assert_eq!(fake.count("POST", "/retry"), 0);
    assert_eq!(fake.count("POST", "/submissions"), 0);
    assert_eq!(fake.count("POST", "/execute"), 1);
}

#[test]
fn a_lost_request_that_sent_nothing_is_not_reported_as_submitted() {
    let files = Files::new();
    let mut world = World::new();
    world.execute_never_arrives = true;
    let fake = Fake::start(world);
    let error = files.run(&fake, true, &mut Vec::new()).unwrap_err();
    assert!(
        error.to_string().contains("no job has been sent"),
        "{error}"
    );
    // Repeating after the network recovers finishes it, with one job.
    fake.world.lock().unwrap().execute_never_arrives = false;
    let value = files.run(&fake, true, &mut Vec::new()).unwrap();
    assert_eq!(value["status"], "VERIFIED");
    assert_eq!(
        fake.count("POST", "/execute"),
        2,
        "one lost and one that arrived"
    );
    assert_eq!(fake.count("POST", "/reconcile"), 0);
}

#[test]
fn following_a_slurm_job_reads_the_view_that_has_the_latest_state_and_ends() {
    let files = Files::new();
    let mut world = World::new();
    world.cloud = false;
    world.job_states = vec![("RUNNING", ""), ("SUCCEEDED", "")];
    let fake = Fake::start(world);
    let mut shown = Vec::new();
    let value = files
        .drive(&fake, true, &mut Vec::new(), true, &mut shown)
        .unwrap();
    assert_eq!(value["job_state"], "SUCCEEDED");
    assert_eq!(fake.count("POST", "/refresh"), 0, "no SSH call per poll");
}

#[test]
fn watching_says_so_when_the_server_reports_nothing_and_does_not_claim_success() {
    let files = Files::new();
    let mut world = World::new();
    world.cloud = false;
    world.polling = false;
    let fake = Fake::start(world);
    let mut shown = Vec::new();
    let value = files
        .drive(&fake, true, &mut Vec::new(), true, &mut shown)
        .unwrap();
    assert_eq!(value["watching_stopped"], true);
    assert_eq!(value["job_over"], false);
    assert!(submit::failure(&value).is_some(), "not read as success");
    assert!(
        shown.iter().any(|line| line.contains("not reporting")),
        "{shown:?}"
    );
}

#[test]
fn a_long_cloud_job_is_watched_for_as_long_as_it_runs() {
    let files = Files::new();
    let mut world = World::new();
    world.job_states = vec![("RUNNING", "Running"); 400];
    world.job_states.push(("SUCCEEDED", "Succeeded"));
    let fake = Fake::start(world);
    let mut shown = Vec::new();
    let value = files
        .drive(&fake, true, &mut Vec::new(), true, &mut shown)
        .unwrap();
    assert_eq!(
        value["job_state"], "SUCCEEDED",
        "well past the 30 ms silence limit"
    );
    assert_eq!(value["watching_stopped"], false);
}

#[test]
fn retry_on_a_server_not_polling_slurm_leaves_the_decision_to_the_server() {
    let files = Files::new();
    let mut world = ended();
    world.cloud = false;
    world.polling = false;
    let fake = Fake::start(world);
    submitted_train(&files, &fake);
    let states = files.root.path().join("states");
    submit::retry(&fake.client(), &states, None, quiet(&mut |_| Ok(true))).unwrap();
    assert_eq!(fake.count("POST", "/retry"), 1);
}

#[test]
fn a_build_left_running_by_a_restart_is_not_called_failed() {
    let files = Files::new();
    let mut world = World::new();
    world.restarted = true;
    let fake = Fake::start(world);
    let value = files.run(&fake, true, &mut Vec::new()).unwrap();
    assert_eq!(value["status"], "VERIFIED");
}

#[test]
fn a_build_that_left_nothing_says_its_outcome_is_unresolved() {
    let files = Files::new();
    let mut world = World::new();
    world.restarted = true;
    world.builder_gone = true;
    world.continues_until_ready = u32::MAX;
    let fake = Fake::start(world);
    let text = files
        .run(&fake, true, &mut Vec::new())
        .unwrap_err()
        .to_string();
    assert!(text.contains("unresolved"), "{text}");
    assert_eq!(fake.count("POST", "/execute"), 0);
}

#[test]
fn a_job_stays_reachable_when_its_executor_has_gone() {
    let files = Files::new();
    let fake = Fake::start(World::new());
    files.run(&fake, true, &mut Vec::new()).unwrap();
    fake.world.lock().unwrap().no_executors = true;
    let mut asked = Vec::new();
    let value = files.run(&fake, true, &mut asked).unwrap();
    assert_eq!(value["already_submitted"], true);
    assert!(asked.is_empty());
}

#[test]
fn a_changed_limit_shows_and_runs_what_was_already_prepared() {
    let files = Files::new();
    let mut world = World::new();
    world.execute_never_arrives = true;
    let fake = Fake::start(world);
    assert!(files.run(&fake, true, &mut Vec::new()).is_err());
    {
        let mut world = fake.world.lock().unwrap();
        world.execute_never_arrives = false;
        world.cpu_limit = 4;
    }
    let mut asked = Vec::new();
    let value = files.run(&fake, true, &mut asked).unwrap();
    assert_eq!(value["status"], "VERIFIED");
    assert_eq!(fake.count("POST", "/experiment-preparations"), 1);
    assert!(asked[0].contains("8 CPU"), "the frozen value: {asked:?}");
}

#[test]
fn a_prepare_that_never_arrived_does_not_stop_the_next_one() {
    let files = Files::new();
    let mut world = World::new();
    world.prepare_never_arrives = true;
    let fake = Fake::start(world);
    assert!(files.run(&fake, true, &mut Vec::new()).is_err());
    // The local record holds the old request; the server has none, so it cannot matter.
    {
        let mut world = fake.world.lock().unwrap();
        world.prepare_never_arrives = false;
        world.cpu_limit = 4;
    }
    let value = files.run(&fake, true, &mut Vec::new()).unwrap();
    assert_eq!(value["status"], "VERIFIED");
}

#[test]
fn watching_stops_when_the_provider_can_no_longer_be_read() {
    let files = Files::new();
    let mut world = World::new();
    world.job_states = vec![("RUNNING", "Running")];
    world.provider_read_failing = true;
    let fake = Fake::start(world);
    let mut shown = Vec::new();
    let value = files
        .drive(&fake, true, &mut Vec::new(), true, &mut shown)
        .unwrap();
    assert_eq!(value["watching_stopped"], true);
    assert!(
        shown.iter().any(|line| line.contains("cannot be read")),
        "{shown:?}"
    );
}

#[test]
fn watching_stops_when_the_provider_has_paused_the_job() {
    let files = Files::new();
    let mut world = World::new();
    world.job_states = vec![("CANCELLED", "SUSPENDED")];
    let fake = Fake::start(world);
    let mut shown = Vec::new();
    let value = files
        .drive(&fake, true, &mut Vec::new(), true, &mut shown)
        .unwrap();
    assert_eq!(value["watching_stopped"], true);
    assert_eq!(value["job_over"], false, "it may still run");
    assert!(
        shown.iter().any(|line| line.contains("paused")),
        "{shown:?}"
    );
}

#[test]
fn another_server_does_not_inherit_this_servers_state_file() {
    let files = Files::new();
    let first = Fake::start(World::new());
    files.run(&first, true, &mut Vec::new()).unwrap();
    let second = Fake::start(World::new());
    let value = files.run(&second, true, &mut Vec::new()).unwrap();
    assert_eq!(value["status"], "VERIFIED");
    assert_eq!(second.count("POST", "/execute"), 1);
}

#[test]
fn an_unresolved_send_stays_unsettled_and_retry_reconciles_it() {
    let files = Files::new();
    let fake = Fake::start(ended());
    submitted_train(&files, &fake);
    {
        let mut world = fake.world.lock().unwrap();
        world.status = "RECONCILE_REQUIRED";
        world.next_action = "RECONCILE";
    }
    let states = files.root.path().join("states");
    let value = submit::retry(&fake.client(), &states, None, quiet(&mut |_| Ok(true))).unwrap();
    assert_eq!(fake.count("POST", "/reconcile"), 1);
    // The fake's reconcile confirms it, so this was a real request for a new Attempt.
    assert_eq!(value["status"], "VERIFIED");
}

#[test]
fn a_checkpoint_is_not_silently_ignored_for_an_attempt_already_prepared() {
    let files = Files::new();
    let fake = Fake::start(ended());
    submitted_train(&files, &fake);
    fake.world.lock().unwrap().status = "PREPARED";
    let states = files.root.path().join("states");
    let text = submit::retry(
        &fake.client(),
        &states,
        Some(format!("checkpoint.{}", "a".repeat(64))),
        quiet(&mut |_| Ok(true)),
    )
    .unwrap_err()
    .to_string();
    assert!(text.contains("--checkpoint"), "{text}");
    assert_eq!(fake.count("POST", "/execute"), 1);
}

#[test]
fn cancel_works_on_a_cloud_job_which_keeps_no_cancellation_record() {
    let files = Files::new();
    let fake = Fake::start(World::new());
    submitted_train(&files, &fake);
    let states = files.root.path().join("states");
    submit::cancel(&fake.client(), &states, quiet(&mut |_| Ok(true))).unwrap();
    assert_eq!(fake.count("POST", "/cancel"), 1);
}

#[test]
fn cancel_checks_a_slurm_job_cancellation_record() {
    let files = Files::new();
    let mut world = World::new();
    world.cloud = false;
    let fake = Fake::start(world);
    submitted_train(&files, &fake);
    let states = files.root.path().join("states");
    submit::cancel(&fake.client(), &states, quiet(&mut |_| Ok(true))).unwrap();
    assert_eq!(fake.count("POST", "/cancel"), 1);
}

#[test]
fn nothing_can_act_on_a_submission_that_is_still_being_prepared() {
    let files = Files::new();
    let mut world = World::new();
    world.build_failed = true;
    let fake = Fake::start(world);
    assert!(files.run(&fake, true, &mut Vec::new()).is_err());
    let states = files.root.path().join("states");
    assert!(submit::latest(&states).unwrap().submission_id.is_none());
    let mutations = fake.mutations();
    let text = submit::cancel(&fake.client(), &states, quiet(&mut |_| Ok(true)))
        .unwrap_err()
        .to_string();
    assert!(text.contains("has not sent a job"), "{text}");
    assert!(submit::retry(&fake.client(), &states, None, quiet(&mut |_| Ok(true))).is_err());
    assert_eq!(fake.mutations(), mutations);
}

fn unsettle(files: &Files) {
    let path = files.root.path().join("states/latest.json");
    let mut latest = serde_json::from_slice::<Value>(&std::fs::read(&path).unwrap()).unwrap();
    latest["unsettled"] = json!(true);
    ml_exp_io::write_atomic(&path, latest.to_string().as_bytes(), false).unwrap();
}

#[test]
fn repeating_retry_after_an_unknown_outcome_only_finds_out_what_happened() {
    let files = Files::new();
    let fake = Fake::start(ended());
    submitted_train(&files, &fake);
    unsettle(&files);
    let states = files.root.path().join("states");
    let mut asked = 0;
    let value = submit::retry(
        &fake.client(),
        &states,
        None,
        quiet(&mut |_| {
            asked += 1;
            Ok(true)
        }),
    )
    .unwrap();
    assert_eq!(asked, 0);
    assert_eq!(value["already_submitted"], true);
    assert_eq!(fake.count("POST", "/retry"), 0, "no new Attempt");
    assert_eq!(fake.count("POST", "/execute"), 1, "no new job");
    assert!(!submit::latest(&states).unwrap().unsettled);
    // Now that it is known, asking again is a real request for another Attempt.
    submit::retry(&fake.client(), &states, None, quiet(&mut |_| Ok(true))).unwrap();
    assert_eq!(fake.count("POST", "/retry"), 1);
    assert_eq!(fake.count("POST", "/execute"), 2);
}

#[test]
fn retry_is_refused_while_a_cloud_job_has_not_ended() {
    let files = Files::new();
    let mut world = World::new();
    world.job_states = vec![("RUNNING", "Running")];
    let fake = Fake::start(world);
    submitted_train(&files, &fake);
    let states = files.root.path().join("states");
    let text = submit::retry(&fake.client(), &states, None, quiet(&mut |_| Ok(true)))
        .unwrap_err()
        .to_string();
    assert!(text.contains("has not ended"), "{text}");
    assert_eq!(fake.count("POST", "/retry"), 0);
}

#[test]
fn retry_finishes_a_prepared_attempt_rather_than_starting_another() {
    let files = Files::new();
    let fake = Fake::start(ended());
    submitted_train(&files, &fake);
    {
        let mut world = fake.world.lock().unwrap();
        world.status = "PREPARED";
        world.next_action = "EXECUTE";
    }
    let states = files.root.path().join("states");
    let value = submit::retry(&fake.client(), &states, None, quiet(&mut |_| Ok(true))).unwrap();
    assert_eq!(value["status"], "VERIFIED");
    assert_eq!(fake.count("POST", "/retry"), 0);
    assert_eq!(fake.count("POST", "/execute"), 2);
}

#[test]
fn repeating_submit_after_a_retry_keeps_the_newer_attempt_as_the_latest() {
    let files = Files::new();
    let fake = Fake::start(ended());
    submitted_train(&files, &fake);
    let states = files.root.path().join("states");
    submit::retry(&fake.client(), &states, None, quiet(&mut |_| Ok(true))).unwrap();
    fake.world.lock().unwrap().preparation_submission = Some(ID);
    submitted_train(&files, &fake);
    assert_eq!(
        submit::latest(&states).unwrap().submission_id.as_deref(),
        Some(SECOND)
    );
}

/// One saved batch with one child, and the fake answering for that child.
struct SavedBatch {
    root: tempfile::TempDir,
    fake: Fake,
    client: Client,
    health: Value,
}
impl SavedBatch {
    fn new() -> Self {
        let batch: ml_exp_core::preparation::Batch = serde_json::from_value(json!({
            "request_id":"b".repeat(32),
            "experiment":{"request_id":"a".repeat(32),
                "run":{"run_id":"trial","runtime_id":RUNTIME,"executor":"dlc","command":["train"],
                    "resources":{"gpus":1,"cpus":2,"memory_gb":4},"walltime_seconds":300,"seed":42},
                "approval":{"max_gpu_hours":1.0,"reason":"test"}}}))
        .unwrap();
        let child = batch.expand("demo").unwrap().remove(0);
        let mut world = World::new();
        world.prepared = true;
        world.continues = 2;
        world.run_id = child.run["run_id"].as_str().unwrap().to_owned();
        let fake = Fake::start(world);
        let client = fake.client();
        let health = client.health().unwrap();
        let root = tempfile::tempdir().unwrap();
        let manifest = json!({"contract":"native-batch.v1","server":client.base_url(),
            "project":"demo","batch":batch});
        ml_exp_io::directory::Directory::open(root.path(), false)
            .unwrap()
            .publish_new(
                std::ffi::OsStr::new("batch.json"),
                &serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
        Self {
            root,
            fake,
            client,
            health,
        }
    }
    fn dispatch(&self) -> Result<Value, Error> {
        ml_exp_client::batch::dispatch(
            &self.client,
            self.root.path(),
            1,
            &format!("SUBMIT BATCH {}", "b".repeat(32)),
            &self.health,
        )
    }
    fn claim_file(&self) -> bool {
        self.root
            .path()
            .join(format!("{ID}.dispatch.json"))
            .exists()
    }
}

#[test]
fn a_batch_sends_a_ready_child_once_with_its_reason() {
    let batch = SavedBatch::new();
    let state = batch.dispatch().unwrap();
    assert_eq!(state["experiments"][0]["submission"]["status"], "VERIFIED");
    assert_eq!(state["occupied_slots"], 1);
    assert!(batch.claim_file());
    assert_eq!(
        batch.fake.execute_request(),
        json!({"confirmation":format!("EXECUTE {ID}"),
            "reason":format!("SUBMIT BATCH {}", "b".repeat(32))})
    );
    assert_eq!(batch.fake.mutations(), 1, "no other step");
    let again = batch.dispatch().unwrap();
    assert_eq!(again["occupied_slots"], 1);
    assert_eq!(batch.fake.count("POST", "/execute"), 1);
}

#[test]
fn a_batch_child_past_its_gate_writes_no_claim_and_takes_no_slot() {
    let batch = SavedBatch::new();
    batch.fake.world.lock().unwrap().next_action = "REPREPARE";
    let state = batch.dispatch().unwrap();
    assert_eq!(state["occupied_slots"], 0);
    let diagnostic = state["experiments"][0]["diagnostic"].as_str().unwrap();
    assert!(diagnostic.starts_with("GATE_EXPIRED"), "{diagnostic}");
    assert!(!batch.claim_file());
    assert_eq!(batch.fake.mutations(), 0);
}

#[test]
fn a_refused_batch_child_holds_its_slot_only_until_its_gate_expires() {
    let batch = SavedBatch::new();
    batch.fake.world.lock().unwrap().execute_refused = true;
    assert!(matches!(batch.dispatch(), Err(Error::Http(_))));
    assert!(batch.claim_file());
    // The client cannot tell this refusal from a request still in flight, so it is not
    // sent again and it keeps the slot.
    let state = batch.dispatch().unwrap();
    assert_eq!(state["occupied_slots"], 1);
    let diagnostic = state["experiments"][0]["diagnostic"].as_str().unwrap();
    assert!(
        diagnostic.starts_with("DISPATCH_ALREADY_REQUESTED"),
        "{diagnostic}"
    );
    assert_eq!(batch.fake.count("POST", "/execute"), 1);
    // Past the gate nothing can start from this Action, so the slot is free again.
    batch.fake.world.lock().unwrap().next_action = "REPREPARE";
    let state = batch.dispatch().unwrap();
    assert_eq!(state["occupied_slots"], 0);
    let diagnostic = state["experiments"][0]["diagnostic"].as_str().unwrap();
    assert!(diagnostic.starts_with("GATE_EXPIRED"), "{diagnostic}");
    assert_eq!(batch.fake.count("POST", "/execute"), 1);
}

#[test]
fn a_dockerfile_git_ignores_is_named_before_anything_is_sent() {
    let files = Files::new();
    let source = files.root.path().join("source");
    let status = std::process::Command::new("git")
        .arg("-C")
        .arg(&source)
        .args(["init", "-q"])
        .status()
        .unwrap();
    assert!(status.success());
    std::fs::write(source.join(".gitignore"), "Dockerfile\n").unwrap();
    let fake = Fake::start(World::new());
    let mut asked = Vec::new();
    let text = files.run(&fake, true, &mut asked).unwrap_err().to_string();
    assert!(text.contains(".gitignore"), "{text}");
    assert!(asked.is_empty());
    assert_eq!(fake.mutations(), 0);
}
