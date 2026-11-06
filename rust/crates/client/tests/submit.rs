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
    authorize_never_arrives: bool,
    execute_never_arrives: bool,
    /// Cloud reports through its observation; Slurm through the Submission's result.
    cloud: bool,
    /// The backend's states, oldest first; the last one stays.
    job_states: Vec<(&'static str, &'static str)>,
    reads_down: bool,
    progress_reads: usize,
    reads: usize,
    /// What a preparation names, which stays the first Submission after a retry.
    preparation_submission: Option<&'static str>,
    action: &'static str,
    attempt: &'static str,
}
impl World {
    fn new() -> Self {
        Self {
            prepared: false,
            continues: 0,
            continues_until_ready: 1,
            status: "PREPARED",
            next_action: "AUTHORIZE",
            build_failed: false,
            lose_execute_reply: false,
            authorize_never_arrives: false,
            execute_never_arrives: false,
            cloud: true,
            job_states: vec![("RUNNING", "Running"), ("SUCCEEDED", "Succeeded")],
            reads_down: false,
            progress_reads: 0,
            reads: 0,
            preparation_submission: None,
            action: ID,
            attempt: "attempt-001",
        }
    }
    fn submission(&self) -> Value {
        let (state, own) = self.job_states[0];
        let mut value = json!({"submission_id":self.action,"project":"demo","run_id":"train",
            "attempt_id":self.attempt,
            "operation":"SUBMIT_RUN","status":self.status,"next_action":self.next_action,
            "confirmation":format!("EXECUTE {}",self.action),
            "preflight_summary":{"resources":{"gpus":1},"max_gpu_hours":0.5},
            "execution":{"status":self.status,"revision":0,
                "result":{"submission":{"backend_job_id":"123"}}}});
        if self.status == "VERIFIED" {
            if self.cloud {
                value["cloud_observation"] = json!({"observation":{"job_id":"job",
                    "provider_state":own,"state":state,"queue":null}});
            } else {
                value["execution"]["result"]["observation"] = json!({"status":{"state":state}});
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
        json!({"project":"demo","preparation_id":id,"run_id":"train",
            "runtime":{"runtime_id":RUNTIME,
                "status":if self.build_failed {"RECONCILE_REQUIRED"} else if ready {"READY"} else {"EXECUTING"},
                "build_error":if self.build_failed {json!({"code":"BAD_DOCKERFILE"})} else {Value::Null}},
            "phase":"ENVIRONMENT_PREPARATION","status":"PREPARING",
            "confirmation":format!("PREPARE {}",id.unwrap_or_default()),
            "submission":if ready {
                let mut value = self.submission();
                if let Some(id) = self.preparation_submission {
                    value["submission_id"] = json!(id);
                }
                value
            } else {Value::Null},
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
            ("GET", "/api/executors") => (
                200,
                json!({"executors":[{"executor_id":"pai","enabled":true,"run_creation":true,
                    "limits":{"gpus":1,"cpus":8,"memory_gb":32,"walltime_seconds":3600}}]}),
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
                self.prepared = true;
                let request: Value = serde_json::from_slice(body).unwrap();
                let id = ml_exp_core::preparation::id("demo", request["request_id"].as_str()?);
                (200, json!({"project":"demo","preparation_id":id.unwrap()}))
            }
            ("POST", r) if r.ends_with("/continue") => {
                self.continues += 1;
                (200, self.view(r))
            }
            ("GET", r) if r.starts_with("/api/submissions/") && r.matches('/').count() == 3 => {
                let mut value = self.submission();
                value["submission_id"] = json!(r.rsplit('/').next());
                // Slurm's view has the latest state; a cloud view waits for a refresh.
                if !self.cloud && self.status == "VERIFIED" {
                    self.advance_job();
                }
                (200, value)
            }
            ("POST", r) if r.ends_with("/cancel") => (200, self.submission()),
            ("POST", r) if r.ends_with("/retry") => {
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
                self.next_action = "AUTHORIZE";
                (200, self.submission())
            }
            ("POST", r) if r.ends_with("/authorize") => {
                if self.authorize_never_arrives {
                    return None;
                }
                self.status = "AUTHORIZED";
                self.next_action = "EXECUTE";
                (200, self.submission())
            }
            ("POST", r) if r.ends_with("/execute") => {
                if self.execute_never_arrives {
                    return None;
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
                        "progress":{"phase":"BUILDING_ON_DSW","message":"Docker build running"}}),
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
    assert_eq!(fake.count("POST", "/authorize"), 1);
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
    assert_eq!(fake.count("POST", "/authorize"), 0);
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
    assert_eq!(fake.count("POST", "/authorize"), 0);
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
    assert_eq!(fake.count("POST", "/authorize"), 0);
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
fn quiet<'a>(confirm: &'a mut dyn FnMut(&str) -> Result<bool, Error>) -> Hooks<'a> {
    Hooks {
        confirm,
        progress: Box::leak(Box::new(|_: &str| {})),
        poll: Duration::from_millis(1),
        deadline: Duration::from_secs(5),
        follow: false,
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
    let mutations = fake.mutations();
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
            follow_poll: Duration::from_millis(1),
        },
    );
    assert!(declined.is_err());
    assert_eq!(fake.mutations(), mutations);
}

#[test]
fn a_lost_request_that_sent_nothing_is_not_reported_as_submitted() {
    for lost_at_authorize in [true, false] {
        let files = Files::new();
        let mut world = World::new();
        world.authorize_never_arrives = lost_at_authorize;
        world.execute_never_arrives = !lost_at_authorize;
        let fake = Fake::start(world);
        let error = files.run(&fake, true, &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("no job was sent"), "{error}");
        // Repeating after the network recovers finishes it, with one job.
        {
            let mut world = fake.world.lock().unwrap();
            world.authorize_never_arrives = false;
            world.execute_never_arrives = false;
        }
        let value = files.run(&fake, true, &mut Vec::new()).unwrap();
        assert_eq!(value["status"], "VERIFIED");
    }
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
fn following_stops_when_nothing_changes_for_a_long_time() {
    let files = Files::new();
    let mut world = World::new();
    world.job_states = vec![("RUNNING", "Running")];
    world.status = "VERIFIED";
    world.prepared = true;
    world.continues = 2;
    let fake = Fake::start(world);
    let mut shown = Vec::new();
    let value = submit::run(
        &fake.client(),
        &files.spec(),
        &files.root.path().join("states"),
        Hooks {
            confirm: &mut |_| Ok(true),
            progress: &mut |line| shown.push(line.to_owned()),
            poll: Duration::from_millis(1),
            deadline: Duration::from_millis(30),
            follow: true,
            follow_poll: Duration::from_millis(1),
        },
    )
    .unwrap();
    assert_eq!(value["job_state"], "RUNNING");
    assert!(
        shown.iter().any(|line| line.contains("keeps running")),
        "{shown:?}"
    );
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
    assert!(text.contains("still being prepared"), "{text}");
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
fn retry_is_refused_while_the_job_has_not_ended() {
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
        world.status = "AUTHORIZED";
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
