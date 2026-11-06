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
    /// The backend's states, oldest first; the last one stays.
    job_states: Vec<(&'static str, &'static str)>,
    reads_down: bool,
    progress_reads: usize,
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
            job_states: vec![("RUNNING", "Running"), ("SUCCEEDED", "Succeeded")],
            reads_down: false,
            progress_reads: 0,
        }
    }
    fn submission(&self) -> Value {
        json!({"submission_id":ID,"project":"demo","run_id":"train","attempt_id":"attempt-001",
            "operation":"SUBMIT_RUN","status":self.status,"next_action":self.next_action,
            "confirmation":format!("EXECUTE {ID}"),
            "execution":{"status":self.status,"revision":0}})
    }
    fn view(&self, path: &str) -> Value {
        let id = path.rsplit('/').find(|p| p.starts_with("preparation-"));
        let ready = self.continues > self.continues_until_ready;
        json!({"project":"demo","preparation_id":id,"run_id":"train",
            "runtime":{"runtime_id":RUNTIME,
                "status":if self.build_failed {"PREPARED"} else if ready {"READY"} else {"EXECUTING"},
                "build_error":if self.build_failed {json!({"code":"BAD_DOCKERFILE"})} else {Value::Null}},
            "phase":"ENVIRONMENT_PREPARATION","status":"PREPARING",
            "confirmation":format!("PREPARE {}",id.unwrap_or_default()),
            "submission":if ready {self.submission()} else {Value::Null},
            "training_authorized":false,"scheduler_submitted":false})
    }
    fn reply(&mut self, method: &str, path: &str, body: &[u8]) -> Option<(u16, Value)> {
        let route = path.split('?').next().unwrap_or(path);
        Some(match (method, route) {
            ("GET", "/api/health") => (
                200,
                json!({"capabilities":["native-submissions.v1","source-archive.v1",
                    "native-experiment-preparations.v1","native-runtime-builds.v1",
                    "native-scheduler-observation.v1"],
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
            ("GET", "/api/submissions/action-0123456789abcdef") => (200, self.submission()),
            ("POST", r) if r.ends_with("/authorize") => {
                self.status = "AUTHORIZED";
                self.next_action = "EXECUTE";
                (200, self.submission())
            }
            ("POST", r) if r.ends_with("/execute") => {
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
                let (state, own) = if self.job_states.len() > 1 {
                    self.job_states.remove(0)
                } else {
                    self.job_states[0]
                };
                let mut value = self.submission();
                value["cloud_observation"] = json!({"observation":{"job_id":"job",
                    "provider_state":own,"state":state,"queue":null}});
                (200, value)
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
        let world = Mutex::new(world);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = {
            let (requests, stop) = (requests.clone(), stop.clone());
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
            run_id: None,
            data: vec![],
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
