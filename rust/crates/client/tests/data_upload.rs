#![cfg(unix)]
use ml_exp_client::{
    Client, Error,
    datasets::{self, ObjectStore, Part},
};
use ml_exp_core::{datasets::State, multipart::PartRecord, slurm::helper::Target};
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

/// Nothing listens on this port, so anything that reaches the network comes back as
/// an unconfirmed transfer rather than the local refusal.
fn upload(files: usize) -> Error {
    let directory = tempfile::tempdir().unwrap();
    for n in 0..files {
        let name = format!("dataset-tasks-sample-{n:05}.bin");
        fs::write(directory.path().join(name), b"x").unwrap();
    }
    let client = Client::new("http://127.0.0.1:9", "source-test-secret".into()).unwrap();
    let health = json!({"capabilities":["native-datasets.v1"]});
    datasets::send(&client, "demo", "slurm", directory.path(), &health, false).unwrap_err()
}

#[test]
fn a_dataset_no_run_could_mount_is_refused_before_any_request() {
    assert!(matches!(
        upload(2500),
        Error::Observation(message) if message.contains("256 KiB")
    ));
    assert!(matches!(upload(3), Error::DatasetOutcomeUnknown { .. }));
}

const TWO_PARTS: usize = 17 * 1024 * 1024;
const DIRECT: &str = "native-dataset-direct-upload.v1";

/// The dataset routes as the real service answers them, over a real socket.
#[derive(Default)]
struct Server {
    state: Mutex<Option<State>>,
    /// The method and route of each request, without its asset.
    seen: Mutex<Vec<String>>,
    /// Answered to every ticket request, as (status, error code).
    refuse_tickets: Mutex<Option<(u16, &'static str)>>,
    ticket_url: Mutex<Option<String>>,
}
impl Server {
    fn start(self: &Arc<Self>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = self.clone();
        std::thread::spawn(move || {
            for socket in listener.incoming() {
                let mut socket = socket.unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let mut words = line.split_whitespace();
                let method = words.next().unwrap_or_default().to_owned();
                let path = words.next().unwrap_or_default().to_owned();
                let mut length = 0;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    let lower = header.trim().to_lowercase();
                    if lower.is_empty() {
                        break;
                    }
                    if let Some(value) = lower.strip_prefix("content-length: ") {
                        length = value.parse().unwrap();
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let (status, code, reply) = server.answer(&method, &path, &body);
                let reply = reply.to_string();
                let _ = write!(
                    socket,
                    "HTTP/1.1 {status} X\r\nX-ML-Expd-Error-Code: {code}\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n{reply}",
                    reply.len()
                );
            }
        });
        base
    }
    fn view(&self) -> Value {
        let mut value = json!(self.state.lock().unwrap().as_ref().unwrap());
        value["archive_verified"] = json!(false);
        value
    }
    fn record(&self, part: u64, sha256: &str) {
        let mut state = self.state.lock().unwrap();
        let bytes = state.as_ref().unwrap().definition.part_size(part).unwrap();
        let record = PartRecord {
            sha256: sha256.into(),
            bytes,
        };
        let next = state.as_ref().unwrap().record_part(part, record).unwrap();
        *state = Some(next);
    }
    fn answer(&self, method: &str, path: &str, body: &[u8]) -> (u16, &'static str, Value) {
        let route = path.rsplit('/').next().unwrap();
        let hashed = path.contains("/parts/");
        self.seen
            .lock()
            .unwrap()
            .push(format!("{method} {}", if hashed { "parts" } else { route }));
        match (method, route) {
            ("POST", "s") => {
                let definition = serde_json::from_slice(body).unwrap();
                let mut state = self.state.lock().unwrap();
                if state.is_none() {
                    let target = Target {
                        backend_id: "s".into(),
                        profile_sha256: format!("sha256:{}", "3".repeat(64)),
                        storage_scope: format!("sha256:{}", "4".repeat(64)),
                    };
                    *state = Some(State::new("demo".into(), target, definition).unwrap());
                }
                drop(state);
                (200, "", self.view())
            }
            ("POST", "upload-ticket") => {
                if let Some((status, code)) = *self.refuse_tickets.lock().unwrap() {
                    return (status, code, json!({"code":code,"detail":"refused"}));
                }
                let asked: Value = serde_json::from_slice(body).unwrap();
                let part = asked["part"].as_u64().unwrap();
                let state = self.state.lock().unwrap();
                let definition = &state.as_ref().unwrap().definition;
                let url = self.ticket_url.lock().unwrap().clone().unwrap_or(format!(
                    "https://bucket.oss-cn-example.aliyuncs.com/users/a/b/datasets/demo/s/{}/parts/{part:05}?Signature=TICKET-SECRET",
                    definition.sha256
                ));
                (
                    200,
                    "",
                    json!({"part":part,"bytes":definition.part_size(part).unwrap(),"url":url,
                    "headers":{"x-oss-forbid-overwrite":"true"},"expires_in_seconds":300}),
                )
            }
            ("POST", "part-uploaded") => {
                let asked: Value = serde_json::from_slice(body).unwrap();
                self.record(
                    asked["part"].as_u64().unwrap(),
                    asked["sha256"].as_str().unwrap(),
                );
                (200, "", self.view())
            }
            ("PUT", sha256) => {
                let part = path
                    .split("/parts/")
                    .nth(1)
                    .unwrap()
                    .split('/')
                    .next()
                    .unwrap();
                self.record(part.parse().unwrap(), sha256);
                (200, "", self.view())
            }
            ("POST", "complete") => {
                let mut value = self.view();
                value["archive_verified"] = json!(true);
                (200, "", value)
            }
            _ => (404, "", json!({})),
        }
    }
    fn count(&self, what: &str) -> usize {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .filter(|s| *s == what)
            .count()
    }
}
/// The object store. Each PUT takes the next scripted answer, then 200.
#[derive(Default)]
struct Store {
    puts: Vec<(String, u64)>,
    script: Vec<Result<u16, String>>,
    waits: Vec<Duration>,
}
impl ObjectStore for Store {
    fn put(&mut self, url: &url::Url, part: &Part) -> Result<u16, String> {
        let bytes: &[u8] = part.as_ref();
        self.puts.push((url.to_string(), bytes.len() as u64));
        if self.script.is_empty() {
            Ok(200)
        } else {
            self.script.remove(0)
        }
    }
    fn wait(&mut self, delay: Duration) {
        self.waits.push(delay);
    }
}
/// Enough to need a second part, which costs time to pack: only where two matter.
fn two_parts() -> (tempfile::TempDir, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("train.bin"), vec![1u8; TWO_PARTS]).unwrap();
    let path = directory.path().to_owned();
    (directory, path)
}
fn one_part() -> (tempfile::TempDir, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("train.bin"), b"data").unwrap();
    let path = directory.path().to_owned();
    (directory, path)
}
fn health(direct: bool) -> Value {
    let mut names = vec!["native-datasets.v1"];
    if direct {
        names.push(DIRECT);
    }
    json!({"capabilities":names})
}
fn client(server: &Arc<Server>) -> Client {
    Client::new(&server.start(), "source-test-secret".into()).unwrap()
}
fn send(
    client: &Client,
    path: &Path,
    health: &Value,
    via_server: bool,
    store: &mut Store,
) -> Result<Value, Error> {
    datasets::upload_with(client, "demo", "s", path, health, via_server, store)
}
#[test]
fn parts_go_to_object_storage_and_the_server_receives_only_metadata() {
    let (_directory, path) = two_parts();
    let server = Arc::new(Server::default());
    let client = client(&server);
    let mut store = Store::default();
    let value = send(&client, &path, &health(true), false, &mut store).unwrap();
    assert_eq!(value["archive_verified"], true);
    assert_eq!(store.puts.len(), 2);
    assert!(store.puts[0].0.contains("/parts/00000?"));
    assert!(store.puts[1].0.contains("/parts/00001?"));
    assert_eq!(store.puts[0].1, 16 * 1024 * 1024);
    assert_eq!(server.count("POST upload-ticket"), 2);
    assert_eq!(server.count("POST part-uploaded"), 2);
    assert_eq!(server.count("PUT parts"), 0);
    assert_eq!(server.count("POST complete"), 1);
}
#[test]
fn a_repeat_skips_accepted_parts_and_a_refusal_names_the_way_out_without_the_ticket() {
    let (_directory, path) = two_parts();
    let server = Arc::new(Server::default());
    let client = client(&server);
    let mut store = Store {
        script: vec![Ok(200), Ok(403)],
        ..Store::default()
    };
    let error = send(&client, &path, &health(true), false, &mut store).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("--via-server"), "{text}");
    assert!(!text.contains("TICKET-SECRET"), "{text}");
    assert!(!text.contains("aliyuncs"), "{text}");
    // Told what happened, and nothing was quietly sent another way.
    assert_eq!(store.puts.len(), 2);
    assert_eq!(server.count("PUT parts"), 0);
    assert_eq!(server.count("POST complete"), 0);
    assert_eq!(
        server.state.lock().unwrap().as_ref().unwrap().parts.len(),
        1
    );

    let mut store = Store::default();
    send(&client, &path, &health(true), false, &mut store).unwrap();
    assert_eq!(store.puts.len(), 1, "only the part not accepted");
    assert!(store.puts[0].0.contains("/parts/00001?"));
    // The rest can go the other way, because both record the same parts.
    let server = Arc::new(Server::default());
    let client2 = self::client(&server);
    let mut store = Store {
        script: vec![Ok(200), Ok(403)],
        ..Store::default()
    };
    send(&client2, &path, &health(true), false, &mut store).unwrap_err();
    let mut store = Store::default();
    send(&client2, &path, &health(true), true, &mut store).unwrap();
    assert!(store.puts.is_empty());
    assert_eq!(server.count("PUT parts"), 1);
}
#[test]
fn retries_are_bounded_each_with_a_fresh_ticket_and_only_for_what_may_pass() {
    let (_directory, path) = one_part();
    // Two passing failures, then 409 meaning already stored, which is a success.
    let server = Arc::new(Server::default());
    let client1 = client(&server);
    let mut store = Store {
        script: vec![Ok(503), Err("connection reset".into()), Ok(409)],
        ..Store::default()
    };
    send(&client1, &path, &health(true), false, &mut store).unwrap();
    assert_eq!(store.puts.len(), 3);
    assert_eq!(store.waits.len(), 2);
    assert_eq!(server.count("POST upload-ticket"), 3);
    // Forever failing stops at the worker's bound, and names the fallback.
    let server = Arc::new(Server::default());
    let client2 = client(&server);
    let mut store = Store {
        script: vec![Ok(503); 100],
        ..Store::default()
    };
    let text = send(&client2, &path, &health(true), false, &mut store)
        .unwrap_err()
        .to_string();
    assert!(text.contains("--via-server"), "{text}");
    assert_eq!(store.puts.len(), ml_exp_core::retry::ATTEMPTS as usize);
    assert_eq!(server.count("PUT parts"), 0);
    // A refusal is not repeated.
    let server = Arc::new(Server::default());
    let client3 = client(&server);
    let mut store = Store {
        script: vec![Ok(400)],
        ..Store::default()
    };
    send(&client3, &path, &health(true), false, &mut store).unwrap_err();
    assert_eq!(store.puts.len(), 1);
    assert!(store.waits.is_empty());
}
#[test]
fn the_server_path_is_chosen_only_before_anything_is_sent() {
    let (_directory, path) = one_part();
    // Asked for.
    let server = Arc::new(Server::default());
    let c = client(&server);
    let mut store = Store::default();
    send(&c, &path, &health(true), true, &mut store).unwrap();
    assert!(store.puts.is_empty());
    assert_eq!(server.count("POST upload-ticket"), 0);
    assert_eq!(server.count("PUT parts"), 1);
    // An older server offers no tickets, and is not asked for one.
    let server = Arc::new(Server::default());
    let c = client(&server);
    let mut store = Store::default();
    send(&c, &path, &health(false), false, &mut store).unwrap();
    assert!(store.puts.is_empty());
    assert_eq!(server.count("POST upload-ticket"), 0);
    assert_eq!(server.count("PUT parts"), 1);
    // An executor with no object store, such as Slurm, on a server that has one.
    let server = Arc::new(Server::default());
    *server.refuse_tickets.lock().unwrap() = Some((422, "DATASET_DIRECT_UPLOAD_UNSUPPORTED"));
    let c = client(&server);
    let mut store = Store::default();
    send(&c, &path, &health(true), false, &mut store).unwrap();
    assert!(store.puts.is_empty());
    assert_eq!(server.count("POST upload-ticket"), 1, "asked once");
    assert_eq!(server.count("PUT parts"), 1);
    // Any other refusal of a ticket is not a reason to send another way.
    let server = Arc::new(Server::default());
    *server.refuse_tickets.lock().unwrap() = Some((409, "CLOUD_EVIDENCE"));
    let c = client(&server);
    let mut store = Store::default();
    send(&c, &path, &health(true), false, &mut store).unwrap_err();
    assert_eq!(server.count("PUT parts"), 0);
}
#[test]
fn a_ticket_that_is_not_for_this_part_on_https_object_storage_is_never_used() {
    let (_directory, path) = one_part();
    let tail = "datasets/demo/s/SHA/parts/00000?Signature=x";
    for url in [
        format!("http://bucket.oss-cn-example.aliyuncs.com/{tail}"),
        format!("https://bucket.example.com/{tail}"),
        format!("https://aliyuncs.com.example.com/{tail}"),
        format!("https://user:pass@bucket.oss-cn-example.aliyuncs.com/{tail}"),
        "https://bucket.oss-cn-example.aliyuncs.com/datasets/demo/other/parts/00000".to_owned(),
        "not a url".to_owned(),
    ] {
        let server = Arc::new(Server::default());
        *server.ticket_url.lock().unwrap() = Some(url.clone());
        let c = client(&server);
        let mut store = Store::default();
        send(&c, &path, &health(true), false, &mut store).unwrap_err();
        assert!(store.puts.is_empty(), "{url}");
    }
}
