//! An in-process MLflow REST server for tests. It implements the calls the
//! publisher makes with the SQL store's behaviour that publication relies on:
//! one row per identical (key, timestamp, step, value) point, writes accepted
//! on a finished run, a non-idempotent `runs/create`, HTTP Basic auth, and the
//! `runs/log-batch` limits. It can commit a write and then drop its response.
use base64::Engine as _;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

/// One stored point: key, timestamp, step and the value's bits.
pub type Point = (String, i64, i64, u64);

#[derive(Clone, Debug, Default)]
pub struct Run {
    pub id: String,
    pub experiment: String,
    pub name: String,
    pub status: String,
    pub lifecycle: String,
    pub start_time: i64,
    pub tags: BTreeMap<String, String>,
    pub params: BTreeMap<String, String>,
    pub points: BTreeSet<Point>,
}

#[derive(Default)]
pub struct Data {
    /// id, name, lifecycle stage
    pub experiments: Vec<(String, String, String)>,
    pub runs: BTreeMap<String, Run>,
    /// Calls received per endpoint, such as `runs/log-batch`.
    pub calls: BTreeMap<String, u64>,
    /// Commit the next N calls to an endpoint, then close without a response.
    pub drop_after_commit: BTreeMap<String, u64>,
    /// Answer every call with this status and do nothing.
    pub fail_with: Option<u16>,
}

pub struct Server {
    pub url: String,
    pub data: Arc<Mutex<Data>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Server {
    pub fn start(username: &str, password: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let data = Arc::new(Mutex::new(Data::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let expected = format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode(format!("{username}:{password}"))
        );
        let (shared, stopping) = (data.clone(), stop.clone());
        let thread = thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => handle(stream, &shared, &expected),
                    Err(_) => thread::sleep(Duration::from_millis(2)),
                }
            }
        });
        Self {
            url,
            data,
            stop,
            thread: Some(thread),
        }
    }
    pub fn with<T>(&self, f: impl FnOnce(&mut Data) -> T) -> T {
        f(&mut self.data.lock().unwrap())
    }
    pub fn calls(&self, endpoint: &str) -> u64 {
        self.with(|data| data.calls.get(endpoint).copied().unwrap_or(0))
    }
    pub fn runs(&self) -> Vec<Run> {
        self.with(|data| data.runs.values().cloned().collect())
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn handle(stream: TcpStream, data: &Mutex<Data>, expected: &str) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return;
    };
    let (method, target) = (method.to_owned(), target.to_owned());
    let mut length = 0;
    let mut authorization = String::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).is_err() {
            return;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            match name.trim().to_ascii_lowercase().as_str() {
                "content-length" => length = value.trim().parse().unwrap_or(0),
                "authorization" => authorization = value.trim().to_owned(),
                _ => {}
            }
        }
    }
    let mut body = vec![0; length];
    if reader.read_exact(&mut body).is_err() {
        return;
    }
    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
    let endpoint = path
        .strip_prefix("/api/2.0/mlflow/")
        .unwrap_or(path)
        .to_owned();
    let query: BTreeMap<String, String> = url::form_urlencoded::parse(query.as_bytes())
        .into_owned()
        .collect();
    let mut data = data.lock().unwrap();
    *data.calls.entry(endpoint.clone()).or_default() += 1;
    let (status, reply) = if let Some(status) = data.fail_with {
        (status, json!({"error_code":"INTERNAL_ERROR"}))
    } else if authorization != expected {
        (401, json!({"error_code":"UNAUTHENTICATED"}))
    } else if body.len() > 1024 * 1024 {
        (400, invalid("request too large"))
    } else {
        let request: Value = if body.is_empty() {
            json!({})
        } else {
            serde_json::from_slice(&body).unwrap_or(Value::Null)
        };
        route(&mut data, &method, &endpoint, &query, &request)
    };
    let dropped = data.drop_after_commit.get_mut(&endpoint).is_some_and(|n| {
        let drop = *n > 0;
        *n = n.saturating_sub(1);
        drop
    });
    drop(data);
    let mut stream = reader.into_inner();
    if dropped {
        let _ = stream.shutdown(std::net::Shutdown::Both);
        return;
    }
    let text = reply.to_string();
    let _ = write!(
        stream,
        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    );
    let _ = stream.flush();
}

fn invalid(message: &str) -> Value {
    json!({"error_code":"INVALID_PARAMETER_VALUE","message":message})
}
fn missing() -> (u16, Value) {
    (404, json!({"error_code":"RESOURCE_DOES_NOT_EXIST"}))
}
fn info(run: &Run) -> Value {
    json!({"run_id":run.id,"experiment_id":run.experiment,"run_name":run.name,"status":run.status,
        "lifecycle_stage":run.lifecycle,"start_time":run.start_time})
}
fn document(run: &Run) -> Value {
    let pairs = |map: &BTreeMap<String, String>| -> Vec<Value> {
        map.iter()
            .map(|(k, v)| json!({"key":k,"value":v}))
            .collect()
    };
    json!({"info":info(run),"data":{"tags":pairs(&run.tags),"params":pairs(&run.params)}})
}
/// MLflow's name rule, as the real server checks it.
fn valid_name(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 250
        && key
            .chars()
            .all(|c| c.is_alphanumeric() || "_-./ ".contains(c))
        && !key.starts_with('/')
        && key.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

fn route(
    data: &mut Data,
    method: &str,
    endpoint: &str,
    query: &BTreeMap<String, String>,
    request: &Value,
) -> (u16, Value) {
    match (method, endpoint) {
        ("GET", "experiments/get-by-name") => {
            let name = query.get("experiment_name").cloned().unwrap_or_default();
            match data.experiments.iter().find(|e| e.1 == name) {
                Some((id, name, stage)) => (
                    200,
                    json!({"experiment":{"experiment_id":id,"name":name,"lifecycle_stage":stage}}),
                ),
                None => missing(),
            }
        }
        ("POST", "experiments/create") => {
            let name = request["name"].as_str().unwrap_or_default().to_owned();
            if data.experiments.iter().any(|e| e.1 == name) {
                return (400, json!({"error_code":"RESOURCE_ALREADY_EXISTS"}));
            }
            let id = (data.experiments.len() + 1).to_string();
            data.experiments
                .push((id.clone(), name, "active".to_owned()));
            (200, json!({"experiment_id":id}))
        }
        ("POST", "runs/search") => {
            let filter = request["filter"].as_str().unwrap_or_default();
            let key = filter.split('`').nth(1).unwrap_or_default();
            let value = filter.split('\'').nth(1).unwrap_or_default();
            let experiments: Vec<&str> = request["experiment_ids"]
                .as_array()
                .map(|ids| ids.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let runs: Vec<Value> = data
                .runs
                .values()
                .filter(|run| {
                    experiments.contains(&run.experiment.as_str())
                        && run.lifecycle == "active"
                        && run.tags.get(key).is_some_and(|v| v == value)
                })
                .map(document)
                .collect();
            (200, json!({"runs":runs}))
        }
        ("POST", "runs/create") => {
            let id = format!("run{:029}", data.runs.len() + 1);
            let mut run = Run {
                id: id.clone(),
                experiment: request["experiment_id"].as_str().unwrap_or_default().into(),
                name: request["run_name"].as_str().unwrap_or_default().into(),
                status: "RUNNING".into(),
                lifecycle: "active".into(),
                start_time: request["start_time"].as_i64().unwrap_or(0),
                ..Run::default()
            };
            for tag in request["tags"].as_array().into_iter().flatten() {
                run.tags.insert(
                    tag["key"].as_str().unwrap_or_default().into(),
                    tag["value"].as_str().unwrap_or_default().into(),
                );
            }
            let reply = json!({"run":document(&run)});
            data.runs.insert(id, run);
            (200, reply)
        }
        ("GET", "runs/get") => {
            match data
                .runs
                .get(query.get("run_id").map_or("", String::as_str))
            {
                Some(run) => (200, json!({"run":document(run)})),
                None => missing(),
            }
        }
        ("POST", "runs/update") => {
            let Some(run) = data
                .runs
                .get_mut(request["run_id"].as_str().unwrap_or_default())
            else {
                return missing();
            };
            run.status = request["status"].as_str().unwrap_or_default().into();
            (200, json!({"run_info":info(run)}))
        }
        ("POST", "runs/log-batch") => log_batch(data, request),
        ("GET", "metrics/get-history") => {
            let Some(run) = data
                .runs
                .get(query.get("run_id").map_or("", String::as_str))
            else {
                return missing();
            };
            let key = query.get("metric_key").cloned().unwrap_or_default();
            let metrics: Vec<Value> = run
                .points
                .iter()
                .filter(|point| point.0 == key)
                .map(|(key, timestamp, step, bits)| {
                    json!({"key":key,"value":f64::from_bits(*bits),"timestamp":timestamp,"step":step})
                })
                .collect();
            (200, json!({"metrics":metrics}))
        }
        _ => (404, json!({"error_code":"ENDPOINT_NOT_FOUND"})),
    }
}

fn log_batch(data: &mut Data, request: &Value) -> (u16, Value) {
    let empty = Vec::new();
    let metrics = request["metrics"].as_array().unwrap_or(&empty);
    let params = request["params"].as_array().unwrap_or(&empty);
    if metrics.len() > 1000 || params.len() > 100 || metrics.len() + params.len() > 1000 {
        return (400, invalid("batch limit"));
    }
    let Some(run) = data
        .runs
        .get_mut(request["run_id"].as_str().unwrap_or_default())
    else {
        return missing();
    };
    if run.lifecycle != "active" {
        return (400, invalid("run must be active"));
    }
    let mut points = Vec::new();
    for metric in metrics {
        let key = metric["key"].as_str().unwrap_or_default();
        let (Some(value), Some(timestamp)) =
            (metric["value"].as_f64(), metric["timestamp"].as_i64())
        else {
            return (400, invalid("metric"));
        };
        if !valid_name(key) {
            return (400, invalid("metric name"));
        }
        points.push((
            key.to_owned(),
            timestamp,
            metric["step"].as_i64().unwrap_or(0),
            value.to_bits(),
        ));
    }
    let mut new_params = BTreeMap::new();
    for param in params {
        let key = param["key"].as_str().unwrap_or_default();
        let value = param["value"].as_str().unwrap_or_default();
        if !valid_name(key) || value.chars().count() > 6000 {
            return (400, invalid("param"));
        }
        // Params are immutable: the same value again is accepted, another is not.
        if run.params.get(key).is_some_and(|old| old != value) {
            return (400, invalid("param changed"));
        }
        new_params.insert(key.to_owned(), value.to_owned());
    }
    run.params.extend(new_params);
    run.points.extend(points);
    (200, json!({}))
}

/// A runtime that signs in to `url` as `admin` with `password`.
pub fn runtime(url: &str, dir: &std::path::Path, password: &str) -> super::Runtime {
    use std::os::unix::fs::PermissionsExt;
    let file = dir.join(format!("mlflow-{password}.password"));
    std::fs::write(&file, password).unwrap();
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
    super::Runtime::open(
        super::Config::from_value(json!({"url":url,"username":"admin","password_file":file}))
            .unwrap(),
    )
    .unwrap()
}
/// An address nothing listens on.
pub fn closed_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    format!("http://{}", listener.local_addr().unwrap())
}
