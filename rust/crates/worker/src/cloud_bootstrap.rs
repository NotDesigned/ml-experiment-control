//! The worker in an image is only a bootstrap. A package names the worker its Run froze,
//! and when that is another one, this fetches it, proves it and becomes it, with the same
//! arguments, before anything else happens. This reads nothing from a package except
//! `worker_sha256`, because the package may come from a server far newer than this image.
//!
//! The worker goes in a local temporary directory, not on shared storage. The Attempt's
//! directory cannot exist yet: `cloud::prepare` reads an existing one as a launch that
//! began. And shared storage may be mounted noexec, which nothing here can know before
//! trying. A local copy costs one download each time a job starts, and a restarted job
//! makes that download again.
use crate::managed::{Error, Result};
use ml_exp_core::{cloud::launch::MAX_WORKER_BYTES, slurm::helper};
use std::{
    ffi::OsString,
    fs,
    io::{Read, Write},
    os::unix::{
        fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
        process::CommandExt,
    },
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

/// Set only by this module on the worker it starts, naming the hash that worker must have.
const STARTED_AS: &str = "ML_EXPD_WORKER_BOOTSTRAPPED";
const PROGRAM: &str = "ml-exp-worker-rs";

fn refused(why: impl std::fmt::Display) -> Error {
    Error::Bootstrap(experiment_redact::redact_line(&why.to_string()))
}

#[derive(Debug, PartialEq, Eq)]
pub enum Step {
    Run,
    Fetch,
}
/// A worker started by a bootstrap never fetches again, so a bad hash cannot become a loop.
pub fn next(running: &str, required: &str, started_as: Option<&str>) -> Result<Step> {
    if running == required {
        Ok(Step::Run)
    } else if let Some(started_as) = started_as {
        Err(refused(format!(
            "a bootstrap started this worker as {started_as}, the Attempt requires {required}, \
             and this worker is {running}"
        )))
    } else {
        Ok(Step::Fetch)
    }
}
/// The one field of a package this worker relies on. Its other fields are not its business.
pub fn required(package: &[u8]) -> Result<String> {
    let value = ml_exp_core::json::mapping(package)
        .map_err(|_| refused("the launch package cannot be read"))?;
    value
        .get("worker_sha256")
        .and_then(|worker| worker.as_str())
        .filter(|worker| helper::valid_digest(worker))
        .map(str::to_owned)
        .ok_or_else(|| refused("the launch package names no valid worker"))
}

enum Failure {
    /// May pass by itself, and repeating the read has no effect on the server.
    Again(String),
    Final(Error),
}
fn read_once(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
    expected: &str,
) -> std::result::Result<Vec<u8>, Failure> {
    let response = client
        .get(url)
        .bearer_auth(token)
        .send()
        .map_err(|error| Failure::Again(error.without_url().to_string()))?;
    let status = response.status().as_u16();
    if status != 200 {
        return Err(if crate::retry::may_pass(status) {
            Failure::Again(format!("HTTP {status}"))
        } else if status == 409 {
            Failure::Final(refused(format!(
                "the server does not hold worker {expected}, or its copy differs (HTTP 409)"
            )))
        } else {
            Failure::Final(refused(format!(
                "the server refused the worker download with HTTP {status}"
            )))
        });
    }
    let mut bytes = Vec::new();
    response
        .take(MAX_WORKER_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| Failure::Again(format!("the worker's content: {error}")))?;
    if bytes.len() as u64 > MAX_WORKER_BYTES {
        return Err(Failure::Final(refused("the worker is too large")));
    }
    let actual = helper::sha256_bytes(&bytes);
    if actual != expected {
        return Err(Failure::Final(refused(format!(
            "the worker downloaded is {actual}, not {expected}"
        ))));
    }
    Ok(bytes)
}
/// Only the worker this Attempt froze: the request names nothing, and the token is its own.
pub fn fetch(url: &str, token: &str, expected: &str) -> Result<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(refused)?;
    let mut count = 0;
    loop {
        count += 1;
        match read_once(&client, url, token, expected) {
            Ok(bytes) => return Ok(bytes),
            Err(Failure::Final(error)) => return Err(error),
            Err(Failure::Again(why)) => {
                if count >= crate::retry::ATTEMPTS {
                    return Err(refused(format!("the worker download gave up: {why}")));
                }
                let why = experiment_redact::redact_line(&why);
                eprintln!("ML_EXPD_WORKER_BOOTSTRAP=RETRY {count}: {why}");
                std::thread::sleep(crate::retry::delay(count));
            }
        }
    }
}

fn install(directory: &Path, bytes: &[u8]) -> std::io::Result<PathBuf> {
    let mut random = [0u8; 16];
    getrandom::fill(&mut random).map_err(|error| std::io::Error::other(error.to_string()))?;
    let name: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
    let private = directory.join(format!("ml-expd-worker-{name}"));
    fs::DirBuilder::new().mode(0o700).create(&private)?;
    let path = private.join(PROGRAM);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o700)
        .open(&path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file); // Executing a file still open for writing fails with ETXTBSY.
    fs::set_permissions(&path, fs::Permissions::from_mode(0o500))?;
    Ok(path)
}
/// Returns only if the next worker could not be started, in any of `directories`.
pub fn become_worker(
    directories: &[PathBuf],
    bytes: &[u8],
    required: &str,
    arguments: &[OsString],
) -> Error {
    let mut reasons = Vec::new();
    for directory in directories {
        let shown = directory.display();
        let path = match install(directory, bytes) {
            Ok(path) => path,
            Err(error) => {
                reasons.push(format!("{shown}: {error}"));
                continue;
            }
        };
        // The file on disk is what runs, not the bytes that were checked.
        let actual = ml_exp_io::sha256_file(&path).map(|sha| format!("sha256:{sha}"));
        let reason = match actual {
            Ok(actual) if actual == required => {
                eprintln!("ML_EXPD_WORKER_BOOTSTRAP=START {required}");
                let error = Command::new(&path)
                    .args(arguments)
                    .env(STARTED_AS, required)
                    .exec();
                format!("{shown}: cannot be executed: {error}")
            }
            Ok(actual) => format!("{shown}: reads back as {actual}, not {required}"),
            Err(error) => format!("{shown}: cannot be read back: {error}"),
        };
        reasons.push(reason);
        if let Some(parent) = path.parent() {
            let _ = fs::remove_dir_all(parent);
        }
    }
    refused(format!(
        "no local directory can run the worker ({})",
        reasons.join("; ")
    ))
}
fn local_directories() -> Vec<PathBuf> {
    let mut directories = vec![std::env::temp_dir()];
    for directory in ["/tmp", "/var/tmp"] {
        let directory = PathBuf::from(directory);
        if !directories.contains(&directory) {
            directories.push(directory);
        }
    }
    directories
}
/// Returns when this worker is the required one. Otherwise it is replaced, or this fails.
pub fn enter(package: &[u8], package_url: &str, token: &str) -> Result<()> {
    let path = std::env::current_exe().map_err(refused)?;
    let running = format!("sha256:{}", ml_exp_io::sha256_file(&path).map_err(refused)?);
    let started_as = std::env::var(STARTED_AS).ok();
    let arguments: Vec<OsString> = std::env::args_os().skip(1).collect();
    enter_as(
        &running,
        started_as.as_deref(),
        package,
        package_url,
        token,
        &local_directories(),
        &arguments,
    )
}
fn enter_as(
    running: &str,
    started_as: Option<&str>,
    package: &[u8],
    package_url: &str,
    token: &str,
    directories: &[PathBuf],
    arguments: &[OsString],
) -> Result<()> {
    let required = required(package)?;
    if next(running, &required, started_as)? == Step::Run {
        return Ok(());
    }
    let base = package_url
        .strip_suffix("/package")
        .ok_or_else(|| refused("the package address"))?;
    eprintln!("ML_EXPD_WORKER_BOOTSTRAP=FETCH {running} -> {required}");
    let bytes = fetch(&format!("{base}/worker"), token, &required)?;
    Err(become_worker(directories, &bytes, &required, arguments))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_core::cloud::launch;
    use std::net::TcpListener;

    const RUNNING: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";
    const REQUIRED: &str =
        "sha256:2222222222222222222222222222222222222222222222222222222222222222";

    fn shown(error: Error) -> String {
        error.to_string()
    }
    /// Answers each request in turn with a status and body, and says what it was asked.
    fn server(replies: Vec<(u16, Vec<u8>)>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for (status, body) in replies {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                let mut byte = [0u8; 1];
                while !request.ends_with(b"\r\n\r\n") {
                    if stream.read(&mut byte).unwrap() == 0 {
                        break;
                    }
                    request.push(byte[0]);
                }
                let text = String::from_utf8_lossy(&request).to_string();
                let path = text.split_whitespace().nth(1).unwrap_or("").to_owned();
                let token = text
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("authorization: bearer ")
                            .map(str::to_owned)
                    })
                    .unwrap_or_default();
                seen.push(format!("{path} {token}"));
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .unwrap();
                stream.write_all(&body).unwrap();
            }
            seen
        });
        (url, thread)
    }
    #[test]
    fn the_decision_runs_a_match_fetches_a_difference_and_never_fetches_twice() {
        assert_eq!(next(REQUIRED, REQUIRED, None).unwrap(), Step::Run);
        assert_eq!(next(REQUIRED, REQUIRED, Some(REQUIRED)).unwrap(), Step::Run);
        assert_eq!(next(RUNNING, REQUIRED, None).unwrap(), Step::Fetch);
        let refused = shown(next(RUNNING, REQUIRED, Some(REQUIRED)).unwrap_err());
        assert!(refused.contains(RUNNING), "{refused}");
        assert!(refused.contains(REQUIRED), "{refused}");
        assert!(refused.contains("training was not started"), "{refused}");
    }
    #[test]
    fn a_package_from_a_later_server_is_read_for_its_worker_alone() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("runs");
        let package = launch::compile(launch::Input {
            root: root.to_str().unwrap(),
            project: "demo",
            run: "train",
            attempt: "attempt-001",
            workdir: "/workspace",
            argv: &["train".to_owned()],
            timeout_seconds: 30,
            worker_sha256: REQUIRED,
            storage: launch::StorageLayout {
                mount: temp.path().into(),
                runs: root.clone(),
                datasets: temp.path().join("datasets"),
            },
            channel: ml_exp_core::cloud::exchange::Channel {
                url: "https://127.0.0.1:1/api/worker-exchange/demo/train/attempt-001".into(),
                token: "b".repeat(64),
            },
        })
        .unwrap();
        let mut later = serde_json::to_value(&package).unwrap();
        crate::cloud::decode(&serde_json::to_vec(&later).unwrap()).unwrap();
        later["a_field_of_the_future"] = serde_json::json!({"x":1});
        let bytes = serde_json::to_vec(&later).unwrap();
        // Full decoding refuses it, which is why the bootstrap comes first.
        assert!(crate::cloud::decode(&bytes).is_err());
        assert_eq!(required(&bytes).unwrap(), REQUIRED);
        for bad in [
            serde_json::json!({}),
            serde_json::json!({"worker_sha256":"sha256:short"}),
            serde_json::json!({"worker_sha256":1}),
        ] {
            assert!(required(&serde_json::to_vec(&bad).unwrap()).is_err());
        }
        assert!(required(b"not json").is_err());
    }
    #[test]
    fn a_worker_with_the_wrong_hash_is_refused_and_not_tried_again() {
        let good = b"the frozen worker".to_vec();
        let expected = helper::sha256_bytes(&good);
        let (url, thread) = server(vec![(200, b"another worker".to_vec())]);
        let refused = shown(fetch(&format!("{url}/worker"), "tok", &expected).unwrap_err());
        assert!(refused.contains(&expected), "{refused}");
        assert_eq!(thread.join().unwrap().len(), 1);
    }
    #[test]
    fn a_passing_failure_is_tried_again_and_a_refusal_is_not() {
        let good = b"the frozen worker".to_vec();
        let expected = helper::sha256_bytes(&good);
        let (url, thread) = server(vec![(503, Vec::new()), (200, good.clone())]);
        assert_eq!(
            fetch(&format!("{url}/worker"), "tok", &expected).unwrap(),
            good
        );
        assert_eq!(thread.join().unwrap(), ["/worker tok", "/worker tok"]);
        for (status, words) in [(401, "HTTP 401"), (404, "HTTP 404"), (409, "does not hold")] {
            let (url, thread) = server(vec![(status, Vec::new())]);
            let refused = shown(fetch(&format!("{url}/worker"), "tok", &expected).unwrap_err());
            assert!(refused.contains(words), "{refused}");
            assert!(!refused.contains("tok"), "{refused}");
            assert_eq!(thread.join().unwrap().len(), 1);
        }
    }
    #[test]
    fn what_cannot_be_executed_names_each_place_and_stops() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("absent");
        let bytes = b"not a program";
        let refused = shown(become_worker(
            &[missing.clone(), temp.path().into()],
            bytes,
            &helper::sha256_bytes(bytes),
            &[],
        ));
        assert!(refused.contains(missing.to_str().unwrap()), "{refused}");
        assert!(refused.contains("cannot be executed"), "{refused}");
        assert!(refused.contains("training was not started"), "{refused}");
        // Nothing that failed stays behind.
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
        let refused = shown(become_worker(&[temp.path().into()], bytes, REQUIRED, &[]));
        assert!(refused.contains("reads back as"), "{refused}");
    }
    /// The whole way through a real exec, which must happen in a process of its own.
    #[test]
    #[ignore = "started by the test below"]
    fn exec_child() {
        let (Ok(url), Ok(directory)) = (std::env::var("BOOT_URL"), std::env::var("BOOT_DIR"))
        else {
            return;
        };
        let required = std::env::var("BOOT_REQUIRED").unwrap();
        let package = serde_json::to_vec(&serde_json::json!({"worker_sha256":required})).unwrap();
        let error = enter_as(
            RUNNING,
            None,
            &package,
            &format!("{url}/api/worker-exchange/demo/train/attempt-001/package"),
            "tok",
            &[PathBuf::from(directory)],
            &["cloud-launch".into(), "--package-url".into(), "same".into()],
        )
        .unwrap_err();
        panic!("still here: {error}");
    }
    #[test]
    fn a_worker_fetched_from_the_attempts_endpoint_really_replaces_this_process() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("seen");
        let program = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$0\" \"$@\" \"$ML_EXPD_WORKER_BOOTSTRAPPED\" > {}\n",
            output.display()
        )
        .into_bytes();
        let required = helper::sha256_bytes(&program);
        let (url, thread) = server(vec![(200, program)]);
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "cloud_bootstrap::tests::exec_child",
                "--ignored",
                "--nocapture",
            ])
            .env("BOOT_URL", &url)
            .env("BOOT_DIR", temp.path())
            .env("BOOT_REQUIRED", &required)
            .status()
            .unwrap();
        assert!(status.success());
        assert_eq!(
            thread.join().unwrap(),
            ["/api/worker-exchange/demo/train/attempt-001/worker tok"]
        );
        let seen = fs::read_to_string(output).unwrap();
        let lines: Vec<&str> = seen.lines().collect();
        assert!(
            lines[0].starts_with(temp.path().to_str().unwrap()),
            "{seen}"
        );
        assert_eq!(lines[1..4], ["cloud-launch", "--package-url", "same"]);
        assert_eq!(lines[4], required);
    }
}
