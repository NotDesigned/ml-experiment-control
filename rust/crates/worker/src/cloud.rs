//! Cloud bootstrap around the existing sealed worker. It never installs packages,
//! picks a checkpoint or replays an uncertain process. The only code it fetches is the
//! source frozen in its package, which `cloud_code` verifies before anything starts.
use crate::managed;
use ml_exp_core::{
    cloud::launch::{MAX_PACKAGE_BYTES, Package},
    managed::evidence::Observation,
    slurm::helper,
};
use ml_exp_io::directory::Directory;
use std::{collections::BTreeMap, ffi::OsStr, sync::atomic::AtomicBool};
pub fn decode(bytes: &[u8]) -> managed::Result<Package> {
    if bytes.len() > MAX_PACKAGE_BYTES {
        return Err(managed::Error::Configuration);
    }
    let value = ml_exp_core::json::mapping(bytes).map_err(|_| managed::Error::Configuration)?;
    let package: Package = serde_json::from_value(serde_json::Value::Object(value))
        .map_err(|_| managed::Error::Configuration)?;
    package
        .validate()
        .map_err(|_| managed::Error::Configuration)?;
    Ok(package)
}
fn download_error(operation: &'static str, http_status: Option<u16>) -> managed::Error {
    managed::Error::PackageDownload {
        operation,
        http_status,
    }
}
fn download_bytes(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
    expected: &str,
    timeout: std::time::Duration,
) -> managed::Result<Vec<u8>> {
    use std::io::Read;
    let response = client
        .get(url)
        .bearer_auth(token)
        .timeout(timeout)
        .send()
        .map_err(|_| download_error("request", None))?;
    let status = response.status().as_u16();
    if status != 200 {
        return Err(download_error("request", Some(status)));
    }
    if response
        .content_length()
        .is_some_and(|n| n > MAX_PACKAGE_BYTES as u64)
    {
        return Err(download_error("size", Some(status)));
    }
    let mut bytes = Vec::new();
    response
        .take(MAX_PACKAGE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| download_error("response_body", Some(status)))?;
    if bytes.len() > MAX_PACKAGE_BYTES {
        return Err(download_error("size", Some(status)));
    }
    if helper::sha256_bytes(&bytes) != expected {
        return Err(download_error("sha256", Some(status)));
    }
    Ok(bytes)
}
fn download_with_retries(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
    expected: &str,
) -> managed::Result<Vec<u8>> {
    let started = std::time::Instant::now();
    let budget = std::time::Duration::from_secs(60);
    for attempt in 0..3 {
        let result = download_bytes(
            client,
            url,
            token,
            expected,
            budget.saturating_sub(started.elapsed()),
        );
        let retryable = matches!(
            &result,
            Err(managed::Error::PackageDownload {
                operation: "request" | "response_body",
                http_status: None | Some(200 | 429 | 502 | 503 | 504)
            })
        );
        if !retryable || attempt == 2 || started.elapsed() >= budget {
            return result;
        }
        if let Err(error) = &result {
            eprintln!("ML_EXPD_PACKAGE_DOWNLOAD=RETRY {error}");
        }
        std::thread::sleep(
            std::time::Duration::from_secs(attempt + 1)
                .min(budget.saturating_sub(started.elapsed())),
        );
    }
    unreachable!("bounded package download attempts")
}
/// Download metadata only through the existing exact Attempt capability.
/// Redirects are disabled so its token never follows a different origin.
pub fn download(url: &str, token: &str, expected: &str) -> managed::Result<Package> {
    let bytes = download_raw(url, token, expected)?;
    scoped(url, token, &bytes)
}
/// The exact bytes of a package this worker may not be new enough to read.
pub fn download_raw(url: &str, token: &str, expected: &str) -> managed::Result<Vec<u8>> {
    let uri = reqwest::Url::parse(url).map_err(|_| download_error("url", None))?;
    if uri.scheme() != "https"
        || uri.host_str().is_none()
        || !uri.username().is_empty()
        || uri.password().is_some()
        || uri.query().is_some()
        || uri.fragment().is_some()
        || !helper::valid_digest(expected)
        || !helper::valid_digest(&format!("sha256:{token}"))
    {
        return Err(download_error("url", None));
    }
    if !url.ends_with("/package") {
        return Err(download_error("url", None));
    }
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|_| download_error("client", None))?;
    eprintln!("ML_EXPD_PACKAGE_DOWNLOAD=START");
    download_with_retries(&client, url, token, expected)
}
/// A package for this worker, of exactly the Attempt it was asked for.
pub fn scoped(url: &str, token: &str, bytes: &[u8]) -> managed::Result<Package> {
    let channel = ml_exp_core::cloud::exchange::Channel {
        url: url
            .strip_suffix("/package")
            .ok_or_else(|| download_error("url", None))?
            .into(),
        token: token.into(),
    };
    let package = decode(bytes).map_err(|_| download_error("decode", Some(200)))?;
    if package.channel != channel {
        return Err(download_error("scope", Some(200)));
    }
    eprintln!("ML_EXPD_PACKAGE_DOWNLOAD=READY bytes={}", bytes.len());
    Ok(package)
}
fn startup(stage: &'static str, error: std::io::Error) -> managed::Error {
    managed::Error::Startup {
        stage,
        errno: error.raw_os_error(),
    }
}
fn same_or_new(
    directory: &Directory,
    name: &OsStr,
    bytes: &[u8],
    stage: &'static str,
) -> managed::Result<()> {
    if directory
        .entry_exists(name)
        .map_err(crate::managed::storage_failure!())?
    {
        let original = directory
            .read_regular(name, MAX_PACKAGE_BYTES as u64)
            .map_err(crate::managed::storage_failure!())?;
        if original != bytes {
            return Err(managed::Error::Binding);
        }
    } else {
        directory
            .publish_new(name, bytes)
            .map_err(|error| startup(stage, error))?;
    }
    directory
        .require_private()
        .map_err(crate::managed::storage_failure!())
}
fn child(directory: &Directory, name: &str) -> managed::Result<Directory> {
    let result = if directory
        .entry_exists(OsStr::new(name))
        .map_err(crate::managed::storage_failure!())?
    {
        directory.child(OsStr::new(name), false)
    } else {
        directory.create_new_child(OsStr::new(name))
    }
    .map_err(crate::managed::storage_failure!())?;
    result
        .require_private()
        .map_err(crate::managed::storage_failure!())?;
    Ok(result)
}
pub fn prepare(package: &Package, runtime: &BTreeMap<String, String>) -> managed::Result<()> {
    package
        .validate()
        .map_err(|_| managed::Error::Configuration)?;
    crate::cloud_storage::prepare(package)?;
    let binding = package.config.binding()?;
    let root = Directory::open(&package.root, false).map_err(crate::managed::storage_failure!())?;
    root.require_private()
        .map_err(crate::managed::storage_failure!())?;
    let project = child(&root, &binding.identity.project)?;
    let run = child(&project, &binding.identity.run_id)?;
    let package_name = format!("{}.package.json", binding.identity.attempt_id);
    same_or_new(
        &run,
        OsStr::new(&package_name),
        &serde_json::to_vec(package).map_err(|_| managed::Error::Configuration)?,
        "package_publication",
    )?;
    same_or_new(
        &run,
        package
            .config
            .manifest_path
            .file_name()
            .ok_or(managed::Error::Configuration)?,
        &package
            .manifest
            .encode()
            .map_err(|_| managed::Error::Manifest)?,
        "manifest_publication",
    )?;
    if run
        .entry_exists(OsStr::new(&binding.identity.attempt_id))
        .map_err(crate::managed::storage_failure!())?
    {
        // Existing-but-incomplete initialization stays uncertain. No repair or
        // overwrite of the original environment, phase or launch identity.
        managed::inspect(&package.config)?;
    } else {
        managed::initialize_with_environment(&package.config, runtime)?;
    }
    let attempt = run
        .child(OsStr::new(&binding.identity.attempt_id), false)
        .map_err(crate::managed::storage_failure!())?;
    child(&attempt, "state")?;
    Ok(())
}
pub fn run(
    package: &Package,
    runtime: &BTreeMap<String, String>,
    cancelled: &AtomicBool,
) -> managed::Result<Observation> {
    prepare(package, runtime)?;
    let original = managed::inspect(&package.config)?;
    if original.phase == managed::Phase::Finished {
        let _ = crate::cloud_exchange::drain(package, cancelled);
        if package.result_upload {
            report_deferred(crate::cloud_artifacts::publish(package));
        }
        return Ok(original);
    }
    let stop = std::sync::atomic::AtomicBool::new(false);
    let wake = (std::sync::Mutex::new(()), std::sync::Condvar::new());
    let result = std::thread::scope(|scope| {
        let stop_ref = &stop;
        let wake_ref = &wake;
        let sender = scope.spawn(move || {
            let Ok(transport) = crate::cloud_exchange::Https::new() else {
                return;
            };
            while !stop_ref.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = crate::cloud_exchange::once(package, &transport);
                let Ok(guard) = wake_ref.0.lock() else {
                    break;
                };
                if stop_ref.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let _ = wake_ref
                    .1
                    .wait_timeout(guard, std::time::Duration::from_secs(5));
            }
        });
        // STARTED cannot run again, so it has no use for code and `managed::run` says so.
        let delivered = if original.phase == managed::Phase::Prepared {
            crate::cloud_code::deliver(package, cancelled)
        } else {
            Ok(())
        };
        let result = delivered
            .and_then(|_| crate::cloud_data::materialize(package, cancelled))
            .and_then(|_| managed::run(&package.config, cancelled));
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Ok(_guard) = wake.0.lock() {
            wake.1.notify_all();
        }
        if sender.join().is_err() {
            return Err(managed::Error::Uncertain);
        }
        result
    });
    // Upload failure never rewrites the training exit status or relaunches it.
    // A later CPU-only sync drains this same durable stream on shared storage.
    let delivered = if result.is_ok() {
        crate::cloud_exchange::drain(package, cancelled).unwrap_or(false)
    } else {
        crate::cloud_exchange::Https::new()
            .and_then(|transport| crate::cloud_exchange::once(package, &transport))
            .is_ok()
    };
    if !delivered {
        eprintln!("ML_EXPD_EXCHANGE=DEFERRED; retained on backend storage");
    }
    if result.is_ok() && package.result_upload {
        report_deferred(crate::cloud_artifacts::publish(package));
    }
    result
}
/// The job's outcome does not depend on this, and the reason is all anyone will have.
fn report_deferred(published: managed::Result<()>) {
    if let Err(error) = published {
        eprintln!("ML_EXPD_RESULTS=DEFERRED; retained on backend storage: {error}");
    }
}
pub fn package_sha(package: &Package) -> managed::Result<String> {
    package
        .validate()
        .map_err(|_| managed::Error::Configuration)?;
    Ok(helper::sha256_bytes(
        &serde_json::to_vec(package).map_err(|_| managed::Error::Configuration)?,
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bootstrap_executes_once_and_preserves_complete_state_across_restarts() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("runs");
        let workdir = temp.path().join("code");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(&workdir).unwrap();
        let storage = ml_exp_core::cloud::launch::StorageLayout {
            mount: temp.path().into(),
            runs: root.clone(),
            datasets: temp.path().join("datasets"),
        };
        let argv=vec!["/bin/sh".into(),"-c".into(),"printf once >> \"$STATE_DIR/count\"; printf '{\"name\":\"loss\",\"value\":1,\"unit\":\"nats/token\",\"step\":1}\\n' > \"$METRICS_PATH\"".into()];
        let package = ml_exp_core::cloud::launch::compile(ml_exp_core::cloud::launch::Input {
            root: root.to_str().unwrap(),
            project: "demo",
            run: "train",
            attempt: "attempt-001",
            workdir: workdir.to_str().unwrap(),
            argv: &argv,
            timeout_seconds: 3,
            worker_sha256: &format!("sha256:{}", "a".repeat(64)),
            storage,
            // Nothing listens here, so delivery defers without leaving this host.
            channel: ml_exp_core::cloud::exchange::Channel {
                url: "https://127.0.0.1:1/api/worker-exchange/demo/train/attempt-001".into(),
                token: "b".repeat(64),
            },
        })
        .unwrap();
        let receipt = run(&package, &BTreeMap::new(), &AtomicBool::new(false)).unwrap();
        assert_eq!(
            receipt.receipt.as_ref().unwrap().training,
            ml_exp_core::managed::evidence::Training::Completed
        );
        assert_eq!(
            run(&package, &BTreeMap::new(), &AtomicBool::new(false)).unwrap(),
            receipt
        );
        assert_eq!(
            std::fs::read(package.config.attempt_root.join("state/count")).unwrap(),
            b"once"
        );
        assert!(package.config.output_path().join("metrics.jsonl").is_file());
        let mut changed = package.clone();
        changed.manifest.argv = vec!["/bin/false".into()];
        assert!(prepare(&changed, &BTreeMap::new()).is_err());
        let ledger = package.config.attempt_root.join(".managed-launch.lock");
        let original = std::fs::read(&ledger).unwrap();
        let first_end = original.iter().position(|b| *b == b'\n').unwrap();
        let rest = &original[first_end + 1..];
        let second_end = rest.iter().position(|b| *b == b'\n').unwrap() + first_end + 2;
        std::fs::write(&ledger, &original[..second_end]).unwrap();
        assert!(run(&package, &BTreeMap::new(), &AtomicBool::new(false)).is_err());
        assert_eq!(
            std::fs::read(package.config.attempt_root.join("state/count")).unwrap(),
            b"once"
        );
    }
}

#[cfg(test)]
mod download_tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        time::Duration,
    };

    fn server(responses: Vec<(u16, Vec<u8>)>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/package", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for (status, body) in responses {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = Vec::new();
                while !request.windows(4).any(|v| v == b"\r\n\r\n") {
                    let mut chunk = [0u8; 1024];
                    let read = socket.read(&mut chunk).unwrap();
                    assert!(read > 0);
                    request.extend_from_slice(&chunk[..read]);
                }
                requests.push(String::from_utf8(request).unwrap());
                write!(
                    socket,
                    "HTTP/1.1 {status} fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                socket.write_all(&body).unwrap();
            }
            requests
        });
        (url, thread)
    }
    fn client() -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap()
    }
    fn large_package() -> Package {
        use ml_exp_core::{
            artifacts::FileReceipt,
            datasets::{self, Definition, Mount, Receipt},
        };
        let channel = ml_exp_core::cloud::exchange::Channel {
            url: "https://api.example/api/worker-exchange/demo/train/attempt-001".into(),
            token: "b".repeat(64),
        };
        let mut package = ml_exp_core::cloud::launch::compile(ml_exp_core::cloud::launch::Input {
            root: "/mnt/runs",
            project: "demo",
            run: "train",
            attempt: "attempt-001",
            workdir: "/workspace",
            argv: &["/bin/true".into()],
            timeout_seconds: 60,
            worker_sha256: &format!("sha256:{}", "a".repeat(64)),
            storage: ml_exp_core::cloud::launch::StorageLayout {
                mount: "/mnt".into(),
                runs: "/mnt/runs".into(),
                datasets: "/mnt/data".into(),
            },
            channel: channel.clone(),
        })
        .unwrap();
        let target = helper::Target {
            backend_id: "pai".into(),
            profile_sha256: format!("sha256:{}", "c".repeat(64)),
            storage_scope: format!("sha256:{}", "d".repeat(64)),
        };
        let definition = Definition {
            sha256: "e".repeat(64),
            archive_bytes: 1024 * 1024,
            files: (0..304)
                .map(|i| FileReceipt {
                    path: format!("dataset/full-inventory/part-{i:04}.bin"),
                    bytes: 128,
                    sha256: "f".repeat(64),
                })
                .collect(),
        };
        package
            .attach(
                "/mnt/data",
                vec![Mount {
                    name: "data".into(),
                    receipt: Receipt {
                        contract: datasets::CONTRACT.into(),
                        project: "demo".into(),
                        target: target.clone(),
                        asset_id: definition.asset_id(),
                        root: format!("/mnt/data/demo/{}/tree", definition.asset_id()),
                        definition: definition.clone(),
                    },
                }],
                None,
            )
            .unwrap();
        package.imports.push(datasets::import::Config {
            root: "/mnt/data".into(),
            project: "demo".into(),
            target,
            definition,
            source: datasets::import::Source::ExchangeParts { channel },
            timeout_seconds: 3600,
        });
        package.validate().unwrap();
        package
    }
    #[test]
    fn full_304_file_package_downloads_and_decodes_without_entering_argv_or_starting_training() {
        let package = large_package();
        let bytes = serde_json::to_vec(&serde_json::to_value(&package).unwrap()).unwrap();
        assert!(bytes.len() > 64 * 1024);
        let command = package.download_command("/worker").unwrap();
        assert!(command.join(" ").len() < 1024);
        let expected = helper::sha256_bytes(&bytes);
        assert_eq!(command[7], expected);
        let (url, server) = server(vec![(200, bytes.clone())]);
        let got = download_bytes(
            &client(),
            &url,
            &package.channel.token,
            &expected,
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(got, bytes);
        let decoded = decode(&got).unwrap();
        assert_eq!(
            decoded.config.datasets[0].receipt.definition.files.len(),
            304
        );
        assert_eq!(decoded.imports.len(), 1);
        assert!(decoded == package);
        assert_eq!(server.join().unwrap().len(), 1);
    }
    #[test]
    fn request_status_and_checksum_failures_are_actionable_without_exposing_credentials() {
        for (status, expected, operation, http_status) in [
            (503, helper::sha256_bytes(b"body"), "request", Some(503)),
            (200, helper::sha256_bytes(b"different"), "sha256", Some(200)),
        ] {
            let (url, server) = server(vec![(status, b"body".to_vec())]);
            let error = download_bytes(
                &client(),
                &url,
                "fixture-private-token",
                &expected,
                Duration::from_secs(5),
            )
            .unwrap_err();
            assert!(
                matches!(&error, managed::Error::PackageDownload { operation: op, http_status: code } if *op == operation && *code == http_status)
            );
            assert!(!error.to_string().contains("fixture-private-token"));
            assert!(!error.to_string().contains(&url));
            assert_eq!(server.join().unwrap().len(), 1);
        }
    }
    #[test]
    fn transient_503_retries_only_the_same_metadata_get() {
        let body = b"same frozen metadata".to_vec();
        let (url, server) = server(vec![(503, Vec::new()), (200, body.clone())]);
        assert_eq!(
            download_with_retries(
                &client(),
                &url,
                "fixture-token",
                &helper::sha256_bytes(&body)
            )
            .unwrap(),
            body
        );
        let requests = server.join().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(
            requests
                .iter()
                .all(|r| r.starts_with("GET /package HTTP/1.1\r\n"))
        );
        assert_eq!(requests[0], requests[1]);
    }
}
