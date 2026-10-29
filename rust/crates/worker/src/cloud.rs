//! Cloud bootstrap around the existing sealed worker. It never fetches code,
//! installs packages, picks a checkpoint or replays an uncertain process.
use crate::managed;
use ml_exp_core::{cloud::launch::Package, managed::evidence::Observation, slurm::helper};
use ml_exp_io::directory::Directory;
use std::{collections::BTreeMap, ffi::OsStr, path::Path, sync::atomic::AtomicBool};
pub fn decode(bytes: &[u8]) -> managed::Result<Package> {
    if bytes.len() > 64 * 1024 {
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
fn same_or_new(directory: &Directory, name: &OsStr, bytes: &[u8]) -> managed::Result<()> {
    if directory
        .entry_exists(name)
        .map_err(|_| managed::Error::Storage)?
    {
        let original = directory
            .read_regular(name, 256 * 1024)
            .map_err(|_| managed::Error::Storage)?;
        if original != bytes {
            return Err(managed::Error::Binding);
        }
    } else {
        directory
            .publish_new(name, bytes)
            .map_err(|_| managed::Error::Uncertain)?;
    }
    directory
        .verify_private_parent()
        .map_err(|_| managed::Error::Storage)
}
fn child(directory: &Directory, name: &str) -> managed::Result<Directory> {
    let result = if directory
        .entry_exists(OsStr::new(name))
        .map_err(|_| managed::Error::Storage)?
    {
        directory.child(OsStr::new(name), false)
    } else {
        directory.create_new_child(OsStr::new(name))
    }
    .map_err(|_| managed::Error::Storage)?;
    result
        .verify_private_parent()
        .map_err(|_| managed::Error::Storage)?;
    Ok(result)
}
/// Verify the native executable against the trusted published-image receipt.
pub fn verify_binary(path: &Path, expected: &str) -> managed::Result<()> {
    let sha = ml_exp_io::sha256_file(path).map_err(|_| managed::Error::Binding)?;
    if format!("sha256:{sha}") != expected {
        return Err(managed::Error::Binding);
    }
    Ok(())
}
pub fn prepare(package: &Package, runtime: &BTreeMap<String, String>) -> managed::Result<()> {
    package
        .validate()
        .map_err(|_| managed::Error::Configuration)?;
    crate::cloud_storage::prepare(package)?;
    let binding = package.config.binding()?;
    let root = Directory::open(&package.root, false).map_err(|_| managed::Error::Storage)?;
    root.verify_private_parent()
        .map_err(|_| managed::Error::Storage)?;
    let project = child(&root, &binding.identity.project)?;
    let run = child(&project, &binding.identity.run_id)?;
    let package_name = format!("{}.package.json", binding.identity.attempt_id);
    same_or_new(
        &run,
        OsStr::new(&package_name),
        &serde_json::to_vec(package).map_err(|_| managed::Error::Configuration)?,
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
    )?;
    if run
        .entry_exists(OsStr::new(&binding.identity.attempt_id))
        .map_err(|_| managed::Error::Storage)?
    {
        // Existing-but-incomplete initialization stays uncertain. No repair or
        // overwrite of the original environment, phase or launch identity.
        managed::inspect(&package.config)?;
    } else {
        managed::initialize_with_environment(&package.config, runtime)?;
    }
    let attempt = run
        .child(OsStr::new(&binding.identity.attempt_id), false)
        .map_err(|_| managed::Error::Storage)?;
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
        if package.channel.is_some() {
            let _ = crate::cloud_exchange::drain(package, cancelled);
        }
        if package.result_upload && crate::cloud_artifacts::publish(package).is_err() {
            eprintln!("ML_EXPD_RESULTS=DEFERRED; retained on backend storage");
        }
        return Ok(original);
    }
    if package.channel.is_none() {
        crate::cloud_data::materialize(package, cancelled)?;
        return managed::run(&package.config, cancelled);
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
        let result = crate::cloud_data::materialize(package, cancelled)
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
    if result.is_ok() && package.result_upload && crate::cloud_artifacts::publish(package).is_err()
    {
        eprintln!("ML_EXPD_RESULTS=DEFERRED; retained on backend storage");
    }
    result
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
