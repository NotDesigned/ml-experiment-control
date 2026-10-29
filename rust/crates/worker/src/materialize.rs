//! Gateway-only OCI conversion. No API-host image layers, no scheduler calls.
use ml_exp_core::{cloud::materialize::Request, first_submission::native::Runtime, slurm::managed};
use ml_exp_io::{
    directory::Directory,
    process::{self, Capture, Environment, Input},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    path::PathBuf,
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub root: PathBuf,
    pub executor: String,
    pub apptainer: PathBuf,
    pub registry_auth_file: Option<PathBuf>,
    pub timeout_seconds: u64,
    pub min_free_bytes: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    request: Request,
    runtime: Runtime,
}
fn invalid() -> std::io::Error {
    std::io::Error::other("native OCI conversion configuration, receipt or filesystem is invalid")
}
impl Config {
    fn validate(&self) -> std::io::Result<()> {
        if !ml_exp_core::managed::absolute(&self.root)
            || !ml_exp_core::managed::absolute(&self.apptainer)
            || !(60..=7200).contains(&self.timeout_seconds)
            || self.min_free_bytes == 0
        {
            return Err(invalid());
        }
        ml_exp_core::identity::validate_id(&self.executor)
            .map(|_| ())
            .map_err(|_| invalid())
    }
}
pub fn run(config: Config, mut request: Request) -> std::io::Result<serde_json::Value> {
    config.validate()?;
    request.validate().map_err(|_| invalid())?;
    if config.executor != request.executor {
        return Err(invalid());
    }
    let root = Directory::open(&config.root, false)?;
    root.verify_private_parent()?;
    let lock = root.open_lock(OsStr::new(".materializer.lock"))?;
    fs2::FileExt::try_lock_exclusive(&lock)?;
    let name = OsStr::new(&request.runtime.runtime_id);
    let directory = if root.entry_exists(name)? {
        root.child(name, false)?
    } else {
        if request.operation == "get" {
            return Err(invalid());
        }
        root.create_new_child(name)?
    };
    let build = request.operation == "build";
    request.operation = "get".into();
    let encoded = serde_json::to_vec(&request)?;
    if directory.entry_exists(OsStr::new("intent.json"))? {
        if directory.read_regular(OsStr::new("intent.json"), 65536)? != encoded {
            return Err(invalid());
        }
    } else {
        if !build || !directory.entry_names_limited(1)?.is_empty() {
            return Err(invalid());
        }
        directory.publish_new(OsStr::new("intent.json"), &encoded)?;
    }
    let image = directory.path().join("image.sif");
    if directory.entry_exists(OsStr::new("receipt.json"))? {
        let receipt: Receipt =
            serde_json::from_slice(&directory.read_regular(OsStr::new("receipt.json"), 65536)?)?;
        let mut file = directory.open_regular_read(OsStr::new("image.sif"))?;
        let (sha, _) = ml_exp_io::artifacts::hash(&mut file)?;
        if receipt.request != request
            || receipt.runtime.sif_sha256 != format!("sha256:{sha}")
            || receipt.runtime.sif_path != image.to_str().ok_or_else(invalid)?
        {
            return Err(invalid());
        }
        directory.validate_regular_entry(OsStr::new("image.sif"), &file)?;
        return Ok(serde_json::json!({"runtime":receipt.runtime}));
    }
    if !build {
        return Err(invalid());
    }
    // Only this operation's unpublished temporary file can be replaced.
    if directory.entry_exists(OsStr::new("image.partial.sif"))? {
        let file = directory.open_regular_read(OsStr::new("image.partial.sif"))?;
        directory.remove_staged_regular(OsStr::new("image.partial.sif"), &file)?;
    }
    if directory.entry_exists(OsStr::new("image.sif"))? {
        return Err(invalid());
    }
    let capacity = directory.capacity()?;
    if capacity.insufficient(config.min_free_bytes, 1024) {
        return Err(std::io::Error::other("BUILD_STORAGE_INSUFFICIENT"));
    }
    let mut environment = BTreeMap::from([
        ("PATH".into(), "/usr/bin:/bin".into()),
        (
            "HOME".into(),
            directory.path().to_string_lossy().into_owned(),
        ),
        (
            "APPTAINER_TMPDIR".into(),
            directory.path().to_string_lossy().into_owned(),
        ),
    ]);
    if let Some(path) = config.registry_auth_file {
        let value = ml_exp_io::read_private(&path, 1024 * 1024, true)?;
        let _: serde_json::Value = serde_json::from_slice(&value)?;
        environment.insert(
            "APPTAINER_AUTHFILE".into(),
            path.to_string_lossy().into_owned(),
        );
    }
    let argv = vec![
        config.apptainer.to_str().ok_or_else(invalid)?.to_owned(),
        "build".into(),
        "--disable-cache".into(),
        directory
            .path()
            .join("image.partial.sif")
            .to_str()
            .ok_or_else(invalid)?
            .to_owned(),
        format!("docker://{}", request.runtime.image),
    ];
    let result = process::execute(process::Request {
        argv: &argv,
        cwd: &directory.path(),
        environment: Environment::Replace(&environment),
        stdin: Input::Null,
        stdout: Capture::Memory { limit: 65536 },
        stderr: Capture::Memory { limit: 65536 },
        deadline: Instant::now() + Duration::from_secs(config.timeout_seconds),
        termination_grace: Duration::from_secs(3),
        drain_grace: Duration::from_secs(3),
        cancelled: &AtomicBool::new(false),
    })?;
    if result.exit_code != Some(0)
        || result.timed_out
        || result.cancelled
        || result.failure.is_some()
    {
        return Err(std::io::Error::other("OCI_TO_SIF_FAILED"));
    }
    let file = directory.open_regular_read(OsStr::new("image.partial.sif"))?;
    let (sha, _) = ml_exp_io::artifacts::hash(&mut &file)?;
    directory.validate_regular_entry(OsStr::new("image.partial.sif"), &file)?;
    // The destination is new, operation-owned and held under the exclusive
    // gateway lock. Atomic publication keeps unfinished SIFs out of receipts.
    directory.publish_regular_as(
        OsStr::new("image.partial.sif"),
        OsStr::new("image.sif"),
        &file,
    )?;
    let runtime = Runtime {
        runtime_id: request.runtime.runtime_id.clone(),
        source_id: request.runtime.source_id.clone(),
        image_id: request
            .runtime
            .image
            .rsplit_once('@')
            .ok_or_else(invalid)?
            .1
            .into(),
        sif_path: image.to_str().ok_or_else(invalid)?.into(),
        sif_sha256: format!("sha256:{sha}"),
        oci_image: request.runtime.image.clone(),
        source_dir: Some(request.runtime.source_dir.clone()),
        worker: Some(managed::Runtime {
            program: request.runtime.worker_path.clone(),
            base_environment: BTreeMap::new(),
            runtime_environment: vec!["CUDA_VISIBLE_DEVICES".into(), "LD_LIBRARY_PATH".into()],
        }),
    };
    runtime
        .worker
        .as_ref()
        .ok_or_else(invalid)?
        .validate()
        .map_err(|_| invalid())?;
    directory.seal_regular(OsStr::new("image.sif"), &file, false)?;
    let receipt = Receipt { request, runtime };
    directory.publish_new(OsStr::new("receipt.json"), &serde_json::to_vec(&receipt)?)?;
    Ok(serde_json::json!({"runtime":receipt.runtime}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt};
    #[test]
    fn conversion_reuses_exact_receipt_and_rejects_tampering_or_low_capacity() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("images");
        fs::create_dir(&root).unwrap();
        let binary = temp.path().join("apptainer");
        fs::write(
            &binary,
            b"#!/bin/sh\nprintf 'native fake SIF' > \"$3\"\nprintf x >> calls\n",
        )
        .unwrap();
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
        let config = |min_free_bytes| Config {
            root: root.clone(),
            executor: "slurm-test".into(),
            apptainer: binary.clone(),
            registry_auth_file: None,
            timeout_seconds: 60,
            min_free_bytes,
        };
        let request = |operation: &str, digit: &str| Request {
            operation: operation.into(),
            executor: "slurm-test".into(),
            runtime: ml_exp_core::cloud::run::Runtime {
                runtime_id: format!("runtime.{}", digit.repeat(64)),
                source_id: format!("source.{}", "a".repeat(64)),
                image: format!("registry.example/code@sha256:{}", "b".repeat(64)),
                source_dir: "/code".into(),
                worker_path: "/bin/ml-exp-worker-rs".into(),
                worker_sha256: format!("sha256:{}", "c".repeat(64)),
                worker_features: Default::default(),
            },
        };
        assert!(run(config(1), request("get", "1")).is_err());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1); // exclusive lock only
        let first = run(config(1), request("build", "1")).unwrap();
        assert_eq!(first["runtime"]["source_dir"], "/code");
        assert_eq!(run(config(1), request("get", "1")).unwrap(), first);
        assert_eq!(run(config(1), request("build", "1")).unwrap(), first);
        let directory = root.join(format!("runtime.{}", "1".repeat(64)));
        assert_eq!(fs::read(directory.join("calls")).unwrap(), b"x");
        let image = directory.join("image.sif");
        fs::set_permissions(&image, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&image, b"corrupt").unwrap();
        assert!(run(config(1), request("get", "1")).is_err());
        assert!(run(config(1), request("build", "1")).is_err());
        assert_eq!(fs::read(&image).unwrap(), b"corrupt"); // no destructive repair
        let error = run(config(u64::MAX), request("build", "2")).unwrap_err();
        assert!(error.to_string().contains("BUILD_STORAGE_INSUFFICIENT"));
        assert!(
            !root
                .join(format!("runtime.{}/calls", "2".repeat(64)))
                .exists()
        );
        assert!(run(config(1), request("get", "2")).is_err());
    }
}
