//! Sealed local managed execution for already-staged code and inputs.
//!
//! Trusted private configuration fixes the launch digest, identity and paths.
//! There is no manifest fetch, remote command RPC, provider call or upload in
//! this slice. Unsupported managed I/O is refused before the durable start.
use fs2::FileExt;
pub use ml_exp_core::managed::Config;
use ml_exp_core::{launch, managed::absolute};
use ml_exp_io::directory::Directory;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

const LOCK: &str = ".managed-launch.lock";
const MANIFEST: &str = "launch-manifest.json";
const RECEIPT: &str = "managed-receipt.json";
const LEDGER_BYTES: u64 = 16 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid private managed launch configuration")]
    Configuration,
    #[error("sealed launch identity, checksum or encoding is invalid")]
    Manifest,
    #[error(
        "managed input, checkpoint, record or publication I/O has not been ported; no process started"
    )]
    Unsupported,
    #[error("managed launch paths or existing state are unavailable or changed")]
    Storage,
    #[error("managed launch identity differs from its frozen configuration")]
    Binding,
    #[error("managed launch is already being observed or executed")]
    Busy,
    #[error("managed launch may already have started; inspect without replay")]
    Uncertain,
    #[error("managed launch was cancelled before execution")]
    Cancelled,
}
pub type Result<T> = std::result::Result<T, Error>;

impl From<ml_exp_core::managed::Error> for Error {
    fn from(_: ml_exp_core::managed::Error) -> Self {
        Self::Configuration
    }
}
/// The shell reads private bytes; identity and configuration decisions are pure.
pub fn load_config(path: &Path, expected_sha256: Option<&str>) -> Result<Config> {
    let bytes = private_file(path, 1024 * 1024).map_err(|_| Error::Configuration)?;
    if expected_sha256.is_some_and(|value| !hex(value) || sha(&bytes) != value) {
        return Err(Error::Binding);
    }
    let config: Config = serde_json::from_value(serde_json::Value::Object(
        ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Configuration)?,
    ))
    .map_err(|_| Error::Configuration)?;
    config.validate()?;
    Ok(config)
}

pub use ml_exp_core::managed::evidence::*;
#[derive(PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    contract: String,
    phase: Phase,
    binding_sha256: String,
    receipt_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_environment_sha256: Option<String>,
}
impl From<crate::Execution> for ProcessEvidence {
    fn from(value: crate::Execution) -> Self {
        Self {
            exit_code: value.exit_code,
            termination_signal: value.termination_signal,
            timed_out: value.timed_out,
            cancelled: value.cancelled,
            error: value.error.map(|error| Diagnostic {
                stage: error.stage.into(),
                code: error.code.into(),
            }),
            stdout_sha256: value.stdout_sha256,
            stderr_sha256: value.stderr_sha256,
        }
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn private_bytes(root: &Directory, name: &OsStr, maximum: u64) -> Result<Vec<u8>> {
    root.verify_private_parent().map_err(|_| Error::Storage)?;
    let mut file = root.open_regular_read(name).map_err(|_| Error::Storage)?;
    let metadata = file.metadata().map_err(|_| Error::Storage)?;
    if metadata.mode() & 0o077 != 0 || metadata.len() > maximum {
        return Err(Error::Storage);
    }
    let mut bytes = Vec::new();
    (&mut file)
        .take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Error::Storage)?;
    root.validate_regular_entry(name, &file)
        .map_err(|_| Error::Storage)?;
    if bytes.len() as u64 != metadata.len() {
        return Err(Error::Storage);
    }
    Ok(bytes)
}
fn private_file(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    if !absolute(path) {
        return Err(Error::Configuration);
    }
    let root = Directory::open(path.parent().ok_or(Error::Configuration)?, false)
        .map_err(|_| Error::Storage)?;
    private_bytes(
        &root,
        path.file_name().ok_or(Error::Configuration)?,
        maximum,
    )
}
fn verified(
    config: &Config,
    bytes: &[u8],
) -> Result<(launch::Binding, launch::Manifest, BTreeMap<String, String>)> {
    config.validate()?;
    let target = config.binding()?;
    let manifest = launch::verify_binding(bytes, &target).map_err(|_| Error::Manifest)?;
    let values = ml_exp_core::managed::execution_environment(config, &manifest).map_err(
        |error| match error {
            ml_exp_core::managed::AdmissionError::Configuration => Error::Configuration,
            ml_exp_core::managed::AdmissionError::Binding => Error::Binding,
            ml_exp_core::managed::AdmissionError::Unsupported => Error::Unsupported,
        },
    )?;
    Ok((target, manifest, values))
}

struct Held {
    root: Directory,
    file: File,
    binding: String,
}
impl Drop for Held {
    fn drop(&mut self) {
        // A concurrent Command fork can briefly inherit this open description
        // before CLOEXEC closes it. Closing only our FD would retain the lock
        // in that child; explicit unlock releases the owner's completed guard.
        let _ = FileExt::unlock(&self.file);
    }
}
impl Held {
    fn open(config: &Config, exclusive: bool) -> Result<Self> {
        config.validate()?;
        let root = Directory::open(&config.attempt_root, false).map_err(|_| Error::Storage)?;
        let file = root
            .open_existing_lock_rw(OsStr::new(LOCK))
            .map_err(|_| Error::Storage)?;
        let result = if exclusive {
            FileExt::try_lock_exclusive(&file)
        } else {
            FileExt::try_lock_shared(&file)
        };
        result.map_err(|e| {
            if e.kind() == std::io::ErrorKind::WouldBlock {
                Error::Busy
            } else {
                Error::Storage
            }
        })?;
        let held = Self {
            root,
            file,
            binding: config.digest()?,
        };
        held.validate()?;
        Ok(held)
    }
    fn validate(&self) -> Result<()> {
        self.root
            .verify_private_parent()
            .map_err(|_| Error::Storage)?;
        self.root
            .validate_regular_entry(OsStr::new(LOCK), &self.file)
            .map_err(|_| Error::Storage)
    }
    fn state(&mut self) -> Result<Record> {
        self.validate()?;
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|_| Error::Storage)?;
        let mut bytes = Vec::new();
        (&mut self.file)
            .take(LEDGER_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| Error::Storage)?;
        self.validate()?;
        if bytes.len() as u64 > LEDGER_BYTES || !bytes.ends_with(b"\n") {
            return Err(Error::Uncertain);
        }
        let lines = bytes[..bytes.len() - 1]
            .split(|c| *c == b'\n')
            .collect::<Vec<_>>();
        if !(1..=3).contains(&lines.len()) {
            return Err(Error::Uncertain);
        }
        let mut last: Option<Record> = None;
        for (line, phase) in
            lines
                .into_iter()
                .zip([Phase::Prepared, Phase::Started, Phase::Finished])
        {
            let record: Record = serde_json::from_slice(line).map_err(|_| Error::Uncertain)?;
            if record.contract != CONTRACT || record.binding_sha256 != self.binding {
                return Err(Error::Binding);
            }
            if last.as_ref().is_some_and(|previous| {
                previous.runtime_environment_sha256 != record.runtime_environment_sha256
            }) || record
                .runtime_environment_sha256
                .as_ref()
                .is_some_and(|value| !hex(value))
                || record.phase != phase
                || (phase == Phase::Finished)
                    != record.receipt_sha256.as_ref().is_some_and(|v| hex(v))
                || phase != Phase::Finished && record.receipt_sha256.is_some()
            {
                return Err(Error::Uncertain);
            }
            last = Some(record);
        }
        last.ok_or(Error::Uncertain)
    }
    fn append(
        &self,
        phase: Phase,
        receipt: Option<String>,
        runtime_sha256: Option<String>,
    ) -> Result<()> {
        self.validate()?;
        let record = Record {
            contract: CONTRACT.into(),
            phase,
            binding_sha256: self.binding.clone(),
            receipt_sha256: receipt,
            runtime_environment_sha256: runtime_sha256,
        };
        let mut bytes = serde_json::to_vec(&record).map_err(|_| Error::Storage)?;
        bytes.push(b'\n');
        self.root
            .append_regular(OsStr::new(LOCK), &bytes, LEDGER_BYTES)
            .map_err(|_| Error::Storage)?;
        self.validate()
    }
    fn runtime(&mut self, config: &Config) -> Result<BTreeMap<String, String>> {
        let record = self.state()?;
        if config.runtime_environment.is_empty() {
            if record.runtime_environment_sha256.is_some() {
                return Err(Error::Binding);
            }
            return Ok(BTreeMap::new());
        }
        let bytes = private_bytes(
            &self.root,
            OsStr::new("runtime-environment.json"),
            128 * 1024,
        )?;
        if record.runtime_environment_sha256.as_deref() != Some(sha(&bytes).as_str()) {
            return Err(Error::Binding);
        }
        let values: BTreeMap<String, String> = serde_json::from_value(serde_json::Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Binding)?,
        ))
        .map_err(|_| Error::Binding)?;
        config.validate_runtime(&values)?;
        Ok(values)
    }
    fn observe(&mut self, config: &Config) -> Result<Observation> {
        let record = self.state()?;
        self.runtime(config)?;
        let bytes = private_bytes(&self.root, OsStr::new(MANIFEST), launch::MAX_BYTES as u64)?;
        let (target, _, _) = verified(config, &bytes)?;
        let receipt = if record.phase == Phase::Finished {
            let bytes = private_bytes(&self.root, OsStr::new(RECEIPT), 64 * 1024)?;
            if record.receipt_sha256.as_deref() != Some(sha(&bytes).as_str()) {
                return Err(Error::Uncertain);
            }
            let value: Receipt = serde_json::from_slice(&bytes).map_err(|_| Error::Uncertain)?;
            if value.contract != CONTRACT
                || value.identity != target.identity
                || value.manifest_sha256 != target.sha256
                || value.download_available
                || value.runtime_environment_sha256 != record.runtime_environment_sha256
                || value.training != training(&value.process)
                || value
                    .process
                    .stdout_sha256
                    .as_ref()
                    .is_some_and(|v| !hex(v))
                || value
                    .process
                    .stderr_sha256
                    .as_ref()
                    .is_some_and(|v| !hex(v))
            {
                return Err(Error::Uncertain);
            }
            Some(value)
        } else {
            None
        };
        self.validate()?;
        Ok(Observation {
            contract: CONTRACT.into(),
            identity: target.identity,
            manifest_sha256: target.sha256,
            phase: record.phase,
            safe_to_retry: false,
            receipt,
            runtime_environment_sha256: record.runtime_environment_sha256,
        })
    }
}

/// Explicit bootstrap of one fresh, dedicated Attempt root. Existing paths are
/// never adopted, reset or repaired. Its frozen OUTPUT_DIR fixes this location.
pub fn initialize(config: &Config) -> Result<Observation> {
    initialize_with_environment(config, &BTreeMap::new())
}

/// The caller selects the approved runtime variables. Their values are sealed
/// once, never inherited again during inspect/restart or silently reinterpreted.
pub fn initialize_with_environment(
    config: &Config,
    runtime: &BTreeMap<String, String>,
) -> Result<Observation> {
    config.validate()?;
    config.validate_runtime(runtime)?;
    let bytes = private_file(&config.manifest_path, launch::MAX_BYTES as u64)?;
    verified(config, &bytes)?;
    let workdir = Directory::open(&config.workdir, false).map_err(|_| Error::Storage)?;
    workdir
        .verify_private_parent()
        .map_err(|_| Error::Storage)?;
    let parent = Directory::open(
        config.attempt_root.parent().ok_or(Error::Configuration)?,
        false,
    )
    .map_err(|_| Error::Storage)?;
    let root = parent
        .create_new_child(
            config
                .attempt_root
                .file_name()
                .ok_or(Error::Configuration)?,
        )
        .map_err(|_| Error::Storage)?;
    root.publish_new(OsStr::new(MANIFEST), &bytes)
        .map_err(|_| Error::Storage)?;
    let file = root
        .open_lock(OsStr::new(LOCK))
        .map_err(|_| Error::Storage)?;
    FileExt::try_lock_exclusive(&file).map_err(|_| Error::Busy)?;
    let mut held = Held {
        root,
        file,
        binding: config.digest()?,
    };
    let runtime_sha256 = if config.runtime_environment.is_empty() {
        None
    } else {
        let bytes = serde_json::to_vec(runtime).map_err(|_| Error::Configuration)?;
        held.root
            .publish_new(OsStr::new("runtime-environment.json"), &bytes)
            .map_err(|_| Error::Storage)?;
        Some(sha(&bytes))
    };
    held.append(Phase::Prepared, None, runtime_sha256)?;
    workdir
        .verify_private_parent()
        .map_err(|_| Error::Storage)?;
    held.observe(config)
}

/// Observe existing durable state only. STARTED is possible/unknown execution,
/// never evidence that the process is currently alive or safe to replay.
pub fn inspect(config: &Config) -> Result<Observation> {
    Held::open(config, false)?.observe(config)
}

/// Read a stable point-in-time ledger while execution owns its exclusive lock.
/// The descriptor is read-only and no lock/phase is mutated. STARTED proves
/// possible execution only; torn or changing records yield uncertainty.
pub fn sample(config: &Config) -> Result<Observation> {
    config.validate()?;
    let root = Directory::open(&config.attempt_root, false).map_err(|_| Error::Storage)?;
    let file = root
        .open_regular_read(OsStr::new(LOCK))
        .map_err(|_| Error::Storage)?;
    let mut held = Held {
        root,
        file,
        binding: config.digest()?,
    };
    let before = held.state()?;
    let value = held.observe(config)?;
    if held.state()? != before {
        return Err(Error::Uncertain);
    }
    Ok(value)
}

/// Start at most once for this frozen Attempt root. A lost reply can be
/// inspected; STARTED, damaged state, or missing locks never trigger a replay.
pub fn run(config: &Config, cancelled: &AtomicBool) -> Result<Observation> {
    let mut held = Held::open(config, true)?;
    let initial = held.observe(config)?;
    if initial.phase == Phase::Finished {
        return Ok(initial);
    }
    if initial.phase != Phase::Prepared {
        return Err(Error::Uncertain);
    }
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    let output_path = config.output_path();
    let output_parent = Directory::open(output_path.parent().ok_or(Error::Configuration)?, false)
        .map_err(|_| Error::Storage)?;
    output_parent
        .verify_private_parent()
        .map_err(|_| Error::Storage)?;
    if output_parent
        .entry_exists(output_path.file_name().ok_or(Error::Configuration)?)
        .map_err(|_| Error::Storage)?
        || held
            .root
            .entry_exists(OsStr::new(RECEIPT))
            .map_err(|_| Error::Storage)?
    {
        return Err(Error::Uncertain);
    }
    let workdir = Directory::open(&config.workdir, false).map_err(|_| Error::Storage)?;
    workdir
        .verify_private_parent()
        .map_err(|_| Error::Storage)?;
    let bytes = private_bytes(&held.root, OsStr::new(MANIFEST), launch::MAX_BYTES as u64)?;
    let (target, manifest, mut environment) = verified(config, &bytes)?;
    environment.extend(held.runtime(config)?);
    if !ml_exp_core::managed::environment(&environment) {
        return Err(Error::Configuration);
    }
    // This fsync completes before any spawn attempt. Any later error leaves a
    // non-replayable STARTED record, including setup errors with no known child.
    held.append(
        Phase::Started,
        None,
        initial.runtime_environment_sha256.clone(),
    )?;
    output_parent
        .verify_private_parent()
        .map_err(|_| Error::Uncertain)?;
    let execution = crate::process::execute_explicit(
        &config.workdir,
        &output_path,
        &manifest.argv,
        Duration::from_secs(manifest.timeout_seconds),
        cancelled,
        &environment,
    )
    .map_err(|_| Error::Uncertain)?;
    held.validate().map_err(|_| Error::Uncertain)?;
    output_parent
        .verify_private_parent()
        .map_err(|_| Error::Uncertain)?;
    Directory::open(&output_path, false)
        .and_then(|outputs| outputs.verify_private_parent())
        .map_err(|_| Error::Uncertain)?;
    workdir
        .verify_private_parent()
        .map_err(|_| Error::Uncertain)?;
    let process = ProcessEvidence::from(execution);
    let receipt = Receipt {
        contract: CONTRACT.into(),
        identity: target.identity,
        manifest_sha256: target.sha256,
        training: training(&process),
        process,
        publication: Publication::NotImplemented,
        download_available: false,
        runtime_environment_sha256: initial.runtime_environment_sha256.clone(),
    };
    let bytes = serde_json::to_vec(&receipt).map_err(|_| Error::Uncertain)?;
    held.root
        .publish_new(OsStr::new(RECEIPT), &bytes)
        .map_err(|_| Error::Uncertain)?;
    held.append(
        Phase::Finished,
        Some(sha(&bytes)),
        initial.runtime_environment_sha256,
    )
    .map_err(|_| Error::Uncertain)?;
    held.observe(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_guard_unlocks_even_when_an_inherited_description_remains_open() {
        let temp = tempfile::tempdir().unwrap();
        let root = Directory::open(temp.path(), false).unwrap();
        let file = root.open_lock(OsStr::new(LOCK)).unwrap();
        FileExt::try_lock_exclusive(&file).unwrap();
        // try_clone shares the kernel open-file-description, as fork does.
        let inherited = file.try_clone().unwrap();
        let held = Held {
            root: root.clone(),
            file,
            binding: "synthetic".into(),
        };
        drop(held);
        let next = root.open_existing_lock_rw(OsStr::new(LOCK)).unwrap();
        FileExt::try_lock_exclusive(&next).unwrap();
        FileExt::unlock(&next).unwrap();
        drop(inherited);
    }
}
