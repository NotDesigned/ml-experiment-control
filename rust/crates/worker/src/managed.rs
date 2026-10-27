//! Sealed local managed execution for already-staged code and inputs.
//!
//! Trusted private configuration fixes the launch digest, identity and paths.
//! There is no manifest fetch, remote command RPC, provider call or upload in
//! this slice. Unsupported managed I/O is refused before the durable start.
use fs2::FileExt;
use ml_exp_core::{canonical, launch};
use ml_exp_io::directory::Directory;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

pub const CONTRACT: &str = "rust-managed-launch.v1";
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

// Environment and paths are private configuration, never Debug/diagnostic text.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub manifest_path: PathBuf,
    pub endpoint: String,
    pub attempt_root: PathBuf,
    pub workdir: PathBuf,
    pub maximum_timeout_seconds: u64,
    #[serde(default)]
    pub base_environment: BTreeMap<String, String>,
}

fn absolute(path: &Path) -> bool {
    path.is_absolute()
        && path.to_str().is_some()
        && path
            .components()
            .all(|part| matches!(part, Component::RootDir | Component::Normal(_)))
}
fn key(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
        && bytes.all(|c| c.is_ascii_alphanumeric() || c == b'_')
}
fn environment(values: &BTreeMap<String, String>) -> bool {
    values.len() <= 512
        && values
            .iter()
            .all(|(name, value)| key(name) && !value.contains('\0'))
        && values
            .iter()
            .try_fold(0usize, |total, (k, v)| {
                total.checked_add(k.len() + v.len() + 2)
            })
            .is_some_and(|bytes| bytes <= 128 * 1024)
}
impl Config {
    /// Pure configuration checks; no environment, clock or filesystem lookup.
    pub fn validate(&self) -> Result<()> {
        if !absolute(&self.manifest_path)
            || !absolute(&self.attempt_root)
            || self.attempt_root.file_name().is_none()
            || !absolute(&self.workdir)
            || !(1..1000 * 3600).contains(&self.maximum_timeout_seconds)
            || !environment(&self.base_environment)
            || self.base_environment.keys().any(|name| {
                name.starts_with("ML_EXPD_")
                    || ["OUTPUT_DIR", "PROJECT_NAME", "RUN_ID", "ATTEMPT_ID"]
                        .contains(&name.as_str())
            })
            || launch::endpoint(&self.endpoint).is_err()
        {
            return Err(Error::Configuration);
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = private_file(path, 1024 * 1024).map_err(|_| Error::Configuration)?;
        let config: Self = serde_json::from_slice(&bytes).map_err(|_| Error::Configuration)?;
        config.validate()?;
        Ok(config)
    }
    fn digest(&self) -> Result<String> {
        canonical::digest(&serde_json::to_value(self).map_err(|_| Error::Configuration)?)
            .map_err(|_| Error::Configuration)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Phase {
    Prepared,
    Started,
    Finished,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    contract: String,
    phase: Phase,
    binding_sha256: String,
    receipt_sha256: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Diagnostic {
    pub stage: String,
    pub code: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessEvidence {
    pub exit_code: Option<i32>,
    pub termination_signal: Option<i32>,
    pub timed_out: bool,
    pub cancelled: bool,
    pub error: Option<Diagnostic>,
    pub stdout_sha256: Option<String>,
    pub stderr_sha256: Option<String>,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Training {
    Completed,
    Failed,
    TimedOut,
    Cancelled,
    Unknown,
}
fn training(process: &ProcessEvidence) -> Training {
    if process.timed_out {
        Training::TimedOut
    } else if process.cancelled {
        Training::Cancelled
    } else if process.exit_code == Some(0) {
        Training::Completed
    } else if process.exit_code.is_some() || process.termination_signal.is_some() {
        Training::Failed
    } else {
        Training::Unknown
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Publication {
    NotImplemented,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub contract: String,
    pub identity: launch::Identity,
    pub manifest_sha256: String,
    pub process: ProcessEvidence,
    pub training: Training,
    pub publication: Publication,
    pub download_available: bool,
}
impl Receipt {
    /// Success covers the supervised local process and verified captured logs.
    /// It makes no assertion that outputs were archived, uploaded or published.
    pub fn succeeded(&self) -> bool {
        self.training == Training::Completed
            && self.process.error.is_none()
            && self.process.stdout_sha256.is_some()
            && self.process.stderr_sha256.is_some()
    }
}
#[derive(Debug, Serialize)]
pub struct Observation {
    pub contract: &'static str,
    pub identity: launch::Identity,
    pub manifest_sha256: String,
    pub phase: Phase,
    /// Even PREPARED is not a statement authorizing automatic replay.
    pub safe_to_retry: bool,
    pub receipt: Option<Receipt>,
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
) -> Result<(launch::Endpoint, launch::Manifest, BTreeMap<String, String>)> {
    config.validate()?;
    let target = launch::endpoint(&config.endpoint).map_err(|_| Error::Configuration)?;
    let manifest = launch::verify(bytes, &target).map_err(|_| Error::Manifest)?;
    if manifest.workdir != config.workdir.to_str().ok_or(Error::Configuration)?
        || manifest.environment.get("OUTPUT_DIR").map(String::as_str)
            != config.attempt_root.join("outputs").to_str()
        || manifest.timeout_seconds > config.maximum_timeout_seconds
    {
        return Err(Error::Binding);
    }
    if manifest
        .environment
        .keys()
        .any(|name| name.starts_with("ML_EXPD_"))
    {
        return Err(Error::Unsupported);
    }
    let mut values = config.base_environment.clone();
    values.extend(manifest.environment.clone());
    for (name, expected) in [
        ("PROJECT_NAME", &target.identity.project),
        ("RUN_ID", &target.identity.run_id),
        ("ATTEMPT_ID", &target.identity.attempt_id),
    ] {
        if values.get(name).is_some_and(|actual| actual != expected) {
            return Err(Error::Binding);
        }
        values.insert(name.into(), expected.clone());
    }
    if !environment(&values) {
        return Err(Error::Configuration);
    }
    let program = Path::new(&manifest.argv[0]);
    if !program.is_absolute() {
        if manifest.argv[0].contains('/')
            || !values.get("PATH").is_some_and(|path| {
                !path.is_empty() && path.split(':').all(|part| absolute(Path::new(part)))
            })
        {
            return Err(Error::Configuration);
        }
    } else if !absolute(program) {
        return Err(Error::Configuration);
    }
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
        let mut last = None;
        for (line, phase) in
            lines
                .into_iter()
                .zip([Phase::Prepared, Phase::Started, Phase::Finished])
        {
            let record: Record = serde_json::from_slice(line).map_err(|_| Error::Uncertain)?;
            if record.contract != CONTRACT || record.binding_sha256 != self.binding {
                return Err(Error::Binding);
            }
            if record.phase != phase
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
    fn append(&self, phase: Phase, receipt: Option<String>) -> Result<()> {
        self.validate()?;
        let record = Record {
            contract: CONTRACT.into(),
            phase,
            binding_sha256: self.binding.clone(),
            receipt_sha256: receipt,
        };
        let mut bytes = serde_json::to_vec(&record).map_err(|_| Error::Storage)?;
        bytes.push(b'\n');
        self.root
            .append_regular(OsStr::new(LOCK), &bytes, LEDGER_BYTES)
            .map_err(|_| Error::Storage)?;
        self.validate()
    }
    fn observe(&mut self, config: &Config) -> Result<Observation> {
        let record = self.state()?;
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
            contract: CONTRACT,
            identity: target.identity,
            manifest_sha256: target.sha256,
            phase: record.phase,
            safe_to_retry: false,
            receipt,
        })
    }
}

/// Explicit bootstrap of one fresh, dedicated Attempt root. Existing paths are
/// never adopted, reset or repaired. Its frozen OUTPUT_DIR fixes this location.
pub fn initialize(config: &Config) -> Result<Observation> {
    config.validate()?;
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
    held.append(Phase::Prepared, None)?;
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
    if held
        .root
        .entry_exists(OsStr::new("outputs"))
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
    let (target, manifest, environment) = verified(config, &bytes)?;
    // This fsync completes before any spawn attempt. Any later error leaves a
    // non-replayable STARTED record, including setup errors with no known child.
    held.append(Phase::Started, None)?;
    let execution = crate::process::execute_explicit(
        &config.workdir,
        &config.attempt_root.join("outputs"),
        &manifest.argv,
        Duration::from_secs(manifest.timeout_seconds),
        cancelled,
        &environment,
    )
    .map_err(|_| Error::Uncertain)?;
    held.validate().map_err(|_| Error::Uncertain)?;
    held.root
        .child(OsStr::new("outputs"), false)
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
    };
    let bytes = serde_json::to_vec(&receipt).map_err(|_| Error::Uncertain)?;
    held.root
        .publish_new(OsStr::new(RECEIPT), &bytes)
        .map_err(|_| Error::Uncertain)?;
    held.append(Phase::Finished, Some(sha(&bytes)))
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
