#![forbid(unsafe_code)]
mod artifacts;
mod checkpoints;
mod history;
mod jobs;
mod observation;
mod staging;
use fs2::FileExt;
use ml_exp_core::identity::submission_marker;
use ml_exp_core::slurm::{
    helper::{
        self, FrozenIntent, LedgerPhase, LedgerRecord, OperationTag, Rejection, Reply,
        ReplyOutcome, Request, SubmitDecision, Target, UnknownReason,
    },
    parse_submit_response,
};
use ml_exp_io::{directory::Directory, process};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fs::File,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use super::{Error, Result};
const ROOT_MARKER: &str = "root.json";
const REGISTRY: &str = "registry.json";
const LOCK: &str = ".slurm-helper.lock";
const STATE: &str = "state.json";
const INTENT: &str = "intent.json";
const MANIFEST: &str = "manifest.yaml";
const SCRIPT: &str = "job.sbatch";
const START: &str = "submit.started.json";
const RECEIPT: &str = "receipt.json";

// No Debug: trusted environment may contain operator credentials. The protocol
// cannot select or override any executable, path, environment or time budget.
pub struct HelperConfig {
    pub root: PathBuf,
    pub shared_run_root: Option<PathBuf>,
    pub target: Target,
    pub sbatch_program: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub lock_timeout: Duration,
    pub execution_timeout: Duration,
    pub scheduler: Option<SchedulerTools>,
    pub data_preparer: Option<PathBuf>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerTools {
    pub sacct_program: PathBuf,
    pub squeue_program: PathBuf,
    pub scancel_program: PathBuf,
    pub partition: String,
}
impl SchedulerTools {
    fn validate(&self) -> Result<()> {
        if [
            &self.sacct_program,
            &self.squeue_program,
            &self.scancel_program,
        ]
        .iter()
        .any(|path| {
            !path.is_absolute()
                || path
                    .to_str()
                    .is_none_or(|s| s.len() > 4096 || s.chars().any(char::is_control))
        }) || self.partition.is_empty()
            || self.partition.len() > 128
            || !self
                .partition
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
        {
            return Err(Error::Configuration);
        }
        Ok(())
    }
}
impl HelperConfig {
    /// Pure trusted-configuration checks. This does not open paths, inspect
    /// process environment, read a clock, or initialize/repair helper state.
    pub fn validate(&self) -> Result<()> {
        if self
            .data_preparer
            .as_ref()
            .is_some_and(|p| !ml_exp_core::managed::absolute(p))
        {
            return Err(Error::Configuration);
        }
        self.target.validate()?;
        if let Some(root) = &self.shared_run_root {
            let root = root.to_str().ok_or(Error::Configuration)?;
            ml_exp_core::slurm::staging::Requirement {
                worker: None,
                runs_root: root.into(),
            }
            .validate()?;
            let helper_root = self.root.to_str().ok_or(Error::Configuration)?;
            if ml_exp_core::slurm::path_within(root, helper_root)
                || ml_exp_core::slurm::path_within(helper_root, root)
            {
                return Err(Error::Configuration);
            }
        }
        if let Some(scheduler) = &self.scheduler {
            scheduler.validate()?;
        }
        if !self.root.is_absolute()
            || !self.sbatch_program.is_absolute()
            || self.lock_timeout.is_zero()
            || self.lock_timeout > Duration::from_secs(30)
            || self.execution_timeout.is_zero()
            || self.execution_timeout > Duration::from_secs(300)
            || self
                .sbatch_program
                .to_str()
                .is_none_or(|v| v.contains(['\0', '\n', '\r']))
        {
            return Err(Error::Configuration);
        }
        let environment_bytes = self
            .environment
            .iter()
            .try_fold(0usize, |total, (key, value)| {
                total
                    .checked_add(key.len())?
                    .checked_add(value.len())?
                    .checked_add(2)
            });
        if self.environment.len() > 128
            || environment_bytes.is_none_or(|bytes| bytes > 65536)
            || self.environment.iter().any(|(key, value)| {
                !key.as_bytes()
                    .first()
                    .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
                    || !key.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
                    || value.contains('\0')
                    // sbatch environment flags can override the frozen script
                    // resource budget or silently target another cluster.
                    || ["SBATCH_", "SQUEUE_", "SACCT_", "SCANCEL_"].iter().any(|prefix| key.starts_with(prefix))
                    || matches!(key.as_str(), "SLURM_CLUSTERS" | "SLURM_CONF" | "SLURM_CONF_SERVER")
            })
        {
            return Err(Error::Configuration);
        }
        Ok(())
    }
}
pub struct Helper {
    config: HelperConfig,
    root: Directory,
    shared: Option<Directory>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RootMarker {
    version: u16,
    target: Target,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared_storage: Option<staging::StorageBinding>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    version: u16,
    claims: BTreeMap<String, String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Started {
    version: u16,
    intent_sha256: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    version: u16,
    intent_sha256: String,
    job_id: String,
}

struct Lock {
    directory: Directory,
    file: File,
}
impl Lock {
    fn validate(&self) -> Result<()> {
        Ok(self
            .directory
            .validate_regular_entry(OsStr::new(LOCK), &self.file)?)
    }
}
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

struct Snapshot {
    directory: Directory,
    record: LedgerRecord,
    start_exists: bool,
    receipt: Option<Receipt>,
}
enum ReadLedger {
    Missing,
    Unknown(UnknownReason),
    Conflict,
    Found(Snapshot),
}

fn json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}
fn read_json<T: serde::de::DeserializeOwned>(directory: &Directory, name: &str) -> Result<T> {
    Ok(serde_json::from_slice(&directory.read_regular(
        OsStr::new(name),
        helper::MAX_METADATA_BYTES as u64,
    )?)?)
}

impl Helper {
    /// Operator bootstrap only. Refuse every nonempty root, including a marker
    /// with a missing lock; RPC requests can never initialize/repair this state.
    pub fn initialize(config: HelperConfig) -> Result<Self> {
        config.validate()?;
        let shared = staging::open_root(&config)?;
        let root = Directory::open(&config.root, true)?;
        root.verify_private_parent()?;
        if !root.entry_names()?.is_empty() {
            return Err(Error::Bootstrap);
        }
        let file = root.open_lock(OsStr::new(LOCK))?;
        let lock = Self::acquire(&root, file, config.lock_timeout, &AtomicBool::new(false))?;
        let names = root.entry_names()?;
        if names != [std::ffi::OsString::from(LOCK)] {
            return Err(Error::Bootstrap);
        }
        root.publish_new(
            OsStr::new(ROOT_MARKER),
            &json(&RootMarker {
                version: helper::PROTOCOL_VERSION,
                target: config.target.clone(),
                shared_storage: staging::binding(shared.as_ref())?,
            })?,
        )?;
        root.publish_new(
            OsStr::new(REGISTRY),
            &json(&Registry {
                version: helper::PROTOCOL_VERSION,
                claims: BTreeMap::new(),
            })?,
        )?;
        lock.validate()?;
        drop(lock);
        Ok(Self {
            config,
            root,
            shared,
        })
    }
    pub fn open(config: HelperConfig) -> Result<Self> {
        config.validate()?;
        let shared = staging::open_root(&config)?;
        let root = Directory::open(&config.root, false)?;
        root.verify_private_parent()?;
        let result = Self {
            config,
            root,
            shared,
        };
        let lock = result.lock(&AtomicBool::new(false))?;
        result.verify_root()?;
        lock.validate()?;
        drop(lock);
        Ok(result)
    }
    fn verify_root(&self) -> Result<()> {
        let marker: RootMarker = read_json(&self.root, ROOT_MARKER)?;
        if marker.version != helper::PROTOCOL_VERSION
            || marker.target != self.config.target
            || marker.shared_storage != staging::binding(self.shared.as_ref())?
        {
            return Err(Error::Bootstrap);
        }
        self.registry()?;
        self.root.verify_attached()?;
        Ok(())
    }
    fn registry(&self) -> Result<Registry> {
        let registry: Registry = read_json(&self.root, REGISTRY)?;
        if registry.version != helper::PROTOCOL_VERSION
            || registry.claims.iter().any(|(scope, intent)| {
                !helper::valid_digest(scope) || !helper::valid_digest(intent)
            })
        {
            return Err(Error::Bootstrap);
        }
        Ok(registry)
    }
    fn verify_issued(&self, intent: &FrozenIntent, directory: &Directory) -> Result<()> {
        self.verify_root()?;
        let registry = self.registry()?;
        if registry.claims.get(&intent.scope.digest()?) != Some(&intent.digest()?) {
            return Err(Error::Bootstrap);
        }
        directory.verify_attached()?;
        Ok(())
    }
    fn acquire(
        directory: &Directory,
        file: File,
        timeout: Duration,
        cancelled: &AtomicBool,
    ) -> Result<Lock> {
        let deadline = Instant::now() + timeout;
        loop {
            match FileExt::try_lock_exclusive(&file) {
                Ok(()) => {
                    let lock = Lock {
                        directory: directory.clone(),
                        file,
                    };
                    lock.validate()?;
                    return Ok(lock);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if cancelled.load(Ordering::Relaxed) || Instant::now() >= deadline {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "helper metadata lock unavailable",
                        )
                        .into());
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    fn lock(&self, cancelled: &AtomicBool) -> Result<Lock> {
        let file = self.root.open_existing_lock(OsStr::new(LOCK))?;
        let lock = Self::acquire(&self.root, file, self.config.lock_timeout, cancelled)?;
        self.verify_root()?;
        lock.validate()?;
        Ok(lock)
    }
    fn run(&self, intent: &FrozenIntent, create: bool) -> Result<Directory> {
        Ok(self
            .root
            .child(OsStr::new(&intent.scope.project), create)?
            .child(OsStr::new("runs"), create)?
            .child(OsStr::new(&intent.scope.run_id), create)?)
    }
    fn claim_name(intent: &FrozenIntent) -> String {
        format!(".submission-{}", intent.scope.attempt_id)
    }
    fn read_ledger(&self, intent: &FrozenIntent) -> Result<ReadLedger> {
        let registry = self.registry()?;
        let registered = registry.claims.get(&intent.scope.digest()?).cloned();
        let expected_digest = intent.digest()?;
        if registered
            .as_ref()
            .is_some_and(|digest| digest != &expected_digest)
        {
            return Ok(ReadLedger::Conflict);
        }
        let run = match self.run(intent, false) {
            Ok(run) => run,
            Err(Error::Storage(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(if registered.is_some() {
                    ReadLedger::Unknown(UnknownReason::MissingLedger)
                } else {
                    ReadLedger::Missing
                });
            }
            Err(error) => return Err(error),
        };
        let name = Self::claim_name(intent);
        if !run.entry_exists(OsStr::new(&name))? {
            return Ok(if registered.is_some() {
                ReadLedger::Unknown(UnknownReason::MissingLedger)
            } else {
                ReadLedger::Missing
            });
        }
        if registered.is_none() {
            return Ok(ReadLedger::Unknown(UnknownReason::LegacyClaim));
        }
        let directory = run.child(OsStr::new(&name), false)?;
        if !directory.entry_exists(OsStr::new(STATE))? {
            return Ok(ReadLedger::Unknown(
                if directory.entry_exists(OsStr::new(INTENT))? {
                    UnknownReason::MalformedLedger
                } else {
                    UnknownReason::LegacyClaim
                },
            ));
        }
        let record: LedgerRecord = match read_json(&directory, STATE) {
            Ok(record) => record,
            Err(_) => return Ok(ReadLedger::Unknown(UnknownReason::MalformedLedger)),
        };
        let stored: FrozenIntent = match read_json(&directory, INTENT) {
            Ok(record) => record,
            Err(_) => return Ok(ReadLedger::Unknown(UnknownReason::MalformedLedger)),
        };
        if record.validate().is_err() || stored.validate().is_err() || record.intent != stored {
            return Ok(ReadLedger::Unknown(UnknownReason::MalformedLedger));
        }
        if &record.intent != intent {
            return Ok(ReadLedger::Conflict);
        }
        let manifest =
            directory.read_regular(OsStr::new(MANIFEST), helper::MAX_METADATA_BYTES as u64)?;
        let script =
            directory.read_regular(OsStr::new(SCRIPT), helper::MAX_METADATA_BYTES as u64)?;
        if helper::sha256_bytes(&manifest) != intent.manifest_sha256
            || helper::sha256_bytes(&script) != intent.script_sha256
        {
            return Ok(ReadLedger::Unknown(UnknownReason::MalformedLedger));
        }
        let start_exists = directory.entry_exists(OsStr::new(START))?;
        // After dispatch, mutable training directories cannot erase the known
        // scheduler identity or prevent its observation/cancellation. A new
        // submission still requires the original complete staging receipt.
        if !start_exists && self.verify_shared(intent, &directory).is_err() {
            return Ok(ReadLedger::Unknown(UnknownReason::StorageUnavailable));
        }
        if start_exists {
            let start: Started = match read_json(&directory, START) {
                Ok(value) => value,
                Err(_) => return Ok(ReadLedger::Unknown(UnknownReason::MalformedLedger)),
            };
            if start.version != helper::PROTOCOL_VERSION
                || start.intent_sha256 != intent.digest()?
            {
                return Ok(ReadLedger::Unknown(UnknownReason::MalformedLedger));
            }
        }
        let receipt = if directory.entry_exists(OsStr::new(RECEIPT))? {
            let receipt: Receipt = match read_json(&directory, RECEIPT) {
                Ok(value) => value,
                Err(_) => return Ok(ReadLedger::Unknown(UnknownReason::MalformedLedger)),
            };
            let valid = receipt.version == helper::PROTOCOL_VERSION
                && receipt.intent_sha256 == intent.digest()?
                && parse_submit_response(&receipt.job_id)
                    .is_ok_and(|job| job.as_str() == receipt.job_id);
            if !valid || !start_exists || matches!(record.phase, LedgerPhase::Prepared) {
                return Ok(ReadLedger::Unknown(UnknownReason::MalformedLedger));
            }
            Some(receipt)
        } else {
            None
        };
        if let LedgerPhase::Submitted { job_id } = &record.phase {
            if receipt
                .as_ref()
                .is_none_or(|receipt| &receipt.job_id != job_id)
            {
                return Ok(ReadLedger::Unknown(UnknownReason::MalformedLedger));
            }
        }
        Ok(ReadLedger::Found(Snapshot {
            directory,
            record,
            start_exists,
            receipt,
        }))
    }
    fn observed(ledger: &ReadLedger) -> ReplyOutcome {
        match ledger {
            ReadLedger::Missing => ReplyOutcome::Unknown {
                reason: UnknownReason::MissingLedger,
            },
            ReadLedger::Unknown(reason) => ReplyOutcome::Unknown { reason: *reason },
            ReadLedger::Conflict => ReplyOutcome::Rejected {
                code: Rejection::IdentityConflict,
            },
            ReadLedger::Found(snapshot) => helper::ledger_outcome(
                &snapshot.record,
                snapshot.start_exists,
                snapshot
                    .receipt
                    .as_ref()
                    .map(|receipt| receipt.job_id.as_str()),
            )
            .unwrap_or(ReplyOutcome::Unknown {
                reason: UnknownReason::MalformedLedger,
            }),
        }
    }
    /// Exactly one scoped operation. It never replays a started submission,
    /// clears a claim, guesses scheduler absence, or resets the original token.
    pub fn handle(&self, request: &Request, cancelled: &AtomicBool) -> Result<Reply> {
        request.validate()?;
        if request.intent().target != self.config.target {
            return Ok(Reply::for_request(
                request,
                ReplyOutcome::Rejected {
                    code: Rejection::IdentityConflict,
                },
            )?);
        }
        if request.intent().staging.as_ref().is_some_and(|required| {
            self.config
                .shared_run_root
                .as_ref()
                .and_then(|p| p.to_str())
                != Some(required.runs_root.as_str())
        }) {
            return Ok(Reply::for_request(
                request,
                ReplyOutcome::Rejected {
                    code: Rejection::UnsupportedOperation,
                },
            )?);
        }
        if cancelled.load(Ordering::Relaxed) {
            return Ok(Reply::for_request(
                request,
                ReplyOutcome::Unknown {
                    reason: UnknownReason::Cancelled,
                },
            )?);
        }
        let lock = self.lock(cancelled)?;
        let ledger = self.read_ledger(request.intent())?;
        // Keep the exact directory used by this operation, rather than only
        // reopening its pathname after a scheduler call may have changed it.
        let retained = match &ledger {
            ReadLedger::Found(snapshot) => Some(snapshot.directory.clone()),
            _ => None,
        };
        let outcome = match request.operation() {
            OperationTag::PrepareSubmission if matches!(ledger, ReadLedger::Missing) => {
                return self.prepare(request, lock, cancelled);
            }
            OperationTag::Submit => match ledger {
                ReadLedger::Found(snapshot) => self.submit(request, snapshot, &lock, cancelled)?,
                ReadLedger::Missing => ReplyOutcome::Rejected {
                    code: Rejection::Unprepared,
                },
                other => Self::observed(&other),
            },
            OperationTag::ReadMetricHistory => match ledger {
                ReadLedger::Found(snapshot) => self.metric_history(request, &snapshot, &lock)?,
                other => Self::observed(&other),
            },
            OperationTag::ObserveWorker => match ledger {
                ReadLedger::Found(snapshot) => {
                    self.worker_observation(request, &snapshot, &lock)?
                }
                other => Self::observed(&other),
            },
            OperationTag::ObserveCheckpoint => match ledger {
                ReadLedger::Found(snapshot) => {
                    self.checkpoint_observation(request, &snapshot, &lock)?
                }
                other => Self::observed(&other),
            },
            OperationTag::Artifacts => match ledger {
                ReadLedger::Found(snapshot) => {
                    self.artifact_operation(request, &snapshot, &lock, cancelled)?
                }
                other => Self::observed(&other),
            },
            OperationTag::ObserveJob | OperationTag::CancelJob => match ledger {
                ReadLedger::Found(snapshot) => {
                    self.job_operation(request, snapshot, &lock, cancelled)?
                }
                ReadLedger::Missing => ReplyOutcome::Rejected {
                    code: Rejection::Unprepared,
                },
                other => Self::observed(&other),
            },
            _ => Self::observed(&ledger),
        };
        self.verify_root()?;
        if matches!(
            outcome,
            ReplyOutcome::Prepared | ReplyOutcome::Submitted { .. }
        ) {
            let directory = retained.as_ref().ok_or(Error::Bootstrap)?;
            self.verify_issued(request.intent(), directory)?;
            if Self::observed(&self.read_ledger(request.intent())?) != outcome {
                return Err(Error::Bootstrap);
            }
            self.verify_issued(request.intent(), directory)?;
        } else if let Some(directory) = retained.as_ref() {
            directory.verify_attached()?;
        }
        lock.validate()?;
        Ok(Reply::for_request(request, outcome)?)
    }
    fn prepare(&self, request: &Request, lock: Lock, cancelled: &AtomicBool) -> Result<Reply> {
        let (manifest, script) = request.prepared_content().ok_or(Error::Configuration)?;
        let mut registry = self.registry()?;
        let scope = request.intent().scope.digest()?;
        if registry.claims.contains_key(&scope) {
            return Ok(Reply::for_request(
                request,
                ReplyOutcome::Unknown {
                    reason: UnknownReason::MissingLedger,
                },
            )?);
        }
        registry.claims.insert(scope, request.intent().digest()?);
        let registry_bytes = json(&registry)?;
        if registry_bytes.len() > helper::MAX_METADATA_BYTES {
            return Ok(Reply::for_request(
                request,
                ReplyOutcome::Rejected {
                    code: Rejection::CapacityFull,
                },
            )?);
        }
        lock.validate()?;
        self.root
            .replace_regular(OsStr::new(REGISTRY), &registry_bytes)?;
        lock.validate()?;
        let run = self.run(request.intent(), true)?;
        let name = Self::claim_name(request.intent());
        // Never repair an empty old claim or race another marker's creator.
        if run.entry_exists(OsStr::new(&name))? {
            return Ok(Reply::for_request(
                request,
                ReplyOutcome::Unknown {
                    reason: UnknownReason::LegacyClaim,
                },
            )?);
        }
        let directory = run.child(OsStr::new(&name), true)?;
        lock.validate()?;
        directory.publish_new(OsStr::new(INTENT), &json(request.intent())?)?;
        directory.publish_new(OsStr::new(MANIFEST), manifest.as_bytes())?;
        directory.publish_new(OsStr::new(SCRIPT), script.as_bytes())?;
        // The durable claim prevents replay while a download is running. Do
        // not hold the global metadata lock across that potentially long job:
        // observations and cancellation of other Runs must remain available.
        lock.validate()?;
        drop(lock);
        self.prepare_shared(request.intent(), manifest.as_bytes(), &directory, cancelled)?;
        let lock = self.lock(cancelled)?;
        self.verify_issued(request.intent(), &directory)?;
        for (name, expected) in [
            (INTENT, json(request.intent())?),
            (MANIFEST, manifest.as_bytes().to_vec()),
            (SCRIPT, script.as_bytes().to_vec()),
        ] {
            if directory.read_regular(OsStr::new(name), helper::MAX_METADATA_BYTES as u64)?
                != expected
            {
                return Err(Error::Bootstrap);
            }
        }
        if cancelled.load(Ordering::Relaxed)
            || [STATE, START, RECEIPT]
                .into_iter()
                .any(|name| directory.entry_exists(OsStr::new(name)).unwrap_or(true))
        {
            return Err(Error::Bootstrap);
        }
        directory.publish_new(
            OsStr::new(STATE),
            &json(&LedgerRecord::prepared(request.intent().clone())?)?,
        )?;
        directory.verify_attached()?;
        if Self::observed(&self.read_ledger(request.intent())?) != ReplyOutcome::Prepared {
            return Err(Error::Bootstrap);
        }
        self.verify_issued(request.intent(), &directory)?;
        lock.validate()?;
        Ok(Reply::for_request(request, ReplyOutcome::Prepared)?)
    }
    fn submit(
        &self,
        request: &Request,
        mut snapshot: Snapshot,
        lock: &Lock,
        cancelled: &AtomicBool,
    ) -> Result<ReplyOutcome> {
        if let Some(receipt) = &snapshot.receipt {
            return Ok(helper::ledger_outcome(
                &snapshot.record,
                snapshot.start_exists,
                Some(&receipt.job_id),
            )?);
        }
        match helper::submission_gate(request.intent(), &snapshot.record, snapshot.start_exists)? {
            SubmitDecision::StartOnce => {}
            SubmitDecision::ReturnExisting { job_id } => {
                return Ok(ReplyOutcome::Submitted { job_id });
            }
            SubmitDecision::ReconcileRequired => {
                return Ok(ReplyOutcome::Unknown {
                    reason: UnknownReason::SubmitStarted,
                });
            }
        }
        let script = snapshot
            .directory
            .read_regular(OsStr::new(SCRIPT), helper::MAX_METADATA_BYTES as u64)?;
        if helper::sha256_bytes(&script) != request.intent().script_sha256 {
            return Ok(ReplyOutcome::Unknown {
                reason: UnknownReason::MalformedLedger,
            });
        }
        lock.validate()?;
        snapshot.directory.publish_new(
            OsStr::new(START),
            &json(&Started {
                version: helper::PROTOCOL_VERSION,
                intent_sha256: request.intent().digest()?,
            })?,
        )?;
        snapshot.record.phase = LedgerPhase::SubmitStarted;
        snapshot
            .directory
            .replace_regular(OsStr::new(STATE), &json(&snapshot.record)?)?;
        self.verify_issued(request.intent(), &snapshot.directory)?;
        self.verify_shared(request.intent(), &snapshot.directory)?;
        lock.validate()?;
        let argv = vec![
            self.config.sbatch_program.display().to_string(),
            "--parsable".into(),
            // Slurm CLI flags outrank both script directives and input env.
            // Header inspection alone cannot catch every short-option alias.
            format!("--job-name={}", request.intent().scheduler_name),
            format!(
                "--comment={}",
                submission_marker(&request.intent().submission_token)
                    .map_err(|_| Error::Configuration)?
            ),
        ];
        // sbatch accepts the frozen script on stdin. Never ask it to reopen an
        // attacker-replaceable script path after our byte/digest validation.
        let local = process::execute(process::Request {
            argv: &argv,
            cwd: &self.config.root,
            environment: process::Environment::Replace(&self.config.environment),
            stdin: process::Input::Bytes(&script),
            stdout: process::Capture::Memory { limit: 4096 },
            stderr: process::Capture::Memory { limit: 16384 },
            deadline: Instant::now() + self.config.execution_timeout,
            termination_grace: Duration::from_millis(250),
            drain_grace: Duration::from_millis(100),
            cancelled,
        })?;
        lock.validate()?;
        self.verify_issued(request.intent(), &snapshot.directory)?;
        if local.exit_code != Some(0)
            || local.timed_out
            || local.cancelled
            || local.failure.is_some()
            || !local.stdin_complete
            || !local.stdout.complete
            || local.stdout.truncated
        {
            return Ok(ReplyOutcome::Unknown {
                reason: UnknownReason::UncertainProcessResult,
            });
        }
        let job = match std::str::from_utf8(&local.stdout.bytes)
            .ok()
            .and_then(|text| parse_submit_response(text).ok())
        {
            Some(job) => job,
            None => {
                return Ok(ReplyOutcome::Unknown {
                    reason: UnknownReason::UncertainProcessResult,
                });
            }
        };
        snapshot.directory.publish_new(
            OsStr::new(RECEIPT),
            &json(&Receipt {
                version: helper::PROTOCOL_VERSION,
                intent_sha256: request.intent().digest()?,
                job_id: job.as_str().into(),
            })?,
        )?;
        snapshot.record.phase = LedgerPhase::Submitted {
            job_id: job.as_str().into(),
        };
        snapshot
            .directory
            .replace_regular(OsStr::new(STATE), &json(&snapshot.record)?)?;
        lock.validate()?;
        Ok(ReplyOutcome::Submitted {
            job_id: job.as_str().into(),
        })
    }
}
