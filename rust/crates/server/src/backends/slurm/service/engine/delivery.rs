//! Idempotent preparation before a submission: the worker an Attempt froze, and its
//! code, placed on the cluster's shared storage over the backend's own SSH channel.
//! Each is named by its content, proven where it lands and never replaced, so any
//! number of Runs and Attempts reuse one copy. Nothing here asks the scheduler for
//! anything; the one submission stays in `action.rs`.
use super::*;
use ml_exp_core::slurm::delivery::{self, Frozen, Settings};
use ml_exp_io::process::{self, Capture, Environment, Input};
use std::{
    io::{Seek, Write},
    time::{Duration, Instant},
};

/// Hashing a worker or a code source on shared storage takes longer than a metadata call.
const CHECK: Duration = Duration::from_secs(600);
/// Up to 64 MiB of code, or a worker, through the gateway.
const TRANSFER: Duration = Duration::from_secs(1200);

/// What one gateway command did. A transport failure is never an answer.
struct Answer {
    succeeded: bool,
    stdout: String,
}
impl Service {
    /// What a new Run on this backend freezes: its delivery root and this server's
    /// worker. None when either is missing, and then the Run works from its image.
    pub(super) fn delivery_offer(&self, backend: &Backend) -> Option<(Settings, String)> {
        let settings = backend.delivery.clone()?;
        let worker = self.runtime_builder.as_ref()?.current_worker()?;
        Some((settings, worker))
    }
    /// Whether a Run on this executor may name a code source to deliver.
    pub fn delivers_code(&self, executor: &str) -> bool {
        self.backends
            .get(executor)
            .is_some_and(|backend| self.delivery_offer(backend).is_some())
    }
    /// One fixed command on the gateway: the backend's pinned SSH invocation, with the
    /// helper and its arguments replaced by `tail`.
    fn gateway(
        &self,
        backend: &Backend,
        tail: Vec<String>,
        stdin: Input<'_>,
        timeout: Duration,
        cancelled: &AtomicBool,
    ) -> Result<Answer> {
        let ssh = &backend.ssh;
        ssh.validate().map_err(why!(Error::Configuration))?;
        let mut argv = ssh.argv();
        let at = argv
            .iter()
            .position(|arg| arg == &ssh.helper_path)
            .ok_or(Error::Configuration)?;
        argv.truncate(at);
        argv.extend(tail);
        let environment = ssh.environment();
        let outcome = process::execute(process::Request {
            argv: &argv,
            cwd: &ssh.local_cwd,
            environment: Environment::Replace(&environment),
            stdin,
            stdout: Capture::Memory { limit: 4096 },
            stderr: Capture::Memory { limit: 8192 },
            deadline: Instant::now() + timeout,
            termination_grace: ssh.budget.termination_grace,
            drain_grace: ssh.budget.drain_grace,
            cancelled,
        })
        .map_err(why!(Error::DeliveryUnavailable))?;
        if outcome.timed_out
            || outcome.cancelled
            || outcome.failure.is_some()
            || !outcome.stdin_complete
            || !outcome.stdout.complete
            || outcome.stdout.truncated
        {
            crate::why::record(file!(), line!(), &outcome.diagnostic().to_string());
            return Err(Error::DeliveryUnavailable);
        }
        let succeeded = outcome.exit_code == Some(0) && outcome.termination_signal.is_none();
        if !succeeded {
            // The worker names a file that differs, which is what an operator needs.
            crate::why::record(file!(), line!(), &outcome.diagnostic().to_string());
        }
        Ok(Answer {
            succeeded,
            stdout: String::from_utf8(outcome.stdout.bytes)
                .map_err(why!(Error::DeliveryUnavailable))?,
        })
    }
    fn worker_state(
        &self,
        backend: &Backend,
        frozen: &Frozen,
        helper: bool,
        cancelled: &AtomicBool,
    ) -> Result<&'static str> {
        let tail =
            delivery::check_worker(frozen, helper.then_some(backend.ssh.helper_path.as_str()))
                .map_err(why!(Error::Configuration))?;
        let answer = self.gateway(backend, tail, Input::Null, CHECK, cancelled)?;
        match delivery::answer(&answer.stdout).filter(|_| answer.succeeded) {
            Some(delivery::PRESENT) => Ok(delivery::PRESENT),
            Some(delivery::ABSENT) => Ok(delivery::ABSENT),
            Some(delivery::CORRUPT) => Err(Error::DeliveryRefused),
            _ => Err(Error::DeliveryUnavailable),
        }
    }
    /// The exact worker on shared storage, sent from this server's store only when it is
    /// absent. With `helper`, the gateway's fixed helper path is also made this worker.
    fn ensure_worker(
        &self,
        backend: &Backend,
        frozen: &Frozen,
        helper: bool,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        if self.worker_state(backend, frozen, helper, cancelled)? == delivery::PRESENT {
            return Ok(());
        }
        let builder = self
            .runtime_builder
            .as_ref()
            .ok_or(Error::WorkerUnavailable)?;
        let (length, file) = builder.worker_file(&frozen.worker_sha256)?;
        let tail = delivery::place_worker(frozen).map_err(why!(Error::Configuration))?;
        let answer = self.gateway(
            backend,
            tail,
            Input::File {
                file: &file,
                length,
            },
            TRANSFER,
            cancelled,
        )?;
        if !answer.succeeded || delivery::answer(&answer.stdout) != Some(delivery::PLACED) {
            return Err(Error::DeliveryUnavailable);
        }
        // Placed is what the gateway said; present is what it proves now.
        match self.worker_state(backend, frozen, helper, cancelled)? {
            delivery::PRESENT => Ok(()),
            _ => Err(Error::DeliveryUnavailable),
        }
    }
    /// This server's worker, and the gateway's helper with it, so that a release needs
    /// nothing uploaded by hand. Only where a delivery root is configured.
    pub(super) fn refresh_helper(&self, backend: &Backend, cancelled: &AtomicBool) -> Result<()> {
        let Some((settings, worker)) = self.delivery_offer(backend) else {
            return Ok(());
        };
        let current = settings
            .freeze(&worker, None)
            .map_err(why!(Error::Configuration))?;
        self.ensure_worker(backend, &current, true, cancelled)
    }
    fn code_state(
        &self,
        backend: &Backend,
        frozen: &Frozen,
        cancelled: &AtomicBool,
    ) -> Result<bool> {
        let tail = delivery::receive_code(frozen, true).map_err(why!(Error::Configuration))?;
        let answer = self.gateway(backend, tail, Input::Null, CHECK, cancelled)?;
        match (answer.succeeded, answer.stdout.trim_end()) {
            (true, delivery::CODE_READY) => Ok(true),
            (true, delivery::CODE_ABSENT) => Ok(false),
            // The Run's worker ran and refused the copy that is there.
            (false, delivery::CODE_REFUSED) => Err(Error::DeliveryRefused),
            _ => Err(Error::DeliveryUnavailable),
        }
    }
    /// The Run's code on shared storage: checked first, and sent only when absent. The
    /// gateway proves every file and the whole tree before it publishes the listing.
    fn ensure_code(
        &self,
        backend: &Backend,
        project: &str,
        frozen: &Frozen,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let code = frozen.code_source_id.as_deref().ok_or(Error::Authority)?;
        if self.code_state(backend, frozen, cancelled)? {
            return Ok(());
        }
        let manifest = self
            .sources
            .code_manifest(project, code)
            .map_err(why!(Error::Authority))?;
        let listing = serde_json::to_vec(&manifest).map_err(why!(Error::Configuration))?;
        let size = u32::try_from(listing.len()).map_err(why!(Error::Configuration))?;
        // Spooled to an unnamed file, so that 64 MiB of code is never held in memory.
        let mut file = tempfile::tempfile().map_err(why!(Error::DeliveryUnavailable))?;
        file.write_all(&size.to_be_bytes())
            .map_err(why!(Error::DeliveryUnavailable))?;
        file.write_all(&listing)
            .map_err(why!(Error::DeliveryUnavailable))?;
        self.sources
            .code_content(project, code)
            .map_err(why!(Error::Authority))?
            .each_chunk(|chunk| file.write_all(&chunk))
            .map_err(why!(Error::DeliveryUnavailable))?;
        file.flush().map_err(why!(Error::DeliveryUnavailable))?;
        let length = file
            .stream_position()
            .map_err(why!(Error::DeliveryUnavailable))?;
        let tail = delivery::receive_code(frozen, false).map_err(why!(Error::Configuration))?;
        let answer = self.gateway(
            backend,
            tail,
            Input::File {
                file: &file,
                length,
            },
            TRANSFER,
            cancelled,
        )?;
        match (answer.succeeded, answer.stdout.trim_end()) {
            (true, delivery::CODE_READY | delivery::CODE_REUSED) => Ok(()),
            (false, delivery::CODE_REFUSED) => Err(Error::DeliveryRefused),
            _ => Err(Error::DeliveryUnavailable),
        }
    }
    /// Before the one submission of a prepared Attempt. A Run that froze no delivery
    /// works from its image; only the gateway's helper is kept current for it.
    pub(super) fn deliver(
        &self,
        backend: &Backend,
        prepared: &Prepared,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        self.refresh_helper(backend, cancelled)?;
        let (manifest, _) = prepared
            .request
            .prepared_content()
            .ok_or(Error::Authority)?;
        let document = Value::Object(
            ml_exp_core::json::mapping(manifest.as_bytes()).map_err(why!(Error::Authority))?,
        );
        let Some(frozen) = Frozen::from_document(&document).map_err(why!(Error::Authority))? else {
            return Ok(());
        };
        let current = self
            .delivery_offer(backend)
            .and_then(|(settings, worker)| settings.freeze(&worker, None).ok());
        if current.as_ref().is_none_or(|current| {
            current.root != frozen.root || current.worker_sha256 != frozen.worker_sha256
        }) {
            self.ensure_worker(backend, &frozen, false, cancelled)?;
        }
        if frozen.code_source_id.is_some() {
            self.ensure_code(backend, &prepared.binding.project, &frozen, cancelled)?;
        }
        Ok(())
    }
}
