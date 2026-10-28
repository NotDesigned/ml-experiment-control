//! Explicit checkpoint verification on shared storage. Only the named generation
//! is sealed; payload bytes never pass through the API host or scheduler.
use super::*;
use ml_exp_core::checkpoints;

impl Helper {
    pub(super) fn checkpoint_observation(
        &self,
        request: &Request,
        snapshot: &Snapshot,
        lock: &Lock,
    ) -> Result<ReplyOutcome> {
        let Some(receipt) = &snapshot.receipt else {
            return Ok(ReplyOutcome::Unknown {
                reason: UnknownReason::SubmitStarted,
            });
        };
        if request.requested_job_id() != Some(receipt.job_id.as_str()) {
            return Ok(ReplyOutcome::Rejected {
                code: Rejection::IdentityConflict,
            });
        }
        self.verify_issued(request.intent(), &snapshot.directory)?;
        lock.validate()?;
        let read = || -> Result<serde_json::Value> {
            let (plan, directories) = self
                .observe_shared(request.intent(), &snapshot.directory, false)?
                .ok_or(Error::Configuration)?;
            let package = plan.managed.as_ref().ok_or(Error::Configuration)?;
            // The sealed worker ledger proves this output tree belongs to the
            // issued job. It does not claim the training process is alive.
            crate::managed::sample(&package.config).map_err(|_| Error::Metadata)?;
            let outputs = directories[5].child(OsStr::new("payload"), false)?;
            let before = outputs.read_regular(
                OsStr::new("checkpoint.ready.json"),
                checkpoints::MANIFEST_LIMIT as u64,
            )?;
            let ready = serde_json::Value::Object(
                ml_exp_core::json::mapping(&before).map_err(|_| Error::Metadata)?,
            );
            let manifest = checkpoints::manifest(&ready).map_err(|_| Error::Metadata)?;
            ml_exp_io::checkpoints::verify(&outputs, &manifest, true)
                .map_err(|_| Error::Metadata)?;
            if outputs.read_regular(
                OsStr::new("checkpoint.ready.json"),
                checkpoints::MANIFEST_LIMIT as u64,
            )? != before
            {
                return Err(Error::Metadata);
            }
            outputs.verify_private_parent()?;
            self.observe_shared(request.intent(), &snapshot.directory, false)?;
            let attempt = snapshot
                .directory
                .read_regular(OsStr::new(MANIFEST), helper::MAX_METADATA_BYTES as u64)?;
            let context =
                ml_exp_core::slurm::staging::checkpoint_context(request.intent(), &attempt)?;
            checkpoints::document(&context, &ready).map_err(|_| Error::Metadata)
        };
        let outcome = match read() {
            Ok(document) => ReplyOutcome::CheckpointObserved {
                document: Box::new(document),
            },
            Err(_) => ReplyOutcome::Unknown {
                reason: UnknownReason::StorageUnavailable,
            },
        };
        self.verify_issued(request.intent(), &snapshot.directory)?;
        lock.validate()?;
        Ok(outcome)
    }
}
