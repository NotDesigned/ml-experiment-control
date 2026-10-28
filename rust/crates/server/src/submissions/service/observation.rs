//! Explicit read-only provider sampling and bounded local observation cache.
//! Frozen manifests, submission Actions and provider resources are never changed.
use super::*;
use crate::backends::slurm::{CallOutcome, call};
use crate::experiments::Snapshot;
use ml_exp_core::managed::remote::Evidence;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    request_sha256: String,
    observed_at: String,
    evidence: Evidence,
}
impl Service {
    fn observation_name(prepared: &Prepared) -> Result<String> {
        Ok(format!(
            "{}.observation.json",
            native::attempt_slot(
                &prepared.binding.project,
                &prepared.binding.run_id,
                &prepared.binding.attempt_id
            )?
        ))
    }
    fn observation(&self, prepared: &Prepared) -> Result<Option<Saved>> {
        let name = Self::observation_name(prepared)?;
        if !self
            .slots
            .entry_exists(OsStr::new(&name))
            .map_err(|_| Error::Authority)?
        {
            return Ok(None);
        }
        let identity = self
            .slots
            .regular_identity(OsStr::new(&name))
            .map_err(|_| Error::Authority)?
            .ok_or(Error::Authority)?;
        if !identity.is_private_for(effective_uid()) || identity.mode & 0o077 != 0 {
            return Err(Error::Authority);
        }
        let bytes = self
            .slots
            .read_regular(OsStr::new(&name), helper::MAX_METADATA_BYTES as u64)
            .map_err(|_| Error::Authority)?;
        let saved: Saved = serde_json::from_slice(&bytes).map_err(|_| Error::Authority)?;
        if saved.request_sha256
            != prepared
                .request
                .identity()
                .map_err(|_| Error::Authority)?
                .request_sha256
        {
            return Err(Error::Authority);
        }
        time::OffsetDateTime::parse(
            &saved.observed_at,
            &time::format_description::well_known::Rfc3339,
        )
        .map_err(|_| Error::Authority)?;
        self.validate_observation(prepared, &saved.evidence)?;
        Ok(Some(saved))
    }
    fn validate_observation(&self, prepared: &Prepared, evidence: &Evidence) -> Result<()> {
        let snapshot = self.actions.observe(
            prepared.plan["action_id"]
                .as_str()
                .ok_or(Error::Authority)?,
        )?;
        let job = snapshot["execution"]["result"]["submission"]["backend_job_id"]
            .as_str()
            .ok_or(Error::Authority)?;
        if snapshot["execution"]["status"] != "VERIFIED" {
            return Err(Error::Conflict);
        }
        evidence
            .validate_scope(&prepared.request.intent().scope, job)
            .map_err(|_| Error::Authority)?;
        let (manifest, _) = prepared
            .request
            .prepared_content()
            .ok_or(Error::Authority)?;
        let plan =
            ml_exp_core::slurm::staging::plan(prepared.request.intent(), manifest.as_bytes())
                .map_err(|_| Error::Authority)?
                .ok_or(Error::Authority)?;
        let package = plan.managed.ok_or(Error::Authority)?;
        evidence
            .validate_config(&package.config)
            .map_err(|_| Error::Authority)
    }
    /// Refresh the already-owned exact job. No prepare, submit, cancel, claim,
    /// Action recovery or reconciliation is reachable from this operation.
    pub fn refresh(&self, action: &str, now: &str, cancelled: &AtomicBool) -> Result<Value> {
        time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
            .map_err(|_| Error::Configuration)?;
        let lock = self.lock()?;
        let prepared = self.load(action)?;
        let guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        let snapshot = self.actions.observe(action)?;
        if snapshot["execution"]["status"] != "VERIFIED" {
            return Err(Error::Conflict);
        }
        let job = snapshot["execution"]["result"]["submission"]["backend_job_id"]
            .as_str()
            .ok_or(Error::Authority)?;
        let request =
            helper::Request::observe_worker(prepared.request.intent().clone(), job.into())
                .map_err(|_| Error::Authority)?;
        let backend = self
            .backends
            .values()
            .find(|b| b.profile.target == request.intent().target)
            .ok_or(Error::Authority)?;
        let outcome =
            call(&backend.ssh, &request, cancelled).map_err(|_| Error::ObservationUnavailable)?;
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        self.check(&prepared)?;
        let CallOutcome::Receipt { reply, .. } = outcome else {
            return Err(Error::ObservationUnavailable);
        };
        let helper::ReplyOutcome::WorkerObserved { mut evidence } = reply.outcome else {
            return Err(Error::ObservationUnavailable);
        };
        self.validate_observation(&prepared, &evidence)?;
        if let Some(previous) = self.observation(&prepared)? {
            evidence
                .follows(&previous.evidence)
                .map_err(|_| Error::ObservationConflict)?;
            evidence.retain_windows(&previous.evidence);
        }
        let saved = Saved {
            request_sha256: prepared
                .request
                .identity()
                .map_err(|_| Error::Authority)?
                .request_sha256,
            observed_at: now.into(),
            evidence: *evidence,
        };
        let bytes = serde_json::to_vec(&saved).map_err(|_| Error::Authority)?;
        if bytes.len() > helper::MAX_METADATA_BYTES {
            return Err(Error::Authority);
        }
        let name = Self::observation_name(&prepared)?;
        lock.validate()?;
        if self
            .slots
            .entry_exists(OsStr::new(&name))
            .map_err(|_| Error::Authority)?
        {
            self.slots
                .replace_regular(OsStr::new(&name), &bytes)
                .map_err(|_| Error::Uncertain)?;
        } else {
            self.slots
                .publish_new(OsStr::new(&name), &bytes)
                .map_err(|_| Error::Uncertain)?;
        }
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        lock.validate()?;
        let mut response = self.public(snapshot, now)?;
        response["worker_observation"] = json!({"status":"OBSERVED","phase":saved.evidence.worker.phase,"received_at":saved.observed_at,"provider_mutations":false,"source":"native-worker-observation"});
        Ok(response)
    }
    /// GET projects only a previously accepted sample, never makes an RPC.
    pub fn apply_observation(&self, snapshot: &mut Snapshot) -> Result<()> {
        let ids: Vec<_> = snapshot
            .attempts
            .iter()
            .map(|v| v.attempt_id.clone())
            .collect();
        for id in ids {
            let Some(prepared) = self.saved_attempt(&snapshot.project, &snapshot.run_id, &id)?
            else {
                continue;
            };
            let Some(saved) = self.observation(&prepared)? else {
                continue;
            };
            crate::read_model::project_worker(
                snapshot,
                &prepared.binding,
                &saved.evidence,
                prepared.plan["action_id"]
                    .as_str()
                    .ok_or(Error::Authority)?,
                &saved.observed_at,
            )
            .map_err(|_| Error::Authority)?;
        }
        Ok(())
    }
}
