//! Exact scheduler reads have a cache separate from the immutable Action commit.
use super::*;
use crate::backends::slurm::{CallOutcome, call};
use ml_exp_core::slurm::{JobStatus, StatusObservation};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Saved {
    request_sha256: String,
    pub(super) observation: StatusObservation,
}
impl Service {
    fn scheduler_name(prepared: &Prepared) -> Result<String> {
        Ok(format!(
            "{}.scheduler.json",
            native::attempt_slot(
                &prepared.binding.project,
                &prepared.binding.run_id,
                &prepared.binding.attempt_id
            )?
        ))
    }
    pub(super) fn scheduler_observation(&self, prepared: &Prepared) -> Result<Option<Saved>> {
        let name = Self::scheduler_name(prepared)?;
        if !self
            .slots
            .entry_exists(OsStr::new(&name))
            .map_err(|_| Error::Authority)?
        {
            return Ok(None);
        }
        let bytes = self
            .slots
            .read_regular(OsStr::new(&name), 65536)
            .map_err(|_| Error::Authority)?;
        let saved: Saved = serde_json::from_value(Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Authority)?,
        ))
        .map_err(|_| Error::Authority)?;
        let original: JobStatus = serde_json::from_value(
            self.actions.observe(
                prepared.plan["action_id"]
                    .as_str()
                    .ok_or(Error::Authority)?,
            )?["execution"]["result"]["observation"]["status"]
                .clone(),
        )
        .map_err(|_| Error::Authority)?;
        if saved.request_sha256
            != prepared
                .request
                .identity()
                .map_err(|_| Error::Authority)?
                .request_sha256
            || !ml_exp_core::polling::scheduler_follows(&original, &saved.observation)
        {
            return Err(Error::Authority);
        }
        Ok(Some(saved))
    }
    pub fn refresh_scheduler(
        &self,
        action: &str,
        _now: &str,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        self.refresh_scheduler_with_limit(action, _now, cancelled, None)
    }
    pub(super) fn refresh_scheduler_with_limit(
        &self,
        action: &str,
        _now: &str,
        cancelled: &AtomicBool,
        limit: Option<std::time::Duration>,
    ) -> Result<Value> {
        let lock = self.lock()?;
        let prepared = self.load(action)?;
        let guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        self.check(&prepared)?;
        let snapshot = self.actions.observe(action)?;
        if snapshot["execution"]["status"] != "VERIFIED" {
            return Err(Error::Conflict);
        }
        let original: JobStatus = serde_json::from_value(
            snapshot["execution"]["result"]["observation"]["status"].clone(),
        )
        .map_err(|_| Error::Authority)?;
        let request = helper::Request::observe_job(
            prepared.request.intent().clone(),
            original.job_id.as_str().into(),
        )
        .map_err(|_| Error::Authority)?;
        let backend = self
            .backends
            .values()
            .find(|b| b.profile.target == request.intent().target)
            .ok_or(Error::Authority)?;
        let outcome = call(&Self::observation_ssh(backend, limit), &request, cancelled)
            .map_err(|_| Error::ObservationUnavailable)?;
        let CallOutcome::Receipt { reply, .. } = outcome else {
            return Err(Error::ObservationUnavailable);
        };
        if reply.identity != request.identity().map_err(|_| Error::Authority)? {
            return Err(Error::Authority);
        }
        let helper::ReplyOutcome::JobObserved { observation } = reply.outcome else {
            return Err(Error::ObservationUnavailable);
        };
        let previous = self.scheduler_observation(&prepared)?;
        if !ml_exp_core::polling::scheduler_follows(
            previous
                .as_ref()
                .map_or(&original, |s| &s.observation.status),
            &observation,
        ) {
            return Err(Error::ObservationConflict);
        }
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        self.check(&prepared)?;
        lock.validate()?;
        let saved = Saved {
            request_sha256: prepared
                .request
                .identity()
                .map_err(|_| Error::Authority)?
                .request_sha256,
            observation: *observation,
        };
        self.slots
            .replace_regular(
                OsStr::new(&Self::scheduler_name(&prepared)?),
                &serde_json::to_vec(&saved).map_err(|_| Error::Authority)?,
            )
            .map_err(|_| Error::Uncertain)?;
        Ok(
            json!({"submission_id":action,"scheduler_observation":saved.observation,"scheduler_mutation_performed":false}),
        )
    }
}
