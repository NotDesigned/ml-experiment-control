//! Persist cancellation intent before the exact-job RPC. Repeated explicit
//! calls reconcile the helper's durable cancel outbox; they never resubmit.
use super::*;
use crate::backends::slurm::{CallOutcome, call};
use ml_exp_core::slurm::cancellation::State;
impl Service {
    fn cancellation_request(&self, prepared: &Prepared) -> Result<helper::Request> {
        let snapshot = self.actions.observe(
            prepared.plan["action_id"]
                .as_str()
                .ok_or(Error::Authority)?,
        )?;
        if snapshot["execution"]["status"] != "VERIFIED" {
            return Err(Error::Conflict);
        }
        let job = snapshot["execution"]["result"]["submission"]["backend_job_id"]
            .as_str()
            .ok_or(Error::Authority)?;
        helper::Request::cancel_job(prepared.request.intent().clone(), job.into())
            .map_err(|_| Error::Authority)
    }
    fn cancellation_name(prepared: &Prepared) -> Result<String> {
        Ok(format!(
            "{}.cancellation.json",
            native::attempt_slot(
                &prepared.binding.project,
                &prepared.binding.run_id,
                &prepared.binding.attempt_id
            )?
        ))
    }
    pub(super) fn cancellation_state(&self, prepared: &Prepared) -> Result<Option<State>> {
        let name = Self::cancellation_name(prepared)?;
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
        let mapping = ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Authority)?;
        let value: State =
            serde_json::from_value(Value::Object(mapping)).map_err(|_| Error::Authority)?;
        value
            .bind(&self.cancellation_request(prepared)?)
            .map_err(|_| Error::Authority)?;
        Ok(Some(value))
    }
    pub fn cancel(
        &self,
        action: &str,
        confirmation: &str,
        now: &str,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        if !self.enabled {
            return Err(Error::Disabled);
        }
        if confirmation != format!("CANCEL {action}") {
            return Err(Error::Conflict);
        }
        let _lock = self.lock()?;
        let prepared = self.load(action)?;
        self.check(&prepared)?;
        let guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        let request = self.cancellation_request(&prepared)?;
        let state = self
            .cancellation_state(&prepared)?
            .map(Ok)
            .unwrap_or_else(|| State::new(&request, now).map_err(|_| Error::Authority))?;
        if state.termination_verified {
            return self.get(action, now);
        }
        let name = Self::cancellation_name(&prepared)?;
        self.slots
            .replace_regular(
                OsStr::new(&name),
                &serde_json::to_vec(&state).map_err(|_| Error::Authority)?,
            )
            .map_err(|_| Error::Uncertain)?;
        let backend = self
            .backends
            .values()
            .find(|b| b.profile.target == request.intent().target)
            .ok_or(Error::Authority)?;
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        self.check(&prepared)?;
        let outcome = call(&backend.ssh, &request, cancelled).map_err(|_| Error::Uncertain)?;
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Uncertain)?;
        self.check(&prepared).map_err(|_| Error::Uncertain)?;
        if let CallOutcome::Receipt { reply, .. } = outcome {
            let next = state
                .observed(&request, &reply)
                .map_err(|_| Error::Uncertain)?;
            self.slots
                .replace_regular(
                    OsStr::new(&name),
                    &serde_json::to_vec(&next).map_err(|_| Error::Uncertain)?,
                )
                .map_err(|_| Error::Uncertain)?;
        }
        // An uncertain response retains REQUESTED; never invent CANCELLED.
        self.get(action, now)
    }
}
