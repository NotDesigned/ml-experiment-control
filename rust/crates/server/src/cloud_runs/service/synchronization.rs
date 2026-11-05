//! Explicit one-child CPU recovery retains both the failed job and its outbox.
//! Local GET is read-only; only a confirmed retry POST can create the child.
use super::*;
use ml_exp_core::cloud::exchange::{Sync, SyncRetry};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Retry {
    action: String,
    request_id: String,
    request: SyncRetry,
    original: ml_exp_core::cloud::JobObservation,
    training: ml_exp_core::cloud::JobObservation,
    created_at: String,
    job: ml_exp_core::cloud::Job,
}
fn slot(action: &str, request_id: &str) -> cloud::Result<String> {
    Sync {
        request_id: request_id.into(),
        confirmation: format!("SYNC {action}"),
    }
    .validate(action)
    .map_err(|_| cloud::Error::configuration())?;
    Ok(format!(
        "sync-retry-{}.json",
        canonical::digest(&json!([action, request_id])).map_err(|_| cloud::Error::evidence())?
    ))
}
fn failed() -> cloud::Error {
    cloud::Error {
        code: "SYNC_RETRY_NOT_ELIGIBLE".into(),
        http_status: Some(409),
        uncertain: false,
    }
}
fn observation(status: &Value) -> cloud::Result<ml_exp_core::cloud::JobObservation> {
    let evidence: ml_exp_core::cloud::JobObservation =
        serde_json::from_value(status["observation"].clone()).map_err(|_| failed())?;
    if !status["diagnostic"].is_null() || status["handle"]["job_id"] != evidence.job_id {
        return Err(failed());
    }
    Ok(evidence)
}
impl Service {
    pub(super) fn sync_recovery_job(
        &self,
        prepared: &Prepared,
        action: &str,
        request_id: &str,
    ) -> cloud::Result<ml_exp_core::cloud::Job> {
        if let Some(saved) = self.sync_retry_saved(prepared, action, request_id)? {
            return Ok(saved.job);
        }
        Sync {
            request_id: request_id.into(),
            confirmation: format!("SYNC {action}"),
        }
        .job(&prepared.execution.job)
        .map_err(|_| cloud::Error::evidence())
    }
    pub(super) fn sync_recovery_request(
        &self,
        action: &str,
        request_id: &str,
    ) -> cloud::Result<Option<SyncRetry>> {
        Ok(self
            .sync_retry_saved(&self.load(action)?, action, request_id)?
            .map(|s| s.request))
    }
    fn sync_retry_saved(
        &self,
        prepared: &Prepared,
        action: &str,
        request_id: &str,
    ) -> cloud::Result<Option<Retry>> {
        let name = slot(action, request_id)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(None);
        }
        let saved: Retry = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(|_| cloud::Error::evidence())?;
        let job = saved
            .request
            .job(action, request_id, &prepared.execution.job)
            .map_err(|_| cloud::Error::evidence())?;
        if saved.action != action
            || saved.request_id != request_id
            || saved.job != job
            || saved.original.state != ml_exp_core::state::RunState::Failed
            || saved.original.job_id != saved.request.expected_failed_job_id
            || !saved.training.state.is_terminal()
        {
            return Err(cloud::Error::evidence());
        }
        Ok(Some(saved))
    }
    pub fn sync_status(&self, action: &str, request_id: &str, now: &str) -> cloud::Result<Value> {
        let _lock = self.lock()?;
        let prepared = self.load(action)?;
        self.check(&prepared)?;
        let request = Sync {
            request_id: request_id.into(),
            confirmation: format!("SYNC {action}"),
        };
        request
            .validate(action)
            .map_err(|_| cloud::Error::configuration())?;
        let original = request
            .job(&prepared.execution.job)
            .map_err(|_| cloud::Error::evidence())?;
        let original_status = self
            .sync_outbox
            .optional_status(&original)?
            .ok_or_else(missing)?;
        let retry = self.sync_retry_saved(&prepared, action, request_id)?;
        let retry_status = retry
            .as_ref()
            .map(|r| self.sync_outbox.optional_status(&r.job))
            .transpose()?
            .flatten();
        let mut value = self.get(action, now)?;
        value["synchronization"] = json!({"request_id":request_id,"resources":{"gpus":0,"cpus":2,"memory_gb":4},"max_time_seconds":600,"training_replayed":false,
            "original_job":original_status,"job":retry_status.as_ref().unwrap_or(&original_status),"retry_limit":1,
            "retry":retry.as_ref().map(|r|json!({"expected_failed_job_id":r.request.expected_failed_job_id,"created_at":r.created_at,"job":retry_status,"attempt":1})),
            "source":"retained-sync-evidence","remote_refresh_performed":false});
        Ok(value)
    }
    pub fn retry_sync(
        &self,
        action: &str,
        request_id: &str,
        request: &SyncRetry,
        now: &str,
        cancelled: &AtomicBool,
    ) -> cloud::Result<Value> {
        request
            .validate(action, request_id)
            .map_err(|_| cloud::Error::configuration())?;
        if !self.enabled || self.exchange.is_none() {
            return Err(cloud::Error::configuration());
        }
        let lock = self.lock()?;
        let prepared = self.load(action)?;
        if self.actions.observe(action).map_err(op)?["execution"]["status"] != "VERIFIED" {
            return Err(failed());
        }
        let collector = self.backends[&prepared.execution.backend_id]
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let guard = self
            .fence
            .existing_observation_guard(
                &prepared.execution.job.project,
                &prepared.execution.job.run_id,
            )
            .map_err(|_| cloud::Error::evidence())?;
        let check = || {
            lock.validate()?;
            guard
                .require_observed_live(
                    &prepared.execution.job.project,
                    &prepared.execution.job.run_id,
                )
                .map_err(|_| cloud::Error::evidence())?;
            self.check(&prepared)
        };
        let allocation = ml_exp_core::cloud::pai::Allocation::Guaranteed;
        let job = request
            .job(action, request_id, &prepared.execution.job)
            .map_err(|_| cloud::Error::configuration())?;
        if let Some(saved) = self.sync_retry_saved(&prepared, action, request_id)? {
            if saved.request != *request {
                return Err(failed());
            }
        } else {
            let adapter = self.execution_adapter(&prepared.execution)?;
            let training = observation(&self.outbox.reconcile(
                &adapter,
                &prepared.execution.job,
                prepared.execution.allocation,
                &check,
                cancelled,
            )?)?;
            if !training.state.is_terminal() {
                return Err(failed());
            }
            let original = Sync {
                request_id: request_id.into(),
                confirmation: format!("SYNC {action}"),
            }
            .job(&prepared.execution.job)
            .map_err(|_| cloud::Error::configuration())?;
            if self.sync_outbox.optional_status(&original)?.is_none() {
                return Err(missing());
            }
            let original = observation(
                &self
                    .sync_outbox
                    .reconcile(collector, &original, allocation, &check, cancelled)?,
            )?;
            if original.state != ml_exp_core::state::RunState::Failed
                || original.job_id != request.expected_failed_job_id
            {
                return Err(failed());
            }
            let saved = Retry {
                action: action.into(),
                request_id: request_id.into(),
                request: request.clone(),
                original,
                training,
                created_at: now.into(),
                job: job.clone(),
            };
            check()?;
            write(&self.slots, &slot(action, request_id)?, &saved)?;
        }
        self.sync_outbox
            .prepare(&collector.profile, &job, allocation, &check)?;
        let state = self.sync_outbox.status(&job)?;
        if state["retry_safe"] == true {
            self.sync_outbox.execute(
                collector,
                &job,
                allocation,
                &format!("SUBMIT {}", Outbox::slot(&job)?),
                &check,
                cancelled,
            )?;
        } else {
            self.sync_outbox
                .reconcile(collector, &job, allocation, &check, cancelled)?;
        }
        drop(lock);
        self.sync_status(action, request_id, now)
    }
}
