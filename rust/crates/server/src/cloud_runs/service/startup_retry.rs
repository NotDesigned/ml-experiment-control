//! A fresh CPU observation fences a checkpoint-free restart. A retained
//! pre-terminal packet alone cannot prove that training never started.
use super::*;
use ml_exp_core::{
    cloud::{exchange::Sync, retry::StartupProof},
    first_submission::native::retry::Request,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fence {
    parent_action: String,
    parent_binding: Binding,
    parent_job_sha256: String,
    termination: ml_exp_core::cloud::JobObservation,
    sync: Sync,
    baseline_sequence: u64,
    fenced_at: String,
}
fn unavailable(code: &str) -> cloud::Error {
    cloud::Error {
        code: code.into(),
        http_status: Some(409),
        uncertain: false,
    }
}
fn name(action: &str, sync: &str) -> String {
    format!("{action}.startup-{sync}.json")
}
impl Service {
    pub(super) fn fence_startup_sync(
        &self,
        prepared: &Prepared,
        request: &Sync,
        lock: &Lock<'_>,
        now: &str,
        cancelled: &AtomicBool,
    ) -> cloud::Result<()> {
        let Some(saved) = self.exchange_saved(prepared)? else {
            return Ok(());
        };
        if saved.evidence.worker.phase != ml_exp_core::managed::evidence::Phase::Prepared
            || !saved.inputs.iter().any(|r| {
                r.phase == ml_exp_core::input_progress::Phase::Failed
                    && r.code.as_deref() == Some("DATA_CACHE_UNAVAILABLE")
            })
        {
            return Ok(());
        }
        // Only PAI's single Worker/RestartPolicy=Never launch is admitted here.
        // Other backends retain checkpoint retry until equivalent proof exists.
        if !matches!(
            self.backends[&prepared.execution.backend_id]
                .adapter
                .profile,
            cloud::adapter::Profile::Pai(_)
        ) {
            return Ok(());
        }
        let action = prepared.plan["action_id"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        let path = name(action, &request.request_id);
        if self.slots.entry_exists(OsStr::new(&path)).map_err(io)? {
            return Ok(());
        }
        let sync_job = request
            .job(&prepared.execution.job)
            .map_err(|_| cloud::Error::evidence())?;
        if self
            .sync_outbox
            .optional_status(&sync_job)?
            .is_some_and(|s| s["phase"] != "PREPARED")
        {
            // An already-started sync cannot be relabelled as a fresh probe.
            return Ok(());
        }
        let adapter = self.execution_adapter(&prepared.execution)?;
        let state = self.outbox.reconcile(
            &adapter,
            &prepared.execution.job,
            prepared.execution.allocation,
            &|| {
                lock.validate()?;
                self.check(prepared)
            },
            cancelled,
        )?;
        let Ok(termination) = serde_json::from_value::<ml_exp_core::cloud::JobObservation>(
            state["observation"].clone(),
        ) else {
            return Ok(());
        };
        if state["diagnostic"].is_null()
            && state["handle"]["job_id"] == termination.job_id
            && termination.state.is_terminal()
            && termination.state != ml_exp_core::state::RunState::Succeeded
        {
            let saved = self
                .exchange_saved(prepared)?
                .ok_or_else(cloud::Error::evidence)?;
            if saved.evidence.worker.phase != ml_exp_core::managed::evidence::Phase::Prepared {
                return Ok(());
            }
            let fence = Fence {
                parent_action: action.into(),
                parent_binding: prepared.execution.binding.clone(),
                parent_job_sha256: cloud::adapter::job_digest(&prepared.execution.job)?,
                termination,
                sync: request.clone(),
                baseline_sequence: saved.sequence,
                fenced_at: now.into(),
            };
            lock.validate()?;
            write(&self.slots, &path, &fence)?;
        }
        Ok(())
    }
    pub(super) fn startup_proof(
        &self,
        prepared: &Prepared,
        request: &Request,
        termination: &ml_exp_core::cloud::JobObservation,
        lock: &Lock<'_>,
        cancelled: &AtomicBool,
    ) -> cloud::Result<StartupProof> {
        let start = request
            .restart_from
            .as_ref()
            .ok_or_else(cloud::Error::evidence)?;
        let action = prepared.plan["action_id"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        let path = name(action, &start.sync_request_id);
        if !self.slots.entry_exists(OsStr::new(&path)).map_err(io)? {
            return Err(unavailable("STARTUP_RETRY_SYNC_REQUIRED"));
        }
        let fence: Fence = serde_json::from_value(read(&self.slots, &path)?)
            .map_err(|_| cloud::Error::evidence())?;
        if fence.parent_action != action
            || fence.parent_binding != prepared.execution.binding
            || fence.parent_job_sha256 != cloud::adapter::job_digest(&prepared.execution.job)?
            || fence.termination != *termination
            || fence.sync.request_id != start.sync_request_id
        {
            return Err(cloud::Error::evidence());
        }
        fence
            .sync
            .validate(action)
            .map_err(|_| cloud::Error::evidence())?;
        let collector = self.backends[&prepared.execution.backend_id]
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let job = fence
            .sync
            .job(&prepared.execution.job)
            .map_err(|_| cloud::Error::evidence())?;
        let state = self.sync_outbox.reconcile(
            collector,
            &job,
            ml_exp_core::cloud::pai::Allocation::Guaranteed,
            &|| {
                lock.validate()?;
                self.check(prepared)
            },
            cancelled,
        )?;
        let observation: ml_exp_core::cloud::JobObservation =
            serde_json::from_value(state["observation"].clone())
                .map_err(|_| unavailable("STARTUP_RETRY_SYNC_PENDING"))?;
        if !state["diagnostic"].is_null()
            || state["handle"]["job_id"] != observation.job_id
            || !observation.state.is_terminal()
        {
            return Err(unavailable("STARTUP_RETRY_SYNC_PENDING"));
        }
        let saved = self
            .exchange_saved(prepared)?
            .ok_or_else(|| unavailable("STARTUP_RETRY_EVIDENCE_REQUIRED"))?;
        let package = prepared
            .execution
            .package()
            .map_err(|_| cloud::Error::evidence())?;
        saved
            .evidence
            .validate_config(&package.config)
            .map_err(|_| cloud::Error::evidence())?;
        if saved.evidence.stdout.total_bytes != 0
            || saved.evidence.stderr.total_bytes != 0
            || saved.evidence.metrics.total_bytes != 0
            || saved.cursor.is_some()
            || saved.checkpoint_error.is_some()
        {
            return Err(unavailable("STARTUP_RETRY_TRAINING_MAY_HAVE_STARTED"));
        }
        let proof = StartupProof {
            sync_request_id: start.sync_request_id.clone(),
            baseline_sequence: fence.baseline_sequence,
            sequence: saved.sequence,
            fenced_at: fence.fenced_at,
            observed_at: saved.observed_at,
            packet_sha256: saved.packet_sha256,
            worker: saved.evidence.worker,
            inputs: saved.inputs,
        };
        proof
            .validate(&prepared.execution.binding, request, termination)
            .map_err(|_| unavailable("STARTUP_RETRY_EVIDENCE_REQUIRED"))?;
        Ok(proof)
    }
}
