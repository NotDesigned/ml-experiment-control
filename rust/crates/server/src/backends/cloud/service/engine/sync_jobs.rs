//! The one path a synchronous CPU job takes to a provider. Write-ahead ordering
//! stays in the outbox: this only decides between its two entry points.
use super::*;
use ml_exp_core::cloud::{Job, JobObservation, pai::Allocation};

/// A fresh read proving that this exact job ended. A failed read leaves a diagnostic
/// beside the old observation, so that observation proves nothing. An unreadable
/// observation is an error; anything else that falls short returns None.
pub(super) fn proven_terminal(state: &Value) -> cloud::Result<Option<JobObservation>> {
    let observation: JobObservation = serde_json::from_value(state["observation"].clone())
        .map_err(why!(cloud::Error::evidence()))?;
    Ok((state["diagnostic"].is_null()
        && state["handle"]["job_id"] == observation.job_id
        && observation.state.is_terminal())
    .then_some(observation))
}

impl Service {
    /// Prepare durably, then send only if the entry never started. Once started,
    /// a repeated call observes the same job and can never reach a create.
    pub(super) fn submit_once_or_reconcile(
        &self,
        adapter: &Adapter<Shared>,
        job: &Job,
        allocation: Allocation,
        check: &dyn Fn() -> cloud::Result<()>,
        cancelled: &AtomicBool,
    ) -> cloud::Result<Value> {
        self.sync_outbox
            .prepare(&adapter.profile, job, allocation, check)?;
        if self.sync_outbox.status(job)?["retry_safe"] == true {
            self.sync_outbox.execute(
                adapter,
                job,
                allocation,
                &format!("SUBMIT {}", Outbox::slot(job)?),
                check,
                cancelled,
            )
        } else {
            self.sync_outbox
                .reconcile(adapter, job, allocation, check, cancelled)
        }
    }
}
