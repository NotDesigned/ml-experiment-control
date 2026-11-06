//! The one path a synchronous CPU job takes to a provider. Write-ahead ordering
//! stays in the outbox: this only decides between its two entry points.
use super::*;
use ml_exp_core::cloud::{Job, pai::Allocation};

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
