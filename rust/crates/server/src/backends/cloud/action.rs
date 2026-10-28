//! Existing reviewed Actions execute native cloud effects. The caller supplies
//! its frozen Run/source/deletion guard; this adapter cannot choose a new Run.
use super::{
    Error, Result,
    adapter::{Adapter, Transport},
    outbox::Outbox,
};
use crate::operations::{Execute, Store};
use ml_exp_core::{
    cloud::action::{self, Execution, Outcome},
    operations::Dispatch,
};
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
pub struct Runner<'a, T> {
    pub actions: &'a Store,
    pub adapter: &'a Adapter<T>,
    pub outbox: &'a Outbox,
}
fn action_error(_: crate::operations::Error) -> Error {
    Error {
        code: "CLOUD_ACTION_UNAVAILABLE".into(),
        http_status: None,
        uncertain: false,
    }
}
impl<T: Transport> Runner<'_, T> {
    fn validate(&self, snapshot: &Value, request: &Execution) -> Result<()> {
        action::validate(snapshot, request).map_err(|_| Error::evidence())?;
        if request.profile_sha256 != self.adapter.profile.digest()? {
            return Err(Error::evidence());
        }
        Ok(())
    }
    pub fn execute(
        &self,
        request: &Execution,
        command: Execute<'_>,
        guard: &dyn Fn() -> Result<()>,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::transport(false));
        }
        guard()?;
        let snapshot = self
            .actions
            .observe(command.action_id)
            .map_err(action_error)?;
        self.validate(&snapshot, request)?;
        self.actions
            .bind_cloud_request(command.action_id, request)
            .map_err(action_error)?;
        let prepared = self.outbox.prepare(
            &self.adapter.profile,
            &request.job,
            request.allocation,
            guard,
        )?;
        guard()?;
        let claimed = self
            .actions
            .claim(
                command.action_id,
                command.confirmation,
                command.policy,
                command.files,
                command.claim,
            )
            .map_err(action_error)?;
        let Some(dispatch) = claimed.dispatch else {
            return Ok(claimed.snapshot);
        };
        if dispatch != Dispatch::Scheduler {
            return Err(Error::evidence());
        }
        self.validate(&claimed.snapshot, request)?;
        let state = guard()
            .and_then(|_| {
                self.outbox.execute(
                    self.adapter,
                    &request.job,
                    request.allocation,
                    prepared["confirmation"]
                        .as_str()
                        .ok_or_else(Error::evidence)?,
                    guard,
                    cancelled,
                )
            })
            .and_then(|_| {
                self.outbox.reconcile(
                    self.adapter,
                    &request.job,
                    request.allocation,
                    guard,
                    cancelled,
                )
            });
        self.persist(&claimed.snapshot, request, state, guard)
    }
    /// A lost or invalid reply never re-enters create, even if ListJobs is empty.
    pub fn reconcile(
        &self,
        action_id: &str,
        guard: &dyn Fn() -> Result<()>,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        guard()?;
        let request = self
            .actions
            .cloud_request(action_id)
            .map_err(action_error)?;
        let snapshot = self.actions.observe(action_id).map_err(action_error)?;
        self.validate(&snapshot, &request)?;
        match snapshot["execution"]["status"].as_str() {
            Some("VERIFIED" | "EXECUTING") => return Ok(snapshot),
            Some("RECONCILE_REQUIRED") => {}
            _ => return Err(Error::evidence()),
        }
        let state = self.outbox.reconcile(
            self.adapter,
            &request.job,
            request.allocation,
            guard,
            cancelled,
        );
        self.persist(&snapshot, &request, state, guard)
    }
    fn persist(
        &self,
        snapshot: &Value,
        request: &Execution,
        state: Result<Value>,
        guard: &dyn Fn() -> Result<()>,
    ) -> Result<Value> {
        let state = state.ok();
        let job_id = state
            .as_ref()
            .and_then(|s| s["handle"]["job_id"].as_str())
            .map(str::to_owned);
        let outcome = if guard().is_ok() {
            match (
                job_id.clone(),
                state
                    .as_ref()
                    .and_then(|s| serde_json::from_value(s["observation"].clone()).ok()),
            ) {
                (Some(job_id), Some(observation)) => Outcome::Owned {
                    job_id,
                    observation,
                },
                _ => Outcome::Uncertain { job_id },
            }
        } else {
            Outcome::Uncertain { job_id }
        };
        let (now, id) = crate::projects::transition_context().map_err(|_| Error::evidence())?;
        let transition =
            action::finish(snapshot, request, outcome, &now).map_err(|_| Error::evidence())?;
        self.actions
            .commit(
                snapshot["action_id"].as_str().ok_or_else(Error::evidence)?,
                &transition,
                &id,
                &now,
            )
            .map_err(action_error)
    }
}
