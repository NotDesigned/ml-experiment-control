//! Durable native submission execution. The caller retains and supplies checks
//! for its Run/deletion/source guards. No bootstrap, authorization, Run selection
//! or automatic provider replay is performed here.
use super::{CallOutcome, SshConfig, call};
use crate::operations::{self, ClaimContext, Store};
use ml_exp_core::{
    operations::{Dispatch, FrozenFiles, Policy},
    slurm::{
        action::{self, Outcome, Uncertain},
        helper::{Reply, ReplyOutcome, Request},
    },
};
use serde_json::Value;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("native Slurm Action binding is invalid; no replay is permitted")]
    Binding,
    #[error("native Slurm execution authority is unavailable; no replay is permitted")]
    Guard,
    #[error("native Slurm execution context is unavailable")]
    Context,
    #[error("native Slurm execution was cancelled before claiming the Action")]
    Cancelled,
    #[error(transparent)]
    Action(#[from] operations::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
pub struct Execute<'a> {
    pub action_id: &'a str,
    pub confirmation: &'a str,
    pub policy: &'a Policy,
    pub files: &'a FrozenFiles,
    pub claim: ClaimContext<'a>,
}
pub struct Runner<'a> {
    pub actions: &'a Store,
    pub ssh: &'a SshConfig,
}
impl Runner<'_> {
    /// A successful claim is durably committed before the first RPC. A caller
    /// timeout cannot roll it back. Repeated execution must reconcile instead.
    pub fn execute(
        &self,
        prepare: &Request,
        command: Execute<'_>,
        guard: &dyn Fn() -> Result<()>,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        self.validate(prepare)?;
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        guard()?;
        let snapshot = self.actions.observe(command.action_id)?;
        action::validate(&snapshot, prepare).map_err(|_| Error::Binding)?;
        guard()?;
        self.actions
            .bind_slurm_request(command.action_id, prepare)?;
        let claimed = self.actions.claim(
            command.action_id,
            command.confirmation,
            command.policy,
            command.files,
            command.claim,
        )?;
        let Some(dispatch) = claimed.dispatch else {
            return Ok(claimed.snapshot);
        };
        if dispatch != Dispatch::Scheduler {
            return Err(Error::Binding);
        }
        let snapshot = claimed.snapshot;
        action::validate(&snapshot, prepare).map_err(|_| Error::Binding)?;
        let deadline = Instant::now() + self.ssh.budget.timeout;
        let outcome = if guard().is_err() {
            Outcome::Uncertain {
                job_id: None,
                reason: Uncertain::Guard,
            }
        } else {
            match self.rpc(prepare, deadline, guard, cancelled) {
                Some(Reply {
                    outcome: ReplyOutcome::Prepared,
                    ..
                }) => {
                    let submit =
                        Request::submit(prepare.intent().clone()).map_err(|_| Error::Binding)?;
                    self.after_receipt(
                        prepare,
                        self.rpc(&submit, deadline, guard, cancelled),
                        Uncertain::Submission,
                        deadline,
                        guard,
                        cancelled,
                    )
                }
                reply => self.after_receipt(
                    prepare,
                    reply,
                    Uncertain::Preparation,
                    deadline,
                    guard,
                    cancelled,
                ),
            }
        };
        self.persist(&snapshot, prepare, outcome, guard)
    }
    /// Reconciliation sends INSPECT/OBSERVE only. PREPARED or a missing ledger
    /// cannot fence a delayed submit, so neither permits another mutation.
    pub fn reconcile(
        &self,
        action_id: &str,
        guard: &dyn Fn() -> Result<()>,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        guard()?;
        let prepare = self.actions.slurm_request(action_id)?;
        let prepare = &prepare;
        self.validate(prepare)?;
        let snapshot = self.actions.observe(action_id)?;
        action::validate(&snapshot, prepare).map_err(|_| Error::Binding)?;
        match snapshot["execution"]["status"].as_str() {
            Some("VERIFIED" | "EXECUTING") => return Ok(snapshot),
            Some("RECONCILE_REQUIRED") => {}
            _ => return Err(Error::Binding),
        }
        let deadline = Instant::now() + self.ssh.budget.timeout;
        let inspect = Request::inspect(prepare.intent().clone()).map_err(|_| Error::Binding)?;
        let outcome = self.after_receipt(
            prepare,
            self.rpc(&inspect, deadline, guard, cancelled),
            Uncertain::Ledger,
            deadline,
            guard,
            cancelled,
        );
        self.persist(&snapshot, prepare, outcome, guard)
    }
    fn validate(&self, prepare: &Request) -> Result<()> {
        self.ssh.validate().map_err(|_| Error::Binding)?;
        prepare.validate().map_err(|_| Error::Binding)?;
        if prepare.prepared_content().is_none() || prepare.intent().target != self.ssh.target {
            return Err(Error::Binding);
        }
        Ok(())
    }
    fn rpc(
        &self,
        request: &Request,
        deadline: Instant,
        guard: &dyn Fn() -> Result<()>,
        cancelled: &AtomicBool,
    ) -> Option<Reply> {
        if guard().is_err() || cancelled.load(Ordering::Relaxed) {
            return None;
        }
        let remaining = deadline.checked_duration_since(Instant::now())?;
        if remaining.is_zero() {
            return None;
        }
        let mut ssh = self.ssh.clone();
        ssh.budget.timeout = remaining;
        let outcome = call(&ssh, request, cancelled).ok()?;
        if guard().is_err() {
            return None;
        }
        match outcome {
            CallOutcome::Receipt { reply, .. } => Some(reply),
            CallOutcome::Uncertain { .. } => None,
        }
    }
    fn after_receipt(
        &self,
        prepare: &Request,
        reply: Option<Reply>,
        unknown: Uncertain,
        deadline: Instant,
        guard: &dyn Fn() -> Result<()>,
        cancelled: &AtomicBool,
    ) -> Outcome {
        let Some(Reply {
            outcome: ReplyOutcome::Submitted { job_id },
            ..
        }) = reply
        else {
            return Outcome::Uncertain {
                job_id: None,
                reason: unknown,
            };
        };
        let Ok(observe) = Request::observe_job(prepare.intent().clone(), job_id.clone()) else {
            return Outcome::Uncertain {
                job_id: None,
                reason: Uncertain::Observation,
            };
        };
        if let Some(reply) = self.rpc(&observe, deadline, guard, cancelled) {
            if matches!(reply.outcome, ReplyOutcome::JobObserved { .. }) {
                return Outcome::Owned { job_id, reply };
            }
        }
        Outcome::Uncertain {
            job_id: Some(job_id),
            reason: Uncertain::Observation,
        }
    }
    fn persist(
        &self,
        snapshot: &Value,
        prepare: &Request,
        outcome: Outcome,
        guard: &dyn Fn() -> Result<()>,
    ) -> Result<Value> {
        let outcome = if guard().is_err() {
            let job_id = match outcome {
                Outcome::Owned { job_id, .. } => Some(job_id),
                Outcome::Uncertain { job_id, .. } => job_id,
            };
            Outcome::Uncertain {
                job_id,
                reason: Uncertain::Guard,
            }
        } else {
            outcome
        };
        let (now, id) = crate::projects::transition_context().map_err(|_| Error::Context)?;
        let transition =
            action::finish(snapshot, prepare, outcome, &now).map_err(|_| Error::Binding)?;
        let action_id = snapshot["action_id"].as_str().ok_or(Error::Binding)?;
        Ok(self.actions.commit(action_id, &transition, &id, &now)?)
    }
}
