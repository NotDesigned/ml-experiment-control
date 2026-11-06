//! The one decision about which backend owns a Run or an Action. It sits above
//! each backend's service and reaches a provider only by calling those services.
//! A failed read is an answer of its own: it never falls through to a backend.
use crate::{
    api::Api,
    backends::{cloud::service as cloud, slurm::service as slurm},
    error::{Error, error},
};
use ml_exp_core::records::{ActionKind, RunKind};
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, atomic::AtomicBool};

pub enum Owner {
    Slurm(Arc<slurm::Service>),
    Cloud(Arc<cloud::Service>),
}
type Cloud = Option<Arc<cloud::Service>>;
type Slurm = Option<Arc<slurm::Service>>;

fn cloud_absent() -> Error {
    error(
        501,
        "CLOUD_NOT_CONFIGURED",
        "cloud submission service is not configured",
    )
}
fn slurm_absent() -> Error {
    error(
        501,
        "SUBMISSION_NOT_CONFIGURED",
        "native submissions are not explicitly configured",
    )
}
impl Owner {
    /// The Action store is shared, so either configured service can read it. An
    /// unknown Action keeps the reading service's own answer.
    pub fn of_action(cloud: Cloud, slurm: Slurm, action: &str) -> Result<Self, Error> {
        let kind = match (&cloud, &slurm) {
            (Some(service), _) => service.action_kind(action)?,
            (None, Some(service)) => service.action_kind(action)?,
            (None, None) => return Err(slurm_absent()),
        };
        Self::of_action_kind(kind, cloud, slurm)
    }
    fn of_action_kind(kind: ActionKind, cloud: Cloud, slurm: Slurm) -> Result<Self, Error> {
        match kind {
            ActionKind::Cloud => cloud.map(Self::Cloud).ok_or_else(cloud_absent),
            ActionKind::Slurm => slurm.map(Self::Slurm).ok_or_else(slurm_absent),
            ActionKind::Other => Err(error(
                404,
                "UNKNOWN_SUBMISSION",
                "Action is not an experiment Submission",
            )),
        }
    }
    /// Its only failure is an absent service, which a read-only view can tolerate.
    /// A Local or Authored Run reaches Slurm, whose admission refuses it by format.
    pub(crate) fn of_kind(kind: RunKind, cloud: Cloud, slurm: Slurm) -> Result<Self, Error> {
        match kind {
            RunKind::Cloud => cloud.map(Self::Cloud).ok_or_else(cloud_absent),
            RunKind::Slurm | RunKind::Local | RunKind::Authored => {
                slurm.map(Self::Slurm).ok_or_else(slurm_absent)
            }
        }
    }
    /// The owner, once it has confirmed the workspace, the deletion fence and an
    /// active Project. Every Run route passes this before any effect.
    pub fn of_run(
        kind: RunKind,
        cloud: Cloud,
        slurm: Slurm,
        project: &str,
        run: &str,
    ) -> Result<Self, Error> {
        let owner = Self::of_kind(kind, cloud, slurm)?;
        owner.gate(project, run)?;
        Ok(owner)
    }
    pub(crate) fn gate(&self, project: &str, run: &str) -> Result<(), Error> {
        match self {
            Self::Cloud(service) => service.validate_run(project, run)?,
            Self::Slurm(service) => service.validate_workspace(project, run)?,
        }
        Ok(())
    }
}
impl Api {
    pub fn action_owner(&self, action: &str) -> Result<Owner, Error> {
        Owner::of_action(self.cloud(), self.submissions(), action)
    }
    pub fn run_owner(&self, project: &str, run: &str) -> Result<Owner, Error> {
        Owner::of_run(
            self.run_kind(project, run)?,
            self.cloud(),
            self.submissions(),
            project,
            run,
        )
    }
    /// For routes only the cloud service has. This chooses nothing.
    pub fn cloud_only(&self) -> Result<Arc<cloud::Service>, Error> {
        self.cloud().ok_or_else(cloud_absent)
    }
    /// For routes only the Slurm service has. This chooses nothing.
    pub fn slurm_only(&self) -> Result<Arc<slurm::Service>, Error> {
        self.submissions().ok_or_else(slurm_absent)
    }
}

/// The clock and identities of one operator request.
#[derive(Clone, Copy)]
pub struct Context<'a> {
    pub now: &'a str,
    pub expires: &'a str,
    pub transition_id: &'a str,
}
pub enum Operation {
    DownloadTicket(u64),
    Sync(ml_exp_core::cloud::exchange::Sync),
    Authorize(String),
    Renew(ml_exp_core::renewal::Request),
    Execute(String),
    Cancel(String),
    Retry(ml_exp_core::first_submission::native::retry::Request),
    Reconcile,
    Refresh,
    History,
    Checkpoint,
    Collect,
    Acknowledge(Acknowledge),
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Acknowledge {
    pub proof: ml_exp_core::artifacts::DownloadProof,
    pub release: bool,
}
/// One operator change to an existing Submission, on the backend that owns it.
/// The two arms are written out on purpose: each operation keeps its own call and
/// its own ordering rules inside that backend's service.
pub fn apply(
    owner: Owner,
    action: &str,
    operation: Operation,
    context: Context<'_>,
) -> Result<Value, Error> {
    let cancelled = AtomicBool::new(false);
    let Context {
        now,
        transition_id: transition,
        ..
    } = context;
    match owner {
        Owner::Cloud(cloud) => match operation {
            Operation::DownloadTicket(part) => cloud.download_ticket(action, part, now),
            Operation::Sync(request) => cloud.synchronize(action, &request, now, &cancelled),
            Operation::Authorize(reason) => cloud.authorize(action, &reason, now, transition),
            Operation::Renew(request) => cloud.renew(action, request, context),
            Operation::Execute(confirmation) => {
                cloud.execute(action, &confirmation, now, transition, &cancelled)
            }
            Operation::Cancel(confirmation) => cloud.cancel(action, &confirmation, now, &cancelled),
            Operation::Retry(request) => cloud.retry(action, request, now, &cancelled),
            Operation::Reconcile => cloud.reconcile(action, now, transition, &cancelled),
            Operation::Refresh => cloud.refresh(action, now, &cancelled),
            // Counts of what the worker already pushed. Nothing is fetched.
            Operation::History => cloud.history(action),
            // The newest checkpoint the worker registered. Nothing is fetched.
            Operation::Checkpoint => cloud.latest_registered_checkpoint(action),
            Operation::Collect => cloud.collect_artifacts(action, now, &cancelled),
            Operation::Acknowledge(request) => {
                cloud.acknowledge_artifacts(action, request.proof, request.release, now)
            }
        }
        .map_err(Into::into),
        Owner::Slurm(slurm) => {
            let refreshing = matches!(
                &operation,
                Operation::History
                    | Operation::Refresh
                    | Operation::Checkpoint
                    | Operation::Collect
                    | Operation::Acknowledge(_)
            );
            match operation {
                Operation::DownloadTicket(_) => {
                    return Err(error(
                        422,
                        "DIRECT_DOWNLOAD_UNAVAILABLE",
                        "this backend streams downloads through its gateway",
                    ));
                }
                Operation::Sync(_) => {
                    return Err(error(
                        501,
                        "CLOUD_SYNC_NOT_CONFIGURED",
                        "CPU cloud synchronization is not configured for this submission",
                    ));
                }
                Operation::Authorize(reason) => {
                    slurm.authorize(action, "bearer-client", &reason, context)
                }
                Operation::Renew(request) => slurm.renew(action, request, context),
                Operation::Execute(confirmation) => {
                    slurm.execute(action, &confirmation, context, &cancelled)
                }
                Operation::Cancel(confirmation) => {
                    slurm.cancel(action, &confirmation, now, &cancelled)
                }
                Operation::Retry(request) => slurm.retry(action, request, now, &cancelled),
                Operation::Reconcile => slurm.reconcile(action, now, &cancelled),
                Operation::Refresh => slurm.refresh(action, now, &cancelled),
                // Pulls new metric lines over the gateway, because no worker pushes them.
                Operation::History => slurm.ingest_history(action, &cancelled),
                // Verifies the backend's ready marker before registering a checkpoint.
                Operation::Checkpoint => slurm.refresh_checkpoint(action, now, &cancelled),
                Operation::Collect => slurm.collect_artifacts(action, &cancelled),
                Operation::Acknowledge(request) => slurm.acknowledge_artifacts(
                    action,
                    request.proof,
                    request.release,
                    now,
                    &cancelled,
                ),
            }
            .map_err(|cause| slurm_failure(refreshing, cause))
        }
    }
}
fn slurm_failure(refreshing: bool, cause: slurm::Error) -> Error {
    if refreshing && matches!(cause, slurm::Error::Uncertain) {
        error(
            503,
            "WORKER_OBSERVATION_CACHE_UNCERTAIN",
            "the sample may have been saved; query it or repeat refresh; no scheduler effect was requested",
        )
    } else {
        cause.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_slurm_refresh_reports_an_uncertain_sample_instead_of_an_uncertain_effect() {
        let refreshed = slurm_failure(true, slurm::Error::Uncertain);
        assert_eq!(
            (refreshed.status, refreshed.code),
            (503, "WORKER_OBSERVATION_CACHE_UNCERTAIN")
        );
        let effect = slurm_failure(false, slurm::Error::Uncertain);
        let public = Error::from(slurm::Error::Uncertain);
        assert_eq!((effect.status, effect.code), (public.status, public.code));
        let other = slurm_failure(true, slurm::Error::Busy);
        assert_eq!((other.status, other.code), (503, "SUBMISSION_BUSY"));
    }

    fn answer(result: Result<Owner, Error>) -> (u16, &'static str) {
        let error = result.err().unwrap();
        (error.status, error.code)
    }
    #[test]
    fn an_absent_owner_answers_501_and_an_action_with_no_owner_is_unknown() {
        for (kind, expected) in [
            (ActionKind::Cloud, (501, "CLOUD_NOT_CONFIGURED")),
            (ActionKind::Slurm, (501, "SUBMISSION_NOT_CONFIGURED")),
            (ActionKind::Other, (404, "UNKNOWN_SUBMISSION")),
        ] {
            assert_eq!(answer(Owner::of_action_kind(kind, None, None)), expected);
        }
        for (kind, expected) in [
            (RunKind::Cloud, (501, "CLOUD_NOT_CONFIGURED")),
            (RunKind::Slurm, (501, "SUBMISSION_NOT_CONFIGURED")),
            (RunKind::Local, (501, "SUBMISSION_NOT_CONFIGURED")),
            (RunKind::Authored, (501, "SUBMISSION_NOT_CONFIGURED")),
        ] {
            assert_eq!(answer(Owner::of_kind(kind, None, None)), expected);
        }
        assert_eq!(
            answer(Owner::of_action(None, None, "action-1111111111111111")),
            (501, "SUBMISSION_NOT_CONFIGURED")
        );
    }
}
