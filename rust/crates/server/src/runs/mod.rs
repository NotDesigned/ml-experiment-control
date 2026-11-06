//! The one decision about which backend owns a Run or an Action. It sits above
//! each backend's service and reaches a provider only by calling those services.
//! A failed read is an answer of its own: it never falls through to a backend.
use crate::{
    cloud_runs,
    error::{Error, error},
    read_api::Api,
    submissions,
};
use ml_exp_core::records::{ActionKind, RunKind};
use std::sync::Arc;

pub enum Owner {
    Slurm(Arc<submissions::Service>),
    Cloud(Arc<cloud_runs::Service>),
}
type Cloud = Option<Arc<cloud_runs::Service>>;
type Slurm = Option<Arc<submissions::Service>>;

fn cloud_absent() -> Error {
    error(
        501,
        "CLOUD_NOT_CONFIGURED",
        "cloud submission service is not configured",
    )
}
pub(crate) fn slurm_absent() -> Error {
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
    pub fn cloud_only(&self) -> Result<Arc<cloud_runs::Service>, Error> {
        self.cloud().ok_or_else(cloud_absent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
