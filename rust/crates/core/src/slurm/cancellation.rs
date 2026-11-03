//! Pure client-visible cancellation intent. Scheduler effects and durable
//! outbox deduplication belong to the existing exact-job helper boundary.
use super::{StatusObservation, helper};
use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub request_sha256: String,
    pub job_id: String,
    pub requested_at: String,
    pub termination_verified: bool,
    pub observation: Option<StatusObservation>,
}
impl State {
    pub fn new(request: &helper::Request, now: &str) -> helper::Result<Self> {
        let value = Self {
            request_sha256: request.identity()?.request_sha256,
            job_id: request
                .requested_job_id()
                .ok_or(helper::Error::Identity)?
                .into(),
            requested_at: now.into(),
            termination_verified: false,
            observation: None,
        };
        value.bind(request)?;
        Ok(value)
    }
    pub fn bind(&self, request: &helper::Request) -> helper::Result<()> {
        request.validate()?;
        if request.operation() != helper::OperationTag::CancelJob
            || self.request_sha256 != request.identity()?.request_sha256
            || Some(self.job_id.as_str()) != request.requested_job_id()
            || OffsetDateTime::parse(&self.requested_at, &Rfc3339).is_err()
        {
            return Err(helper::Error::Identity);
        }
        self.validate()
    }
    pub fn validate(&self) -> helper::Result<()> {
        if !helper::valid_digest(&self.request_sha256)
            || super::JobId::parse(&self.job_id).is_err()
            || OffsetDateTime::parse(&self.requested_at, &Rfc3339).is_err()
        {
            return Err(helper::Error::Identity);
        }
        if let Some(observation) = &self.observation {
            observation
                .status
                .validate()
                .map_err(|_| helper::Error::Identity)?;
            if observation.status.job_id.as_str() != self.job_id
                || observation.retained_previous
                || observation.status.state.is_terminal() != self.termination_verified
            {
                return Err(helper::Error::Identity);
            }
        } else if self.termination_verified {
            return Err(helper::Error::Identity);
        }
        Ok(())
    }
    pub fn observed(
        &self,
        request: &helper::Request,
        reply: &helper::Reply,
    ) -> helper::Result<Self> {
        self.bind(request)?;
        if reply.identity != request.identity()? {
            return Err(helper::Error::Identity);
        }
        let mut next = self.clone();
        if let helper::ReplyOutcome::JobObserved { observation } = &reply.outcome {
            if !observation.retained_previous
                && (!self.termination_verified
                    || self.observation.as_ref() == Some(observation.as_ref()))
            {
                next.termination_verified = observation.status.state.is_terminal();
                next.observation = Some(observation.as_ref().clone());
            }
        }
        next.bind(request)?;
        Ok(next)
    }
}
