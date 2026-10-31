//! Cloud continuation is a new immutable Attempt, never a replay of an unknown
//! provider create. The caller supplies a freshly reconciled exact parent job.
use super::*;
use crate::{
    checkpoints,
    first_submission::native::{
        Binding,
        retry::{self, Request},
    },
};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
pub const CONTRACT: &str = "native-cloud-retry.v1";
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartupProof {
    pub sync_request_id: String,
    pub baseline_sequence: u64,
    pub sequence: u64,
    pub fenced_at: String,
    pub observed_at: String,
    pub packet_sha256: String,
    pub worker: crate::managed::evidence::Observation,
    pub inputs: Vec<crate::input_progress::Record>,
}
impl StartupProof {
    pub fn validate(
        &self,
        binding: &Binding,
        request: &Request,
        termination: &JobObservation,
    ) -> Result<()> {
        let start = request
            .restart_from
            .as_ref()
            .ok_or(Error("startup source"))?;
        let fenced = OffsetDateTime::parse(&self.fenced_at, &Rfc3339)
            .map_err(|_| Error("startup fence time"))?;
        let observed = OffsetDateTime::parse(&self.observed_at, &Rfc3339)
            .map_err(|_| Error("startup observation time"))?;
        // There can be one durable unsent packet at the fence. Two subsequent
        // sequence advances prove at least one sample was taken after the
        // confirmed terminal parent; timestamps alone cannot prove freshness.
        if self.sync_request_id != start.sync_request_id
            || self.sequence
                < self
                    .baseline_sequence
                    .checked_add(2)
                    .ok_or(Error("startup sequence"))?
            || observed < fenced
            || !helper::valid_digest(&self.packet_sha256)
            || !helper::valid_digest(&self.worker.manifest_sha256)
            || self.worker.contract != crate::managed::evidence::CONTRACT
            || self.worker.identity.project != binding.project
            || self.worker.identity.run_id != binding.run_id
            || self.worker.identity.attempt_id != binding.attempt_id
            || self.worker.phase != crate::managed::evidence::Phase::Prepared
            || self.worker.receipt.is_some()
            || self.worker.safe_to_retry
            || !matches!(
                termination.state,
                crate::state::RunState::Failed
                    | crate::state::RunState::Cancelled
                    | crate::state::RunState::Preempted
            )
            || self.inputs.is_empty()
            || self.inputs.len() > 32
            || !self.inputs.iter().any(|r| {
                r.phase == crate::input_progress::Phase::Failed
                    && r.code.as_deref() == Some("DATA_CACHE_UNAVAILABLE")
            })
        {
            return Err(Error("unproven startup failure"));
        }
        for input in &self.inputs {
            let line = format!(
                "{}{}",
                crate::input_progress::MARKER,
                serde_json::to_string(input).map_err(|_| Error("startup input"))?
            );
            if crate::input_progress::parse_line(&line).record.as_ref() != Some(input) {
                return Err(Error("startup input"));
            }
        }
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub contract: String,
    pub parent_action: String,
    pub parent_binding: Binding,
    pub parent_job_sha256: String,
    pub termination: JobObservation,
    pub request: Request,
    pub request_sha256: String,
    pub reserved_gpu_seconds: u64,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_failure: Option<StartupProof>,
}
impl Receipt {
    pub fn validate(&self, run: &Value, attempt: &str) -> Result<()> {
        self.request
            .validate(&self.parent_action)
            .map_err(|_| Error("retry request"))?;
        let b = &self.parent_binding;
        let count = retry::number(attempt).map_err(|_| Error("retry count"))?;
        let reserved = run["resources"]["gpus"]
            .as_u64()
            .and_then(|g| g.checked_mul(run["native_cloud"]["walltime_seconds"].as_u64()?))
            .and_then(|s| s.checked_mul(u64::from(count)))
            .ok_or(Error("retry budget"))?;
        if self.contract != CONTRACT
            || retry::next(&b.attempt_id).map_err(|_| Error("retry sequence"))? != attempt
            || run["project"] != b.project
            || run["run_id"] != b.run_id
            || !helper::valid_digest(&self.parent_job_sha256)
            || self.request.parent().map_err(|_| Error("retry parent"))?
                != (b.run_id.as_str(), b.attempt_id.as_str())
            || !self.termination.state.is_terminal()
            || self.termination.job_id.is_empty()
            || self.request_sha256
                != self
                    .request
                    .identity(&self.parent_action)
                    .map_err(|_| Error("retry identity"))?
            || self.reserved_gpu_seconds != reserved
            || reserved as f64 > self.request.max_gpu_hours * 3600.
            || OffsetDateTime::parse(&self.created_at, &Rfc3339).is_err()
            || [
                &b.project_sha256,
                &b.run_sha256,
                &b.attempt_sha256,
                &b.source_receipt_sha256,
                &b.backend_config_sha256,
            ]
            .iter()
            .any(|v| !helper::valid_digest(v))
        {
            return Err(Error("cloud retry authority"));
        }
        match (&self.request.restart_from, &self.startup_failure) {
            (Some(_), Some(proof)) => {
                if !run["resume_from"].is_null() {
                    return Err(Error("startup retry would discard restore source"));
                }
                proof.validate(b, &self.request, &self.termination)?;
            }
            (None, None) => {}
            _ => return Err(Error("startup proof required")),
        }
        Ok(())
    }
}
pub fn plan(run: &Value, receipt: Receipt, checkpoint: &Value) -> Result<Value> {
    let next =
        retry::next(&receipt.parent_binding.attempt_id).map_err(|_| Error("retry sequence"))?;
    receipt.validate(run, &next)?;
    if let Some(reference) = &receipt.request.resume_from {
        let (context, _) =
            checkpoints::validate_document(checkpoint, None, Some(&reference.checkpoint_id))
                .map_err(|_| Error("retry checkpoint"))?;
        if context.project != receipt.parent_binding.project
            || context.run_id != reference.run_id
            || context.attempt_id != reference.attempt_id
            || run["source_id"] != context.source_id
            || run["image_id"] != context.image_id
        {
            return Err(Error("retry checkpoint provenance"));
        }
    } else if !checkpoint.is_null() {
        return Err(Error("startup retry cannot restore a checkpoint"));
    }
    let mut attempt = run.clone();
    attempt["attempt_id"] = json!(next);
    attempt["attempt_created_at"] = json!(receipt.created_at);
    if receipt.request.resume_from.is_some() {
        attempt["resume_from"] = json!(receipt.request.resume_from);
        attempt["native_checkpoint"] = checkpoint.clone();
    }
    attempt["native_cloud_retry"] = json!(receipt);
    Ok(attempt)
}
pub fn validate_attempt(run: &Value, attempt: &Value, binding: &Binding) -> Result<()> {
    let receipt: Receipt = serde_json::from_value(attempt["native_cloud_retry"].clone())
        .map_err(|_| Error("retry receipt"))?;
    receipt.validate(run, &binding.attempt_id)?;
    let b = &receipt.parent_binding;
    if b.project != binding.project
        || b.run_id != binding.run_id
        || b.project_sha256 != binding.project_sha256
        || b.run_sha256 != binding.run_sha256
        || b.source_receipt_sha256 != binding.source_receipt_sha256
        || b.backend_config_sha256 != binding.backend_config_sha256
        || plan(run, receipt, &attempt["native_checkpoint"])? != *attempt
    {
        return Err(Error("retry frozen manifest"));
    }
    Ok(())
}
#[cfg(test)]
mod tests;
