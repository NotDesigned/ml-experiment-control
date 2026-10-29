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
            || self.request.resume_from.run_id != b.run_id
            || self.request.resume_from.attempt_id != b.attempt_id
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
        Ok(())
    }
}
pub fn plan(run: &Value, receipt: Receipt, checkpoint: &Value) -> Result<Value> {
    let next =
        retry::next(&receipt.parent_binding.attempt_id).map_err(|_| Error("retry sequence"))?;
    receipt.validate(run, &next)?;
    let (context, _) = checkpoints::validate_document(
        checkpoint,
        None,
        Some(&receipt.request.resume_from.checkpoint_id),
    )
    .map_err(|_| Error("retry checkpoint"))?;
    if context.project != receipt.parent_binding.project
        || context.run_id != receipt.request.resume_from.run_id
        || context.attempt_id != receipt.request.resume_from.attempt_id
        || run["source_id"] != context.source_id
        || run["image_id"] != context.image_id
    {
        return Err(Error("retry checkpoint provenance"));
    }
    let mut attempt = run.clone();
    attempt["attempt_id"] = json!(next);
    attempt["attempt_created_at"] = json!(receipt.created_at);
    attempt["resume_from"] = json!(receipt.request.resume_from);
    attempt["native_checkpoint"] = checkpoint.clone();
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
