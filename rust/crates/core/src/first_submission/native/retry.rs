//! Explicit same-Run continuation. No clock, checkpoint selection or scheduler
//! effect is hidden in this module. Every Attempt reserves its full GPU limit.
use super::{Binding, Error, Result, field};
use crate::{canonical, checkpoints, identity, slurm};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

pub const CONTRACT: &str = "native-attempt-retry.v1";
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub confirmation: String,
    pub resume_from: checkpoints::Reference,
    /// Cumulative approval, including all previous Attempt reservations.
    pub max_gpu_hours: f64,
    #[serde(default)]
    pub reason: String,
}
impl Request {
    pub fn validate(&self, action: &str) -> Result<()> {
        crate::operations::validate_submission_id(action).map_err(|_| Error("parent Action"))?;
        self.resume_from
            .validate()
            .map_err(|_| Error("retry checkpoint"))?;
        crate::first_submission::validate_gpu_budget(Some(self.max_gpu_hours), "budget_cap")
            .map_err(|_| Error("cumulative budget"))?;
        if self.confirmation != format!("RETRY {action}")
            || self.reason.chars().count() > 4000
            || self.reason.chars().any(char::is_control)
        {
            return Err(Error("retry confirmation or reason"));
        }
        Ok(())
    }
    pub fn identity(&self, action: &str) -> Result<String> {
        self.validate(action)?;
        canonical::protocol_identity(
            &json!({"contract":CONTRACT,"parent_action":action,"request":self}),
        )
        .map_err(|_| Error("retry request identity"))
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub contract: String,
    pub parent_action: String,
    pub parent_binding: Binding,
    pub parent_job_id: String,
    pub backend_id: String,
    pub termination: slurm::StatusObservation,
    pub request: Request,
    pub request_sha256: String,
    pub reserved_gpu_seconds: u64,
    pub created_at: String,
}

pub fn number(attempt: &str) -> Result<u32> {
    identity::validate_attempt(attempt).map_err(|_| Error("Attempt identity"))?;
    let n: u32 = attempt
        .strip_prefix("attempt-")
        .ok_or(Error("Attempt identity"))?
        .parse()
        .map_err(|_| Error("Attempt identity"))?;
    if n == 0 || n > 256 || format!("attempt-{n:03}") != attempt {
        return Err(Error("Attempt limit"));
    }
    Ok(n)
}
pub fn next(attempt: &str) -> Result<String> {
    let n = number(attempt)?
        .checked_add(1)
        .ok_or(Error("Attempt limit"))?;
    let value = format!("attempt-{n:03}");
    number(&value)?;
    Ok(value)
}
fn reservation(run: &Value, count: u32) -> Result<u64> {
    let gpus = run["resources"]["gpus"]
        .as_u64()
        .ok_or(Error("GPU reservation"))?;
    let seconds = slurm::walltime_seconds(field(&run["backend"], "time")?)
        .map_err(|_| Error("walltime reservation"))?;
    gpus.checked_mul(seconds)
        .and_then(|v| v.checked_mul(u64::from(count)))
        .filter(|v| *v > 0)
        .ok_or(Error("cumulative reservation"))
}
impl Receipt {
    pub fn validate(&self, run: &Value, attempt_id: &str) -> Result<()> {
        let parent = &self.parent_binding;
        let status = &self.termination.status;
        let count = number(attempt_id)?;
        status
            .validate()
            .map_err(|_| Error("retry terminal evidence"))?;
        if self.contract != CONTRACT
            || next(&parent.attempt_id)? != attempt_id
            || parent.project != field(run, "project")?
            || parent.run_id != field(run, "run_id")?
            || self.request.resume_from.run_id != parent.run_id
            || self.request.resume_from.attempt_id != parent.attempt_id
            || self.request_sha256 != self.request.identity(&self.parent_action)?
            || self.termination.retained_previous
            || !status.state.is_terminal()
            || status.job_id.as_str() != self.parent_job_id
            || status.scheduler_name
                != slurm::scheduler_job_name(&parent.run_id, &parent.attempt_id)
                    .map_err(|_| Error("parent scheduler name"))?
            || status.ssh_alias != self.backend_id
            || !slurm::component(&self.backend_id)
            || status.partition != field(&run["backend"], "partition")?
            || self.reserved_gpu_seconds != reservation(run, count)?
            || self.reserved_gpu_seconds as f64 > self.request.max_gpu_hours * 3600.
            || OffsetDateTime::parse(&self.created_at, &Rfc3339).is_err()
            || [
                &parent.project_sha256,
                &parent.run_sha256,
                &parent.attempt_sha256,
                &parent.source_receipt_sha256,
                &parent.backend_config_sha256,
            ]
            .iter()
            .any(|v| !slurm::helper::valid_digest(v))
        {
            return Err(Error("retry receipt or cumulative budget"));
        }
        Ok(())
    }
}
pub fn validate_attempt(run: &Value, attempt: &Value, binding: &Binding) -> Result<Receipt> {
    let receipt: Receipt = serde_json::from_value(attempt["native_retry"].clone())
        .map_err(|_| Error("retry receipt"))?;
    receipt.validate(run, &binding.attempt_id)?;
    if receipt.parent_binding.run_sha256 != binding.run_sha256
        || receipt.parent_binding.project_sha256 != binding.project_sha256
        || receipt.parent_binding.source_receipt_sha256 != binding.source_receipt_sha256
        || receipt.parent_binding.backend_config_sha256 != binding.backend_config_sha256
        || attempt["resume_from"] != json!(receipt.request.resume_from)
        || attempt["attempt_created_at"] != receipt.created_at
    {
        return Err(Error("retry authority binding"));
    }
    let strip = |v: &Value| -> Result<Value> {
        let mut value = v.as_object().cloned().ok_or(Error("retry manifest"))?;
        for key in [
            "attempt_id",
            "attempt_created_at",
            "native_retry",
            "resume_from",
            "native_checkpoint",
        ] {
            value.remove(key);
        }
        Ok(Value::Object(value))
    };
    if strip(run)? != strip(attempt)? {
        return Err(Error("retry changed scientific manifest"));
    }
    Ok(receipt)
}
pub struct Authority<'a> {
    pub backend_id: &'a str,
    pub parent: Binding,
    pub action: &'a str,
    pub job: &'a str,
    pub termination: slurm::StatusObservation,
    pub checkpoint: &'a Value,
}
pub fn plan(run: &Value, request: Request, authority: Authority<'_>, now: &str) -> Result<Value> {
    let Authority {
        backend_id,
        parent,
        action,
        job,
        termination,
        checkpoint,
    } = authority;
    let attempt_id = next(&parent.attempt_id)?;
    let receipt = Receipt {
        contract: CONTRACT.into(),
        parent_action: action.into(),
        parent_binding: parent,
        parent_job_id: job.into(),
        backend_id: backend_id.into(),
        termination,
        request: request.clone(),
        request_sha256: request.identity(action)?,
        reserved_gpu_seconds: reservation(run, number(&attempt_id)?)?,
        created_at: now.into(),
    };
    receipt.validate(run, &attempt_id)?;
    let (context, _) =
        checkpoints::validate_document(checkpoint, None, Some(&request.resume_from.checkpoint_id))
            .map_err(|_| Error("retry checkpoint receipt"))?;
    if context.project != field(run, "project")?
        || context.run_id != request.resume_from.run_id
        || context.attempt_id != request.resume_from.attempt_id
        || context.source_id != field(run, "source_id")?
        || context.image_id != field(run, "image_id")?
    {
        return Err(Error("retry checkpoint provenance"));
    }
    let mut attempt = run.clone();
    attempt["attempt_id"] = json!(attempt_id);
    attempt["attempt_created_at"] = json!(now);
    attempt["resume_from"] = json!(request.resume_from);
    attempt["native_checkpoint"] = checkpoint.clone();
    attempt["native_retry"] = json!(receipt);
    Ok(attempt)
}
