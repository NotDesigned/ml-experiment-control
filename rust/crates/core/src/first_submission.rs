//! Pure planning for the first Attempt submission.
//!
//! The shell supplies already-read Run evidence, a campaign-file locator and
//! an optional verified source binding.  This module never reads those files,
//! contacts a source service, authorizes an Action or submits a scheduler job.
use crate::{canonical, identity};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid GPU-hour budget or resource approval mode")]
    InvalidBudget,
    #[error("project has no controller configuration")]
    ControllerUnavailable,
    #[error("Run is not eligible for first submission")]
    RunNotSubmittable,
    #[error("Run already has submitted Attempt evidence")]
    RunAlreadySubmitted,
    #[error("invalid first-submission identity or source binding")]
    Identity,
    #[error("source revision evidence is unavailable")]
    SourceUnavailable,
    #[error("invalid evidence digest")]
    Evidence,
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptEvidence {
    pub attempt_id: String,
    pub has_submission: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunEvidence {
    pub project: String,
    pub run_id: String,
    pub controller_configured: bool,
    pub scheduler_state: String,
    pub attempts: Vec<AttemptEvidence>,
    pub campaign_source_id: Option<String>,
    pub source_verified: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub campaign_file: String,
    pub max_gpu_hours: Option<f64>,
    pub resource_approval: String,
    pub reason: String,
    pub evidence: Value,
    pub submission_number: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntentPlan {
    pub kind: String,
    pub title: String,
    pub target: String,
    pub change_summary: String,
    pub resource_estimate: String,
    pub rationale: String,
    pub risk: String,
    pub draft: Value,
    pub evidence_digest: String,
    pub idempotency_key: Option<String>,
}

pub fn validate_gpu_budget(max_gpu_hours: Option<f64>, resource_approval: &str) -> Result<()> {
    if !matches!(resource_approval, "budget_cap" | "review_exact")
        || (resource_approval == "budget_cap"
            && !max_gpu_hours.is_some_and(|value| value.is_finite() && value > 0.0))
    {
        return Err(Error::InvalidBudget);
    }
    Ok(())
}

fn attempt_id(evidence: &RunEvidence) -> Result<String> {
    if evidence
        .attempts
        .iter()
        .any(|attempt| attempt.has_submission)
    {
        return Err(Error::RunAlreadySubmitted);
    }
    let id = evidence
        .attempts
        .last()
        .map(|attempt| attempt.attempt_id.clone())
        .unwrap_or_else(|| "attempt-001".into());
    identity::validate_attempt(&id).map_err(|_| Error::Identity)?;
    Ok(id)
}

fn source_id(evidence: &RunEvidence) -> Result<Option<String>> {
    let Some(source) = evidence.campaign_source_id.as_deref() else {
        return Ok(None);
    };
    if !source.starts_with("source.") || source.len() != 71 {
        return Err(Error::Identity);
    }
    if !source[7..]
        .bytes()
        .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
    {
        return Err(Error::Identity);
    }
    if !evidence.source_verified {
        return Err(Error::SourceUnavailable);
    }
    Ok(Some(source.to_owned()))
}

pub fn plan(evidence: &RunEvidence, request: &Request) -> Result<IntentPlan> {
    validate_gpu_budget(request.max_gpu_hours, &request.resource_approval)?;
    identity::validate_id(&evidence.project).map_err(|_| Error::Identity)?;
    identity::validate_id(&evidence.run_id).map_err(|_| Error::Identity)?;
    if !evidence.controller_configured {
        return Err(Error::ControllerUnavailable);
    }
    if !evidence
        .scheduler_state
        .eq_ignore_ascii_case("NOT_SUBMITTED")
    {
        return Err(Error::RunNotSubmittable);
    }
    let attempt_id = attempt_id(evidence)?;
    let source = source_id(evidence)?;
    if request.campaign_file.is_empty()
        || request.campaign_file.len() > 4096
        || request.campaign_file.chars().any(char::is_control)
    {
        return Err(Error::Identity);
    }
    let evidence_digest =
        canonical::protocol_identity(&request.evidence).map_err(|_| Error::Evidence)?;
    let mut draft = json!({
        "campaign_file": request.campaign_file,
        "run_id": evidence.run_id,
        "attempt_id": attempt_id,
        "resource_approval": request.resource_approval,
    });
    if let Some(source) = source {
        draft["expected_source_id"] = Value::String(source);
    }
    if let Some(hours) = request.max_gpu_hours {
        draft["max_gpu_hours"] = serde_json::Number::from_f64(hours)
            .map(Value::Number)
            .ok_or(Error::InvalidBudget)?;
    }
    let reason = if request.reason.is_empty() {
        "submit the authored Run's first Attempt"
    } else {
        request.reason.as_str()
    };
    let resource_estimate = if request.resource_approval == "review_exact" {
        "exact dry-run resources require human Action confirmation".into()
    } else {
        format!("up to {} GPU-hours", request.max_gpu_hours.unwrap())
    };
    let idempotency_key = if let Some(number) = request.submission_number {
        let value = format!(
            "{}:{}:{}:{}",
            evidence.project, evidence.run_id, attempt_id, number
        );
        let digest = canonical::digest(&Value::String(value)).map_err(|_| Error::Evidence)?;
        Some(format!("submission-{}", &digest[..20]))
    } else {
        None
    };
    Ok(IntentPlan {
        kind: "SUBMIT_RUN".into(),
        title: format!("Launch {} as {}", evidence.run_id, attempt_id),
        target: format!("run://{}/{}", evidence.project, evidence.run_id),
        change_summary: reason.into(),
        resource_estimate,
        rationale: "materialize one authored Campaign membership".into(),
        risk: "scheduler mutation requires separate authorization and execution".into(),
        draft,
        evidence_digest,
        idempotency_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> RunEvidence {
        RunEvidence {
            project: "demo".into(),
            run_id: "trial".into(),
            controller_configured: true,
            scheduler_state: "NOT_SUBMITTED".into(),
            attempts: Vec::new(),
            campaign_source_id: None,
            source_verified: false,
        }
    }

    fn request() -> Request {
        Request {
            campaign_file: "/workspace/campaign.yaml".into(),
            max_gpu_hours: Some(2.5),
            resource_approval: "budget_cap".into(),
            reason: String::new(),
            evidence: json!({"run":"trial","state":"NOT_SUBMITTED"}),
            submission_number: Some(3),
        }
    }

    #[test]
    fn plan_freezes_first_attempt_and_idempotency_identity() {
        let plan = plan(&evidence(), &request()).unwrap();
        assert_eq!(plan.kind, "SUBMIT_RUN");
        assert_eq!(plan.draft["attempt_id"], "attempt-001");
        assert_eq!(plan.idempotency_key.as_deref().unwrap().len(), 32);
        assert!(plan.evidence_digest.starts_with("sha256:"));
    }

    #[test]
    fn existing_attempts_and_submitted_runs_are_rejected() {
        let mut value = evidence();
        value.attempts.push(AttemptEvidence {
            attempt_id: "attempt-001".into(),
            has_submission: true,
        });
        assert_eq!(plan(&value, &request()), Err(Error::RunAlreadySubmitted));
        value.attempts[0].has_submission = false;
        value.scheduler_state = "QUEUED".into();
        assert_eq!(plan(&value, &request()), Err(Error::RunNotSubmittable));
    }

    #[test]
    fn budget_source_and_controller_gates_fail_closed() {
        assert_eq!(
            validate_gpu_budget(None, "budget_cap"),
            Err(Error::InvalidBudget)
        );
        let mut value = evidence();
        value.controller_configured = false;
        assert_eq!(plan(&value, &request()), Err(Error::ControllerUnavailable));
        let mut source = evidence();
        source.campaign_source_id = Some(format!("source.{}", "a".repeat(64)));
        assert_eq!(plan(&source, &request()), Err(Error::SourceUnavailable));
    }
}
