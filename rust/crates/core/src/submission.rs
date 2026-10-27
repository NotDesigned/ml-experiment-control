//! Pure submission intent, reconciliation and cancellation decisions.
//!
//! Provider adapters remain imperative: they query Slurm/PAI/SenseCore and
//! persist the returned records.  This module decides what those adapters may
//! do after an exact, frozen identity has been supplied.  In particular, an
//! unavailable provider query is never treated as proof that no job exists.
use crate::{canonical, identity, state::RunState};
use experiment_redact::redact_json;
use serde_json::{Map, Value, json};

pub const TOKEN_BYTES: usize = 16;
pub const MAX_REQUEST_BYTES: usize = 1 << 20;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid submission identity: {0}")]
    Identity(&'static str),
    #[error("invalid submission request: {0}")]
    Request(&'static str),
    #[error("invalid submission intent: {0}")]
    Intent(&'static str),
    #[error("submission intent is legacy and unsafe to recover")]
    LegacyUnsafe,
    #[error("submission intent conflicts in {0}")]
    Conflict(String),
    #[error("submission intent is required before reconciliation")]
    MissingIntent,
    #[error("invalid provider observation: {0}")]
    Observation(&'static str),
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetIdentity {
    pub project: String,
    pub run_id: String,
    pub attempt_id: String,
}

impl TargetIdentity {
    pub fn validate(&self) -> Result<()> {
        identity::validate_id(&self.project).map_err(|_| Error::Identity("project"))?;
        identity::validate_id(&self.run_id).map_err(|_| Error::Identity("run_id"))?;
        identity::validate_attempt(&self.attempt_id).map_err(|_| Error::Identity("attempt_id"))?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionIdentity {
    pub target: TargetIdentity,
    pub backend: String,
}

impl SubmissionIdentity {
    pub fn validate(&self) -> Result<()> {
        self.target.validate()?;
        identity::validate_id(&self.backend).map_err(|_| Error::Identity("backend"))?;
        Ok(())
    }
}

fn token_valid(value: &str) -> bool {
    value.len() == TOKEN_BYTES * 2
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Validate a caller-supplied 128-bit lowercase token. Random generation is
/// deliberately outside the pure core; the shell supplies fresh entropy.
pub fn validate_submission_token(value: &str) -> Result<&str> {
    token_valid(value)
        .then_some(value)
        .ok_or(Error::Identity("submission_token"))
}

pub fn submission_marker(token: &str) -> Result<String> {
    validate_submission_token(token)?;
    Ok(format!("ml-exp-{token}"))
}

fn request_object(value: &Value) -> Result<&Map<String, Value>> {
    value
        .as_object()
        .ok_or(Error::Request("request must be a mapping"))
}

/// Apply the same secret-bearing JSON policy used by outward diagnostics, then
/// enforce a bounded object. The returned value is safe to freeze in an intent.
pub fn sanitize_request(value: &Value) -> Result<Value> {
    request_object(value)?;
    let sanitized = redact_json(value.clone());
    let encoded = canonical::python_json_bytes(&sanitized)
        .map_err(|_| Error::Request("request is not canonical JSON"))?;
    if encoded.len() > MAX_REQUEST_BYTES {
        return Err(Error::Request("request exceeds size limit"));
    }
    Ok(sanitized)
}

/// Stable digest of the immutable scheduler identity and redacted request.
/// Secret changes therefore cannot silently create a second submission intent.
pub fn frozen_request_digest(identity: &SubmissionIdentity, request: &Value) -> Result<String> {
    identity.validate()?;
    let request = sanitize_request(request)?;
    canonical::protocol_identity(&json!({
        "project": identity.target.project,
        "run_id": identity.target.run_id,
        "attempt_id": identity.target.attempt_id,
        "backend": identity.backend,
        "request": request,
    }))
    .map_err(|_| Error::Request("request identity"))
}

fn field<'a>(value: &'a Value, name: &'static str) -> Result<&'a Value> {
    value
        .as_object()
        .and_then(|object| object.get(name))
        .ok_or(Error::Intent(name))
}

fn string_field<'a>(value: &'a Value, name: &'static str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or(Error::Intent(name))
}

fn optional_job(value: &Value) -> Result<Option<String>> {
    match value
        .as_object()
        .and_then(|object| object.get("backend_job_id"))
    {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(job)) if !job.is_empty() => {
            if job.len() > 256 || job.chars().any(|character| character.is_control()) {
                Err(Error::Intent("backend_job_id"))
            } else {
                Ok(Some(job.clone()))
            }
        }
        Some(_) => Err(Error::Intent("backend_job_id")),
    }
}

fn check_existing(
    existing: &Value,
    identity: &SubmissionIdentity,
    request: &Value,
) -> Result<(Value, Option<String>)> {
    identity.validate()?;
    let object = existing.as_object().ok_or(Error::Intent("object"))?;
    let mut conflicts = Vec::new();
    for (key, expected) in [
        ("project", &identity.target.project),
        ("run_id", &identity.target.run_id),
        ("attempt_id", &identity.target.attempt_id),
        ("backend", &identity.backend),
    ] {
        if object.get(key).and_then(Value::as_str) != Some(expected.as_str()) {
            conflicts.push(key.to_owned());
        }
    }
    let request = sanitize_request(request)?;
    let existing_request = object
        .get("request")
        .ok_or(Error::Intent("request"))
        .and_then(sanitize_request)?;
    if existing_request != request {
        conflicts.push("request".into());
    }
    if !conflicts.is_empty() {
        return Err(Error::Conflict(conflicts.join(", ")));
    }
    let token = match object.get("submission_token") {
        None | Some(Value::Null) => None,
        Some(Value::String(token)) => Some(validate_submission_token(token)?.to_owned()),
        Some(_) => return Err(Error::Intent("submission_token")),
    };
    if token.is_none() && object.get("state").and_then(Value::as_str) == Some("SUBMITTING") {
        return Err(Error::LegacyUnsafe);
    }
    Ok((existing.clone(), optional_job(existing)?))
}

#[derive(Clone, Debug, PartialEq)]
pub enum BeginDecision {
    Create {
        intent: Value,
        request_digest: String,
    },
    Reuse {
        intent: Value,
        request_digest: String,
    },
    Conflict {
        fields: Vec<String>,
    },
    LegacyUnsafe,
}

/// Decide whether to publish a new intent or reuse the exact immutable one.
/// `None` is the only case that creates an intent. A missing token on a legacy
/// `SUBMITTING` record is rejected rather than repaired or replayed.
pub fn begin_submission(
    identity: &SubmissionIdentity,
    request: &Value,
    existing: Option<&Value>,
    token: &str,
    created_at: &str,
) -> Result<BeginDecision> {
    identity.validate()?;
    validate_submission_token(token)?;
    if created_at.is_empty() || created_at.len() > 128 || created_at.chars().any(char::is_control) {
        return Err(Error::Intent("created_at"));
    }
    let request = sanitize_request(request)?;
    let request_digest = frozen_request_digest(identity, &request)?;
    if let Some(existing) = existing {
        return match check_existing(existing, identity, &request) {
            Ok((intent, _)) => {
                // The token is part of the scheduler mutation identity.  A
                // caller that presents a different token is asking about a
                // different mutation, even when all frozen request fields
                // happen to match.  Reusing the old intent in that case would
                // allow an accidental second submission to masquerade as an
                // idempotent retry.
                let existing_token = intent.get("submission_token").and_then(Value::as_str);
                if existing_token.is_some_and(|value| value != token) {
                    return Ok(BeginDecision::Conflict {
                        fields: vec!["submission_token".into()],
                    });
                }
                Ok(BeginDecision::Reuse {
                    intent,
                    request_digest,
                })
            }
            Err(Error::Conflict(fields)) => Ok(BeginDecision::Conflict {
                fields: fields.split(", ").map(str::to_owned).collect(),
            }),
            Err(Error::LegacyUnsafe) => Ok(BeginDecision::LegacyUnsafe),
            Err(error) => Err(error),
        };
    }
    Ok(BeginDecision::Create {
        intent: json!({
            "project": identity.target.project,
            "run_id": identity.target.run_id,
            "attempt_id": identity.target.attempt_id,
            "backend": identity.backend,
            "request": request,
            "submission_token": token,
            "state": "SUBMITTING",
            "created_at": created_at,
        }),
        request_digest,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactTarget {
    pub identity: TargetIdentity,
}

impl ExactTarget {
    pub fn validate(&self) -> Result<()> {
        self.identity.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderObservation {
    ExactAbsent { target: ExactTarget },
    ExactFound { target: ExactTarget, job_id: String },
    UnavailableOrAmbiguous { reason: String },
}

impl ProviderObservation {
    fn validate(&self) -> Result<()> {
        match self {
            Self::ExactAbsent { target } => target.validate(),
            Self::ExactFound { target, job_id } => {
                target.validate()?;
                if job_id.is_empty() || job_id.len() > 256 || job_id.chars().any(char::is_control) {
                    return Err(Error::Observation("job_id"));
                }
                Ok(())
            }
            Self::UnavailableOrAmbiguous { reason } => {
                if reason.is_empty() || reason.len() > 2048 || reason.chars().any(char::is_control)
                {
                    return Err(Error::Observation("reason"));
                }
                Ok(())
            }
        }
    }
}

fn same_target(identity: &SubmissionIdentity, target: &ExactTarget) -> bool {
    identity.target == target.identity
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReconcileDecision {
    Attached { intent: Value, job_id: String },
    NotSubmittedSafeToRetry { reason: String },
    UnknownDoNotRetry { reason: String },
}

/// Reconcile one exact provider observation. Only a positive exact match may
/// attach a job. Exact absence is retry-safe only while the intent has no job
/// and is still SUBMITTING/NOT_SUBMITTED.
pub fn reconcile_submission(
    identity: &SubmissionIdentity,
    existing: Option<&Value>,
    observation: &ProviderObservation,
    _state: RunState,
    reconciled_at: &str,
) -> Result<ReconcileDecision> {
    identity.validate()?;
    observation.validate()?;
    let existing = existing.ok_or(Error::MissingIntent)?;
    let (mut intent, current_job) = check_existing(
        existing,
        identity,
        existing
            .as_object()
            .and_then(|value| value.get("request"))
            .ok_or(Error::Intent("request"))?,
    )?;
    if reconciled_at.is_empty()
        || reconciled_at.len() > 128
        || reconciled_at.chars().any(char::is_control)
    {
        return Err(Error::Intent("reconciled_at"));
    }
    // The first successful reconciliation fixes the receipt timestamp.  A
    // repeated provider read must be byte-for-byte idempotent rather than
    // rewriting the frozen intent with the poller's later wall clock.
    let effective_reconciled_at = match intent.get("reconciled_at") {
        None | Some(Value::Null) => reconciled_at.to_owned(),
        Some(Value::String(previous))
            if !previous.is_empty()
                && previous.len() <= 128
                && !previous.chars().any(char::is_control) =>
        {
            previous.clone()
        }
        Some(_) => return Err(Error::Intent("reconciled_at")),
    };
    match observation {
        ProviderObservation::UnavailableOrAmbiguous { reason } => {
            Ok(ReconcileDecision::UnknownDoNotRetry {
                reason: reason.clone(),
            })
        }
        ProviderObservation::ExactAbsent { target } => {
            if !same_target(identity, target) {
                return Ok(ReconcileDecision::UnknownDoNotRetry {
                    reason: "provider absence was not bound to this Run/Attempt".into(),
                });
            }
            let current_state = intent.get("state").and_then(Value::as_str).unwrap_or("");
            if current_job.is_none() && matches!(current_state, "SUBMITTING" | "NOT_SUBMITTED") {
                Ok(ReconcileDecision::NotSubmittedSafeToRetry {
                    reason: "provider confirms that this exact Run/Attempt was not submitted"
                        .into(),
                })
            } else {
                Ok(ReconcileDecision::UnknownDoNotRetry {
                    reason: "provider absence conflicts with an existing submission intent".into(),
                })
            }
        }
        ProviderObservation::ExactFound { target, job_id } => {
            if !same_target(identity, target) {
                return Ok(ReconcileDecision::UnknownDoNotRetry {
                    reason: "provider job was not bound to this Run/Attempt".into(),
                });
            }
            if current_job
                .as_deref()
                .is_some_and(|current| current != job_id)
            {
                return Err(Error::Conflict("backend_job_id".into()));
            }
            // Provider identity is authoritative for this reconciliation.  A
            // scheduler may accept a job before the local lifecycle reader
            // observes QUEUED/RUNNING (and a caller may still report the
            // durable SUBMITTING state), so do not reject an exact match based
            // on a lagging sampled state.
            let object = intent.as_object_mut().ok_or(Error::Intent("object"))?;
            object.insert("state".into(), Value::String("SUBMITTED".into()));
            object.insert("backend_job_id".into(), Value::String(job_id.clone()));
            object.insert(
                "reconciled_at".into(),
                Value::String(effective_reconciled_at),
            );
            Ok(ReconcileDecision::Attached {
                intent,
                job_id: job_id.clone(),
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CancelIdentity {
    pub target: TargetIdentity,
    pub backend: Option<String>,
    pub backend_job_id: String,
}

impl CancelIdentity {
    fn validate(&self) -> Result<()> {
        self.target.validate()?;
        if let Some(backend) = &self.backend {
            identity::validate_id(backend).map_err(|_| Error::Identity("backend"))?;
        }
        if self.backend_job_id.is_empty()
            || self.backend_job_id.len() > 256
            || self.backend_job_id.chars().any(char::is_control)
        {
            return Err(Error::Identity("backend_job_id"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CancelStatus {
    Exact {
        target: ExactTarget,
        job_id: String,
        state: String,
        result: Value,
    },
    UnavailableOrAmbiguous {
        reason: String,
    },
}

impl CancelStatus {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Exact {
                target,
                job_id,
                state,
                result,
            } => {
                target.validate()?;
                if job_id.is_empty()
                    || job_id.len() > 256
                    || job_id.chars().any(char::is_control)
                    || state.is_empty()
                    || state.len() > 64
                    || state.chars().any(char::is_control)
                    || !result.is_object()
                {
                    return Err(Error::Observation("cancel status"));
                }
                Ok(())
            }
            Self::UnavailableOrAmbiguous { reason } => {
                if reason.is_empty() || reason.len() > 2048 || reason.chars().any(char::is_control)
                {
                    return Err(Error::Observation("reason"));
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CancelDecision {
    CreateRequested { intent: Value, issue_job_id: String },
    VerifyTerminal { intent: Value, result: Value },
    AlreadyVerified { result: Value },
    Wait { reason: String },
}

fn terminal_state(state: &str) -> bool {
    matches!(
        state.to_ascii_uppercase().as_str(),
        "SUCCEEDED" | "FAILED" | "CANCELLED" | "CANCELED" | "PREEMPTED" | "COMPLETED"
    )
}

fn cancel_identity_matches(intent: &Value, identity: &CancelIdentity) -> bool {
    let Some(object) = intent.as_object() else {
        return false;
    };
    object.get("project").and_then(Value::as_str) == Some(identity.target.project.as_str())
        && object.get("run_id").and_then(Value::as_str) == Some(identity.target.run_id.as_str())
        && object.get("attempt_id").and_then(Value::as_str)
            == Some(identity.target.attempt_id.as_str())
        && object.get("backend_job_id").and_then(Value::as_str)
            == Some(identity.backend_job_id.as_str())
        && match (&identity.backend, object.get("backend")) {
            (None, None | Some(Value::Null)) => true,
            (Some(expected), Some(Value::String(actual))) => expected == actual,
            _ => false,
        }
}

/// Reduce the cancel outbox before any provider mutation. A fresh request
/// publishes REQUESTED and returns the one permitted cancel call. A persisted
/// REQUESTED intent never issues a second cancel until an exact terminal read.
pub fn plan_cancel(
    identity: &CancelIdentity,
    existing: Option<&Value>,
    status: Option<&CancelStatus>,
    now: &str,
) -> Result<CancelDecision> {
    identity.validate()?;
    if now.is_empty() || now.len() > 128 || now.chars().any(char::is_control) {
        return Err(Error::Intent("requested_at"));
    }
    if let Some(status) = status {
        status.validate()?;
    }
    let Some(existing) = existing else {
        return Ok(CancelDecision::CreateRequested {
            intent: json!({
                "schema_version": 1,
                "state": "REQUESTED",
                "requested_at": now,
                "project": identity.target.project,
                "run_id": identity.target.run_id,
                "attempt_id": identity.target.attempt_id,
                "backend": identity.backend,
                "backend_job_id": identity.backend_job_id,
            }),
            issue_job_id: identity.backend_job_id.clone(),
        });
    };
    if !cancel_identity_matches(existing, identity) {
        return Err(Error::Conflict("cancel scheduler identity".into()));
    }
    let state = string_field(existing, "state")?;
    if state == "VERIFIED" {
        let result = existing
            .get("result")
            .filter(|value| value.is_object())
            .cloned()
            .ok_or(Error::Intent("result"))?;
        return Ok(CancelDecision::AlreadyVerified { result });
    }
    if state != "REQUESTED" {
        return Err(Error::Intent("cancel state"));
    }
    let Some(status) = status else {
        return Ok(CancelDecision::Wait {
            reason: "a persisted cancel request requires an exact status observation".into(),
        });
    };
    match status {
        CancelStatus::UnavailableOrAmbiguous { reason } => Ok(CancelDecision::Wait {
            reason: reason.clone(),
        }),
        CancelStatus::Exact {
            target,
            job_id,
            state,
            result,
        } => {
            if target.identity != identity.target || job_id != &identity.backend_job_id {
                return Ok(CancelDecision::Wait {
                    reason: "status is not bound to the exact scheduler job".into(),
                });
            }
            if terminal_state(state) {
                let mut intent = existing.clone();
                let object = intent.as_object_mut().ok_or(Error::Intent("object"))?;
                object.insert("state".into(), Value::String("VERIFIED".into()));
                object.insert("verified_at".into(), Value::String(now.into()));
                object.insert("result".into(), result.clone());
                Ok(CancelDecision::VerifyTerminal {
                    intent,
                    result: result.clone(),
                })
            } else {
                Ok(CancelDecision::Wait {
                    reason: "exact scheduler job remains nonterminal; no second cancel is allowed"
                        .into(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: &str = "0123456789abcdef0123456789abcdef";
    fn identity() -> SubmissionIdentity {
        SubmissionIdentity {
            target: TargetIdentity {
                project: "project".into(),
                run_id: "run-s0".into(),
                attempt_id: "attempt-001".into(),
            },
            backend: "slurm".into(),
        }
    }
    fn target() -> ExactTarget {
        ExactTarget {
            identity: identity().target,
        }
    }
    fn created() -> Value {
        match begin_submission(
            &identity(),
            &json!({"argv":["sbatch"],"environment":{"API_TOKEN":"secret"}}),
            None,
            TOKEN,
            "2026-10-28T00:00:00Z",
        )
        .unwrap()
        {
            BeginDecision::Create { intent, .. } => intent,
            _ => unreachable!(),
        }
    }

    #[test]
    fn begin_is_idempotent_and_redacts_frozen_request() {
        let request = json!({
            "argv":["sbatch"],
            "environment":{"API_TOKEN":"secret","SEED":"7"},
            "callback":"https://user:password@example.invalid/status"
        });
        let first = begin_submission(&identity(), &request, None, TOKEN, "now").unwrap();
        let intent = match &first {
            BeginDecision::Create { intent, .. } => intent.clone(),
            _ => unreachable!(),
        };
        let second =
            begin_submission(&identity(), &request, Some(&intent), TOKEN, "later").unwrap();
        match second {
            BeginDecision::Reuse { intent: reused, .. } => assert_eq!(reused, intent),
            _ => unreachable!(),
        }
        assert_eq!(intent["request"]["environment"]["API_TOKEN"], "<redacted>");
        assert_eq!(
            intent["request"]["callback"],
            "https://<redacted>@example.invalid/status"
        );
    }

    #[test]
    fn begin_rejects_every_immutable_change_and_legacy_submitting() {
        let base = created();
        for (name, altered) in [
            ("project", json!({"project":"other"})),
            ("run_id", json!({"run_id":"other"})),
            ("attempt_id", json!({"attempt_id":"attempt-002"})),
            ("backend", json!({"backend":"pai"})),
            ("request", json!({"request":{"different":true}})),
        ] {
            let mut candidate = identity();
            let mut request = json!({"argv":["sbatch"],"environment":{"API_TOKEN":"secret"}});
            if name == "project" {
                candidate.target.project = altered["project"].as_str().unwrap().into();
            }
            if name == "run_id" {
                candidate.target.run_id = altered["run_id"].as_str().unwrap().into();
            }
            if name == "attempt_id" {
                candidate.target.attempt_id = altered["attempt_id"].as_str().unwrap().into();
            }
            if name == "backend" {
                candidate.backend = altered["backend"].as_str().unwrap().into();
            }
            if name == "request" {
                request = altered["request"].clone();
            }
            assert!(
                matches!(
                    begin_submission(&candidate, &request, Some(&base), TOKEN, "now").unwrap(),
                    BeginDecision::Conflict { .. }
                ),
                "{name}"
            );
        }
        let legacy = json!({
            "project": "project",
            "run_id": "run-s0",
            "attempt_id": "attempt-001",
            "backend": "slurm",
            "request": {},
            "state": "SUBMITTING"
        });
        assert_eq!(
            begin_submission(&identity(), &json!({}), Some(&legacy), TOKEN, "now").unwrap(),
            BeginDecision::LegacyUnsafe
        );
    }

    #[test]
    fn begin_rejects_a_different_submission_token() {
        let base = created();
        let other = "fedcba9876543210fedcba9876543210";
        assert!(matches!(
            begin_submission(
                &identity(),
                &json!({"argv":["sbatch"],"environment":{"API_TOKEN":"secret"}}),
                Some(&base),
                other,
                "later",
            )
            .unwrap(),
            BeginDecision::Conflict { fields }
                if fields == vec!["submission_token".to_owned()]
        ));
    }

    #[test]
    fn exact_provider_absence_is_the_only_retry_safe_reconciliation() {
        let intent = created();
        let absent = reconcile_submission(
            &identity(),
            Some(&intent),
            &ProviderObservation::ExactAbsent { target: target() },
            RunState::Submitting,
            "now",
        )
        .unwrap();
        assert!(matches!(
            absent,
            ReconcileDecision::NotSubmittedSafeToRetry { .. }
        ));
        let unknown = reconcile_submission(
            &identity(),
            Some(&intent),
            &ProviderObservation::UnavailableOrAmbiguous {
                reason: "timeout".into(),
            },
            RunState::Submitting,
            "now",
        )
        .unwrap();
        assert!(matches!(
            unknown,
            ReconcileDecision::UnknownDoNotRetry { .. }
        ));
    }

    #[test]
    fn exact_job_attachment_is_idempotent_and_conflicting_jobs_are_rejected() {
        let intent = created();
        let attached = reconcile_submission(
            &identity(),
            Some(&intent),
            &ProviderObservation::ExactFound {
                target: target(),
                job_id: "job-1".into(),
            },
            RunState::Queued,
            "now",
        )
        .unwrap();
        let attached_intent = match attached {
            ReconcileDecision::Attached { intent, .. } => intent,
            _ => unreachable!(),
        };
        let repeated = reconcile_submission(
            &identity(),
            Some(&attached_intent),
            &ProviderObservation::ExactFound {
                target: target(),
                job_id: "job-1".into(),
            },
            RunState::Queued,
            "later",
        )
        .unwrap();
        let repeated_intent = match repeated {
            ReconcileDecision::Attached { intent, .. } => intent,
            _ => unreachable!(),
        };
        assert_eq!(repeated_intent, attached_intent);
        assert!(matches!(
            reconcile_submission(
                &identity(),
                Some(&attached_intent),
                &ProviderObservation::ExactFound {
                    target: target(),
                    job_id: "job-2".into()
                },
                RunState::Queued,
                "later",
            ),
            Err(Error::Conflict(_))
        ));
    }

    #[test]
    fn exact_job_attachment_is_allowed_while_local_state_is_submitting() {
        let intent = created();
        let attached = reconcile_submission(
            &identity(),
            Some(&intent),
            &ProviderObservation::ExactFound {
                target: target(),
                job_id: "job-1".into(),
            },
            RunState::Submitting,
            "now",
        )
        .unwrap();
        assert!(matches!(
            attached,
            ReconcileDecision::Attached { job_id, .. } if job_id == "job-1"
        ));
    }

    #[test]
    fn cancel_outbox_writes_once_and_never_reissues_for_nonterminal_job() {
        let cancel = CancelIdentity {
            target: identity().target,
            backend: Some("slurm".into()),
            backend_job_id: "job-1".into(),
        };
        let requested = plan_cancel(&cancel, None, None, "now").unwrap();
        let intent = match requested {
            CancelDecision::CreateRequested { intent, .. } => intent,
            _ => unreachable!(),
        };
        let waiting = plan_cancel(
            &cancel,
            Some(&intent),
            Some(&CancelStatus::Exact {
                target: target(),
                job_id: "job-1".into(),
                state: "RUNNING".into(),
                result: json!({"state":"RUNNING","backend_job_id":"job-1"}),
            }),
            "later",
        )
        .unwrap();
        assert!(matches!(waiting, CancelDecision::Wait { .. }));
    }

    #[test]
    fn cancel_terminal_status_verifies_without_mutating_provider_again() {
        let cancel = CancelIdentity {
            target: identity().target,
            backend: Some("slurm".into()),
            backend_job_id: "job-1".into(),
        };
        let intent = match plan_cancel(&cancel, None, None, "now").unwrap() {
            CancelDecision::CreateRequested { intent, .. } => intent,
            _ => unreachable!(),
        };
        let result = json!({"state":"CANCELLED","backend_job_id":"job-1"});
        let verified = plan_cancel(
            &cancel,
            Some(&intent),
            Some(&CancelStatus::Exact {
                target: target(),
                job_id: "job-1".into(),
                state: "CANCELLED".into(),
                result: result.clone(),
            }),
            "later",
        )
        .unwrap();
        let verified_intent = match verified {
            CancelDecision::VerifyTerminal { intent, .. } => intent,
            _ => unreachable!(),
        };
        assert!(matches!(
            plan_cancel(&cancel, Some(&verified_intent), None, "again").unwrap(),
            CancelDecision::AlreadyVerified { .. }
        ));
    }

    #[test]
    fn cancel_rejects_a_corrupt_verified_receipt() {
        let cancel = CancelIdentity {
            target: identity().target,
            backend: Some("slurm".into()),
            backend_job_id: "job-1".into(),
        };
        let corrupt = json!({
            "state": "VERIFIED",
            "project": "project",
            "run_id": "run-s0",
            "attempt_id": "attempt-001",
            "backend": "slurm",
            "backend_job_id": "job-1",
            "result": null,
        });
        assert!(matches!(
            plan_cancel(&cancel, Some(&corrupt), None, "now"),
            Err(Error::Intent("result"))
        ));
    }
}
