//! Submission request limits, token validation and pure cancellation decisions.
//!
//! The worker queries Slurm and persists the returned intents. This module
//! decides what it may do next: a persisted cancel request never issues a second
//! cancel, and an unavailable query is never treated as proof of termination.
use crate::identity;
use serde_json::{Value, json};

pub const TOKEN_BYTES: usize = 16;
pub const MAX_REQUEST_BYTES: usize = 1 << 20;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid submission identity: {0}")]
    Identity(&'static str),
    #[error("invalid submission intent: {0}")]
    Intent(&'static str),
    #[error("submission intent conflicts in {0}")]
    Conflict(String),
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
        let result = match existing.get("result") {
            Some(value) if value.is_object() => value.clone(),
            _ => return Err(Error::Intent("result")),
        };
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

    fn identity() -> TargetIdentity {
        TargetIdentity {
            project: "project".into(),
            run_id: "run-s0".into(),
            attempt_id: "attempt-001".into(),
        }
    }
    fn target() -> ExactTarget {
        ExactTarget {
            identity: identity(),
        }
    }

    #[test]
    fn cancel_outbox_writes_once_and_never_reissues_for_nonterminal_job() {
        let cancel = CancelIdentity {
            target: identity(),
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
            target: identity(),
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
            target: identity(),
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
