//! Native Action binding and outcome reduction. No clocks, files or RPC calls.
//! Legacy plans cannot acquire this execution protocol by interpretation.
use super::helper::{self, OperationTag, Reply, ReplyOutcome, Request};
use crate::{
    canonical,
    operations::{self, Scope, ScopeType, Transition},
};
use serde_json::{Value, json};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("native Slurm Action binding or execution evidence is invalid")]
pub struct Error;
pub type Result<T> = std::result::Result<T, Error>;
const FORMAT: &str = "native-slurm-action.v1";
const HASH_FIELDS: [&str; 4] = [
    "action_id",
    "request_digest",
    "intent_digest",
    "gate_bundle_digest",
];
fn digest(value: &Value) -> Result<String> {
    canonical::protocol_identity(value).map_err(|_| Error)
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value.get(key).and_then(Value::as_str).ok_or(Error)
}
fn body(plan: &Value) -> Result<Value> {
    let mut value = plan.as_object().cloned().ok_or(Error)?;
    for key in HASH_FIELDS.into_iter().chain(["execution", "journal"]) {
        value.remove(key);
    }
    Ok(Value::Object(value))
}
fn scope(plan: &Value, request: &Request) -> Result<Scope> {
    request.validate().map_err(|_| Error)?;
    let scope: Scope = serde_json::from_value(plan["scope"].clone()).map_err(|_| Error)?;
    if scope.scope_type != ScopeType::Run
        || scope.project != request.intent().scope.project
        || scope.object_id != request.intent().scope.run_id
        || text(plan, "run_id")? != request.intent().scope.run_id
        || text(plan, "attempt_id")? != request.intent().scope.attempt_id
        || text(plan, "operation")? != "SUBMIT_RUN"
        || text(plan, "execution_manifest_sha256")? != request.intent().manifest_sha256
        || request.operation() != OperationTag::PrepareSubmission
    {
        return Err(Error);
    }
    for key in ["execution_campaign_sha256", "execution_manifest_sha256"] {
        if !helper::valid_digest(text(plan, key)?) {
            return Err(Error);
        }
    }
    if !plan["authored_campaign_sha256"].is_null()
        && !helper::valid_digest(text(plan, "authored_campaign_sha256")?)
    {
        return Err(Error);
    }
    let created = OffsetDateTime::parse(text(plan, "created_at")?, &Rfc3339).map_err(|_| Error)?;
    let expires =
        OffsetDateTime::parse(text(plan, "gate_expires_at")?, &Rfc3339).map_err(|_| Error)?;
    if expires <= created || !plan["gates"].is_array() || !plan["ready"].is_boolean() {
        return Err(Error);
    }
    if serde_json::to_string(plan)
        .map_err(|_| Error)?
        .contains(&request.intent().submission_token)
    {
        return Err(Error);
    }
    Ok(scope)
}
/// Called only while preparing a NEW reviewed plan, before authorization.
/// The native binding is covered by both approval digests and its stable ID.
pub fn prepare(mut plan: Value, request: &Request) -> Result<Value> {
    if plan.get("execution").is_some()
        || plan.get("journal").is_some()
        || plan.get("native_slurm").is_some()
    {
        return Err(Error);
    }
    let scope = scope(&plan, request)?;
    plan["native_slurm"] =
        json!({"format":FORMAT,"intent_sha256":request.intent().digest().map_err(|_| Error)?});
    let intent = digest(&body(&plan)?)?;
    plan["action_id"] = json!(operations::action_id(&scope, &intent));
    plan["request_digest"] = json!(intent);
    plan["intent_digest"] = json!(intent);
    plan["gate_bundle_digest"] = json!(digest(
        &json!({"intent_digest":intent,"gates":plan["gates"]})
    )?);
    validate(&plan, request)?;
    Ok(plan)
}
/// Recompute all native binding digests from the persisted immutable plan.
pub fn validate(plan: &Value, request: &Request) -> Result<()> {
    let scope = scope(plan, request)?;
    if plan["native_slurm"]
        != json!({"format":FORMAT,"intent_sha256":request.intent().digest().map_err(|_| Error)?})
    {
        return Err(Error);
    }
    let intent = digest(&body(plan)?)?;
    if text(plan, "request_digest")? != intent
        || text(plan, "intent_digest")? != intent
        || text(plan, "action_id")? != operations::action_id(&scope, &intent)
        || text(plan, "gate_bundle_digest")?
            != digest(&json!({"intent_digest":intent,"gates":plan["gates"]}))?
    {
        return Err(Error);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uncertain {
    Preparation,
    Submission,
    Ledger,
    Observation,
    Guard,
}
impl Uncertain {
    fn message(self) -> &'static str {
        match self {
            Self::Preparation => {
                "native helper preparation is uncertain; inspect the original ledger"
            }
            Self::Submission => {
                "native helper submission is uncertain; inspect the original ledger"
            }
            Self::Ledger => "native helper ledger does not prove an owned scheduler job",
            Self::Observation => "the original scheduler job has not been verified",
            Self::Guard => "local execution authority changed; inspect the original effect",
        }
    }
}
pub enum Outcome {
    Owned {
        job_id: String,
        reply: Reply,
    },
    Uncertain {
        job_id: Option<String>,
        reason: Uncertain,
    },
}
/// Reduce only a claimed or interrupted submission. VERIFIED means submission
/// ownership was observed, never that training/results have completed.
pub fn finish(plan: &Value, request: &Request, outcome: Outcome, now: &str) -> Result<Transition> {
    validate(plan, request)?;
    OffsetDateTime::parse(now, &Rfc3339).map_err(|_| Error)?;
    let execution = plan["execution"].as_object().ok_or(Error)?;
    let expected = text(&plan["execution"], "status")?;
    if !matches!(expected, "EXECUTING" | "RECONCILE_REQUIRED") {
        return Err(Error);
    }
    let mut result = match execution.get("result") {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(value)) => value.clone(),
        _ => return Err(Error),
    };
    let previous_job = match result.get("submission") {
        None | Some(Value::Null) => None,
        Some(Value::Object(value)) => {
            if value.get("backend").and_then(Value::as_str)
                != Some(request.intent().target.backend_id.as_str())
            {
                return Err(Error);
            }
            Some(
                value
                    .get("backend_job_id")
                    .and_then(Value::as_str)
                    .ok_or(Error)?
                    .to_owned(),
            )
        }
        _ => return Err(Error),
    };
    let (job, observation, error) = match outcome {
        Outcome::Owned { job_id, reply } => {
            let expected = Request::observe_job(request.intent().clone(), job_id.clone())
                .map_err(|_| Error)?;
            let checked =
                Reply::for_request(&expected, reply.outcome.clone()).map_err(|_| Error)?;
            if checked.identity != reply.identity {
                return Err(Error);
            }
            let ReplyOutcome::JobObserved { observation } = reply.outcome else {
                return Err(Error);
            };
            (Some(job_id), Some(observation), None)
        }
        Outcome::Uncertain { job_id, reason } => (job_id, None, Some(reason.message())),
    };
    let job = job.or_else(|| previous_job.clone());
    if let Some(job) = &job {
        let parsed = super::JobId::parse(job).map_err(|_| Error)?;
        if parsed.as_str().contains('_') || previous_job.as_ref().is_some_and(|old| old != job) {
            return Err(Error);
        }
        result.insert(
            "submission".into(),
            json!({"backend":request.intent().target.backend_id,"backend_job_id":job}),
        );
    }
    if let Some(observation) = observation {
        result.insert(
            "observation".into(),
            serde_json::to_value(observation).map_err(|_| Error)?,
        );
    }
    let verified = error.is_none();
    let status = if verified {
        "VERIFIED"
    } else {
        "RECONCILE_REQUIRED"
    };
    let mut next = execution.clone();
    next.extend(
        json!({"status":status,"error":error,"result":result,"safe_to_retry":false,
        "resolution":if verified {"APPLIED"} else {"UNKNOWN_DO_NOT_RETRY"},
        "next_action":if verified {"OBSERVE_RESULT"} else {"RECONCILE"},
        "last_reconciled_at":now})
        .as_object()
        .ok_or(Error)?
        .clone(),
    );
    if verified {
        next.insert("finished_at".into(), json!(now));
    }
    Ok(Transition {
        execution: Value::Object(next),
        event: json!({"timestamp":now,"event":if verified {"native_slurm_submission_verified"} else {"native_slurm_submission_uncertain"},
            "payload":{"status":status,"error":error,"native_intent_sha256":request.intent().digest().map_err(|_| Error)?}}),
        expected_status: Some(expected.to_owned()),
    })
}
