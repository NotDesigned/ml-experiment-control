//! Pure Action decisions. All clocks, identities and observations are caller inputs.
//! Fields a transition does not name survive it; no function dispatches I/O.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

const SUBMIT_RUN: &str = "SUBMIT_RUN";

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("invalid Action record: {0}")]
    Invalid(&'static str),
    #[error("Action transition is not permitted: {0}")]
    State(&'static str),
    #[error("Action gate has expired; prepare and approve a fresh intent")]
    Expired,
    #[error("Action frozen identity changed: {0}")]
    Identity(&'static str),
    #[error("Action revision changed")]
    RevisionConflict,
    #[error("Action status changed")]
    StatusConflict,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScopeType {
    Run,
}

impl ScopeType {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Run => "run",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Scope {
    pub project: String,
    pub scope_type: ScopeType,
    pub object_id: String,
}

pub fn action_id(scope: &Scope, intent_key: &str) -> String {
    let raw = format!(
        "{}:{}:{}:{intent_key}",
        scope.project,
        scope.scope_type.name(),
        scope.object_id
    );
    format!("action-{:x}", Sha256::digest(raw.as_bytes()))[..23].to_owned()
}

/// Exactly what `action_id` generates, action-[a-f0-9]{16}, wherever an Action ID
/// is read: a request, a directory name or a database key.
pub fn validate_action_id(value: &str) -> Result<(), Error> {
    if value.strip_prefix("action-").is_some_and(|suffix| {
        suffix.len() == 16
            && suffix
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    }) {
        Ok(())
    } else {
        Err(Error::Invalid("action_id"))
    }
}

fn object(value: &Value) -> Result<&Map<String, Value>, Error> {
    value
        .as_object()
        .ok_or(Error::Invalid("JSON object required"))
}
fn field<'a>(value: &'a Value, name: &'static str) -> Result<&'a str, Error> {
    value
        .get(name)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or(Error::Invalid(name))
}
fn timestamp(value: &str) -> Result<OffsetDateTime, Error> {
    OffsetDateTime::parse(value, &Rfc3339).map_err(|_| Error::Invalid("RFC3339 timestamp"))
}
fn digest<'a>(value: &'a Value, name: &'static str) -> Result<&'a str, Error> {
    let value = field(value, name)?;
    if !value.strip_prefix("sha256:").is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    }) {
        return Err(Error::Invalid(name));
    }
    Ok(value)
}
pub fn revision(execution: &Value) -> Result<i64, Error> {
    match execution.get("revision") {
        None => Ok(0),
        Some(value) => value
            .as_i64()
            .filter(|n| *n >= 0)
            .ok_or(Error::Invalid("revision")),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    pub execution: Value,
    pub event: Value,
    pub expected_status: Option<String>,
}

fn transition(
    execution: &Value,
    updates: Value,
    now: &str,
    event: &str,
    expected: Option<&str>,
    details: Value,
) -> Result<Transition, Error> {
    timestamp(now)?;
    let mut next = object(execution)?.clone();
    next.extend(object(&updates)?.clone());
    let mut payload = Map::from_iter([
        (
            "status".into(),
            next.get("status").cloned().unwrap_or(Value::Null),
        ),
        (
            "error".into(),
            next.get("error").cloned().unwrap_or(Value::Null),
        ),
    ]);
    payload.extend(object(&details)?.clone());
    Ok(Transition {
        execution: Value::Object(next),
        event: json!({"timestamp":now,"event":event,"payload":payload}),
        expected_status: expected.map(str::to_owned),
    })
}

pub fn initial_execution(plan: &Value) -> Value {
    json!({"revision":0,"status":if plan.get("ready") == Some(&Value::Bool(true)) {"PREPARED"} else {"BLOCKED"},
        "started_at":null,"finished_at":null,"result":null,"error":null})
}

/// Public Submission projection of an already observed Action. Time is supplied
/// by the shell; this does not execute, repair or refresh anything.
pub fn submission_view(action: &Value, now: &str) -> Result<Value, Error> {
    let now = timestamp(now)?;
    let id = field(action, "action_id")?;
    validate_action_id(id)?;
    let project = field(&action["scope"], "project")?;
    let run = field(action, "run_id")?;
    let attempt = field(action, "attempt_id")?;
    crate::identity::validate_id(project).map_err(|_| Error::Identity("project"))?;
    crate::identity::validate_observation_id(run).map_err(|_| Error::Identity("run_id"))?;
    crate::identity::validate_attempt(attempt).map_err(|_| Error::Identity("attempt_id"))?;
    let operation = field(action, "operation")?;
    if operation != SUBMIT_RUN {
        return Err(Error::Invalid("Action is not a Submission"));
    }
    let execution = action
        .get("execution")
        .filter(|v| v.is_object())
        .ok_or(Error::Invalid("execution"))?;
    revision(execution)?;
    let status = field(execution, "status")?;
    let next = match status {
        "PREPARED" => {
            if action
                .get("gate_expires_at")
                .and_then(Value::as_str)
                .and_then(|value| timestamp(value).ok())
                .is_none_or(|expiry| expiry <= now)
            {
                "REPREPARE"
            } else {
                "EXECUTE"
            }
        }
        "EXECUTING" | "RECONCILE_REQUIRED" => "RECONCILE",
        "BLOCKED" => "REPREPARE",
        _ => "NONE",
    };
    let summary = action
        .get("preflight_summary")
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let git_commit = summary
        .get("git_commit")
        .filter(|v| !v.is_null() && **v != "")
        .cloned()
        .or_else(|| {
            action
                .get("semantic_changes")
                .and_then(Value::as_array)
                .and_then(|changes| changes.iter().find(|change| change["path"] == "git_commit"))
                .and_then(|change| change.get("after"))
                .cloned()
        });
    Ok(json!({
        "submission_id":id, "project":project, "run_id":run, "attempt_id":attempt,
        "operation":operation, "status":status, "ready":action.get("ready") == Some(&Value::Bool(true)),
        "next_action":next, "reused":false, "confirmation":format!("EXECUTE {id}"),
        "gate_expires_at":action.get("gate_expires_at"), "intent_digest":action.get("intent_digest"),
        "gates":action.get("gates").filter(|v| v.is_array()).cloned().unwrap_or_else(|| json!([])),
        "preflight_summary":summary, "git_commit":git_commit, "execution":execution,
    }))
}

/// Prepare exactly the SQLite CAS update and its audit event, without mutating input.
pub fn commit_transition(
    current: &Value,
    proposed: &Value,
    event: &Value,
    expected_status: Option<&str>,
    transition_id: &str,
) -> Result<(Value, Value), Error> {
    if expected_status
        .is_some_and(|expected| current.get("status").and_then(Value::as_str) != Some(expected))
    {
        return Err(Error::StatusConflict);
    }
    let current_revision = revision(current)?;
    if proposed.get("revision").and_then(Value::as_i64) != Some(current_revision) {
        return Err(Error::RevisionConflict);
    }
    if transition_id.len() != 32
        || !transition_id
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(Error::Invalid("transition_id"));
    }
    let next = current_revision
        .checked_add(1)
        .ok_or(Error::Invalid("revision overflow"))?;
    let mut execution = object(proposed)?.clone();
    execution.insert("revision".into(), json!(next));
    let mut event = object(event)?.clone();
    event.insert("transition_id".into(), json!(transition_id));
    event.insert("revision".into(), json!(next));
    Ok((Value::Object(execution), Value::Object(event)))
}

/// Every Action is a Submission; a plan naming anything else cannot be executed.
fn require_submission(plan: &Value) -> Result<(), Error> {
    if field(plan, "operation")? == SUBMIT_RUN {
        Ok(())
    } else {
        Err(Error::State("operation cannot be dispatched"))
    }
}

/// What `claim` refuses from the request and the clock alone. A shell that makes a
/// durable record ahead of its claim calls this first, so that a request which could
/// never be claimed leaves nothing behind.
pub fn check_request(
    plan: &Value,
    confirmation: &str,
    reason: &str,
    now: &str,
) -> Result<(), Error> {
    let id = field(plan, "action_id")?;
    if confirmation != format!("EXECUTE {id}") {
        return Err(Error::Invalid("execution confirmation"));
    }
    if timestamp(now)? >= timestamp(field(plan, "gate_expires_at")?)? {
        return Err(Error::Expired);
    }
    if reason.chars().count() > 4000 || reason.chars().any(char::is_control) {
        return Err(Error::Invalid("execution reason"));
    }
    Ok(())
}

/// Files are hashed by the I/O shell. Missing observations never count as matches.
#[derive(Clone, Debug, Default)]
pub struct FrozenFiles {
    pub execution_campaign_sha256: Option<String>,
    pub execution_manifest_sha256: Option<String>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Claim {
    AlreadyVerified,
    Execute { transition: Transition },
}

pub fn claim(
    plan: &Value,
    execution: &Value,
    confirmation: &str,
    files: &FrozenFiles,
    reason: &str,
    now: &str,
) -> Result<Claim, Error> {
    if execution.get("status").and_then(Value::as_str) == Some("VERIFIED") {
        return Ok(Claim::AlreadyVerified);
    }
    let status = execution.get("status").and_then(Value::as_str);
    if matches!(status, Some("EXECUTING" | "RECONCILE_REQUIRED")) {
        return Err(Error::State(
            "execution intent already exists; reconcile instead of replaying",
        ));
    }
    if plan.get("ready") != Some(&Value::Bool(true)) || status != Some("PREPARED") {
        return Err(Error::State("only a ready PREPARED action can be executed"));
    }
    check_request(plan, confirmation, reason, now)?;
    digest(plan, "intent_digest")?;
    digest(plan, "gate_bundle_digest")?;
    require_submission(plan)?;
    for (key, observed) in [
        (
            "execution_campaign_sha256",
            &files.execution_campaign_sha256,
        ),
        (
            "execution_manifest_sha256",
            &files.execution_manifest_sha256,
        ),
    ] {
        if observed.as_deref() != Some(digest(plan, key)?) {
            return Err(Error::Identity(key));
        }
    }
    Ok(Claim::Execute {
        transition: transition(
            execution,
            json!({"status":"EXECUTING","started_at":now,"execution_reason":reason,"error":null,"resolution":"STILL_IN_PROGRESS","safe_to_retry":false,"next_action":"WAIT"}),
            now,
            "execution_started",
            Some("PREPARED"),
            json!({"intent_digest":plan["intent_digest"],"reason":reason}),
        )?,
    })
}

pub fn interrupted(execution: &Value, now: &str) -> Result<Option<Transition>, Error> {
    if execution.get("status").and_then(Value::as_str) != Some("EXECUTING") {
        return Ok(None);
    }
    let error = execution
        .get("error")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("daemon restarted after execution was claimed; observe effects before retry");
    Ok(Some(transition(
        execution,
        json!({"status":"RECONCILE_REQUIRED","error":error,"resolution":"UNKNOWN_DO_NOT_RETRY","safe_to_retry":false,"next_action":"RECONCILE"}),
        now,
        "execution_interrupted_by_restart",
        Some("EXECUTING"),
        json!({}),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    const NOW: &str = "2026-10-27T00:00:00Z";
    const CONFIRMATION: &str = "EXECUTE action-0123456789abcdef";
    fn hash(c: char) -> String {
        format!("sha256:{}", c.to_string().repeat(64))
    }
    fn plan(operation: &str) -> Value {
        json!({"action_id":"action-0123456789abcdef","ready":true,"operation":operation,"gate_expires_at":"2026-10-27T00:30:00Z",
            "intent_digest":hash('a'),"gate_bundle_digest":hash('b'),"execution_campaign_sha256":hash('c'),"execution_manifest_sha256":hash('d'),
            "run_id":"train","attempt_id":"attempt-001",
            "frozen_unknown":{"resume_from":"checkpoint.immutable","budget":2,"submission_token":"frozen"}})
    }
    fn files() -> FrozenFiles {
        FrozenFiles {
            execution_campaign_sha256: Some(hash('c')),
            execution_manifest_sha256: Some(hash('d')),
        }
    }
    #[test]
    fn claim_is_immutable_and_requires_exact_frozen_inputs() {
        let plan = plan("SUBMIT_RUN");
        let initial = initial_execution(&plan);
        let original = initial.clone();
        assert!(matches!(
            claim(&plan, &initial, "wrong", &files(), "", NOW),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            claim(
                &plan,
                &initial,
                CONFIRMATION,
                &FrozenFiles::default(),
                "",
                NOW
            ),
            Err(Error::Identity(_))
        ));
        let Claim::Execute { transition } =
            claim(&plan, &initial, CONFIRMATION, &files(), "approved", NOW).unwrap()
        else {
            panic!()
        };
        assert_eq!(initial, original);
        assert_eq!(transition.expected_status.as_deref(), Some("PREPARED"));
        assert_eq!(transition.execution["status"], "EXECUTING");
        assert_eq!(transition.execution["safe_to_retry"], false);
        assert_eq!(transition.execution["execution_reason"], "approved");
        assert_eq!(transition.event["payload"]["reason"], "approved");
        assert_eq!(
            plan["frozen_unknown"]["resume_from"],
            "checkpoint.immutable"
        );
        assert!(matches!(
            claim(
                &plan,
                &transition.execution,
                CONFIRMATION,
                &files(),
                "",
                NOW
            ),
            Err(Error::State(_))
        ));
        // Either digest must still be a well formed digest in the frozen plan.
        for key in ["intent_digest", "gate_bundle_digest"] {
            let mut changed = plan.clone();
            changed[key] = json!("not-a-digest");
            assert_eq!(
                claim(&changed, &initial, CONFIRMATION, &files(), "", NOW).unwrap_err(),
                Error::Invalid(key)
            );
        }
    }
    #[test]
    fn claim_refuses_an_expired_gate_and_a_bad_clock_or_reason() {
        let plan = plan("SUBMIT_RUN");
        let initial = initial_execution(&plan);
        for expired in ["2026-10-27T00:30:00Z", "2026-10-27T01:00:00Z"] {
            assert_eq!(
                claim(&plan, &initial, CONFIRMATION, &files(), "", expired).unwrap_err(),
                Error::Expired
            );
            assert_eq!(
                check_request(&plan, CONFIRMATION, "", expired),
                Err(Error::Expired)
            );
        }
        assert!(matches!(
            claim(&plan, &initial, CONFIRMATION, &files(), "", "invalid"),
            Err(Error::Invalid(_))
        ));
        for reason in ["bad\nreason".to_owned(), "r".repeat(4001)] {
            assert!(matches!(
                claim(&plan, &initial, CONFIRMATION, &files(), &reason, NOW),
                Err(Error::Invalid("execution reason"))
            ));
        }
        assert!(
            claim(
                &plan,
                &initial,
                CONFIRMATION,
                &files(),
                &"r".repeat(4000),
                NOW
            )
            .is_ok()
        );
        assert_eq!(check_request(&plan, CONFIRMATION, "", NOW), Ok(()));
        assert_eq!(
            check_request(&plan, "wrong", "", NOW),
            Err(Error::Invalid("execution confirmation"))
        );
    }
    #[test]
    fn only_a_ready_prepared_action_can_be_claimed() {
        let plan = plan("SUBMIT_RUN");
        let initial = initial_execution(&plan);
        let mut blocked = plan.clone();
        blocked["ready"] = json!(false);
        let refused = Error::State("only a ready PREPARED action can be executed");
        // A plan that is not ready, whichever status its execution row claims.
        assert_eq!(
            claim(&blocked, &initial, CONFIRMATION, &files(), "", NOW).unwrap_err(),
            refused
        );
        assert_eq!(initial_execution(&blocked)["status"], "BLOCKED");
        assert_eq!(
            claim(
                &blocked,
                &initial_execution(&blocked),
                CONFIRMATION,
                &files(),
                "",
                NOW
            )
            .unwrap_err(),
            refused
        );
        // A status this version never writes is not treated as PREPARED.
        for status in ["AUTHORIZED", "FAILED", "UNKNOWN"] {
            let mut other = initial.clone();
            other["status"] = json!(status);
            assert_eq!(
                claim(&plan, &other, CONFIRMATION, &files(), "", NOW).unwrap_err(),
                refused
            );
        }
    }
    #[test]
    fn commit_is_cas_and_restart_preserves_identity_without_replaying() {
        let current = json!({"revision":4,"status":"PREPARED","unknown_field":{"keep":true}});
        let mut proposed = current.clone();
        proposed["status"] = json!("EXECUTING");
        let (committed, event) = commit_transition(
            &current,
            &proposed,
            &json!({"event":"execution_started"}),
            Some("PREPARED"),
            "0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        assert_eq!(committed["revision"], 5);
        assert_eq!(event["revision"], 5);
        assert_eq!(committed["unknown_field"], current["unknown_field"]);
        assert!(matches!(
            commit_transition(
                &committed,
                &proposed,
                &json!({}),
                None,
                "0123456789abcdef0123456789abcdef"
            ),
            Err(Error::RevisionConflict)
        ));
        assert!(matches!(
            commit_transition(
                &current,
                &proposed,
                &json!({}),
                Some("EXECUTING"),
                "0123456789abcdef0123456789abcdef"
            ),
            Err(Error::StatusConflict)
        ));
        let restarted = interrupted(&committed, NOW).unwrap().unwrap();
        assert_eq!(restarted.execution["status"], "RECONCILE_REQUIRED");
        assert_eq!(restarted.execution["safe_to_retry"], false);
        assert_eq!(restarted.execution["next_action"], "RECONCILE");
        assert_eq!(
            restarted.execution["unknown_field"],
            current["unknown_field"]
        );
        assert!(interrupted(&restarted.execution, NOW).unwrap().is_none());
    }
    #[test]
    fn an_action_id_is_exactly_what_the_generator_produces() {
        let scope = Scope {
            project: "demo".into(),
            scope_type: ScopeType::Run,
            object_id: "train".into(),
        };
        assert!(validate_action_id(&action_id(&scope, "submit")).is_ok());
        assert!(validate_action_id("action-0123456789abcdef").is_ok());
        for refused in [
            "",
            "action-",
            "action-a",
            "action-0123456789abcde",
            "action-0123456789abcdef0",
            "action-0123456789ABCDEF",
            "action-0123456789abcdeg",
            "action-é",
            "action-0123456789abcdé",
            "action-../../../etc",
            "0123456789abcdef",
        ] {
            assert_eq!(
                validate_action_id(refused),
                Err(Error::Invalid("action_id")),
                "{refused}"
            );
        }
    }
    #[test]
    fn only_a_submission_can_be_claimed() {
        let mut other = plan("SUBMIT_RUN");
        other["operation"] = json!("CANCEL_RUN");
        assert_eq!(
            claim(
                &other,
                &initial_execution(&other),
                CONFIRMATION,
                &files(),
                "",
                NOW
            )
            .unwrap_err(),
            Error::State("operation cannot be dispatched")
        );
    }
}
