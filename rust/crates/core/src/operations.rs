//! Pure Action decisions. All clocks, identities and observations are caller inputs.
//! Unknown historical fields survive every transition; no function dispatches I/O.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

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
    #[error("Action operation is disabled by policy")]
    Disabled,
    #[error("Action revision changed")]
    RevisionConflict,
    #[error("Action status changed")]
    StatusConflict,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScopeType {
    Project,
    Campaign,
    Run,
    Attempt,
}

impl ScopeType {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Campaign => "campaign",
            Self::Run => "run",
            Self::Attempt => "attempt",
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

/// Match the historical directory validator without accepting path separators.
/// The stricter HTTP request identity remains action-[a-f0-9]{16}.
pub fn validate_action_id(value: &str) -> Result<(), Error> {
    if !value
        .strip_prefix("action-")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(char::is_alphanumeric))
    {
        return Err(Error::Invalid("action_id"));
    }
    Ok(())
}

/// The public Submission protocol uses the deterministic 16-hex Action ID;
/// the historical on-disk Action validator deliberately remains broader.
pub fn validate_submission_id(value: &str) -> Result<(), Error> {
    if value.strip_prefix("action-").is_some_and(|suffix| {
        suffix.len() == 16
            && suffix
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    }) {
        Ok(())
    } else {
        Err(Error::Identity("action_id"))
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
        "authorized_at":null,"authorization_note":"","started_at":null,"finished_at":null,"result":null,"error":null})
}

/// Public Submission projection of an already observed Action. Time is supplied
/// by the shell; this does not authorize, execute, repair or refresh anything.
pub fn submission_view(action: &Value, now: &str) -> Result<Value, Error> {
    let now = timestamp(now)?;
    let id = field(action, "action_id")?;
    validate_submission_id(id)?;
    let project = field(&action["scope"], "project")?;
    let run = field(action, "run_id")?;
    let attempt = field(action, "attempt_id")?;
    crate::identity::validate_id(project).map_err(|_| Error::Identity("project"))?;
    crate::identity::validate_observation_id(run).map_err(|_| Error::Identity("run_id"))?;
    crate::identity::validate_attempt(attempt).map_err(|_| Error::Identity("attempt_id"))?;
    let operation = field(action, "operation")?;
    if !matches!(operation, "SUBMIT_RUN" | "RETRY_ATTEMPT" | "RUN_EVALUATION") {
        return Err(Error::Invalid("Action is not a Submission"));
    }
    let execution = action
        .get("execution")
        .filter(|v| v.is_object())
        .ok_or(Error::Invalid("execution"))?;
    revision(execution)?;
    let status = field(execution, "status")?;
    let next = match status {
        "PREPARED" | "AUTHORIZED" => {
            if action
                .get("gate_expires_at")
                .and_then(Value::as_str)
                .and_then(|value| timestamp(value).ok())
                .is_none_or(|expiry| expiry <= now)
            {
                "REPREPARE"
            } else if status == "PREPARED" {
                "AUTHORIZE"
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Policy {
    pub allow_project_writes: bool,
    pub allow_scheduler_mutations: bool,
    pub allow_local_evidence_rebuild: bool,
    pub allow_resource_deletions: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Dispatch {
    ProjectWrite,
    LocalEvidenceRebuild,
    ResourceDeletion,
    Scheduler,
}

fn dispatch(operation: &str, policy: &Policy) -> Result<Dispatch, Error> {
    let (dispatch, allowed) = match operation {
        "WRITE_CAMPAIGN"
        | "WRITE_CAMPAIGN_ARCHIVE"
        | "WRITE_RUN_ARCHIVE"
        | "WRITE_ATTEMPT_ARCHIVE" => (Dispatch::ProjectWrite, policy.allow_project_writes),
        "REBUILD_LOCAL_EVIDENCE" => (
            Dispatch::LocalEvidenceRebuild,
            policy.allow_local_evidence_rebuild,
        ),
        "DELETE_RUN_RESOURCES" => (Dispatch::ResourceDeletion, policy.allow_resource_deletions),
        "SUBMIT_RUN" | "RETRY_ATTEMPT" | "RUN_EVALUATION" | "CANCEL_RUN" => {
            (Dispatch::Scheduler, policy.allow_scheduler_mutations)
        }
        _ => return Err(Error::State("historical operation is read-only")),
    };
    if allowed {
        Ok(dispatch)
    } else {
        Err(Error::Disabled)
    }
}

pub fn authorize(
    plan: &Value,
    execution: &Value,
    actor: &str,
    note: &str,
    now: &str,
) -> Result<Transition, Error> {
    // Approval may precede enabling an executor, but an unsupported historical
    // operation is a read-only record and must not acquire new authorization.
    dispatch(
        field(plan, "operation")?,
        &Policy {
            allow_project_writes: true,
            allow_scheduler_mutations: true,
            allow_local_evidence_rebuild: true,
            allow_resource_deletions: true,
        },
    )?;
    if plan.get("ready") != Some(&Value::Bool(true))
        || execution.get("status").and_then(Value::as_str) != Some("PREPARED")
    {
        return Err(Error::State(
            "only a ready PREPARED action can be authorized",
        ));
    }
    if timestamp(now)? >= timestamp(field(plan, "gate_expires_at")?)? {
        return Err(Error::Expired);
    }
    let actor = actor.trim();
    if actor.is_empty() || actor.chars().any(char::is_control) || note.chars().count() > 4000 {
        return Err(Error::Invalid("trusted authorization actor or note"));
    }
    let intent = digest(plan, "intent_digest")?;
    let gates = digest(plan, "gate_bundle_digest")?;
    transition(
        execution,
        json!({"status":"AUTHORIZED","authorized_at":now,"authorization_note":note,"authorization_actor":actor,
        "authorized_intent_digest":intent,"authorized_gate_bundle_digest":gates,"authorization_expires_at":plan["gate_expires_at"]}),
        now,
        "execution_authorized",
        Some("PREPARED"),
        json!({}),
    )
}

/// Files are hashed by the I/O shell. Missing observations never count as matches.
#[derive(Clone, Debug, Default)]
pub struct FrozenFiles {
    pub execution_campaign_sha256: Option<String>,
    pub authored_campaign_sha256: Option<String>,
    pub execution_manifest_sha256: Option<String>,
    pub collection_sha256: Option<String>,
    pub controller_snapshot_verified: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Claim {
    AlreadyVerified,
    Execute {
        transition: Transition,
        dispatch: Dispatch,
    },
}

pub fn claim(
    plan: &Value,
    execution: &Value,
    confirmation: &str,
    policy: &Policy,
    files: &FrozenFiles,
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
    if status != Some("AUTHORIZED") {
        return Err(Error::State(
            "a separate execution authorization is required",
        ));
    }
    let id = field(plan, "action_id")?;
    if confirmation != format!("EXECUTE {id}") {
        return Err(Error::Invalid("execution confirmation"));
    }
    if timestamp(now)? >= timestamp(field(plan, "gate_expires_at")?)? {
        return Err(Error::Expired);
    }
    if execution
        .get("authorized_intent_digest")
        .and_then(Value::as_str)
        != Some(digest(plan, "intent_digest")?)
    {
        return Err(Error::Identity("authorized intent"));
    }
    if execution
        .get("authorized_gate_bundle_digest")
        .and_then(Value::as_str)
        != Some(digest(plan, "gate_bundle_digest")?)
    {
        return Err(Error::Identity("authorized gates"));
    }
    let operation = field(plan, "operation")?;
    let dispatch = dispatch(operation, policy)?;
    if matches!(operation, "SUBMIT_RUN" | "RETRY_ATTEMPT" | "RUN_EVALUATION") {
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
        if !plan
            .get("authored_campaign_sha256")
            .unwrap_or(&Value::Null)
            .is_null()
            && files.authored_campaign_sha256.as_deref()
                != Some(digest(plan, "authored_campaign_sha256")?)
        {
            return Err(Error::Identity("authored campaign"));
        }
    }
    if dispatch == Dispatch::LocalEvidenceRebuild
        && (files.collection_sha256.as_deref() != Some(digest(plan, "expected_collection_sha256")?)
            || !files.controller_snapshot_verified)
    {
        return Err(Error::Identity(
            "local collection or approved controller snapshot",
        ));
    }
    Ok(Claim::Execute {
        transition: transition(
            execution,
            json!({"status":"EXECUTING","started_at":now,"error":null,"resolution":"STILL_IN_PROGRESS","safe_to_retry":false,"next_action":"WAIT"}),
            now,
            "execution_started",
            Some("AUTHORIZED"),
            json!({"intent_digest":plan["intent_digest"]}),
        )?,
        dispatch,
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

/// A successful status observation must come from the exact frozen command/cwd.
/// No provider call or submit command is executed by the reconciliation planner.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    pub command: Vec<String>,
    pub cwd: String,
    pub timed_out: bool,
    pub return_code: Option<i32>,
    pub payload: Value,
}
#[derive(Clone, Debug, Default)]
pub struct Evidence {
    pub observation: Observation,
    pub collection_sha256: Option<String>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Reconcile {
    Unchanged,
    Commit(Transition),
    /// Recovery of an approved idempotent transaction; never a scheduler submit.
    RecoverTransaction(Dispatch),
}
fn single(payload: &Value) -> Option<&Value> {
    if payload.is_object() {
        Some(payload)
    } else {
        payload
            .as_array()
            .filter(|a| a.len() == 1)
            .and_then(|a| a.first())
            .filter(|v| v.is_object())
    }
}

/// Historical SQLite/worker observations may represent a scheduler ID as an
/// integer. Preserve its exact decimal identity; an unsupported non-null value
/// is unknown evidence, never proof that no scheduler mutation occurred.
fn scheduler_identity(value: Option<&Value>) -> Result<Option<String>, ()> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.is_empty() => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(Value::Number(value)) => {
            let text = value.to_string();
            if text.bytes().all(|c| c.is_ascii_digit()) {
                Ok(Some(text))
            } else {
                Err(())
            }
        }
        Some(_) => Err(()),
    }
}
fn bound_observation<'a>(
    plan: &Value,
    evidence: &'a Observation,
) -> Result<&'a Value, &'static str> {
    let expected = plan
        .get("verification_command_preview")
        .and_then(Value::as_array);
    let command: Vec<Value> = evidence.command.iter().map(|s| json!(s)).collect();
    let expected_cwd = plan
        .get("verification_cwd")
        .and_then(Value::as_str)
        .or_else(|| plan.get("cwd").and_then(Value::as_str));
    if expected.is_none_or(|a| a.is_empty() || a != &command)
        || expected_cwd != Some(evidence.cwd.as_str())
    {
        return Err("status observation is not bound to the immutable verification command");
    }
    if evidence.timed_out {
        return Err("exact Attempt status verification timed out");
    }
    if evidence.return_code != Some(0) {
        return Err("exact Attempt status verification failed");
    }
    let observed =
        single(&evidence.payload).ok_or("status did not return exactly one Attempt record")?;
    for key in ["run_id", "attempt_id"] {
        if let (Some(actual), Some(expected)) = (
            observed.get(key).and_then(Value::as_str),
            plan.get(key).and_then(Value::as_str),
        ) {
            if !actual.is_empty() && !expected.is_empty() && actual != expected {
                return Err("status identity differs from the frozen Attempt");
            }
        }
    }
    Ok(observed)
}

pub fn reconcile(
    plan: &Value,
    execution: &Value,
    policy: &Policy,
    evidence: &Evidence,
    now: &str,
) -> Result<Reconcile, Error> {
    let status = execution.get("status").and_then(Value::as_str);
    if matches!(status, Some("VERIFIED" | "EXECUTING")) {
        return Ok(Reconcile::Unchanged);
    }
    if status != Some("RECONCILE_REQUIRED") {
        return Err(Error::State("action is not awaiting reconciliation"));
    }
    let operation = field(plan, "operation")?;
    if matches!(
        operation,
        "DELETE_RUN_RESOURCES"
            | "WRITE_CAMPAIGN"
            | "WRITE_CAMPAIGN_ARCHIVE"
            | "WRITE_RUN_ARCHIVE"
            | "WRITE_ATTEMPT_ARCHIVE"
    ) {
        return Ok(Reconcile::RecoverTransaction(dispatch(operation, policy)?));
    }
    if operation == "REBUILD_LOCAL_EVIDENCE" {
        dispatch(operation, policy)?;
        let actual = evidence.collection_sha256.as_deref();
        let new = digest(plan, "expected_new_collection_sha256")?;
        let old = digest(plan, "expected_collection_sha256")?;
        let previous = execution
            .get("result")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let mut result = previous;
        result.extend(object(&json!({"observed_collection_sha256":actual,"expected_new_collection_sha256":new,"reviewed_old_collection_sha256":old,"reconciled_read_only":true}))?.clone());
        let applied = actual == Some(new);
        let absent = actual == Some(old)
            && plan.get("atomic_collection_replace") == Some(&Value::Bool(true))
            && plan.get("write_protocol").and_then(Value::as_str) == Some("dirfd-fsync-rename-v1");
        let (status, event, error, resolution, next) = if applied {
            (
                "VERIFIED",
                "local_evidence_rebuild_reconciled",
                Value::Null,
                "APPLIED",
                "OBSERVE_RESULT",
            )
        } else if absent {
            (
                "FAILED",
                "local_evidence_rebuild_not_executed",
                json!(
                    "collection remains at the reviewed old digest; the atomic local evidence write did not execute"
                ),
                "NOT_APPLIED_SAFE_TO_RETRY",
                "PREPARE_NEW_ACTION",
            )
        } else {
            (
                "RECONCILE_REQUIRED",
                "local_evidence_rebuild_reconcile_blocked",
                json!(
                    "collection digest is neither the reviewed old digest nor the frozen expected new digest; manual investigation is required"
                ),
                "UNKNOWN_DO_NOT_RETRY",
                "RECONCILE",
            )
        };
        let mut update = json!({"status":status,"last_reconciled_at":now,"error":error,"result":result,"resolution":resolution,"safe_to_retry":absent,"next_action":next});
        if applied || absent {
            update["finished_at"] = json!(now);
        }
        return Ok(Reconcile::Commit(transition(
            execution,
            update,
            now,
            event,
            Some("RECONCILE_REQUIRED"),
            json!({}),
        )?));
    }
    if !matches!(
        operation,
        "SUBMIT_RUN" | "RETRY_ATTEMPT" | "RUN_EVALUATION" | "CANCEL_RUN"
    ) {
        return Err(Error::State("historical operation is read-only"));
    }
    // Observation remains allowed when mutation policy is disabled.
    let observed = bound_observation(plan, &evidence.observation);
    let previous = execution
        .get("result")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let result_shape_valid = execution
        .get("result")
        .is_none_or(|value| value.is_null() || value.is_object());
    let expected_job = match previous.get("submission") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(submission)) => scheduler_identity(submission.get("backend_job_id")),
        Some(_) => Err(()),
    };
    let record = observed.as_ref().ok().copied();
    let job = scheduler_identity(record.and_then(|r| r.get("backend_job_id")));
    let observed_state = record
        .and_then(|r| r.get("state"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_uppercase();
    let cancelled = operation == "CANCEL_RUN";
    let exact_job = if cancelled {
        let frozen_job = scheduler_identity(plan.get("backend_job_id"));
        matches!((&frozen_job, &job), (Ok(Some(expected)), Ok(Some(actual))) if expected == actual)
    } else {
        matches!(&job, Ok(Some(_)))
            && match &expected_job {
                Ok(None) => true,
                Ok(Some(expected)) => job == Ok(Some(expected.clone())),
                Err(()) => false,
            }
    };
    let terminal = matches!(
        observed_state.as_str(),
        "CANCELLED" | "CANCELED" | "COMPLETED" | "FAILED" | "SUCCEEDED"
    );
    let verified = result_shape_valid && observed.is_ok() && exact_job && (!cancelled || terminal);
    // Absence is positive evidence only for the exact Run/Attempt and a clean
    // bound status response; unavailable, foreign and duplicate replies stay unknown.
    let exact_attempt = record.is_some_and(|r| {
        ["run_id", "attempt_id"].iter().all(|key| {
            plan.get(*key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .is_some_and(|expected| r.get(*key).and_then(Value::as_str) == Some(expected))
        })
    });
    let absent = !cancelled
        && observed.is_ok()
        && result_shape_valid
        && expected_job == Ok(None)
        && observed_state == "NOT_SUBMITTED"
        && exact_attempt
        && job == Ok(None);
    let mut result = previous;
    result.insert("observation".into(), record.cloned().unwrap_or(Value::Null));
    result.insert("verification_command".into(),json!({"returncode":evidence.observation.return_code,"timeout":evidence.observation.timed_out,"payload":evidence.observation.payload}));
    let error = if verified {
        Value::Null
    } else if absent {
        json!("scheduler confirms that submission did not occur")
    } else {
        json!(observed.err().unwrap_or(if cancelled {
            "cancellation is not yet terminal"
        } else {
            "exact Attempt has no matching unique backend job identity"
        }))
    };
    let mut update = json!({"status":if verified {"VERIFIED"} else if absent {"FAILED"} else {"RECONCILE_REQUIRED"},"last_reconciled_at":now,"error":error,"result":result,
        "resolution":if verified {if cancelled {"APPLIED"}else{"SUBMITTED"}} else if absent {"NOT_SUBMITTED_SAFE_TO_RETRY"}else{"UNKNOWN_DO_NOT_RETRY"},
        "safe_to_retry":absent,"next_action":if verified {if cancelled {"OBSERVE_RESULT"} else {"MONITOR_RUN"}} else if absent {"PREPARE_NEW_ACTION"}else{"RECONCILE"}});
    if verified || absent {
        update["finished_at"] = json!(now);
    }
    let event = if cancelled {
        if verified {
            "cancellation_verified"
        } else {
            "cancellation_reconcile_pending"
        }
    } else if verified {
        "submission_reconciled"
    } else {
        "submission_reconcile_pending"
    };
    Ok(Reconcile::Commit(transition(
        execution,
        update,
        now,
        event,
        Some("RECONCILE_REQUIRED"),
        json!({}),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    const NOW: &str = "2026-10-27T00:00:00Z";
    fn hash(c: char) -> String {
        format!("sha256:{}", c.to_string().repeat(64))
    }
    fn plan(operation: &str) -> Value {
        json!({"action_id":"action-0123456789abcdef","ready":true,"operation":operation,"gate_expires_at":"2026-10-27T00:30:00Z",
            "intent_digest":hash('a'),"gate_bundle_digest":hash('b'),"execution_campaign_sha256":hash('c'),"execution_manifest_sha256":hash('d'),
            "run_id":"train","attempt_id":"attempt-001","verification_command_preview":["status","train","attempt-001"],"verification_cwd":"/reviewed",
            "frozen_unknown":{"resume_from":"checkpoint.immutable","budget":2,"submission_token":"frozen"}})
    }
    fn policy() -> Policy {
        Policy {
            allow_project_writes: true,
            allow_scheduler_mutations: true,
            allow_local_evidence_rebuild: true,
            allow_resource_deletions: true,
        }
    }
    fn files() -> FrozenFiles {
        FrozenFiles {
            execution_campaign_sha256: Some(hash('c')),
            execution_manifest_sha256: Some(hash('d')),
            ..Default::default()
        }
    }
    fn authorized(plan: &Value) -> Value {
        authorize(
            plan,
            &initial_execution(plan),
            "actor:test",
            "approved",
            NOW,
        )
        .unwrap()
        .execution
    }
    fn uncertain() -> Value {
        json!({"revision":3,"status":"RECONCILE_REQUIRED","result":null,"finished_at":null,"private_extension":{"keep":true}})
    }
    fn observation(payload: Value) -> Evidence {
        Evidence {
            observation: Observation {
                command: vec!["status".into(), "train".into(), "attempt-001".into()],
                cwd: "/reviewed".into(),
                return_code: Some(0),
                payload,
                ..Default::default()
            },
            ..Default::default()
        }
    }
    fn decided(value: Reconcile) -> Value {
        match value {
            Reconcile::Commit(t) => t.execution,
            _ => panic!("expected committed decision"),
        }
    }

    #[test]
    fn authorization_and_claim_are_immutable_and_require_exact_frozen_inputs() {
        let plan = plan("SUBMIT_RUN");
        let initial = initial_execution(&plan);
        let original = initial.clone();
        let authorized = authorize(&plan, &initial, " actor:test ", "approved", NOW).unwrap();
        assert_eq!(initial, original);
        assert_eq!(authorized.execution["authorization_actor"], "actor:test");
        assert!(matches!(
            claim(
                &plan,
                &authorized.execution,
                "wrong",
                &policy(),
                &files(),
                NOW
            ),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            claim(
                &plan,
                &authorized.execution,
                "EXECUTE action-0123456789abcdef",
                &Policy::default(),
                &files(),
                NOW
            ),
            Err(Error::Disabled)
        ));
        assert!(matches!(
            claim(
                &plan,
                &authorized.execution,
                "EXECUTE action-0123456789abcdef",
                &policy(),
                &FrozenFiles::default(),
                NOW
            ),
            Err(Error::Identity(_))
        ));
        let Claim::Execute {
            transition,
            dispatch,
        } = claim(
            &plan,
            &authorized.execution,
            "EXECUTE action-0123456789abcdef",
            &policy(),
            &files(),
            NOW,
        )
        .unwrap()
        else {
            panic!()
        };
        assert_eq!(dispatch, Dispatch::Scheduler);
        assert_eq!(transition.expected_status.as_deref(), Some("AUTHORIZED"));
        assert_eq!(transition.execution["status"], "EXECUTING");
        assert_eq!(transition.execution["safe_to_retry"], false);
        assert_eq!(
            plan["frozen_unknown"]["resume_from"],
            "checkpoint.immutable"
        );
        assert!(matches!(
            claim(
                &plan,
                &transition.execution,
                "EXECUTE action-0123456789abcdef",
                &policy(),
                &files(),
                NOW
            ),
            Err(Error::State(_))
        ));
        let mut changed = authorized.execution.clone();
        changed["authorized_intent_digest"] = json!(hash('f'));
        assert!(matches!(
            claim(
                &plan,
                &changed,
                "EXECUTE action-0123456789abcdef",
                &policy(),
                &files(),
                NOW
            ),
            Err(Error::Identity(_))
        ));
        assert!(matches!(
            authorize(
                &plan,
                &initial,
                "actor:test",
                "approved",
                "2026-10-27T00:30:00Z"
            ),
            Err(Error::Expired)
        ));
        assert!(matches!(
            authorize(&plan, &initial, " ", "approved", NOW),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            authorize(&plan, &initial, "actor:test", "approved", "invalid"),
            Err(Error::Invalid(_))
        ));
    }
    #[test]
    fn commit_is_cas_and_restart_preserves_identity_without_replaying() {
        let current = json!({"revision":4,"status":"AUTHORIZED","unknown_field":{"keep":true}});
        let mut proposed = current.clone();
        proposed["status"] = json!("EXECUTING");
        let (committed, event) = commit_transition(
            &current,
            &proposed,
            &json!({"event":"execution_started"}),
            Some("AUTHORIZED"),
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
                Some("PREPARED"),
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
    fn submission_absence_needs_clean_exact_bound_evidence_and_no_previous_job() {
        let plan = plan("SUBMIT_RUN");
        let record = json!({"run_id":"train","attempt_id":"attempt-001","state":"NOT_SUBMITTED"});
        let absent = decided(
            reconcile(
                &plan,
                &uncertain(),
                &Policy::default(),
                &observation(record.clone()),
                NOW,
            )
            .unwrap(),
        );
        assert_eq!(absent["status"], "FAILED");
        assert_eq!(absent["safe_to_retry"], true);
        assert_eq!(absent["next_action"], "PREPARE_NEW_ACTION");
        let mut foreign_record = record.clone();
        foreign_record["run_id"] = json!("foreign");
        let mut unknowns = vec![
            observation(json!([record.clone(), record.clone()])),
            observation(foreign_record),
        ];
        let mut timedout = observation(record.clone());
        timedout.observation.timed_out = true;
        unknowns.push(timedout);
        let mut foreign = observation(record.clone());
        foreign.observation.command = vec!["other-command".into()];
        unknowns.push(foreign);
        unknowns.push(observation(json!({"state":"NOT_SUBMITTED"})));
        for evidence in unknowns {
            let unknown =
                decided(reconcile(&plan, &uncertain(), &policy(), &evidence, NOW).unwrap());
            assert_eq!(unknown["status"], "RECONCILE_REQUIRED");
            assert_eq!(unknown["safe_to_retry"], false);
        }
        let mut previous = uncertain();
        previous["result"] = json!({"submission":{"backend_job_id":"job-1"}});
        let unknown =
            decided(reconcile(&plan, &previous, &policy(), &observation(record), NOW).unwrap());
        assert_eq!(unknown["safe_to_retry"], false);
        let visible = json!({"run_id":"train","attempt_id":"attempt-001","backend_job_id":"job-1","state":"RUNNING"});
        let verified = decided(
            reconcile(
                &plan,
                &previous,
                &Policy::default(),
                &observation(visible),
                NOW,
            )
            .unwrap(),
        );
        assert_eq!(verified["status"], "VERIFIED");
        assert_eq!(verified["next_action"], "MONITOR_RUN");
        assert_eq!(verified["private_extension"]["keep"], true);
    }
    #[test]
    fn legacy_numeric_job_ids_and_malformed_evidence_never_authorize_replay() {
        let plan = plan("SUBMIT_RUN");
        let absent = json!({"run_id":"train","attempt_id":"attempt-001","state":"NOT_SUBMITTED"});
        for value in [
            json!(24827),
            json!(true),
            json!(false),
            json!(1.5),
            json!({"id":24827}),
            json!([]),
            json!(0),
        ] {
            let mut previous = uncertain();
            previous["result"] = json!({"submission":{"backend_job_id":value}});
            let result = decided(
                reconcile(
                    &plan,
                    &previous,
                    &policy(),
                    &observation(absent.clone()),
                    NOW,
                )
                .unwrap(),
            );
            assert_eq!(result["status"], "RECONCILE_REQUIRED");
            assert_eq!(result["safe_to_retry"], false);
        }
        for malformed in [
            json!({"submission":24827}),
            json!({"submission":true}),
            json!("unknown prior result"),
        ] {
            let mut previous = uncertain();
            previous["result"] = malformed;
            let result = decided(
                reconcile(
                    &plan,
                    &previous,
                    &policy(),
                    &observation(absent.clone()),
                    NOW,
                )
                .unwrap(),
            );
            assert_eq!(result["safe_to_retry"], false);
        }
        for identity in [
            "24827",
            "9007199254740993",
            "123456789012345678901234567890",
        ] {
            let mut previous = uncertain();
            let numeric: Value = serde_json::from_str(identity).unwrap();
            previous["result"] = json!({"submission":{"backend_job_id":numeric}});
            for value in [json!(identity), numeric] {
                let visible = json!({"run_id":"train","attempt_id":"attempt-001","backend_job_id":value,"state":"RUNNING"});
                let result = decided(
                    reconcile(&plan, &previous, &policy(), &observation(visible), NOW).unwrap(),
                );
                assert_eq!(result["status"], "VERIFIED");
                assert_eq!(result["safe_to_retry"], false);
            }
            let foreign = json!({"run_id":"train","attempt_id":"attempt-001","backend_job_id":"9007199254740992","state":"RUNNING"});
            assert_eq!(
                decided(
                    reconcile(&plan, &previous, &policy(), &observation(foreign), NOW).unwrap()
                )["status"],
                "RECONCILE_REQUIRED"
            );
        }
        let mut corrupt_absence = absent.clone();
        corrupt_absence["backend_job_id"] = json!(true);
        assert_eq!(
            decided(
                reconcile(
                    &plan,
                    &uncertain(),
                    &policy(),
                    &observation(corrupt_absence),
                    NOW
                )
                .unwrap()
            )["safe_to_retry"],
            false
        );
        let mut cancel = plan.clone();
        cancel["operation"] = json!("CANCEL_RUN");
        cancel["backend_job_id"] = json!(24827);
        let terminal = json!({"run_id":"train","attempt_id":"attempt-001","backend_job_id":"24827","state":"CANCELLED"});
        assert_eq!(
            decided(
                reconcile(
                    &cancel,
                    &uncertain(),
                    &policy(),
                    &observation(terminal),
                    NOW
                )
                .unwrap()
            )["status"],
            "VERIFIED"
        );
    }
    #[test]
    fn local_atomic_recovery_distinguishes_applied_safe_absent_and_unknown() {
        let mut plan = plan("REBUILD_LOCAL_EVIDENCE");
        plan["expected_collection_sha256"] = json!(hash('c'));
        plan["expected_new_collection_sha256"] = json!(hash('d'));
        plan["atomic_collection_replace"] = json!(true);
        plan["write_protocol"] = json!("dirfd-fsync-rename-v1");
        for (actual, status, safe) in [
            (Some(hash('d')), "VERIFIED", false),
            (Some(hash('c')), "FAILED", true),
            (Some(hash('e')), "RECONCILE_REQUIRED", false),
            (None, "RECONCILE_REQUIRED", false),
        ] {
            let evidence = Evidence {
                collection_sha256: actual,
                ..Default::default()
            };
            let result =
                decided(reconcile(&plan, &uncertain(), &policy(), &evidence, NOW).unwrap());
            assert_eq!(result["status"], status);
            assert_eq!(result["safe_to_retry"], safe);
        }
        plan["atomic_collection_replace"] = json!(false);
        let result = decided(
            reconcile(
                &plan,
                &uncertain(),
                &policy(),
                &Evidence {
                    collection_sha256: Some(hash('c')),
                    ..Default::default()
                },
                NOW,
            )
            .unwrap(),
        );
        assert_eq!(result["safe_to_retry"], false);
        assert!(matches!(
            reconcile(
                &plan,
                &uncertain(),
                &Policy::default(),
                &Evidence::default(),
                NOW
            ),
            Err(Error::Disabled)
        ));
    }
    #[test]
    fn cancellation_and_transaction_recovery_never_schedule_a_new_submission() {
        let mut cancel = plan("CANCEL_RUN");
        cancel["backend_job_id"] = json!("job-1");
        for (job, state, status) in [
            ("job-1", "CANCELLED", "VERIFIED"),
            ("job-2", "CANCELLED", "RECONCILE_REQUIRED"),
            ("job-1", "RUNNING", "RECONCILE_REQUIRED"),
        ] {
            let result=decided(reconcile(&cancel,&uncertain(),&Policy::default(),&observation(json!({"run_id":"train","attempt_id":"attempt-001","backend_job_id":job,"state":state})),NOW).unwrap());
            assert_eq!(result["status"], status);
            assert_eq!(result["safe_to_retry"], false);
        }
        assert_eq!(
            reconcile(
                &plan("WRITE_CAMPAIGN"),
                &uncertain(),
                &policy(),
                &Evidence::default(),
                NOW
            )
            .unwrap(),
            Reconcile::RecoverTransaction(Dispatch::ProjectWrite)
        );
        assert_eq!(
            reconcile(
                &plan("DELETE_RUN_RESOURCES"),
                &uncertain(),
                &policy(),
                &Evidence::default(),
                NOW
            )
            .unwrap(),
            Reconcile::RecoverTransaction(Dispatch::ResourceDeletion)
        );
        assert!(matches!(
            claim(
                &plan("REMOVED_HISTORICAL_OPERATION"),
                &authorized(&plan("SUBMIT_RUN")),
                "EXECUTE action-0123456789abcdef",
                &policy(),
                &files(),
                NOW
            ),
            Err(Error::State(_))
        ));
    }
}
