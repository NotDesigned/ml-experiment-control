//! Saved-identity observation as a pure request/response state machine. Planning
//! returns GET requests; only `resolve` below executes the HTTP boundary.
use crate::{Client, Error};
use ml_exp_core::identity::validate_attempt;
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    Experiment,
    TrainingOperation,
}

#[derive(Default)]
pub struct Selection<'a> {
    pub project: Option<&'a str>,
    pub run: Option<&'a str>,
    pub attempt: Option<&'a str>,
}
#[derive(Clone, Serialize)]
pub struct Hints {
    pub server: Option<String>,
    pub project: Option<String>,
    pub run_id: Option<String>,
    pub attempt_id: Option<String>,
    pub state_kind: Kind,
    pub saved: Value,
}
#[derive(Clone, Serialize)]
pub struct Reference {
    pub server: String,
    pub project: String,
    pub run_id: String,
    pub attempt_id: Option<String>,
    pub state_kind: Option<Kind>,
    pub saved: Value,
    pub preparation: Option<Value>,
    pub submission: Option<Value>,
    pub action: Option<Value>,
}
pub enum Step {
    Complete(Reference),
    Read(Read),
}
pub struct Read {
    path: String,
    reference: Reference,
    phase: Phase,
}
enum Phase {
    Preparation(String),
    Submission(String),
    Action,
}

fn fail(message: &'static str) -> Error {
    Error::Observation(message)
}
fn truth(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Number(value) => value.as_f64().is_none_or(|v| v != 0.0),
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        Value::Bool(true) => true,
    }
}
fn named_hex(value: &str, prefix: &str, len: usize) -> bool {
    value.strip_prefix(prefix).is_some_and(|suffix| {
        suffix.len() == len
            && suffix
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}
fn attempt(value: Option<&Value>) -> Result<String, Error> {
    value
        .and_then(Value::as_str)
        .filter(|v| validate_attempt(v).is_ok())
        .map(str::to_owned)
        .ok_or_else(|| fail("choose an exact attempt-NNN identity"))
}
fn action_id(value: Option<&Value>) -> Result<String, Error> {
    value
        .and_then(Value::as_str)
        .filter(|v| named_hex(v, "action-", 16))
        .map(str::to_owned)
        .ok_or_else(|| fail("invalid saved Submission identity"))
}

/// API read IDs intentionally retain the baseline's broader historical grammar.
/// They are never reused as filesystem paths or as managed resource identities.
pub fn segment(value: &str) -> Result<String, Error> {
    if value.is_empty()
        || value.len() > 256
        || !value.as_bytes()[0].is_ascii_alphanumeric()
        || !value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.:-".contains(&c))
    {
        return Err(fail("invalid API identity"));
    }
    Ok(value.replace(':', "%3A"))
}

fn optional_hint(saved: &Value, key: &str) -> Result<Option<String>, Error> {
    match saved.get(key) {
        None => Ok(None),
        Some(value) => value
            .as_str()
            .filter(|v| !v.is_empty())
            .map(|v| Some(v.into()))
            .ok_or_else(|| fail("invalid saved server or Run identity")),
    }
}

pub fn server_hint(saved: &Value) -> Result<Option<String>, Error> {
    if !saved.is_object() {
        return Err(fail(
            "observation requires a saved experiment or training-operation state",
        ));
    }
    optional_hint(saved, "server")
}

pub fn reference(saved: &Value) -> Result<Hints, Error> {
    let server = server_hint(saved)?;
    let (state_kind, target) = if saved.get("contract").and_then(Value::as_str)
        == Some("training-operation.v1")
    {
        if !matches!(
            saved.get("operation").and_then(Value::as_str),
            Some("cancel" | "retry")
        ) || !saved.get("plan").is_some_and(Value::is_object)
        {
            return Err(fail(
                "training preparation reply is unknown; inspect its recovery URL and attach the original Action; no Attempt was guessed",
            ));
        }
        let target = attempt(saved["plan"].get("attempt_id"))?;
        attempt(saved.get("source_attempt_id"))?;
        if action_id(saved["plan"].get("action_id")).is_err() {
            return Err(fail("invalid saved training Action identity"));
        }
        (Kind::TrainingOperation, Some(target))
    } else if saved.get("config_sha256").is_some() {
        for key in ["preparation", "submission"] {
            if saved
                .get(key)
                .is_some_and(|v| !v.is_null() && !v.is_object())
            {
                return Err(fail("invalid saved preparation or Submission identity"));
            }
        }
        let submission = saved.get("submission").filter(|v| truth(v)).or_else(|| {
            saved
                .get("preparation")
                .and_then(|p| p.get("submission"))
                .filter(|v| truth(v))
        });
        if submission.is_some_and(|v| !v.is_object()) {
            return Err(fail("invalid saved Submission identity"));
        }
        let target = if let Some(submission) = submission {
            let target = attempt(submission.get("attempt_id"))?;
            action_id(submission.get("submission_id"))?;
            Some(target)
        } else {
            None
        };
        (Kind::Experiment, target)
    } else {
        return Err(fail(
            "observation requires an experiment state or training-operation.v1 plan",
        ));
    };
    let project = optional_hint(saved, "project")?;
    let run_id = optional_hint(saved, "run_id")?;
    for value in [&project, &run_id].into_iter().flatten() {
        segment(value)?;
    }
    Ok(Hints {
        server,
        project,
        run_id,
        attempt_id: target,
        state_kind,
        saved: saved.clone(),
    })
}

pub fn bound_response<'a>(
    value: &'a Value,
    expected: &[(&str, Value)],
    message: &'static str,
) -> Result<&'a Value, Error> {
    if !value.is_object()
        || expected
            .iter()
            .any(|(key, wanted)| value.get(*key) != Some(wanted))
    {
        return Err(fail(message));
    }
    Ok(value)
}

fn bind(base: &str, hints: Option<Hints>, selection: Selection<'_>) -> Result<Reference, Error> {
    if hints
        .as_ref()
        .and_then(|h| h.server.as_deref())
        .is_some_and(|server| server.trim_end_matches('/') != base.trim_end_matches('/'))
    {
        return Err(fail(
            "observation state belongs to a different server or Run",
        ));
    }
    let old_project = hints.as_ref().and_then(|h| h.project.as_deref());
    let old_run = hints.as_ref().and_then(|h| h.run_id.as_deref());
    let old_attempt = hints.as_ref().and_then(|h| h.attempt_id.as_deref());
    for (supplied, previous) in [
        (selection.project, old_project),
        (selection.run, old_run),
        (selection.attempt, old_attempt),
    ] {
        if supplied.zip(previous).is_some_and(|(a, b)| a != b) {
            return Err(fail(
                "requested identity differs from the saved exact Run/Attempt",
            ));
        }
    }
    let project = selection
        .project
        .or(old_project)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            fail("provide --project and --run for old state without a complete Run identity")
        })?
        .to_owned();
    let run_id = selection
        .run
        .or(old_run)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            fail("provide --project and --run for old state without a complete Run identity")
        })?
        .to_owned();
    segment(&project)?;
    segment(&run_id)?;
    let attempt_id = selection
        .attempt
        .or(old_attempt)
        .filter(|v| !v.is_empty())
        .map(str::to_owned);
    if let Some(attempt) = &attempt_id {
        validate_attempt(attempt).map_err(|_| fail("choose an exact attempt-NNN identity"))?;
    }
    Ok(Reference {
        server: base.into(),
        project,
        run_id,
        attempt_id,
        state_kind: hints.as_ref().map(|h| h.state_kind),
        saved: hints.map(|h| h.saved).unwrap_or_else(|| json!({})),
        preparation: None,
        submission: None,
        action: None,
    })
}

fn action_identity(action: &Value, reference: &Reference) -> Result<(), Error> {
    let saved = &reference.saved;
    let plan = &saved["plan"];
    let source = attempt(saved.get("source_attempt_id"))?;
    let target = reference
        .attempt_id
        .as_deref()
        .ok_or_else(|| fail("choose an exact attempt-NNN identity"))?;
    let operation = match saved["operation"].as_str() {
        Some("retry") => "RETRY_ATTEMPT",
        Some("cancel") => "CANCEL_RUN",
        _ => return Err(fail("invalid saved training Action identity")),
    };
    let scope = json!({"project":reference.project,"scope_type":"attempt","object_id":format!("{}::{source}",reference.run_id)});
    if !action.is_object()
        || action["operation"] != operation
        || action["scope"] != scope
        || action["run_id"] != reference.run_id
        || action["attempt_id"] != target
        || !action["execution"].is_object()
        || ["intent_digest", "gate_bundle_digest"].iter().any(|key| {
            !action
                .get(*key)
                .and_then(Value::as_str)
                .is_some_and(|v| named_hex(v, "sha256:", 64))
        })
        || ["action_id", "intent_digest", "gate_bundle_digest"]
            .iter()
            .any(|key| action.get(*key) != plan.get(*key))
    {
        return Err(fail(
            "Action identity differs from the saved exact training operation",
        ));
    }
    if operation == "CANCEL_RUN" && target != source
        || operation == "RETRY_ATTEMPT" && target == source
    {
        return Err(fail("saved training Action targets an invalid Attempt"));
    }
    let request = saved
        .get("request")
        .filter(|v| v.is_object())
        .ok_or_else(|| fail("Action target differs from the saved retry request"))?;
    if request
        .get("new_attempt_id")
        .is_some_and(|value| value.as_str() != Some(target))
    {
        return Err(fail("Action target differs from the saved retry request"));
    }
    Ok(())
}

pub fn begin(base: &str, saved: Option<&Value>, selection: Selection<'_>) -> Result<Step, Error> {
    let reference = bind(base, saved.map(reference).transpose()?, selection)?;
    if reference.state_kind == Some(Kind::Experiment) {
        let original = reference
            .saved
            .get("submission")
            .filter(|v| truth(v))
            .cloned();
        if original.is_none() {
            if let Some(previous) = reference.saved.get("preparation").filter(|v| truth(v)) {
                let id = previous
                    .get("preparation_id")
                    .and_then(Value::as_str)
                    .filter(|id| named_hex(id, "preparation-", 32))
                    .ok_or_else(|| fail("Preparation identity differs from the saved experiment"))?
                    .to_owned();
                if previous
                    .get("project")
                    .is_some_and(|v| v.as_str() != Some(&reference.project))
                    || previous
                        .get("run_id")
                        .is_some_and(|v| v.as_str() != Some(&reference.run_id))
                {
                    return Err(fail(
                        "Preparation identity differs from the saved experiment",
                    ));
                }
                let path = format!(
                    "/api/projects/{}/experiment-preparations/{id}",
                    segment(&reference.project)?
                );
                return Ok(Step::Read(Read {
                    path,
                    reference,
                    phase: Phase::Preparation(id),
                }));
            }
        }
        return submission_step(reference, original.as_ref());
    }
    if reference.state_kind == Some(Kind::TrainingOperation) {
        action_identity(&reference.saved["plan"], &reference)?;
        if let Some(action) = reference.saved.get("action").filter(|v| !v.is_null()) {
            action_identity(action, &reference)?;
        }
        let path = format!(
            "/api/actions/{}",
            segment(
                reference.saved["plan"]["action_id"]
                    .as_str()
                    .ok_or_else(|| fail("invalid saved training Action identity"))?
            )?
        );
        return Ok(Step::Read(Read {
            path,
            reference,
            phase: Phase::Action,
        }));
    }
    Ok(Step::Complete(reference))
}

fn submission_step(reference: Reference, original: Option<&Value>) -> Result<Step, Error> {
    let Some(original) = original.filter(|v| truth(v)) else {
        return Ok(Step::Complete(reference));
    };
    let id = action_id(original.get("submission_id"))?;
    for (key, value) in [
        ("project", json!(reference.project)),
        ("run_id", json!(reference.run_id)),
        ("attempt_id", json!(reference.attempt_id)),
    ] {
        if original.get(key).is_some_and(|old| old != &value) {
            return Err(fail("saved Submission belongs to a different Run/Attempt"));
        }
    }
    Ok(Step::Read(Read {
        path: format!("/api/submissions/{}", segment(&id)?),
        reference,
        phase: Phase::Submission(id),
    }))
}

impl Read {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn accept(self, value: Value) -> Result<Step, Error> {
        let mut reference = self.reference;
        match self.phase {
            Phase::Preparation(id) => {
                bound_response(
                    &value,
                    &[
                        ("project", json!(reference.project)),
                        ("run_id", json!(reference.run_id)),
                        ("preparation_id", json!(id)),
                    ],
                    "Preparation identity differs from the saved experiment",
                )?;
                let original = value.get("submission").filter(|v| !v.is_null());
                if let Some(original) = original {
                    bound_response(
                        original,
                        &[
                            ("project", json!(reference.project)),
                            ("run_id", json!(reference.run_id)),
                        ],
                        "prepared Submission belongs to a different Run",
                    )?;
                    let target = attempt(original.get("attempt_id"))?;
                    if reference
                        .attempt_id
                        .as_ref()
                        .is_some_and(|previous| previous != &target)
                    {
                        return Err(fail(
                            "prepared Submission differs from the saved exact Run/Attempt",
                        ));
                    }
                    let previous = reference
                        .saved
                        .get("preparation")
                        .and_then(|p| p.get("submission"))
                        .filter(|v| truth(v));
                    if previous.is_some_and(|previous| {
                        original.get("submission_id") != previous.get("submission_id")
                    }) {
                        return Err(fail(
                            "prepared Submission differs from the saved exact identity",
                        ));
                    }
                    reference.attempt_id = Some(target);
                }
                reference.preparation = Some(value.clone());
                submission_step(reference, original)
            }
            Phase::Submission(id) => {
                bound_response(
                    &value,
                    &[
                        ("project", json!(reference.project)),
                        ("run_id", json!(reference.run_id)),
                        ("attempt_id", json!(reference.attempt_id)),
                        ("submission_id", json!(id)),
                    ],
                    "Submission identity differs from the saved exact Attempt",
                )?;
                reference.submission = Some(value);
                Ok(Step::Complete(reference))
            }
            Phase::Action => {
                action_identity(&value, &reference)?;
                reference.action = Some(value);
                Ok(Step::Complete(reference))
            }
        }
    }
}

/// Effect shell. No source file lookup, state mutation, provider refresh, POST,
/// retry or fallback to a different Attempt is possible on this path.
pub fn resolve(
    client: &Client,
    saved: Option<&Value>,
    selection: Selection<'_>,
) -> Result<Reference, Error> {
    let mut step = begin(client.base_url(), saved, selection)?;
    loop {
        match step {
            Step::Complete(reference) => return Ok(reference),
            Step::Read(read) => {
                let value = client.get(&read.path)?;
                step = read.accept(value)?;
            }
        }
    }
}
