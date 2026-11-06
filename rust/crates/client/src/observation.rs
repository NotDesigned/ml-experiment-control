//! Saved-identity observation as a pure request/response state machine. Planning
//! returns GET requests; only `resolve` below executes the HTTP boundary.
use crate::{Client, Error};
use ml_exp_core::identity::validate_attempt;
use serde::Serialize;
use serde_json::{Value, json};

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
    pub saved: Value,
}
#[derive(Clone, Serialize)]
pub struct Reference {
    pub server: String,
    pub project: String,
    pub run_id: String,
    pub attempt_id: Option<String>,
    pub saved: Value,
    pub preparation: Option<Value>,
    pub submission: Option<Value>,
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

/// API read IDs have a broader grammar than managed identities. They are never
/// reused as filesystem paths or as managed resource identities.
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
        return Err(fail("observation requires a saved experiment state"));
    }
    optional_hint(saved, "server")
}

pub fn reference(saved: &Value) -> Result<Hints, Error> {
    let server = server_hint(saved)?;
    if saved.get("config_sha256").is_none() {
        return Err(fail("observation requires an experiment state"));
    }
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
            fail("provide --project and --run for a saved state without a complete Run identity")
        })?
        .to_owned();
    let run_id = selection
        .run
        .or(old_run)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            fail("provide --project and --run for a saved state without a complete Run identity")
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
        saved: hints.map(|h| h.saved).unwrap_or_else(|| json!({})),
        preparation: None,
        submission: None,
    })
}

pub fn begin(base: &str, saved: Option<&Value>, selection: Selection<'_>) -> Result<Step, Error> {
    let reference = bind(base, saved.map(reference).transpose()?, selection)?;
    if saved.is_some() {
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
