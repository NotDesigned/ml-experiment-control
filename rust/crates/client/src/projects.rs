//! Project request planning is pure. Only `execute` sends the already validated
//! typed request; it never retries a mutation or chooses a new project identity.
use crate::{Client, Error, session};
use ml_exp_core::{
    identity,
    projects::{LifecycleRecord, LifecycleState},
};
use serde_json::{Value, json};
use std::collections::BTreeSet;

pub const MAX_REQUEST_BYTES: usize = 64 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Pause,
    Resume,
    Archive,
    Restore,
    Unregister,
}
impl Action {
    fn name(self) -> &'static str {
        match self {
            Self::Pause => "pause",
            Self::Resume => "resume",
            Self::Archive => "archive",
            Self::Restore => "restore",
            Self::Unregister => "unregister",
        }
    }
    fn state(self) -> Option<LifecycleState> {
        match self {
            Self::Pause | Self::Restore => Some(LifecycleState::Paused),
            Self::Resume => Some(LifecycleState::Active),
            Self::Archive => Some(LifecycleState::Archived),
            Self::Unregister => None,
        }
    }
}
/// A daemon path refers to the server filesystem, never a client-local upload.
pub enum Command<'a> {
    List,
    Register {
        manifest_path: &'a str,
        expected_revision: Option<u64>,
    },
    Change {
        project: &'a str,
        action: Action,
        reason: &'a str,
        expected_revision: Option<u64>,
    },
    UnregisterAll {
        reason: &'a str,
        expected_revision: Option<u64>,
    },
}
enum Expected {
    List,
    Register { manifest_path: String },
    Change { project: String, action: Action },
    UnregisterAll,
}
/// Routes and encoded bodies cannot be changed by callers. There is no generic
/// public POST constructor; this type represents project lifecycle only.
pub struct Plan {
    path: String,
    body: Option<Vec<u8>>,
    expected: Expected,
}
impl Plan {
    pub(crate) fn path(&self) -> &str {
        &self.path
    }
    pub(crate) fn body(&self) -> Option<&[u8]> {
        self.body.as_deref()
    }
    pub fn is_mutation(&self) -> bool {
        self.body.is_some()
    }
}
fn invalid() -> Error {
    Error::Observation(
        "invalid project operation; use a safe project identity, an absolute daemon manifest path, and a reason of at most 4000 characters",
    )
}
fn reason(value: &str) -> Result<(), Error> {
    if value.chars().count() > 4000 {
        Err(invalid())
    } else {
        Ok(())
    }
}
fn has_reason(value: &str) -> bool {
    !value
        .trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'))
        .is_empty()
}
pub fn plan(command: Command<'_>) -> Result<Plan, Error> {
    let (path, body, expected, revision) = match command {
        Command::List => {
            return Ok(Plan {
                path: "/api/project-lifecycle".into(),
                body: None,
                expected: Expected::List,
            });
        }
        Command::Register {
            manifest_path,
            expected_revision,
        } => {
            if !manifest_path.starts_with('/')
                || manifest_path.chars().count() > 4096
                || manifest_path.contains('\0')
            {
                return Err(invalid());
            }
            (
                "/api/project-lifecycle/register".into(),
                json!({"source":{"kind":"daemon_path","manifest_path":manifest_path}}),
                Expected::Register {
                    manifest_path: manifest_path.into(),
                },
                expected_revision,
            )
        }
        Command::Change {
            project,
            action,
            reason: value,
            expected_revision,
        } => {
            identity::validate_id(project).map_err(|_| invalid())?;
            reason(value)?;
            if action == Action::Archive && !has_reason(value) {
                return Err(Error::Observation(
                    "archiving a project requires a nonempty reason; no request was sent",
                ));
            }
            (
                format!("/api/project-lifecycle/{project}/{}", action.name()),
                json!({"reason":value}),
                Expected::Change {
                    project: project.into(),
                    action,
                },
                expected_revision,
            )
        }
        Command::UnregisterAll {
            reason: value,
            expected_revision,
        } => {
            reason(value)?;
            (
                "/api/project-lifecycle/unregister-all".into(),
                json!({"reason":value}),
                Expected::UnregisterAll,
                expected_revision,
            )
        }
    };
    let mut body = body;
    if let Some(revision) = revision {
        body["expected_revision"] = json!(revision);
    }
    let bytes = serde_json::to_vec(&body).map_err(|_| invalid())?;
    if bytes.len() > MAX_REQUEST_BYTES {
        return Err(Error::Observation(
            "project request exceeds the 64 KiB limit; no request was sent",
        ));
    }
    Ok(Plan {
        path,
        body: Some(bytes),
        expected,
    })
}
fn record(value: &Value) -> Result<LifecycleRecord, Error> {
    let record = LifecycleRecord::from_value(value).map_err(|_| Error::Response)?;
    if !record.project_file.starts_with('/') || record.project_file.contains('\0') {
        return Err(Error::Response);
    }
    Ok(record)
}
fn record_array(value: &Value) -> Result<(), Error> {
    let values = value.as_array().ok_or(Error::Response)?;
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(record(value)?.project) {
            return Err(Error::Response);
        }
    }
    Ok(())
}
fn response(plan: &Plan, value: Value) -> Result<Value, Error> {
    let fields = value.as_object().ok_or(Error::Response)?;
    match &plan.expected {
        Expected::List => record_array(fields.get("projects").ok_or(Error::Response)?)?,
        Expected::Register { manifest_path } => {
            let result = record(fields.get("project").ok_or(Error::Response)?)?;
            if result.state != LifecycleState::Active || result.project_file != *manifest_path {
                return Err(Error::Response);
            }
        }
        Expected::Change { project, action } => {
            let result = record(
                fields
                    .get(if *action == Action::Unregister {
                        "unregistered"
                    } else {
                        "project"
                    })
                    .ok_or(Error::Response)?,
            )?;
            if result.project != *project
                || action.state().is_some_and(|state| result.state != state)
            {
                return Err(Error::Response);
            }
        }
        Expected::UnregisterAll => {
            record_array(fields.get("unregistered").ok_or(Error::Response)?)?
        }
    }
    Ok(value)
}
/// Health is an explicit authenticated observation, not an inferred capability.
/// On a lost or malformed mutation response, query the lifecycle before deciding
/// whether a new request is appropriate; this function never replays it.
pub fn execute(client: &Client, plan: &Plan, health: &Value) -> Result<Value, Error> {
    session::validate_health(health)?;
    if !health
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|values| values.iter().any(|v| v == "project-lifecycle.v1"))
    {
        return Err(Error::Observation(
            "daemon has not enabled project-lifecycle.v1; no project request was sent",
        ));
    }
    client
        .send_project(plan)
        .and_then(|value| response(plan, value))
        .map_err(|error| {
            if plan.is_mutation() && matches!(error, Error::Transport | Error::Response) {
                Error::ProjectOutcomeUnknown(Box::new(error))
            } else {
                error
            }
        })
}
