//! Pure request and response planning; one explicit transport call per command.
//! Mutation response loss retains uncertainty and never triggers a second POST.
use crate::{Client, Error, session};
use ml_exp_core::{
    first_submission::validate_gpu_budget, identity, operations::validate_submission_id,
};
use serde_json::{Value, json};

pub enum Command<'a> {
    Prepare {
        project: &'a str,
        run: &'a str,
        max_gpu_hours: f64,
        reason: &'a str,
    },
    Existing {
        project: &'a str,
        run: &'a str,
    },
    Show {
        submission: &'a str,
    },
    Authorize {
        submission: &'a str,
        reason: &'a str,
    },
    Execute {
        submission: &'a str,
        confirmation: &'a str,
    },
    Reconcile {
        submission: &'a str,
    },
}
enum Expected {
    Run { project: String, run: String },
    Submission(String),
}
pub struct Plan {
    path: String,
    body: Option<Vec<u8>>,
    expected: Expected,
    executes: bool,
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
        "invalid submission identity, finite positive budget, reason or exact execution confirmation; no request sent",
    )
}
fn reason(value: &str) -> Result<(), Error> {
    if value.chars().count() > 4000 || value.chars().any(char::is_control) {
        Err(invalid())
    } else {
        Ok(())
    }
}
pub fn plan(command: Command<'_>) -> Result<Plan, Error> {
    let (project, run, body) = match command {
        Command::Prepare {
            project,
            run,
            max_gpu_hours,
            reason: value,
        } => {
            validate_gpu_budget(Some(max_gpu_hours), "budget_cap").map_err(|_| invalid())?;
            reason(value)?;
            (
                project,
                run,
                Some(json!({"max_gpu_hours":max_gpu_hours,"reason":value})),
            )
        }
        Command::Existing { project, run } => (project, run, None),
        command => {
            let (id, suffix, body, executes) = match command {
                Command::Show { submission } => (submission, "", None, false),
                Command::Authorize {
                    submission,
                    reason: value,
                } => {
                    reason(value)?;
                    (
                        submission,
                        "/authorize",
                        Some(json!({"reason":value})),
                        false,
                    )
                }
                Command::Execute {
                    submission,
                    confirmation,
                } => {
                    if confirmation != format!("EXECUTE {submission}") {
                        return Err(invalid());
                    }
                    (
                        submission,
                        "/execute",
                        Some(json!({"confirmation":confirmation})),
                        true,
                    )
                }
                Command::Reconcile { submission } => {
                    (submission, "/reconcile", Some(json!({})), false)
                }
                _ => unreachable!(),
            };
            validate_submission_id(id).map_err(|_| invalid())?;
            return Ok(Plan {
                path: format!("/api/submissions/{id}{suffix}"),
                body: body
                    .map(|v| serde_json::to_vec(&v).map_err(|_| invalid()))
                    .transpose()?,
                expected: Expected::Submission(id.into()),
                executes,
            });
        }
    };
    identity::validate_id(project).map_err(|_| invalid())?;
    identity::validate_id(run).map_err(|_| invalid())?;
    let suffix = if body.is_some() {
        "submissions"
    } else {
        "submission"
    };
    Ok(Plan {
        path: format!("/api/runs/{project}/{run}/{suffix}"),
        body: body
            .map(|v| serde_json::to_vec(&v).map_err(|_| invalid()))
            .transpose()?,
        expected: Expected::Run {
            project: project.into(),
            run: run.into(),
        },
        executes: false,
    })
}
pub fn response(plan: &Plan, value: Value) -> Result<Value, Error> {
    let id = value["submission_id"].as_str().ok_or(Error::Response)?;
    validate_submission_id(id).map_err(|_| Error::Response)?;
    let project = value["project"].as_str().ok_or(Error::Response)?;
    let run = value["run_id"].as_str().ok_or(Error::Response)?;
    identity::validate_id(project).map_err(|_| Error::Response)?;
    identity::validate_id(run).map_err(|_| Error::Response)?;
    if value["attempt_id"] != "attempt-001"
        || value["operation"] != "SUBMIT_RUN"
        || value["confirmation"] != format!("EXECUTE {id}")
        || value["execution"]["status"] != value["status"]
        || !matches!(
            value["status"].as_str(),
            Some(
                "PREPARED"
                    | "AUTHORIZED"
                    | "EXECUTING"
                    | "RECONCILE_REQUIRED"
                    | "VERIFIED"
                    | "BLOCKED"
            )
        )
        || value["execution"]["revision"]
            .as_i64()
            .is_none_or(|revision| revision < 0)
    {
        return Err(Error::Response);
    }
    match &plan.expected {
        Expected::Run {
            project: expected,
            run: expected_run,
        } if project != expected || run != expected_run => return Err(Error::Response),
        Expected::Submission(expected) if id != expected => return Err(Error::Response),
        _ => {}
    }
    Ok(value)
}
pub fn execute(client: &Client, plan: &Plan, health: &Value) -> Result<Value, Error> {
    session::validate_health(health)?;
    if !health["capabilities"]
        .as_array()
        .is_some_and(|caps| caps.iter().any(|v| v == "native-slurm-submissions.v1"))
        || plan.executes && health["scheduler_mutations"] != true
    {
        return Err(Error::Observation(
            "daemon has not enabled this native submission operation; no request sent",
        ));
    }
    client
        .send_submission(plan)
        .and_then(|value| response(plan, value))
        .map_err(|error| {
            if plan.is_mutation() && matches!(error, Error::Transport | Error::Response) {
                Error::SubmissionOutcomeUnknown(Box::new(error))
            } else {
                error
            }
        })
}
