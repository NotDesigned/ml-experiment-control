//! Pure request and response planning; one explicit transport call per command.
//! Mutation response loss retains uncertainty and never triggers a second POST.
use crate::{Client, Error, session};
use ml_exp_core::{
    first_submission::validate_gpu_budget, identity, operations::validate_submission_id,
};
use serde_json::{Value, json};

pub enum Command<'a> {
    Create {
        project: &'a str,
        request: &'a ml_exp_core::first_submission::native::creation::Request,
    },
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
    Refresh {
        submission: &'a str,
    },
    Checkpoint {
        submission: &'a str,
    },
    Checkpoints {
        project: &'a str,
        run: &'a str,
        attempt: &'a str,
        checkpoint: Option<&'a str>,
    },
    Reconcile {
        submission: &'a str,
    },
}
enum Expected {
    Checkpoints {
        project: String,
        run: String,
        attempt: String,
        checkpoint: Option<String>,
    },
    Creation {
        project: String,
        run: String,
        runtime: String,
        request_sha256: String,
    },
    Run {
        project: String,
        run: String,
    },
    Submission(String),
}
pub struct Plan {
    path: String,
    body: Option<Vec<u8>>,
    expected: Expected,
    executes: bool,
    refreshes: bool,
    checkpoint: bool,
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
    let refreshes = matches!(
        &command,
        Command::Refresh { .. } | Command::Checkpoint { .. }
    );
    let checkpoint = matches!(&command, Command::Checkpoint { .. });
    if let Command::Checkpoints {
        project,
        run,
        attempt,
        checkpoint,
    } = &command
    {
        identity::validate_id(project).map_err(|_| invalid())?;
        identity::validate_id(run).map_err(|_| invalid())?;
        identity::validate_attempt(attempt).map_err(|_| invalid())?;
        if let Some(id) = checkpoint {
            ml_exp_core::checkpoints::validate_checkpoint_id(id).map_err(|_| invalid())?;
        }
        return Ok(Plan {
            path: format!(
                "/api/runs/{project}/{run}/attempts/{attempt}/checkpoints{}",
                checkpoint.map(|id| format!("/{id}")).unwrap_or_default()
            ),
            body: None,
            expected: Expected::Checkpoints {
                project: (*project).into(),
                run: (*run).into(),
                attempt: (*attempt).into(),
                checkpoint: checkpoint.map(str::to_owned),
            },
            executes: false,
            refreshes: false,
            checkpoint: false,
        });
    }
    if let Command::Create { project, request } = &command {
        let request_sha256 = request
            .identity(project)
            .map_err(|_| Error::Observation("invalid Run creation request; no request sent"))?;
        return Ok(Plan {
            path: format!("/api/projects/{project}/runs"),
            body: Some(serde_json::to_vec(request).map_err(|_| invalid())?),
            expected: Expected::Creation {
                project: (*project).into(),
                run: request.run_id.clone(),
                runtime: request.runtime_id.clone(),
                request_sha256,
            },
            executes: false,
            refreshes,
            checkpoint,
        });
    }
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
                Command::Refresh { submission } => (submission, "/refresh", Some(json!({})), false),
                Command::Checkpoint { submission } => {
                    (submission, "/checkpoint", Some(json!({})), false)
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
                refreshes,
                checkpoint,
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
        refreshes,
        checkpoint,
    })
}
pub fn response(plan: &Plan, value: Value) -> Result<Value, Error> {
    if let Expected::Checkpoints {
        project,
        run,
        attempt,
        checkpoint,
    } = &plan.expected
    {
        if value["project"] != *project
            || value["run_id"] != *run
            || value["attempt_id"] != *attempt
        {
            return Err(Error::Response);
        }
        let rows = if checkpoint.is_some() {
            vec![&value]
        } else {
            value["checkpoints"]
                .as_array()
                .ok_or(Error::Response)?
                .iter()
                .collect()
        };
        for row in rows {
            let (context, _) =
                ml_exp_core::checkpoints::validate_document(row, None, checkpoint.as_deref())
                    .map_err(|_| Error::Response)?;
            if context.project != *project
                || context.run_id != *run
                || context.attempt_id != *attempt
                || row["status"] != "REGISTERED"
            {
                return Err(Error::Response);
            }
        }
        return Ok(value);
    }
    if plan.checkpoint {
        ml_exp_core::checkpoints::validate_document(&value["checkpoint"], None, None)
            .map_err(|_| Error::Response)?;
        if !matches!(&plan.expected,Expected::Submission(id) if value["submission_id"]==*id)
            || value["checkpoint"]["status"] != "REGISTERED"
            || value["scheduler_mutation_performed"] != false
        {
            return Err(Error::Response);
        }
        return Ok(value);
    }
    if let Expected::Creation {
        project,
        run,
        runtime,
        request_sha256,
    } = &plan.expected
    {
        if value["project"] != *project
            || value["run_id"] != *run
            || value["attempt_id"] != "attempt-001"
            || value["runtime_id"] != *runtime
            || value["source_id"]
                .as_str()
                .is_none_or(|id| ml_exp_core::sources::validate_source_id(id).is_err())
            || value["request_sha256"] != *request_sha256
            || value["creation_state"] != "CREATED"
            || value["scheduler_mutation_performed"] != false
            || !value["reused"].is_boolean()
            || !["run_sha256", "attempt_sha256", "protocol_id"]
                .iter()
                .all(|key| {
                    value[key]
                        .as_str()
                        .is_some_and(ml_exp_core::slurm::helper::valid_digest)
                })
        {
            return Err(Error::Response);
        }
        return Ok(value);
    }
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
    let creates = matches!(plan.expected, Expected::Creation { .. });
    let capability = if creates {
        "native-run-creation.v1"
    } else if plan.checkpoint || matches!(plan.expected, Expected::Checkpoints { .. }) {
        "native-checkpoints.v1"
    } else if plan.refreshes {
        "native-worker-observation.v1"
    } else {
        "native-slurm-submissions.v1"
    };
    if !health["capabilities"]
        .as_array()
        .is_some_and(|caps| caps.iter().any(|v| v == capability))
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
            if plan.is_mutation()
                && !plan.refreshes
                && matches!(error, Error::Transport | Error::Response)
            {
                if creates {
                    Error::RunCreationOutcomeUnknown(Box::new(error))
                } else {
                    Error::SubmissionOutcomeUnknown(Box::new(error))
                }
            } else {
                error
            }
        })
}
