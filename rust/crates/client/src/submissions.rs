//! Pure request and response planning; one explicit transport call per command.
//! Mutation response loss retains uncertainty and never triggers a second POST.
use crate::{Client, Error, session};
use ml_exp_core::{identity, operations::validate_action_id};
use serde_json::{Value, json};

/// The server fills in an Attempt a request leaves out; the same name as the server's.
pub const RUN_CENTRIC: &str = "run-centric-attempts.v1";

pub enum Command<'a> {
    Renew {
        submission: &'a str,
        request_id: &'a str,
        confirmation: &'a str,
    },
    Sync {
        submission: &'a str,
        request_id: &'a str,
        confirmation: &'a str,
    },
    SyncStatus {
        submission: &'a str,
        request_id: &'a str,
    },
    SyncDiagnostic {
        submission: &'a str,
        request_id: &'a str,
        status: bool,
    },
    SyncRetry {
        submission: &'a str,
        request_id: &'a str,
        request: ml_exp_core::cloud::exchange::SyncRetry,
    },
    Discovery {
        project: &'a str,
        run: &'a str,
    },
    Tracking {
        project: &'a str,
        run: &'a str,
    },
    Retry {
        submission: &'a str,
        request: &'a ml_exp_core::first_submission::native::retry::Request,
    },
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
    Execute {
        submission: &'a str,
        confirmation: &'a str,
        reason: &'a str,
    },
    Cancel {
        submission: &'a str,
        confirmation: &'a str,
    },
    History {
        submission: &'a str,
    },
    Refresh {
        submission: &'a str,
    },
    Checkpoint {
        submission: &'a str,
    },
    /// Without an Attempt: the Run's current one, or the one holding `checkpoint`.
    Checkpoints {
        project: &'a str,
        run: &'a str,
        attempt: Option<&'a str>,
        checkpoint: Option<&'a str>,
    },
    Reconcile {
        submission: &'a str,
    },
}
enum Expected {
    Diagnostic(Value),
    Discovery {
        project: String,
        run: String,
    },
    Synchronization {
        submission: String,
        request_id: String,
        failed_job: Option<String>,
    },
    Renewal {
        original: String,
        request_id: String,
    },
    Tracking {
        project: String,
        run: String,
    },
    /// A request that left its Attempt out is checked once the answer names it.
    Retry {
        parent: String,
        request: ml_exp_core::first_submission::native::retry::Request,
    },
    Checkpoints {
        project: String,
        run: String,
        attempt: Option<String>,
        checkpoint: Option<String>,
    },
    Creation {
        project: String,
        run: String,
        runtime: String,
        request: Box<ml_exp_core::first_submission::native::creation::Request>,
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
        "invalid submission identity, numeric metadata, reason or execution confirmation; no request sent",
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
    if let Command::SyncDiagnostic {
        submission,
        request_id,
        status,
    } = &command
    {
        ml_exp_core::cloud::exchange::Sync {
            request_id: (*request_id).into(),
            confirmation: format!("SYNC {submission}"),
        }
        .validate(submission)
        .map_err(|_| invalid())?;
        return Ok(Plan {
            path: format!("/api/submissions/{submission}/sync/{request_id}/diagnostic"),
            body: if *status { None } else { Some(b"{}".to_vec()) },
            expected: Expected::Diagnostic(json!({"action":submission,"request_id":request_id})),
            executes: !*status,
            refreshes: false,
            checkpoint: false,
        });
    }
    if let Command::Discovery { project, run } = &command {
        identity::validate_id(project)
            .and_then(|_| identity::validate_id(run))
            .map_err(|_| invalid())?;
        return Ok(Plan {
            path: format!("/api/runs/{project}/{run}/preparation"),
            body: None,
            expected: Expected::Discovery {
                project: (*project).into(),
                run: (*run).into(),
            },
            executes: false,
            refreshes: false,
            checkpoint: false,
        });
    }
    if let Command::SyncStatus {
        submission,
        request_id,
    }
    | Command::SyncRetry {
        submission,
        request_id,
        ..
    } = &command
    {
        ml_exp_core::cloud::exchange::Sync {
            request_id: (*request_id).into(),
            confirmation: format!("SYNC {submission}"),
        }
        .validate(submission)
        .map_err(|_| invalid())?;
        let retry = match &command {
            Command::SyncRetry { request, .. } => {
                request
                    .validate(submission, request_id)
                    .map_err(|_| invalid())?;
                Some(request)
            }
            _ => None,
        };
        return Ok(Plan {
            path: format!(
                "/api/submissions/{submission}/sync/{request_id}{}",
                if retry.is_some() { "/retry" } else { "" }
            ),
            body: retry
                .map(|r| serde_json::to_vec(r).map_err(|_| invalid()))
                .transpose()?,
            expected: Expected::Synchronization {
                submission: (*submission).into(),
                request_id: (*request_id).into(),
                failed_job: retry.map(|r| r.expected_failed_job_id.clone()),
            },
            executes: retry.is_some(),
            refreshes: false,
            checkpoint: false,
        });
    }
    if let Command::Renew {
        submission,
        request_id,
        confirmation,
    } = &command
    {
        let request = ml_exp_core::renewal::Request {
            request_id: (*request_id).into(),
            confirmation: (*confirmation).into(),
        };
        request.validate(submission).map_err(|_| invalid())?;
        return Ok(Plan {
            path: format!("/api/submissions/{submission}/renew"),
            body: Some(serde_json::to_vec(&request).map_err(|_| invalid())?),
            expected: Expected::Renewal {
                original: (*submission).into(),
                request_id: (*request_id).into(),
            },
            executes: false,
            refreshes: false,
            checkpoint: false,
        });
    }
    if let Command::Tracking { project, run } = &command {
        identity::validate_id(project).map_err(|_| invalid())?;
        identity::validate_id(run).map_err(|_| invalid())?;
        return Ok(Plan {
            path: format!("/api/runs/{project}/{run}/tracking"),
            body: None,
            expected: Expected::Tracking {
                project: (*project).into(),
                run: (*run).into(),
            },
            executes: false,
            refreshes: false,
            checkpoint: false,
        });
    }
    let refreshes = matches!(
        &command,
        Command::History { .. } | Command::Refresh { .. } | Command::Checkpoint { .. }
    );
    let checkpoint = matches!(&command, Command::Checkpoint { .. });
    if let Command::Retry {
        submission,
        request,
    } = &command
    {
        request
            .validate_received(submission)
            .map_err(|_| invalid())?;
        return Ok(Plan {
            path: format!("/api/submissions/{submission}/retry"),
            body: Some(serde_json::to_vec(request).map_err(|_| invalid())?),
            expected: Expected::Retry {
                parent: (*submission).into(),
                request: (*request).clone(),
            },
            executes: false,
            refreshes: false,
            checkpoint: false,
        });
    }
    if let Command::Checkpoints {
        project,
        run,
        attempt,
        checkpoint,
    } = &command
    {
        identity::validate_id(project).map_err(|_| invalid())?;
        identity::validate_id(run).map_err(|_| invalid())?;
        if let Some(attempt) = attempt {
            identity::validate_attempt(attempt).map_err(|_| invalid())?;
        }
        if let Some(id) = checkpoint {
            ml_exp_core::checkpoints::validate_checkpoint_id(id).map_err(|_| invalid())?;
        }
        return Ok(Plan {
            path: format!(
                "/api/runs/{project}/{run}{}/checkpoints{}",
                attempt
                    .map(|attempt| format!("/attempts/{attempt}"))
                    .unwrap_or_default(),
                checkpoint.map(|id| format!("/{id}")).unwrap_or_default()
            ),
            body: None,
            expected: Expected::Checkpoints {
                project: (*project).into(),
                run: (*run).into(),
                attempt: attempt.map(str::to_owned),
                checkpoint: checkpoint.map(str::to_owned),
            },
            executes: false,
            refreshes: false,
            checkpoint: false,
        });
    }
    if let Command::Create { project, request } = &command {
        request
            .identity(project)
            .map_err(|_| Error::Observation("invalid Run creation request; no request sent"))?;
        return Ok(Plan {
            path: format!("/api/projects/{project}/runs"),
            body: Some(serde_json::to_vec(request).map_err(|_| invalid())?),
            expected: Expected::Creation {
                project: (*project).into(),
                run: request.run_id.clone(),
                runtime: request.runtime_id.clone(),
                request: Box::new((*request).clone()),
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
            if !max_gpu_hours.is_finite() {
                return Err(invalid());
            }
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
                Command::Sync {
                    submission,
                    request_id,
                    confirmation,
                } => {
                    let request = ml_exp_core::cloud::exchange::Sync {
                        request_id: request_id.into(),
                        confirmation: confirmation.into(),
                    };
                    request.validate(submission).map_err(|_| invalid())?;
                    (submission, "/sync", Some(json!(request)), true)
                }
                Command::Cancel {
                    submission,
                    confirmation,
                } => {
                    if confirmation != format!("CANCEL {submission}") {
                        return Err(invalid());
                    }
                    (
                        submission,
                        "/cancel",
                        Some(json!({"confirmation":confirmation})),
                        true,
                    )
                }
                Command::Show { submission } => (submission, "", None, false),
                Command::Execute {
                    submission,
                    confirmation,
                    reason: value,
                } => {
                    if confirmation != format!("EXECUTE {submission}") {
                        return Err(invalid());
                    }
                    reason(value)?;
                    (
                        submission,
                        "/execute",
                        Some(json!({"confirmation":confirmation,"reason":value})),
                        true,
                    )
                }
                Command::History { submission } => (submission, "/history", Some(json!({})), false),
                Command::Refresh { submission } => (submission, "/refresh", Some(json!({})), false),
                Command::Checkpoint { submission } => {
                    (submission, "/checkpoint", Some(json!({})), false)
                }
                Command::Reconcile { submission } => {
                    (submission, "/reconcile", Some(json!({})), false)
                }
                _ => unreachable!(),
            };
            validate_action_id(id).map_err(|_| invalid())?;
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
/// The server configures a CPU job's size and limit, so only their shape is checked.
/// That the job holds no GPU is the part a client can rely on.
fn cpu_only_job(sync: &Value) -> bool {
    let positive = |value: &Value| value.as_u64().is_some_and(|n| n > 0);
    sync["resources"]["gpus"] == 0
        && positive(&sync["resources"]["cpus"])
        && positive(&sync["resources"]["memory_gb"])
        && positive(&sync["max_time_seconds"])
}
pub fn diagnostic_response(value: Value, lineage: &Value) -> Result<Value, Error> {
    if value["contract"] != "readonly-diagnostic.v1"
        || value["lineage"] != *lineage
        || value["resources"] != json!({"gpus":0,"cpus":2,"memory_gb":4})
        || value["max_time_seconds"] != 300
        || value["storage_read_only"] != true
        || value["training_replayed"] != false
    {
        return Err(Error::Response);
    }
    Ok(value)
}
/// The reference a request left its Attempt out of, as the answer says it was frozen:
/// the same Run and checkpoint with a valid Attempt. A named Attempt stays as it was.
fn resolved_reference(
    sent: &ml_exp_core::checkpoints::Reference,
    frozen: &Value,
) -> Result<ml_exp_core::checkpoints::Reference, Error> {
    if !sent.unresolved() {
        return Ok(sent.clone());
    }
    let attempt = frozen["attempt_id"].as_str().ok_or(Error::Response)?;
    if frozen["run_id"] != sent.run_id.as_str()
        || frozen["checkpoint_id"] != sent.checkpoint_id.as_str()
        || identity::validate_attempt(attempt).is_err()
    {
        return Err(Error::Response);
    }
    Ok(ml_exp_core::checkpoints::Reference {
        attempt_id: attempt.into(),
        ..sent.clone()
    })
}
fn resolved_retry(
    sent: &ml_exp_core::first_submission::native::retry::Request,
    value: &Value,
) -> Result<ml_exp_core::first_submission::native::retry::Request, Error> {
    let mut request = sent.clone();
    if let Some(reference) = &sent.resume_from {
        request.resume_from = Some(resolved_reference(reference, &value["resume_from"])?);
    }
    if let Some(start) = request
        .restart_from
        .as_mut()
        .filter(|start| start.attempt_id.is_empty())
    {
        if value["restart_from"]["run_id"] != start.run_id.as_str() {
            return Err(Error::Response);
        }
        start.attempt_id = value["restart_from"]["attempt_id"]
            .as_str()
            .ok_or(Error::Response)?
            .into();
    }
    Ok(request)
}
fn resolved_creation(
    sent: &ml_exp_core::first_submission::native::creation::Request,
    value: &Value,
) -> Result<ml_exp_core::first_submission::native::creation::Request, Error> {
    let mut request = sent.clone();
    if let Some(reference) = &sent.resume_from {
        request.resume_from = Some(resolved_reference(reference, &value["resume_from"])?);
    }
    Ok(request)
}
/// Whether the plan leaves an Attempt for the server to choose.
fn leaves_attempt(plan: &Plan) -> bool {
    match &plan.expected {
        Expected::Checkpoints { attempt, .. } => attempt.is_none(),
        Expected::Retry { request, .. } => request.unresolved(),
        Expected::Creation { request, .. } => request
            .resume_from
            .as_ref()
            .is_some_and(ml_exp_core::checkpoints::Reference::unresolved),
        _ => false,
    }
}
pub fn response(plan: &Plan, value: Value) -> Result<Value, Error> {
    if let Expected::Diagnostic(lineage) = &plan.expected {
        return diagnostic_response(value, lineage);
    }
    if let Expected::Discovery { project, run } = &plan.expected {
        if !value["preparation_id"].is_null() {
            let id = value["preparation_id"].as_str().ok_or(Error::Response)?;
            if value["project"] != *project
                || value["run_id"] != *run
                || !id.strip_prefix("preparation-").is_some_and(|s| {
                    s.len() == 32
                        && s.bytes()
                            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                })
                || value["confirmation"] != format!("PREPARE {id}")
                || value["training_authorized"] != false
                || value["scheduler_submitted"] != false
            {
                return Err(Error::Response);
            }
            return Ok(value);
        }
    }
    if let Expected::Synchronization {
        submission,
        request_id,
        failed_job,
    } = &plan.expected
    {
        let sync = &value["synchronization"];
        if value["submission_id"] != *submission
            || sync["request_id"] != *request_id
            || !cpu_only_job(sync)
            || sync["training_replayed"] != false
            || sync["retry_limit"] != 1
            || failed_job
                .as_ref()
                .is_some_and(|id| sync["retry"]["expected_failed_job_id"] != *id)
        {
            return Err(Error::Response);
        }
        return Ok(value);
    }
    if let Expected::Tracking { project, run } = &plan.expected {
        if value["project"] != *project
            || value["run_id"] != *run
            || !value["scopes"].is_array()
            || !value["wandb"]["enabled"].is_boolean()
        {
            return Err(Error::Response);
        }
        return Ok(value);
    }
    if plan.path.ends_with("/history") {
        if !matches!(&plan.expected,Expected::Submission(id) if value["submission_id"]==*id)
            || !matches!(value["status"].as_str(), Some("PARTIAL" | "COMPLETE"))
            || value["scheduler_mutation_performed"] != false
            || value["source"] != "native-shared-metric-history"
            || ["received_bytes", "records", "invalid_lines"]
                .iter()
                .any(|k| value[k].as_u64().is_none())
        {
            return Err(Error::Response);
        }
        return Ok(value);
    }
    if let Expected::Retry { parent, request } = &plan.expected {
        let request = resolved_retry(request, &value)?;
        let request_sha256 = request.identity(parent).map_err(|_| Error::Response)?;
        let (run, attempt) = request.parent().map_err(|_| Error::Response)?;
        let next = ml_exp_core::first_submission::native::retry::next(attempt)
            .map_err(|_| Error::Response)?;
        if value["parent_submission_id"] != *parent
            || value["run_id"] != run
            || value["attempt_id"] != next
            || value["request_sha256"] != request_sha256
            || value["resume_from"] != json!(request.resume_from)
            || value["restart_from"] != json!(request.restart_from)
            || value["max_gpu_hours"] != request.max_gpu_hours
            || value["retry_state"] != "CREATED"
            || value["scheduler_mutation_performed"] != false
            || value["next_action"] != "PREPARE_NEW_ACTION"
            || !value["reused"].is_boolean()
            || value["project"]
                .as_str()
                .is_none_or(|v| identity::validate_id(v).is_err())
            || value["reserved_gpu_seconds"].as_u64().is_none()
            || ["run_sha256", "attempt_sha256"].iter().any(|k| {
                value[k]
                    .as_str()
                    .is_none_or(|v| !ml_exp_core::slurm::helper::valid_digest(v))
            })
        {
            return Err(Error::Response);
        }
        return Ok(value);
    }
    if let Expected::Checkpoints {
        project,
        run,
        attempt,
        checkpoint,
    } = &plan.expected
    {
        // Named by the Run alone, the answer names the Attempt the server chose.
        let attempt = match attempt {
            Some(attempt) => attempt.as_str(),
            None => value["attempt_id"].as_str().ok_or(Error::Response)?,
        };
        identity::validate_attempt(attempt).map_err(|_| Error::Response)?;
        if value["project"] != *project || value["run_id"] != *run || value["attempt_id"] != attempt
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
        request,
    } = &plan.expected
    {
        let request_sha256 = resolved_creation(request, &value)?
            .identity(project)
            .map_err(|_| Error::Response)?;
        if value["project"] != *project
            || value["run_id"] != *run
            || value["attempt_id"] != "attempt-001"
            || value["runtime_id"] != *runtime
            || value["source_id"]
                .as_str()
                .is_none_or(|id| ml_exp_core::sources::validate_source_id(id).is_err())
            || value["request_sha256"] != request_sha256
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
    // Input preparation can need a CPU job before a training Action exists.
    // This response is valid only for the explicit Run prepare request and
    // cannot be used as execute evidence for a Submission.
    if value["operation"] == "PREPARE_INPUT" {
        let Expected::Run { project, run } = &plan.expected else {
            return Err(Error::Response);
        };
        if plan.body.is_none()
            || value["project"] != *project
            || value["run_id"] != *run
            || value["attempt_id"] != "attempt-001"
            || value["phase"] != "DATA_PREPARATION"
            || !value["submission_id"].is_null()
            || value["scheduler_submitted"] != false
            || value["training_authorized"] != false
            || !matches!(
                value["status"].as_str(),
                Some("PREPARED" | "PREPARING" | "FAILED" | "RECONCILE_REQUIRED")
            )
        {
            return Err(Error::Response);
        }
        return Ok(value);
    }
    let id = value["submission_id"].as_str().ok_or(Error::Response)?;
    validate_action_id(id).map_err(|_| Error::Response)?;
    let project = value["project"].as_str().ok_or(Error::Response)?;
    let run = value["run_id"].as_str().ok_or(Error::Response)?;
    identity::validate_id(project).map_err(|_| Error::Response)?;
    identity::validate_id(run).map_err(|_| Error::Response)?;
    if value["attempt_id"]
        .as_str()
        .is_none_or(|v| ml_exp_core::first_submission::native::retry::number(v).is_err())
        || value["operation"] != "SUBMIT_RUN"
        || value["confirmation"] != format!("EXECUTE {id}")
        || value["execution"]["status"] != value["status"]
        || !matches!(
            value["status"].as_str(),
            Some("PREPARED" | "EXECUTING" | "RECONCILE_REQUIRED" | "VERIFIED" | "BLOCKED")
        )
        || value["execution"]["revision"]
            .as_i64()
            .is_none_or(|revision| revision < 0)
    {
        return Err(Error::Response);
    }
    if let Expected::Renewal {
        original,
        request_id,
    } = &plan.expected
    {
        if id == original
            || value["renewed_from"] != *original
            || value["renewal_request_id"] != *request_id
            || value["scheduler_mutation_performed"] != false
        {
            return Err(Error::Response);
        }
    }
    match &plan.expected {
        Expected::Run {
            project: expected,
            run: expected_run,
        }
        | Expected::Discovery {
            project: expected,
            run: expected_run,
        } if project != expected || run != expected_run => return Err(Error::Response),
        Expected::Submission(expected) if id != expected => return Err(Error::Response),
        _ => {}
    }
    // Only Slurm keeps a cancellation record. A cloud job reports through its observation.
    if plan.path.ends_with("/cancel") && value["cloud_observation"].is_null() {
        let state: ml_exp_core::slurm::cancellation::State =
            serde_json::from_value(value["cancellation"].clone()).map_err(|_| Error::Response)?;
        state.validate().map_err(|_| Error::Response)?;
        if value["execution"]["result"]["submission"]["backend_job_id"] != state.job_id {
            return Err(Error::Response);
        }
    }
    if plan.path.ends_with("/sync") {
        let request: ml_exp_core::cloud::exchange::Sync =
            serde_json::from_slice(plan.body.as_deref().ok_or(Error::Response)?)
                .map_err(|_| Error::Response)?;
        if value["synchronization"]["request_id"] != request.request_id
            || !cpu_only_job(&value["synchronization"])
            || value["synchronization"]["training_replayed"] != false
        {
            return Err(Error::Response);
        }
    }
    Ok(value)
}
pub fn execute(client: &Client, plan: &Plan, health: &Value) -> Result<Value, Error> {
    session::validate_health(health)?;
    let creates = matches!(plan.expected, Expected::Creation { .. });
    let capability = if matches!(plan.expected, Expected::Tracking { .. }) {
        "native-tracking-status.v1"
    } else if plan.path.ends_with("/sync")
        || matches!(plan.expected, Expected::Synchronization { .. })
        || matches!(plan.expected, Expected::Diagnostic(_))
    {
        "native-cloud-sync.v1"
    } else if plan.path.ends_with("/history") {
        "native-metric-history.v1"
    } else if matches!(plan.expected, Expected::Retry { .. }) {
        "native-attempt-retry.v1"
    } else if plan.path.ends_with("/cancel") {
        "native-cancellation.v1"
    } else if creates {
        "native-run-creation.v1"
    } else if plan.checkpoint || matches!(plan.expected, Expected::Checkpoints { .. }) {
        "native-checkpoints.v1"
    } else if plan.refreshes {
        "native-worker-observation.v1"
    } else {
        "native-submissions.v1"
    };
    let scheduler_refresh = plan.path.ends_with("/refresh")
        && health["capabilities"]
            .as_array()
            .is_some_and(|caps| caps.iter().any(|v| v == "native-scheduler-observation.v1"));
    let checkpoint_read = matches!(plan.expected, Expected::Checkpoints { .. })
        && health["capabilities"]
            .as_array()
            .is_some_and(|caps| caps.iter().any(|v| v == "checkpoint-record-read.v1"));
    if !(health["capabilities"]
        .as_array()
        .is_some_and(|caps| caps.iter().any(|v| v == capability))
        || scheduler_refresh
        || checkpoint_read)
        || plan.executes && health["scheduler_mutations"] != true
    {
        return Err(Error::Observation(
            "daemon has not enabled this native submission operation; no request sent",
        ));
    }
    if leaves_attempt(plan)
        && !health["capabilities"]
            .as_array()
            .is_some_and(|caps| caps.iter().any(|v| v == RUN_CENTRIC))
    {
        return Err(Error::Observation(
            "this server cannot choose the Attempt; name it with --attempt or its attempt_id; no request sent",
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
