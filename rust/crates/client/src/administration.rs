use crate::{Client, Error};
use clap::Subcommand;
use serde_json::{Value, json};
#[derive(Subcommand)]
pub enum Command {
    /// Prepare removal of one terminal Attempt's outputs or persistent state.
    Plan {
        #[arg(long)]
        submission: String,
        #[arg(long)]
        request_id: String,
        #[arg(long,value_parser=["outputs","state","checkpoint"])]
        kind: String,
        #[arg(long, required_if_eq("kind", "checkpoint"))]
        checkpoint: Option<String>,
    },
    Show {
        identity: String,
    },
    Execute {
        identity: String,
        #[arg(long)]
        confirmation: String,
    },
    /// Retire a terminal Run registration, keeping small audit records.
    Retire {
        #[arg(long)]
        project: String,
        #[arg(long)]
        run: String,
        #[arg(long)]
        confirmation: String,
    },
}
pub struct Plan {
    path: String,
    body: Option<Value>,
}
fn invalid() -> Error {
    Error::Observation(
        "invalid administrative selection, identity or confirmation; no request sent",
    )
}
impl Command {
    pub fn plan(&self) -> Result<Plan, Error> {
        let (path, body) = match self {
            Self::Plan {
                submission,
                request_id,
                kind,
                checkpoint,
            } => {
                ml_exp_core::operations::validate_submission_id(submission)
                    .map_err(|_| invalid())?;
                ml_exp_core::submission::validate_submission_token(request_id)
                    .map_err(|_| invalid())?;
                let selection = if kind == "checkpoint" {
                    let checkpoint = checkpoint.as_ref().ok_or_else(invalid)?;
                    ml_exp_core::checkpoints::validate_checkpoint_id(checkpoint)
                        .map_err(|_| invalid())?;
                    json!({"kind":kind,"checkpoint_id":checkpoint})
                } else if matches!(kind.as_str(), "state" | "outputs") && checkpoint.is_none() {
                    json!({"kind":kind})
                } else {
                    return Err(invalid());
                };
                (
                    format!("/api/admin/submissions/{submission}/maintenance"),
                    Some(json!({"request_id":request_id,"selection":selection})),
                )
            }
            Self::Show { identity } | Self::Execute { identity, .. } => {
                ml_exp_core::sources::validate_archive_digest(identity).map_err(|_| invalid())?;
                if let Self::Execute { confirmation, .. } = self {
                    if confirmation != &format!("DELETE {identity}") {
                        return Err(invalid());
                    }
                    (
                        format!("/api/admin/maintenance/{identity}/execute"),
                        Some(json!({"confirmation":confirmation})),
                    )
                } else {
                    (format!("/api/admin/maintenance/{identity}"), None)
                }
            }
            Self::Retire {
                project,
                run,
                confirmation,
            } => {
                ml_exp_core::identity::validate_id(project)
                    .and_then(|_| ml_exp_core::identity::validate_id(run))
                    .map_err(|_| invalid())?;
                if confirmation != &format!("DELETE {project}/{run}") {
                    return Err(invalid());
                }
                (
                    format!("/api/admin/runs/{project}/{run}/retire"),
                    Some(json!({"confirmation":confirmation})),
                )
            }
        };
        Ok(Plan { path, body })
    }
}
pub fn execute(client: &Client, plan: Plan, health: &Value) -> Result<Value, Error> {
    if !health["capabilities"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v == "native-administration.v1"))
    {
        return Err(Error::Observation(
            "server administrative deletion is not enabled",
        ));
    }
    if let Some(body) = plan.body {
        client.post(&plan.path, &body)
    } else {
        client.get(&plan.path)
    }
}
