//! Runtime commands share one bounded request plan, independent of credentials.
use crate::{Client, Error};
use clap::Subcommand;
use serde_json::{Value, json};
use std::{io::Read, path::PathBuf};
#[derive(Subcommand)]
pub enum Command {
    List {
        #[arg(long)]
        project: String,
    },
    /// Freeze a source + Dockerfile build; does not submit training.
    Prepare {
        #[arg(long)]
        project: String,
        #[arg(long)]
        file: PathBuf,
    },
    Show {
        #[arg(long)]
        project: String,
        runtime: String,
    },
    Execute {
        #[arg(long)]
        project: String,
        runtime: String,
        #[arg(long)]
        confirmation: String,
    },
    Materialize {
        #[arg(long)]
        project: String,
        runtime: String,
    },
    Reconcile {
        #[arg(long)]
        project: String,
        runtime: String,
    },
    Logs {
        #[arg(long)]
        project: String,
        runtime: String,
    },
    Progress {
        #[arg(long)]
        project: String,
        runtime: String,
    },
}
pub struct Plan {
    path: String,
    body: Option<Value>,
    project: String,
    runtime: Option<String>,
}
fn invalid() -> Error {
    Error::Observation("invalid runtime identity or request; no request sent")
}
impl Command {
    pub fn plan(&self) -> Result<Plan, Error> {
        let (project, runtime) = match self {
            Self::Prepare { project, .. } | Self::List { project } => (project, None),
            Self::Show { project, runtime }
            | Self::Execute {
                project, runtime, ..
            }
            | Self::Materialize { project, runtime }
            | Self::Reconcile { project, runtime }
            | Self::Logs { project, runtime }
            | Self::Progress { project, runtime } => (project, Some(runtime)),
        };
        ml_exp_core::identity::validate_id(project).map_err(|_| invalid())?;
        if runtime.is_some_and(|id| {
            id.strip_prefix("runtime.")
                .is_none_or(|s| ml_exp_core::sources::validate_archive_digest(s).is_err())
        }) {
            return Err(invalid());
        }
        let base = format!("/api/projects/{project}/runtimes");
        let (path, body) = match self {
            Self::List { .. } => (base, None),
            Self::Prepare { file, .. } => {
                let mut bytes = Vec::new();
                std::fs::File::open(file)
                    .map_err(|_| invalid())?
                    .take(65537)
                    .read_to_end(&mut bytes)
                    .map_err(|_| invalid())?;
                if bytes.len() > 65536 {
                    return Err(invalid());
                }
                let definition: ml_exp_core::cloud::build::Definition = serde_json::from_value(
                    Value::Object(ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?),
                )
                .map_err(|_| invalid())?;
                definition.validate().map_err(|_| invalid())?;
                (base, Some(json!(definition)))
            }
            Self::Execute {
                runtime,
                confirmation,
                ..
            } => {
                if confirmation != &format!("BUILD {runtime}") {
                    return Err(invalid());
                }
                (
                    format!("{base}/{runtime}/execute"),
                    Some(json!({"confirmation":confirmation})),
                )
            }
            Self::Materialize { runtime, .. } => {
                (format!("{base}/{runtime}/materialize"), Some(json!({})))
            }
            Self::Reconcile { runtime, .. } => {
                (format!("{base}/{runtime}/reconcile"), Some(json!({})))
            }
            Self::Show { runtime, .. } => (format!("{base}/{runtime}"), None),
            Self::Logs { runtime, .. } => (format!("{base}/{runtime}/logs"), None),
            Self::Progress { runtime, .. } => (format!("{base}/{runtime}/progress"), None),
        };
        Ok(Plan {
            path,
            body,
            project: project.clone(),
            runtime: runtime.cloned(),
        })
    }
}
pub fn execute(client: &Client, plan: Plan, health: &Value) -> Result<Value, Error> {
    if !health["capabilities"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v == "native-runtime-builds.v1"))
    {
        return Err(Error::Observation(
            "server has not configured native runtime builds",
        ));
    }
    let value = match plan.body {
        Some(body) => client.post(&plan.path, &body)?,
        None => client.get(&plan.path)?,
    };
    if plan
        .runtime
        .as_ref()
        .is_some_and(|id| value["runtime_id"] != *id)
        || value.get("project").is_some_and(|p| p != &plan.project)
    {
        return Err(Error::Response);
    }
    Ok(value)
}
