//! Explicit debug-machine requests. Keys stay client-side; only the public key
//! in a validated request is sent. Status never implicitly creates or stops.
use crate::{Client, Error};
use clap::Subcommand;
use serde_json::{Value, json};
use std::{io::Read, path::PathBuf};
#[derive(Subcommand)]
pub enum Command {
    List,
    Create {
        #[arg(long)]
        file: PathBuf,
    },
    Show {
        identity: String,
    },
    Refresh {
        identity: String,
    },
    Stop {
        identity: String,
        #[arg(long)]
        confirmation: String,
    },
    Delete {
        identity: String,
        #[arg(long)]
        confirmation: String,
    },
}
pub struct Plan {
    path: String,
    body: Option<Value>,
    identity: Option<String>,
}
fn invalid() -> Error {
    Error::Observation("invalid debug request or exact confirmation; no request sent")
}
impl Command {
    pub fn plan(&self) -> Result<Plan, Error> {
        let identity = match self {
            Self::List | Self::Create { .. } => None,
            Self::Show { identity }
            | Self::Refresh { identity }
            | Self::Stop { identity, .. }
            | Self::Delete { identity, .. } => Some(identity.clone()),
        };
        if identity.as_ref().is_some_and(|s| {
            s.strip_prefix("debug.")
                .is_none_or(|s| ml_exp_core::sources::validate_archive_digest(s).is_err())
        }) {
            return Err(invalid());
        }
        let (path, body) = match self {
            Self::List => ("/api/debug".into(), None),
            Self::Create { file } => {
                let mut bytes = Vec::new();
                std::fs::File::open(file)
                    .map_err(|_| invalid())?
                    .take(16385)
                    .read_to_end(&mut bytes)
                    .map_err(|_| invalid())?;
                if bytes.len() > 16384 {
                    return Err(invalid());
                }
                let request: ml_exp_core::cloud::debug::Create = serde_json::from_value(
                    Value::Object(ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?),
                )
                .map_err(|_| invalid())?;
                request.validate().map_err(|_| invalid())?;
                ("/api/debug".into(), Some(json!(request)))
            }
            Self::Show { identity } => (format!("/api/debug/{identity}"), None),
            Self::Refresh { identity } => {
                (format!("/api/debug/{identity}/refresh"), Some(json!({})))
            }
            Self::Stop {
                identity,
                confirmation,
            }
            | Self::Delete {
                identity,
                confirmation,
            } => {
                let (op, verb) = if matches!(self, Self::Delete { .. }) {
                    ("delete", "DELETE")
                } else {
                    ("stop", "STOP")
                };
                if confirmation != &format!("{verb} {identity}") {
                    return Err(invalid());
                }
                (
                    format!("/api/debug/{identity}/{op}"),
                    Some(json!({"confirmation":confirmation})),
                )
            }
        };
        Ok(Plan {
            path,
            body,
            identity,
        })
    }
}
pub fn execute(client: &Client, plan: Plan, health: &Value) -> Result<Value, Error> {
    if !health["capabilities"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v == "native-debug-machines.v1"))
    {
        return Err(Error::Observation(
            "server has not configured native debug machines",
        ));
    }
    let value = if let Some(body) = plan.body {
        client.post(&plan.path, &body)?
    } else {
        client.get(&plan.path)?
    };
    if plan.identity.is_some_and(|id| value["debug_id"] != id) {
        return Err(Error::Response);
    }
    Ok(value)
}
