//! Provider request compilers. No credentials, clocks, network or persistence.
//! A compiled request is an effect description, never authority to execute it.
pub mod action;
pub mod launch;
mod observation;
pub mod pai;
pub mod run;
pub mod sensecore;
use crate::{canonical, identity, slurm::helper, submission};
pub use observation::{JobObservation, Page, QueueDiagnostic};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid cloud provider configuration, request or exact-job evidence: {0}")]
pub struct Error(pub &'static str);
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Method {
    Get,
    Post,
    Delete,
    Put,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Product {
    PaiDlc,
    PaiDsw,
    PaiStudio,
    AiWorkspace,
}
impl Product {
    pub fn name(self) -> &'static str {
        match self {
            Self::PaiDlc => "pai-dlc",
            Self::PaiDsw => "pai-dsw",
            Self::PaiStudio => "paistudio",
            Self::AiWorkspace => "aiworkspace",
        }
    }
    pub fn endpoint(self) -> &'static str {
        if self == Self::PaiStudio {
            "pai"
        } else {
            self.name()
        }
    }
    pub fn version(self) -> &'static str {
        match self {
            Self::PaiDlc => "2020-12-03",
            Self::PaiDsw => "2022-01-01",
            Self::PaiStudio => "2022-01-12",
            Self::AiWorkspace => "2021-02-04",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
pub enum Endpoint {
    Pai {
        product: Product,
        region: String,
    },
    Sensecore {
        service: String,
        region: Option<String>,
    },
}
impl Endpoint {
    pub fn host(&self) -> Result<String> {
        match self {
            Self::Pai { product, region } => {
                region_id(region)?;
                Ok(format!("{}.{}.aliyuncs.com", product.endpoint(), region))
            }
            Self::Sensecore { service, region } => {
                if !matches!(service.as_str(), "aec2" | "management" | "iam" | "monitor") {
                    return Err(Error("service"));
                }
                match region {
                    Some(region) => {
                        sense_region(region)?;
                        if service != "aec2" {
                            return Err(Error("regional service"));
                        }
                        Ok(format!("{service}.{region}.sensecoreapi.cn"))
                    }
                    None if service != "aec2" => Ok(format!("{service}.sensecoreapi.cn")),
                    _ => Err(Error("service region")),
                }
            }
        }
    }
}
// No Debug: a command or image URI may contain user-authored sensitive values.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub endpoint: Endpoint,
    pub method: Method,
    pub action: String,
    pub path: String,
    pub query: BTreeMap<String, String>,
    pub body: Option<Value>,
    /// Offline log search is a POST with no provider mutation.
    pub read_only: bool,
}
impl Request {
    pub fn validate(&self) -> Result<()> {
        self.endpoint.host()?;
        component(&self.action)?;
        if !self.path.starts_with('/')
            || self.path.len() > 4096
            || self.path.chars().any(char::is_control)
            || self.path.contains(['?', '#', '\\', '%'])
            || self.path.split('/').any(|p| matches!(p, "." | ".."))
            || self.query.len() > 32
            || self.query.iter().any(|(k, v)| {
                component(k).is_err() || v.len() > 16384 || v.chars().any(char::is_control)
            })
            || self.method == Method::Get && self.body.is_some()
            || serde_json::to_vec(&self.body)
                .map_err(|_| Error("body"))?
                .len()
                > submission::MAX_REQUEST_BYTES
        {
            return Err(Error("request"));
        }
        Ok(())
    }
}
pub(super) fn request(
    endpoint: Endpoint,
    method: Method,
    action: &str,
    path: String,
    query: BTreeMap<String, String>,
    body: Option<Value>,
    read_only: bool,
) -> Result<Request> {
    let request = Request {
        endpoint,
        method,
        action: action.into(),
        path,
        query,
        body,
        read_only,
    };
    request.validate()?;
    Ok(request)
}
pub fn component(value: &str) -> Result<&str> {
    if value.is_empty()
        || value.len() > 256
        || !value.as_bytes()[0].is_ascii_alphanumeric()
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
    {
        Err(Error("resource component"))
    } else {
        Ok(value)
    }
}
pub fn region_id(value: &str) -> Result<&str> {
    if value.len() < 3
        || value.len() > 64
        || !value.as_bytes()[0].is_ascii_lowercase()
        || !value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        Err(Error("region"))
    } else {
        Ok(value)
    }
}
pub(super) fn sense_region(value: &str) -> Result<&str> {
    region_id(value)?;
    if !regex::Regex::new(r"^cn-[a-z]+-[0-9]+$")
        .expect("constant region regex")
        .is_match(value)
    {
        return Err(Error("SenseCore region"));
    }
    Ok(value)
}
pub(super) fn absolute_path(value: &str) -> Result<&str> {
    if !value.starts_with('/')
        || value.len() > 4096
        || value.contains(['\0', '\r', '\n', '\\'])
        || value.split('/').any(|p| matches!(p, "." | ".."))
    {
        return Err(Error("mount path"));
    }
    Ok(value)
}
pub(super) fn digest(value: &Value) -> Result<String> {
    canonical::protocol_identity(value).map_err(|_| Error("identity"))
}
pub fn shell_join(args: &[String]) -> Result<String> {
    if args.is_empty()
        || args.len() > 256
        || args.iter().any(|s| s.len() > 65536 || s.contains('\0'))
        || args[0].is_empty()
    {
        return Err(Error("command"));
    }
    let command = args
        .iter()
        .map(|s| format!("'{}'", s.replace('\'', "'\"'\"'")))
        .collect::<Vec<_>>()
        .join(" ");
    if command.len() > 128 * 1024 {
        return Err(Error("command length"));
    }
    Ok(command)
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub project: String,
    pub run_id: String,
    pub attempt_id: String,
    pub submission_token: String,
    pub source_id: String,
    pub image: String,
    pub command: Vec<String>,
    pub gpus: u32,
    pub cpus: u32,
    pub memory_gb: u32,
    pub max_time_seconds: u64,
    #[serde(default = "priority")]
    pub priority: u8,
}
#[cfg(test)]
mod tests;
fn priority() -> u8 {
    5
}
impl Job {
    pub fn validate(&self) -> Result<()> {
        identity::validate_id(&self.project).map_err(|_| Error("project"))?;
        identity::validate_id(&self.run_id).map_err(|_| Error("run"))?;
        identity::validate_attempt(&self.attempt_id).map_err(|_| Error("attempt"))?;
        submission::validate_submission_token(&self.submission_token)
            .map_err(|_| Error("token"))?;
        let source = self
            .source_id
            .strip_prefix("source.")
            .ok_or(Error("source"))?;
        if !helper::valid_digest(&format!("sha256:{source}")) {
            return Err(Error("source"));
        }
        let (repository, hash) = self
            .image
            .split_once("@sha256:")
            .ok_or(Error("pinned image"))?;
        if repository.is_empty()
            || repository.len() > 2048
            || repository.contains(['@', ':', '?', '#'])
            || !repository.contains('/')
            || !repository
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"/._-".contains(&c))
            || !helper::valid_digest(&format!("sha256:{hash}"))
        {
            return Err(Error("pinned image"));
        }
        if self.cpus == 0
            || self.cpus > 65536
            || self.memory_gb == 0
            || self.memory_gb > 1048576
            || self.gpus > 4096
            || self.max_time_seconds == 0
            || self.max_time_seconds > 1000 * 3600
            || !(1..=9).contains(&self.priority)
        {
            return Err(Error("resources"));
        }
        shell_join(&self.command)?;
        Ok(())
    }
    pub fn name(&self) -> Result<String> {
        self.validate()?;
        Ok(format!("ml-exp-{}", self.submission_token))
    }
    pub fn tags(&self, scope: &str) -> Result<Value> {
        self.validate()?;
        if !helper::valid_digest(scope) {
            return Err(Error("scope"));
        }
        Ok(
            json!({"ml-expd.project":self.project,"ml-expd.run":self.run_id,"ml-expd.attempt":self.attempt_id,
            "ml-expd.scope":scope,"ml-expd.submission":self.submission_token}),
        )
    }
}
