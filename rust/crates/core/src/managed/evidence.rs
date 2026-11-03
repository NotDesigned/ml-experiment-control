//! Pure supervised-process evidence and bounded remote observation contracts.
use crate::launch;
use serde::{Deserialize, Serialize};
pub const CONTRACT: &str = "rust-managed-launch.v1";
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Phase {
    Prepared,
    Started,
    Finished,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Diagnostic {
    pub stage: String,
    pub code: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessEvidence {
    pub exit_code: Option<i32>,
    pub termination_signal: Option<i32>,
    pub timed_out: bool,
    pub cancelled: bool,
    pub error: Option<Diagnostic>,
    pub stdout_sha256: Option<String>,
    pub stderr_sha256: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Training {
    Completed,
    Failed,
    TimedOut,
    Cancelled,
    Unknown,
}
pub fn training(process: &ProcessEvidence) -> Training {
    if process.timed_out {
        Training::TimedOut
    } else if process.cancelled {
        Training::Cancelled
    } else if process.exit_code == Some(0) {
        Training::Completed
    } else if process.exit_code.is_some() || process.termination_signal.is_some() {
        Training::Failed
    } else {
        Training::Unknown
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Publication {
    NotImplemented,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub contract: String,
    pub identity: launch::Identity,
    pub manifest_sha256: String,
    pub process: ProcessEvidence,
    pub training: Training,
    pub publication: Publication,
    pub download_available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_environment_sha256: Option<String>,
}
impl Receipt {
    /// Success covers the supervised local process and verified captured logs.
    /// It makes no assertion that outputs were archived, uploaded or published.
    pub fn succeeded(&self) -> bool {
        self.training == Training::Completed
            && self.process.error.is_none()
            && self.process.stdout_sha256.is_some()
            && self.process.stderr_sha256.is_some()
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub contract: String,
    pub identity: launch::Identity,
    pub manifest_sha256: String,
    pub phase: Phase,
    /// Even PREPARED is not a statement authorizing automatic replay.
    pub safe_to_retry: bool,
    pub receipt: Option<Receipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_environment_sha256: Option<String>,
}
