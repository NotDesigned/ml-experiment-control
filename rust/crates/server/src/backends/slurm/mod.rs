//! The Slurm backend. `transport` is one bounded SSH invocation of the native
//! Slurm helper: the caller must have persisted its immutable intent
//! before calling, and it does not select Attempts, consume budgets, retry or
//! execute other command plans. `action` runs that intent; `service` sits above.
mod transport;
pub use transport::*;
pub mod action;
pub mod service;

#[cfg(test)]
mod tests;

pub(crate) mod artifact_stream;

/// Trusted operator settings for reaching one helper over SSH. The Slurm service
/// and the cloud builder both read this shape.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ssh {
    pub binary: std::path::PathBuf,
    pub config_file: std::path::PathBuf,
    pub known_hosts_file: std::path::PathBuf,
    pub helper_path: String,
    pub helper_config_path: String,
    pub local_cwd: std::path::PathBuf,
    pub agent_socket: Option<std::path::PathBuf>,
    pub timeout_seconds: u32,
    pub connect_timeout_seconds: u32,
}
