//! Pure Slurm protocol domain: validate inputs, plan commands, and reduce evidence.
//!
//! No SSH, scheduler, filesystem, clock, environment, random or process access is
//! performed here. A command plan is not authorization to execute it. The server
//! must durably persist and review intents, hold its leases, execute each effect
//! once, and reconcile uncertain outcomes before preparing another effect.

mod allocation;
mod config;
mod observation;
mod plans;

pub use allocation::*;
pub use config::*;
pub use observation::*;
pub use plans::*;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid Slurm configuration field: {0}")]
    Configuration(&'static str),
    #[error("invalid Slurm identity: {0}")]
    Identity(&'static str),
    #[error("malformed Slurm evidence: {0}")]
    Evidence(&'static str),
    #[error("Slurm effect needs exact owned identity")]
    Ownership,
    #[error("Slurm scheduler identity is ambiguous or conflicting")]
    Ambiguous,
    #[error("Slurm evidence is unavailable; do not replay effects")]
    Unavailable,
}

pub(crate) fn printable(value: &str, limit: usize) -> bool {
    !value.is_empty() && value.len() <= limit && !value.chars().any(char::is_control)
}

pub(crate) fn component(value: &str) -> bool {
    value.len() <= 128
        && value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
}
