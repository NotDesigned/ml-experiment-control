//! Action persistence I/O shell. No scheduler/provider execution is implicit.
mod database;
mod store;
pub use database::{Database, Error, Result};
pub use store::{ClaimContext, Claimed, CommandPhase, Store};

#[cfg(test)]
mod tests;

/// Provider-neutral reviewed execution command. Credentials and provider
/// effects never become part of an approval request.
pub struct Execute<'a> {
    pub action_id: &'a str,
    pub confirmation: &'a str,
    pub policy: &'a ml_exp_core::operations::Policy,
    pub files: &'a ml_exp_core::operations::FrozenFiles,
    pub claim: ClaimContext<'a>,
}
