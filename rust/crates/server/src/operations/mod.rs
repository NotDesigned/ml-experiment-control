//! Action persistence I/O shell. No scheduler/provider execution is implicit.
mod database;
mod store;
pub use database::{Database, Error, Result};
pub use store::{ClaimContext, Claimed, CommandPhase, Store};

#[cfg(test)]
mod tests;
