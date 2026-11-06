//! Workspace-local project tracking, separate from research files and jobs.
//! The core chooses transitions; the store owns a recoverable JSON ledger.
pub(crate) mod http;
mod service;
mod store;
pub use service::{Service, transition_context};
pub use store::{Error, Operation, Snapshot, Store, canonical_path};
