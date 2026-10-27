//! Read-only authoritative metadata. Files are observations, not scheduler
//! authorization; raw SHA binds file bytes, not the legacy scientific digest.
//! No index, workspace bootstrap, mutation, provider or latest-Attempt lookup.
#![forbid(unsafe_code)]
mod domain;
pub mod scanner;
pub mod snapshot;
mod store;

pub use domain::{
    Document, DocumentError, Format, ManifestSource, Provenance, parse_mapping, validate_attempt,
    validate_project, validate_run,
};
pub use scanner::{Discovery, DiscoveryError, DiscoveryLimits, LocatedRun};
pub use snapshot::{
    AttemptSnapshot, Diagnostic, EvidenceDocument, EvidenceProblem, LogTail, MetricTail, Snapshot,
};
pub use store::{AttemptRecord, Error, ExpectedHashes, ProjectBinding, RunRef, Store};

#[cfg(test)]
mod tests;
