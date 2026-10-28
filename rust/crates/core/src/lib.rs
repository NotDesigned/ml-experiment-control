//! Native shared contracts. No scheduler I/O, configuration loading or Python runtime.
#![forbid(unsafe_code)]
#![deny(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]
pub mod artifacts;
pub mod auth;
pub mod canonical;
pub mod capabilities;
pub mod checkpoints;
pub mod diagnostic;
pub mod durable;
pub mod failures;
pub mod first_submission;
pub mod identity;
pub mod index;
pub mod input_progress;
pub mod launch;
pub mod manifest;
pub mod metrics;
pub mod multipart;
pub mod observation;
pub mod operation_catalog;
pub mod operations;
pub mod projects;
pub mod records;
pub mod result_collection;
pub mod slurm;
pub mod sources;
pub mod state;
pub mod submission;
pub mod tracking;
pub mod transfers;

pub const API_PROTOCOL_VERSION: u32 = 2;
pub const CLIENT_PROTOCOL_HEADER: &str = "X-ML-Expd-Client-Protocol";
