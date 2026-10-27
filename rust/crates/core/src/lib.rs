//! Native shared contracts. No scheduler I/O, configuration loading or Python runtime.
#![forbid(unsafe_code)]
#![deny(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]
pub mod auth;
pub mod canonical;
pub mod diagnostic;
pub mod durable;
pub mod failures;
pub mod identity;
pub mod index;
pub mod launch;
pub mod manifest;
pub mod metrics;
pub mod observation;
pub mod operations;
pub mod projects;
pub mod records;
pub mod slurm;
pub mod state;
pub mod tracking;

pub const API_PROTOCOL_VERSION: u32 = 2;
pub const CLIENT_PROTOCOL_HEADER: &str = "X-ML-Expd-Client-Protocol";
