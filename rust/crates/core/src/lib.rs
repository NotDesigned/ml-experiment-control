//! Native shared contracts. No scheduler I/O, configuration loading or Python runtime.
#![forbid(unsafe_code)]
#![deny(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]
pub mod artifacts;
pub mod auth;
pub mod canonical;
pub mod checkpoints;
pub mod cloud;
pub mod data_preparation;
pub mod datasets;
pub mod diagnostic;
pub mod distribution;
pub mod durable;
pub mod environment;
pub mod failures;
pub mod first_submission;
pub mod identity;
pub mod input_progress;
pub mod json;
pub mod launch;
pub mod local;
pub mod managed;
pub mod metrics;
pub mod multipart;
pub mod observation;
pub mod operations;
pub mod polling;
pub mod projects;
pub mod records;
pub mod retry;
pub mod slurm;
pub mod sources;
pub mod state;
pub mod submission;
pub mod tracking;

pub mod capacity;
pub mod maintenance;
pub mod preparation;

pub mod renewal;
