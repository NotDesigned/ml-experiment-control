//! Native shared contracts. No scheduler I/O, configuration loading or Python runtime.
#![forbid(unsafe_code)]
#![deny(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]
pub mod auth;
pub mod identity;
pub mod metrics;
pub mod state;

pub const API_PROTOCOL_VERSION: u32 = 2;
pub const CLIENT_PROTOCOL_HEADER: &str = "X-ML-Expd-Client-Protocol";
