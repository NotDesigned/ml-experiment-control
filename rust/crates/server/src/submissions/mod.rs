//! Native first-submission service. Trusted configuration and authored files
//! provide execution authority. Client input is a budget/reason or an exact
//! Action confirmation; it cannot supply a script, source receipt or Run path.
pub mod config;
pub(crate) mod http;
mod service;
pub use service::{Backend, Context, Error, Service};

mod artifact_http;
pub(crate) mod dataset_http;
