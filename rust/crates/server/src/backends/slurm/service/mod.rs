//! Native first-submission service. Trusted configuration and authored files
//! provide execution authority. Client input is a budget/reason or an exact
//! Action confirmation; it cannot supply a script, source receipt or Run path.
pub mod config;
mod engine;
pub(crate) mod http;
pub use engine::{Backend, Context, Error, RuntimeBuilder, Service};

mod artifact_http;
pub(crate) mod dataset_http;
