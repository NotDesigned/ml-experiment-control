//! Public Run/Action composition for native cloud adapters. Configuration,
//! filesystem authority and external effects stay outside the pure compilers.
mod config;
mod service;
pub use config::{Definition, Settings};
pub use service::Service;
pub(crate) use service::failure;
