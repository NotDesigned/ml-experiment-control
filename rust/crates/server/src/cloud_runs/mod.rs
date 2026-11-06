//! Public Run/Action composition for native cloud adapters. Configuration,
//! filesystem authority and external effects stay outside the pure compilers.
mod build_http;
mod config;
mod http;
mod observer;
mod service;
pub(crate) use build_http::routes as build_routes;
pub(crate) use config::RUN_CREATION_REQUIRES_EXCHANGE;
pub use config::{Definition, Settings};
pub(crate) fn routes() -> axum::Router<std::sync::Arc<crate::AppState>> {
    http::routes().merge(sync_http::routes())
}
pub use service::{Continue, ExternalExecutors, Service};

mod debug_http;
mod sync_http;
pub(crate) use debug_http::routes as debug_routes;
