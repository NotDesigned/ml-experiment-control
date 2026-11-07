//! Public Run/Action composition for native cloud adapters. Configuration,
//! filesystem authority and external effects stay outside the pure compilers.
mod build_http;
mod config;
mod engine;
mod http;
mod observer;
pub(crate) use build_http::routes as build_routes;
pub(crate) use config::{BACKEND_RETIRED, RUN_CREATION_REQUIRES_EXCHANGE};
pub use config::{Definition, Settings};
pub(crate) fn routes() -> axum::Router<std::sync::Arc<crate::AppState>> {
    http::routes()
        .merge(sync_http::routes())
        .merge(derived_http::routes())
        .merge(rebuild_http::routes())
}
pub use engine::{Continue, ExternalExecutors, Service};

mod debug_http;
mod derived_http;
mod rebuild_http;
mod sync_http;
pub(crate) use debug_http::routes as debug_routes;
