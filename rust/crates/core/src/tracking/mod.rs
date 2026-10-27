//! Optional publication decisions and disposable metric displays. No filesystem,
//! environment, clock, SDK, HTTP or database access; accepted journal records
//! remain authoritative regardless of any publication/display failure.
mod contract;
mod display;
pub mod publication;

pub use contract::{
    RoutingSettings, WandbOptions, WandbSettings, finite_parameters, finite_parameters_json,
    resolve, resolve_preparation,
};
pub use display::{
    DefinitionSpec, Projection, SummaryPatch, curve_record, definition_specs, fingerprint,
    identity, number, stem, summary_patch,
};

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid tracking contract: {0}")]
    Contract(&'static str),
    #[error("invalid metric display: {0}")]
    Display(&'static str),
    #[error(transparent)]
    Canonical(#[from] crate::canonical::CanonicalError),
}
pub type Result<T> = std::result::Result<T, Error>;
