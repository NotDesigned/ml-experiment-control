//! Native read-only client: pure connection planning, private credential I/O,
//! and HTTP transport are separate boundaries. No mutation replay.

pub mod config;
pub mod credentials;
pub mod observation;
pub mod status;
pub mod transport;

pub use config::{ConnectionPlan, Settings, TokenSource, normalize_url};
pub use transport::Client;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid API URL; use HTTPS or loopback HTTP without userinfo, query or fragment")]
    Url,
    #[error("invalid API token")]
    Token,
    #[error("cannot read or write private client configuration")]
    Config,
    #[error("saved credentials belong to a different API URL")]
    CredentialScope,
    #[error("API path must remain under the configured /api/ prefix")]
    Path,
    #[error("HTTP request failed; no automatic retry was attempted")]
    Transport,
    #[error("{0}")]
    Http(Box<ml_exp_core::diagnostic::HttpFailure>),
    #[error("API response is too large or is not JSON")]
    Response,
    #[error("{0}")]
    Observation(&'static str),
}
