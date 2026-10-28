//! Native client: pure connection and business request planning, private credential I/O,
//! and HTTP transport are separate boundaries. No mutation replay.

pub mod config;
pub mod credentials;
pub mod datasets;
pub mod observation;
pub mod presentation;
pub mod projects;
pub mod read;
pub mod session;
pub mod sources;
pub mod status;
pub mod submissions;
pub mod transport;
pub mod watch;

pub use config::{ConnectionPlan, Settings, TokenSource, normalize_url};
pub use transport::Client;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "dataset {asset} transfer is unconfirmed; repeat upload with the same directory to reconcile durable parts: {source}"
    )]
    DatasetOutcomeUnknown { asset: String, source: Box<Error> },
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
    #[error("daemon did not provide a supported protocol-2 handshake")]
    Protocol,
    #[error(
        "project change outcome is not confirmed; query project list before retrying; no automatic replay was attempted: {0}"
    )]
    ProjectOutcomeUnknown(Box<Error>),
    #[error(
        "source publication outcome is not confirmed; use source lookup --project <project> --sha256 <archive-sha256> before retrying; no automatic replay was attempted: {0}"
    )]
    SourceOutcomeUnknown(Box<Error>),
    #[error(
        "submission outcome is unknown; query the original Run preparation or Submission, then reconcile; no automatic replay was attempted: {0}"
    )]
    SubmissionOutcomeUnknown(Box<Error>),
    #[error(
        "Run creation outcome is unknown; query the same Project/Run before retrying; no automatic replay was attempted: {0}"
    )]
    RunCreationOutcomeUnknown(Box<Error>),
    #[error("{0}")]
    Observation(&'static str),
}

pub mod artifacts;
