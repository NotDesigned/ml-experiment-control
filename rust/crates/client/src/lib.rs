//! Native client: pure connection and business request planning, private credential I/O,
//! and HTTP transport are separate boundaries. No mutation replay.

pub mod batch;
pub mod config;
pub mod credentials;
pub mod datasets;
pub mod derived;
pub mod dockerfile;
pub mod local;
pub mod observation;
pub mod preparations;
pub mod presentation;
pub mod projects;
pub mod purge_state;
pub mod read;
pub mod runtime;
pub mod session;
pub mod sources;
pub mod status;
pub mod submissions;
pub mod submit;
pub mod transport;
pub mod update;
pub mod wandb_rebuild;
pub mod watch;

pub use config::{ConnectionPlan, Settings, TokenSource, normalize_url};
pub use transport::Client;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "dataset {asset} transfer is unconfirmed; inspect this asset and repeat the same dataset operation to reconcile durable parts: {source}"
    )]
    DatasetOutcomeUnknown { asset: String, source: Box<Error> },
    #[error(
        "invalid API URL; use HTTPS or loopback HTTP without userinfo, query or fragment. \
         A configuration from before 0.2 needs `ml-exp config set` again"
    )]
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
    #[error("API response failed size, format or identity validation")]
    Response,
    #[error("daemon did not provide a valid health document")]
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
        "archive saved and verified locally; acknowledgement or release is unconfirmed; repeat the same download command without redownloading: {0}"
    )]
    ArtifactAcknowledgementUnconfirmed(Box<Error>),
    #[error(
        "Run creation outcome is unknown; query the same Project/Run before retrying; no automatic replay was attempted: {0}"
    )]
    RunCreationOutcomeUnknown(Box<Error>),
    #[error("{0}")]
    Observation(&'static str),
    #[error("{0}")]
    Submit(String),
}
impl Error {
    /// A stable machine code: the server's own code for an HTTP failure.
    pub fn code(&self) -> String {
        match self {
            Self::DatasetOutcomeUnknown { .. } => "DATASET_OUTCOME_UNKNOWN",
            Self::Url => "INVALID_API_URL",
            Self::Token => "INVALID_API_TOKEN",
            Self::Config => "CLIENT_CONFIG_UNAVAILABLE",
            Self::CredentialScope => "CREDENTIAL_SCOPE_MISMATCH",
            Self::Path => "INVALID_API_PATH",
            Self::Transport => "HTTP_TRANSPORT_FAILED",
            Self::Http(failure) => return failure.error_code.clone(),
            Self::Response => "INVALID_API_RESPONSE",
            Self::Protocol => "INVALID_HEALTH_DOCUMENT",
            Self::ProjectOutcomeUnknown(_) => "PROJECT_OUTCOME_UNKNOWN",
            Self::SourceOutcomeUnknown(_) => "SOURCE_OUTCOME_UNKNOWN",
            Self::SubmissionOutcomeUnknown(_) => "SUBMISSION_OUTCOME_UNKNOWN",
            Self::ArtifactAcknowledgementUnconfirmed(_) => "ARTIFACT_ACKNOWLEDGEMENT_UNCONFIRMED",
            Self::RunCreationOutcomeUnknown(_) => "RUN_CREATION_OUTCOME_UNKNOWN",
            Self::Observation(_) | Self::Submit(_) => "CLIENT_ERROR",
        }
        .into()
    }
    /// The HTTP failure behind this error, also through an unknown outcome.
    pub fn http(&self) -> Option<&ml_exp_core::diagnostic::HttpFailure> {
        match self {
            Self::Http(failure) => Some(failure),
            Self::DatasetOutcomeUnknown { source, .. } => source.http(),
            Self::ProjectOutcomeUnknown(inner)
            | Self::SourceOutcomeUnknown(inner)
            | Self::SubmissionOutcomeUnknown(inner)
            | Self::ArtifactAcknowledgementUnconfirmed(inner)
            | Self::RunCreationOutcomeUnknown(inner) => inner.http(),
            _ => None,
        }
    }
}
/// What `--json` prints on stdout when a command fails, so a caller that parses stdout
/// always reads one JSON document. The message is the one shown on stderr.
pub fn error_document(error: &(dyn std::error::Error + 'static)) -> serde_json::Value {
    let mut document =
        serde_json::json!({"code":"CLIENT_ERROR","http_status":null,"message":error.to_string()});
    if let Some(error) = error.downcast_ref::<Error>() {
        document["code"] = error.code().into();
        if let Some(failure) = error.http() {
            document["http_status"] = failure.status.into();
            if let Some(seconds) = failure.retry_after {
                document["retry_after_seconds"] = seconds.into();
            }
            if !matches!(error, Error::Http(_)) {
                document["http_code"] = failure.error_code.clone().into();
            }
        }
    }
    serde_json::json!({ "error": document })
}
#[cfg(test)]
mod error_tests {
    use super::*;
    #[test]
    fn a_json_error_keeps_the_server_code_status_and_message() {
        let failure = ml_exp_core::diagnostic::HttpFailure::from_response(
            503,
            Some("CLOUD_RUN_BUSY"),
            Some("5"),
            br#"{"code":"CLOUD_RUN_BUSY","detail":"another operation holds the lock"}"#,
            "secret-token",
            0,
        );
        let error = Error::Http(Box::new(failure));
        let document = error_document(&error);
        assert_eq!(document["error"]["code"], "CLOUD_RUN_BUSY");
        assert_eq!(document["error"]["http_status"], 503);
        assert_eq!(document["error"]["message"], error.to_string());
        let wrapped = Error::SubmissionOutcomeUnknown(Box::new(error));
        let document = error_document(&wrapped);
        assert_eq!(document["error"]["code"], "SUBMISSION_OUTCOME_UNKNOWN");
        assert_eq!(document["error"]["http_code"], "CLOUD_RUN_BUSY");
        assert_eq!(document["error"]["http_status"], 503);
        let other: Box<dyn std::error::Error> = "not a client error".into();
        let document = error_document(&*other);
        assert_eq!(
            document,
            serde_json::json!({"error":{"code":"CLIENT_ERROR","http_status":null,"message":"not a client error"}})
        );
        assert_eq!(
            error_document(&Error::Transport)["error"]["code"],
            "HTTP_TRANSPORT_FAILED"
        );
    }
}

pub mod artifacts;

pub mod debug;

pub mod administration;
