//! Pure, bounded outward views of already-read exact Run/Attempt evidence.
//! No filesystem, index, provider, clock, repair or publication effects occur here.
use serde_json::Value;

mod metrics;
mod projection;
pub use metrics::{
    MetricFormat, MetricQuery, merge_metrics, metric_payload, validate_journal_observations,
};
pub use projection::{
    LiveMetrics, LogQuery, LogStream, attempt_collection, attempt_logs, attempt_metrics,
    attempt_show, outward_metadata, run_attempts, run_detail,
};

pub const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_METRIC_OBSERVATIONS: usize = 100_000;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid read query: {0}")]
    Query(&'static str),
    #[error("read snapshot identity does not match the exact requested object")]
    Identity,
    #[error("exact Attempt is unavailable in this snapshot")]
    UnknownAttempt,
    #[error("read evidence is unavailable or invalid")]
    Evidence,
    #[error("read response or complete-history validation exceeds the bounded limit")]
    OutputLimit,
    #[error(transparent)]
    Metric(#[from] ml_exp_core::metrics::MetricError),
    #[error(transparent)]
    Canonical(#[from] ml_exp_core::canonical::CanonicalError),
    #[error(transparent)]
    Failure(#[from] ml_exp_core::failures::FailureError),
}
pub type Result<T> = std::result::Result<T, Error>;

fn bounded(value: Value) -> Result<Value> {
    struct Limit(usize);
    impl std::io::Write for Limit {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len())
                .ok_or_else(|| std::io::Error::other("bounded JSON output overflow"))?;
            if self.0 > MAX_RESPONSE_BYTES {
                return Err(std::io::Error::other("bounded JSON output limit"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(Limit(0), &value).map_err(|_| Error::OutputLimit)?;
    Ok(value)
}
