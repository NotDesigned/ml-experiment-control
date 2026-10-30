//! Pure, bounded outward views of already-read exact Run/Attempt evidence.
//! No filesystem, index, provider, clock, repair or publication effects occur here.
use serde_json::Value;

mod metrics;
mod projection;
mod remote;
pub use metrics::{
    MetricFormat, MetricQuery, merge_metrics, metric_payload, validate_journal_observations,
};
pub use projection::{
    LiveMetrics, LogQuery, LogStream, attempt_collection, attempt_logs, attempt_metrics,
    attempt_show, historical_collection, outward_metadata, project_summary, run_attempts,
    run_detail, run_summary,
};
pub(crate) use remote::{project_artifacts, project_worker};

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
    encoded_size(&value)?;
    Ok(value)
}

/// Count a projection without allocating another serialized payload.
pub(crate) fn encoded_size(value: &Value) -> Result<usize> {
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
    let mut limit = Limit(0);
    serde_json::to_writer(&mut limit, value).map_err(|_| Error::OutputLimit)?;
    Ok(limit.0)
}

/// Account for the next member before retaining it in a bounded JSON array.
pub(crate) fn array_size(size: usize, value: &Value) -> Result<usize> {
    size.checked_add(encoded_size(value)?)
        .and_then(|size| size.checked_add(1))
        .filter(|size| *size <= MAX_RESPONSE_BYTES)
        .ok_or(Error::OutputLimit)
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn aggregate_budget_counts_escaped_bytes_and_refuses_growth_before_retention() {
        let unicode = Value::String("训练\n\"".into());
        assert_eq!(
            encoded_size(&unicode).unwrap(),
            serde_json::to_vec(&unicode).unwrap().len()
        );
        let large = Value::String("x".repeat(MAX_RESPONSE_BYTES / 2));
        let first = array_size(2, &large).unwrap();
        assert!(matches!(array_size(first, &large), Err(Error::OutputLimit)));
        assert!(matches!(
            array_size(usize::MAX, &Value::Null),
            Err(Error::OutputLimit)
        ));
    }
}
