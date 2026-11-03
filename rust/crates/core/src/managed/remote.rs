//! Pure validation of bounded, exact-worker observations. No provider mutation,
//! path lookup, wall clock or scientific metric normalization occurs here.
use super::{Config, evidence::*};
use crate::slurm::helper::{self, Scope};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const WINDOW_BYTES: usize = 128 * 1024;
pub const OBSERVATION_CONTRACT: &str = "managed-worker-observation.v1";
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Problem {
    Missing,
    Truncated,
    InvalidUtf8,
    Changed,
    Unavailable,
    HashMismatch,
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Window {
    pub text: String,
    pub offset_bytes: u64,
    pub total_bytes: u64,
    pub problem: Option<Problem>,
}
impl std::fmt::Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundedWorkerWindow")
            .field("offset_bytes", &self.offset_bytes)
            .field("total_bytes", &self.total_bytes)
            .field("problem", &self.problem)
            .finish_non_exhaustive()
    }
}
impl Window {
    pub fn unavailable(problem: Problem) -> Self {
        Self {
            text: String::new(),
            offset_bytes: 0,
            total_bytes: 0,
            problem: Some(problem),
        }
    }
    pub fn validate(&self) -> helper::Result<()> {
        if self.text.len() > WINDOW_BYTES
            || self.offset_bytes > self.total_bytes
            || self.text.len() as u64 > self.total_bytes - self.offset_bytes
            || self.problem.is_none()
                && (self.offset_bytes != 0 || self.text.len() as u64 != self.total_bytes)
        {
            return Err(helper::Error::Metadata);
        }
        Ok(())
    }
    /// Complete JSONL records only. A partial final line or malformed record
    /// remains a diagnostic; missing, failed and truncated history is not zero.
    pub fn metrics(&self) -> (Vec<Value>, bool, usize) {
        let mut records = Vec::new();
        let mut invalid = 0;
        let mut lines = self.text.split_inclusive('\n');
        if self.offset_bytes > 0 {
            lines.next();
        }
        let mut partial = false;
        for line in lines {
            if !line.ends_with('\n') {
                partial = true;
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            match crate::json::mapping(line.as_bytes()) {
                Ok(map) => records.push(Value::Object(map)),
                Err(_) => invalid += 1,
            }
        }
        (
            records,
            self.problem.is_some() || partial || invalid > 0,
            invalid,
        )
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub contract: String,
    pub config_sha256: String,
    pub job_id: String,
    pub worker: Observation,
    pub stdout: Window,
    pub stderr: Window,
    pub metrics: Window,
}
impl Evidence {
    pub fn validate_scope(&self, scope: &Scope, job: &str) -> helper::Result<()> {
        scope.validate()?;
        let identity = &self.worker.identity;
        if self.contract != OBSERVATION_CONTRACT
            || self.job_id != job
            || !helper::valid_digest(&self.config_sha256)
            || identity.project != scope.project
            || identity.run_id != scope.run_id
            || identity.attempt_id != scope.attempt_id
            || self.worker.contract != CONTRACT
            || self.worker.safe_to_retry
            || (self.worker.phase == Phase::Finished) != self.worker.receipt.is_some()
        {
            return Err(helper::Error::Identity);
        }
        for window in [&self.stdout, &self.stderr, &self.metrics] {
            window.validate()?;
        }
        Ok(())
    }
    pub fn validate_config(&self, config: &Config) -> helper::Result<()> {
        let binding = config.binding().map_err(|_| helper::Error::Metadata)?;
        let bytes = serde_json::to_vec(config).map_err(|_| helper::Error::Metadata)?;
        if self.config_sha256 != helper::sha256_bytes(&bytes)
            || self.worker.identity != binding.identity
            || (self.worker.phase == Phase::Finished) != self.worker.receipt.is_some()
            || self.worker.manifest_sha256 != binding.sha256
            || self.worker.runtime_environment_sha256.is_some()
                == config.runtime_environment.is_empty()
        {
            return Err(helper::Error::Identity);
        }
        if let Some(receipt) = &self.worker.receipt {
            if receipt.identity != binding.identity
                || receipt.contract != CONTRACT
                || receipt.manifest_sha256 != binding.sha256
                || receipt.download_available
                || receipt.runtime_environment_sha256 != self.worker.runtime_environment_sha256
                || receipt.training != training(&receipt.process)
            {
                return Err(helper::Error::Metadata);
            }
        }
        Ok(())
    }
    pub fn metric_records(&self) -> (Vec<Value>, bool, usize, usize) {
        let (records, mut partial, invalid) = self.metrics.metrics();
        let identity = &self.worker.identity;
        let mut foreign = 0;
        let records = records
            .into_iter()
            .filter(|record| {
                let conflict = [
                    ("project", &identity.project),
                    ("run_id", &identity.run_id),
                    ("attempt_id", &identity.attempt_id),
                ]
                .into_iter()
                .any(|(key, expected)| {
                    record.get(key).is_some_and(|value| {
                        !value.is_null() && value.as_str() != Some(expected.as_str())
                    })
                });
                if conflict {
                    foreign += 1;
                }
                !conflict
            })
            .collect();
        partial |= foreign > 0;
        (records, partial, invalid, foreign)
    }
    /// Failed file samples retain the last accepted bounded text with the new
    /// diagnostic. They never turn a transient storage error into empty history.
    pub fn retain_windows(&mut self, previous: &Self) {
        for (next, old) in [
            (&mut self.stdout, &previous.stdout),
            (&mut self.stderr, &previous.stderr),
            (&mut self.metrics, &previous.metrics),
        ] {
            if matches!(
                next.problem,
                Some(Problem::Missing | Problem::Unavailable | Problem::Changed)
            ) && !old.text.is_empty()
            {
                let problem = next.problem.clone();
                *next = old.clone();
                next.problem = problem;
            }
        }
    }
    /// Finished process evidence is immutable. Failed observations never erase
    /// a known receipt, and STARTED remains possible execution, not liveness.
    pub fn follows(&self, previous: &Self) -> helper::Result<()> {
        self.metrics.validate()?;
        previous.metrics.validate()?;
        let rank = |phase| match phase {
            Phase::Prepared => 0,
            Phase::Started => 1,
            Phase::Finished => 2,
        };
        if self.config_sha256 != previous.config_sha256
            || self.job_id != previous.job_id
            || self.worker.identity != previous.worker.identity
            || self.worker.runtime_environment_sha256 != previous.worker.runtime_environment_sha256
            || rank(self.worker.phase) < rank(previous.worker.phase)
            || previous.worker.phase == Phase::Finished && self.worker != previous.worker
        {
            return Err(helper::Error::Content);
        }
        // Compare accepted complete-line bytes wherever the bounded windows
        // overlap. A failed read can retain those bytes, so its diagnostic must
        // not disable later history checks. A partial final line may complete.
        if matches!(self.metrics.problem, None | Some(Problem::Truncated)) {
            let old = &previous.metrics;
            let next = &self.metrics;
            let bytes = old.text.as_bytes();
            let first = if old.offset_bytes == 0 {
                0
            } else {
                bytes
                    .iter()
                    .position(|b| *b == b'\n')
                    .map_or(bytes.len(), |i| i + 1)
            };
            let last = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
            if last > first {
                let old_start = old.offset_bytes + first as u64;
                let old_end = old.offset_bytes + last as u64;
                let next_end = next.offset_bytes + next.text.len() as u64;
                if next.offset_bytes == 0 && next_end < old_end {
                    return Err(helper::Error::Content);
                }
                let start = old_start.max(next.offset_bytes);
                let end = old_end.min(next_end);
                if start < end
                    && bytes[(start - old.offset_bytes) as usize..(end - old.offset_bytes) as usize]
                        != next.text.as_bytes()[(start - next.offset_bytes) as usize
                            ..(end - next.offset_bytes) as usize]
                {
                    return Err(helper::Error::Content);
                }
            }
        }
        Ok(())
    }
}
