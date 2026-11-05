use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::identity::submission_marker;
use crate::state::{FailureClass, RunState};

use super::{Error, component, printable};

const MAX_OUTPUT: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct JobId(String);
impl JobId {
    /// An existing single array element is readable. Step IDs and compressed
    /// arrays are not exact job identities and cannot authorize cancellation.
    pub fn parse(value: &str) -> Result<Self, Error> {
        let parts: Vec<_> = value.split('_').collect();
        if value.len() > 64
            || parts.len() > 2
            || parts
                .iter()
                .any(|part| part.is_empty() || !part.bytes().all(|c| c.is_ascii_digit()))
        {
            return Err(Error::Identity("job_id"));
        }
        Ok(Self(value.to_owned()))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl<'de> Deserialize<'de> for JobId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

pub fn normalize_state(raw: &str, exit_code: Option<&str>) -> RunState {
    let raw = raw
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches('+')
        .to_ascii_uppercase();
    match raw.as_str() {
        "PENDING" | "CONFIGURING" | "REQUEUED" | "REQUEUE_FED" => RunState::Queued,
        "RUNNING" | "COMPLETING" => RunState::Running,
        "COMPLETED" => {
            if matches!(exit_code, None | Some("") | Some("0:0")) {
                RunState::Succeeded
            } else {
                RunState::Failed
            }
        }
        "PREEMPTED" => RunState::Preempted,
        "FAILED" | "NODE_FAIL" | "OUT_OF_MEMORY" | "TIMEOUT" => RunState::Failed,
        "CANCELLED" => RunState::Cancelled,
        _ => RunState::Unknown,
    }
}
pub fn failure_class(raw: &str) -> Option<FailureClass> {
    match raw {
        "PREEMPTED" => Some(FailureClass::Preemption),
        "NODE_FAIL" | "BOOT_FAIL" => Some(FailureClass::Scheduler),
        "OUT_OF_MEMORY" | "TIMEOUT" => Some(FailureClass::Resource),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobScope {
    pub ssh_alias: String,
    pub job_id: JobId,
    pub scheduler_name: String,
    pub partition: String,
}
impl JobScope {
    pub fn validate(&self) -> Result<(), Error> {
        if !component(&self.ssh_alias)
            || !component(&self.partition)
            || !printable(&self.scheduler_name, 128)
        {
            return Err(Error::Identity("job scope"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StatusSource {
    Sacct,
    Squeue,
    Scontrol,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Availability {
    Observed,
    NotObserved,
    Unavailable,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueDetail {
    pub source: String,
    pub availability: Availability,
    pub reason: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobStatus {
    pub ssh_alias: String,
    pub job_id: JobId,
    pub scheduler_name: String,
    pub partition: String,
    pub state: RunState,
    pub raw_state: String,
    pub elapsed: Option<String>,
    pub exit_code: Option<String>,
    pub reason: Option<String>,
    pub failure_class: Option<FailureClass>,
    pub observation_source: Option<StatusSource>,
    pub observed_at: Option<String>,
    pub queue: Option<QueueDetail>,
}
impl JobStatus {
    /// Validate an IPC observation independently of serde's field types.
    pub fn validate(&self) -> Result<(), Error> {
        let source = self
            .observation_source
            .ok_or(Error::Evidence("status source"))?;
        let scope = JobScope {
            ssh_alias: self.ssh_alias.clone(),
            job_id: self.job_id.clone(),
            scheduler_name: self.scheduler_name.clone(),
            partition: self.partition.clone(),
        };
        let mut row = vec![
            self.job_id.as_str(),
            &self.scheduler_name,
            &self.partition,
            &self.raw_state,
            self.elapsed.as_deref().unwrap_or(""),
            self.exit_code.as_deref().unwrap_or(""),
        ];
        if source != StatusSource::Sacct {
            if source == StatusSource::Squeue && self.exit_code.is_some() {
                return Err(Error::Evidence("unobserved queue exit code"));
            }
            row.push(self.reason.as_deref().unwrap_or(""));
        }
        let mut parsed =
            parse_job_status(&row.join("|"), &scope, source, self.observed_at.as_deref())?
                .ok_or(Error::Evidence("missing status"))?;
        if let Some(queue) = &self.queue {
            if self.state != RunState::Queued
                || source != StatusSource::Sacct
                || queue.source != "slurm-squeue"
                || queue.reason != self.reason
                || self.reason.as_deref().is_some_and(|v| !printable(v, 1024))
                || queue.availability != Availability::Observed && self.reason.is_some()
            {
                return Err(Error::Evidence("queue detail"));
            }
            parsed.reason = self.reason.clone();
            parsed.queue = self.queue.clone();
        }
        if &parsed != self {
            return Err(Error::Evidence("status projection"));
        }
        Ok(())
    }
    fn unknown(scope: &JobScope) -> Self {
        Self {
            ssh_alias: scope.ssh_alias.clone(),
            job_id: scope.job_id.clone(),
            scheduler_name: scope.scheduler_name.clone(),
            partition: scope.partition.clone(),
            state: RunState::Unknown,
            raw_state: "UNKNOWN".into(),
            elapsed: None,
            exit_code: None,
            reason: None,
            failure_class: None,
            observation_source: None,
            observed_at: None,
            queue: None,
        }
    }
    fn matches_scope(&self, scope: &JobScope) -> bool {
        self.ssh_alias == scope.ssh_alias
            && self.job_id == scope.job_id
            && self.scheduler_name == scope.scheduler_name
            && self.partition == scope.partition
    }
}

pub(crate) fn fields(line: &str, count: usize) -> Result<Vec<&str>, Error> {
    let mut values: Vec<_> = line.split('|').collect();
    if values.len() == count + 1 && values.last() == Some(&"") {
        values.pop();
    }
    if values.len() != count {
        return Err(Error::Evidence("field count"));
    }
    Ok(values)
}
fn exit_code(value: &str) -> bool {
    let Some((code, signal)) = value.split_once(':') else {
        return false;
    };
    [code, signal].iter().all(|v| {
        !v.is_empty()
            && v.len() <= 5
            && v.bytes().all(|c| c.is_ascii_digit())
            && v.parse::<u16>().is_ok()
    })
}

/// Parse exactly one parent-job row. Foreign, duplicate, truncated or step rows
/// are errors, never candidates selected by position. No timestamp is invented.
pub fn parse_job_status(
    output: &str,
    scope: &JobScope,
    source: StatusSource,
    observed_at: Option<&str>,
) -> Result<Option<JobStatus>, Error> {
    scope.validate()?;
    if output.len() > MAX_OUTPUT {
        return Err(Error::Evidence("status size"));
    }
    if observed_at.is_some_and(|v| !printable(v, 64)) {
        return Err(Error::Evidence("observation timestamp"));
    }
    let rows: Vec<_> = output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.len() != 1 {
        return Err(Error::Evidence("duplicate status row"));
    }
    let row = fields(rows[0], if source == StatusSource::Sacct { 6 } else { 7 })?;
    if row[0] != scope.job_id.as_str()
        || row[1] != scope.scheduler_name
        || row[2] != scope.partition
    {
        return Err(Error::Evidence("foreign job scope"));
    }
    let raw = row[3]
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches('+');
    if !printable(raw, 64) || !raw.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
        return Err(Error::Evidence("scheduler state"));
    }
    if !row[4].is_empty() && !printable(row[4], 64) {
        return Err(Error::Evidence("elapsed"));
    }
    if !row[5].is_empty() && !exit_code(row[5]) {
        return Err(Error::Evidence("exit code"));
    }
    // %R changes from reason to node list after allocation. Only pending or
    // configuring states supply a scheduler blocker. '0:0' in the squeue format
    // is a formatting placeholder, NOT observed process exit evidence.
    let reason = if source != StatusSource::Sacct
        && matches!(raw, "PENDING" | "CONFIGURING")
        && !row[6].trim().is_empty()
    {
        if !printable(row[6].trim(), 1024) {
            return Err(Error::Evidence("pending reason"));
        }
        Some(row[6].trim().to_owned())
    } else {
        None
    };
    let observed_exit =
        (source != StatusSource::Squeue && !row[5].is_empty()).then(|| row[5].to_owned());
    Ok(Some(JobStatus {
        ssh_alias: scope.ssh_alias.clone(),
        job_id: scope.job_id.clone(),
        scheduler_name: scope.scheduler_name.clone(),
        partition: scope.partition.clone(),
        state: normalize_state(raw, observed_exit.as_deref()),
        raw_state: raw.into(),
        elapsed: (!row[4].is_empty()).then(|| row[4].to_owned()),
        exit_code: observed_exit,
        reason,
        failure_class: failure_class(raw),
        observation_source: Some(source),
        observed_at: observed_at.map(str::to_owned),
        queue: None,
    }))
}

/// Parse status and ownership in the same scheduler row. Looking up a Comment
/// separately and then reading status would permit a job-ID reuse race.
pub fn parse_owned_job_status(
    output: &str,
    scope: &JobScope,
    token: &str,
    source: StatusSource,
    observed_at: Option<&str>,
) -> Result<Option<JobStatus>, Error> {
    let marker = submission_marker(token).map_err(|_| Error::Ownership)?;
    if output.len() > MAX_OUTPUT {
        return Err(Error::Evidence("status size"));
    }
    let rows: Vec<_> = output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if rows.is_empty() {
        return parse_job_status("", scope, source, observed_at);
    }
    if rows.len() != 1 {
        return Err(Error::Ambiguous);
    }
    let row = fields(rows[0], if source == StatusSource::Sacct { 7 } else { 8 })?;
    if row[2] != marker {
        return Err(Error::Ownership);
    }
    let projected = row[..2]
        .iter()
        .chain(row[3..].iter())
        .copied()
        .collect::<Vec<_>>()
        .join("|");
    parse_job_status(&projected, scope, source, observed_at)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Query<'a> {
    NotRequested,
    Failed,
    Success(&'a str),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Diagnostic {
    NoEvidence,
    OwnershipNotRecorded,
    QueryUnavailable,
    MalformedOrForeignEvidence,
    QueueNotObserved,
    QueueUnavailable,
    RegressiveState,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusObservation {
    pub status: JobStatus,
    pub checked_at: Option<String>,
    pub diagnostic: Option<Diagnostic>,
    pub retained_previous: bool,
}

/// Same reducer with a submission-marker check in every accepted row. Query
/// failures preserve previous evidence; no missing row proves scheduler absence.
pub fn observe_owned_status(
    scope: &JobScope,
    token: &str,
    previous: Option<&JobStatus>,
    accounting: Query<'_>,
    queue: Query<'_>,
    observed_at: Option<&str>,
) -> Result<StatusObservation, Error> {
    submission_marker(token).map_err(|_| Error::Ownership)?;
    // Some clusters disable accounting Comment retention. A structurally exact
    // row with an empty Comment is not ownership proof. In that one case the
    // queue must independently prove BOTH ownership and state in its own row;
    // no accounting field is spliced into that observation.
    let accounting = match accounting {
        Query::Success(text) if accounting_omits_ownership(text, scope) => {
            let mut value = observe_with(
                scope,
                previous,
                Query::NotRequested,
                queue,
                observed_at,
                &|text, scope, source, at| parse_owned_job_status(text, scope, token, source, at),
            )?;
            if value.diagnostic == Some(Diagnostic::NoEvidence) {
                value.diagnostic = Some(Diagnostic::OwnershipNotRecorded);
            }
            return Ok(value);
        }
        other => other,
    };
    observe_with(
        scope,
        previous,
        accounting,
        queue,
        observed_at,
        &|text, scope, source, at| parse_owned_job_status(text, scope, token, source, at),
    )
}

fn accounting_omits_ownership(text: &str, scope: &JobScope) -> bool {
    if text.len() > MAX_OUTPUT {
        return false;
    }
    let rows: Vec<_> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let [line] = rows.as_slice() else {
        return false;
    };
    let Ok(row) = fields(line, 7) else {
        return false;
    };
    row[2].is_empty()
        && parse_job_status(
            &row[..2]
                .iter()
                .chain(row[3..].iter())
                .copied()
                .collect::<Vec<_>>()
                .join("|"),
            scope,
            StatusSource::Sacct,
            None,
        )
        .is_ok_and(|value| value.is_some())
}

/// Controller observations are useful when accounting does not retain Comment,
/// including just-ended jobs no longer visible in squeue. Every required field
/// comes from one exact controller row; a separate identity lookup is forbidden.
pub fn parse_owned_control_status(
    output: &str,
    scope: &JobScope,
    token: &str,
    observed_at: Option<&str>,
) -> Result<Option<JobStatus>, Error> {
    if output.len() > MAX_OUTPUT {
        return Err(Error::Evidence("controller size"));
    }
    let rows: Vec<_> = output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let [line] = rows.as_slice() else {
        return if rows.is_empty() {
            Ok(None)
        } else {
            Err(Error::Ambiguous)
        };
    };
    let required = [
        "JobId",
        "JobName",
        "Comment",
        "Partition",
        "JobState",
        "RunTime",
        "ExitCode",
        "Reason",
    ];
    let mut values = BTreeMap::new();
    for item in line.split_whitespace() {
        if let Some((key, value)) = item.split_once('=') {
            if required.contains(&key) && values.insert(key, value).is_some() {
                return Err(Error::Ambiguous);
            }
        }
    }
    let row = required
        .iter()
        .map(|key| {
            values
                .get(key)
                .copied()
                .ok_or(Error::Evidence("controller field"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if row.iter().any(|value| value.contains('|')) {
        return Err(Error::Evidence("controller field"));
    }
    parse_owned_job_status(
        &row.join("|"),
        scope,
        token,
        StatusSource::Scontrol,
        observed_at,
    )
}

type StatusParser<'a> =
    dyn Fn(&str, &JobScope, StatusSource, Option<&str>) -> Result<Option<JobStatus>, Error> + 'a;
fn observe_with(
    scope: &JobScope,
    previous: Option<&JobStatus>,
    accounting: Query<'_>,
    queue: Query<'_>,
    observed_at: Option<&str>,
    parse: &StatusParser<'_>,
) -> Result<StatusObservation, Error> {
    scope.validate()?;
    if observed_at.is_some_and(|v| !printable(v, 64)) {
        return Err(Error::Evidence("observation timestamp"));
    }
    if previous.is_some_and(|status| !status.matches_scope(scope)) {
        return Err(Error::Identity("previous status scope"));
    }
    let retain = |diagnostic| StatusObservation {
        status: previous
            .cloned()
            .unwrap_or_else(|| JobStatus::unknown(scope)),
        checked_at: observed_at.map(str::to_owned),
        diagnostic: Some(diagnostic),
        retained_previous: previous.is_some(),
    };
    let mut status = match accounting {
        Query::Failed => return Ok(retain(Diagnostic::QueryUnavailable)),
        Query::Success(text) => match parse(text, scope, StatusSource::Sacct, observed_at) {
            Ok(Some(status)) => status,
            Err(_) => return Ok(retain(Diagnostic::MalformedOrForeignEvidence)),
            Ok(None) => return observe_queue(scope, previous, queue, observed_at, parse),
        },
        Query::NotRequested => return observe_queue(scope, previous, queue, observed_at, parse),
    };
    if previous.is_some_and(|prior| !prior.state.allows(status.state)) {
        return Ok(retain(Diagnostic::RegressiveState));
    }
    let mut diagnostic = None;
    if status.state == RunState::Queued {
        let mut detail = QueueDetail {
            source: "slurm-squeue".into(),
            availability: Availability::NotObserved,
            reason: None,
        };
        match queue {
            Query::Failed => {
                detail.availability = Availability::Unavailable;
                diagnostic = Some(Diagnostic::QueueUnavailable);
            }
            Query::NotRequested => diagnostic = Some(Diagnostic::QueueNotObserved),
            Query::Success(text) => match parse(text, scope, StatusSource::Squeue, observed_at) {
                Ok(Some(other)) if other.state == RunState::Queued => {
                    status.reason = other.reason;
                    detail.reason = status.reason.clone();
                    detail.availability = Availability::Observed;
                }
                Ok(_) => diagnostic = Some(Diagnostic::QueueNotObserved),
                Err(_) => diagnostic = Some(Diagnostic::MalformedOrForeignEvidence),
            },
        }
        status.queue = Some(detail);
    }
    Ok(StatusObservation {
        status,
        checked_at: observed_at.map(str::to_owned),
        diagnostic,
        retained_previous: false,
    })
}
fn observe_queue(
    scope: &JobScope,
    previous: Option<&JobStatus>,
    queue: Query<'_>,
    observed_at: Option<&str>,
    parse: &StatusParser<'_>,
) -> Result<StatusObservation, Error> {
    let result = match queue {
        Query::Success(text) => parse(text, scope, StatusSource::Squeue, observed_at),
        Query::Failed => Err(Error::Unavailable),
        Query::NotRequested => Ok(None),
    };
    let (status, diagnostic, retained_previous) = match result {
        Ok(Some(status)) if previous.is_some_and(|prior| !prior.state.allows(status.state)) => (
            previous.unwrap().clone(),
            Some(Diagnostic::RegressiveState),
            true,
        ),
        Ok(Some(status)) => (status, None, false),
        other => (
            previous
                .cloned()
                .unwrap_or_else(|| JobStatus::unknown(scope)),
            Some(match other {
                Err(Error::Unavailable) => Diagnostic::QueryUnavailable,
                Err(_) => Diagnostic::MalformedOrForeignEvidence,
                _ => Diagnostic::NoEvidence,
            }),
            previous.is_some(),
        ),
    };
    Ok(StatusObservation {
        status,
        checked_at: observed_at.map(str::to_owned),
        diagnostic,
        retained_previous,
    })
}
