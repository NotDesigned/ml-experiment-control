//! Pure result-collection planning and recovery decisions.
//!
//! The Python server remains responsible for locks, durable JSON, clocks,
//! provider REST/SSH calls, archive streaming and object storage.  This module
//! owns only immutable identities and reducers so a provider timeout, daemon
//! restart, or missing receipt cannot accidentally replay a recovery request.
use crate::identity::{validate_attempt, validate_id};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const CONTRACT: &str = "result-collection.v1";
pub const DEFAULT_MAX_RETRIES: u32 = 3;
pub const DAEMON_INTERRUPTED: &str = "DAEMON_INTERRUPTED";
pub const EXACT_JOB_UNCONFIRMED: &str = "EXACT_CPU_JOB_UNCONFIRMED";
pub const JOB_QUERY_UNAVAILABLE: &str = "CPU_JOB_QUERY_UNAVAILABLE";
pub const JOB_AMBIGUOUS: &str = "CPU_JOB_AMBIGUOUS";
pub const JOB_EXITED_WITHOUT_RECEIPT: &str = "CPU_JOB_EXITED_WITHOUT_RECEIPT";

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid result collection identity")]
    Identity,
    #[error("the original Attempt is not terminal")]
    OriginalNotTerminal,
    #[error("result collection retry budget exhausted")]
    RetryLimit,
    #[error("archive identity conflict")]
    ArchiveConflict,
    #[error("archive identity is invalid")]
    ArchiveIdentity,
    #[error("provider observation conflicts with the frozen job identity")]
    ProviderIdentityConflict,
    #[error("archive size arithmetic overflow")]
    CapacityOverflow,
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionIdentity {
    pub project: String,
    pub run_id: String,
    pub attempt_id: String,
}

impl CollectionIdentity {
    pub fn validate(&self) -> Result<()> {
        validate_id(&self.project).map_err(|_| Error::Identity)?;
        validate_id(&self.run_id).map_err(|_| Error::Identity)?;
        validate_attempt(&self.attempt_id).map_err(|_| Error::Identity)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CollectionStatus {
    Queued,
    Collecting,
    Submitting,
    Waiting,
    Available,
    ReconcileRequired,
    Failed,
    Released,
}

impl CollectionStatus {
    pub fn active(self) -> bool {
        matches!(
            self,
            Self::Queued | Self::Collecting | Self::Submitting | Self::Waiting
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CollectionPhase {
    Checking,
    CpuSubmission,
    NasRecovery,
    Archiving,
    Verifying,
    Publishing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobState {
    Queued,
    Pending,
    Starting,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobState {
    pub fn terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveIdentity {
    pub sha256: String,
    pub bytes: u64,
}

impl ArchiveIdentity {
    pub fn validate(&self) -> Result<()> {
        if self.bytes == 0
            || self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(Error::ArchiveIdentity);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionState {
    pub identity: CollectionIdentity,
    pub status: CollectionStatus,
    pub phase: CollectionPhase,
    pub attempts: u32,
    pub scheduler_name: Option<String>,
    pub job_uid: Option<String>,
    pub reconcile: bool,
    pub diagnostic: Option<String>,
    pub receipt: Option<ArchiveIdentity>,
}

impl CollectionState {
    pub fn validate(&self) -> Result<()> {
        self.identity.validate()?;
        if let Some(receipt) = &self.receipt {
            receipt.validate()?;
        }
        if self.attempts > DEFAULT_MAX_RETRIES {
            return Err(Error::RetryLimit);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OriginalAttempt {
    pub terminal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeginRequest {
    pub retry: bool,
    pub reconcile: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeginAction {
    None,
    CreateRecoveryJob,
    ReconcileRecoveryJob,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeginOutcome {
    pub state: CollectionState,
    pub action: BeginAction,
}

/// Derive the same stable recovery-job name for an identity and attempt.
/// Randomness and persistence stay in the Python shell; the name is a pure
/// deterministic function so a restart can safely refer to the old request.
pub fn scheduler_name(identity: &CollectionIdentity, attempt: u32) -> Result<String> {
    identity.validate()?;
    if attempt == 0 {
        return Err(Error::RetryLimit);
    }
    let mut digest = Sha256::new();
    digest.update(identity.project.as_bytes());
    digest.update([0]);
    digest.update(identity.run_id.as_bytes());
    digest.update([0]);
    digest.update(identity.attempt_id.as_bytes());
    digest.update([0]);
    digest.update(attempt.to_string().as_bytes());
    let hex = format!("{:x}", digest.finalize());
    Ok(format!("ml-expd-result-{}", &hex[..32]))
}

/// Plan a collection request. A retry is for the CPU/NAS collector only; it
/// never reopens or resubmits the original training Attempt.
pub fn begin(
    current: Option<&CollectionState>,
    identity: CollectionIdentity,
    original: OriginalAttempt,
    request: BeginRequest,
) -> Result<BeginOutcome> {
    identity.validate()?;
    if !original.terminal {
        return Err(Error::OriginalNotTerminal);
    }
    if let Some(state) = current {
        state.validate()?;
        if state.identity != identity {
            return Err(Error::Identity);
        }
        if state.receipt.is_some() {
            return Ok(BeginOutcome {
                state: state.clone(),
                action: BeginAction::None,
            });
        }
        if state.status.active() {
            return Ok(BeginOutcome {
                state: state.clone(),
                action: BeginAction::None,
            });
        }
        if state.status == CollectionStatus::ReconcileRequired {
            if !request.reconcile {
                return Ok(BeginOutcome {
                    state: state.clone(),
                    action: BeginAction::None,
                });
            }
            let mut next = state.clone();
            next.status = CollectionStatus::Queued;
            next.phase = CollectionPhase::Checking;
            next.reconcile = true;
            next.diagnostic = None;
            return Ok(BeginOutcome {
                state: next,
                action: BeginAction::ReconcileRecoveryJob,
            });
        }
        if !request.retry {
            return Ok(BeginOutcome {
                state: state.clone(),
                action: BeginAction::None,
            });
        }
        let attempt = state.attempts.checked_add(1).ok_or(Error::RetryLimit)?;
        if attempt > DEFAULT_MAX_RETRIES {
            return Err(Error::RetryLimit);
        }
        let next = new_state(identity, attempt)?;
        return Ok(BeginOutcome {
            state: next,
            action: BeginAction::CreateRecoveryJob,
        });
    }
    let next = new_state(identity, 1)?;
    Ok(BeginOutcome {
        state: next,
        action: BeginAction::CreateRecoveryJob,
    })
}

fn new_state(identity: CollectionIdentity, attempts: u32) -> Result<CollectionState> {
    Ok(CollectionState {
        scheduler_name: Some(scheduler_name(&identity, attempts)?),
        identity,
        status: CollectionStatus::Queued,
        phase: CollectionPhase::Checking,
        attempts,
        job_uid: None,
        reconcile: false,
        diagnostic: None,
        receipt: None,
    })
}

/// Turn work that was active when the daemon stopped into an explicit
/// reconciliation state. This is intentionally idempotent.
pub fn recover_interrupted(state: &CollectionState) -> Result<CollectionState> {
    state.validate()?;
    if !state.status.active() {
        return Ok(state.clone());
    }
    let mut next = state.clone();
    next.status = CollectionStatus::ReconcileRequired;
    next.diagnostic = Some(DAEMON_INTERRUPTED.to_owned());
    Ok(next)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobObservation {
    Exact {
        uid: String,
        state: JobState,
        exit_code: Option<i32>,
    },
    Absent,
    Ambiguous,
    Unavailable,
}

/// Apply a read-only provider observation. An unknown or ambiguous provider
/// response never permits a new create request. A durable receipt wins over a
/// late provider state because the archive is the collection's terminal proof.
pub fn observe_job(
    state: &CollectionState,
    observation: JobObservation,
) -> Result<CollectionState> {
    state.validate()?;
    if state.receipt.is_some() {
        let mut next = state.clone();
        next.status = CollectionStatus::Available;
        next.phase = CollectionPhase::Verifying;
        next.diagnostic = None;
        return Ok(next);
    }
    let Some(expected_uid) = state.job_uid.as_deref() else {
        return Ok(reconcile_state(state, EXACT_JOB_UNCONFIRMED));
    };
    let mut next = state.clone();
    match observation {
        JobObservation::Exact { uid, state: _, .. } if uid != expected_uid => {
            Err(Error::ProviderIdentityConflict)
        }
        JobObservation::Exact {
            state: job_state, ..
        } if job_state.terminal() => {
            next.status = CollectionStatus::Failed;
            next.phase = CollectionPhase::NasRecovery;
            next.diagnostic = Some(JOB_EXITED_WITHOUT_RECEIPT.to_owned());
            Ok(next)
        }
        JobObservation::Exact { .. } => {
            next.status = CollectionStatus::Waiting;
            next.phase = CollectionPhase::NasRecovery;
            next.diagnostic = None;
            Ok(next)
        }
        JobObservation::Absent => Ok(reconcile_state(&next, EXACT_JOB_UNCONFIRMED)),
        JobObservation::Ambiguous => Ok(reconcile_state(&next, JOB_AMBIGUOUS)),
        JobObservation::Unavailable => Ok(reconcile_state(&next, JOB_QUERY_UNAVAILABLE)),
    }
}

fn reconcile_state(state: &CollectionState, diagnostic: &str) -> CollectionState {
    let mut next = state.clone();
    next.status = CollectionStatus::ReconcileRequired;
    next.diagnostic = Some(diagnostic.to_owned());
    next
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputFile {
    pub bytes: u64,
    pub regular: bool,
    pub hidden: bool,
    pub selected: bool,
}

/// Estimate peak bytes for local tar + validation tree + object write. The
/// caller supplies already-validated file selection; no filesystem access or
/// glob implementation belongs in the pure core.
pub fn archive_peak(files: &[OutputFile], block_bytes: u64) -> Result<u64> {
    if block_bytes == 0 {
        return Err(Error::CapacityOverflow);
    }
    let mut total = 0u64;
    let mut count = 0u64;
    for file in files {
        if !file.regular || file.hidden || !file.selected {
            continue;
        }
        let rounded = file
            .bytes
            .checked_add(block_bytes - 1)
            .and_then(|value| value.checked_div(block_bytes))
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .and_then(|value| value.checked_add(8192))
            .ok_or(Error::CapacityOverflow)?;
        total = total.checked_add(rounded).ok_or(Error::CapacityOverflow)?;
        count = count.checked_add(1).ok_or(Error::CapacityOverflow)?;
    }
    total
        .checked_add(10240)
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| value.checked_add(count.checked_mul(4096)?))
        .ok_or(Error::CapacityOverflow)
}

pub fn unfinished_archive_identity(
    candidates: impl IntoIterator<Item = ArchiveIdentity>,
) -> Result<Option<ArchiveIdentity>> {
    let mut unique = BTreeSet::new();
    for candidate in candidates {
        candidate.validate()?;
        unique.insert((candidate.sha256, candidate.bytes));
    }
    if unique.len() > 1 {
        return Err(Error::ArchiveConflict);
    }
    Ok(unique
        .into_iter()
        .next()
        .map(|(sha256, bytes)| ArchiveIdentity { sha256, bytes }))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PublishAction {
    Upload,
    AlreadyPublished,
    NotFound,
    ArchiveConflict,
    InsufficientStorage,
    Interrupted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishInput<'a> {
    pub local: Option<&'a ArchiveIdentity>,
    pub expected: Option<&'a ArchiveIdentity>,
    pub receipt: Option<&'a ArchiveIdentity>,
    pub capacity_ok: bool,
    pub stopping: bool,
}

/// Decide whether the imperative shell may perform one exact archive upload.
/// It never performs the upload, deletes a cache, or ACKs a download.
pub fn publish_action(input: PublishInput<'_>) -> PublishAction {
    if input.stopping {
        return PublishAction::Interrupted;
    }
    let Some(local) = input.local else {
        return PublishAction::NotFound;
    };
    if input.expected.is_some_and(|expected| expected != local) {
        return PublishAction::ArchiveConflict;
    }
    if input.receipt.is_some_and(|receipt| receipt == local) {
        return PublishAction::AlreadyPublished;
    }
    if !input.capacity_ok {
        return PublishAction::InsufficientStorage;
    }
    PublishAction::Upload
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> CollectionIdentity {
        CollectionIdentity {
            project: "demo".into(),
            run_id: "run-a".into(),
            attempt_id: "attempt-001".into(),
        }
    }

    fn state(status: CollectionStatus) -> CollectionState {
        CollectionState {
            identity: identity(),
            status,
            phase: CollectionPhase::Checking,
            attempts: 1,
            scheduler_name: Some("ml-expd-result-example".into()),
            job_uid: Some("job-1".into()),
            reconcile: false,
            diagnostic: None,
            receipt: None,
        }
    }

    #[test]
    fn begin_table_preserves_terminal_and_reconcile_gates() {
        let cases = [
            (
                None,
                BeginRequest {
                    retry: false,
                    reconcile: false,
                },
                BeginAction::CreateRecoveryJob,
            ),
            (
                Some(CollectionStatus::Queued),
                BeginRequest {
                    retry: false,
                    reconcile: false,
                },
                BeginAction::None,
            ),
            (
                Some(CollectionStatus::Failed),
                BeginRequest {
                    retry: false,
                    reconcile: false,
                },
                BeginAction::None,
            ),
            (
                Some(CollectionStatus::Failed),
                BeginRequest {
                    retry: true,
                    reconcile: false,
                },
                BeginAction::CreateRecoveryJob,
            ),
            (
                Some(CollectionStatus::ReconcileRequired),
                BeginRequest {
                    retry: true,
                    reconcile: false,
                },
                BeginAction::None,
            ),
            (
                Some(CollectionStatus::ReconcileRequired),
                BeginRequest {
                    retry: false,
                    reconcile: true,
                },
                BeginAction::ReconcileRecoveryJob,
            ),
        ];
        for (existing, request, action) in cases {
            let old = existing.map(state);
            let result = begin(
                old.as_ref(),
                identity(),
                OriginalAttempt { terminal: true },
                request,
            )
            .unwrap();
            assert_eq!(result.action, action);
        }
        assert_eq!(
            begin(
                None,
                identity(),
                OriginalAttempt { terminal: false },
                BeginRequest {
                    retry: false,
                    reconcile: false
                }
            )
            .unwrap_err(),
            Error::OriginalNotTerminal
        );
    }

    #[test]
    fn retry_budget_and_identity_are_fail_closed() {
        let mut old = state(CollectionStatus::Failed);
        old.attempts = DEFAULT_MAX_RETRIES;
        assert_eq!(
            begin(
                Some(&old),
                identity(),
                OriginalAttempt { terminal: true },
                BeginRequest {
                    retry: true,
                    reconcile: false
                }
            )
            .unwrap_err(),
            Error::RetryLimit
        );
        let mut foreign = identity();
        foreign.run_id = "other".into();
        assert_eq!(
            begin(
                Some(&old),
                foreign,
                OriginalAttempt { terminal: true },
                BeginRequest {
                    retry: false,
                    reconcile: false
                }
            )
            .unwrap_err(),
            Error::Identity
        );
    }

    #[test]
    fn interruption_is_idempotent_and_never_replays_submit() {
        for status in [
            CollectionStatus::Queued,
            CollectionStatus::Collecting,
            CollectionStatus::Submitting,
            CollectionStatus::Waiting,
        ] {
            let first = recover_interrupted(&state(status)).unwrap();
            let second = recover_interrupted(&first).unwrap();
            assert_eq!(first.status, CollectionStatus::ReconcileRequired);
            assert_eq!(first.diagnostic.as_deref(), Some(DAEMON_INTERRUPTED));
            assert_eq!(first, second);
        }
        let terminal = recover_interrupted(&state(CollectionStatus::Failed)).unwrap();
        assert_eq!(terminal.status, CollectionStatus::Failed);
    }

    #[test]
    fn provider_observation_requires_exact_uid_and_receipt_wins() {
        let mut current = state(CollectionStatus::Waiting);
        assert_eq!(
            observe_job(
                &current,
                JobObservation::Exact {
                    uid: "other".into(),
                    state: JobState::Running,
                    exit_code: None
                }
            )
            .unwrap_err(),
            Error::ProviderIdentityConflict
        );
        current.job_uid = Some("job-1".into());
        let failed = observe_job(
            &current,
            JobObservation::Exact {
                uid: "job-1".into(),
                state: JobState::Failed,
                exit_code: Some(74),
            },
        )
        .unwrap();
        assert_eq!(failed.status, CollectionStatus::Failed);
        assert_eq!(
            failed.diagnostic.as_deref(),
            Some(JOB_EXITED_WITHOUT_RECEIPT)
        );
        let held = observe_job(&current, JobObservation::Unavailable).unwrap();
        assert_eq!(held.status, CollectionStatus::ReconcileRequired);
        let mut published = current;
        published.receipt = Some(ArchiveIdentity {
            sha256: "a".repeat(64),
            bytes: 1,
        });
        assert_eq!(
            observe_job(&published, JobObservation::Absent)
                .unwrap()
                .status,
            CollectionStatus::Available
        );
    }

    #[test]
    fn archive_identity_conflicts_are_not_silently_replaced() {
        let one = ArchiveIdentity {
            sha256: "a".repeat(64),
            bytes: 10,
        };
        assert_eq!(unfinished_archive_identity(vec![]).unwrap(), None);
        assert_eq!(
            unfinished_archive_identity(vec![one.clone(), one.clone()]).unwrap(),
            Some(one.clone())
        );
        let other = ArchiveIdentity {
            sha256: "b".repeat(64),
            bytes: 10,
        };
        assert_eq!(
            unfinished_archive_identity(vec![one, other]).unwrap_err(),
            Error::ArchiveConflict
        );
    }

    #[test]
    fn archive_peak_matches_python_formula_and_skips_unselected_files() {
        let files = vec![
            OutputFile {
                bytes: 7,
                regular: true,
                hidden: false,
                selected: true,
            },
            OutputFile {
                bytes: 100,
                regular: true,
                hidden: true,
                selected: true,
            },
            OutputFile {
                bytes: 100,
                regular: true,
                hidden: false,
                selected: false,
            },
            OutputFile {
                bytes: 100,
                regular: false,
                hidden: false,
                selected: true,
            },
        ];
        assert_eq!(
            archive_peak(&files, 512).unwrap(),
            3 * (512 + 8192 + 10240) + 4096
        );
    }

    #[test]
    fn publish_action_is_ordered_fail_closed() {
        let local = ArchiveIdentity {
            sha256: "a".repeat(64),
            bytes: 10,
        };
        let other = ArchiveIdentity {
            sha256: "b".repeat(64),
            bytes: 10,
        };
        assert_eq!(
            publish_action(PublishInput {
                local: None,
                expected: None,
                receipt: None,
                capacity_ok: true,
                stopping: false
            }),
            PublishAction::NotFound
        );
        assert_eq!(
            publish_action(PublishInput {
                local: Some(&local),
                expected: Some(&other),
                receipt: None,
                capacity_ok: true,
                stopping: false
            }),
            PublishAction::ArchiveConflict
        );
        assert_eq!(
            publish_action(PublishInput {
                local: Some(&local),
                expected: None,
                receipt: Some(&local),
                capacity_ok: false,
                stopping: false
            }),
            PublishAction::AlreadyPublished
        );
        assert_eq!(
            publish_action(PublishInput {
                local: Some(&local),
                expected: None,
                receipt: None,
                capacity_ok: false,
                stopping: false
            }),
            PublishAction::InsufficientStorage
        );
        assert_eq!(
            publish_action(PublishInput {
                local: Some(&local),
                expected: None,
                receipt: None,
                capacity_ok: true,
                stopping: true
            }),
            PublishAction::Interrupted
        );
        assert_eq!(
            publish_action(PublishInput {
                local: Some(&local),
                expected: None,
                receipt: None,
                capacity_ok: true,
                stopping: false,
            }),
            PublishAction::Upload
        );
    }
}
