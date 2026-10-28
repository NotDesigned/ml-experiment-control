//! Native helper metadata and decisions. This module never opens a path,
//! invokes a tool, reads a clock, or proves that a scheduler owns a job.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{JobId, component, scheduler_job_name};
use crate::{
    canonical,
    identity::{submission_marker, validate_attempt, validate_id},
};

pub const PROTOCOL_VERSION: u16 = 1;
pub const MAX_METADATA_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_FRAME_BYTES: usize = MAX_METADATA_BYTES + 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid Slurm helper protocol identity")]
    Identity,
    #[error("unsupported Slurm helper protocol version")]
    Version,
    #[error("invalid Slurm helper metadata frame")]
    Frame,
    #[error("invalid or unsupported Slurm helper metadata")]
    Metadata,
    #[error("Slurm helper metadata exceeds its bounded control-message limit")]
    TooLarge,
    #[error("Slurm helper frozen content differs")]
    Content,
}
pub type Result<T> = std::result::Result<T, Error>;

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}
pub fn valid_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|suffix| {
        suffix.len() == 64
            && suffix
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}
fn digest<T: Serialize>(value: &T) -> Result<String> {
    let value = serde_json::to_value(value).map_err(|_| Error::Metadata)?;
    canonical::protocol_identity(&value).map_err(|_| Error::Metadata)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub backend_id: String,
    pub profile_sha256: String,
    pub storage_scope: String,
}
impl Target {
    pub fn validate(&self) -> Result<()> {
        if !component(&self.backend_id)
            || !valid_digest(&self.profile_sha256)
            || !valid_digest(&self.storage_scope)
        {
            return Err(Error::Identity);
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    pub project: String,
    pub run_id: String,
    pub attempt_id: String,
}
impl Scope {
    pub fn validate(&self) -> Result<()> {
        validate_id(&self.project).map_err(|_| Error::Identity)?;
        validate_id(&self.run_id).map_err(|_| Error::Identity)?;
        validate_attempt(&self.attempt_id).map_err(|_| Error::Identity)?;
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        digest(self)
    }
}

// Deliberately no Debug: a token identifies an approved scheduler effect.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenIntent {
    pub target: Target,
    pub scope: Scope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub staging: Option<super::staging::Requirement>,
    pub submission_token: String,
    pub scheduler_name: String,
    pub manifest_sha256: String,
    pub script_sha256: String,
}
impl FrozenIntent {
    pub fn validate(&self) -> Result<()> {
        self.target.validate()?;
        self.scope.validate()?;
        if let Some(requirement) = &self.staging {
            requirement.validate()?;
        }
        submission_marker(&self.submission_token).map_err(|_| Error::Identity)?;
        if scheduler_job_name(&self.scope.run_id, &self.scope.attempt_id)
            .map_err(|_| Error::Identity)?
            != self.scheduler_name
            || !valid_digest(&self.manifest_sha256)
            || !valid_digest(&self.script_sha256)
        {
            return Err(Error::Identity);
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        digest(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationTag {
    PrepareSubmission,
    Submit,
    InspectSubmission,
    ObserveJob,
    ObserveWorker,
    ObserveCheckpoint,
    CancelJob,
}

// Never debug-format a manifest, script or request frame.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Operation {
    PrepareSubmission { manifest: String, script: String },
    Submit,
    InspectSubmission,
    ObserveJob { job_id: String },
    ObserveWorker { job_id: String },
    ObserveCheckpoint { job_id: String },
    CancelJob { job_id: String },
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    version: u16,
    intent: FrozenIntent,
    operation: Operation,
}
impl Request {
    pub fn prepare(intent: FrozenIntent, manifest: String, script: String) -> Result<Self> {
        let request = Self {
            version: PROTOCOL_VERSION,
            intent,
            operation: Operation::PrepareSubmission { manifest, script },
        };
        request.validate()?;
        Ok(request)
    }
    pub fn submit(intent: FrozenIntent) -> Result<Self> {
        let request = Self {
            version: PROTOCOL_VERSION,
            intent,
            operation: Operation::Submit,
        };
        request.validate()?;
        Ok(request)
    }
    pub fn inspect(intent: FrozenIntent) -> Result<Self> {
        let request = Self {
            version: PROTOCOL_VERSION,
            intent,
            operation: Operation::InspectSubmission,
        };
        request.validate()?;
        Ok(request)
    }
    pub fn observe_job(intent: FrozenIntent, job_id: String) -> Result<Self> {
        let request = Self {
            version: PROTOCOL_VERSION,
            intent,
            operation: Operation::ObserveJob { job_id },
        };
        request.validate()?;
        Ok(request)
    }
    pub fn observe_worker(intent: FrozenIntent, job_id: String) -> Result<Self> {
        let request = Self {
            version: PROTOCOL_VERSION,
            intent,
            operation: Operation::ObserveWorker { job_id },
        };
        request.validate()?;
        Ok(request)
    }
    pub fn observe_checkpoint(intent: FrozenIntent, job_id: String) -> Result<Self> {
        let request = Self {
            version: PROTOCOL_VERSION,
            intent,
            operation: Operation::ObserveCheckpoint { job_id },
        };
        request.validate()?;
        Ok(request)
    }
    pub fn cancel_job(intent: FrozenIntent, job_id: String) -> Result<Self> {
        let request = Self {
            version: PROTOCOL_VERSION,
            intent,
            operation: Operation::CancelJob { job_id },
        };
        request.validate()?;
        Ok(request)
    }
    pub fn requested_job_id(&self) -> Option<&str> {
        match &self.operation {
            Operation::ObserveJob { job_id }
            | Operation::ObserveWorker { job_id }
            | Operation::ObserveCheckpoint { job_id }
            | Operation::CancelJob { job_id } => Some(job_id),
            _ => None,
        }
    }
    pub fn intent(&self) -> &FrozenIntent {
        &self.intent
    }
    pub fn operation(&self) -> OperationTag {
        match self.operation {
            Operation::PrepareSubmission { .. } => OperationTag::PrepareSubmission,
            Operation::Submit => OperationTag::Submit,
            Operation::InspectSubmission => OperationTag::InspectSubmission,
            Operation::ObserveJob { .. } => OperationTag::ObserveJob,
            Operation::ObserveWorker { .. } => OperationTag::ObserveWorker,
            Operation::ObserveCheckpoint { .. } => OperationTag::ObserveCheckpoint,
            Operation::CancelJob { .. } => OperationTag::CancelJob,
        }
    }
    pub fn prepared_content(&self) -> Option<(&str, &str)> {
        match &self.operation {
            Operation::PrepareSubmission { manifest, script } => Some((manifest, script)),
            _ => None,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.version != PROTOCOL_VERSION {
            return Err(Error::Version);
        }
        self.intent.validate()?;
        if let Some(job_id) = self.requested_job_id() {
            let job = JobId::parse(job_id).map_err(|_| Error::Identity)?;
            if job.as_str().contains('_') {
                return Err(Error::Identity);
            }
        }
        if let Some((manifest, script)) = self.prepared_content() {
            if manifest.is_empty()
                || script.is_empty()
                || manifest.contains('\0')
                || script.contains('\0')
                || manifest.len() > MAX_METADATA_BYTES
                || script.len() > MAX_METADATA_BYTES
                || sha256_bytes(manifest.as_bytes()) != self.intent.manifest_sha256
                || sha256_bytes(script.as_bytes()) != self.intent.script_sha256
            {
                return Err(Error::Content);
            }
            super::staging::plan(&self.intent, manifest.as_bytes())?;
            // Require the generated long-form identity directives in pinned
            // bytes, but do not pretend this is a complete sbatch parser. The
            // executor must also force these fields on the sbatch CLI, whose
            // flags outrank environment values and short script aliases.
            let name = format!("#SBATCH --job-name={}", self.intent.scheduler_name);
            let marker = format!(
                "#SBATCH --comment={}",
                submission_marker(&self.intent.submission_token).map_err(|_| Error::Identity)?
            );
            let directives: Vec<_> = script
                .lines()
                .take_while(|line| line.is_empty() || line.starts_with('#'))
                .collect();
            if directives
                .iter()
                .filter(|line| line.starts_with("#SBATCH --job-name"))
                .copied()
                .collect::<Vec<_>>()
                != [name.as_str()]
                || directives
                    .iter()
                    .filter(|line| line.starts_with("#SBATCH --comment"))
                    .copied()
                    .collect::<Vec<_>>()
                    != [marker.as_str()]
            {
                return Err(Error::Content);
            }
        }
        if serde_json::to_vec(self).map_err(|_| Error::Metadata)?.len() > MAX_METADATA_BYTES {
            return Err(Error::TooLarge);
        }
        Ok(())
    }
    pub fn identity(&self) -> Result<RequestIdentity> {
        self.validate()?;
        Ok(RequestIdentity {
            request_sha256: digest(self)?,
            intent_sha256: self.intent.digest()?,
            operation: self.operation(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestIdentity {
    pub request_sha256: String,
    pub intent_sha256: String,
    pub operation: OperationTag,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UnknownReason {
    MissingLedger,
    LegacyClaim,
    MalformedLedger,
    SubmitStarted,
    UncertainProcessResult,
    StorageUnavailable,
    LockUnavailable,
    Cancelled,
    SchedulerNotObserved,
    SchedulerUnavailable,
    SchedulerEvidenceInvalid,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Rejection {
    IdentityConflict,
    Unprepared,
    UnsupportedOperation,
    CapacityFull,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "state",
    rename_all = "SCREAMING_SNAKE_CASE",
    deny_unknown_fields
)]
pub enum ReplyOutcome {
    Prepared,
    /// A durable ledger receipt only. This does NOT prove scheduler ownership.
    Submitted {
        job_id: String,
    },
    /// Current scheduler evidence with exact ownership checked in the same row.
    JobObserved {
        observation: Box<super::StatusObservation>,
    },
    WorkerObserved {
        evidence: Box<crate::managed::remote::Evidence>,
    },
    CheckpointObserved {
        document: Box<serde_json::Value>,
    },
    /// The durable cancel marker exists; no terminal provider read is proved.
    CancelPending {
        job_id: String,
    },
    Unknown {
        reason: UnknownReason,
    },
    Rejected {
        code: Rejection,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reply {
    pub identity: RequestIdentity,
    pub outcome: ReplyOutcome,
}
impl Reply {
    pub fn for_request(request: &Request, outcome: ReplyOutcome) -> Result<Self> {
        let reply = Self {
            identity: request.identity()?,
            outcome,
        };
        reply.validate_for(request)?;
        Ok(reply)
    }
    fn validate_for(&self, request: &Request) -> Result<()> {
        self.validate()?;
        if self.identity != request.identity()? {
            return Err(Error::Identity);
        }
        if let ReplyOutcome::WorkerObserved { evidence } = &self.outcome {
            evidence.validate_scope(
                &request.intent.scope,
                request.requested_job_id().ok_or(Error::Identity)?,
            )?;
        }
        if let ReplyOutcome::CheckpointObserved { document } = &self.outcome {
            let (context, _) = crate::checkpoints::validate_document(document, None, None)
                .map_err(|_| Error::Metadata)?;
            if self.identity.operation != OperationTag::ObserveCheckpoint
                || context.project != request.intent.scope.project
                || context.run_id != request.intent.scope.run_id
                || context.attempt_id != request.intent.scope.attempt_id
                || context.storage_scope != request.intent.target.storage_scope
            {
                return Err(Error::Identity);
            }
        }
        match &self.outcome {
            ReplyOutcome::WorkerObserved { .. }
                if self.identity.operation != OperationTag::ObserveWorker =>
            {
                return Err(Error::Metadata);
            }
            ReplyOutcome::JobObserved { observation }
                if request.requested_job_id() != Some(observation.status.job_id.as_str())
                    || observation.status.scheduler_name != request.intent.scheduler_name
                    || observation.status.ssh_alias != request.intent.target.backend_id =>
            {
                return Err(Error::Identity);
            }
            ReplyOutcome::CancelPending { job_id }
                if request.requested_job_id() != Some(job_id.as_str()) =>
            {
                return Err(Error::Identity);
            }
            _ => {}
        }
        // A scheduler reason is untrusted text. Refuse known-token echoes;
        // never rewrite frozen identity or scientific content to sanitize it.
        if serde_json::to_string(&self.outcome)
            .map_err(|_| Error::Metadata)?
            .contains(&request.intent.submission_token)
        {
            return Err(Error::Metadata);
        }
        Ok(())
    }
    fn validate(&self) -> Result<()> {
        if !valid_digest(&self.identity.request_sha256)
            || !valid_digest(&self.identity.intent_sha256)
        {
            return Err(Error::Identity);
        }
        if let ReplyOutcome::Submitted { job_id } = &self.outcome {
            let job = JobId::parse(job_id).map_err(|_| Error::Identity)?;
            if job.as_str().contains('_') {
                return Err(Error::Identity);
            }
        }
        match &self.outcome {
            ReplyOutcome::JobObserved { observation } => {
                observation.status.validate().map_err(|_| Error::Metadata)?;
                if observation.retained_previous
                    || observation.checked_at != observation.status.observed_at
                    || !matches!(
                        self.identity.operation,
                        OperationTag::ObserveJob | OperationTag::CancelJob
                    )
                {
                    return Err(Error::Metadata);
                }
            }
            ReplyOutcome::CancelPending { job_id } => {
                let job = JobId::parse(job_id).map_err(|_| Error::Identity)?;
                if job.as_str().contains('_') || self.identity.operation != OperationTag::CancelJob
                {
                    return Err(Error::Metadata);
                }
            }
            ReplyOutcome::Prepared | ReplyOutcome::Submitted { .. }
                if matches!(
                    self.identity.operation,
                    OperationTag::ObserveJob
                        | OperationTag::ObserveWorker
                        | OperationTag::ObserveCheckpoint
                        | OperationTag::CancelJob
                ) =>
            {
                return Err(Error::Metadata);
            }
            _ => {}
        }
        if self.identity.operation == OperationTag::Submit && self.outcome == ReplyOutcome::Prepared
        {
            return Err(Error::Metadata);
        }
        Ok(())
    }
    /// Point-in-time PREPARED and scheduler absence do not fence a delayed
    /// Submit request. This protocol has no CloseUnstarted operation.
    pub fn safe_to_retry(&self) -> bool {
        false
    }
}

/// The single durable head is mandatory after preparation. Immutable start and
/// receipt entries are additional consistency checks, never absence evidence.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerRecord {
    pub version: u16,
    pub intent: FrozenIntent,
    pub phase: LedgerPhase,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "state",
    rename_all = "SCREAMING_SNAKE_CASE",
    deny_unknown_fields
)]
pub enum LedgerPhase {
    Prepared,
    SubmitStarted,
    Submitted { job_id: String },
}
impl LedgerRecord {
    pub fn prepared(intent: FrozenIntent) -> Result<Self> {
        intent.validate()?;
        Ok(Self {
            version: PROTOCOL_VERSION,
            intent,
            phase: LedgerPhase::Prepared,
        })
    }
    pub fn validate(&self) -> Result<()> {
        if self.version != PROTOCOL_VERSION {
            return Err(Error::Version);
        }
        self.intent.validate()?;
        if let LedgerPhase::Submitted { job_id } = &self.phase {
            let job = JobId::parse(job_id).map_err(|_| Error::Identity)?;
            if job.as_str().contains('_') {
                return Err(Error::Identity);
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmitDecision {
    StartOnce,
    ReturnExisting { job_id: String },
    ReconcileRequired,
}
/// Interpret separately validated immutable receipt bytes without repairing the
/// durable head. A receipt can recover a response lost between receipt fsync
/// and head replacement, but never proves scheduler ownership.
pub fn ledger_outcome(
    record: &LedgerRecord,
    start_exists: bool,
    receipt_job: Option<&str>,
) -> Result<ReplyOutcome> {
    record.validate()?;
    if let Some(job_id) = receipt_job {
        let job = JobId::parse(job_id).map_err(|_| Error::Identity)?;
        if job.as_str().contains('_')
            || !start_exists
            || matches!(record.phase, LedgerPhase::Prepared)
            || matches!(&record.phase, LedgerPhase::Submitted { job_id: expected } if expected != job_id)
        {
            return Err(Error::Content);
        }
        return Ok(ReplyOutcome::Submitted {
            job_id: job_id.into(),
        });
    }
    Ok(match record.phase {
        LedgerPhase::Prepared if !start_exists => ReplyOutcome::Prepared,
        LedgerPhase::Submitted { .. } => ReplyOutcome::Unknown {
            reason: UnknownReason::MalformedLedger,
        },
        _ => ReplyOutcome::Unknown {
            reason: UnknownReason::SubmitStarted,
        },
    })
}
pub fn submission_gate(
    expected: &FrozenIntent,
    record: &LedgerRecord,
    start_exists: bool,
) -> Result<SubmitDecision> {
    expected.validate()?;
    record.validate()?;
    if &record.intent != expected {
        return Err(Error::Identity);
    }
    Ok(match &record.phase {
        LedgerPhase::Prepared if !start_exists => SubmitDecision::StartOnce,
        LedgerPhase::Submitted { job_id } if start_exists => SubmitDecision::ReturnExisting {
            job_id: job_id.clone(),
        },
        _ => SubmitDecision::ReconcileRequired,
    })
}

fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let body = serde_json::to_vec(value).map_err(|_| Error::Metadata)?;
    if body.is_empty() || body.len() > MAX_METADATA_BYTES {
        return Err(Error::TooLarge);
    }
    let mut frame = Vec::with_capacity(body.len() + 4);
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(&body);
    Ok(frame)
}
fn body(frame: &[u8]) -> Result<&[u8]> {
    let header: [u8; 4] = frame
        .get(..4)
        .ok_or(Error::Frame)?
        .try_into()
        .map_err(|_| Error::Frame)?;
    let length = u32::from_be_bytes(header) as usize;
    if length == 0 || length > MAX_METADATA_BYTES {
        return Err(Error::TooLarge);
    }
    if frame.len() != length + 4 {
        return Err(Error::Frame);
    }
    Ok(&frame[4..])
}
pub fn encode_request(request: &Request) -> Result<Vec<u8>> {
    request.validate()?;
    encode(request)
}
pub fn decode_request(frame: &[u8]) -> Result<Request> {
    let request: Request = serde_json::from_slice(body(frame)?).map_err(|_| Error::Metadata)?;
    request.validate()?;
    Ok(request)
}
pub fn encode_reply(reply: &Reply) -> Result<Vec<u8>> {
    reply.validate()?;
    encode(reply)
}
pub fn decode_reply(frame: &[u8], expected: &Request) -> Result<Reply> {
    let reply: Reply = serde_json::from_slice(body(frame)?).map_err(|_| Error::Metadata)?;
    reply.validate_for(expected)?;
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (FrozenIntent, String, String) {
        let manifest = "run_id: train\n".to_owned();
        let script = format!(
            "#!/usr/bin/env bash\n#SBATCH --job-name=train--attempt-001\n#SBATCH --comment=ml-exp-{}\n\ntrue\n",
            "a".repeat(32)
        );
        let intent = FrozenIntent {
            staging: None,
            target: Target {
                backend_id: "slurm-test".into(),
                profile_sha256: sha256_bytes(b"profile"),
                storage_scope: sha256_bytes(b"storage"),
            },
            scope: Scope {
                project: "demo".into(),
                run_id: "train".into(),
                attempt_id: "attempt-001".into(),
            },
            submission_token: "a".repeat(32),
            scheduler_name: "train--attempt-001".into(),
            manifest_sha256: sha256_bytes(manifest.as_bytes()),
            script_sha256: sha256_bytes(script.as_bytes()),
        };
        (intent, manifest, script)
    }
    #[test]
    fn strict_frames_bind_every_operation_and_frozen_byte() {
        let (intent, manifest, script) = fixture();
        let request = Request::prepare(intent.clone(), manifest, script).unwrap();
        let frame = encode_request(&request).unwrap();
        assert!(decode_request(&frame).unwrap() == request);
        for bad in [
            &frame[..frame.len() - 1],
            &[frame.clone(), vec![0]].concat(),
            &[0, 0, 0, 0],
            &[0, 128, 0, 1],
        ] {
            assert!(decode_request(bad).is_err());
        }
        let reply = Reply::for_request(&request, ReplyOutcome::Prepared).unwrap();
        assert!(!reply.safe_to_retry());
        assert!(
            Reply::for_request(
                &Request::submit(request.intent().clone()).unwrap(),
                ReplyOutcome::Prepared
            )
            .is_err()
        );
        assert_eq!(
            decode_reply(&encode_reply(&reply).unwrap(), &request).unwrap(),
            reply
        );
        assert!(
            decode_reply(
                &encode_reply(&reply).unwrap(),
                &Request::submit(intent).unwrap()
            )
            .is_err()
        );
        let mut json = serde_json::to_value(&request).unwrap();
        json["unknown"] = serde_json::json!(true);
        assert!(decode_request(&encode(&json).unwrap()).is_err());
        json.as_object_mut().unwrap().remove("unknown");
        json["intent"]["scope"]["run_id"] = serde_json::json!("../foreign");
        assert!(decode_request(&encode(&json).unwrap()).is_err());
    }
    #[test]
    fn identities_content_and_submit_gate_fail_closed() {
        let (intent, manifest, script) = fixture();
        let mut foreign = intent.clone();
        foreign.scheduler_name = "foreign".into();
        assert!(Request::submit(foreign).is_err());
        assert!(Request::prepare(intent.clone(), manifest, script + "tampered").is_err());
        let mut record = LedgerRecord::prepared(intent.clone()).unwrap();
        assert_eq!(
            submission_gate(&intent, &record, false).unwrap(),
            SubmitDecision::StartOnce
        );
        assert_eq!(
            submission_gate(&intent, &record, true).unwrap(),
            SubmitDecision::ReconcileRequired
        );
        record.phase = LedgerPhase::SubmitStarted;
        assert_eq!(
            ledger_outcome(&record, true, Some("123")).unwrap(),
            ReplyOutcome::Submitted {
                job_id: "123".into()
            }
        );
        assert!(ledger_outcome(&record, false, Some("123")).is_err());
        assert_eq!(
            submission_gate(&intent, &record, false).unwrap(),
            SubmitDecision::ReconcileRequired
        );
        record.phase = LedgerPhase::Submitted {
            job_id: "123".into(),
        };
        assert!(ledger_outcome(&record, true, Some("456")).is_err());
        assert_eq!(
            ledger_outcome(&record, true, None).unwrap(),
            ReplyOutcome::Unknown {
                reason: UnknownReason::MalformedLedger
            }
        );
        assert_eq!(
            submission_gate(&intent, &record, true).unwrap(),
            SubmitDecision::ReturnExisting {
                job_id: "123".into()
            }
        );
        assert_eq!(
            submission_gate(&intent, &record, false).unwrap(),
            SubmitDecision::ReconcileRequired
        );
        let mut foreign = intent;
        foreign.submission_token = "b".repeat(32);
        assert!(submission_gate(&foreign, &record, true).is_err());
    }
}
