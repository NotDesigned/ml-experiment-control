//! Pure contracts for resumable archive uploads.
//!
//! The Python implementation still owns locks, staging files, capacity checks,
//! streaming hashes and publication.  This module contains the decisions that
//! must remain stable across those effects: the content-addressed upload ID,
//! policy and part arithmetic, session reuse/restart, idempotent part writes,
//! and the checks required before publication.
use crate::canonical::{self, JsonOptions};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const UPLOAD_ID_PREFIX: &str = "upload.";
pub const DEFAULT_PART_BYTES: u64 = 16 * 1024 * 1024;
pub const MIN_PART_BYTES: u64 = 1024;
pub const MAX_PART_BYTES: u64 = 64 * 1024 * 1024;
pub const DEFAULT_SESSION_SECONDS: u64 = 86_400;
pub const MIN_SESSION_SECONDS: u64 = 300;
pub const MAX_SESSION_SECONDS: u64 = 604_800;
pub const DEFAULT_MAX_PARTS: u64 = 65_536;
pub const MAX_MAX_PARTS: u64 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid multipart upload policy")]
    Policy,
    #[error("invalid multipart upload identity")]
    Identity,
    #[error("multipart upload exceeds its configured limit")]
    Limit,
    #[error("multipart upload record does not match its requested identity")]
    RecordConflict,
    #[error("multipart upload record is invalid")]
    Record,
    #[error("completed multipart uploads cannot be aborted")]
    Completed,
    #[error("invalid multipart upload part number")]
    PartNumber,
    #[error("multipart upload part length differs")]
    PartSize,
    #[error("multipart upload part digest is invalid")]
    PartDigest,
    #[error("multipart upload part is already sealed with another digest")]
    PartConflict,
    #[error("multipart upload is missing parts")]
    MissingParts,
    #[error("multipart archive SHA256 differs from upload identity")]
    WholeDigest,
}
pub type Result<T> = std::result::Result<T, Error>;

fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// The Python implementation uses `json.dumps([binding, digest, size],
/// sort_keys=True)` with its default separators and `ensure_ascii=True`.
/// `compact=false` is therefore intentional: the spaces are part of the
/// historical upload identity and changing them would strand resumable
/// sessions created by Python.
pub fn upload_id(binding: &Value, digest: &str, size: u64) -> Result<String> {
    let body = canonical::encode(
        &json!([binding, digest, size]),
        JsonOptions {
            sort_keys: true,
            ensure_ascii: true,
            compact: false,
        },
    )
    .map_err(|_| Error::Identity)?;
    Ok(format!("{}{:x}", UPLOAD_ID_PREFIX, Sha256::digest(body)))
}

/// Compatibility spelling used by the Python `UploadStore`.
pub fn identity(binding: &Value, digest: &str, size: u64) -> Result<String> {
    upload_id(binding, digest, size)
}

pub fn validate_upload_id(value: &str) -> Result<()> {
    if value.strip_prefix(UPLOAD_ID_PREFIX).is_some_and(sha256) {
        Ok(())
    } else {
        Err(Error::Identity)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub part_bytes: u64,
    pub session_seconds: u64,
    pub max_parts: u64,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            part_bytes: DEFAULT_PART_BYTES,
            session_seconds: DEFAULT_SESSION_SECONDS,
            max_parts: DEFAULT_MAX_PARTS,
        }
    }
}

impl Policy {
    pub fn validate(&self) -> Result<()> {
        if !(MIN_PART_BYTES..=MAX_PART_BYTES).contains(&self.part_bytes)
            || !(MIN_SESSION_SECONDS..=MAX_SESSION_SECONDS).contains(&self.session_seconds)
            || !(1..=MAX_MAX_PARTS).contains(&self.max_parts)
        {
            return Err(Error::Policy);
        }
        Ok(())
    }
}

/// Return the number of parts for one archive and reject a policy/size pair
/// that cannot be represented by the configured maximum.
pub fn part_count(size: u64, policy: Policy) -> Result<u64> {
    policy.validate()?;
    if size == 0 {
        return Err(Error::Limit);
    }
    let count = size
        .checked_add(policy.part_bytes - 1)
        .ok_or(Error::Limit)?
        / policy.part_bytes;
    if count == 0 || count > policy.max_parts {
        return Err(Error::Limit);
    }
    Ok(count)
}

/// Return the exact expected bytes for a zero-based part number.
pub fn part_size(size: u64, policy: Policy, number: u64) -> Result<u64> {
    let count = part_count(size, policy)?;
    if number >= count {
        return Err(Error::PartNumber);
    }
    Ok(std::cmp::min(
        policy.part_bytes,
        size - number * policy.part_bytes,
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UploadPlan {
    pub upload_id: String,
    pub binding: Value,
    pub sha256: String,
    pub bytes: u64,
    pub part_bytes: u64,
    pub part_count: u64,
}

impl UploadPlan {
    pub fn new(
        binding: Value,
        digest: impl Into<String>,
        size: u64,
        limit: Option<u64>,
        policy: Policy,
    ) -> Result<Self> {
        policy.validate()?;
        let digest = digest.into();
        if !sha256(&digest) || size == 0 || limit.is_some_and(|limit| size > limit) {
            return Err(Error::Limit);
        }
        let count = part_count(size, policy)?;
        let upload_id = upload_id(&binding, &digest, size)?;
        Ok(Self {
            upload_id,
            binding,
            sha256: digest,
            bytes: size,
            part_bytes: policy.part_bytes,
            part_count: count,
        })
    }

    pub fn part_size(&self, number: u64) -> Result<u64> {
        part_size(
            self.bytes,
            Policy {
                part_bytes: self.part_bytes,
                ..Policy::default()
            },
            number,
        )
    }

    pub fn matches_record(&self, record: &UploadRecord) -> bool {
        self.upload_id == record.upload_id
            && self.binding == record.binding
            && self.sha256 == record.sha256
            && self.bytes == record.bytes
            && self.part_bytes == record.part_bytes
            && self.part_count == record.part_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UploadStatus {
    Uploading,
    Completed,
    Aborted,
    Expired,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartRecord {
    pub sha256: String,
    pub bytes: u64,
}

impl PartRecord {
    pub fn validate(&self) -> Result<()> {
        if !sha256(&self.sha256) || self.bytes == 0 {
            Err(Error::Record)
        } else {
            Ok(())
        }
    }
}

/// The pure portion of the durable `upload.json` record. `result`, error
/// diagnostics and provider-specific fields remain in the Python shell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UploadRecord {
    pub upload_id: String,
    pub binding: Value,
    pub sha256: String,
    pub bytes: u64,
    pub part_bytes: u64,
    pub part_count: u64,
    #[serde(default)]
    pub parts: BTreeMap<u64, PartRecord>,
    pub status: UploadStatus,
}

impl UploadRecord {
    pub fn from_plan(plan: &UploadPlan) -> Self {
        Self {
            upload_id: plan.upload_id.clone(),
            binding: plan.binding.clone(),
            sha256: plan.sha256.clone(),
            bytes: plan.bytes,
            part_bytes: plan.part_bytes,
            part_count: plan.part_count,
            parts: BTreeMap::new(),
            status: UploadStatus::Uploading,
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_upload_id(&self.upload_id)?;
        if !sha256(&self.sha256)
            || self.bytes == 0
            || !(MIN_PART_BYTES..=MAX_PART_BYTES).contains(&self.part_bytes)
            || !(1..=MAX_MAX_PARTS).contains(&self.part_count)
        {
            return Err(Error::Record);
        }
        if upload_id(&self.binding, &self.sha256, self.bytes)? != self.upload_id {
            return Err(Error::Record);
        }
        let expected = self
            .bytes
            .checked_add(self.part_bytes - 1)
            .ok_or(Error::Record)?
            / self.part_bytes;
        if expected != self.part_count || self.parts.keys().any(|number| *number >= self.part_count)
        {
            return Err(Error::Record);
        }
        for part in self.parts.values() {
            part.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestartReason {
    Expired,
    Aborted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CreateDecision {
    Create {
        plan: UploadPlan,
    },
    Reuse {
        record: UploadRecord,
    },
    Restart {
        plan: UploadPlan,
        reason: RestartReason,
    },
}

/// Decide whether an upload request creates a session, reuses an immutable
/// active/completed record, or starts a new generation after expiry/abort.
/// `expired` is supplied by the clock-owning shell; this function has no clock
/// or timestamp dependency.
pub fn create_or_reuse(
    binding: Value,
    digest: impl Into<String>,
    size: u64,
    limit: Option<u64>,
    policy: Policy,
    existing: Option<&UploadRecord>,
    expired: bool,
) -> Result<CreateDecision> {
    let plan = UploadPlan::new(binding, digest, size, limit, policy)?;
    let Some(record) = existing else {
        return Ok(CreateDecision::Create { plan });
    };
    record.validate()?;
    if !plan.matches_record(record) {
        return Err(Error::RecordConflict);
    }
    match record.status {
        UploadStatus::Uploading if !expired => Ok(CreateDecision::Reuse {
            record: record.clone(),
        }),
        UploadStatus::Completed => Ok(CreateDecision::Reuse {
            record: record.clone(),
        }),
        UploadStatus::Uploading | UploadStatus::Expired => Ok(CreateDecision::Restart {
            plan,
            reason: RestartReason::Expired,
        }),
        UploadStatus::Aborted => Ok(CreateDecision::Restart {
            plan,
            reason: RestartReason::Aborted,
        }),
    }
}

/// Convert an expired active record into the explicit terminal state used by
/// the Python pruner. The caller supplies the expiry observation; this function
/// never reads a clock or removes files.
pub fn mark_expired(record: &UploadRecord, expired: bool) -> Result<UploadRecord> {
    record.validate()?;
    if expired && record.status == UploadStatus::Uploading {
        Ok(UploadRecord {
            status: UploadStatus::Expired,
            parts: BTreeMap::new(),
            ..record.clone()
        })
    } else {
        Ok(record.clone())
    }
}

/// Record an explicit abort. The shell persists this returned state before it
/// removes staged parts. A completed publication is immutable and cannot be
/// aborted.
pub fn abort(record: &UploadRecord) -> Result<UploadRecord> {
    record.validate()?;
    if record.status == UploadStatus::Completed {
        return Err(Error::Completed);
    }
    Ok(UploadRecord {
        status: UploadStatus::Aborted,
        parts: BTreeMap::new(),
        ..record.clone()
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartDecision {
    Accept { part: PartRecord },
    Duplicate { part: PartRecord },
}

/// Check one part before the shell moves a temporary file into durable
/// staging. An exact retry is idempotent; a different digest/length for an
/// already sealed number is a conflict and must not replace the old bytes.
pub fn check_part(
    plan: &UploadPlan,
    existing: Option<&PartRecord>,
    number: u64,
    digest: impl Into<String>,
    bytes: u64,
) -> Result<PartDecision> {
    let digest = digest.into();
    let expected = plan.part_size(number)?;
    if !sha256(&digest) {
        return Err(Error::PartDigest);
    }
    if bytes != expected {
        return Err(Error::PartSize);
    }
    let part = PartRecord {
        sha256: digest,
        bytes,
    };
    match existing {
        None => Ok(PartDecision::Accept { part }),
        Some(previous) => {
            previous.validate()?;
            if previous == &part {
                Ok(PartDecision::Duplicate {
                    part: previous.clone(),
                })
            } else {
                Err(Error::PartConflict)
            }
        }
    }
}

/// Validate all preconditions that can be checked after the shell has streamed
/// the parts and computed the whole-archive digest. The map is keyed by the
/// zero-based Python part number; no provider or file-system work happens here.
pub fn check_complete(
    plan: &UploadPlan,
    parts: &BTreeMap<u64, PartRecord>,
    whole_sha256: &str,
) -> Result<()> {
    if parts.len() as u64 != plan.part_count || !parts.keys().copied().eq(0..plan.part_count) {
        return Err(Error::MissingParts);
    }
    if whole_sha256 != plan.sha256 {
        return Err(Error::WholeDigest);
    }
    for (number, part) in parts {
        let expected = plan.part_size(*number)?;
        if !sha256(&part.sha256) || part.bytes != expected {
            return Err(Error::Record);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> Value {
        json!({"kind":"asset", "project":"demo", "unicode":"模型"})
    }
    fn plan(size: u64) -> UploadPlan {
        UploadPlan::new(
            binding(),
            "a".repeat(64),
            size,
            Some(100_000),
            Policy {
                part_bytes: 1024,
                ..Policy::default()
            },
        )
        .unwrap()
    }
    fn part(digest: char, bytes: u64) -> PartRecord {
        PartRecord {
            sha256: digest.to_string().repeat(64),
            bytes,
        }
    }

    #[test]
    fn upload_id_matches_python_default_json_spacing_and_ascii() {
        let id = upload_id(&binding(), &"a".repeat(64), 12).unwrap();
        assert_eq!(id.len(), 71);
        assert!(id.starts_with("upload."));
        assert_eq!(
            id,
            "upload.5cefcdc48912cfd1fd570fc51e7e8f18cd6ddb2f1121bba89cf7a0f44097819c"
        );
        assert_eq!(id, identity(&binding(), &"a".repeat(64), 12).unwrap());
        assert_ne!(
            id,
            upload_id(
                &json!({"kind":"asset", "project":"demo", "unicode":"模型"}),
                &"a".repeat(64),
                13
            )
            .unwrap()
        );
    }

    #[test]
    fn policy_accepts_reference_bounds_and_rejects_outside() {
        assert!(Policy::default().validate().is_ok());
        assert!(
            Policy {
                part_bytes: 1023,
                ..Policy::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Policy {
                part_bytes: 64 * 1024 * 1024 + 1,
                ..Policy::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Policy {
                session_seconds: 299,
                ..Policy::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Policy {
                max_parts: 65_537,
                ..Policy::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn part_arithmetic_uses_zero_based_final_short_part() {
        let policy = Policy {
            part_bytes: 1024,
            ..Policy::default()
        };
        assert_eq!(part_count(2048, policy).unwrap(), 2);
        assert_eq!(part_size(2050, policy, 0).unwrap(), 1024);
        assert_eq!(part_size(2050, policy, 1).unwrap(), 1024);
        assert_eq!(part_size(2050, policy, 2).unwrap(), 2);
        assert_eq!(part_size(2050, policy, 3), Err(Error::PartNumber));
    }

    #[test]
    fn plan_rejects_invalid_digest_limit_and_part_count() {
        let p = Policy {
            part_bytes: 1024,
            max_parts: 2,
            ..Policy::default()
        };
        assert_eq!(
            UploadPlan::new(binding(), "bad", 1, None, p),
            Err(Error::Limit)
        );
        assert_eq!(
            UploadPlan::new(binding(), "a".repeat(64), 1, Some(0), p),
            Err(Error::Limit)
        );
        assert_eq!(
            UploadPlan::new(binding(), "a".repeat(64), 2049, None, p),
            Err(Error::Limit)
        );
    }

    #[test]
    fn create_missing_session_returns_create() {
        let decision = create_or_reuse(
            binding(),
            "a".repeat(64),
            100,
            None,
            Policy::default(),
            None,
            false,
        )
        .unwrap();
        assert!(matches!(decision, CreateDecision::Create { .. }));
    }

    #[test]
    fn create_reuses_active_and_completed_sessions() {
        let p = plan(100);
        let record = UploadRecord::from_plan(&p);
        assert!(matches!(
            create_or_reuse(
                binding(),
                p.sha256.clone(),
                100,
                None,
                Policy {
                    part_bytes: 1024,
                    ..Policy::default()
                },
                Some(&record),
                false
            )
            .unwrap(),
            CreateDecision::Reuse { .. }
        ));
        let completed = UploadRecord {
            status: UploadStatus::Completed,
            ..record
        };
        assert!(matches!(
            create_or_reuse(
                binding(),
                p.sha256,
                100,
                None,
                Policy {
                    part_bytes: 1024,
                    ..Policy::default()
                },
                Some(&completed),
                true
            )
            .unwrap(),
            CreateDecision::Reuse { .. }
        ));
    }

    #[test]
    fn create_restarts_expired_or_aborted_sessions() {
        let p = plan(100);
        let expired = UploadRecord {
            status: UploadStatus::Uploading,
            ..UploadRecord::from_plan(&p)
        };
        assert!(matches!(
            create_or_reuse(
                binding(),
                p.sha256.clone(),
                100,
                None,
                Policy {
                    part_bytes: 1024,
                    ..Policy::default()
                },
                Some(&expired),
                true
            )
            .unwrap(),
            CreateDecision::Restart {
                reason: RestartReason::Expired,
                ..
            }
        ));
        let aborted = UploadRecord {
            status: UploadStatus::Aborted,
            ..expired
        };
        assert!(matches!(
            create_or_reuse(
                binding(),
                p.sha256,
                100,
                None,
                Policy {
                    part_bytes: 1024,
                    ..Policy::default()
                },
                Some(&aborted),
                false
            )
            .unwrap(),
            CreateDecision::Restart {
                reason: RestartReason::Aborted,
                ..
            }
        ));
    }

    #[test]
    fn expiry_clears_only_expired_active_parts_and_abort_is_terminal() {
        let p = plan(100);
        let mut active = UploadRecord::from_plan(&p);
        active.parts.insert(0, part('b', 100));
        let expired = mark_expired(&active, true).unwrap();
        assert_eq!(expired.status, UploadStatus::Expired);
        assert!(expired.parts.is_empty());
        assert_eq!(mark_expired(&expired, true).unwrap(), expired);
        let aborted = abort(&active).unwrap();
        assert_eq!(aborted.status, UploadStatus::Aborted);
        assert!(aborted.parts.is_empty());
        let completed = UploadRecord {
            status: UploadStatus::Completed,
            ..active
        };
        assert_eq!(abort(&completed), Err(Error::Completed));
    }

    #[test]
    fn create_rejects_same_upload_id_with_changed_binding_or_size() {
        let p = plan(100);
        let record = UploadRecord::from_plan(&p);
        assert_eq!(
            create_or_reuse(
                json!({"project":"other"}),
                p.sha256.clone(),
                100,
                None,
                Policy {
                    part_bytes: 1024,
                    ..Policy::default()
                },
                Some(&record),
                false
            ),
            Err(Error::RecordConflict)
        );
        assert_eq!(
            create_or_reuse(
                binding(),
                p.sha256,
                101,
                None,
                Policy {
                    part_bytes: 1024,
                    ..Policy::default()
                },
                Some(&record),
                false
            ),
            Err(Error::RecordConflict)
        );
    }

    #[test]
    fn part_accepts_new_and_repeats_exact_part_idempotently() {
        let p = plan(2050);
        let one = check_part(&p, None, 0, "b".repeat(64), 1024).unwrap();
        let PartDecision::Accept { part } = one else {
            panic!("expected new part")
        };
        assert!(matches!(
            check_part(&p, Some(&part), 0, part.sha256.clone(), 1024).unwrap(),
            PartDecision::Duplicate { .. }
        ));
    }

    #[test]
    fn part_rejects_wrong_number_size_digest_and_conflict() {
        let p = plan(2050);
        assert_eq!(
            check_part(&p, None, 3, "b".repeat(64), 1),
            Err(Error::PartNumber)
        );
        assert_eq!(
            check_part(&p, None, 0, "b".repeat(64), 1),
            Err(Error::PartSize)
        );
        assert_eq!(check_part(&p, None, 0, "bad", 1024), Err(Error::PartDigest));
        assert_eq!(
            check_part(&p, Some(&part('b', 1024)), 0, "c".repeat(64), 1024),
            Err(Error::PartConflict)
        );
    }

    #[test]
    fn complete_requires_exact_part_set_and_whole_digest() {
        let p = plan(2050);
        let mut parts = BTreeMap::new();
        parts.insert(0, part('b', 1024));
        assert_eq!(
            check_complete(&p, &parts, &p.sha256),
            Err(Error::MissingParts)
        );
        parts.insert(1, part('c', 1024));
        parts.insert(2, part('d', 2));
        assert_eq!(
            check_complete(&p, &parts, &"b".repeat(64)),
            Err(Error::WholeDigest)
        );
        assert_eq!(check_complete(&p, &parts, &p.sha256), Ok(()));
        parts.remove(&1);
        parts.insert(3, part('c', 1024));
        assert_eq!(
            check_complete(&p, &parts, &p.sha256),
            Err(Error::MissingParts)
        );
    }
}
