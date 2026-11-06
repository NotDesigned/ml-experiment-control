//! Pure artifact receipt, download-proof and retention decisions.
//!
//! Object storage, filesystem reads and clocks stay outside this module. The
//! functions here only validate frozen identities and return the next durable
//! value; callers must persist that value before performing an external effect.
pub mod publication;

use crate::identity::{validate_attempt, validate_id};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

pub const MAX_FILES: usize = 20_000;
pub const MAX_PATH: usize = 4096;
const MAX_RECEIPT_BYTES: usize = 8 * 1024 * 1024;
/// Budget for `listed_cost`, below the 8 MiB limit on a saved or sent State and
/// leaving room for the skip list and the rest of the record.
pub const MAX_LISTED_BYTES: usize = 6 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileReceipt {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveReceipt {
    pub project: String,
    pub run_id: String,
    pub attempt_id: String,
    pub sha256: String,
    pub bytes: u64,
    pub files: Vec<FileReceipt>,
    pub object_key: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RetentionStatus {
    Keep,
    Scheduled,
    Releasing,
    Released,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Retention {
    pub status: RetentionStatus,
    pub sha256: String,
    pub release_after: Option<String>,
    pub released_at: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadProof {
    pub archive_sha256: String,
    pub archive_bytes: u64,
    pub files: std::collections::BTreeMap<String, FileReceipt>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Acknowledgement {
    pub proof: DownloadProof,
    pub verified_at: String,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid artifact identity")]
    Identity,
    #[error("invalid artifact receipt")]
    Receipt,
    #[error("download proof differs from the immutable receipt")]
    ProofConflict,
    #[error("artifact is already released or expired")]
    Unavailable,
    #[error("invalid retention transition")]
    Transition,
}
pub type Result<T> = std::result::Result<T, Error>;

fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
/// Any relative path inside OUTPUT_DIR. A hidden name or one called `outputs` is an
/// ordinary result; an archive member is always `outputs/` plus exactly this path.
pub fn receipt_path(value: &str) -> bool {
    value.len() <= MAX_PATH
        && !value.is_empty()
        && !value.contains(['\\', '\0'])
        && value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}
/// What one archived file adds to its saved State: a receipt row, and after ACK a
/// proof row that repeats the path in its `outputs/` key.
pub fn listed_cost(path: &str) -> usize {
    240 + 3 * (path.len() + path.matches('"').count())
}
fn bounded(value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(value).map_err(|_| Error::Receipt)?;
    if bytes.len() > MAX_RECEIPT_BYTES {
        Err(Error::Receipt)
    } else {
        Ok(())
    }
}

fn timestamp(value: &str) -> Result<OffsetDateTime> {
    if value.is_empty() || value.len() > 128 {
        return Err(Error::Transition);
    }
    OffsetDateTime::parse(value, &Rfc3339).map_err(|_| Error::Transition)
}

fn validate_retention(retention: &Retention, expected_sha: &str) -> Result<()> {
    if retention.sha256 != expected_sha || !sha256(&retention.sha256) {
        return Err(Error::Transition);
    }
    match retention.status {
        RetentionStatus::Keep => {
            if retention.release_after.is_some() || retention.released_at.is_some() {
                return Err(Error::Transition);
            }
        }
        RetentionStatus::Scheduled | RetentionStatus::Releasing => {
            let release_after = retention
                .release_after
                .as_deref()
                .ok_or(Error::Transition)?;
            timestamp(release_after)?;
            if retention.released_at.is_some() {
                return Err(Error::Transition);
            }
        }
        RetentionStatus::Released => {
            if retention.release_after.is_none() || retention.released_at.is_none() {
                return Err(Error::Transition);
            }
            timestamp(retention.release_after.as_deref().unwrap())?;
            timestamp(retention.released_at.as_deref().unwrap())?;
        }
    }
    Ok(())
}

pub fn validate_receipt(value: &ArchiveReceipt) -> Result<()> {
    validate_id(&value.project).map_err(|_| Error::Identity)?;
    validate_id(&value.run_id).map_err(|_| Error::Identity)?;
    validate_attempt(&value.attempt_id).map_err(|_| Error::Identity)?;
    if value.bytes == 0 || !sha256(&value.sha256) || value.files.len() > MAX_FILES {
        return Err(Error::Receipt);
    }
    let mut paths = BTreeSet::new();
    for file in &value.files {
        if !receipt_path(&file.path) || !sha256(&file.sha256) || !paths.insert(file.path.as_str()) {
            return Err(Error::Receipt);
        }
    }
    if value.object_key
        != format!(
            "{}/{}/{}/{}.tar",
            value.project, value.run_id, value.attempt_id, value.sha256
        )
    {
        return Err(Error::Receipt);
    }
    bounded(&serde_json::to_value(value).map_err(|_| Error::Receipt)?)
}

pub fn proof_matches(receipt: &ArchiveReceipt, proof: &DownloadProof) -> Result<()> {
    validate_receipt(receipt)?;
    if proof.archive_sha256 != receipt.sha256
        || proof.archive_bytes != receipt.bytes
        || proof.files.len() != receipt.files.len()
    {
        return Err(Error::ProofConflict);
    }
    for file in &receipt.files {
        let proof_path = format!("outputs/{}", file.path);
        let Some(observed) = proof.files.get(&proof_path) else {
            return Err(Error::ProofConflict);
        };
        if observed.path != file.path
            || observed.bytes != file.bytes
            || observed.sha256 != file.sha256
        {
            return Err(Error::ProofConflict);
        }
    }
    bounded(&serde_json::to_value(proof).map_err(|_| Error::ProofConflict)?)
}

pub fn acknowledge(
    receipt: &ArchiveReceipt,
    proof: DownloadProof,
    verified_at: &str,
    release: bool,
) -> Result<(Acknowledgement, Retention)> {
    proof_matches(receipt, &proof)?;
    timestamp(verified_at).map_err(|_| Error::ProofConflict)?;
    let acknowledgement = Acknowledgement {
        proof,
        verified_at: verified_at.into(),
    };
    let retention = Retention {
        status: if release {
            RetentionStatus::Scheduled
        } else {
            RetentionStatus::Keep
        },
        sha256: receipt.sha256.clone(),
        release_after: if release {
            Some(verified_at.into())
        } else {
            None
        },
        released_at: None,
    };
    Ok((acknowledgement, retention))
}

pub fn begin_release(receipt: &ArchiveReceipt, retention: &Retention) -> Result<Retention> {
    validate_receipt(receipt)?;
    validate_retention(retention, &receipt.sha256)?;
    match retention.status {
        RetentionStatus::Scheduled | RetentionStatus::Releasing => Ok(Retention {
            status: RetentionStatus::Releasing,
            ..retention.clone()
        }),
        RetentionStatus::Released => Ok(retention.clone()),
        RetentionStatus::Keep => Err(Error::Unavailable),
    }
}

pub fn finish_release(
    receipt: &ArchiveReceipt,
    retention: &Retention,
    released_at: &str,
) -> Result<Retention> {
    validate_receipt(receipt)?;
    validate_retention(retention, &receipt.sha256)?;
    if retention.status != RetentionStatus::Releasing {
        return Err(Error::Transition);
    }
    let released = timestamp(released_at)?;
    let release_after = timestamp(
        retention
            .release_after
            .as_deref()
            .ok_or(Error::Transition)?,
    )?;
    if released < release_after {
        return Err(Error::Transition);
    }
    Ok(Retention {
        status: RetentionStatus::Released,
        released_at: Some(released_at.into()),
        ..retention.clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt() -> ArchiveReceipt {
        ArchiveReceipt {
            project: "demo".into(),
            run_id: "run-1".into(),
            attempt_id: "attempt-001".into(),
            sha256: "a".repeat(64),
            bytes: 10,
            files: vec![FileReceipt {
                path: "model.bin".into(),
                bytes: 10,
                sha256: "b".repeat(64),
            }],
            object_key: format!("demo/run-1/attempt-001/{}.tar", "a".repeat(64)),
        }
    }
    #[test]
    fn nothing_expires_an_archive() {
        for status in ["EXPIRING", "EXPIRED"] {
            assert!(serde_json::from_value::<RetentionStatus>(status.into()).is_err());
        }
        assert_eq!(
            serde_json::from_value::<RetentionStatus>("RELEASED".into()).unwrap(),
            RetentionStatus::Released
        );
    }
    #[test]
    fn proof_and_release_are_exact_and_replay_safe() {
        let receipt = receipt();
        let proof = DownloadProof {
            archive_sha256: receipt.sha256.clone(),
            archive_bytes: receipt.bytes,
            files: receipt
                .files
                .iter()
                .map(|file| (format!("outputs/{}", file.path), file.clone()))
                .collect(),
        };
        let (ack, keep) =
            acknowledge(&receipt, proof.clone(), "2026-10-28T00:00:00Z", false).unwrap();
        assert_eq!(keep.status, RetentionStatus::Keep);
        assert_eq!(ack.proof, proof);
        let (_, scheduled) = acknowledge(&receipt, proof, "2026-10-28T00:00:00Z", true).unwrap();
        let releasing = begin_release(&receipt, &scheduled).unwrap();
        assert_eq!(releasing.status, RetentionStatus::Releasing);
        assert_eq!(begin_release(&receipt, &releasing).unwrap(), releasing);
        assert_eq!(
            finish_release(&receipt, &releasing, "2026-10-28T00:01:00Z")
                .unwrap()
                .status,
            RetentionStatus::Released
        );
    }
    #[test]
    fn foreign_files_paths_hashes_and_attempts_never_match() {
        let receipt = receipt();
        let mut proof = DownloadProof {
            archive_sha256: receipt.sha256.clone(),
            archive_bytes: receipt.bytes,
            files: receipt
                .files
                .iter()
                .map(|file| (format!("outputs/{}", file.path), file.clone()))
                .collect(),
        };
        proof
            .files
            .insert("outputs/other.bin".into(), receipt.files[0].clone());
        assert_eq!(proof_matches(&receipt, &proof), Err(Error::ProofConflict));
        let mut bad = receipt.clone();
        bad.attempt_id = "attempt-002".into();
        bad.object_key = format!("demo/run-1/attempt-001/{}.tar", "a".repeat(64));
        assert_eq!(validate_receipt(&bad), Err(Error::Receipt));
    }

    #[test]
    fn zero_byte_regular_outputs_are_valid_receipt_members() {
        let mut receipt = receipt();
        receipt.bytes = 1;
        receipt.files[0].bytes = 0;
        assert!(validate_receipt(&receipt).is_ok());
    }

    #[test]
    fn an_archive_with_no_members_is_a_valid_receipt() {
        let mut receipt = receipt();
        receipt.files.clear();
        assert!(validate_receipt(&receipt).is_ok());
    }

    #[test]
    fn hidden_names_and_outputs_are_paths_while_dot_components_stay_refused() {
        for path in [
            ".hydra/config.yaml",
            "a/.b/c",
            "..data",
            "...",
            "outputs",
            "outputs/model.pt",
        ] {
            assert!(receipt_path(path), "{path}");
        }
        for path in ["", ".", "..", "./a", "a/.", "../a", "a/../b", "a//b", "/a"] {
            assert!(!receipt_path(path), "{path}");
        }
    }
}
