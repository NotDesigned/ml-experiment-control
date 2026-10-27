//! Pure contracts for the bounded upload transfer buffer.
//!
//! The Python shell still owns file descriptors, locks, streaming hashes and
//! deletion fences.  This module owns the part identity, accounting and
//! receipt/state transitions so a restart cannot turn an unacknowledged part
//! into either an uncharged leak or an unsafe deletion.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const BUFFER_BYTES: u64 = 1024 * 1024 * 1024;
pub const LOCK_STRIPES: u64 = 128;
pub const SIDE_CAR_BYTES: u64 = 2048;
pub const MAX_PART_NUMBER: u32 = 65_535;
pub const RECEIPT_CONTRACT: &str = "upload-part.v1";

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid transfer part identity")]
    Identity,
    #[error("invalid transfer buffer state")]
    State,
    #[error("remote durable upload receipt differs")]
    Receipt,
    #[error("local buffered part length or SHA256 differs")]
    Digest,
    #[error("transfer buffer is full or this part is in flight")]
    Busy,
    #[error("insufficient transfer buffer space")]
    Storage,
    #[error("invalid transfer buffer accounting")]
    Accounting,
}
pub type Result<T> = std::result::Result<T, Error>;

fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn upload_id(value: &str) -> bool {
    value.strip_prefix("upload.").is_some_and(sha256)
}

/// Exact durable identity of one part. The shell may use the same value for
/// the part metadata JSON and the request body it is about to stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartIdentity {
    pub upload_id: String,
    pub number: u32,
    pub sha256: String,
    pub bytes: u64,
}

impl PartIdentity {
    pub fn validate(&self) -> Result<()> {
        if !upload_id(&self.upload_id)
            || self.number > MAX_PART_NUMBER
            || !sha256(&self.sha256)
            || self.bytes == 0
            || self.bytes > BUFFER_BYTES
        {
            return Err(Error::Identity);
        }
        Ok(())
    }

    pub fn key(&self) -> Result<String> {
        self.validate()?;
        Ok(format!(
            "{:x}",
            Sha256::digest(format!("{}:{}", self.upload_id, self.number).as_bytes())
        ))
    }

    pub fn metadata_name(&self) -> Result<String> {
        Ok(format!("part-{}.json", self.key()?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TransferStatus {
    Receiving,
    Sealed,
    Confirmed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableReceipt {
    pub contract: String,
    pub upload_id: String,
    pub number: u32,
    pub sha256: String,
    pub bytes: u64,
    pub durable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferState {
    pub upload_id: String,
    pub number: u32,
    pub sha256: String,
    pub bytes: u64,
    pub status: TransferStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<DurableReceipt>,
}

impl TransferState {
    pub fn identity(&self) -> PartIdentity {
        PartIdentity {
            upload_id: self.upload_id.clone(),
            number: self.number,
            sha256: self.sha256.clone(),
            bytes: self.bytes,
        }
    }

    pub fn validate(&self, metadata_name: &str) -> Result<()> {
        let identity = self.identity();
        identity.validate()?;
        if metadata_name != identity.metadata_name()? {
            return Err(Error::State);
        }
        if let Some(receipt) = &self.receipt {
            check_receipt(&identity, receipt)?;
        }
        Ok(())
    }

    pub fn new(identity: PartIdentity) -> Result<Self> {
        identity.validate()?;
        Ok(Self {
            upload_id: identity.upload_id,
            number: identity.number,
            sha256: identity.sha256,
            bytes: identity.bytes,
            status: TransferStatus::Receiving,
            receipt: None,
        })
    }
}

pub fn check_receipt(identity: &PartIdentity, receipt: &DurableReceipt) -> Result<()> {
    identity.validate()?;
    if receipt.contract != RECEIPT_CONTRACT
        || !receipt.durable
        || receipt.upload_id != identity.upload_id
        || receipt.number != identity.number
        || receipt.sha256 != identity.sha256
        || receipt.bytes != identity.bytes
    {
        return Err(Error::Receipt);
    }
    Ok(())
}

/// Filesystem-block accounting used by the one-GiB process-shared ceiling.
pub fn charged_bytes(bytes: u64, block_bytes: u64) -> Result<u64> {
    if bytes == 0 || bytes > BUFFER_BYTES || block_bytes == 0 {
        return Err(Error::Accounting);
    }
    let round = |value: u64| {
        value
            .checked_add(block_bytes - 1)
            .and_then(|value| value.checked_div(block_bytes))
            .and_then(|blocks| blocks.checked_mul(block_bytes))
    };
    round(bytes)
        .and_then(|payload| round(SIDE_CAR_BYTES)?.checked_mul(2)?.checked_add(payload))
        .and_then(|value| value.checked_add(block_bytes))
        .ok_or(Error::Accounting)
}

pub fn fixed_overhead(block_bytes: u64) -> Result<u64> {
    if block_bytes == 0 {
        return Err(Error::Accounting);
    }
    (LOCK_STRIPES + 1)
        .checked_mul(block_bytes)
        .ok_or(Error::Accounting)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferSnapshot {
    pub limit_bytes: u64,
    pub charged_bytes: u64,
    pub filesystem_block_bytes: u64,
}

impl BufferSnapshot {
    pub fn validate(&self) -> Result<()> {
        if self.limit_bytes == 0
            || self.limit_bytes > BUFFER_BYTES
            || self.filesystem_block_bytes == 0
        {
            return Err(Error::Accounting);
        }
        fixed_overhead(self.filesystem_block_bytes)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    pub state: TransferState,
    /// The shell could not acquire this part's stripe lock during recovery.
    /// Such a claim must remain charged and cannot be cleaned speculatively.
    pub active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryAction {
    KeepActive,
    KeepSealed,
    RemoveReceiving,
    RemoveConfirmed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaimDecision {
    pub identity: PartIdentity,
    pub action: RecoveryAction,
    pub charged_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferAccounting {
    pub limit_bytes: u64,
    pub reserved_bytes: u64,
    pub payload_bytes: u64,
    pub charged_bytes: u64,
    pub fixed_overhead_bytes: u64,
    pub filesystem_block_bytes: u64,
    pub pending_bytes: u64,
    pub available_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoverySnapshot {
    pub accounting: BufferAccounting,
    pub decisions: Vec<ClaimDecision>,
}

/// Reduce the shell's no-follow recovery observations into accounting and
/// cleanup actions. Lock acquisition, receipt file reads and deletion remain
/// effects; a caller must apply only the returned action after its own lock.
pub fn snapshot(
    claims: &[Claim],
    limit_bytes: u64,
    filesystem_block_bytes: u64,
) -> Result<RecoverySnapshot> {
    if limit_bytes == 0 || limit_bytes > BUFFER_BYTES || filesystem_block_bytes == 0 {
        return Err(Error::Accounting);
    }
    let fixed = fixed_overhead(filesystem_block_bytes)?;
    let mut reserved = 0u64;
    let mut pending = 0u64;
    let mut charged = fixed;
    let mut seen = BTreeSet::new();
    let mut decisions = Vec::with_capacity(claims.len());
    for claim in claims {
        let identity = claim.state.identity();
        let key = identity.key()?;
        if !seen.insert(key) {
            return Err(Error::State);
        }
        claim.state.validate(&identity.metadata_name()?)?;
        let keep = claim.active || claim.state.status == TransferStatus::Sealed;
        let (action, charge) = if claim.active {
            (
                RecoveryAction::KeepActive,
                charged_bytes(identity.bytes, filesystem_block_bytes)?,
            )
        } else {
            match claim.state.status {
                TransferStatus::Sealed => {
                    pending = pending
                        .checked_add(identity.bytes)
                        .ok_or(Error::Accounting)?;
                    (
                        RecoveryAction::KeepSealed,
                        charged_bytes(identity.bytes, filesystem_block_bytes)?,
                    )
                }
                TransferStatus::Receiving => (RecoveryAction::RemoveReceiving, 0),
                TransferStatus::Confirmed => {
                    check_receipt(
                        &identity,
                        claim.state.receipt.as_ref().ok_or(Error::Receipt)?,
                    )?;
                    (RecoveryAction::RemoveConfirmed, 0)
                }
            }
        };
        if keep {
            reserved = reserved
                .checked_add(identity.bytes)
                .ok_or(Error::Accounting)?;
            charged = charged.checked_add(charge).ok_or(Error::Accounting)?;
        }
        decisions.push(ClaimDecision {
            identity,
            action,
            charged_bytes: charge,
        });
    }
    Ok(RecoverySnapshot {
        accounting: BufferAccounting {
            limit_bytes,
            reserved_bytes: reserved,
            payload_bytes: reserved,
            charged_bytes: charged,
            fixed_overhead_bytes: fixed,
            filesystem_block_bytes,
            pending_bytes: pending,
            available_bytes: limit_bytes.saturating_sub(charged),
        },
        decisions,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionAction {
    Receive,
    Resume,
    ResetConfirmed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    pub state: TransferState,
    pub action: AdmissionAction,
    pub charged_bytes: u64,
}

/// Decide whether one part may reserve the bounded buffer. Filesystem locks
/// and the actual admission lock stay in Python; this function is the
/// deterministic capacity and identity half of that operation.
pub fn admit(
    existing: Option<&TransferState>,
    expected: PartIdentity,
    snapshot: BufferSnapshot,
    filesystem_free_bytes: u64,
) -> Result<Admission> {
    expected.validate()?;
    snapshot.validate()?;
    let expected_charge = charged_bytes(expected.bytes, snapshot.filesystem_block_bytes)?;
    let (state, old_charge, action) = match existing {
        None => (
            TransferState::new(expected.clone())?,
            0,
            AdmissionAction::Receive,
        ),
        Some(value) => {
            value.validate(&value.identity().metadata_name()?)?;
            if value.identity() != expected {
                return Err(Error::Identity);
            }
            match value.status {
                TransferStatus::Confirmed => {
                    check_receipt(&expected, value.receipt.as_ref().ok_or(Error::Receipt)?)?;
                    (
                        TransferState::new(expected.clone())?,
                        // A confirmed record is discarded by the shell's
                        // recovery snapshot before this decision is applied.
                        0,
                        AdmissionAction::ResetConfirmed,
                    )
                }
                TransferStatus::Receiving => {
                    (value.clone(), expected_charge, AdmissionAction::Resume)
                }
                TransferStatus::Sealed => (value.clone(), expected_charge, AdmissionAction::Resume),
            }
        }
    };
    let available_charge = snapshot
        .charged_bytes
        .checked_sub(old_charge)
        .ok_or(Error::Accounting)?
        .checked_add(expected_charge)
        .ok_or(Error::Accounting)?;
    if available_charge > snapshot.limit_bytes {
        return Err(Error::Busy);
    }
    if existing.is_none() && filesystem_free_bytes < expected.bytes {
        return Err(Error::Storage);
    }
    Ok(Admission {
        state,
        action,
        charged_bytes: available_charge,
    })
}

/// Seal a complete local body after the I/O shell has computed its size and
/// SHA256. A repeated seal of the same immutable part is idempotent.
pub fn seal(
    state: &TransferState,
    observed_bytes: u64,
    observed_sha256: &str,
) -> Result<TransferState> {
    state.validate(&state.identity().metadata_name()?)?;
    if observed_bytes != state.bytes || observed_sha256 != state.sha256 {
        return Err(Error::Digest);
    }
    match state.status {
        TransferStatus::Receiving => Ok(TransferState {
            status: TransferStatus::Sealed,
            ..state.clone()
        }),
        TransferStatus::Sealed => Ok(state.clone()),
        TransferStatus::Confirmed => Err(Error::State),
    }
}

/// Persist the durable remote receipt before the shell removes local bytes.
pub fn acknowledge(state: &TransferState, receipt: DurableReceipt) -> Result<TransferState> {
    state.validate(&state.identity().metadata_name()?)?;
    if state.status != TransferStatus::Sealed {
        return Err(Error::State);
    }
    check_receipt(&state.identity(), &receipt)?;
    Ok(TransferState {
        status: TransferStatus::Confirmed,
        receipt: Some(receipt),
        ..state.clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(number: u32, bytes: u64) -> PartIdentity {
        PartIdentity {
            upload_id: format!("upload.{}", "a".repeat(64)),
            number,
            sha256: "b".repeat(64),
            bytes,
        }
    }
    fn receipt(identity: &PartIdentity) -> DurableReceipt {
        DurableReceipt {
            contract: RECEIPT_CONTRACT.into(),
            upload_id: identity.upload_id.clone(),
            number: identity.number,
            sha256: identity.sha256.clone(),
            bytes: identity.bytes,
            durable: true,
        }
    }
    fn buffer_snapshot() -> BufferSnapshot {
        BufferSnapshot {
            limit_bytes: BUFFER_BYTES,
            charged_bytes: fixed_overhead(4096).unwrap(),
            filesystem_block_bytes: 4096,
        }
    }

    #[test]
    fn part_identity_and_metadata_name_are_deterministic() {
        let part = identity(3, 100);
        let key = part.key().unwrap();
        assert_eq!(key.len(), 64);
        assert_eq!(part.metadata_name().unwrap(), format!("part-{key}.json"));
        assert_eq!(part, part.clone());
    }

    #[test]
    fn identities_reject_wrong_upload_number_digest_and_size() {
        let mut part = identity(0, 1);
        part.upload_id = "upload.bad".into();
        assert_eq!(part.validate(), Err(Error::Identity));
        let part = identity(MAX_PART_NUMBER + 1, 1);
        assert_eq!(part.validate(), Err(Error::Identity));
        let mut part = identity(0, 1);
        part.sha256 = "c".repeat(63);
        assert_eq!(part.validate(), Err(Error::Identity));
        let part = identity(0, BUFFER_BYTES + 1);
        assert_eq!(part.validate(), Err(Error::Identity));
    }

    #[test]
    fn charged_bytes_include_payload_sidecars_and_fixed_entry() {
        assert_eq!(charged_bytes(1, 4096).unwrap(), 16_384);
        assert_eq!(fixed_overhead(4096).unwrap(), 129 * 4096);
        assert_eq!(charged_bytes(BUFFER_BYTES, 4096).unwrap(), 1_073_754_112);
    }

    #[test]
    fn receipts_must_be_durable_and_exact() {
        let part = identity(0, 100);
        let good = receipt(&part);
        assert!(check_receipt(&part, &good).is_ok());
        let mut bad = good.clone();
        bad.durable = false;
        assert_eq!(check_receipt(&part, &bad), Err(Error::Receipt));
        let mut bad = good;
        bad.number = 1;
        assert_eq!(check_receipt(&part, &bad), Err(Error::Receipt));
    }

    #[test]
    fn admission_new_part_uses_charged_capacity_and_filesystem_bytes() {
        let part = identity(0, 4096);
        let admitted = admit(None, part, buffer_snapshot(), 4096).unwrap();
        assert_eq!(admitted.action, AdmissionAction::Receive);
        assert_eq!(admitted.state.status, TransferStatus::Receiving);
        assert!(admitted.charged_bytes > admitted.state.bytes);
        assert_eq!(
            admit(None, identity(0, 4096), buffer_snapshot(), 0),
            Err(Error::Storage)
        );
    }

    #[test]
    fn sealed_part_resumes_without_double_charging() {
        let part = identity(0, 4096);
        let state = seal(
            &TransferState::new(part.clone()).unwrap(),
            4096,
            &part.sha256,
        )
        .unwrap();
        let admitted = admit(Some(&state), part, buffer_snapshot(), 0).unwrap();
        assert_eq!(admitted.action, AdmissionAction::Resume);
        assert_eq!(admitted.charged_bytes, buffer_snapshot().charged_bytes);
    }

    #[test]
    fn confirmed_part_requires_receipt_then_resets_after_safe_proof() {
        let part = identity(0, 4096);
        let state = seal(
            &TransferState::new(part.clone()).unwrap(),
            4096,
            &part.sha256,
        )
        .unwrap();
        let confirmed = acknowledge(&state, receipt(&part)).unwrap();
        let admitted = admit(Some(&confirmed), part, buffer_snapshot(), 0).unwrap();
        assert_eq!(admitted.action, AdmissionAction::ResetConfirmed);
        assert_eq!(admitted.state.status, TransferStatus::Receiving);
    }

    #[test]
    fn admission_rejects_global_charged_capacity_even_when_payload_is_small() {
        let part = identity(0, 4096);
        let full = BufferSnapshot {
            charged_bytes: BUFFER_BYTES,
            ..buffer_snapshot()
        };
        assert_eq!(admit(None, part, full, 4096), Err(Error::Busy));
    }

    #[test]
    fn wrong_local_digest_and_wrong_state_never_publish_or_acknowledge() {
        let part = identity(0, 100);
        let receiving = TransferState::new(part.clone()).unwrap();
        assert_eq!(seal(&receiving, 99, &part.sha256), Err(Error::Digest));
        assert_eq!(acknowledge(&receiving, receipt(&part)), Err(Error::State));
        let sealed = seal(&receiving, part.bytes, &part.sha256).unwrap();
        let confirmed = acknowledge(&sealed, receipt(&part)).unwrap();
        assert_eq!(acknowledge(&confirmed, receipt(&part)), Err(Error::State));
    }

    #[test]
    fn a_different_digest_cannot_reuse_an_existing_part_slot() {
        let original = identity(0, 100);
        let state = TransferState::new(original).unwrap();
        let mut changed = identity(0, 100);
        changed.sha256 = "c".repeat(64);
        assert_eq!(
            admit(Some(&state), changed, buffer_snapshot(), 0),
            Err(Error::Identity)
        );
    }

    #[test]
    fn recovery_keeps_active_and_sealed_claims_but_cleans_receiving_and_confirmed() {
        let receiving = TransferState::new(identity(0, 100)).unwrap();
        let sealed_part = identity(1, 200);
        let sealed = seal(
            &TransferState::new(sealed_part.clone()).unwrap(),
            sealed_part.bytes,
            &sealed_part.sha256,
        )
        .unwrap();
        let confirmed_part = identity(2, 300);
        let confirmed = acknowledge(
            &seal(
                &TransferState::new(confirmed_part.clone()).unwrap(),
                confirmed_part.bytes,
                &confirmed_part.sha256,
            )
            .unwrap(),
            receipt(&confirmed_part),
        )
        .unwrap();
        let result = snapshot(
            &[
                Claim {
                    state: receiving,
                    active: true,
                },
                Claim {
                    state: sealed,
                    active: false,
                },
                Claim {
                    state: TransferState::new(identity(3, 50)).unwrap(),
                    active: false,
                },
                Claim {
                    state: confirmed,
                    active: false,
                },
            ],
            BUFFER_BYTES,
            4096,
        )
        .unwrap();
        assert_eq!(result.decisions[0].action, RecoveryAction::KeepActive);
        assert_eq!(result.decisions[1].action, RecoveryAction::KeepSealed);
        assert_eq!(result.decisions[2].action, RecoveryAction::RemoveReceiving);
        assert_eq!(result.decisions[3].action, RecoveryAction::RemoveConfirmed);
        assert_eq!(result.accounting.reserved_bytes, 300);
        assert_eq!(result.accounting.pending_bytes, 200);
        assert!(result.accounting.charged_bytes > result.accounting.fixed_overhead_bytes);
    }

    #[test]
    fn recovery_rejects_duplicate_claims_and_unproved_confirmed_cleanup() {
        let state = TransferState::new(identity(0, 100)).unwrap();
        assert_eq!(
            snapshot(
                &[
                    Claim {
                        state: state.clone(),
                        active: true,
                    },
                    Claim {
                        state,
                        active: false,
                    },
                ],
                BUFFER_BYTES,
                4096,
            ),
            Err(Error::State)
        );
        let mut corrupt = TransferState::new(identity(1, 100)).unwrap();
        corrupt.status = TransferStatus::Confirmed;
        assert_eq!(
            snapshot(
                &[Claim {
                    state: corrupt,
                    active: false,
                }],
                BUFFER_BYTES,
                4096,
            ),
            Err(Error::Receipt)
        );
    }
}
