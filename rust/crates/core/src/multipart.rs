//! Part arithmetic and part records for resumable archive uploads.
use serde::{Deserialize, Serialize};

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
    #[error("multipart upload exceeds its configured limit")]
    Limit,
    #[error("invalid multipart upload part number")]
    PartNumber,
}
pub type Result<T> = std::result::Result<T, Error>;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartRecord {
    pub sha256: String,
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_accepts_reference_bounds_and_rejects_outside() {
        assert!(Policy::default().validate().is_ok());
        for policy in [
            Policy {
                part_bytes: 1023,
                ..Policy::default()
            },
            Policy {
                part_bytes: 64 * 1024 * 1024 + 1,
                ..Policy::default()
            },
            Policy {
                session_seconds: 299,
                ..Policy::default()
            },
            Policy {
                max_parts: 65_537,
                ..Policy::default()
            },
        ] {
            assert!(policy.validate().is_err());
        }
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
        assert_eq!(part_count(0, policy), Err(Error::Limit));
    }
}
