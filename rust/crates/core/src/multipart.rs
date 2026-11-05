//! Part arithmetic and part records for resumable archive uploads.
use serde::{Deserialize, Serialize};

/// Every archive part is this size except a shorter final one.
pub const PART_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_PARTS: u64 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("multipart upload exceeds its limit")]
    Limit,
    #[error("invalid multipart upload part number")]
    PartNumber,
}
pub type Result<T> = std::result::Result<T, Error>;

/// Return the number of parts for one archive and reject a size that cannot
/// be represented in `MAX_PARTS`.
pub fn part_count(size: u64) -> Result<u64> {
    if size == 0 {
        return Err(Error::Limit);
    }
    let count = size.div_ceil(PART_BYTES);
    if count > MAX_PARTS {
        return Err(Error::Limit);
    }
    Ok(count)
}

/// Return the exact expected bytes for a zero-based part number.
pub fn part_size(size: u64, number: u64) -> Result<u64> {
    let count = part_count(size)?;
    if number >= count {
        return Err(Error::PartNumber);
    }
    Ok(std::cmp::min(PART_BYTES, size - number * PART_BYTES))
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
    fn part_arithmetic_uses_zero_based_final_short_part() {
        let size = 2 * PART_BYTES + 2;
        assert_eq!(part_count(2 * PART_BYTES).unwrap(), 2);
        assert_eq!(part_size(size, 0).unwrap(), PART_BYTES);
        assert_eq!(part_size(size, 1).unwrap(), PART_BYTES);
        assert_eq!(part_size(size, 2).unwrap(), 2);
        assert_eq!(part_size(size, 3), Err(Error::PartNumber));
        assert_eq!(part_count(0), Err(Error::Limit));
        assert_eq!(part_count(MAX_PARTS * PART_BYTES).unwrap(), MAX_PARTS);
        assert_eq!(part_count(MAX_PARTS * PART_BYTES + 1), Err(Error::Limit));
    }
}
