//! A worker's job is finished before these calls, and a client's upload may have run
//! for hours: one busy reply or a restarting server must not lose that. Each call
//! retried with this is safe to repeat. Pure: the caller does the sleeping.
use std::time::Duration;

pub const ATTEMPTS: u32 = 12;

pub fn delay(attempt: u32) -> Duration {
    Duration::from_secs((1u64 << attempt.min(6)).min(60))
}

pub fn may_pass(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_a_passing_failure_is_tried_again() {
        for status in [408, 429, 500, 502, 503, 504] {
            assert!(may_pass(status), "{status}");
        }
        for status in [400, 401, 403, 404, 409, 410, 422] {
            assert!(!may_pass(status), "{status}");
        }
    }
    #[test]
    fn the_wait_grows_and_stays_bounded() {
        let waits: Vec<u64> = (1..ATTEMPTS).map(|n| delay(n).as_secs()).collect();
        assert_eq!(waits[0], 2);
        assert!(waits.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(waits.iter().all(|wait| *wait <= 60));
        assert!(waits.iter().sum::<u64>() >= 300, "covers a long-held lock");
    }
}
