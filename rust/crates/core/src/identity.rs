use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
#[error("invalid {0} identity")]
pub struct IdentityError(&'static str);

pub fn validate_id(value: &str) -> Result<&str, IdentityError> {
    let first = value.as_bytes().first().copied();
    if value.len() > 128
        || !first.is_some_and(|c| c.is_ascii_alphanumeric())
        || !value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
    {
        return Err(IdentityError("resource"));
    }
    Ok(value)
}

pub fn validate_attempt(value: &str) -> Result<&str, IdentityError> {
    let suffix = value
        .strip_prefix("attempt-")
        .ok_or(IdentityError("attempt"))?;
    if suffix.len() < 3 || !suffix.bytes().all(|c| c.is_ascii_digit()) {
        return Err(IdentityError("attempt"));
    }
    Ok(value)
}

/// The two daemon-generated companion identities are readable and fenced with
/// their original parent ID, even when the suffix takes them over 128 bytes.
pub fn validate_observation_id(value: &str) -> Result<&str, IdentityError> {
    if validate_id(value).is_ok()
        || ["--preparation", "--backfill"].iter().any(|suffix| {
            value
                .strip_suffix(suffix)
                .is_some_and(|parent| validate_id(parent).is_ok())
        })
    {
        Ok(value)
    } else {
        Err(IdentityError("observation"))
    }
}

pub fn submission_marker(token: &str) -> Result<String, IdentityError> {
    if token.len() != 32
        || !token
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(IdentityError("submission token"));
    }
    Ok(format!("ml-exp-{token}"))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityReport {
    pub available: bool,
    pub ambiguous: bool,
    #[serde(default)]
    pub scheduler_job_ids: Vec<String>,
    pub remote_manifest_exists: Option<bool>,
    pub remote_manifest_matches: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reject_path_injection_and_unbound_attempts() {
        for value in ["", "../run", "a/b", "a\\b", "a\n", ".hidden", "模型"] {
            assert!(validate_id(value).is_err());
        }
        assert!(validate_id("a-._99").is_ok());
        assert!(validate_attempt("attempt-001").is_ok());
        for value in ["latest", "attempt-01", "attempt-001/..", "attempt-00x"] {
            assert!(validate_attempt(value).is_err());
        }
        assert!(submission_marker("0123456789abcdef0123456789abcdef").is_ok());
        assert!(submission_marker("0123456789ABCDEF0123456789ABCDEF").is_err());
    }
}
