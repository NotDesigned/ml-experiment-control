//! Pure parser for bounded worker input-delivery progress records.
//!
//! The Python shell reads the exact Attempt log and supplies individual lines.
//! This module validates the marker payload and returns observational data; it
//! never treats a log line as execution authority or as proof of training.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const MARKER: &str = "ML_EXPD_INPUT_PROGRESS=";
pub const LEGACY_FAILURE: &str = "ML_EXPD_INPUT_DELIVERY=FAILED";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Phase {
    Initializing,
    WaitingForCache,
    Downloading,
    Extracting,
    Verifying,
    Publishing,
    Mounting,
    Ready,
    Failed,
}

impl Phase {
    fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "INITIALIZING" => Self::Initializing,
            "WAITING_FOR_CACHE" => Self::WaitingForCache,
            "DOWNLOADING" => Self::Downloading,
            "EXTRACTING" => Self::Extracting,
            "VERIFYING" => Self::Verifying,
            "PUBLISHING" => Self::Publishing,
            "MOUNTING" => Self::Mounting,
            "READY" => Self::Ready,
            "FAILED" => Self::Failed,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub at: String,
    pub phase: Phase,
    pub asset_id: String,
    pub received_bytes: Option<f64>,
    pub expected_bytes: Option<f64>,
    pub elapsed_seconds: Option<f64>,
    pub bytes_per_second: Option<f64>,
    pub http_status: Option<f64>,
    pub errno: Option<f64>,
    pub code: Option<String>,
    pub error_class: Option<String>,
    pub failed_phase: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParsedLine {
    pub record: Option<Record>,
    pub legacy_failed: bool,
}

fn object(value: &Value) -> Option<&Map<String, Value>> {
    value.as_object()
}

fn bounded_number(fields: &Map<String, Value>, key: &str) -> Option<f64> {
    let value = fields.get(key)?.as_f64()?;
    (value.is_finite() && (0.0..1e20).contains(&value)).then_some(value)
}

fn bounded_word(fields: &Map<String, Value>, key: &str) -> Option<String> {
    let value = fields.get(key)?.as_str()?;
    if (1..=80).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        Some(value.to_owned())
    } else {
        None
    }
}

fn valid_asset(value: &str) -> bool {
    value == "none"
        || value.strip_prefix("asset.").is_some_and(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

/// Parse one complete log line. Invalid marker payloads are ignored exactly
/// like arbitrary user log text; the legacy failure marker is retained.
pub fn parse_line(line: &str) -> ParsedLine {
    if line == LEGACY_FAILURE {
        return ParsedLine {
            record: None,
            legacy_failed: true,
        };
    }
    let Some(payload) = line.strip_prefix(MARKER) else {
        return ParsedLine {
            record: None,
            legacy_failed: false,
        };
    };
    let Ok(value) = serde_json::from_str::<Value>(payload) else {
        return ParsedLine {
            record: None,
            legacy_failed: false,
        };
    };
    let Some(fields) = object(&value) else {
        return ParsedLine {
            record: None,
            legacy_failed: false,
        };
    };
    let at = fields.get("at").and_then(Value::as_str);
    let phase = fields
        .get("phase")
        .and_then(Value::as_str)
        .and_then(Phase::parse);
    let asset_id = fields.get("asset_id").and_then(Value::as_str);
    let Some((at, phase, asset_id)) = at.zip(phase).zip(asset_id).map(|((a, p), i)| (a, p, i)) else {
        return ParsedLine {
            record: None,
            legacy_failed: false,
        };
    };
    if at.is_empty()
        || at.len() > 256
        || at.chars().any(char::is_control)
        || !valid_asset(asset_id)
    {
        return ParsedLine {
            record: None,
            legacy_failed: false,
        };
    }
    ParsedLine {
        record: Some(Record {
            at: at.to_owned(),
            phase,
            asset_id: asset_id.to_owned(),
            received_bytes: bounded_number(fields, "received_bytes"),
            expected_bytes: bounded_number(fields, "expected_bytes"),
            elapsed_seconds: bounded_number(fields, "elapsed_seconds"),
            bytes_per_second: bounded_number(fields, "bytes_per_second"),
            http_status: bounded_number(fields, "http_status"),
            errno: bounded_number(fields, "errno"),
            code: bounded_word(fields, "code"),
            error_class: bounded_word(fields, "error_class"),
            failed_phase: bounded_word(fields, "failed_phase"),
        }),
        legacy_failed: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_records_keep_bounded_observational_fields() {
        let parsed = parse_line(&format!(
            "{}{{\"at\":\"2026-10-28T10:00:00+00:00\",\"phase\":\"DOWNLOADING\",\"asset_id\":\"asset.{}\",\"received_bytes\":12,\"code\":\"HTTP_200\"}}",
            MARKER,
            "a".repeat(64)
        ));
        let record = parsed.record.unwrap();
        assert_eq!(record.phase, Phase::Downloading);
        assert_eq!(record.received_bytes, Some(12.0));
        assert_eq!(record.code.as_deref(), Some("HTTP_200"));
    }

    #[test]
    fn malformed_or_untrusted_log_text_is_ignored() {
        assert!(parse_line("ordinary user output").record.is_none());
        assert!(parse_line("ML_EXPD_INPUT_PROGRESS={bad").record.is_none());
        assert!(parse_line(&format!(
            "{}{{\"at\":\"now\",\"phase\":\"READY\",\"asset_id\":\"asset.{}\",\"code\":\"bad-value\"}}",
            MARKER,
            "A".repeat(64)
        ))
        .record
        .is_none());
    }

    #[test]
    fn legacy_failure_marker_remains_distinguishable() {
        let parsed = parse_line(LEGACY_FAILURE);
        assert!(parsed.record.is_none());
        assert!(parsed.legacy_failed);
    }
}
