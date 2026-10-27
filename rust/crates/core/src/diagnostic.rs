//! Bounded HTTP diagnostics. Clock, headers, body and credential are explicit
//! inputs; this module never sends a request, sleeps or chooses to retry it.
use regex::Regex;
use serde::Serialize;
use serde_json::{Map, Value};
use std::sync::LazyLock;
use time::{OffsetDateTime, format_description::well_known::Rfc2822};

pub const ERROR_BODY_BYTES: usize = 16 * 1024;
const REDACTED: &str = "[REDACTED]";
static PRIVATE_KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----")
        .expect("constant private-key pattern")
});
static URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\b[a-z][a-z0-9+.-]*://[^\s\"'<>]+"#).expect("constant URL pattern")
});
static BEARER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)\bBearer\s+[^\s,;\"']+"#).expect("constant bearer pattern"));
static ASSIGNMENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)([\w-]*(?:token|secret|password|passwd|proxy|credential|authorization|cookie|(?:api|private|access|client)[_-]?key|signature)[\w-]*[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"#)
        .expect("constant diagnostic assignment pattern")
});
static SENSITIVE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)token|secret|password|credential|authorization|cookie|(?:api|private|access|client)[_-]?key|signature|url|argv|command|environment|^env$|^input$|headers|body|^auth$|^key$")
        .expect("constant diagnostic field pattern")
});

/// HTTP errors add a stricter policy to the shared log redactor: whole URLs,
/// private key blocks, quoted assignments and request-bearing JSON fields are
/// suppressed. These are diagnostics, never scientific metric projections.
pub fn text(value: &str, credential: &str) -> String {
    let value = if credential.is_empty() {
        value.to_owned()
    } else {
        value.replace(credential, REDACTED)
    };
    let value = PRIVATE_KEY.replace_all(&value, REDACTED);
    let value = URL.replace_all(&value, REDACTED);
    let value = BEARER.replace_all(&value, "Bearer [REDACTED]");
    let value = ASSIGNMENT.replace_all(&value, |c: &regex::Captures<'_>| {
        format!("{}{REDACTED}", &c[1])
    });
    experiment_redact::redact_line_with_marker(&value, REDACTED)
        .chars()
        .map(|c| if (c as u32) < 32 { ' ' } else { c })
        .take(1024)
        .collect()
}

pub fn json(value: &Value, credential: &str) -> Value {
    fn visit(value: &Value, credential: &str, depth: usize, budget: &mut usize) -> Value {
        if *budget == 0 {
            return Value::String("[TRUNCATED]".into());
        }
        *budget -= 1;
        if depth > 6 {
            return Value::String("[TRUNCATED]".into());
        }
        match value {
            Value::Object(fields) => {
                let mut clean = Map::new();
                for (key, value) in fields.iter().take(64) {
                    let name = text(key, credential).chars().take(128).collect::<String>();
                    clean.insert(
                        name,
                        if SENSITIVE.is_match(key) || experiment_redact::sensitive_json_key(key) {
                            Value::String(REDACTED.into())
                        } else {
                            visit(value, credential, depth + 1, budget)
                        },
                    );
                }
                Value::Object(clean)
            }
            Value::Array(values) => Value::Array(
                values
                    .iter()
                    .take(64)
                    .map(|v| visit(v, credential, depth + 1, budget))
                    .collect(),
            ),
            Value::String(value) => Value::String(text(value, credential)),
            Value::Number(number)
                if !credential.is_empty() && number.to_string().contains(credential) =>
            {
                Value::String(REDACTED.into())
            }
            Value::Number(number)
                if number.to_string().contains(['.', 'e', 'E'])
                    && !number.as_f64().is_some_and(f64::is_finite) =>
            {
                Value::Null
            }
            value => value.clone(),
        }
    }
    visit(value, credential, 0, &mut 256)
}

/// Retry-After is an observation only. It never schedules a replay. Non-ASCII
/// and malformed headers are refused rather than interpreting locale digits.
pub fn retry_after(value: Option<&str>, now_unix: i64) -> Option<u32> {
    let value = value.filter(|v| !v.is_empty() && v.len() <= 128 && v.is_ascii())?;
    if value.bytes().all(|c| c.is_ascii_digit()) {
        let seconds = value.bytes().fold(0u32, |n, c| {
            n.saturating_mul(10).saturating_add(u32::from(c - b'0'))
        });
        return Some(seconds.min(86400));
    }
    let instant = OffsetDateTime::parse(value, &Rfc2822).ok()?;
    Some(
        instant
            .unix_timestamp()
            .saturating_sub(now_unix)
            .clamp(0, 86400) as u32,
    )
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HttpFailure {
    pub status: u16,
    pub retryable: bool,
    pub error_code: String,
    pub retry_after: Option<u32>,
    pub details: Option<Value>,
    pub response_body: Option<Value>,
    pub body_truncated: bool,
}
impl HttpFailure {
    /// Only complete, bounded, valid JSON is retained. HTML, malformed JSON,
    /// nonstandard JSON constants and unsupported Unicode are suppressed.
    pub fn from_response(
        status: u16,
        code: Option<&str>,
        retry_header: Option<&str>,
        body: &[u8],
        credential: &str,
        now_unix: i64,
    ) -> Self {
        let code = code
            .filter(|s| {
                !s.is_empty()
                    && s.len() <= 128
                    && s.as_bytes()[0].is_ascii_uppercase()
                    && s.bytes()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_')
                    && (credential.is_empty() || !s.contains(credential))
            })
            .unwrap_or("API_ERROR");
        let body_truncated = body.len() > ERROR_BODY_BYTES;
        let response_body = (!body_truncated)
            .then(|| serde_json::from_slice::<Value>(body).ok())
            .flatten()
            .map(|value| json(&value, credential));
        let details = response_body
            .as_ref()
            .map(|v| v.get("detail").unwrap_or(v).clone());
        Self {
            status,
            retryable: false,
            error_code: code.into(),
            retry_after: retry_after(retry_header, now_unix),
            details,
            response_body,
            body_truncated,
        }
    }
}
impl std::fmt::Display for HttpFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HTTP {} {}; inspect saved identities before retrying",
            self.status, self.error_code
        )
    }
}
