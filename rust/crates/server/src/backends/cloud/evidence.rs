//! Pure, bounded provider-error projection; input credentials are only used for redaction.
use super::Credentials;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Selected response evidence only. Headers, request bodies and arbitrary
/// provider documents are never persisted with a debug-machine failure.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureEvidence {
    pub code: String,
    pub http_status: u16,
    pub message: Option<String>,
    pub request_id: Option<String>,
}

pub(super) fn failure_evidence(
    response: &Value,
    code: &str,
    status: u16,
    credentials: &Credentials,
    request_body: Option<&Value>,
) -> FailureEvidence {
    let (id, secret, token) = match credentials {
        Credentials::Pai {
            access_key_id,
            access_key_secret,
            security_token,
        } => (access_key_id, access_key_secret, security_token.as_deref()),
        Credentials::Sensecore {
            access_key_id,
            access_key_secret,
        } => (access_key_id, access_key_secret, None),
    };
    fn hide_request_secrets(text: &mut String, value: &Value, depth: usize) {
        if depth > 16 {
            return;
        }
        match value {
            Value::Object(fields) => {
                for (key, value) in fields {
                    if experiment_redact::sensitive_json_key(key) {
                        if let Some(secret) = value.as_str().filter(|s| !s.is_empty()) {
                            *text = text.replace(secret, "[REDACTED]");
                        }
                    } else {
                        hide_request_secrets(text, value, depth + 1);
                    }
                }
            }
            Value::Array(values) => {
                for value in values {
                    hide_request_secrets(text, value, depth + 1);
                }
            }
            _ => {}
        }
    }
    let scrub = |raw: &str| {
        // Remove exact credentials before bounding the final text, so a
        // truncation boundary cannot retain the beginning of a secret.
        let mut text = raw.to_owned();
        for credential in [id.as_str(), secret.as_str()].into_iter().chain(token) {
            if !credential.is_empty() {
                text = text.replace(credential, "[REDACTED]");
            }
        }
        if let Some(body) = request_body {
            hide_request_secrets(&mut text, body, 0);
        }
        ml_exp_core::diagnostic::text(&text, "")
    };
    let message = response["Message"]
        .as_str()
        .or_else(|| response["message"].as_str())
        .filter(|s| s.len() <= 16384)
        .map(scrub);
    let request_id = response["RequestId"]
        .as_str()
        .or_else(|| response["request_id"].as_str())
        .filter(|s| {
            !s.is_empty()
                && s.len() <= 128
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
        })
        .filter(|s| scrub(s) == *s)
        .map(str::to_owned);
    FailureEvidence {
        code: if scrub(code) == code {
            code.into()
        } else {
            "CLOUD_REQUEST_FAILED".into()
        },
        http_status: status,
        message,
        request_id,
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn response_evidence_is_bounded_and_scrubs_known_credentials_and_request_secrets() {
        let credentials = Credentials::Pai {
            access_key_id: "known-id".into(),
            access_key_secret: "known-secret".into(),
            security_token: Some("known-token".into()),
        };
        let evidence = failure_evidence(
            &json!({"Message":"invalid role; known-id known-secret known-token image-password https://private.test/?signature=secret\n", "RequestId":"request-123", "Password":"never-copy-this"}),
            "ValidationError",
            400,
            &credentials,
            Some(&json!({"ImageConfig":{"Password":"image-password"}})),
        );
        let encoded = serde_json::to_string(&evidence).unwrap();
        for sensitive in [
            "known-id",
            "known-secret",
            "known-token",
            "image-password",
            "private.test",
            "never-copy-this",
        ] {
            assert!(!encoded.contains(sensitive), "{sensitive}");
        }
        assert!(evidence.message.unwrap().starts_with("invalid role;"));
        assert_eq!(evidence.request_id.as_deref(), Some("request-123"));
        let evidence = failure_evidence(
            &json!({"Message":"a".repeat(2000),"RequestId":"invalid/id"}),
            "ValidationError",
            400,
            &credentials,
            None,
        );
        assert_eq!(evidence.message.unwrap().len(), 1024);
        assert!(evidence.request_id.is_none());
        let evidence = failure_evidence(
            &json!({"Message":"a".repeat(16385)}),
            "ValidationError",
            400,
            &credentials,
            None,
        );
        assert!(evidence.message.is_none());
        let evidence = failure_evidence(
            &json!({"Message":[],"RequestId":"known-secret"}),
            "known-id",
            400,
            &credentials,
            None,
        );
        assert!(evidence.message.is_none() && evidence.request_id.is_none());
        assert_eq!(evidence.code, "CLOUD_REQUEST_FAILED");
    }
}
