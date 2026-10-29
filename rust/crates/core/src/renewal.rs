//! Explicit replacement of an expired, never-started submission gate. The
//! scientific identity and scheduler token stay unchanged; authorization does not.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub request_id: String,
    pub confirmation: String,
}
impl Request {
    pub fn validate(&self, action: &str) -> Result<(), &'static str> {
        crate::operations::validate_submission_id(action).map_err(|_| "renewal action")?;
        crate::submission::validate_submission_token(&self.request_id)
            .map_err(|_| "renewal request")?;
        if self.confirmation != format!("RENEW {action}") {
            return Err("renewal confirmation");
        }
        Ok(())
    }
    pub fn record(&self, action: &str) -> Result<String, &'static str> {
        self.validate(action)?;
        Ok(format!(
            "renewal-{}.json",
            crate::canonical::digest(&serde_json::json!(["native-gate-renewal.v1", action, self]))
                .map_err(|_| "renewal identity")?
        ))
    }
}
pub fn archive(action: &str) -> Result<String, &'static str> {
    crate::operations::validate_submission_id(action).map_err(|_| "renewal action")?;
    Ok(format!("prepared-{action}.json"))
}
pub fn eligible(snapshot: &Value, now: &str) -> Result<(), &'static str> {
    let at = OffsetDateTime::parse(now, &Rfc3339).map_err(|_| "renewal clock")?;
    let expires = snapshot["gate_expires_at"]
        .as_str()
        .and_then(|s| OffsetDateTime::parse(s, &Rfc3339).ok())
        .ok_or("renewal expiry")?;
    if snapshot["operation"] != "SUBMIT_RUN"
        || at < expires
        || !matches!(
            snapshot["execution"]["status"].as_str(),
            Some("PREPARED" | "AUTHORIZED")
        )
        || !snapshot["execution"]["started_at"].is_null()
        || !snapshot["execution"]["result"].is_null()
        || !snapshot["cancellation"].is_null()
    {
        return Err("submission may have started or gate has not expired");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn renewal_never_grants_authority_over_a_possible_execution() {
        let original = json!({"operation":"SUBMIT_RUN","gate_expires_at":"2026-10-29T00:30:00Z","execution":{"status":"AUTHORIZED","started_at":null,"result":null}});
        assert!(eligible(&original, "2026-10-29T00:30:00Z").is_ok());
        assert!(eligible(&original, "2026-10-29T00:29:59Z").is_err());
        for state in [
            "EXECUTING",
            "RECONCILE_REQUIRED",
            "VERIFIED",
            "BLOCKED",
            "UNKNOWN",
        ] {
            let mut v = original.clone();
            v["execution"]["status"] = json!(state);
            assert!(eligible(&v, "2026-10-29T01:00:00Z").is_err());
        }
        for key in ["started_at", "result"] {
            let mut v = original.clone();
            v["execution"][key] = json!("present");
            assert!(eligible(&v, "2026-10-29T01:00:00Z").is_err());
        }
        let request = Request {
            request_id: "a".repeat(32),
            confirmation: "RENEW action-0123456789abcdef".into(),
        };
        assert!(request.record("action-0123456789abcdef").is_ok());
        assert!(request.record("action-1123456789abcdef").is_err());
    }
}
