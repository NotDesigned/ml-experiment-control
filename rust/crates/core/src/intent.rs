//! Pure validation for client-authored operation intents.
//!
//! An intent is descriptive input to Action preparation. It is not an
//! authorization, scheduler request or scientific conclusion.
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Kind {
    CreateCampaignDraft,
    UpdateCampaignDraft,
    DeriveRunDraft,
    RunEvaluation,
    RetryAttempt,
    SubmitRun,
    CancelRun,
    ArchiveCampaign,
    ArchiveRun,
    ArchiveAttempt,
    RebuildLocalEvidence,
    DeleteRunResources,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationIntent {
    pub kind: Kind,
    pub title: String,
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub change_summary: String,
    #[serde(default = "default_resource_estimate")]
    pub resource_estimate: String,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub risk: String,
    pub draft: String,
    #[serde(default)]
    pub evidence_digest: String,
    #[serde(default)]
    pub idempotency_key: Option<String>,
}

fn default_resource_estimate() -> String {
    "unknown".into()
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("intent field is missing or exceeds its limit: {0}")]
    Field(&'static str),
    #[error("intent idempotency key has an invalid format")]
    IdempotencyKey,
    #[error("intent must be a JSON object with the known schema")]
    Schema,
}
pub type Result<T> = std::result::Result<T, Error>;

fn bounded(value: &str, min: usize, max: usize, field: &'static str) -> Result<()> {
    let length = value.chars().count();
    if length < min || length > max || value.chars().any(char::is_control) {
        return Err(Error::Field(field));
    }
    Ok(())
}

fn valid_key(value: &str) -> bool {
    let mut chars = value.bytes();
    chars
        .next()
        .is_some_and(|byte| byte.is_ascii_alphanumeric())
        && chars.all(|byte| byte.is_ascii_alphanumeric() || b"_.:-".contains(&byte))
}

impl OperationIntent {
    pub fn validate(&self) -> Result<()> {
        bounded(&self.title, 1, 1000, "title")?;
        bounded(&self.target, 0, 2000, "target")?;
        bounded(&self.change_summary, 0, 8000, "change_summary")?;
        bounded(&self.resource_estimate, 0, 2000, "resource_estimate")?;
        bounded(&self.rationale, 0, 8000, "rationale")?;
        bounded(&self.risk, 0, 8000, "risk")?;
        bounded(&self.draft, 0, 200_000, "draft")?;
        bounded(&self.evidence_digest, 0, 200, "evidence_digest")?;
        if let Some(key) = &self.idempotency_key {
            if !(1..=200).contains(&key.len()) || !valid_key(key) {
                return Err(Error::IdempotencyKey);
            }
        }
        Ok(())
    }

    pub fn from_value(value: &Value) -> Result<Self> {
        let intent: Self = serde_json::from_value(value.clone()).map_err(|_| Error::Schema)?;
        intent.validate()?;
        Ok(intent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn value() -> Value {
        json!({
            "kind":"SUBMIT_RUN",
            "title":"Launch trial",
            "draft":"{}"
        })
    }

    #[test]
    fn defaults_and_known_fields_match_the_client_contract() {
        let intent = OperationIntent::from_value(&value()).unwrap();
        assert_eq!(intent.resource_estimate, "unknown");
        assert!(intent.target.is_empty());
        assert!(intent.idempotency_key.is_none());
    }

    #[test]
    fn bounds_and_idempotency_are_fail_closed() {
        let mut bad = value();
        bad["title"] = json!("");
        assert_eq!(OperationIntent::from_value(&bad), Err(Error::Field("title")));
        let mut bad_key = value();
        bad_key["idempotency_key"] = json!("bad key");
        assert_eq!(
            OperationIntent::from_value(&bad_key),
            Err(Error::IdempotencyKey)
        );
        let mut unknown = value();
        unknown["extra"] = json!(true);
        assert_eq!(OperationIntent::from_value(&unknown), Err(Error::Schema));
    }
}
