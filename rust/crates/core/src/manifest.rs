//! Scientific Run identity construction. Attempt resume is deliberately excluded.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifestInput {
    pub project: String,
    pub run_id: String,
    pub created_at: String,
    pub config_path: String,
    pub resolved_config: Map<String, Value>,
    pub source_id: String,
    pub runtime_tree_id: String,
    pub git_commit: Option<String>,
    pub campaign_id: Option<String>,
    pub campaign: Option<String>,
    pub image_id: String,
    // Kept in the input contract for existing callers; not scientific identity.
    pub run_dir: String,
    pub max_infra_retries: i64,
    pub backend: Map<String, Value>,
    pub resources: Map<String, Value>,
    pub storage: Map<String, Value>,
    pub command: Vec<String>,
    pub execution: Map<String, Value>,
    pub config_overrides: Option<Vec<String>>,
    pub assets: Option<Vec<Value>>,
    pub checkpoint: Option<Map<String, Value>>,
    pub evaluation: Option<Map<String, Value>>,
    pub research_contract: Option<Value>,
    pub research_role: Option<String>,
}

pub fn build_run_manifest(input: &RunManifestInput) -> Value {
    let scientific: Map<String, Value> = input
        .resolved_config
        .iter()
        .filter(|(key, _)| key.as_str() != "resume")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let seed = scientific.get("seed").cloned().unwrap_or(Value::Null);
    let mut manifest = json!({
        "schema_version":1,"identity_version":2,
        "project":input.project,"run_id":input.run_id,"created_at":input.created_at,
        "config_path":input.config_path,"resolved_config":scientific,
        "source_id":input.source_id,"runtime_tree_id":input.runtime_tree_id,
        "git_commit":input.git_commit,"campaign_id":input.campaign_id,"campaign":input.campaign,
        "image_id":input.image_id,"seed":seed,"backend":input.backend,
        "resources":input.resources,"storage":input.storage,"command":input.command,
        "execution":input.execution,"config_overrides":input.config_overrides.as_deref().unwrap_or(&[]),
        "assets":input.assets.as_deref().unwrap_or(&[]),
        "checkpoint":input.checkpoint.clone().unwrap_or_default(),
        "evaluation":input.evaluation.clone().unwrap_or_default(),
        "resume_policy":{"enabled":true,"max_infra_retries":input.max_infra_retries}
    });
    if let Some(contract) = &input.research_contract {
        manifest["research_contract"] = contract.clone();
        manifest["research_role"] = json!(input.research_role);
    }
    manifest
}

pub fn comparable_manifest(manifest: &Map<String, Value>) -> Value {
    let mut result: Map<String, Value> = manifest
        .iter()
        .filter(|(key, _)| key.as_str() != "created_at")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let Some(Value::Object(resolved)) = result.get_mut("resolved_config") {
        resolved.shift_remove("resume");
    }
    Value::Object(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_only_removes_attempt_operations_and_keeps_unknown_scientific_fields() {
        let manifest = json!({"created_at":"old","resolved_config":{"resume":"step-10","seed":42},"future_science":{"protocol":2}});
        let expected = json!({"resolved_config":{"seed":42},"future_science":{"protocol":2}});
        assert_eq!(comparable_manifest(manifest.as_object().unwrap()), expected);
        assert!(manifest["resolved_config"].get("resume").is_some());
    }
}
