//! Pure shared-storage admission for a frozen Slurm Attempt. No pathname is
//! opened and no directory is created here; the helper supplies those effects.
use super::{ExecutionConfig, JobSpec, Resources, SlurmConfig, Storage, absolute_path, helper};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const CONTRACT: &str = "slurm-shared-attempt.v1";
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requirement {
    pub runs_root: String,
}
impl Requirement {
    pub fn validate(&self) -> helper::Result<()> {
        if !absolute_path(&self.runs_root)
            || self.runs_root == "/"
            || self.runs_root.ends_with('/')
            || self.runs_root.contains("//")
        {
            return Err(helper::Error::Metadata);
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub contract: String,
    pub scope: helper::Scope,
    pub intent_sha256: String,
    pub manifest_sha256: String,
    pub runs_root: String,
    pub run_dir: String,
    pub attempt_dir: String,
}
fn decode<T: serde::de::DeserializeOwned>(value: &Value) -> helper::Result<T> {
    serde_json::from_value(value.clone()).map_err(|_| helper::Error::Metadata)
}
/// Only the immutable Attempt may select its scoped directory. A shared-root
/// capability is independent operator policy and cannot be supplied by a client.
pub fn plan(intent: &helper::FrozenIntent, manifest: &[u8]) -> helper::Result<Option<Plan>> {
    intent.validate()?;
    let Some(requirement) = &intent.staging else {
        return Ok(None);
    };
    if helper::sha256_bytes(manifest) != intent.manifest_sha256 {
        return Err(helper::Error::Content);
    }
    let doc = Value::Object(crate::json::mapping(manifest).map_err(|_| helper::Error::Metadata)?);
    if doc["schema_version"] != 1
        || doc["identity_version"] != 2
        || doc["project"] != intent.scope.project
        || doc["run_id"] != intent.scope.run_id
        || doc["attempt_id"] != intent.scope.attempt_id
    {
        return Err(helper::Error::Identity);
    }
    let spec = JobSpec {
        run_id: intent.scope.run_id.clone(),
        attempt_id: intent.scope.attempt_id.clone(),
        image_id: doc["image_id"]
            .as_str()
            .ok_or(helper::Error::Metadata)?
            .into(),
        backend: decode::<SlurmConfig>(&doc["backend"])?,
        storage: decode::<Storage>(&doc["storage"])?,
        resources: decode::<Resources>(&doc["resources"])?,
        execution: decode::<ExecutionConfig>(&doc["execution"])?,
        command: decode(&doc["command"])?,
    };
    spec.validate().map_err(|_| helper::Error::Metadata)?;
    let run_dir = format!(
        "{}/{}/{}",
        requirement.runs_root, intent.scope.project, intent.scope.run_id
    );
    if spec.storage.run_dir != run_dir {
        return Err(helper::Error::Identity);
    }
    Ok(Some(Plan {
        contract: CONTRACT.into(),
        scope: intent.scope.clone(),
        intent_sha256: intent.digest()?,
        manifest_sha256: intent.manifest_sha256.clone(),
        runs_root: requirement.runs_root.clone(),
        attempt_dir: format!("{run_dir}/attempts/{}", intent.scope.attempt_id),
        run_dir,
    }))
}
