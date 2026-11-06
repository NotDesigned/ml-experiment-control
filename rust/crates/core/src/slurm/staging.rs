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
    pub worker: super::managed::Runtime,
}
impl Requirement {
    pub fn validate(&self) -> helper::Result<()> {
        self.worker.validate()?;
        validate_runs_root(&self.runs_root)
    }
}
/// The operator's shared root, checked on its own before any Attempt names it.
pub fn validate_runs_root(runs_root: &str) -> helper::Result<()> {
    if !absolute_path(runs_root)
        || runs_root == "/"
        || runs_root.ends_with('/')
        || runs_root.contains("//")
    {
        return Err(helper::Error::Metadata);
    }
    Ok(())
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
    pub managed: super::managed::Package,
}
fn decode<T: serde::de::DeserializeOwned>(value: &Value) -> helper::Result<T> {
    serde_json::from_value(value.clone()).map_err(|_| helper::Error::Metadata)
}
/// Only the immutable Attempt may select its scoped directory. A shared-root
/// capability is independent operator policy and cannot be supplied by a client.
pub fn admit(intent: &helper::FrozenIntent, manifest: &[u8]) -> helper::Result<Plan> {
    intent.validate()?;
    let requirement = &intent.staging;
    if helper::sha256_bytes(manifest) != intent.manifest_sha256 {
        return Err(helper::Error::Content);
    }
    let doc = Value::Object(crate::json::mapping(manifest).map_err(|_| helper::Error::Metadata)?);
    if doc["schema_version"] != 1
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
    let datasets = crate::datasets::document_mounts(
        &doc,
        &intent.scope.project,
        &intent.target,
        &spec.storage.project_data_root,
    )
    .map_err(|_| helper::Error::Identity)?;
    let run_dir = format!(
        "{}/{}/{}",
        requirement.runs_root, intent.scope.project, intent.scope.run_id
    );
    if spec.storage.run_dir != run_dir {
        return Err(helper::Error::Identity);
    }
    match (
        doc.get("resume_from").filter(|v| !v.is_null()),
        doc.get("native_checkpoint"),
    ) {
        (Some(reference), Some(value)) => {
            let reference: crate::checkpoints::Reference = decode(reference)?;
            crate::checkpoints::selected(
                &reference,
                value,
                &intent.scope.project,
                &intent.target.storage_scope,
                &requirement.runs_root,
            )
            .map_err(|_| helper::Error::Identity)?;
            if reference.run_id == intent.scope.run_id {
                let receipt: crate::first_submission::native::retry::Receipt =
                    decode(&doc["native_retry"])?;
                receipt
                    .validate(&doc, &intent.scope.attempt_id)
                    .map_err(|_| helper::Error::Identity)?;
                if receipt.request.resume_from.as_ref() != Some(&reference) {
                    return Err(helper::Error::Identity);
                }
            }
        }
        (None, None) => {}
        _ => return Err(helper::Error::Identity),
    }
    Ok(Plan {
        contract: CONTRACT.into(),
        scope: intent.scope.clone(),
        intent_sha256: intent.digest()?,
        manifest_sha256: intent.manifest_sha256.clone(),
        runs_root: requirement.runs_root.clone(),
        attempt_dir: format!("{run_dir}/attempts/{}", intent.scope.attempt_id),
        run_dir,
        managed: super::managed::plan_for_document(
            &intent.scope,
            &spec,
            &requirement.worker,
            &doc,
            &datasets,
        )?,
    })
}

/// What `plan` hands back: the managed package and nothing else.
#[doc(hidden)]
pub struct Admitted {
    pub managed: Option<super::managed::Package>,
}
/// `admit` in the shape `backends/slurm/tests/submission_tests.rs` reads. That
/// file is pinned by the no-replay guard and calls this as
/// `plan(..).unwrap().unwrap().managed.unwrap()`, so the name and the two options
/// stay until its owner allows the edit. Production code calls `admit`.
#[doc(hidden)]
pub fn plan(intent: &helper::FrozenIntent, manifest: &[u8]) -> helper::Result<Option<Admitted>> {
    Ok(Some(Admitted {
        managed: Some(admit(intent, manifest)?.managed),
    }))
}

pub fn checkpoint_context(
    intent: &helper::FrozenIntent,
    manifest: &[u8],
) -> helper::Result<crate::checkpoints::Context> {
    let plan = admit(intent, manifest)?;
    let doc = Value::Object(crate::json::mapping(manifest).map_err(|_| helper::Error::Metadata)?);
    let context = crate::checkpoints::Context {
        project: intent.scope.project.clone(),
        run_id: intent.scope.run_id.clone(),
        attempt_id: intent.scope.attempt_id.clone(),
        source_id: doc["source_id"]
            .as_str()
            .ok_or(helper::Error::Metadata)?
            .into(),
        image_id: doc["image_id"]
            .as_str()
            .ok_or(helper::Error::Metadata)?
            .into(),
        storage_scope: intent.target.storage_scope.clone(),
        state_root: format!("{}/state", plan.attempt_dir),
    };
    Ok(context)
}
