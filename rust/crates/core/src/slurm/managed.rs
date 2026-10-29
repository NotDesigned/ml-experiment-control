//! Sealed native-worker launch packages for an already-frozen Slurm Attempt.
//! The helper stages these bytes; only the worker inside the image opens the
//! container workdir and creates its fresh process/control directories.
use super::{JobSpec, helper};
use crate::{launch, managed};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub const CONFIG_FILE: &str = "managed-config.json";
pub const MANIFEST_FILE: &str = "launch-manifest.json";

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    /// Installed in the immutable SIF/OCI image by the trusted builder.
    pub program: String,
    #[serde(default)]
    pub base_environment: BTreeMap<String, String>,
    #[serde(default)]
    pub runtime_environment: Vec<String>,
}
impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedRuntime").finish_non_exhaustive()
    }
}
impl Runtime {
    pub fn validate(&self) -> helper::Result<()> {
        if !managed::absolute(Path::new(&self.program))
            || Path::new(&self.program).file_name().is_none()
            || !managed::environment_policy(&self.base_environment, &self.runtime_environment)
        {
            return Err(helper::Error::Metadata);
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Package {
    pub config: managed::Config,
    pub manifest: launch::Manifest,
}
impl std::fmt::Debug for Package {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedLaunchPackage")
            .finish_non_exhaustive()
    }
}
impl Package {
    pub fn config_bytes(&self) -> helper::Result<Vec<u8>> {
        self.config
            .validate()
            .map_err(|_| helper::Error::Metadata)?;
        serde_json::to_vec(&self.config).map_err(|_| helper::Error::Metadata)
    }
    pub fn manifest_bytes(&self) -> helper::Result<Vec<u8>> {
        self.manifest.encode().map_err(|_| helper::Error::Metadata)
    }
}
pub fn plan(scope: &helper::Scope, spec: &JobSpec, runtime: &Runtime) -> helper::Result<Package> {
    plan_restoring(scope, spec, runtime, None)
}
pub fn plan_restoring(
    scope: &helper::Scope,
    spec: &JobSpec,
    runtime: &Runtime,
    restore: Option<&serde_json::Value>,
) -> helper::Result<Package> {
    plan_with_inputs(scope, spec, runtime, restore, &[])
}
pub fn plan_with_inputs(
    scope: &helper::Scope,
    spec: &JobSpec,
    runtime: &Runtime,
    restore: Option<&serde_json::Value>,
    datasets: &[crate::datasets::Mount],
) -> helper::Result<Package> {
    scope.validate()?;
    spec.validate().map_err(|_| helper::Error::Metadata)?;
    runtime.validate()?;
    if scope.run_id != spec.run_id
        || scope.attempt_id != spec.attempt_id
        || !spec.execution.managed_io
    {
        return Err(helper::Error::Identity);
    }
    let root = format!("{}/attempts/{}", spec.storage.run_dir, scope.attempt_id);
    // The existing outputs directory belongs to shared staging. A fresh child
    // preserves managed execution's no-adoption/no-replay boundary.
    let output = format!("{root}/outputs/payload");
    let timeout =
        super::walltime_seconds(&spec.backend.time).map_err(|_| helper::Error::Metadata)?;
    let mut manifest = launch::Manifest {
        contract: launch::CONTRACT.into(),
        identity: launch::Identity {
            project: scope.project.clone(),
            run_id: scope.run_id.clone(),
            attempt_id: scope.attempt_id.clone(),
        },
        environment: BTreeMap::from([
            ("OUTPUT_DIR".into(), output.clone()),
            ("INPUT_DIR".into(), format!("{root}/inputs")),
            ("DATA_DIR".into(), spec.storage.project_data_root.clone()),
        ]),
        argv: spec.command.clone(),
        workdir: spec.execution.workdir.clone(),
        timeout_seconds: timeout,
    };
    manifest.environment.extend(spec.execution.env.clone());
    if let Some(value) = restore {
        let plan =
            crate::checkpoints::restore_plan(value, None).map_err(|_| helper::Error::Metadata)?;
        manifest.environment.extend([
            (
                "RESUME_DIR".into(),
                format!("{}/checkpoints/{}", plan.state_root, plan.generation),
            ),
            ("RESUME_CHECKPOINT_ID".into(), plan.checkpoint_id),
            ("RESUME_STEP".into(), plan.step.to_string()),
        ]);
    }
    let bytes = manifest.encode().map_err(|_| helper::Error::Metadata)?;
    if !datasets.is_empty() {
        crate::datasets::mounts(datasets).map_err(|_| helper::Error::Metadata)?;
        let mapping: BTreeMap<_, _> = datasets
            .iter()
            .map(|v| (v.name.clone(), format!("{root}/inputs/{}", v.name)))
            .collect();
        manifest.environment.insert(
            "DATASETS_JSON".into(),
            serde_json::to_string(&mapping).map_err(|_| helper::Error::Metadata)?,
        );
    }
    let bytes = if datasets.is_empty() {
        bytes
    } else {
        manifest.encode().map_err(|_| helper::Error::Metadata)?
    };
    let config = managed::Config {
        manifest_path: PathBuf::from(format!("{root}/{MANIFEST_FILE}")),
        endpoint: String::new(),
        attempt_root: PathBuf::from(format!("{root}/worker")),
        workdir: PathBuf::from(&spec.execution.workdir),
        maximum_timeout_seconds: timeout,
        base_environment: runtime.base_environment.clone(),
        local_binding: Some(launch::Binding {
            identity: manifest.identity.clone(),
            sha256: helper::sha256_bytes(&bytes)[7..].into(),
        }),
        output_directory: Some(PathBuf::from(output)),
        runtime_environment: runtime.runtime_environment.clone(),
        // Frozen Attempt encoding sorts nested mapping keys. Normalize this
        // newly introduced Value before hashing the typed configuration too,
        // so API authorship and helper re-reading derive identical bytes.
        checkpoint_restore: restore
            .map(|value| {
                let bytes = crate::canonical::python_json_bytes(value)
                    .map_err(|_| helper::Error::Metadata)?;
                serde_json::from_slice(&bytes).map_err(|_| helper::Error::Metadata)
            })
            .transpose()?,
        datasets: datasets.to_vec(),
        data_preparation: None,
    };
    managed::execution_environment(&config, &manifest).map_err(|_| helper::Error::Metadata)?;
    Ok(Package { config, manifest })
}
/// Shared by API authorship and gateway staging: both derive exactly the same
/// managed configuration hash, including optional script preparation.
pub fn plan_for_document(
    scope: &helper::Scope,
    spec: &JobSpec,
    runtime: &Runtime,
    document: &serde_json::Value,
    datasets: &[crate::datasets::Mount],
) -> helper::Result<Package> {
    let mut package = plan_with_inputs(
        scope,
        spec,
        runtime,
        document.get("native_checkpoint"),
        datasets,
    )?;
    if let Some(value) = document.get("data_preparation") {
        let plan = crate::data_preparation::Plan {
            contract: crate::data_preparation::CONTRACT.into(),
            request: serde_json::from_value(value.clone()).map_err(|_| helper::Error::Metadata)?,
            source_id: document["source_id"]
                .as_str()
                .ok_or(helper::Error::Metadata)?
                .into(),
            image: spec
                .backend
                .oci_image
                .clone()
                .ok_or(helper::Error::Metadata)?,
            mounted_root: spec.backend.mount_root.clone(),
            cache_root: spec.storage.project_data_root.clone(),
            workdir: spec.execution.workdir.clone(),
            env: spec.execution.env.clone(),
            require_cached: true,
        };
        managed::attach_preparation(&mut package.config, &mut package.manifest, plan)
            .map_err(|_| helper::Error::Metadata)?;
    }
    Ok(package)
}
pub fn command(package: &Package, runtime: &Runtime) -> helper::Result<Vec<String>> {
    let bytes = package.config_bytes()?;
    let root = package
        .config
        .manifest_path
        .parent()
        .and_then(Path::to_str)
        .ok_or(helper::Error::Metadata)?;
    Ok(vec![
        runtime.program.clone(),
        "managed-launch".into(),
        "--config".into(),
        format!("{root}/{CONFIG_FILE}"),
        "--sha256".into(),
        helper::sha256_bytes(&bytes)[7..].into(),
        "--initialize-and-run".into(),
    ])
}
