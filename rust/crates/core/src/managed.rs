//! Pure configuration and environment decisions for a sealed managed launch.
//! Paths are values here; opening files and taking runtime snapshots are effects.
pub mod evidence;
pub mod history;
pub mod remote;
use crate::{canonical, launch};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid sealed managed launch configuration")]
pub struct Error;
type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub manifest_path: PathBuf,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub endpoint: String,
    pub attempt_root: PathBuf,
    pub workdir: PathBuf,
    pub maximum_timeout_seconds: u64,
    #[serde(default)]
    pub base_environment: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_binding: Option<launch::Binding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_directory: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime_environment: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_restore: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<crate::datasets::Mount>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_preparation: Option<crate::data_preparation::Plan>,
}
pub fn absolute(path: &Path) -> bool {
    path.is_absolute()
        && path.to_str().is_some()
        && path
            .components()
            .all(|p| matches!(p, Component::RootDir | Component::Normal(_)))
}
fn key(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
        && bytes.all(|c| c.is_ascii_alphanumeric() || c == b'_')
}
pub fn environment(values: &BTreeMap<String, String>) -> bool {
    values.len() <= 512
        && values
            .iter()
            .all(|(name, value)| key(name) && !value.contains('\0'))
        && values
            .iter()
            .try_fold(0usize, |n, (k, v)| n.checked_add(k.len() + v.len() + 2))
            .is_some_and(|n| n <= 128 * 1024)
}
fn runtime_key(name: &str) -> bool {
    matches!(
        name,
        "CUDA_VISIBLE_DEVICES"
            | "NVIDIA_VISIBLE_DEVICES"
            | "NVIDIA_DRIVER_CAPABILITIES"
            | "LD_LIBRARY_PATH"
    )
}
pub fn environment_policy(base: &BTreeMap<String, String>, runtime: &[String]) -> bool {
    environment(base)
        && !base.keys().any(|name| {
            name.starts_with("ML_EXPD_")
                || [
                    "OUTPUT_DIR",
                    "PROJECT_NAME",
                    "RUN_ID",
                    "ATTEMPT_ID",
                    "DATASETS_JSON",
                ]
                .contains(&name.as_str())
        })
        && runtime.len() <= 4
        && runtime.iter().all(|name| runtime_key(name))
        && runtime.iter().all(|name| !base.contains_key(name))
        && runtime
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == runtime.len()
}
impl Config {
    pub fn binding(&self) -> Result<launch::Binding> {
        match (&self.local_binding, self.endpoint.is_empty()) {
            (Some(binding), true) => {
                binding.validate().map_err(|_| Error)?;
                Ok(binding.clone())
            }
            (None, false) => {
                let target = launch::endpoint(&self.endpoint).map_err(|_| Error)?;
                Ok(launch::Binding {
                    identity: target.identity,
                    sha256: target.sha256,
                })
            }
            _ => Err(Error),
        }
    }
    pub fn output_path(&self) -> PathBuf {
        self.output_directory
            .clone()
            .unwrap_or_else(|| self.attempt_root.join("outputs"))
    }
    pub fn validate(&self) -> Result<()> {
        if let Some(plan) = &self.data_preparation {
            plan.validate().map_err(|_| Error)?;
            if self.workdir.to_str() != Some(plan.workdir.as_str()) {
                return Err(Error);
            }
        }
        crate::datasets::mounts(&self.datasets).map_err(|_| Error)?;
        let project = self.binding()?.identity.project;
        if self.datasets.iter().any(|v| v.receipt.project != project) {
            return Err(Error);
        }
        if let Some(value) = &self.checkpoint_restore {
            crate::checkpoints::restore_plan(value, None).map_err(|_| Error)?;
        }
        if !absolute(&self.manifest_path)
            || !absolute(&self.attempt_root)
            || self.attempt_root.file_name().is_none()
            || !absolute(&self.workdir)
            || !absolute(&self.output_path())
            || self.output_path().file_name().is_none()
            || !(1..1000 * 3600).contains(&self.maximum_timeout_seconds)
            || !environment_policy(&self.base_environment, &self.runtime_environment)
        {
            return Err(Error);
        }
        self.binding().map(|_| ())
    }
    pub fn validate_runtime(&self, values: &BTreeMap<String, String>) -> Result<()> {
        if !environment(values)
            || values.iter().any(|(name, value)| {
                !self.runtime_environment.contains(name) || value.len() > 16384
            })
        {
            return Err(Error);
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        canonical::digest(&serde_json::to_value(self).map_err(|_| Error)?).map_err(|_| Error)
    }
}

/// Attach a trusted, frozen download plan and reseal the launch manifest.
pub fn attach_preparation(
    config: &mut Config,
    manifest: &mut launch::Manifest,
    plan: crate::data_preparation::Plan,
) -> Result<()> {
    plan.validate().map_err(|_| Error)?;
    manifest
        .environment
        .insert("DATA_DIR".into(), plan.tree().map_err(|_| Error)?);
    config.data_preparation = Some(plan);
    config.local_binding.as_mut().ok_or(Error)?.sha256 =
        crate::slurm::helper::sha256_bytes(&manifest.encode().map_err(|_| Error)?)[7..].into();
    config.validate()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionError {
    #[error("invalid managed process configuration")]
    Configuration,
    #[error("managed process identity differs from its sealed manifest")]
    Binding,
    #[error("managed I/O protocol has not been connected")]
    Unsupported,
}
/// Build the entire payload environment from approved immutable inputs. This
/// deliberately has no ambient environment lookup, filesystem access or clock.
pub fn execution_environment(
    config: &Config,
    manifest: &launch::Manifest,
) -> std::result::Result<BTreeMap<String, String>, AdmissionError> {
    config
        .validate()
        .map_err(|_| AdmissionError::Configuration)?;
    let target = config
        .binding()
        .map_err(|_| AdmissionError::Configuration)?;
    let body = manifest
        .encode()
        .map_err(|_| AdmissionError::Configuration)?;
    launch::verify_binding(&body, &target).map_err(|_| AdmissionError::Binding)?;
    if config
        .runtime_environment
        .iter()
        .any(|name| manifest.environment.contains_key(name))
    {
        return Err(AdmissionError::Binding);
    }
    if manifest.workdir
        != config
            .workdir
            .to_str()
            .ok_or(AdmissionError::Configuration)?
        || manifest.environment.get("OUTPUT_DIR").map(String::as_str)
            != config.output_path().to_str()
        || manifest.timeout_seconds > config.maximum_timeout_seconds
    {
        return Err(AdmissionError::Binding);
    }
    if manifest
        .environment
        .keys()
        .any(|name| name.starts_with("ML_EXPD_"))
    {
        return Err(AdmissionError::Unsupported);
    }
    let mut values = config.base_environment.clone();
    values.extend(manifest.environment.clone());
    if let Some(preparation) = &config.data_preparation {
        if values.get("DATA_DIR") != Some(&preparation.tree().map_err(|_| AdmissionError::Binding)?)
        {
            return Err(AdmissionError::Binding);
        }
    }
    if !config.datasets.is_empty() {
        let inputs = values.get("INPUT_DIR").ok_or(AdmissionError::Binding)?;
        let mapping: BTreeMap<_, _> = config
            .datasets
            .iter()
            .map(|v| (v.name.clone(), format!("{inputs}/{}", v.name)))
            .collect();
        let expected = serde_json::to_string(&mapping).map_err(|_| AdmissionError::Binding)?;
        if values.get("DATASETS_JSON") != Some(&expected) {
            let direct: BTreeMap<_, _> = config
                .datasets
                .iter()
                .map(|v| (v.name.clone(), v.receipt.root.clone()))
                .collect();
            if !crate::datasets::root_path(inputs)
                || config
                    .datasets
                    .iter()
                    .any(|v| !v.receipt.root.starts_with(&format!("{inputs}/")))
                || values.get("DATASETS_JSON")
                    != Some(&serde_json::to_string(&direct).map_err(|_| AdmissionError::Binding)?)
            {
                return Err(AdmissionError::Binding);
            }
        }
    } else if values.contains_key("DATASETS_JSON") {
        return Err(AdmissionError::Binding);
    }
    if let Some(value) = &config.checkpoint_restore {
        let plan =
            crate::checkpoints::restore_plan(value, None).map_err(|_| AdmissionError::Binding)?;
        for (key, expected) in [
            (
                "RESUME_DIR",
                format!("{}/checkpoints/{}", plan.state_root, plan.generation),
            ),
            ("RESUME_CHECKPOINT_ID", plan.checkpoint_id),
            ("RESUME_STEP", plan.step.to_string()),
        ] {
            if values.get(key) != Some(&expected) {
                return Err(AdmissionError::Binding);
            }
        }
    } else if ["RESUME_DIR", "RESUME_CHECKPOINT_ID", "RESUME_STEP"]
        .iter()
        .any(|key| values.contains_key(*key))
    {
        return Err(AdmissionError::Binding);
    }
    for (name, expected) in [
        ("PROJECT_NAME", &target.identity.project),
        ("RUN_ID", &target.identity.run_id),
        ("ATTEMPT_ID", &target.identity.attempt_id),
    ] {
        if values.get(name).is_some_and(|actual| actual != expected) {
            return Err(AdmissionError::Binding);
        }
        values.insert(name.into(), expected.clone());
    }
    if !environment(&values) {
        return Err(AdmissionError::Configuration);
    }
    let program = Path::new(manifest.argv.first().ok_or(AdmissionError::Configuration)?);
    if !program.is_absolute() {
        if manifest.argv[0].contains('/')
            || !values.get("PATH").is_some_and(|path| {
                !path.is_empty() && path.split(':').all(|part| absolute(Path::new(part)))
            })
        {
            return Err(AdmissionError::Configuration);
        }
    } else if !absolute(program) {
        return Err(AdmissionError::Configuration);
    }
    Ok(values)
}
