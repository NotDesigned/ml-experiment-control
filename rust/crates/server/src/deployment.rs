//! One private deployment file selects existing independently approved configs.
//! Parsing grants no initialization or provider effects.
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Debug, thiserror::Error)]
#[error("native deployment refused at {stage}; configuration and existing state were not reset")]
pub struct Error {
    pub stage: &'static str,
}
fn at(stage: &'static str) -> Error {
    Error { stage }
}

/// Safe local configuration facts. No credentials or private paths are returned.
#[derive(serde::Serialize)]
pub struct Check {
    pub configuration_valid: bool,
    pub provider_connectivity_checked: bool,
    pub registry_initialized: bool,
    pub projects: usize,
    pub executors: BTreeSet<String>,
    pub historical_reads: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub schema_version: u32,
    #[serde(default = "loopback")]
    pub bind: SocketAddr,
    pub token_file: PathBuf,
    pub read_config: PathBuf,
    pub history_config: Option<PathBuf>,
    pub submission_config: Option<PathBuf>,
    pub cloud_config: Option<PathBuf>,
    pub tracking_config: Option<PathBuf>,
    #[serde(default)]
    pub project_lifecycle: bool,
    #[serde(default)]
    pub source_archives: bool,
    #[serde(default)]
    pub administration: bool,
}
fn loopback() -> SocketAddr {
    ([127, 0, 0, 1], 18765).into()
}
impl Settings {
    pub fn validate(&self) -> Result<(), crate::read_config::Error> {
        if self.schema_version != 1
            || std::iter::once(&self.token_file)
                .chain(std::iter::once(&self.read_config))
                .chain(self.history_config.iter())
                .chain(self.submission_config.iter())
                .chain(self.cloud_config.iter())
                .chain(self.tracking_config.iter())
                .any(|p| {
                    !p.is_absolute()
                        || p.components().any(|c| {
                            matches!(
                                c,
                                std::path::Component::ParentDir | std::path::Component::CurDir
                            )
                        })
                })
            || !self.source_archives
                && (self.submission_config.is_some()
                    || self.cloud_config.is_some()
                    || self.administration)
        {
            return Err(crate::read_config::Error);
        }
        Ok(())
    }
    pub fn from_json(bytes: &[u8]) -> Result<Self, crate::read_config::Error> {
        if bytes.len() > 64 * 1024 {
            return Err(crate::read_config::Error);
        }
        let mapping = ml_exp_core::json::mapping(bytes).map_err(|_| crate::read_config::Error)?;
        let value: Self = serde_json::from_value(serde_json::Value::Object(mapping))
            .map_err(|_| crate::read_config::Error)?;
        value.validate()?;
        Ok(value)
    }
    pub fn load(path: &Path) -> Result<Self, crate::read_config::Error> {
        Self::from_json(
            &ml_exp_io::read_private(path, 64 * 1024, true)
                .map_err(|_| crate::read_config::Error)?,
        )
    }

    /// Read private dependencies before any SDK/provider or initialization
    /// effects. Passing this check makes no claim about a remote connection.
    pub fn check(&self) -> Result<Check, Error> {
        self.validate().map_err(|_| at("deployment"))?;
        if !self.bind.ip().is_loopback() {
            return Err(at("loopback_bind"));
        }
        crate::Auth::load(&self.token_file).map_err(|_| at("api_token"))?;
        let reads = crate::read_config::load(&self.read_config).map_err(|_| at("read_config"))?;
        for project in &reads.projects {
            let parent = ml_exp_io::directory::Directory::open(
                project
                    .project_file
                    .parent()
                    .ok_or_else(|| at("project_definition"))?,
                false,
            )
            .map_err(|_| at("project_definition"))?;
            let bytes = parent
                .read_regular(
                    project
                        .project_file
                        .file_name()
                        .ok_or_else(|| at("project_definition"))?,
                    4 * 1024 * 1024,
                )
                .map_err(|_| at("project_definition"))?;
            let mapping =
                crate::experiments::parse_mapping(&bytes, crate::experiments::Format::Yaml)
                    .map_err(|_| at("project_definition"))?;
            let definition = ml_exp_core::projects::ProjectDefinition::from_value(
                &serde_json::Value::Object(mapping),
            )
            .map_err(|_| at("project_definition"))?;
            if definition.project() != project.project
                || self.project_lifecycle && definition.campaign_files().next().is_some()
            {
                return Err(at("project_definition"));
            }
            for root in &project.run_roots {
                ml_exp_io::directory::Directory::open(root, false)
                    .map_err(|_| at("project_run_root"))?;
            }
        }
        let mut executors = BTreeSet::new();
        if let Some(path) = &self.submission_config {
            let slurm = crate::submissions::config::load(path).map_err(|_| at("slurm_config"))?;
            for backend in slurm.backends {
                if !executors.insert(backend.profile.target.backend_id.clone()) {
                    return Err(at("duplicate_executor"));
                }
                for path in [&backend.ssh.config_file, &backend.ssh.known_hosts_file] {
                    ml_exp_io::read_private(path, 1024 * 1024, true)
                        .map_err(|_| at("slurm_ssh"))?;
                }
            }
        }
        if let Some(path) = &self.cloud_config {
            let cloud = crate::cloud_runs::Settings::load(path).map_err(|_| at("cloud_config"))?;
            for backend in &cloud.backends {
                if !executors.insert(backend.executor.clone()) {
                    return Err(at("duplicate_executor"));
                }
            }
            cloud
                .check_references()
                .map_err(|_| at("cloud_dependencies"))?;
        }
        if let Some(path) = &self.tracking_config {
            crate::tracking::Config::load(path)
                .and_then(|c| c.check_references())
                .map_err(|_| at("tracking_config"))?;
        }
        if let Some(path) = &self.history_config {
            crate::history::Bridge::open(
                crate::history::Settings::load(path).map_err(|_| at("history_config"))?,
            )
            .and_then(|b| b.validate_native(&reads))
            .map_err(|_| at("history_separation"))?;
        }
        let parent = ml_exp_io::directory::Directory::open(
            reads
                .registry_root
                .parent()
                .ok_or_else(|| at("registry_parent"))?,
            false,
        )
        .map_err(|_| at("registry_parent"))?;
        let exists = parent
            .entry_exists(
                reads
                    .registry_root
                    .file_name()
                    .ok_or_else(|| at("registry_parent"))?,
            )
            .map_err(|_| at("registry_parent"))?;
        let registry_initialized =
            exists && crate::read_config::workspace(&reads.registry_root).is_ok();
        if exists && !registry_initialized {
            return Err(at("existing_non_native_registry"));
        }
        Ok(Check {
            configuration_valid: true,
            provider_connectivity_checked: false,
            registry_initialized,
            projects: reads.projects.len(),
            executors,
            historical_reads: self.history_config.is_some(),
        })
    }

    /// Bootstrap one entirely new registry. A repeated or partial bootstrap is
    /// refused, never repaired or erased. Existing granular commands remain
    /// available for an operator to inspect and finish a partial initialization.
    pub fn initialize(&self) -> Result<serde_json::Value, Error> {
        let checked = self.check()?;
        if checked.registry_initialized {
            return Err(at("registry_already_exists"));
        }
        let reads = crate::read_config::load(&self.read_config).map_err(|_| at("read_config"))?;
        let workspace = crate::read_config::initialize(&reads.registry_root)
            .map_err(|_| at("registry_creation"))?;
        let mut stores = vec!["registry"];
        if self.project_lifecycle {
            let (now, identity) =
                crate::projects::transition_context().map_err(|_| at("project_lifecycle"))?;
            crate::projects::Service::initialize(&reads, &now, &identity)
                .map_err(|_| at("project_lifecycle"))?;
            stores.push("project_lifecycle");
        }
        if self.source_archives {
            crate::sources::Service::initialize(&reads).map_err(|_| at("source_archives"))?;
            stores.push("source_archives");
        }
        if self.submission_config.is_some() {
            crate::submissions::Service::initialize(&reads).map_err(|_| at("slurm_namespace"))?;
            stores.push("slurm");
        }
        if self.cloud_config.is_some() {
            crate::cloud_runs::Service::initialize(&reads).map_err(|_| at("cloud_namespace"))?;
            stores.push("cloud");
        }
        if self.administration {
            crate::administration::Service::initialize(&reads).map_err(|_| at("administration"))?;
            stores.push("administration");
        }
        Ok(
            serde_json::json!({"initialized":true,"workspace_id":workspace,"stores":stores,"provider_jobs_created":0}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn deployment_config_keeps_effects_explicit_and_paths_absolute() {
        let base = json!({"schema_version":1,"token_file":"/private/token","read_config":"/private/reads.json"});
        let parse = |v: &serde_json::Value| Settings::from_json(&serde_json::to_vec(v).unwrap());
        let s = parse(&base).unwrap();
        assert_eq!(s.bind, loopback());
        assert!(!s.source_archives && !s.project_lifecycle && !s.administration);
        for (field, value) in [
            ("token_file", json!("relative")),
            ("read_config", json!("/private/../reads.json")),
            ("cloud_config", json!("/private/cloud.json")),
            ("source_archives", json!("true")),
            ("schema_version", json!(2)),
            ("password", json!("never inline secrets")),
        ] {
            let mut v = base.clone();
            v[field] = value;
            assert!(parse(&v).is_err(), "{field}");
        }
        let mut v = base;
        v["source_archives"] = json!(true);
        v["cloud_config"] = json!("/private/cloud.json");
        assert!(parse(&v).is_ok());
        assert!(Settings::from_json(br#"{"schema_version":1,"schema_version":1}"#).is_err());
        Settings::from_json(include_bytes!("../../../deploy/server.example.json")).unwrap();
        crate::read_config::Settings::from_json(include_bytes!(
            "../../../deploy/reads.example.json"
        ))
        .unwrap();
        for bytes in [
            include_bytes!("../../../deploy/pai.example.json").as_slice(),
            include_bytes!("../../../deploy/sensecore.example.json").as_slice(),
        ] {
            crate::cloud_runs::Settings::from_slice(bytes).unwrap();
        }
        crate::submissions::config::Settings::from_slice(include_bytes!(
            "../../../deploy/slurm.example.json"
        ))
        .unwrap()
        .into_backends()
        .unwrap();
    }
}
