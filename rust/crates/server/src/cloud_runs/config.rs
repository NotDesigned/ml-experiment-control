use crate::backends::cloud::{self, adapter::Profile};
use ml_exp_core::cloud::{self as domain, pai::Allocation, run::Runtime};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub executor: String,
    pub profile: Profile,
    pub run_root: String,
    pub credentials_file: PathBuf,
    pub registry_auth_file: Option<PathBuf>,
    pub runtimes: BTreeMap<String, Runtime>,
    pub data_root: Option<String>,
    #[serde(default)]
    pub datasets: Vec<ml_exp_core::datasets::Receipt>,
    #[serde(default)]
    pub imports: Vec<ml_exp_core::datasets::import::Config>,
    pub collector: Option<Profile>,
    pub data_copier: Option<Profile>,
    /// Opt-in highest-SPOT selection among approved same-zone/VPC pools.
    /// The selected profile is frozen per Run and never reselected on retry.
    pub spot_clusters: Option<BTreeSet<String>>,
    #[serde(default)]
    pub allocation: Allocation,
    #[serde(default = "timeout")]
    pub timeout_seconds: u32,
}
fn timeout() -> u32 {
    30
}
impl Definition {
    pub fn data_root(&self) -> cloud::Result<String> {
        let value = match &self.data_root {
            Some(p) => p.clone(),
            None => format!(
                "{}/datasets",
                Path::new(&self.run_root)
                    .parent()
                    .and_then(Path::to_str)
                    .ok_or_else(cloud::Error::configuration)?
            ),
        };
        if !ml_exp_core::datasets::root_path(&value)
            || value == self.run_root
            || value.starts_with(&format!("{}/", self.run_root))
            || self.run_root.starts_with(&format!("{value}/"))
        {
            return Err(cloud::Error::configuration());
        }
        Ok(value)
    }
    pub fn storage_layout(&self) -> cloud::Result<domain::launch::StorageLayout> {
        let mount = match &self.profile {
            Profile::Pai(_) => "/mnt/workspace",
            Profile::Sensecore(p) => &p.mount_path,
        };
        let storage = domain::launch::StorageLayout {
            mount: mount.into(),
            runs: self.run_root.clone().into(),
            datasets: self.data_root()?.into(),
        };
        storage
            .validate()
            .map_err(|_| cloud::Error::configuration())?;
        Ok(storage)
    }
    pub fn target(&self) -> cloud::Result<ml_exp_core::slurm::helper::Target> {
        let profile_sha256 = self.profile.digest()?;
        let storage_scope = format!(
            "sha256:{}",
            ml_exp_core::canonical::digest(
                &serde_json::json!({"profile":profile_sha256,"run_root":self.run_root})
            )
            .map_err(|_| cloud::Error::configuration())?
        );
        Ok(ml_exp_core::slurm::helper::Target {
            backend_id: self.executor.clone(),
            profile_sha256,
            storage_scope,
        })
    }
    pub fn validate(&self) -> cloud::Result<()> {
        ml_exp_core::identity::validate_id(&self.executor)
            .map_err(|_| cloud::Error::configuration())?;
        self.profile.validate()?;
        if let Some(names) = &self.spot_clusters {
            if names.is_empty()
                || names.len() > 1024
                || names.iter().any(|s| {
                    domain::component(s).is_err() || s.to_ascii_lowercase().contains("debug")
                })
                || !matches!(&self.profile,Profile::Sensecore(p) if p.purpose==domain::sensecore::Purpose::Training&&p.quota==domain::sensecore::Quota::Spot)
            {
                return Err(cloud::Error::configuration());
            }
        }
        if let Some(copy) = &self.data_copier {
            copy.validate()?;
            match (&self.profile, copy) {
                (Profile::Sensecore(training), Profile::Sensecore(copy))
                    if copy.purpose == domain::sensecore::Purpose::DataCopy
                        && training.scope == copy.scope
                        && training.storage_volume == copy.storage_volume
                        && training.storage_subdir == copy.storage_subdir
                        && training.mount_path == copy.mount_path => {}
                _ => return Err(cloud::Error::configuration()),
            }
        }
        if let Some(collector) = &self.collector {
            collector.validate()?;
            match (&self.profile, collector) {
                (Profile::Pai(training), Profile::Pai(copy))
                    if training.region == copy.region
                        && training.workspace_id == copy.workspace_id
                        && training.storage == copy.storage
                        && training.vpc == copy.vpc => {}
                (Profile::Sensecore(training), Profile::Sensecore(copy))
                    if copy.purpose == domain::sensecore::Purpose::ResultCollection
                        && training.scope == copy.scope
                        && training.storage_volume == copy.storage_volume
                        && training.storage_subdir == copy.storage_subdir
                        && training.mount_path == copy.mount_path => {}
                _ => return Err(cloud::Error::configuration()),
            }
        }
        if !(1..=120).contains(&self.timeout_seconds)
            || !ml_exp_core::managed::absolute(Path::new(&self.run_root))
            || self.runtimes.len() > 4096
        {
            return Err(cloud::Error::configuration());
        }
        match &self.profile {
            Profile::Pai(p) => {
                if p.storage.is_none() || !self.run_root.starts_with("/mnt/workspace/") {
                    return Err(cloud::Error::configuration());
                }
            }
            Profile::Sensecore(p) => {
                if p.worker_nodes != 1
                    || p.purpose != domain::sensecore::Purpose::Training
                    || self.allocation != Allocation::Guaranteed
                    || !self.run_root.starts_with(&format!("{}/", p.mount_path))
                {
                    return Err(cloud::Error::configuration());
                }
            }
        }
        let data_root = self.data_root()?;
        let prefix = match &self.profile {
            Profile::Pai(_) => "/mnt/workspace/".into(),
            Profile::Sensecore(p) => format!("{}/", p.mount_path),
        };
        if !data_root.starts_with(&prefix) || self.datasets.len() > 4096 {
            return Err(cloud::Error::configuration());
        }
        let target = self.target()?;
        let mut identities = BTreeSet::new();
        for receipt in &self.datasets {
            receipt
                .bind(&receipt.project, &target, &data_root)
                .map_err(|_| cloud::Error::configuration())?;
            if !identities.insert((&receipt.project, &receipt.asset_id)) {
                return Err(cloud::Error::configuration());
            }
        }
        if self.imports.len() > 4096 {
            return Err(cloud::Error::configuration());
        }
        for import in &self.imports {
            import
                .validate()
                .map_err(|_| cloud::Error::configuration())?;
            if import.root != data_root
                || import.target != target
                || !identities.insert((&import.project, &import.definition.sha256))
            {
                return Err(cloud::Error::configuration());
            }
        }
        for (id, runtime) in &self.runtimes {
            runtime
                .validate()
                .map_err(|_| cloud::Error::configuration())?;
            if id != &runtime.runtime_id {
                return Err(cloud::Error::configuration());
            }
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub schema_version: u32,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub allow_run_creation: bool,
    pub backends: Vec<Definition>,
    pub exchange: Option<ExchangeSettings>,
    pub artifacts: Option<crate::objects::Settings>,
    pub builds: Option<super::service::builds::Settings>,
    pub debug: Option<super::service::debug::Settings>,
    #[serde(default)]
    pub polling: ml_exp_core::polling::Policy,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExchangeSettings {
    pub public_url: String,
    pub signing_key_file: PathBuf,
}
impl Settings {
    /// Validate private dependencies without opening any durable store or
    /// sending a provider request. This does not establish network readiness.
    pub fn check_references(self) -> cloud::Result<()> {
        super::service::check_configuration(self)
    }
    pub fn load(path: &Path) -> cloud::Result<Self> {
        let bytes = ml_exp_io::read_private(path, 4 * 1024 * 1024, true)
            .map_err(|_| cloud::Error::configuration())?;
        Self::from_slice(&bytes)
    }
    pub fn from_slice(bytes: &[u8]) -> cloud::Result<Self> {
        let value = domain_json(bytes)?;
        let settings: Self =
            serde_json::from_value(value).map_err(|_| cloud::Error::configuration())?;
        if settings.schema_version != 1
            || (settings.backends.is_empty()
                && settings.builds.is_none()
                && settings.debug.is_none())
            || settings.backends.len() > 64
            || !settings.polling.validate()
            || (settings.builds.is_none()
                && settings.backends.iter().any(|b| b.runtimes.is_empty()))
        {
            return Err(cloud::Error::configuration());
        }
        let mut ids = BTreeSet::new();
        for backend in &settings.backends {
            backend.validate()?;
            if !ids.insert(&backend.executor) {
                return Err(cloud::Error::configuration());
            }
        }
        Ok(settings)
    }
}
fn domain_json(bytes: &[u8]) -> cloud::Result<serde_json::Value> {
    ml_exp_core::json::mapping(bytes)
        .map(serde_json::Value::Object)
        .map_err(|_| cloud::Error::configuration())
}
