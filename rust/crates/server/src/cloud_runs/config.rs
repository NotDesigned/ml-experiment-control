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
    #[serde(default)]
    pub allocation: Allocation,
    #[serde(default = "timeout")]
    pub timeout_seconds: u32,
}
fn timeout() -> u32 {
    30
}
impl Definition {
    pub fn validate(&self) -> cloud::Result<()> {
        ml_exp_core::identity::validate_id(&self.executor)
            .map_err(|_| cloud::Error::configuration())?;
        self.profile.validate()?;
        if !(1..=120).contains(&self.timeout_seconds)
            || !ml_exp_core::managed::absolute(Path::new(&self.run_root))
            || self.runtimes.is_empty()
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
}
impl Settings {
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
            || settings.backends.is_empty()
            || settings.backends.len() > 64
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
