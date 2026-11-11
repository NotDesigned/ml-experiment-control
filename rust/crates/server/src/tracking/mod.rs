//! Optional W&B and MLflow shells. Credentials and SDK effects stay outside pure
//! reducers; canonical record acceptance never depends on cloud publication success.
pub mod mlflow;
mod publisher;
mod transport;
use ml_exp_core::tracking::{RoutingSettings, WandbOptions, WandbSettings};
pub use publisher::Publisher;
pub use publisher::Source;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};
type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid private W&B publisher configuration")]
    Configuration,
    #[error("W&B SDK communication unavailable; rows are resent from the cursor")]
    Transport,
    #[error("W&B SDK is not installed in the configured interpreter")]
    SdkMissing,
    #[error("W&B SDK request failed: {0}; rows are resent from the cursor")]
    SdkRequest(&'static str),
    #[error("W&B publication blocked: {0}; journal and cursor retained")]
    Publication(&'static str),
    #[error("W&B publication identity or state is invalid")]
    Identity,
    #[error("W&B journal or private publication store unavailable")]
    Storage,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub python: PathBuf,
    pub api_key_file: PathBuf,
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default = "enabled")]
    pub enabled: bool,
}
fn enabled() -> bool {
    true
}
#[derive(Clone)]
pub struct Runtime {
    config: Config,
    key: String,
    routing: RoutingSettings,
    diagnostic: Option<&'static str>,
}
impl Config {
    /// Local validation only: never start the SDK or resolve a default entity.
    pub fn check_references(&self) -> Result<()> {
        let bytes = ml_exp_io::read_private(&self.api_key_file, 4096, true)
            .map_err(why!(Error::Configuration, "W&B API key file unreadable"))?;
        let key = std::str::from_utf8(&bytes)
            .map_err(why!(Error::Configuration, "W&B API key encoding invalid"))?
            .trim();
        WandbSettings::from_value(&json!({"api_key":key}))
            .map_err(why!(Error::Configuration, "W&B API key format invalid"))?;
        let metadata = std::fs::metadata(&self.python).map_err(why!(Error::Configuration))?;
        if !metadata.is_file() {
            return Err(Error::Configuration);
        }
        Ok(())
    }
    /// The W&B part of a `tracking_config` file. Its `mlflow` block is read by
    /// [`Settings::load`]; a file without W&B fields is not a W&B configuration.
    pub fn load(path: &Path) -> Result<Self> {
        let mut mapping = read_mapping(path)?;
        mapping.remove("mlflow");
        Self::from_mapping(mapping)
    }
    fn from_mapping(mapping: serde_json::Map<String, Value>) -> Result<Self> {
        let config: Self =
            serde_json::from_value(Value::Object(mapping)).map_err(why!(Error::Configuration))?;
        if config.schema_version != 1
            || !config.python.is_absolute()
            || !config.api_key_file.is_absolute()
        {
            return Err(Error::Configuration);
        }
        WandbOptions::from_value(&json!({"entity":config.entity,"project":config.project}))
            .map_err(why!(Error::Configuration))?;
        Ok(config)
    }
}
fn read_mapping(path: &Path) -> Result<serde_json::Map<String, Value>> {
    let bytes = ml_exp_io::read_private(path, 16384, true).map_err(why!(Error::Configuration))?;
    ml_exp_core::json::mapping(&bytes).map_err(why!(Error::Configuration))
}
/// Both optional destinations of one private `tracking_config` file. The W&B
/// fields stay where they always were; an optional `mlflow` block adds MLflow.
/// A file holding only `schema_version` and `mlflow` configures no W&B publisher.
pub struct Settings {
    pub wandb: Option<Config>,
    pub mlflow: Option<mlflow::Config>,
}
impl Settings {
    pub fn load(path: &Path) -> Result<Self> {
        let mut mapping = read_mapping(path)?;
        let mlflow = mapping
            .remove("mlflow")
            .map(mlflow::Config::from_value)
            .transpose()
            .map_err(|_| Error::Configuration)?;
        let wandb = if mlflow.is_some() && mapping.keys().all(|key| key == "schema_version") {
            if mapping.get("schema_version") != Some(&json!(1)) {
                return Err(Error::Configuration);
            }
            None
        } else {
            Some(Config::from_mapping(mapping)?)
        };
        Ok(Self { wandb, mlflow })
    }
    /// Local validation only: no SDK start and no MLflow request.
    pub fn check_references(&self) -> Result<()> {
        if let Some(wandb) = &self.wandb {
            wandb.check_references()?;
        }
        if let Some(mlflow) = &self.mlflow {
            mlflow
                .check_references()
                .map_err(|_| Error::Configuration)?;
        }
        Ok(())
    }
}
impl Runtime {
    pub fn open(config: Config) -> Result<Self> {
        let bytes = ml_exp_io::read_private(&config.api_key_file, 4096, true)
            .map_err(why!(Error::Configuration, "W&B API key file unreadable"))?;
        let key = std::str::from_utf8(&bytes)
            .map_err(why!(Error::Configuration, "W&B API key encoding invalid"))?
            .trim()
            .to_owned();
        WandbSettings::from_value(&json!({"api_key":key}))
            .map_err(why!(Error::Configuration, "W&B API key format invalid"))?;
        let mut diagnostic = None;
        let entity = if config.enabled && config.entity.is_none() {
            let lookup = || -> Result<String> {
                let mut sdk = transport::Sdk::open(&config.python)?;
                let response = sdk.call(
                    &json!({"operation":"default_entity","api_key":key}),
                    &AtomicBool::new(false),
                )?;
                WandbOptions::from_value(&json!({"entity":response["entity"]}))
                    .map_err(why!(Error::Identity))?
                    .entity
                    .ok_or(Error::Identity)
            };
            match lookup() {
                Ok(entity) => entity,
                Err(error) => {
                    diagnostic = Some(if matches!(error, Error::SdkMissing) {
                        "WANDB_SDK_NOT_INSTALLED"
                    } else {
                        "DEFAULT_ENTITY_UNAVAILABLE"
                    });
                    String::new()
                }
            }
        } else {
            config.entity.clone().unwrap_or_default()
        };
        let routing = RoutingSettings {
            enabled: config.enabled,
            credentials_configured: true,
            resolved_entity: (!entity.is_empty()).then_some(entity),
            project: config.project.clone(),
            ..RoutingSettings::default()
        };
        Ok(Self {
            config,
            key,
            routing,
            diagnostic,
        })
    }
    pub fn routing(&self) -> &RoutingSettings {
        &self.routing
    }
    pub fn diagnostic(&self) -> Option<&'static str> {
        self.diagnostic
    }
}
