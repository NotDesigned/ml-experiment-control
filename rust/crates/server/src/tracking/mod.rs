//! Optional W&B shell. Credentials and SDK effects stay outside pure reducers;
//! canonical record acceptance never depends on cloud publication success.
pub mod handoff;
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
    #[error("W&B SDK communication unavailable; durable reservations retained")]
    Transport,
    #[error("W&B SDK is not installed in the configured interpreter")]
    SdkMissing,
    #[error("W&B SDK request failed: {0}; durable publication state retained")]
    SdkRequest(&'static str),
    #[error("W&B publication identity or remote proof is invalid")]
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
    /// Explicit cutover receipt for a separate historical journal. Never
    /// enables ordinary native authorship in that namespace.
    #[serde(default)]
    pub historical_handoff: Option<PathBuf>,
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
        if let Some(path) = &self.historical_handoff {
            handoff::Config::load(path)?.check()?;
        }
        let bytes = ml_exp_io::read_private(&self.api_key_file, 4096, true)
            .map_err(|_| Error::Configuration)?;
        let key = std::str::from_utf8(&bytes)
            .map_err(|_| Error::Configuration)?
            .trim();
        WandbSettings::from_value(&json!({"api_key":key})).map_err(|_| Error::Configuration)?;
        let metadata = std::fs::metadata(&self.python).map_err(|_| Error::Configuration)?;
        if !metadata.is_file() {
            return Err(Error::Configuration);
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = ml_exp_io::read_private(path, 16384, true).map_err(|_| Error::Configuration)?;
        let config: Self = serde_json::from_value(Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Configuration)?,
        ))
        .map_err(|_| Error::Configuration)?;
        if config.schema_version != 1
            || !config.python.is_absolute()
            || !config.api_key_file.is_absolute()
        {
            return Err(Error::Configuration);
        }
        WandbOptions::from_value(&json!({"entity":config.entity,"project":config.project}))
            .map_err(|_| Error::Configuration)?;
        Ok(config)
    }
}
impl Runtime {
    pub fn open(config: Config) -> Result<Self> {
        let bytes = ml_exp_io::read_private(&config.api_key_file, 4096, true)
            .map_err(|_| Error::Configuration)?;
        let key = std::str::from_utf8(&bytes)
            .map_err(|_| Error::Configuration)?
            .trim()
            .to_owned();
        WandbSettings::from_value(&json!({"api_key":key})).map_err(|_| Error::Configuration)?;
        let mut diagnostic = None;
        let entity = if config.enabled && config.entity.is_none() {
            let lookup = || -> Result<String> {
                let mut sdk = transport::Sdk::open(&config.python)?;
                let response = sdk.call(
                    &json!({"operation":"default_entity","api_key":key}),
                    &AtomicBool::new(false),
                )?;
                WandbOptions::from_value(&json!({"entity":response["entity"]}))
                    .map_err(|_| Error::Identity)?
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
    pub fn historical_handoff(&self) -> Option<&Path> {
        self.config.historical_handoff.as_deref()
    }
}
