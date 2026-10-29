//! One private deployment file selects existing independently approved configs.
//! Parsing grants no initialization or provider effects.
use serde::Deserialize;
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

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
    }
}
