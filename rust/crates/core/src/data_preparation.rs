//! Frozen download scripts and verified persistent caches. No execution here.
use crate::{canonical, datasets, environment, sources};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CONTRACT: &str = "native-script-preparation.v1";
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub script: String,
    #[serde(default = "interpreter")]
    pub interpreter: String,
    #[serde(default)]
    pub arguments: Vec<String>,
    #[serde(default = "timeout")]
    pub timeout_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_content_sha256: Option<String>,
}
fn interpreter() -> String {
    "python3".into()
}
fn timeout() -> u64 {
    1200
}
pub fn content_hash(files: &[crate::artifacts::FileReceipt]) -> Result<String, &'static str> {
    if files.is_empty() || files.len() > crate::artifacts::MAX_FILES {
        return Err("data file count");
    }
    let mut previous = None;
    let mut bytes = 0u64;
    for file in files {
        if !datasets::path(&file.path)
            || sources::validate_archive_digest(&file.sha256).is_err()
            || previous.is_some_and(|p: &str| p >= file.path.as_str())
        {
            return Err("data file identity");
        }
        bytes = bytes.checked_add(file.bytes).ok_or("data byte count")?;
        previous = Some(file.path.as_str());
    }
    let value = serde_json::to_value(files).map_err(|_| "data inventory")?;
    if serde_json::to_vec(&value)
        .map_err(|_| "data inventory")?
        .len()
        > 4 * 1024 * 1024
    {
        return Err("data inventory bound");
    }
    canonical::digest(&value).map_err(|_| "data content hash")
}
impl Request {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !datasets::path(&self.script)
            || self.script.len() > 4096
            || !matches!(self.interpreter.as_str(), "python3" | "/bin/sh")
            || !(5..=86400).contains(&self.timeout_seconds)
            || self.arguments.len() > 128
            || self
                .arguments
                .iter()
                .any(|a| a.contains('\0') || a.len() > 8192)
            || self.arguments.iter().map(String::len).sum::<usize>() > 32768
            || self
                .expected_content_sha256
                .as_deref()
                .is_some_and(|s| sources::validate_archive_digest(s).is_err())
        {
            return Err("invalid data preparation script, arguments, timeout or hash");
        }
        Ok(())
    }
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub contract: String,
    pub request: Request,
    pub source_id: String,
    pub image: String,
    pub cache_root: String,
    pub mounted_root: String,
    pub workdir: String,
    pub env: BTreeMap<String, String>,
    pub require_cached: bool,
}
impl Plan {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.request.validate()?;
        sources::validate_source_id(&self.source_id).map_err(|_| "data preparation source")?;
        environment::validate(&self.env).map_err(|_| "data preparation environment")?;
        if self.contract != CONTRACT
            || !datasets::root_path(&self.cache_root)
            || !datasets::root_path(&self.mounted_root)
            || self.cache_root == self.mounted_root
            || !self
                .cache_root
                .starts_with(&format!("{}/", self.mounted_root))
            || !datasets::root_path(&self.workdir)
            || !self
                .image
                .rsplit_once('@')
                .is_some_and(|(_, hash)| crate::slurm::helper::valid_digest(hash))
        {
            return Err("invalid data preparation scope");
        }
        Ok(())
    }
    pub fn id(&self) -> Result<String, &'static str> {
        self.validate()?;
        // Cache policy and mount location are execution concerns. Code, image,
        // arguments and user environment define the reusable preparation.
        canonical::digest(&serde_json::json!([
            CONTRACT,
            self.request,
            self.source_id,
            self.image,
            self.workdir,
            self.env
        ]))
        .map(|s| format!("preparation.{s}"))
        .map_err(|_| "data preparation identity")
    }
    pub fn tree(&self) -> Result<String, &'static str> {
        Ok(format!(
            "{}/script-preparations/{}/tree",
            self.cache_root,
            self.id()?
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validation_and_identity_separate_data_from_execution_policy() {
        let mut plan = Plan {
            contract: CONTRACT.into(),
            request: serde_json::from_value(serde_json::json!({"script":"fetch.py"})).unwrap(),
            source_id: format!("source.{}", "a".repeat(64)),
            image: format!("registry/image@sha256:{}", "b".repeat(64)),
            mounted_root: "/data".into(),
            cache_root: "/data/demo".into(),
            workdir: "/workspace".into(),
            env: BTreeMap::new(),
            require_cached: false,
        };
        let first = plan.id().unwrap();
        plan.require_cached = true;
        plan.cache_root = "/other/data".into();
        plan.mounted_root = "/other".into();
        assert_eq!(plan.id().unwrap(), first);
        plan.request.arguments.push("different".into());
        assert_ne!(plan.id().unwrap(), first);
        plan.request.script = "../escape".into();
        assert!(plan.validate().is_err());
        plan.request.script = "fetch.py".into();
        plan.env.insert("DATA_DIR".into(), "/escape".into());
        assert!(plan.validate().is_err());
    }
}
