use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{Error, component, printable};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlurmConfig {
    pub kind: String,
    pub ssh_alias: String,
    pub partition: String,
    pub account: String,
    pub qos: String,
    pub gres: String,
    pub time: String,
    pub source_dir: String,
    pub sif_path: String,
    pub mount_root: String,
    #[serde(default)]
    pub oci_image: Option<String>,
    #[serde(default)]
    pub apptainer_cache_dir: Option<String>,
    #[serde(default)]
    pub apptainer_tmp_dir: Option<String>,
    #[serde(default)]
    pub apptainer_unsquash: bool,
    #[serde(default = "stage_timeout")]
    pub image_stage_timeout_seconds: u32,
}
fn stage_timeout() -> u32 {
    1080
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    pub gpus: u32,
    #[serde(default = "cpus")]
    pub cpus: u32,
    #[serde(default = "nodes")]
    pub nodes: u32,
    #[serde(default)]
    pub memory_gb: Option<u32>,
}
fn cpus() -> u32 {
    8
}
fn nodes() -> u32 {
    1
}

impl Resources {
    pub fn validate(&self) -> Result<(), Error> {
        if self.gpus == 0 {
            return Err(Error::Configuration("resources.gpus"));
        }
        self.validate_compute()
    }
    /// Backend-neutral single-node resource shape. CPU Runs are valid here;
    /// Slurm GPU submission additionally calls `validate` above.
    pub fn validate_compute(&self) -> Result<(), Error> {
        if self.cpus == 0 {
            return Err(Error::Configuration("resources.cpus"));
        }
        if self.nodes != 1 {
            return Err(Error::Configuration("resources.nodes"));
        }
        if self.memory_gb == Some(0) {
            return Err(Error::Configuration("resources.memory_gb"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Storage {
    pub run_dir: String,
    pub project_data_root: String,
    /// Other explicitly frozen roots, e.g. data_root/hf_home/hf_datasets_cache.
    #[serde(default)]
    pub additional_roots: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gres {
    pub gpu_type: String,
    pub count: u32,
}

pub fn parse_gres(value: &str) -> Result<Gres, Error> {
    let parts: Vec<_> = value.split(':').collect();
    if parts.len() != 3
        || parts[0] != "gpu"
        || parts[1].is_empty()
        || !parts[1]
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        || parts[2].starts_with('0')
        || !parts[2].bytes().all(|c| c.is_ascii_digit())
    {
        return Err(Error::Configuration("gres"));
    }
    let count = parts[2].parse().map_err(|_| Error::Configuration("gres"))?;
    if count == 0 {
        return Err(Error::Configuration("gres"));
    }
    Ok(Gres {
        gpu_type: parts[1].to_owned(),
        count,
    })
}

/// Positive HH:MM:SS (the public API) or Slurm's D-HH:MM:SS form.
/// This does not infer a provider limit or silently rewrite the supplied string.
pub fn walltime_seconds(value: &str) -> Result<u64, Error> {
    let invalid = || Error::Configuration("time");
    let (days, rest) = match value.split_once('-') {
        Some((day, rest)) => (number(day).ok_or_else(invalid)?, rest),
        None => (0, value),
    };
    let fields: Vec<_> = rest.split(':').collect();
    if fields.len() != 3 || fields[1].len() != 2 || fields[2].len() != 2 {
        return Err(invalid());
    }
    let hours = number(fields[0]).ok_or_else(invalid)?;
    let minutes = number(fields[1]).ok_or_else(invalid)?;
    let seconds = number(fields[2]).ok_or_else(invalid)?;
    if minutes > 59 || seconds > 59 || (value.contains('-') && hours > 23) {
        return Err(invalid());
    }
    let result = days
        .checked_mul(86400)
        .and_then(|n| hours.checked_mul(3600).and_then(|h| n.checked_add(h)))
        .and_then(|n| n.checked_add(minutes * 60 + seconds))
        .ok_or_else(invalid)?;
    if result == 0 {
        return Err(invalid());
    }
    Ok(result)
}
fn number(value: &str) -> Option<u64> {
    if value.is_empty() || !value.bytes().all(|c| c.is_ascii_digit()) {
        None
    } else {
        value.parse().ok()
    }
}

/// Lexical POSIX validation only. The executor must separately prove mount and
/// symlink scope on the remote filesystem; this function never probes a path.
pub fn absolute_path(value: &str) -> bool {
    value.starts_with('/')
        && printable(value, 4096)
        && !value.contains([':', '\\'])
        && !value.split('/').any(|p| p == "." || p == "..")
}
pub fn path_within(path: &str, root: &str) -> bool {
    if !absolute_path(path) || !absolute_path(root) {
        return false;
    }
    let root = root.trim_end_matches('/');
    root.is_empty() || path == root || path.starts_with(&format!("{root}/"))
}

impl SlurmConfig {
    pub fn validate(
        &self,
        resources: &Resources,
        storage: &Storage,
        image_id: &str,
    ) -> Result<(), Error> {
        if self.kind != "slurm" {
            return Err(Error::Configuration("kind"));
        }
        for (name, value) in [
            ("ssh_alias", &self.ssh_alias),
            ("partition", &self.partition),
            ("account", &self.account),
            ("qos", &self.qos),
        ] {
            if !component(value) {
                return Err(Error::Configuration(name));
            }
        }
        resources.validate()?;
        if parse_gres(&self.gres)?.count != resources.gpus {
            return Err(Error::Configuration("resources.gpus/gres"));
        }
        walltime_seconds(&self.time)?;
        if !(30..=1080).contains(&self.image_stage_timeout_seconds) {
            return Err(Error::Configuration("image_stage_timeout_seconds"));
        }
        if !image_digest(image_id) {
            return Err(Error::Configuration("image_id"));
        }
        if let Some(image) = &self.oci_image {
            let Some((repository, digest)) = image.split_once('@') else {
                return Err(Error::Configuration("oci_image"));
            };
            if repository.is_empty()
                || !repository.as_bytes()[0].is_ascii_alphanumeric()
                || !repository
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._:/-".contains(&c))
                || digest != image_id
                || !digest[7..]
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            {
                return Err(Error::Configuration("oci_image"));
            }
        }
        for (name, path) in [
            ("run_dir", &storage.run_dir),
            ("project_data_root", &storage.project_data_root),
            ("source_dir", &self.source_dir),
            ("sif_path", &self.sif_path),
        ] {
            if !path_within(path, &self.mount_root) {
                return Err(Error::Configuration(name));
            }
        }
        for path in storage.additional_roots.values() {
            if !path_within(path, &self.mount_root) {
                return Err(Error::Configuration("additional_roots"));
            }
        }
        for (name, path) in [
            ("apptainer_cache_dir", &self.apptainer_cache_dir),
            ("apptainer_tmp_dir", &self.apptainer_tmp_dir),
        ] {
            if path.as_deref().is_some_and(|p| !absolute_path(p)) {
                return Err(Error::Configuration(name));
            }
        }
        Ok(())
    }
}
fn image_digest(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|hash| hash.len() == 64 && hash.bytes().all(|c| c.is_ascii_hexdigit()))
}

/// Historical *storage* names are readable; `wyd` is not an executable backend.
pub fn compatible_storage_kind(value: &str) -> bool {
    matches!(value, "slurm" | "wyd")
}

pub fn delivery_requested(markers: &[String], asset_id: &str) -> bool {
    markers
        .iter()
        .any(|value| value == &format!("slurm:{asset_id}") || value == &format!("wyd:{asset_id}"))
}
