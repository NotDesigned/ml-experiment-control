//! Explicit isolated read configuration. Validation is pure; opening files and
//! initializing a fresh preview registry are separate, named effects.
use crate::{
    deletion::RunFence,
    experiments::{ProjectBinding, RunRef},
};
use ml_exp_core::identity::{validate_id, validate_observation_id};
use ml_exp_io::directory::Directory;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    path::{Path, PathBuf},
};

const MARKER: &str = ".rust-preview-workspace.json";
const FORMAT: &str = "ml-expd-rust-preview.v1";
#[derive(Debug, thiserror::Error)]
#[error("invalid isolated experiment read configuration or workspace")]
pub struct Error;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub schema_version: u32,
    pub registry_root: PathBuf,
    pub projects: Vec<Project>,
    #[serde(default)]
    pub source_policy: Option<ml_exp_core::sources::Policy>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Project {
    pub project: String,
    pub project_file: PathBuf,
    pub run_roots: Vec<PathBuf>,
    #[serde(default)]
    pub runs: Vec<Run>,
    /// Omitted with no explicit locators enables canonical bounded discovery.
    #[serde(default)]
    pub discover_runs: Option<bool>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Run {
    pub run_id: String,
    pub root_index: usize,
    pub relative_components: Vec<String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Marker {
    contract: String,
    workspace_id: String,
}

fn absolute(path: &Path) -> bool {
    path.is_absolute()
        && path.components().all(|part| {
            matches!(
                part,
                std::path::Component::RootDir | std::path::Component::Normal(_)
            )
        })
}
impl Project {
    pub fn discovery_enabled(&self) -> bool {
        self.discover_runs.unwrap_or(self.runs.is_empty())
    }
}
impl Settings {
    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > 1024 * 1024 {
            return Err(Error);
        }
        let settings: Self = serde_json::from_slice(bytes).map_err(|_| Error)?;
        settings.validate()?;
        Ok(settings)
    }
    pub fn validate(&self) -> Result<(), Error> {
        if self
            .source_policy
            .as_ref()
            .is_some_and(|policy| policy.validate().is_err())
        {
            return Err(Error);
        }
        if self.schema_version != 1 || !absolute(&self.registry_root) || self.projects.len() > 256 {
            return Err(Error);
        }
        let mut projects = BTreeSet::new();
        let mut run_count = 0usize;
        for project in &self.projects {
            validate_id(&project.project).map_err(|_| Error)?;
            if !projects.insert(&project.project)
                || !absolute(&project.project_file)
                || project.run_roots.is_empty()
                || project.run_roots.len() > 64
                || project.run_roots.iter().any(|p| !absolute(p))
                || project.discovery_enabled() && !project.runs.is_empty()
            {
                return Err(Error);
            }
            let mut runs = BTreeSet::new();
            let mut locations = BTreeSet::new();
            for run in &project.runs {
                run_count = run_count.checked_add(1).ok_or(Error)?;
                validate_observation_id(&run.run_id).map_err(|_| Error)?;
                if run_count > 4096
                    || !runs.insert(&run.run_id)
                    || run.root_index >= project.run_roots.len()
                    || run.relative_components.len() > 32
                    || !locations.insert((run.root_index, &run.relative_components))
                {
                    return Err(Error);
                }
                for part in &run.relative_components {
                    validate_observation_id(part).map_err(|_| Error)?;
                }
            }
        }
        Ok(())
    }
    pub fn bindings(&self) -> Vec<ProjectBinding> {
        self.projects
            .iter()
            .map(|p| ProjectBinding {
                project: p.project.clone(),
                project_file: p.project_file.clone(),
                run_roots: p.run_roots.clone(),
            })
            .collect()
    }
    pub fn runs(&self) -> Vec<RunRef> {
        self.projects
            .iter()
            .flat_map(|p| {
                p.runs.iter().map(|r| RunRef {
                    project: p.project.clone(),
                    run_id: r.run_id.clone(),
                    root_index: r.root_index,
                    relative_components: r.relative_components.clone(),
                })
            })
            .collect()
    }
}

pub fn load(path: &Path) -> Result<Settings, Error> {
    Settings::from_json(&ml_exp_io::read_private(path, 1024 * 1024, true).map_err(|_| Error)?)
}

/// Only an explicit CLI initialization may create a preview registry. Existing
/// targets, including empty directories, are refused; no repair or migration.
pub fn initialize(path: &Path) -> Result<String, Error> {
    if !absolute(path) {
        return Err(Error);
    }
    let parent = Directory::open(path.parent().ok_or(Error)?, false).map_err(|_| Error)?;
    let leaf = path.file_name().ok_or(Error)?;
    if parent.entry_exists(leaf).map_err(|_| Error)? {
        return Err(Error);
    }
    let root = parent.create_new_child(leaf).map_err(|_| Error)?;
    root.verify_private_parent().map_err(|_| Error)?;
    if !root.entry_names().map_err(|_| Error)?.is_empty() {
        return Err(Error);
    }
    let mut random = [0u8; 16];
    getrandom::fill(&mut random).map_err(|_| Error)?;
    let workspace_id = format!("rust-preview-{:032x}", u128::from_be_bytes(random));
    let marker = Marker {
        contract: FORMAT.into(),
        workspace_id: workspace_id.clone(),
    };
    root.publish_new(
        OsStr::new(MARKER),
        &serde_json::to_vec(&marker).map_err(|_| Error)?,
    )
    .map_err(|_| Error)?;
    let fence = RunFence::new(path);
    let guard = fence.shared().map_err(|_| Error)?;
    guard.validate().map_err(|_| Error)?;
    root.verify_private_parent().map_err(|_| Error)?;
    Ok(workspace_id)
}

/// Pin the marker and existing deletion lock. The marker explicitly excludes
/// accidental use of a production/Python registry; ordinary reads create none.
pub fn workspace(path: &Path) -> Result<(Directory, String), Error> {
    let root = Directory::open(path, false).map_err(|_| Error)?;
    root.verify_private_parent().map_err(|_| Error)?;
    let data = root
        .read_regular(OsStr::new(MARKER), 4096)
        .map_err(|_| Error)?;
    let marker: Marker = serde_json::from_slice(&data).map_err(|_| Error)?;
    let suffix = marker
        .workspace_id
        .strip_prefix("rust-preview-")
        .ok_or(Error)?;
    if marker.contract != FORMAT
        || suffix.len() != 32
        || !suffix
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(Error);
    }
    let fence = RunFence::new(path);
    let guard = fence.shared_existing().map_err(|_| Error)?;
    guard.validate().map_err(|_| Error)?;
    root.verify_private_parent().map_err(|_| Error)?;
    Ok((root, marker.workspace_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn canonical_discovery_defaults_without_locators_and_refuses_mixed_sources() {
        let mut value = json!({"schema_version":1,"registry_root":"/isolated/registry","projects":[{"project":"demo","project_file":"/isolated/demo/project.yaml","run_roots":["/isolated/demo/runs"]}]});
        let parse =
            |value: &serde_json::Value| Settings::from_json(&serde_json::to_vec(value).unwrap());
        assert!(parse(&value).unwrap().projects[0].discovery_enabled());
        value["projects"][0]["discover_runs"] = json!(false);
        assert!(!parse(&value).unwrap().projects[0].discovery_enabled());
        value["projects"][0]["runs"] =
            json!([{"run_id":"train","root_index":0,"relative_components":["train"]}]);
        value["projects"][0]
            .as_object_mut()
            .unwrap()
            .remove("discover_runs");
        assert!(!parse(&value).unwrap().projects[0].discovery_enabled());
        value["projects"][0]["discover_runs"] = json!(true);
        assert!(parse(&value).is_err());
    }
    #[test]
    fn pure_config_refuses_ambiguous_unsafe_and_duplicate_bindings() {
        let value = json!({"schema_version":1,"registry_root":"/isolated/registry","projects":[{"project":"demo","project_file":"/isolated/demo/research_project.yaml","run_roots":["/isolated/demo/runs"],"runs":[{"run_id":"train","root_index":0,"relative_components":["train"]}]}]});
        assert!(Settings::from_json(&serde_json::to_vec(&value).unwrap()).is_ok());
        let mut cases = Vec::new();
        let mut v = value.clone();
        v["registry_root"] = json!("relative");
        cases.push(v);
        let mut v = value.clone();
        v["projects"][0]["runs"][0]["relative_components"] = json!([".."]);
        cases.push(v);
        let mut v = value.clone();
        v["projects"][0]["runs"][0]["root_index"] = json!(1);
        cases.push(v);
        let mut v = value.clone();
        v["projects"][0]["runs"]
            .as_array_mut()
            .unwrap()
            .push(value["projects"][0]["runs"][0].clone());
        cases.push(v);
        let mut v = value.clone();
        v["projects"][0]["runs"]
            .as_array_mut()
            .unwrap()
            .push(json!({"run_id":"another","root_index":0,"relative_components":["train"]}));
        cases.push(v);
        for v in cases {
            assert!(Settings::from_json(&serde_json::to_vec(&v).unwrap()).is_err());
        }
    }
    #[test]
    fn initialization_is_explicit_fresh_and_never_repairs_existing_state() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("registry");
        assert!(workspace(&path).is_err());
        assert!(!path.exists());
        let id = initialize(&path).unwrap();
        assert_eq!(workspace(&path).unwrap().1, id);
        assert!(initialize(&path).is_err());
        std::fs::remove_file(path.join(".run-deletion.lock")).unwrap();
        assert!(workspace(&path).is_err());
        assert!(!path.join(".run-deletion.lock").exists());
        let empty = temp.path().join("empty");
        std::fs::create_dir(&empty).unwrap();
        assert!(initialize(&empty).is_err());
        assert_eq!(std::fs::read_dir(empty).unwrap().count(), 0);
    }
}
