//! Workspace project registration. Reducers consume explicit path observations
//! and time; they never resolve files, read configuration, index or run a job.
mod definition;
use crate::{identity, observation::python_equal};
pub use definition::{
    ProjectDefinition, ProjectLayout, ProjectPaths, project_layout, project_paths,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LifecycleState {
    #[default]
    Active,
    Paused,
    Archived,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RegistrationSource {
    ConfigSeed,
    LegacyOnboarding,
    Onboarding,
    Manual,
}

/// Paths and reasons may contain private operator data; do not Debug-log them.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleRecord {
    pub project: String,
    pub project_file: String,
    #[serde(default)]
    pub state: LifecycleState,
    pub source: RegistrationSource,
    pub registered_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub state_reason: String,
}
impl LifecycleRecord {
    pub fn from_value(value: &Value) -> Result<Self> {
        let record: Self = serde_json::from_value(value.clone()).map_err(|_| Error::Record)?;
        identity::validate_id(&record.project).map_err(|_| Error::Record)?;
        Ok(record)
    }
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("project record contains only JSON scalars")
    }
}

/// Trusted filesystem-shell observations. `provided` binds the exact stored or
/// requested string; expansion and canonicalization happened outside this pure
/// module, while holding the shell's approved filesystem boundaries.
#[derive(Clone, PartialEq, Eq)]
pub struct PathEvidence {
    pub provided: String,
    pub expanded_absolute: String,
    pub canonical_absolute: String,
}
impl PathEvidence {
    pub fn validate(&self) -> Result<()> {
        if self.provided.contains('\0')
            || !self.expanded_absolute.starts_with('/')
            || self.expanded_absolute.contains('\0')
            || !(self.provided.starts_with('/') || self.provided.starts_with('~'))
            || !canonical_absolute(&self.canonical_absolute)
        {
            return Err(Error::Path);
        }
        Ok(())
    }
}
fn canonical_absolute(value: &str) -> bool {
    value == "/"
        || value.starts_with('/')
            && !value.contains('\0')
            && value[1..]
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != "..")
}
pub struct Seed {
    pub project: String,
    pub path: PathEvidence,
}
pub struct Commit {
    pub payload: Value,
    pub event: Value,
}
pub struct Change<T> {
    pub result: T,
    pub commit: Option<Commit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("project registry must be a mapping with schema version 1")]
    Schema,
    #[error("project registry projects must be an array")]
    Projects,
    #[error("invalid project lifecycle record")]
    Record,
    #[error("project registry contains a duplicate project name")]
    DuplicateProject,
    #[error("project manifest path must have explicit absolute canonical evidence")]
    Path,
    #[error("project manifest path evidence is missing or contradictory")]
    PathEvidence,
    #[error("project registry contains a duplicate canonical manifest path")]
    DuplicateManifest,
    #[error("project manifest is already registered with a different project identity")]
    ManifestIdentityConflict,
    #[error("project is already registered from a different canonical manifest path")]
    ProjectPathConflict,
    #[error("archived project must be restored before reactivation")]
    Archived,
    #[error("unknown registered project")]
    UnknownProject,
    #[error("project lifecycle transition is unavailable from the current state")]
    Transition,
    #[error("unknown project lifecycle action")]
    Action,
    #[error("archiving a project requires a reason")]
    Reason,
    #[error("authored project definition is invalid or unsupported")]
    Definition,
}
pub type Result<T> = std::result::Result<T, Error>;

fn record_values(payload: &Value) -> Result<Vec<LifecycleRecord>> {
    let fields = payload.as_object().ok_or(Error::Schema)?;
    if fields
        .get("schema_version")
        .is_some_and(|v| !python_equal(v, &json!(1)))
    {
        return Err(Error::Schema);
    }
    let empty = Vec::new();
    let values = match fields.get("projects") {
        None => &empty,
        Some(Value::Array(values)) => values,
        _ => return Err(Error::Projects),
    };
    let mut seen = BTreeSet::new();
    values
        .iter()
        .map(|value| {
            let record = LifecycleRecord::from_value(value)?;
            if !seen.insert(record.project.clone()) {
                return Err(Error::DuplicateProject);
            }
            Ok(record)
        })
        .collect()
}

#[derive(Clone)]
pub struct Registry {
    payload: Value,
    records: Vec<LifecycleRecord>,
    canonical_paths: Vec<String>,
}
impl Default for Registry {
    fn default() -> Self {
        Self::empty()
    }
}
impl Registry {
    pub fn empty() -> Self {
        Self {
            payload: json!({"schema_version":1,"projects":[]}),
            records: vec![],
            canonical_paths: vec![],
        }
    }
    /// Parse record shape before the shell resolves any path. This does not
    /// authorize those paths or replace the shell's approved-root policy.
    pub fn required_paths(payload: &Value) -> Result<Vec<String>> {
        Ok(record_values(payload)?
            .into_iter()
            .map(|r| r.project_file)
            .collect())
    }
    pub fn from_value(payload: &Value, paths: &[PathEvidence]) -> Result<Self> {
        let records = record_values(payload)?;
        let mut evidence = BTreeMap::new();
        for path in paths {
            path.validate()?;
            if evidence
                .insert(path.provided.as_str(), path)
                .is_some_and(|previous| previous != path)
            {
                return Err(Error::PathEvidence);
            }
        }
        let mut seen = BTreeSet::new();
        let mut canonical_paths = Vec::new();
        for record in &records {
            let path = evidence
                .get(record.project_file.as_str())
                .ok_or(Error::PathEvidence)?;
            if !seen.insert(path.canonical_absolute.clone()) {
                return Err(Error::DuplicateManifest);
            }
            canonical_paths.push(path.canonical_absolute.clone());
        }
        Ok(Self {
            payload: payload.clone(),
            records,
            canonical_paths,
        })
    }
    pub fn records(&self) -> &[LifecycleRecord] {
        &self.records
    }
    pub fn active_records(&self) -> Vec<&LifecycleRecord> {
        self.records
            .iter()
            .filter(|r| r.state == LifecycleState::Active)
            .collect()
    }
    pub fn payload(&self) -> &Value {
        &self.payload
    }
    fn change<T>(
        &self,
        records: &[LifecycleRecord],
        result: T,
        event: &str,
        project: &str,
        at: &str,
        detail: Value,
    ) -> Change<T> {
        let mut payload = self
            .payload
            .as_object()
            .expect("validated registry mapping")
            .clone();
        payload.insert(
            "projects".into(),
            Value::Array(records.iter().map(LifecycleRecord::to_value).collect()),
        );
        Change {
            result,
            commit: Some(Commit {
                payload: Value::Object(payload),
                event: json!({"event":event,"project":project,"at":at,"detail":detail}),
            }),
        }
    }
    pub fn register(
        &self,
        project: &str,
        path: &PathEvidence,
        source: RegistrationSource,
        now: &str,
    ) -> Result<Change<LifecycleRecord>> {
        identity::validate_id(project).map_err(|_| Error::Record)?;
        path.validate()?;
        if self
            .records
            .iter()
            .zip(&self.canonical_paths)
            .any(|(record, canonical)| {
                *canonical == path.canonical_absolute && record.project != project
            })
        {
            return Err(Error::ManifestIdentityConflict);
        }
        let mut records = self.records.clone();
        let record = if let Some(index) = records.iter().position(|r| r.project == project) {
            if self.canonical_paths[index] != path.canonical_absolute {
                return Err(Error::ProjectPathConflict);
            }
            if records[index].state == LifecycleState::Archived {
                return Err(Error::Archived);
            }
            if records[index].state == LifecycleState::Active {
                return Ok(Change {
                    result: records[index].clone(),
                    commit: None,
                });
            }
            records[index].state = LifecycleState::Active;
            records[index].updated_at = now.into();
            records[index].state_reason = "registered".into();
            records[index].clone()
        } else {
            let record = LifecycleRecord {
                project: project.into(),
                project_file: path.canonical_absolute.clone(),
                state: LifecycleState::Active,
                source,
                registered_at: now.into(),
                updated_at: now.into(),
                state_reason: "registered".into(),
            };
            records.push(record.clone());
            record
        };
        let detail = record.to_value();
        Ok(self.change(&records, record, "REGISTER", project, now, detail))
    }
    pub fn transition(
        &self,
        project: &str,
        target: LifecycleState,
        reason: &str,
        now: &str,
    ) -> Result<Change<LifecycleRecord>> {
        let mut records = self.records.clone();
        let index = records
            .iter()
            .position(|r| r.project == project)
            .ok_or(Error::UnknownProject)?;
        let previous = records[index].state;
        if previous == target {
            return Ok(Change {
                result: records[index].clone(),
                commit: None,
            });
        }
        let allowed = match previous {
            LifecycleState::Active => {
                matches!(target, LifecycleState::Paused | LifecycleState::Archived)
            }
            LifecycleState::Paused => {
                matches!(target, LifecycleState::Active | LifecycleState::Archived)
            }
            LifecycleState::Archived => target == LifecycleState::Paused,
        };
        if !allowed {
            return Err(Error::Transition);
        }
        records[index].state = target;
        records[index].updated_at = now.into();
        records[index].state_reason = reason.into();
        let event = serde_json::to_value(target).expect("enum JSON");
        Ok(self.change(
            &records,
            records[index].clone(),
            event.as_str().expect("enum string"),
            project,
            now,
            json!({"from":previous,"reason":reason}),
        ))
    }
    pub fn action(
        &self,
        project: &str,
        action: &str,
        reason: &str,
        now: &str,
    ) -> Result<Change<LifecycleRecord>> {
        let current = self
            .records
            .iter()
            .find(|r| r.project == project)
            .ok_or(Error::UnknownProject)?;
        let (allowed, target) = match action {
            "pause" => (
                current.state == LifecycleState::Active,
                LifecycleState::Paused,
            ),
            "resume" => (
                current.state == LifecycleState::Paused,
                LifecycleState::Active,
            ),
            "archive" => (
                matches!(
                    current.state,
                    LifecycleState::Active | LifecycleState::Paused
                ),
                LifecycleState::Archived,
            ),
            "restore" => (
                current.state == LifecycleState::Archived,
                LifecycleState::Paused,
            ),
            _ => return Err(Error::Action),
        };
        if !allowed {
            return Err(Error::Transition);
        }
        if target == LifecycleState::Archived
            && reason
                .trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'))
                .is_empty()
        {
            return Err(Error::Reason);
        }
        self.transition(project, target, reason, now)
    }
    pub fn unregister(
        &self,
        project: &str,
        reason: &str,
        now: &str,
    ) -> Result<Change<LifecycleRecord>> {
        let record = self
            .records
            .iter()
            .find(|r| r.project == project)
            .ok_or(Error::UnknownProject)?
            .clone();
        let records = self
            .records
            .iter()
            .filter(|r| r.project != project)
            .cloned()
            .collect::<Vec<_>>();
        let detail =
            json!({"from":record.state,"project_file":record.project_file,"reason":reason});
        Ok(self.change(&records, record, "UNREGISTER", project, now, detail))
    }
    pub fn unregister_all(&self, reason: &str, now: &str) -> Result<Change<Vec<LifecycleRecord>>> {
        if self.records.is_empty() {
            return Ok(Change {
                result: vec![],
                commit: None,
            });
        }
        let detail = json!({"reason":if reason.is_empty(){"unregister_all"}else{reason},"unregistered":self.records.iter()
            .map(|r|json!({"project":r.project,"from":r.state,"project_file":r.project_file})).collect::<Vec<_>>()});
        Ok(self.change(&[], self.records.clone(), "UNREGISTER", "", now, detail))
    }
}

/// This is only a first-creation decision. An existing ledger wins even if the
/// seed configuration changed or projects were explicitly unregistered.
pub fn bootstrap(
    existing: Option<&Registry>,
    seeds: &[Seed],
    now: &str,
) -> Result<Change<Vec<LifecycleRecord>>> {
    if let Some(existing) = existing {
        return Ok(Change {
            result: existing.records.clone(),
            commit: None,
        });
    }
    let mut records = Vec::new();
    let mut paths = BTreeMap::new();
    let mut names = BTreeSet::new();
    for seed in seeds {
        seed.path.validate()?;
        identity::validate_id(&seed.project).map_err(|_| Error::Record)?;
        if let Some(previous) =
            paths.insert(seed.path.canonical_absolute.as_str(), seed.project.as_str())
        {
            if previous != seed.project {
                return Err(Error::ManifestIdentityConflict);
            }
            continue;
        }
        if !names.insert(seed.project.as_str()) {
            return Err(Error::DuplicateProject);
        }
        records.push(LifecycleRecord {
            project: seed.project.clone(),
            project_file: seed.path.canonical_absolute.clone(),
            state: LifecycleState::Active,
            source: RegistrationSource::ConfigSeed,
            registered_at: now.into(),
            updated_at: now.into(),
            state_reason: String::new(),
        });
    }
    let detail =
        json!({"imported":records.iter().map(LifecycleRecord::to_value).collect::<Vec<_>>()});
    Ok(Registry::empty().change(&records, records.clone(), "BOOTSTRAP", "", now, detail))
}
