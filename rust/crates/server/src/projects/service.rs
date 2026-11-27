//! Explicit filesystem application boundary for the project lifecycle.
use super::{Error, Operation, Snapshot, Store, canonical_path};
use crate::{deletion::RunFence, experiments, read_config};
use ml_exp_core::projects::{LifecycleState, ProjectDefinition, Seed};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

pub struct Service {
    workspace_root: PathBuf,
    workspace_id: String,
    projects_root: Option<PathBuf>,
    approved: BTreeMap<String, PathBuf>,
    evidence: experiments::Store,
    store: Store,
}
impl Service {
    pub fn initialize(
        settings: &read_config::Settings,
        now: &str,
        identity: &str,
    ) -> Result<(), Error> {
        settings.validate().map_err(why!(Error::Invalid))?;
        let (_, workspace_id) =
            read_config::workspace(&settings.registry_root).map_err(why!(Error::Invalid))?;
        let evidence = settings
            .open_store(RunFence::new(&settings.registry_root))
            .map_err(why!(Error::Invalid))?;
        let mut seeds = Vec::new();
        let mut observed = Vec::new();
        for project in &settings.projects {
            let document = evidence
                .project(&project.project, None)
                .map_err(why!(Error::Invalid))?;
            let definition = ProjectDefinition::from_value(&Value::Object(document.metadata))?;
            if definition.project() != project.project {
                return Err(Error::Invalid);
            }
            seeds.push(Seed {
                project: project.project.clone(),
                path: canonical_path(project.project_file.to_str().ok_or(Error::Invalid)?)?,
            });
            observed.push((project.project.clone(), document.raw_sha256));
        }
        Store::initialize(
            &settings.registry_root.join("projects"),
            &seeds,
            now,
            identity,
        )?;
        for (project, digest) in observed {
            evidence
                .project(&project, Some(&digest))
                .map_err(why!(Error::EffectUncertain))?;
        }
        let (_, after) = read_config::workspace(&settings.registry_root)
            .map_err(why!(Error::EffectUncertain))?;
        if after != workspace_id {
            return Err(Error::EffectUncertain);
        }
        Ok(())
    }
    pub fn open(settings: &read_config::Settings) -> Result<Self, Error> {
        settings.validate().map_err(why!(Error::Invalid))?;
        let (_, workspace_id) =
            read_config::workspace(&settings.registry_root).map_err(why!(Error::Invalid))?;
        let service = Self {
            workspace_root: settings.registry_root.clone(),
            workspace_id,
            projects_root: settings.projects_root.clone(),
            approved: settings
                .projects
                .iter()
                .map(|p| (p.project.clone(), p.project_file.clone()))
                .collect(),
            evidence: settings
                .open_store(RunFence::new(&settings.registry_root))
                .map_err(why!(Error::Invalid))?,
            store: Store::open(&settings.registry_root.join("projects"))?,
        };
        service.snapshot()?;
        Ok(service)
    }
    pub fn supports_creation(&self) -> bool {
        self.projects_root.is_some()
    }
    fn is_approved_record(&self, project: &str, project_file: &str) -> bool {
        if let Some(path) = self.approved.get(project) {
            path.to_str() == Some(project_file)
        } else if let Some(root) = &self.projects_root {
            root.join(project).join("project.json").to_str() == Some(project_file)
        } else {
            false
        }
    }
    fn validate(&self) -> Result<(), Error> {
        let (_, workspace) =
            read_config::workspace(&self.workspace_root).map_err(why!(Error::Invalid))?;
        if workspace != self.workspace_id {
            return Err(Error::Invalid);
        }
        Ok(())
    }
    pub fn matches_configuration(
        &self,
        root: &std::path::Path,
        projects: &BTreeMap<String, PathBuf>,
        workspace_id: &str,
    ) -> bool {
        self.workspace_root == root
            && &self.approved == projects
            && self.workspace_id == workspace_id
    }
    pub fn snapshot(&self) -> Result<Snapshot, Error> {
        self.validate()?;
        let snapshot = self.store.snapshot(200)?;
        for record in &snapshot.projects {
            if (self.approved.contains_key(&record.project) || self.projects_root.is_some())
                && !self.is_approved_record(&record.project, &record.project_file)
            {
                return Err(Error::Binding);
            }
        }
        self.validate()?;
        Ok(snapshot)
    }
    pub fn list(&self) -> Result<Value, Error> {
        let s = self.snapshot()?;
        Ok(experiment_redact::redact_json(
            json!({"projects":s.projects,"events":s.events,
            "durability":{"revision":s.revision,"journal_pending":s.journal_pending}}),
        ))
    }
    pub fn active_projects(&self) -> Result<BTreeSet<String>, Error> {
        Ok(self
            .snapshot()?
            .projects
            .into_iter()
            .filter(|p| {
                p.state == LifecycleState::Active
                    && self.is_approved_record(&p.project, &p.project_file)
            })
            .map(|p| p.project)
            .collect())
    }
    fn validated_project(&self, project: &str) -> Result<experiments::Document, Error> {
        if !self.approved.contains_key(project) && self.projects_root.is_none() {
            return Err(Error::Invalid);
        }
        let document = self
            .evidence
            .project(project, None)
            .map_err(why!(Error::Invalid))?;
        let definition = ProjectDefinition::from_value(&Value::Object(document.metadata.clone()))?;
        if definition.project() != project {
            return Err(Error::Invalid);
        }
        Ok(document)
    }
    pub fn create_project(
        &self,
        project: &str,
        title: Option<&str>,
        now: &str,
        identity: &str,
    ) -> std::result::Result<Value, crate::error::Error> {
        use ml_exp_io::directory::Directory;
        use std::ffi::OsStr;
        let bad = || crate::error::Error {
            status: 422,
            code: "INVALID_PROJECT_CREATE_REQUEST",
            detail: "invalid project name or title",
        };
        ml_exp_core::identity::validate_id(project).map_err(|_| bad())?;
        let title = title.unwrap_or(project);
        let manifest = json!({
            "schema_version": 1,
            "project": project,
            "title": title,
            "run_roots": ["runs"],
        });
        ProjectDefinition::from_value(&manifest).map_err(|_| bad())?;
        let snapshot = self.snapshot()?;
        if let Some(existing) = snapshot.projects.iter().find(|p| p.project == project) {
            if existing.state != LifecycleState::Active {
                return Err(crate::error::Error {
                    status: 409,
                    code: "PROJECT_INACTIVE",
                    detail: "project already exists and is paused or archived",
                });
            }
            self.validated_project(project)
                .map_err(crate::error::Error::from)?;
            return Ok(experiment_redact::redact_json(
                json!({"created": false, "project": existing}),
            ));
        }
        let Some(root_path) = &self.projects_root else {
            return Err(crate::error::Error {
                status: 501,
                code: "PROJECT_CREATION_NOT_CONFIGURED",
                detail: "managed project creation is not configured on this server",
            });
        };
        if self.approved.contains_key(project) {
            return Err(crate::error::Error {
                status: 409,
                code: "PROJECT_PATH_CONFIGURED",
                detail: "project is preconfigured at a fixed daemon path; register it explicitly",
            });
        }
        if snapshot.projects.len() >= 256 {
            return Err(crate::error::Error {
                status: 409,
                code: "PROJECT_LIMIT_REACHED",
                detail: "workspace has reached its maximum number of projects",
            });
        }
        let io_err = || crate::error::Error {
            status: 500,
            code: "PROJECT_CREATE_UNCERTAIN",
            detail: "project directory may have been created; query /api/projects before retrying",
        };
        let root = Directory::open(root_path, false).map_err(why!(io_err()))?;
        root.require_private().map_err(why!(io_err()))?;
        let dir = if root
            .entry_exists(OsStr::new(project))
            .map_err(why!(io_err()))?
        {
            root.child(OsStr::new(project), false)
                .map_err(why!(io_err()))?
        } else {
            root.create_new_child(OsStr::new(project))
                .map_err(why!(io_err()))?
        };
        if !dir
            .entry_exists(OsStr::new("runs"))
            .map_err(why!(io_err()))?
        {
            dir.create_new_child(OsStr::new("runs"))
                .map_err(why!(io_err()))?;
        } else {
            dir.child(OsStr::new("runs"), false)
                .map_err(why!(io_err()))?;
        }
        if !dir
            .entry_exists(OsStr::new("project.json"))
            .map_err(why!(io_err()))?
        {
            let mut bytes = ml_exp_core::canonical::bytes(&manifest).map_err(|_| bad())?;
            bytes.push(b'\n');
            dir.publish_new(OsStr::new("project.json"), &bytes)
                .map_err(why!(io_err()))?;
        }
        let document = self.validated_project(project)?;
        let path_file = root_path.join(project).join("project.json");
        let path = canonical_path(path_file.to_str().ok_or(Error::Invalid)?)?;
        let result = self.store.execute(
            &Operation::Onboard {
                project: project.into(),
                path,
            },
            now,
            identity,
            None,
            Some((&self.approved, self.projects_root.as_deref())),
        )?;
        self.evidence
            .project(project, Some(&document.raw_sha256))
            .map_err(why!(Error::EffectUncertain))?;
        self.validate().map_err(why!(Error::EffectUncertain))?;
        Ok(experiment_redact::redact_json(
            json!({"created": true, "project": result}),
        ))
    }
    pub fn register(
        &self,
        requested_path: &str,
        now: &str,
        identity: &str,
        revision: Option<u64>,
    ) -> Result<Value, Error> {
        self.validate()?;
        let (project, path) = self
            .approved
            .iter()
            .find(|(_, p)| p.to_str() == Some(requested_path))
            .ok_or(Error::Invalid)?;
        let document = self.validated_project(project)?;
        let result = self.store.execute(
            &Operation::Register {
                project: project.clone(),
                path: canonical_path(path.to_str().ok_or(Error::Invalid)?)?,
            },
            now,
            identity,
            revision,
            Some((&self.approved, self.projects_root.as_deref())),
        )?;
        // Registry identity binds name and canonical source path. A changed
        // authored manifest may be refreshed later; this particular call must
        // not silently race a different Project identity/content observation.
        self.evidence
            .project(project, Some(&document.raw_sha256))
            .map_err(why!(Error::EffectUncertain))?;
        self.validate().map_err(why!(Error::EffectUncertain))?;
        Ok(experiment_redact::redact_json(json!({"project":result,
            "discovery":{"mode":"ON_DEMAND","source":"approved_run_roots"},
            "effect":"Project registration is durable; Run queries discover authoritative metadata in approved roots"})))
    }
    pub fn execute(
        &self,
        operation: Operation,
        now: &str,
        identity: &str,
        revision: Option<u64>,
    ) -> Result<Value, Error> {
        self.validate()?;
        if matches!(
            &operation,
            Operation::Register { .. } | Operation::Onboard { .. }
        ) {
            return Err(Error::Invalid);
        }
        // Returning to active tracking requires fresh approved authored input.
        // Pause/archive/removal still work when the research source is missing.
        let resumed = match &operation {
            Operation::Action {
                project, action, ..
            } if action == "resume" => Some((project.clone(), self.validated_project(project)?)),
            _ => None,
        };
        let result = self.store.execute(
            &operation,
            now,
            identity,
            revision,
            Some((&self.approved, self.projects_root.as_deref())),
        )?;
        if let Some((project, document)) = resumed {
            self.evidence
                .project(&project, Some(&document.raw_sha256))
                .map_err(why!(Error::EffectUncertain))?;
        }
        self.validate().map_err(why!(Error::EffectUncertain))?;
        let body = match operation {
            Operation::Action { .. } => {
                json!({"active":result["state"]=="ACTIVE","project":result,"effect":"Project tracking state changed; research files and jobs are unchanged"})
            }
            Operation::Unregister { .. } | Operation::UnregisterAll { .. } => {
                json!({"unregistered":result,"effect":"Daemon tracking was removed; research files, artifacts and jobs are unchanged"})
            }
            Operation::Register { .. } | Operation::Onboard { .. } => return Err(Error::Invalid),
        };
        Ok(experiment_redact::redact_json(body))
    }
}

/// Named effect boundary; reducers receive these values and never acquire them.
pub fn transition_context() -> Result<(String, String), Error> {
    let now = time::OffsetDateTime::from(std::time::SystemTime::now())
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(why!(Error::Invalid))?;
    let mut random = [0u8; 16];
    getrandom::fill(&mut random).map_err(why!(Error::Invalid))?;
    Ok((now, format!("{:032x}", u128::from_be_bytes(random))))
}
