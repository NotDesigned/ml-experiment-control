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
        settings.validate().map_err(|_| Error::Invalid)?;
        let (_, workspace_id) =
            read_config::workspace(&settings.registry_root).map_err(|_| Error::Invalid)?;
        let evidence =
            experiments::Store::open(settings.bindings(), RunFence::new(&settings.registry_root))
                .map_err(|_| Error::Invalid)?;
        let mut seeds = Vec::new();
        let mut observed = Vec::new();
        for project in &settings.projects {
            let document = evidence
                .project(&project.project, None)
                .map_err(|_| Error::Invalid)?;
            let definition = ProjectDefinition::from_value(&Value::Object(document.metadata))?;
            if definition.campaign_files().next().is_some()
                || !definition.to_value()["campaigns"]
                    .as_array()
                    .is_some_and(Vec::is_empty)
            {
                return Err(Error::UnsupportedDefinition);
            }
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
                .map_err(|_| Error::EffectUncertain)?;
        }
        let (_, after) =
            read_config::workspace(&settings.registry_root).map_err(|_| Error::EffectUncertain)?;
        if after != workspace_id {
            return Err(Error::EffectUncertain);
        }
        Ok(())
    }
    pub fn open(settings: &read_config::Settings) -> Result<Self, Error> {
        settings.validate().map_err(|_| Error::Invalid)?;
        let (_, workspace_id) =
            read_config::workspace(&settings.registry_root).map_err(|_| Error::Invalid)?;
        let service = Self {
            workspace_root: settings.registry_root.clone(),
            workspace_id,
            approved: settings
                .projects
                .iter()
                .map(|p| (p.project.clone(), p.project_file.clone()))
                .collect(),
            evidence: experiments::Store::open(
                settings.bindings(),
                RunFence::new(&settings.registry_root),
            )
            .map_err(|_| Error::Invalid)?,
            store: Store::open(&settings.registry_root.join("projects"))?,
        };
        service.snapshot()?;
        Ok(service)
    }
    fn validate(&self) -> Result<(), Error> {
        let (_, workspace) =
            read_config::workspace(&self.workspace_root).map_err(|_| Error::Invalid)?;
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
            if self
                .approved
                .get(&record.project)
                .is_some_and(|path| path.to_str() != Some(record.project_file.as_str()))
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
            .filter(|p| p.state == LifecycleState::Active && self.approved.contains_key(&p.project))
            .map(|p| p.project)
            .collect())
    }
    fn validated_project(&self, project: &str) -> Result<experiments::Document, Error> {
        if !self.approved.contains_key(project) {
            return Err(Error::Invalid);
        }
        let document = self
            .evidence
            .project(project, None)
            .map_err(|_| Error::Invalid)?;
        let definition = ProjectDefinition::from_value(&Value::Object(document.metadata.clone()))?;
        if definition.project() != project {
            return Err(Error::Invalid);
        }
        if definition.campaign_files().next().is_some()
            || !definition.to_value()["campaigns"]
                .as_array()
                .is_some_and(Vec::is_empty)
        {
            return Err(Error::UnsupportedDefinition);
        }
        Ok(document)
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
            Some(&self.approved),
        )?;
        // Registry identity binds name and canonical source path. A changed
        // authored manifest may be refreshed later; this particular call must
        // not silently race a different Project identity/content observation.
        self.evidence
            .project(project, Some(&document.raw_sha256))
            .map_err(|_| Error::EffectUncertain)?;
        self.validate().map_err(|_| Error::EffectUncertain)?;
        Ok(experiment_redact::redact_json(json!({"project":result,
            "initial_index":{"status":"NOT_IMPLEMENTED","runs":null,"error":null,"unavailable_run_roots":[]},
            "effect":"Project registration is durable; native collection and indexing are not enabled"})))
    }
    pub fn execute(
        &self,
        operation: Operation,
        now: &str,
        identity: &str,
        revision: Option<u64>,
    ) -> Result<Value, Error> {
        self.validate()?;
        if matches!(&operation, Operation::Register { .. }) {
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
        let result =
            self.store
                .execute(&operation, now, identity, revision, Some(&self.approved))?;
        if let Some((project, document)) = resumed {
            self.evidence
                .project(&project, Some(&document.raw_sha256))
                .map_err(|_| Error::EffectUncertain)?;
        }
        self.validate().map_err(|_| Error::EffectUncertain)?;
        let body = match operation {
            Operation::Action { .. } => {
                json!({"active":result["state"]=="ACTIVE","project":result,"effect":"Project tracking state changed; research files and jobs are unchanged"})
            }
            Operation::Unregister { .. } | Operation::UnregisterAll { .. } => {
                json!({"unregistered":result,"effect":"Daemon tracking was removed; research files, artifacts and jobs are unchanged"})
            }
            Operation::Register { .. } => return Err(Error::Invalid),
        };
        Ok(experiment_redact::redact_json(body))
    }
}

/// Named effect boundary; reducers receive these values and never acquire them.
pub fn transition_context() -> Result<(String, String), Error> {
    let now = time::OffsetDateTime::from(std::time::SystemTime::now())
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| Error::Invalid)?;
    let mut random = [0u8; 16];
    getrandom::fill(&mut random).map_err(|_| Error::Invalid)?;
    Ok((now, format!("{:032x}", u128::from_be_bytes(random))))
}
