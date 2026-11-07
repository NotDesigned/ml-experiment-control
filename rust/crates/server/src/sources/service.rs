//! Approved-project authorization around the immutable source store.
use super::{Policy, Store};
use crate::{deletion::RunFence, experiments, projects, read_config};
use ml_exp_core::sources;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{Read, Seek},
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct Service {
    root: PathBuf,
    workspace_id: String,
    approved: BTreeMap<String, PathBuf>,
    lifecycle: Option<Arc<projects::Service>>,
    evidence: experiments::Store,
    store: Store,
    policy: Policy,
}
fn invalid() -> crate::error::Error {
    crate::error::Error {
        status: 409,
        code: "SOURCE_AUTHORITY_UNAVAILABLE",
        detail: "approved Project or isolated source workspace cannot be validated",
    }
}
fn uncertain() -> crate::error::Error {
    crate::error::Error {
        status: 500,
        code: "SOURCE_PUBLICATION_UNCERTAIN",
        detail: "source may have been published; query its archive SHA receipt before deciding whether to retry",
    }
}
fn unavailable() -> crate::error::Error {
    crate::error::Error {
        status: 409,
        code: "SOURCE_STORAGE_UNAVAILABLE",
        detail: "immutable source evidence is unavailable, invalid, changed or beyond its configured bounds",
    }
}
impl Service {
    pub fn initialize(settings: &read_config::Settings) -> Result<(), super::Error> {
        Store::initialize(
            &settings.registry_root,
            settings.source_policy.clone().unwrap_or_default(),
        )?;
        Ok(())
    }
    pub fn open(
        settings: &read_config::Settings,
        lifecycle: Option<Arc<projects::Service>>,
    ) -> Result<Self, read_config::Error> {
        settings.validate()?;
        let (_, workspace_id) = read_config::workspace(&settings.registry_root)?;
        let policy = settings.source_policy.clone().unwrap_or_default();
        let store = Store::open(&settings.registry_root, policy.clone())
            .map_err(why!(read_config::Error))?;
        Ok(Self {
            root: settings.registry_root.clone(),
            workspace_id,
            approved: settings
                .projects
                .iter()
                .map(|p| (p.project.clone(), p.project_file.clone()))
                .collect(),
            lifecycle,
            evidence: experiments::Store::open(
                settings.bindings(),
                RunFence::new(&settings.registry_root),
            )
            .map_err(why!(read_config::Error))?,
            store,
            policy,
        })
    }
    pub fn matches_configuration(
        &self,
        root: &Path,
        files: &BTreeMap<String, PathBuf>,
        workspace: &str,
        lifecycle: Option<&Arc<projects::Service>>,
    ) -> bool {
        self.root == root
            && &self.approved == files
            && self.workspace_id == workspace
            && match (&self.lifecycle, lifecycle) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
    fn scope(&self, project: &str) -> Result<(), crate::error::Error> {
        ml_exp_core::identity::validate_id(project).map_err(why!(invalid()))?;
        if !self.approved.contains_key(project) {
            return Err(crate::error::Error {
                status: 404,
                code: "SOURCE_PROJECT_NOT_APPROVED",
                detail: "source import requires a Project already approved by the isolated server configuration",
            });
        }
        let (_, workspace) = read_config::workspace(&self.root).map_err(why!(invalid()))?;
        if workspace != self.workspace_id {
            return Err(invalid());
        }
        Ok(())
    }
    fn approval(&self, project: &str) -> Result<String, crate::error::Error> {
        self.scope(project)?;
        if let Some(lifecycle) = &self.lifecycle {
            if !lifecycle
                .snapshot()
                .map_err(crate::error::Error::from)?
                .projects
                .iter()
                .any(|record| record.project == project)
            {
                return Err(crate::error::Error {
                    status: 409,
                    code: "SOURCE_PROJECT_NOT_REGISTERED",
                    detail: "register the approved Project before importing source; automatic Project creation is not yet native",
                });
            }
        }
        let document = self
            .evidence
            .project(project, None)
            .map_err(why!(invalid()))?;
        let definition =
            ml_exp_core::projects::ProjectDefinition::from_value(&Value::Object(document.metadata))
                .map_err(why!(invalid()))?;
        if definition.project() != project {
            return Err(invalid());
        }
        Ok(document.raw_sha256)
    }
    /// This check precedes receiving archive bytes and performs no publication.
    pub fn authorize_upload(&self, project: &str, sha256: &str) -> Result<(), crate::error::Error> {
        sources::validate_archive_digest(sha256).map_err(why!(invalid()))?;
        self.approval(project)?;
        Ok(())
    }
    pub fn maximum_archive_bytes(&self) -> u64 {
        self.policy.max_archive_bytes
    }
    pub fn limits(&self) -> Value {
        json!({"archive_bytes":self.policy.max_archive_bytes,"expanded_bytes":self.policy.max_source_bytes,"entries":self.policy.max_source_files})
    }
    pub fn publish<R: Read + Seek>(
        &self,
        project: &str,
        stream: &mut R,
        sha256: &str,
        now: &str,
    ) -> Result<Value, crate::error::Error> {
        let observed = self.approval(project)?;
        let metadata = self
            .store
            .import_archive(project, stream, sha256, now)
            .map_err(store_error)?;
        self.scope(project).map_err(why!(uncertain()))?;
        self.evidence
            .project(project, Some(&observed))
            .map_err(why!(uncertain()))?;
        if !matches!(self.approval(project),Ok(digest) if digest==observed) {
            return Err(uncertain());
        }
        sources::response(
            &metadata,
            &json!({"kind":"archive","sha256":sha256}),
            json!({"existing":true}),
            &self.policy,
        )
        .map_err(why!(uncertain()))
    }
    pub fn read(&self, project: &str, source_id: &str) -> Result<Value, crate::error::Error> {
        self.scope(project)?;
        sources::validate_source_id(source_id).map_err(why!(invalid()))?;
        let metadata = self.store.read(project, source_id).map_err(store_error)?;
        self.scope(project)?;
        Ok(metadata.to_value())
    }
    fn code<T>(
        &self,
        project: &str,
        source_id: &str,
        read: impl FnOnce(&Store) -> Result<T, super::Error>,
    ) -> Result<T, crate::error::Error> {
        self.scope(project)?;
        sources::validate_source_id(source_id).map_err(why!(invalid()))?;
        let value = read(&self.store).map_err(|error| match error {
            super::Error::Limit => crate::error::Error {
                status: 422,
                code: "CODE_SOURCE_TOO_LARGE",
                detail: "the code exceeds 64 MiB, 20000 entries or the length of its listing; move large files to data with `ml-exp data upload`",
            },
            other => store_error(other),
        })?;
        self.scope(project)?;
        Ok(value)
    }
    pub fn check_code(&self, project: &str, source_id: &str) -> Result<(), crate::error::Error> {
        self.code(project, source_id, |store| {
            store.check_code(project, source_id)
        })
    }
    pub fn code_manifest(
        &self,
        project: &str,
        source_id: &str,
    ) -> Result<sources::CodeManifest, crate::error::Error> {
        self.code(project, source_id, |store| {
            store.code_manifest(project, source_id)
        })
    }
    pub fn code_content(
        &self,
        project: &str,
        source_id: &str,
    ) -> Result<super::CodeContent, crate::error::Error> {
        self.code(project, source_id, |store| {
            store.code_content(project, source_id)
        })
    }
    pub fn lookup(&self, project: &str, sha256: &str) -> Result<Value, crate::error::Error> {
        self.scope(project)?;
        sources::validate_archive_digest(sha256).map_err(why!(invalid()))?;
        let metadata = self
            .store
            .find_archive(project, sha256)
            .map_err(store_error)?;
        let metadata=metadata.ok_or(crate::error::Error {status:404,code:"SOURCE_RECEIPT_NOT_OBSERVED",detail:"no archive receipt was observed; this does not prove source publication was absent and does not authorize automatic replay"})?;
        self.scope(project)?;
        sources::response(
            &metadata,
            &json!({"kind":"archive","sha256":sha256}),
            json!({"existing":true}),
            &self.policy,
        )
        .map_err(why!(unavailable()))
    }
}
fn store_error(error: super::Error) -> crate::error::Error {
    // Refined typed mapping lives here, outside pure source identity reducers.
    match error {
        super::Error::EffectUncertain(_) => uncertain(),
        _ => unavailable(),
    }
}
