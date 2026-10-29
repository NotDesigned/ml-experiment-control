//! An explicitly configured historical namespace is observable, never writable
//! native authority. It requires no native marker and never creates one.
use crate::{deletion::RunFence, projects, read_api::Api, read_config};
use ml_exp_io::directory::Directory;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::Arc,
};
mod artifact_tickets;
pub use artifact_tickets::ObjectConfig;

pub(crate) fn identity_conflict() -> crate::read_api::Error {
    crate::read_api::Error {
        status: 409,
        code: "RUN_ID_RESERVED_BY_HISTORY",
        detail: "Run identity is already reserved in historical storage; choose a new Run ID",
    }
}

/// A read-only historical source shared by both request routing and every
/// authorship path (including asynchronous preparations). Never a write adapter.
pub struct Bridge {
    api: Api,
    settings: Settings,
}
impl Bridge {
    pub fn open(settings: Settings) -> Result<Arc<Self>, read_config::Error> {
        let api = open(settings.clone())?;
        Ok(Arc::new(Self { api, settings }))
    }
    pub(crate) fn api(&self) -> &Api {
        &self.api
    }
    pub(crate) fn validate_separation(&self, native: &[PathBuf]) -> Result<(), read_config::Error> {
        let old = self
            .settings
            .reads
            .projects
            .iter()
            .flat_map(|p| p.run_roots.iter())
            .chain(std::iter::once(&self.settings.reads.registry_root));
        for path in old {
            if native
                .iter()
                .any(|p| p.starts_with(path) || path.starts_with(p))
            {
                return Err(read_config::Error);
            }
        }
        if self
            .settings
            .reads
            .projects
            .iter()
            .any(|p| native.iter().any(|r| p.project_file.starts_with(r)))
        {
            return Err(read_config::Error);
        }
        Ok(())
    }
    pub(crate) fn validate_native(
        &self,
        settings: &read_config::Settings,
    ) -> Result<(), read_config::Error> {
        let roots = settings
            .projects
            .iter()
            .flat_map(|p| p.run_roots.clone())
            .chain(std::iter::once(settings.registry_root.clone()))
            .collect::<Vec<_>>();
        self.validate_separation(&roots)
    }
    pub fn claims(&self, project: &str, run: &str) -> Result<bool, crate::read_api::Error> {
        self.api.claims(project, run)
    }
    pub fn require_available(
        &self,
        project: &str,
        run: &str,
    ) -> Result<(), crate::read_api::Error> {
        if self.claims(project, run)? {
            Err(identity_conflict())
        } else {
            Ok(())
        }
    }
    pub(crate) fn checkpoint(
        &self,
        project: &str,
        reference: &ml_exp_core::checkpoints::Reference,
    ) -> Result<Value, crate::read_api::Error> {
        self.api
            .historical_checkpoints(
                project,
                &reference.run_id,
                &reference.attempt_id,
                Some(&reference.checkpoint_id),
            )?
            .ok_or_else(identity_conflict)
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub schema_version: u32,
    pub reads: read_config::Settings,
    /// Original records.meta workspace identity, not a new native seed. None
    /// explicitly means no canonical metric journal may be read in this scope.
    pub journal_workspace_id: Option<String>,
    /// Explicit, hash-pinned original S3 profile. Only GET presigning is allowed.
    #[serde(default)]
    pub artifact_store: Option<ObjectConfig>,
}
impl Settings {
    pub fn validate(&self) -> Result<(), read_config::Error> {
        self.reads.validate()?;
        if let Some(profile) = &self.artifact_store {
            profile.validate()?;
        }
        if self.schema_version != 1
            || self.reads.source_policy.is_some()
            || self.journal_workspace_id.as_ref().is_some_and(|s| {
                s.len() != 32
                    || !s
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            })
        {
            return Err(read_config::Error);
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, read_config::Error> {
        let bytes =
            ml_exp_io::read_private(path, 1024 * 1024, true).map_err(|_| read_config::Error)?;
        let value = ml_exp_core::json::mapping(&bytes).map_err(|_| read_config::Error)?;
        let settings: Self = serde_json::from_value(serde_json::Value::Object(value))
            .map_err(|_| read_config::Error)?;
        settings.validate()?;
        Ok(settings)
    }
}

pub(crate) struct Identity {
    pub journal_workspace_id: Option<String>,
    pub artifact_store: Option<ObjectConfig>,
    lifecycle: projects::Store,
    approved: BTreeMap<String, PathBuf>,
}
impl Identity {
    pub(crate) fn open(
        settings: &Settings,
    ) -> Result<(Directory, String, Self), read_config::Error> {
        settings.validate()?;
        let root = Directory::open(&settings.reads.registry_root, false)
            .map_err(|_| read_config::Error)?;
        root.verify_private_parent()
            .map_err(|_| read_config::Error)?;
        if read_config::has_native_marker(&root)? {
            return Err(read_config::Error);
        }
        let fence = RunFence::new(&settings.reads.registry_root);
        let guard = fence.shared_existing().map_err(|_| read_config::Error)?;
        guard.validate().map_err(|_| read_config::Error)?;
        let identity = Self {
            journal_workspace_id: settings.journal_workspace_id.clone(),
            artifact_store: settings.artifact_store.clone(),
            lifecycle: projects::Store::open(&settings.reads.registry_root)
                .map_err(|_| read_config::Error)?,
            approved: settings
                .reads
                .projects
                .iter()
                .map(|p| (p.project.clone(), p.project_file.clone()))
                .collect(),
        };
        identity.active().map_err(|_| read_config::Error)?;
        identity
            .validate_journal(&settings.reads.registry_root)
            .map_err(|_| read_config::Error)?;
        let digest = ml_exp_core::canonical::protocol_identity(
            &serde_json::to_value(settings).map_err(|_| read_config::Error)?,
        )
        .map_err(|_| read_config::Error)?;
        Ok((
            root,
            format!(
                "historical-{}",
                digest.strip_prefix("sha256:").ok_or(read_config::Error)?
            ),
            identity,
        ))
    }
    pub(crate) fn snapshot(&self) -> Result<projects::Snapshot, projects::Error> {
        let snapshot = self.lifecycle.snapshot(200)?;
        for record in &snapshot.projects {
            if self
                .approved
                .get(&record.project)
                .is_some_and(|path| path.to_str() != Some(record.project_file.as_str()))
            {
                return Err(projects::Error::Binding);
            }
        }
        Ok(snapshot)
    }
    pub(crate) fn active(&self) -> Result<BTreeSet<String>, projects::Error> {
        Ok(self
            .snapshot()?
            .projects
            .into_iter()
            .filter(|p| {
                p.state == ml_exp_core::projects::LifecycleState::Active
                    && self.approved.contains_key(&p.project)
            })
            .map(|p| p.project)
            .collect())
    }
    pub(crate) fn validate_journal(&self, path: &Path) -> Result<(), projects::Error> {
        let registry = Directory::open(path, false)?;
        let tracking = if registry.entry_exists(OsStr::new("tracking"))? {
            Some(registry.child(OsStr::new("tracking"), false)?)
        } else {
            None
        };
        let present = tracking
            .as_ref()
            .map(|d| d.entry_exists(OsStr::new("outbox.sqlite")))
            .transpose()?
            .unwrap_or(false);
        if present != self.journal_workspace_id.is_some() {
            return Err(projects::Error::Binding);
        }
        if let Some(expected) = &self.journal_workspace_id {
            let store = crate::records::Store::read_only(path);
            if store
                .existing_workspace_id()
                .map_err(|_| projects::Error::Invalid)?
                != *expected
            {
                return Err(projects::Error::Binding);
            }
        }
        Ok(())
    }
}

pub fn open(settings: Settings) -> Result<Api, read_config::Error> {
    crate::readonly::run(move || Api::open_historical(settings)).map_err(|_| read_config::Error)?
}

/// Produce a private reader configuration from original local state. This is
/// an explicit offline observation; it neither initializes state nor accepts
/// an absent/different journal later without new operator configuration.
pub fn inspect(reads: read_config::Settings) -> Result<Settings, read_config::Error> {
    crate::readonly::run(move || {
        reads.validate()?;
        let root = Directory::open(&reads.registry_root, false).map_err(|_| read_config::Error)?;
        let present = if root
            .entry_exists(OsStr::new("tracking"))
            .map_err(|_| read_config::Error)?
        {
            root.child(OsStr::new("tracking"), false)
                .map_err(|_| read_config::Error)?
                .entry_exists(OsStr::new("outbox.sqlite"))
                .map_err(|_| read_config::Error)?
        } else {
            false
        };
        let journal_workspace_id = if present {
            Some(
                crate::records::Store::read_only(&reads.registry_root)
                    .existing_workspace_id()
                    .map_err(|_| read_config::Error)?,
            )
        } else {
            None
        };
        let settings = Settings {
            schema_version: 1,
            reads,
            journal_workspace_id,
            artifact_store: None,
        };
        Identity::open(&settings)?;
        Ok(settings)
    })
    .map_err(|_| read_config::Error)?
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn reader_configuration_pins_exact_journal_identity_and_refuses_write_policy() {
        let base = json!({"schema_version":1,"reads":{"schema_version":1,"registry_root":"/original/registry","projects":[]},"journal_workspace_id":"a".repeat(32)});
        let parse = |value: &serde_json::Value| -> Settings {
            serde_json::from_value(value.clone()).unwrap()
        };
        assert!(parse(&base).validate().is_ok());
        for value in [
            json!("a".repeat(31)),
            json!("G".repeat(32)),
            json!("../registry"),
        ] {
            let mut invalid = base.clone();
            invalid["journal_workspace_id"] = value;
            assert!(parse(&invalid).validate().is_err());
        }
        let mut empty = base.clone();
        empty["journal_workspace_id"] = serde_json::Value::Null;
        assert!(parse(&empty).validate().is_ok());
        empty["reads"]["source_policy"] = json!({});
        assert!(parse(&empty).validate().is_err());
        let mut invalid = base;
        invalid["schema_version"] = json!(2);
        assert!(parse(&invalid).validate().is_err());
    }
}
