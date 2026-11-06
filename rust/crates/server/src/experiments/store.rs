use super::domain::{self, Document, DocumentError, ManifestSource, Provenance};
use crate::deletion::RunFence;
use ml_exp_core::identity;
use ml_exp_io::directory::Directory;
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    path::PathBuf,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid trusted experiment metadata binding or locator")]
    Binding,
    #[error("unknown bound project")]
    UnknownProject,
    #[error("authoritative experiment manifest is missing")]
    Missing,
    #[error("frozen manifest bytes changed during observation or differ from the expected SHA256")]
    BytesChanged,
    #[error(transparent)]
    Document(#[from] DocumentError),
    #[error(transparent)]
    Fence(#[from] crate::deletion::Error),
    #[error("experiment metadata filesystem evidence is unavailable")]
    Io(#[from] std::io::Error),
}
type Result<T> = std::result::Result<T, Error>;

/// Trusted operator configuration. An authored document cannot select a new filesystem
/// root. Lifecycle registration and resolving approved roots remain outside this
/// reader; every configured path must already exist.
pub struct ProjectBinding {
    pub project: String,
    pub project_file: PathBuf,
    pub run_roots: Vec<PathBuf>,
}
/// A location selected by a trusted operator/resolver beneath an approved root.
/// It is not an untrusted client/index input. Matching file IDs or SHA cannot
/// establish the unique canonical location when a copied manifest exists. A
/// future scanner must resolve approved layout and reject duplicates first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunRef {
    pub project: String,
    pub run_id: String,
    pub root_index: usize,
    pub relative_components: Vec<String>,
}
#[derive(Clone, Debug, Default)]
pub struct ExpectedHashes {
    pub run: Option<String>,
    pub attempt: Option<String>,
}
// Deliberately no Debug: scientific manifests may include user environment data.
pub struct AttemptRecord {
    pub run: Document,
    pub attempt: Document,
    pub provenance: Provenance,
}
struct ProjectAnchor {
    parent: Directory,
    filename: OsString,
    roots: Vec<Directory>,
}
pub struct Store {
    fence: RunFence,
    projects: BTreeMap<String, ProjectAnchor>,
}

fn check_sha(document: &Document, expected: Option<&str>) -> Result<()> {
    if let Some(expected) = expected {
        let valid = expected.strip_prefix("sha256:").is_some_and(|v| {
            v.len() == 64
                && v.bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        });
        if !valid {
            return Err(Error::Binding);
        }
        if document.raw_sha256 != expected {
            return Err(Error::BytesChanged);
        }
    }
    Ok(())
}
fn read(directory: &Directory, name: &OsStr, source: ManifestSource) -> Result<Document> {
    Ok(Document::decode(&directory.read_whole(name)?, source)?)
}
fn read_fixed(directory: &Directory, source: ManifestSource) -> Result<Document> {
    let name = OsStr::new(source.filename());
    // Symlink/FIFO/directory entries are errors, not absence.
    if !directory.entry_exists(name)? {
        return Err(Error::Missing);
    }
    read(directory, name, source)
}

impl ProjectAnchor {
    fn project(&self, identity: &str) -> Result<Document> {
        let document = read(&self.parent, &self.filename, ManifestSource::ProjectJson)?;
        domain::validate_project(&document.metadata, identity)?;
        Ok(document)
    }
    fn run_directory(&self, reference: &RunRef) -> Result<Directory> {
        identity::validate_id(&reference.project).map_err(why!(Error::Binding))?;
        identity::validate_observation_id(&reference.run_id).map_err(why!(Error::Binding))?;
        if reference.relative_components.len() > super::scanner::MAX_DEPTH {
            return Err(Error::Binding);
        }
        let mut directory = self
            .roots
            .get(reference.root_index)
            .ok_or(Error::Binding)?
            .clone();
        for name in &reference.relative_components {
            identity::validate_observation_id(name).map_err(why!(Error::Binding))?;
            directory = directory.child(OsStr::new(name), false)?;
        }
        Ok(directory)
    }
}

impl Store {
    /// Discover unique authoritative Runs below the bound Project's approved
    /// roots. This observes existing files/fences only and never repairs them.
    pub fn discover(
        &self,
        project: &str,
    ) -> std::result::Result<super::scanner::Discovery, super::scanner::DiscoveryError> {
        let anchor = self
            .projects
            .get(project)
            .ok_or(super::scanner::DiscoveryError::UnknownProject)?;
        super::scanner::scan(
            project,
            &anchor.parent,
            &anchor.filename,
            &anchor.roots,
            &self.fence,
        )
    }

    /// No directory, lock, manifest, SQLite file or recovery artifact is created.
    /// A missing existing deletion fence fails closed instead of bootstrapping.
    pub fn open(bindings: Vec<ProjectBinding>, fence: RunFence) -> Result<Self> {
        let guard = fence.shared_existing()?;
        let mut projects = BTreeMap::new();
        for binding in bindings {
            identity::validate_id(&binding.project).map_err(why!(Error::Binding))?;
            if projects.contains_key(&binding.project)
                || !binding.project_file.is_absolute()
                || binding.run_roots.is_empty()
                || binding.run_roots.len() > 64
            {
                return Err(Error::Binding);
            }
            let parent = binding.project_file.parent().ok_or(Error::Binding)?;
            let filename = binding
                .project_file
                .file_name()
                .ok_or(Error::Binding)?
                .to_owned();
            let anchor = ProjectAnchor {
                parent: Directory::open(parent, false)?,
                filename,
                roots: binding
                    .run_roots
                    .iter()
                    .map(|path| Directory::open(path, false))
                    .collect::<std::io::Result<Vec<_>>>()?,
            };
            anchor.project(&binding.project)?;
            projects.insert(binding.project, anchor);
        }
        drop(guard);
        Ok(Self { fence, projects })
    }
    fn project_anchor(&self, project: &str) -> Result<&ProjectAnchor> {
        identity::validate_id(project).map_err(why!(Error::Binding))?;
        self.projects.get(project).ok_or(Error::UnknownProject)
    }
    pub fn project(&self, project: &str, expected_raw_sha256: Option<&str>) -> Result<Document> {
        let _guard = self.fence.shared_existing()?;
        let anchor = self.project_anchor(project)?;
        let document = anchor.project(project)?;
        check_sha(&document, expected_raw_sha256)?;
        Ok(document)
    }
    pub fn run(&self, reference: &RunRef, expected_raw_sha256: Option<&str>) -> Result<Document> {
        let _guard = self
            .fence
            .existing_observation_guard(&reference.project, &reference.run_id)?;
        let anchor = self.project_anchor(&reference.project)?;
        let directory = anchor.run_directory(reference)?;
        let document = read_fixed(&directory, ManifestSource::RunJson)?;
        domain::validate_run(&document.metadata, &reference.project, &reference.run_id)?;
        check_sha(&document, expected_raw_sha256)?;
        Ok(document)
    }
    pub fn attempt(
        &self,
        reference: &RunRef,
        attempt_id: &str,
        expected: &ExpectedHashes,
    ) -> Result<AttemptRecord> {
        identity::validate_id(attempt_id).map_err(why!(Error::Binding))?;
        let _guard = self
            .fence
            .existing_observation_guard(&reference.project, &reference.run_id)?;
        let anchor = self.project_anchor(&reference.project)?;
        let directory = anchor.run_directory(reference)?;
        let run = read_fixed(&directory, ManifestSource::RunJson)?;
        domain::validate_run(&run.metadata, &reference.project, &reference.run_id)?;
        check_sha(&run, expected.run.as_deref())?;
        let attempts = directory.child(OsStr::new("attempts"), false)?;
        let attempt_directory = attempts.child(OsStr::new(attempt_id), false)?;
        let attempt = read_fixed(&attempt_directory, ManifestSource::AttemptJson)?;
        check_sha(&attempt, expected.attempt.as_deref())?;
        let provenance = domain::validate_attempt(
            &run,
            &attempt,
            &reference.project,
            &reference.run_id,
            attempt_id,
        )?;
        Ok(AttemptRecord {
            run,
            attempt,
            provenance,
        })
    }

    /// Metadata for every bounded Attempt; only the explicitly selected current
    /// Attempt contributes logs/JSONL. Directory sorting never selects current.
    pub fn snapshot(&self, reference: &RunRef) -> Result<super::Snapshot> {
        self.snapshot_selected(reference, None, true)
    }

    /// Bounded Run/Attempt metadata only, for project-level status summaries.
    /// Even the explicitly current Attempt has unloaded log/metric tails.
    pub fn snapshot_metadata(&self, reference: &RunRef) -> Result<super::Snapshot> {
        self.snapshot_selected(reference, None, false)
    }

    /// Read one exact Attempt only. Malformed siblings, Run mirrors
    /// or a different current Attempt cannot supply its payload or identity.
    pub fn snapshot_attempt(
        &self,
        reference: &RunRef,
        attempt_id: &str,
    ) -> Result<super::Snapshot> {
        identity::validate_id(attempt_id).map_err(why!(Error::Binding))?;
        self.snapshot_selected(reference, Some(attempt_id), true)
    }

    fn snapshot_selected(
        &self,
        reference: &RunRef,
        exact: Option<&str>,
        load_heavy: bool,
    ) -> Result<super::Snapshot> {
        use super::snapshot::{self, AttemptSnapshot, Snapshot};
        let _guard = self
            .fence
            .existing_observation_guard(&reference.project, &reference.run_id)?;
        let anchor = self.project_anchor(&reference.project)?;
        let directory = anchor.run_directory(reference)?;
        let run = snapshot::manifest(&directory, ManifestSource::RunJson)?;
        domain::validate_run(&run.metadata, &reference.project, &reference.run_id)?;
        let current_status = snapshot::evidence(
            &directory,
            "status.json",
            "status.json",
            &reference.project,
            &reference.run_id,
            None,
        );
        let current_collection = snapshot::evidence(
            &directory,
            "collection.json",
            "collection.json",
            &reference.project,
            &reference.run_id,
            None,
        );
        let current_decision = snapshot::evidence(
            &directory,
            "decision.json",
            "decision.json",
            &reference.project,
            &reference.run_id,
            None,
        );
        let attempts_directory = if directory.entry_exists(OsStr::new("attempts"))? {
            Some(directory.child(OsStr::new("attempts"), false)?)
        } else {
            None
        };
        let names = match (&attempts_directory, exact) {
            (Some(root), Some(exact)) => {
                if !root.entry_exists(OsStr::new(exact))? {
                    return Err(Error::Missing);
                }
                vec![exact.to_owned()]
            }
            (None, Some(_)) => return Err(Error::Missing),
            (None, None) => Vec::new(),
            (Some(root), None) => root
                .entry_names()?
                .into_iter()
                .filter_map(|name| {
                    name.to_str()
                        .filter(|name| identity::validate_id(name).is_ok())
                        .map(str::to_owned)
                })
                .collect(),
        };
        let (current_attempt_id, current_selection_source, diagnostics) = snapshot::current(
            &[&current_status, &current_collection, &current_decision],
            &names,
        );
        let mut attempts = Vec::new();
        for attempt_id in names {
            let root = attempts_directory.as_ref().ok_or(Error::Missing)?;
            let attempt_directory = root.child(OsStr::new(&attempt_id), false)?;
            let manifest = snapshot::manifest(&attempt_directory, ManifestSource::AttemptJson)?;
            let provenance = domain::validate_attempt(
                &run,
                &manifest,
                &reference.project,
                &reference.run_id,
                &attempt_id,
            )?;
            let prefix = format!("attempts/{attempt_id}");
            let evidence = |name: &str| {
                snapshot::evidence(
                    &attempt_directory,
                    name,
                    &format!("{prefix}/{name}"),
                    &reference.project,
                    &reference.run_id,
                    Some(&attempt_id),
                )
            };
            let status = evidence("status.json");
            let backend = evidence("backend.json");
            let collection = evidence("collection.json");
            let decision = evidence("decision.json");
            let submission = evidence("submission.json");
            let (stdout, stderr, metrics) = if load_heavy
                && (exact.is_some() || current_attempt_id.as_deref() == Some(&attempt_id))
            {
                snapshot::heavy(
                    &attempt_directory,
                    &reference.project,
                    &reference.run_id,
                    &attempt_id,
                )?
            } else {
                snapshot::unloaded(&attempt_id)
            };
            attempts.push(AttemptSnapshot {
                attempt_id,
                manifest,
                provenance,
                status,
                backend,
                collection,
                decision,
                submission,
                stdout,
                stderr,
                metrics,
            });
        }
        Ok(Snapshot {
            project: reference.project.clone(),
            run_id: reference.run_id.clone(),
            run,
            current_status,
            current_collection,
            current_decision,
            current_attempt_id,
            current_selection_source,
            attempts,
            diagnostics,
        })
    }
}
