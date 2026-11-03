use super::domain::{self, Document, DocumentError, ManifestSource, Provenance};
use crate::deletion::{Guard, RunFence};
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
    #[error("experiment snapshot exceeds its bounded metadata or Attempt limit")]
    SnapshotLimit,
    #[error(transparent)]
    Document(#[from] DocumentError),
    #[error(transparent)]
    Fence(#[from] crate::deletion::Error),
    #[error("experiment metadata filesystem evidence is unavailable")]
    Io(#[from] std::io::Error),
}
type Result<T> = std::result::Result<T, Error>;

/// Trusted operator configuration. Authored YAML cannot select a new filesystem
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
fn same(first: &Document, second: &Document) -> Result<()> {
    if first.source == second.source && first.raw_sha256 == second.raw_sha256 {
        Ok(())
    } else {
        Err(Error::BytesChanged)
    }
}
fn read(directory: &Directory, name: &OsStr, source: ManifestSource) -> Result<Document> {
    Ok(Document::decode(
        &directory.read_regular(name, domain::MAX_BYTES as u64)?,
        source,
    )?)
}
fn read_first(directory: &Directory, sources: &[ManifestSource]) -> Result<Document> {
    for source in sources {
        let name = OsStr::new(source.filename());
        // An existing corrupt canonical file must never silently fall back to
        // a legacy copy. Symlink/FIFO/directory entries are errors, not absence.
        if directory.entry_exists(name)? {
            return read(directory, name, *source);
        }
    }
    Err(Error::Missing)
}
const RUN_SOURCES: &[ManifestSource] = &[
    ManifestSource::RunYaml,
    ManifestSource::RunJson,
    ManifestSource::LegacyRunYaml,
];
const ATTEMPT_SOURCES: &[ManifestSource] = &[
    ManifestSource::AttemptYaml,
    ManifestSource::AttemptJson,
    ManifestSource::LegacyAttemptYaml,
];

impl ProjectAnchor {
    fn validate(&self) -> Result<()> {
        self.parent.verify_private_parent()?;
        for root in &self.roots {
            root.verify_private_parent()?;
        }
        Ok(())
    }
    fn project(&self, identity: &str) -> Result<Document> {
        self.validate()?;
        let document = read(&self.parent, &self.filename, ManifestSource::ProjectYaml)?;
        domain::validate_project(&document.metadata, identity)?;
        Ok(document)
    }
    fn run_directory(&self, reference: &RunRef) -> Result<Directory> {
        identity::validate_id(&reference.project).map_err(|_| Error::Binding)?;
        identity::validate_observation_id(&reference.run_id).map_err(|_| Error::Binding)?;
        if reference.relative_components.len() > 32 {
            return Err(Error::Binding);
        }
        let mut directory = self
            .roots
            .get(reference.root_index)
            .ok_or(Error::Binding)?
            .clone();
        for name in &reference.relative_components {
            identity::validate_observation_id(name).map_err(|_| Error::Binding)?;
            directory = directory.child(OsStr::new(name), false)?;
        }
        directory.verify_private_parent()?;
        Ok(directory)
    }
    fn verify_snapshot(
        &self,
        project: &Document,
        run: &Document,
        directory: &Directory,
        reference: &RunRef,
        guard: &Guard<'_>,
    ) -> Result<()> {
        same(project, &self.project(&reference.project)?)?;
        same(run, &read_first(directory, RUN_SOURCES)?)?;
        self.validate()?;
        directory.verify_private_parent()?;
        guard.require_observed_live(&reference.project, &reference.run_id)?;
        Ok(())
    }
}

impl Store {
    /// Discover unique authoritative Runs below the bound Project's approved
    /// roots. This observes existing files/fences only and never repairs them.
    pub fn discover(
        &self,
        project: &str,
        limits: &super::scanner::DiscoveryLimits,
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
            limits,
            RUN_SOURCES,
        )
    }

    /// No directory, lock, manifest, SQLite file or recovery artifact is created.
    /// A missing existing deletion fence fails closed instead of bootstrapping.
    pub fn open(bindings: Vec<ProjectBinding>, fence: RunFence) -> Result<Self> {
        let guard = fence.shared_existing()?;
        let mut projects = BTreeMap::new();
        for binding in bindings {
            identity::validate_id(&binding.project).map_err(|_| Error::Binding)?;
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
        guard.validate()?;
        drop(guard);
        Ok(Self { fence, projects })
    }
    fn project_anchor(&self, project: &str) -> Result<&ProjectAnchor> {
        identity::validate_id(project).map_err(|_| Error::Binding)?;
        self.projects.get(project).ok_or(Error::UnknownProject)
    }
    pub fn project(&self, project: &str, expected_raw_sha256: Option<&str>) -> Result<Document> {
        let guard = self.fence.shared_existing()?;
        let anchor = self.project_anchor(project)?;
        let document = anchor.project(project)?;
        check_sha(&document, expected_raw_sha256)?;
        same(&document, &anchor.project(project)?)?;
        anchor.validate()?;
        guard.validate()?;
        Ok(document)
    }
    pub fn run(&self, reference: &RunRef, expected_raw_sha256: Option<&str>) -> Result<Document> {
        let guard = self
            .fence
            .existing_observation_guard(&reference.project, &reference.run_id)?;
        let anchor = self.project_anchor(&reference.project)?;
        let project = anchor.project(&reference.project)?;
        let directory = anchor.run_directory(reference)?;
        let document = read_first(&directory, RUN_SOURCES)?;
        domain::validate_run(&document.metadata, &reference.project, &reference.run_id)?;
        check_sha(&document, expected_raw_sha256)?;
        anchor.verify_snapshot(&project, &document, &directory, reference, &guard)?;
        Ok(document)
    }
    pub fn attempt(
        &self,
        reference: &RunRef,
        attempt_id: &str,
        expected: &ExpectedHashes,
    ) -> Result<AttemptRecord> {
        identity::validate_id(attempt_id).map_err(|_| Error::Binding)?;
        let guard = self
            .fence
            .existing_observation_guard(&reference.project, &reference.run_id)?;
        let anchor = self.project_anchor(&reference.project)?;
        let project = anchor.project(&reference.project)?;
        let directory = anchor.run_directory(reference)?;
        let run = read_first(&directory, RUN_SOURCES)?;
        domain::validate_run(&run.metadata, &reference.project, &reference.run_id)?;
        check_sha(&run, expected.run.as_deref())?;
        let attempts = directory.child(OsStr::new("attempts"), false)?;
        let attempt_directory = attempts.child(OsStr::new(attempt_id), false)?;
        attempt_directory.verify_private_parent()?;
        let attempt = read_first(&attempt_directory, ATTEMPT_SOURCES)?;
        check_sha(&attempt, expected.attempt.as_deref())?;
        let provenance = domain::validate_attempt(
            &run,
            &attempt,
            &reference.project,
            &reference.run_id,
            attempt_id,
        )?;
        same(&attempt, &read_first(&attempt_directory, ATTEMPT_SOURCES)?)?;
        attempt_directory.verify_private_parent()?;
        anchor.verify_snapshot(&project, &run, &directory, reference, &guard)?;
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

    /// Read one exact Attempt only. Historical malformed siblings, Run mirrors
    /// or a different current Attempt cannot supply its payload or identity.
    pub fn snapshot_attempt(
        &self,
        reference: &RunRef,
        attempt_id: &str,
    ) -> Result<super::Snapshot> {
        identity::validate_id(attempt_id).map_err(|_| Error::Binding)?;
        self.snapshot_selected(reference, Some(attempt_id), true)
    }

    fn snapshot_selected(
        &self,
        reference: &RunRef,
        exact: Option<&str>,
        load_heavy: bool,
    ) -> Result<super::Snapshot> {
        use super::snapshot::{self, AttemptSnapshot, Snapshot};
        let guard = self
            .fence
            .existing_observation_guard(&reference.project, &reference.run_id)?;
        let anchor = self.project_anchor(&reference.project)?;
        let project = anchor.project(&reference.project)?;
        let directory = anchor.run_directory(reference)?;
        let mut budget = snapshot::Budget::new();
        let run = snapshot::manifest(&directory, RUN_SOURCES, &mut budget)?;
        domain::validate_run(&run.metadata, &reference.project, &reference.run_id)?;
        let current_status = snapshot::evidence(
            &directory,
            "status.json",
            "status.json",
            &reference.project,
            &reference.run_id,
            None,
            &mut budget,
        );
        let current_collection = snapshot::evidence(
            &directory,
            "collection.json",
            "collection.json",
            &reference.project,
            &reference.run_id,
            None,
            &mut budget,
        );
        let current_decision = snapshot::evidence(
            &directory,
            "decision.json",
            "decision.json",
            &reference.project,
            &reference.run_id,
            None,
            &mut budget,
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
                .entry_names_limited(snapshot::MAX_ATTEMPTS)?
                .into_iter()
                .filter_map(|name| {
                    name.to_str()
                        .filter(|name| identity::validate_id(name).is_ok())
                        .map(str::to_owned)
                })
                .collect(),
        };
        if names.len() > snapshot::MAX_ATTEMPTS {
            return Err(Error::SnapshotLimit);
        }
        let (current_attempt_id, current_selection_source, diagnostics) = snapshot::current(
            &[&current_status, &current_collection, &current_decision],
            &names,
        );
        let mut attempts = Vec::new();
        let mut retained = Vec::new();
        for attempt_id in names {
            let root = attempts_directory.as_ref().ok_or(Error::Missing)?;
            let attempt_directory = root.child(OsStr::new(&attempt_id), false)?;
            attempt_directory.verify_private_parent()?;
            let manifest = snapshot::manifest(&attempt_directory, ATTEMPT_SOURCES, &mut budget)?;
            let provenance = domain::validate_attempt(
                &run,
                &manifest,
                &reference.project,
                &reference.run_id,
                &attempt_id,
            )?;
            let prefix = format!("attempts/{attempt_id}");
            let mut evidence = |name: &str| {
                snapshot::evidence(
                    &attempt_directory,
                    name,
                    &format!("{prefix}/{name}"),
                    &reference.project,
                    &reference.run_id,
                    Some(&attempt_id),
                    &mut budget,
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
            retained.push(attempt_directory);
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
        for (attempt, root) in attempts.iter().zip(&retained) {
            same(&attempt.manifest, &read_first(root, ATTEMPT_SOURCES)?)?;
            root.verify_private_parent()?;
        }
        if let Some(root) = attempts_directory {
            root.verify_private_parent()?;
        }
        anchor.verify_snapshot(&project, &run, &directory, reference, &guard)?;
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
