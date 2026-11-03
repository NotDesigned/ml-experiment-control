//! Bounded discovery of authoritative Run directories below operator-approved
//! roots. No disposable index, caller locator or observed metric selects a Run.
use super::{Document, DocumentError, ManifestSource, RunRef, domain};
use crate::deletion::RunFence;
use ml_exp_core::identity;
use ml_exp_io::directory::{Directory, RegularIdentity, effective_uid};
use std::{
    collections::BTreeSet,
    ffi::{OsStr, OsString},
    io::{self, Read},
};

#[derive(Clone, Copy, Debug)]
pub struct DiscoveryLimits {
    pub max_depth: usize,
    pub max_directories: usize,
    pub max_entries: usize,
    pub max_runs: usize,
    pub max_metadata_bytes: u64,
}
impl Default for DiscoveryLimits {
    fn default() -> Self {
        Self {
            max_depth: 16,
            max_directories: 4096,
            max_entries: 100_000,
            max_runs: 4096,
            max_metadata_bytes: 32 * 1024 * 1024,
        }
    }
}
impl DiscoveryLimits {
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        if self.max_depth == 0
            || self.max_depth > 32
            || self.max_directories == 0
            || self.max_directories > 16_384
            || self.max_entries == 0
            || self.max_entries > 100_000
            || self.max_runs == 0
            || self.max_runs > 4096
            || self.max_metadata_bytes == 0
            || self.max_metadata_bytes > 32 * 1024 * 1024
        {
            return Err(DiscoveryError::Limits);
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("Project is not configured for authoritative Run discovery")]
    UnknownProject,
    #[error("invalid bounded Run discovery limits")]
    Limits,
    #[error("Run discovery exceeds its bounded traversal or metadata budget")]
    Budget,
    #[error("approved Run roots overlap or repeat")]
    OverlappingRoots,
    #[error("more than one authoritative directory names the same Run")]
    DuplicateRun,
    #[error("Run discovery encountered an unsafe or unsupported directory entry")]
    UnsafeEntry,
    #[error("science-only or payload layout cannot establish an authoritative Run")]
    UnsupportedLayout,
    #[error("Run discovery namespace or frozen metadata changed during observation")]
    Changed,
    #[error(transparent)]
    Document(#[from] DocumentError),
    #[error(transparent)]
    Fence(#[from] crate::deletion::Error),
    #[error("Run discovery filesystem evidence is unavailable")]
    Io(#[from] io::Error),
}

/// A scanner-selected location, not execution authorization or a scientific
/// digest. A subsequent read must retain ordinary Run/deletion checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocatedRun {
    pub reference: RunRef,
    pub raw_sha256: String,
}
pub struct Discovery {
    pub project_raw_sha256: String,
    pub runs: Vec<LocatedRun>,
    /// Existing deletion evidence is kept explicit and never re-admitted live.
    pub deleted: Vec<LocatedRun>,
    pub examined_directories: usize,
    pub examined_entries: usize,
    pub sampled_metadata_bytes: u64,
}
impl Discovery {
    /// Compare authority bindings, not traversal statistics or unrelated file
    /// counts. Callers bracket a composed read with two bounded discoveries.
    pub fn same_bindings(&self, other: &Self) -> bool {
        self.project_raw_sha256 == other.project_raw_sha256
            && self.runs == other.runs
            && self.deleted == other.deleted
    }
}

struct Budget<'a> {
    limits: &'a DiscoveryLimits,
    directories: usize,
    entries: usize,
    runs: usize,
    bytes: u64,
}
impl Budget<'_> {
    fn directory(&mut self) -> Result<(), DiscoveryError> {
        self.directories = self
            .directories
            .checked_add(1)
            .ok_or(DiscoveryError::Budget)?;
        if self.directories > self.limits.max_directories {
            return Err(DiscoveryError::Budget);
        }
        Ok(())
    }
    fn metadata(&mut self, length: u64) -> Result<(), DiscoveryError> {
        self.bytes = self
            .bytes
            .checked_add(length)
            .ok_or(DiscoveryError::Budget)?;
        if self.bytes > self.limits.max_metadata_bytes {
            return Err(DiscoveryError::Budget);
        }
        Ok(())
    }
}
struct Observation {
    directory: Directory,
    names: Option<Vec<OsString>>,
    manifest: Option<Document>,
    files: Vec<(OsString, RegularIdentity)>,
}

fn read_manifest(
    directory: &Directory,
    sources: &[ManifestSource],
    budget: &mut Budget<'_>,
) -> Result<Option<Document>, DiscoveryError> {
    for source in sources {
        let name = OsStr::new(source.filename());
        if !directory.entry_exists(name)? {
            continue;
        }
        let mut file = directory.open_regular_read(name)?;
        let length = file.metadata()?.len();
        if length > domain::MAX_BYTES as u64 {
            return Err(DocumentError::TooLarge.into());
        }
        budget.metadata(length)?;
        let mut bytes = Vec::new();
        (&mut file).take(length + 1).read_to_end(&mut bytes)?;
        directory.validate_regular_entry(name, &file)?;
        if bytes.len() as u64 != length {
            return Err(DiscoveryError::Changed);
        }
        return Ok(Some(Document::decode(&bytes, *source)?));
    }
    Ok(None)
}

fn names(directory: &Directory, budget: &mut Budget<'_>) -> Result<Vec<OsString>, DiscoveryError> {
    let remaining = budget.limits.max_entries.saturating_sub(budget.entries);
    let entries = directory.entry_names_limited(remaining.saturating_add(1))?;
    if entries.len() > remaining {
        return Err(DiscoveryError::Budget);
    }
    budget.entries = budget
        .entries
        .checked_add(entries.len())
        .ok_or(DiscoveryError::Budget)?;
    Ok(entries)
}

fn unchanged(first: &Document, second: Option<Document>) -> Result<(), DiscoveryError> {
    if second.is_some_and(|second| {
        first.source == second.source && first.raw_sha256 == second.raw_sha256
    }) {
        Ok(())
    } else {
        Err(DiscoveryError::Changed)
    }
}

fn verify_observations(
    observed: &[Observation],
    run_sources: &[ManifestSource],
    budget: &mut Budget<'_>,
) -> Result<(), DiscoveryError> {
    for observation in observed {
        observation.directory.verify_private_parent()?;
        if let Some(manifest) = &observation.manifest {
            unchanged(
                manifest,
                read_manifest(&observation.directory, run_sources, budget)?,
            )?;
        } else {
            for (name, identity) in &observation.files {
                if observation.directory.regular_identity(name)? != Some(*identity) {
                    return Err(DiscoveryError::Changed);
                }
            }
            if read_manifest(&observation.directory, run_sources, budget)?.is_some()
                || names(&observation.directory, budget)?
                    != *observation.names.as_ref().ok_or(DiscoveryError::Changed)?
            {
                return Err(DiscoveryError::Changed);
            }
        }
    }
    Ok(())
}

pub(super) fn scan(
    project: &str,
    parent: &Directory,
    project_filename: &OsStr,
    roots: &[Directory],
    fence: &RunFence,
    limits: &DiscoveryLimits,
    run_sources: &[ManifestSource],
) -> Result<Discovery, DiscoveryError> {
    limits.validate()?;
    identity::validate_id(project).map_err(|_| DiscoveryError::UnsafeEntry)?;
    let guard = fence.shared_existing()?;
    parent.verify_private_parent()?;
    let project_bytes = parent.read_regular(project_filename, domain::MAX_BYTES as u64)?;
    let project_before = Document::decode(&project_bytes, ManifestSource::ProjectYaml)?;
    domain::validate_project(&project_before.metadata, project)?;
    let mut budget = Budget {
        limits,
        directories: 0,
        entries: 0,
        runs: 0,
        bytes: 0,
    };
    budget.metadata(project_bytes.len() as u64)?;
    for (index, root) in roots.iter().enumerate() {
        root.verify_private_parent()?;
        if roots[..index].iter().any(|previous| {
            root.path().starts_with(previous.path()) || previous.path().starts_with(root.path())
        }) {
            return Err(DiscoveryError::OverlappingRoots);
        }
    }
    let mut observed = Vec::new();
    let mut seen = BTreeSet::new();
    let mut runs = Vec::new();
    let mut deleted = Vec::new();
    if roots.len() > limits.max_directories {
        return Err(DiscoveryError::Budget);
    }
    let mut pending: Vec<_> = roots
        .iter()
        .enumerate()
        .rev()
        .map(|(index, root)| (index, root.clone(), Vec::<String>::new()))
        .collect();
    while let Some((root_index, directory, components)) = pending.pop() {
        budget.directory()?;
        directory.verify_private_parent()?;
        if let Some(manifest) = read_manifest(&directory, run_sources, &mut budget)? {
            let run_id = manifest
                .metadata
                .get("run_id")
                .and_then(serde_json::Value::as_str)
                .ok_or(DocumentError::Identity)?;
            domain::validate_run(&manifest.metadata, project, run_id)?;
            if !seen.insert(run_id.to_owned()) {
                return Err(DiscoveryError::DuplicateRun);
            }
            budget.runs = budget.runs.checked_add(1).ok_or(DiscoveryError::Budget)?;
            if budget.runs > limits.max_runs {
                return Err(DiscoveryError::Budget);
            }
            let located = LocatedRun {
                reference: RunRef {
                    project: project.into(),
                    run_id: run_id.into(),
                    root_index,
                    relative_components: components,
                },
                raw_sha256: manifest.raw_sha256.clone(),
            };
            if guard.is_deleted(project, run_id)? {
                deleted.push(located);
            } else {
                runs.push(located);
            }
            observed.push(Observation {
                directory,
                names: None,
                manifest: Some(manifest),
                files: Vec::new(),
            });
            // The Run owns all nested artifacts, attempts and collected mirrors.
            // They can never become a second authoritative Run.
            continue;
        }
        let entries = names(&directory, &mut budget)?;
        let mut files = Vec::new();
        for name in entries.iter().rev() {
            match directory.child(name, false) {
                Ok(child) => {
                    child.verify_private_parent()?;
                    let text = name.to_str().ok_or(DiscoveryError::UnsafeEntry)?;
                    identity::validate_observation_id(text)
                        .map_err(|_| DiscoveryError::UnsafeEntry)?;
                    if matches!(text, "attempts" | "collected_run" | "uploaded_outputs") {
                        return Err(DiscoveryError::UnsupportedLayout);
                    }
                    if components.len() >= limits.max_depth {
                        return Err(DiscoveryError::Budget);
                    }
                    if budget.directories + pending.len() + 1 > limits.max_directories {
                        return Err(DiscoveryError::Budget);
                    }
                    let mut child_components = components.clone();
                    child_components.push(text.into());
                    pending.push((root_index, child, child_components));
                }
                Err(error) if error.kind() == io::ErrorKind::NotADirectory => {
                    // Reject links/FIFOs/devices and aliases even when their
                    // names are otherwise unrelated to a scientific manifest.
                    let file = directory
                        .regular_identity(name)?
                        .ok_or(DiscoveryError::Changed)?;
                    if !file.is_private_for(effective_uid()) {
                        return Err(DiscoveryError::UnsafeEntry);
                    }
                    files.push((name.clone(), file));
                }
                Err(error) => return Err(error.into()),
            }
        }
        observed.push(Observation {
            directory,
            names: Some(entries),
            manifest: None,
            files,
        });
    }
    // Discovery is a bounded observation, not an atomic filesystem snapshot.
    // Ordinary additions/removals/replacements are refused before returning;
    // arbitrary same-UID/root ABA mutation is outside directory capabilities.
    let mut validation = Budget {
        limits,
        directories: 0,
        entries: 0,
        runs: 0,
        bytes: 0,
    };
    verify_observations(&observed, run_sources, &mut validation)?;
    let project_after = parent.read_regular(project_filename, domain::MAX_BYTES as u64)?;
    if project_bytes != project_after {
        return Err(DiscoveryError::Changed);
    }
    parent.verify_private_parent()?;
    for root in roots {
        root.verify_private_parent()?;
    }
    guard.validate()?;
    // IDs are unique; order is only stable output, never current selection.
    runs.sort_by(|left, right| left.reference.run_id.cmp(&right.reference.run_id));
    deleted.sort_by(|left, right| left.reference.run_id.cmp(&right.reference.run_id));
    Ok(Discovery {
        project_raw_sha256: project_before.raw_sha256,
        runs,
        deleted,
        examined_directories: budget.directories,
        examined_entries: budget.entries,
        sampled_metadata_bytes: budget.bytes,
    })
}

#[cfg(test)]
mod tests;
