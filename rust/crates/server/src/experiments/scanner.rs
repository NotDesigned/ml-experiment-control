//! Bounded discovery of authoritative Run directories below operator-approved
//! roots. No disposable index, caller locator or observed metric selects a Run.
use super::{Document, DocumentError, ManifestSource, RunRef, domain};
use crate::deletion::RunFence;
use ml_exp_core::identity;
use ml_exp_io::directory::{Directory, effective_uid};
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
    pub runs: Vec<LocatedRun>,
    /// Existing deletion evidence is kept explicit and never re-admitted live.
    pub deleted: Vec<LocatedRun>,
    pub examined_directories: usize,
    pub examined_entries: usize,
    pub sampled_metadata_bytes: u64,
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
        // Discovery reads trees this service did not create, so unlike its own
        // storage, a linked manifest is refused here.
        if !directory
            .regular_identity(name)?
            .is_some_and(|file| file.is_private_for(effective_uid()))
        {
            return Err(DiscoveryError::UnsafeEntry);
        }
        let mut file = directory.open_regular_read(name)?;
        let length = file.metadata()?.len();
        if length > domain::MAX_BYTES as u64 {
            return Err(DocumentError::TooLarge.into());
        }
        budget.metadata(length)?;
        let mut bytes = Vec::new();
        (&mut file).take(length + 1).read_to_end(&mut bytes)?;
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
    identity::validate_id(project).map_err(why!(DiscoveryError::UnsafeEntry))?;
    let guard = fence.shared_existing()?;
    let project_bytes = parent.read_regular(project_filename, domain::MAX_BYTES as u64)?;
    let project_document = Document::decode(&project_bytes, ManifestSource::ProjectYaml)?;
    domain::validate_project(&project_document.metadata, project)?;
    let mut budget = Budget {
        limits,
        directories: 0,
        entries: 0,
        runs: 0,
        bytes: 0,
    };
    budget.metadata(project_bytes.len() as u64)?;
    for (index, root) in roots.iter().enumerate() {
        if roots[..index].iter().any(|previous| {
            root.path().starts_with(previous.path()) || previous.path().starts_with(root.path())
        }) {
            return Err(DiscoveryError::OverlappingRoots);
        }
    }
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
            // The Run owns all nested artifacts, attempts and collected mirrors.
            // They can never become a second authoritative Run.
            continue;
        }
        let entries = names(&directory, &mut budget)?;
        for name in entries.iter().rev() {
            match directory.child(name, false) {
                Ok(child) => {
                    let text = name.to_str().ok_or(DiscoveryError::UnsafeEntry)?;
                    identity::validate_observation_id(text)
                        .map_err(why!(DiscoveryError::UnsafeEntry))?;
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
                // Only a directory can hold a Run. Files and links beside it are
                // never opened or followed.
                Err(error) if error.kind() == io::ErrorKind::NotADirectory => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    // IDs are unique; order is only stable output, never current selection.
    runs.sort_by(|left, right| left.reference.run_id.cmp(&right.reference.run_id));
    deleted.sort_by(|left, right| left.reference.run_id.cmp(&right.reference.run_id));
    Ok(Discovery {
        runs,
        deleted,
        examined_directories: budget.directories,
        examined_entries: budget.entries,
        sampled_metadata_bytes: budget.bytes,
    })
}

#[cfg(test)]
mod tests;
