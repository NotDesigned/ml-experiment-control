//! Discovery of authoritative Run directories below operator-approved
//! roots. No disposable index, caller locator or observed metric selects a Run.
use super::{Document, DocumentError, ManifestSource, RunRef, domain};
use crate::deletion::RunFence;
use ml_exp_core::identity;
use ml_exp_io::directory::{Directory, effective_uid};
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    io::{self, Read},
};

/// Matches the deepest reference `Store` will open for a located Run.
pub(super) const MAX_DEPTH: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("Project is not configured for authoritative Run discovery")]
    UnknownProject,
    #[error("Run discovery exceeds its depth")]
    Depth,
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
}

fn read_manifest(directory: &Directory) -> Result<Option<Document>, DiscoveryError> {
    let source = ManifestSource::RunJson;
    let name = OsStr::new(source.filename());
    if !directory.entry_exists(name)? {
        return Ok(None);
    }
    // Discovery reads trees this service did not create, so a manifest that
    // another group or user could rewrite is refused.
    if !directory
        .regular_identity(name)?
        .is_some_and(|file| file.is_private_for(effective_uid()))
    {
        return Err(DiscoveryError::UnsafeEntry);
    }
    let mut file = directory.open_regular_read(name)?;
    let length = file.metadata()?.len();
    let mut bytes = Vec::new();
    (&mut file).take(length + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != length {
        return Err(DiscoveryError::Changed);
    }
    Ok(Some(Document::decode(&bytes, source)?))
}

pub(super) fn scan(
    project: &str,
    parent: &Directory,
    project_filename: &OsStr,
    roots: &[Directory],
    fence: &RunFence,
) -> Result<Discovery, DiscoveryError> {
    identity::validate_id(project).map_err(why!(DiscoveryError::UnsafeEntry))?;
    let guard = fence.shared_existing()?;
    let project_bytes = parent.read_whole(project_filename)?;
    let project_document = Document::decode(&project_bytes, ManifestSource::ProjectJson)?;
    domain::validate_project(&project_document.metadata, project)?;
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
    let mut pending: Vec<_> = roots
        .iter()
        .enumerate()
        .rev()
        .map(|(index, root)| (index, root.clone(), Vec::<String>::new()))
        .collect();
    while let Some((root_index, directory, components)) = pending.pop() {
        if let Some(manifest) = read_manifest(&directory)? {
            let run_id = manifest
                .metadata
                .get("run_id")
                .and_then(serde_json::Value::as_str)
                .ok_or(DocumentError::Identity)?;
            domain::validate_run(&manifest.metadata, project, run_id)?;
            if !seen.insert(run_id.to_owned()) {
                return Err(DiscoveryError::DuplicateRun);
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
        let entries = directory.entry_names()?;
        for name in entries.iter().rev() {
            match directory.child(name, false) {
                Ok(child) => {
                    let text = name.to_str().ok_or(DiscoveryError::UnsafeEntry)?;
                    identity::validate_observation_id(text)
                        .map_err(why!(DiscoveryError::UnsafeEntry))?;
                    if matches!(text, "attempts" | "collected_run" | "uploaded_outputs") {
                        return Err(DiscoveryError::UnsupportedLayout);
                    }
                    if components.len() >= MAX_DEPTH {
                        return Err(DiscoveryError::Depth);
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
    Ok(Discovery { runs, deleted })
}

#[cfg(test)]
mod tests;
