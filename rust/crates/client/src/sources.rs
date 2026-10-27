//! Immutable source requests have fixed routes and explicit archive identities.
//! Repository files are never executed. Packing and upload are bounded I/O shells;
//! all path policy, request planning, and response binding remain pure.
use crate::{Client, Error, session};
use ml_exp_core::{identity, sources as source_contract};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

pub const MAX_ARCHIVE_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_SOURCE_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_SOURCE_NODES: usize = 20_000;

/// An anonymous temporary archive, rewound to byte zero. It contains no token.
pub struct PreparedArchive {
    pub file: File,
    pub sha256: String,
    pub bytes: u64,
}
/// Callers cannot replace the route, length, or archive identity after planning.
#[derive(Clone, Debug)]
pub struct ArchivePlan {
    path: String,
    project: String,
    sha256: String,
    bytes: u64,
}
impl ArchivePlan {
    pub(crate) fn path(&self) -> &str {
        &self.path
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn archive_sha256(&self) -> &str {
        &self.sha256
    }
}
fn invalid() -> Error {
    Error::Observation("invalid source identity or archive limit; no source request was sent")
}
fn packing() -> Error {
    Error::Observation(
        "cannot pack source: use a stable regular directory without links, special files, credential-looking paths, or unsafe permissions, within the source limits",
    )
}
fn sha256(value: &str) -> bool {
    source_contract::validate_archive_digest(value).is_ok()
}
fn source_digest(source_id: &str) -> Result<&str, Error> {
    source_contract::validate_source_id(source_id).map_err(|_| invalid())?;
    Ok(&source_id["source.".len()..])
}
pub fn plan_archive(project: &str, digest: &str, bytes: u64) -> Result<ArchivePlan, Error> {
    identity::validate_id(project).map_err(|_| invalid())?;
    if !sha256(digest) || bytes == 0 || bytes > MAX_ARCHIVE_BYTES {
        return Err(invalid());
    }
    Ok(ArchivePlan {
        path: format!("/api/source-imports/archive?project={project}&sha256={digest}"),
        project: project.into(),
        sha256: digest.into(),
        bytes,
    })
}
fn enabled(health: &Value) -> Result<(), Error> {
    session::validate_health(health)?;
    if health.get("source_imports") == Some(&Value::Bool(false))
        || !health
            .get("capabilities")
            .and_then(Value::as_array)
            .is_some_and(|items| items.iter().any(|item| item == "source-archive.v1"))
    {
        return Err(Error::Observation(
            "daemon has not enabled source-archive.v1; no source request was sent",
        ));
    }
    Ok(())
}
fn bound<'a>(project: &str, value: &'a Value) -> Result<&'a str, Error> {
    let source_id = value
        .get("source_id")
        .and_then(Value::as_str)
        .ok_or(Error::Response)?;
    let digest = source_digest(source_id).map_err(|_| Error::Response)?;
    if value.get("project").and_then(Value::as_str) != Some(project)
        || value.get("tree_digest").and_then(Value::as_str)
            != Some(format!("sha256:{digest}").as_str())
    {
        return Err(Error::Response);
    }
    Ok(source_id)
}
fn archive_observation(value: &Value, expected: Option<&str>) -> Result<(), Error> {
    let observation = value.get("observation").ok_or(Error::Response)?;
    let digest = observation
        .get("sha256")
        .and_then(Value::as_str)
        .filter(|digest| sha256(digest))
        .ok_or(Error::Response)?;
    if observation.get("kind").and_then(Value::as_str) != Some("archive")
        || expected.is_some_and(|expected| expected != digest)
    {
        return Err(Error::Response);
    }
    Ok(())
}
/// The archive SHA identifies transport bytes. It is deliberately not compared
/// with the source tree digest, which also includes paths and executable bits.
pub fn validate_result(plan: &ArchivePlan, value: Value) -> Result<Value, Error> {
    validate_publication(&plan.project, &plan.sha256, value)
}
fn validate_publication(project: &str, digest: &str, value: Value) -> Result<Value, Error> {
    bound(project, &value)?;
    archive_observation(&value, Some(digest))?;
    if value.pointer("/registration/existing") != Some(&Value::Bool(true)) {
        return Err(Error::Response);
    }
    Ok(value)
}
pub fn validate_metadata(project: &str, source_id: &str, value: Value) -> Result<Value, Error> {
    identity::validate_id(project).map_err(|_| invalid())?;
    source_digest(source_id)?;
    source_contract::Metadata::from_value(
        &value,
        project,
        source_id,
        &source_contract::Policy::default(),
    )
    .map_err(|_| Error::Response)?;
    Ok(value)
}
/// A lost or malformed POST response is not proof of absence. No upload is
/// retried here; the caller must inspect a confirmed exact source identity.
pub fn upload(
    client: &Client,
    plan: &ArchivePlan,
    file: File,
    health: &Value,
) -> Result<Value, Error> {
    enabled(health)?;
    let prepared = inspect_archive(file)?;
    if prepared.bytes != plan.bytes || prepared.sha256 != plan.sha256 {
        return Err(Error::Observation(
            "source archive changed after planning; no source request was sent",
        ));
    }
    client
        .upload_source(plan, prepared.file)
        .and_then(|value| validate_result(plan, value))
        .map_err(|error| {
            if matches!(error, Error::Transport | Error::Response)
                || matches!(&error, Error::Http(value) if value.status >= 500 || value.status == 408)
            {
                Error::SourceOutcomeUnknown(Box::new(error))
            } else {
                error
            }
        })
}

/// One explicit client workflow: validate the daemon capability, create a
/// bounded archive, bind its transport hash, then send exactly one POST.
pub fn import_directory(
    client: &Client,
    project: &str,
    directory: &Path,
    health: &Value,
) -> Result<Value, Error> {
    identity::validate_id(project).map_err(|_| invalid())?;
    enabled(health)?;
    let prepared = pack_directory(directory)?;
    let plan = plan_archive(project, &prepared.sha256, prepared.bytes)?;
    upload(client, &plan, prepared.file, health)
}
pub fn read(
    client: &Client,
    project: &str,
    source_id: &str,
    health: &Value,
) -> Result<Value, Error> {
    identity::validate_id(project).map_err(|_| invalid())?;
    source_digest(source_id)?;
    enabled(health)?;
    let value = client.get(&format!("/api/projects/{project}/sources/{source_id}"))?;
    validate_metadata(project, source_id, value)
}
/// Recover a publication receipt by its original archive identity. A missing
/// receipt does not establish that the source was never published; this read
/// never triggers a new upload or repairs the server state.
pub fn lookup_archive(
    client: &Client,
    project: &str,
    digest: &str,
    health: &Value,
) -> Result<Value, Error> {
    identity::validate_id(project).map_err(|_| invalid())?;
    if !sha256(digest) {
        return Err(invalid());
    }
    enabled(health)?;
    let value = client.get(&format!("/api/projects/{project}/source-imports/{digest}"))?;
    validate_publication(project, digest, value)
}

/// Hash a fixed initial extent and rewind it for a finite streaming upload.
/// Anonymous prepared files have link count zero, so no pathname is consulted.
pub fn inspect_archive(mut file: File) -> Result<PreparedArchive, Error> {
    let before = file.metadata().map_err(|_| packing())?;
    if !before.is_file() || before.len() == 0 || before.len() > MAX_ARCHIVE_BYTES {
        return Err(packing());
    }
    file.seek(SeekFrom::Start(0)).map_err(|_| packing())?;
    let mut digest = Sha256::new();
    let mut reader = (&mut file).take(before.len());
    let mut buffer = [0; 64 * 1024];
    let mut bytes = 0;
    loop {
        let count = reader.read(&mut buffer).map_err(|_| packing())?;
        if count == 0 {
            break;
        }
        bytes += count as u64;
        digest.update(&buffer[..count]);
    }
    let after = file.metadata().map_err(|_| packing())?;
    if bytes != before.len() || !same_metadata(&before, &after) {
        return Err(packing());
    }
    file.seek(SeekFrom::Start(0)).map_err(|_| packing())?;
    Ok(PreparedArchive {
        file,
        sha256: format!("{:x}", digest.finalize()),
        bytes,
    })
}
fn same_metadata(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    let base = before.len() == after.len()
        && before.modified().ok() == after.modified().ok()
        && before.is_file() == after.is_file();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        base && before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.mode() == after.mode()
            && before.nlink() == after.nlink()
            && before.uid() == after.uid()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec()
    }
    #[cfg(not(unix))]
    {
        base
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PathPolicy {
    Include,
    Skip,
    Reject,
}
fn path_policy(path: &str) -> PathPolicy {
    if path.is_empty()
        || path.len() > 4096
        || path.starts_with('/')
        || path.contains(['\\', '\0'])
        || path.split('/').any(|p| matches!(p, "" | "." | ".."))
    {
        return PathPolicy::Reject;
    }
    for part in path.split('/') {
        if matches!(part, ".git" | ".venv" | "__pycache__") {
            return PathPolicy::Skip;
        }
        // The existing client additionally rejects `.env*`, even names that
        // the server's narrower protected-path policy would accept.
        if part.to_lowercase().starts_with(".env") {
            return PathPolicy::Reject;
        }
    }
    match source_contract::archive_path(
        path,
        source_contract::MemberKind::File,
        &source_contract::Policy::default(),
    ) {
        Ok(Some(normalized)) if normalized == path => PathPolicy::Include,
        _ => PathPolicy::Reject,
    }
}

struct BoundedWriter {
    file: File,
    bytes: u64,
}
impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .checked_add(bytes.len() as u64)
            .is_none_or(|length| length > MAX_ARCHIVE_BYTES)
        {
            return Err(std::io::Error::other("source archive exceeds limit"));
        }
        let count = self.file.write(bytes)?;
        self.bytes += count as u64;
        Ok(count)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

/// Pack regular files only, with stable order, gzip mtime zero and normalized
/// tar ownership/timestamps. Executable intent is retained; repository code is
/// never run. The initial names, file metadata, and directory attachment are
/// rechecked before success. This is not an atomic snapshot against root or a
/// hostile process with the same UID changing and restoring the source tree.
#[cfg(unix)]
pub fn pack_directory(path: &Path) -> Result<PreparedArchive, Error> {
    use ml_exp_io::directory::Directory;
    use std::{ffi::OsString, os::unix::fs::MetadataExt, path::Component};
    const MAX_DIRECTORIES: usize = 1024;
    const MAX_DEPTH: usize = 32;
    struct Entry {
        parent: Directory,
        name: OsString,
        path: String,
        metadata: std::fs::Metadata,
    }
    struct Names {
        parent: Directory,
        names: Vec<OsString>,
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map_err(|_| packing())?.join(path)
    };
    let mut normalized = std::path::PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::RootDir | Component::Normal(_) => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            _ => return Err(packing()),
        }
    }
    let root = Directory::open(&normalized, false).map_err(|_| packing())?;
    root.verify_private_parent().map_err(|_| packing())?;
    let mut stack = vec![(root, String::new(), 0)];
    let mut entries = Vec::new();
    let mut observed = Vec::new();
    let mut expanded = 0_u64;
    let mut nodes = 0_usize;
    let mut path_bytes = 0_usize;
    while let Some((directory, relative, depth)) = stack.pop() {
        if depth > MAX_DEPTH || observed.len() >= MAX_DIRECTORIES {
            return Err(packing());
        }
        directory.verify_private_parent().map_err(|_| packing())?;
        let names = directory
            .entry_names_limited(MAX_SOURCE_NODES.saturating_sub(nodes))
            .map_err(|_| packing())?;
        nodes += names.len();
        if nodes > MAX_SOURCE_NODES {
            return Err(packing());
        }
        for name in &names {
            let name_str = name.to_str().ok_or_else(packing)?;
            let path = if relative.is_empty() {
                name_str.into()
            } else {
                format!("{relative}/{name_str}")
            };
            match path_policy(&path) {
                PathPolicy::Skip => continue,
                PathPolicy::Reject => return Err(packing()),
                PathPolicy::Include => {}
            }
            path_bytes = path_bytes.checked_add(path.len()).ok_or_else(packing)?;
            if path_bytes > source_contract::MAX_TOPOLOGY_BYTES {
                return Err(packing());
            }
            if let Ok(child) = directory.child(name, false) {
                if stack.len() + observed.len() >= MAX_DIRECTORIES {
                    return Err(packing());
                }
                stack.push((child, path, depth + 1));
            } else {
                let file = directory.open_regular_read(name).map_err(|_| packing())?;
                let metadata = file.metadata().map_err(|_| packing())?;
                expanded = expanded.checked_add(metadata.len()).ok_or_else(packing)?;
                if expanded > MAX_SOURCE_BYTES {
                    return Err(packing());
                }
                entries.push(Entry {
                    parent: directory.clone(),
                    name: name.clone(),
                    path,
                    metadata,
                });
            }
        }
        observed.push(Names {
            parent: directory,
            names,
        });
    }
    if entries.is_empty() {
        return Err(packing());
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    let writer = BoundedWriter {
        file: tempfile::tempfile().map_err(|_| packing())?,
        bytes: 0,
    };
    let compressed = flate2::GzBuilder::new()
        .mtime(0)
        .write(writer, flate2::Compression::default());
    let mut archive = tar::Builder::new(compressed);
    for entry in &entries {
        let mut file = entry
            .parent
            .open_regular_read(&entry.name)
            .map_err(|_| packing())?;
        if !same_metadata(&entry.metadata, &file.metadata().map_err(|_| packing())?) {
            return Err(packing());
        }
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        header.set_mode(if entry.metadata.mode() & 0o111 != 0 {
            0o500
        } else {
            0o400
        });
        header.set_size(entry.metadata.len());
        header.set_cksum();
        let mut body = (&mut file).take(entry.metadata.len());
        archive
            .append_data(&mut header, &entry.path, &mut body)
            .map_err(|_| packing())?;
        if body.limit() != 0
            || !same_metadata(&entry.metadata, &file.metadata().map_err(|_| packing())?)
        {
            return Err(packing());
        }
        entry
            .parent
            .validate_regular_entry(&entry.name, &file)
            .map_err(|_| packing())?;
    }
    let writer = archive
        .into_inner()
        .and_then(flate2::write::GzEncoder::finish)
        .map_err(|_| packing())?;
    writer.file.sync_all().map_err(|_| packing())?;
    for entry in &entries {
        let file = entry
            .parent
            .open_regular_read(&entry.name)
            .map_err(|_| packing())?;
        if !same_metadata(&entry.metadata, &file.metadata().map_err(|_| packing())?) {
            return Err(packing());
        }
    }
    for proof in observed {
        proof
            .parent
            .verify_private_parent()
            .map_err(|_| packing())?;
        if proof
            .parent
            .entry_names_limited(MAX_SOURCE_NODES)
            .map_err(|_| packing())?
            != proof.names
        {
            return Err(packing());
        }
    }
    inspect_archive(writer.file)
}

#[cfg(not(unix))]
pub fn pack_directory(_path: &Path) -> Result<PreparedArchive, Error> {
    Err(Error::Observation(
        "native source directory packing currently requires Unix descriptor-safe filesystem operations",
    ))
}
