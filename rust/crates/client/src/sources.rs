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

fn policy() -> source_contract::Policy {
    source_contract::Policy::default()
}

/// An anonymous temporary archive, rewound to byte zero. It contains no token.
pub struct PreparedArchive {
    pub file: File,
    pub sha256: String,
    pub bytes: u64,
    /// The members when this side packed it; an archive read back does not list them.
    pub paths: Vec<String>,
    /// Paths, executable bits and contents, so it does not change with the compression.
    pub content_sha256: String,
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
        "cannot pack source: use a stable regular directory without symlinks, special files or credential-looking paths, within the source limits",
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
    if !sha256(digest) || bytes == 0 || bytes > policy().max_archive_bytes {
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
    if !health
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
    if !before.is_file() || before.len() == 0 || before.len() > policy().max_archive_bytes {
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
    let sha256 = format!("{:x}", digest.finalize());
    Ok(PreparedArchive {
        file,
        content_sha256: sha256.clone(),
        sha256,
        bytes,
        paths: Vec::new(),
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
/// `ml-exp results` names its downloads this way, so a download in the source directory
/// does not become part of the source and a different job.
pub const RESULTS_SUFFIX: &str = ".ml-exp-results.tar";
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
        if matches!(part, ".git" | ".venv" | "__pycache__") || part.ends_with(RESULTS_SUFFIX) {
            return PathPolicy::Skip;
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

/// Hashes exactly the bytes that go into the archive, without reading them twice.
struct Hashing<'a, R> {
    inner: R,
    digest: &'a mut Sha256,
}
impl<R: Read> Read for Hashing<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.digest.update(&buffer[..count]);
        Ok(count)
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
            .is_none_or(|length| length > policy().max_archive_bytes)
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

/// What git would keep in this directory: tracked files, and untracked ones it does
/// not ignore, by each nested repository's own rules. A listed directory with no
/// repository of its own is an uninitialized submodule, and holds nothing.
struct Selection {
    files: std::collections::BTreeSet<String>,
    directories: std::collections::BTreeSet<String>,
}
impl Selection {
    fn new(paths: Vec<String>) -> Self {
        let mut directories = std::collections::BTreeSet::new();
        for path in &paths {
            let mut child = path.as_str();
            while let Some((parent, _)) = child.rsplit_once('/') {
                directories.insert(parent.to_owned());
                child = parent;
            }
        }
        Self {
            files: paths.into_iter().collect(),
            directories,
        }
    }
    fn includes(&self, path: &str) -> bool {
        if self.files.contains(path) || self.directories.contains(path) {
            return true;
        }
        let mut child = path;
        while let Some((parent, _)) = child.rsplit_once('/') {
            if self.files.contains(parent) {
                return true;
            }
            child = parent;
        }
        false
    }
}
/// None outside a git work tree. Inside one, git must answer: quietly uploading every
/// file instead would send what .gitignore keeps out, including secrets and data.
/// fsmonitor is the one setting that would make this listing run repository code.
fn git_selection(directory: &Path) -> Result<Option<Selection>, Error> {
    // A worktree or submodule has a `.git` file; a repository's directory has a HEAD. An
    // empty directory named `.git` is not a repository, so it does not count.
    let git = |parent: &Path| parent.join(".git/HEAD").exists() || parent.join(".git").is_file();
    if !directory.ancestors().any(git) {
        return Ok(None);
    }
    let unreadable = || {
        Error::Submit(format!(
            "{} is in a git repository, but git could not list its files, so nothing was \
             uploaded. Check that git is installed and that this directory is a safe.directory",
            directory.display()
        ))
    };
    let list = |root: &Path| -> Option<Vec<String>> {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["-c", "core.fsmonitor=false", "ls-files", "-z"])
            .args(["--cached", "--others", "--exclude-standard"])
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        output
            .stdout
            .split(|byte| *byte == 0)
            .filter(|path| !path.is_empty())
            .map(|path| std::str::from_utf8(path).ok().map(str::to_owned))
            .collect()
    };
    let mut files = Vec::new();
    let mut pending = vec![(directory.to_path_buf(), String::new())];
    while let Some((root, prefix)) = pending.pop() {
        for path in list(&root).ok_or_else(unreadable)? {
            // A nested repository is listed as `nested/` and a submodule as `nested`.
            let path = path.trim_end_matches('/');
            let full = member(&prefix, path);
            let inner = root.join(path);
            // lstat: a symlink back into the repository must not be walked as one.
            let directory = std::fs::symlink_metadata(&inner).is_ok_and(|meta| meta.is_dir());
            if directory && inner.join(".git").exists() {
                pending.push((inner, full));
            } else {
                files.push(full);
            }
        }
    }
    Ok(Some(Selection::new(files)))
}
fn member(relative: &str, name: &str) -> String {
    if relative.is_empty() {
        name.into()
    } else {
        format!("{relative}/{name}")
    }
}
/// Says which file stopped the upload and how to leave it out.
fn refused(path: &str, why: &str) -> Error {
    Error::Submit(format!(
        "cannot pack {path}: {why}. Remove it, or ignore it in .gitignore; \
         upload data separately with `ml-exp data upload`"
    ))
}

/// Pack regular files only, with stable order, gzip mtime zero and normalized
/// tar ownership/timestamps. Executable intent is retained; repository code is
/// never run. The initial names and file metadata are rechecked before success,
/// so an edit during packing fails rather than producing a torn archive.
pub fn pack_directory(path: &Path) -> Result<PreparedArchive, Error> {
    use ml_exp_io::directory::Directory;
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    use std::{ffi::OsString, path::Component};
    struct Entry {
        parent: Directory,
        name: OsString,
        path: String,
        metadata: std::fs::Metadata,
    }
    struct Names {
        parent: Directory,
        relative: String,
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
            #[cfg(windows)]
            Component::Prefix(_) => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            _ => return Err(packing()),
        }
    }
    let root = Directory::open(&normalized, false).map_err(|_| packing())?;
    let limits = policy();
    let selection = git_selection(&normalized)?;
    // Ignored names are not part of the source, so they neither count nor must stay still.
    let visible = |relative: &str, names: Vec<OsString>| -> Vec<OsString> {
        match &selection {
            None => names,
            Some(selection) => names
                .into_iter()
                .filter(|name| {
                    name.to_str()
                        .is_some_and(|name| selection.includes(&member(relative, name)))
                })
                .collect(),
        }
    };
    let read_limit = |remaining: usize| {
        if selection.is_some() {
            usize::MAX
        } else {
            remaining
        }
    };
    let too_many = || {
        Error::Submit(format!(
            "cannot pack the source: it holds more than {} files. Ignore data and outputs in \
             .gitignore, and upload data separately with `ml-exp data upload`",
            limits.max_source_files
        ))
    };
    let mut stack = vec![(root, String::new())];
    let mut entries = Vec::new();
    let mut observed = Vec::new();
    let mut expanded = 0_u64;
    let mut nodes = 0_usize;
    let mut path_bytes = 0_usize;
    while let Some((directory, relative)) = stack.pop() {
        let names = directory
            .entry_names_limited(read_limit(limits.max_source_files.saturating_sub(nodes)))
            .map_err(|_| too_many())?;
        let names = visible(&relative, names);
        nodes += names.len();
        if nodes > limits.max_source_files {
            return Err(too_many());
        }
        for name in &names {
            let name_str = name.to_str().ok_or_else(packing)?;
            let path = member(&relative, name_str);
            match path_policy(&path) {
                PathPolicy::Skip => continue,
                PathPolicy::Reject => {
                    return Err(refused(&path, "its name is not allowed in an upload"));
                }
                PathPolicy::Include => {}
            }
            path_bytes = path_bytes.checked_add(path.len()).ok_or_else(packing)?;
            if path_bytes > source_contract::MAX_TOPOLOGY_BYTES {
                return Err(packing());
            }
            if let Ok(child) = directory.child(name, false) {
                stack.push((child, path));
            } else {
                let file = directory.open_regular_read(name).map_err(|_| {
                    refused(
                        &path,
                        "only regular files upload, and this is a symlink or special file",
                    )
                })?;
                let metadata = file.metadata().map_err(|_| packing())?;
                expanded = expanded.checked_add(metadata.len()).ok_or_else(packing)?;
                if expanded > limits.max_source_bytes {
                    return Err(refused(
                        &path,
                        &format!(
                            "with it the source exceeds {} MiB",
                            limits.max_source_bytes / (1024 * 1024)
                        ),
                    ));
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
            relative,
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
    let mut content = Sha256::new();
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
        #[cfg(unix)]
        let executable = entry.metadata.mode() & 0o111 != 0;
        #[cfg(not(unix))]
        let executable = false;
        header.set_mode(if executable { 0o500 } else { 0o400 });
        header.set_size(entry.metadata.len());
        header.set_cksum();
        content.update(entry.path.as_bytes());
        content.update([0, u8::from(executable)]);
        content.update(entry.metadata.len().to_le_bytes());
        let mut body = Hashing {
            inner: (&mut file).take(entry.metadata.len()),
            digest: &mut content,
        };
        archive
            .append_data(&mut header, &entry.path, &mut body)
            .map_err(|_| {
                refused(
                    &entry.path,
                    &format!(
                        "with it the compressed source exceeds {} MiB, or it changed while packing",
                        limits.max_archive_bytes / (1024 * 1024)
                    ),
                )
            })?;
        if body.inner.limit() != 0
            || !same_metadata(&entry.metadata, &file.metadata().map_err(|_| packing())?)
        {
            return Err(packing());
        }
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
        let names = proof
            .parent
            .entry_names_limited(read_limit(limits.max_source_files))
            .map_err(|_| packing())?;
        if visible(&proof.relative, names) != proof.names {
            return Err(packing());
        }
    }
    let mut prepared = inspect_archive(writer.file)?;
    prepared.paths = entries.into_iter().map(|entry| entry.path).collect();
    prepared.content_sha256 = format!("{:x}", content.finalize());
    Ok(prepared)
}
