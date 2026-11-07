//! Durable immutable source trees and archive receipts.
//!
//! This store never imports a Project implicitly,
//! executes repository code, or retries a publication whose effect is unclear.
use super::archive;
use ml_exp_core::sources::{
    self, CodeFile, CodeManifest, Metadata, Policy, TreeDigest, TreeEntry, TreeKind, TreePlan,
};
use ml_exp_io::directory::{Directory, FileLock, LockOptions, Wait};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::MetadataExt,
    path::Path,
};

const NAMESPACE: &str = "source-revisions";
const SOURCES: &str = "sources";
const IMPORTS: &str = "imports";
const STAGING: &str = "staging";
const LOCK: &str = ".sources.lock";
const POLICY: &str = "policy.json";
const METADATA: &str = "source.json";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("source storage I/O failed")]
    Io(#[from] std::io::Error),
    #[error("source archive or immutable metadata is invalid")]
    Archive,
    #[error("source archive exceeds an explicit bound")]
    Limit,
    #[error("source storage identity or policy is invalid")]
    Identity,
    #[error(
        "source storage operation may have published an effect: {0}; inspect the exact receipt"
    )]
    EffectUncertain(String),
    #[error("source storage is busy")]
    Busy,
}

impl From<sources::Error> for Error {
    fn from(_: sources::Error) -> Self {
        Self::Archive
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct Store {
    namespace: Directory,
    sources: Directory,
    imports: Directory,
    staging: Directory,
    policy: Policy,
}

fn encode(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value).map_err(why!(Error::Identity))?;
    bytes.push(b'\n');
    if bytes.len() > sources::MAX_METADATA_BYTES {
        return Err(Error::Identity);
    }
    Ok(bytes)
}

fn policy_value(policy: &Policy) -> Result<Value> {
    serde_json::to_value(policy).map_err(why!(Error::Identity))
}

impl Store {
    pub fn initialize(workspace_root: &Path, policy: Policy) -> Result<()> {
        policy.validate().map_err(why!(Error::Identity))?;
        let root = Directory::open(workspace_root, false)?;
        let namespace = root.create_new_child(OsStr::new(NAMESPACE))?;
        namespace.publish_new(OsStr::new(POLICY), &encode(&policy_value(&policy)?)?)?;
        let policy_file = namespace.open_regular_read(OsStr::new(POLICY))?;
        ml_exp_io::directory::seal_regular(&policy_file, false)?;
        let lock = namespace.create_new_regular(OsStr::new(LOCK))?;
        lock.sync_all()?;
        namespace.create_new_child(OsStr::new(SOURCES))?;
        namespace.create_new_child(OsStr::new(IMPORTS))?;
        namespace.create_new_child(OsStr::new(STAGING))?;
        Ok(())
    }

    pub fn open(workspace_root: &Path, expected: Policy) -> Result<Self> {
        expected.validate().map_err(why!(Error::Identity))?;
        let root = Directory::open(workspace_root, false)?;
        let namespace = root.child(OsStr::new(NAMESPACE), false)?;
        let policy = serde_json::from_slice::<Policy>(
            &namespace.read_regular(OsStr::new(POLICY), sources::MAX_METADATA_BYTES as u64)?,
        )
        .map_err(why!(Error::Identity))?;
        if policy.validate().is_err()
            || policy.max_archive_bytes != expected.max_archive_bytes
            || policy.max_source_bytes != expected.max_source_bytes
            || policy.max_source_files != expected.max_source_files
        {
            return Err(Error::Identity);
        }
        namespace.open_regular_read(OsStr::new(LOCK))?;
        let sources_dir = namespace.child(OsStr::new(SOURCES), false)?;
        let imports = namespace.child(OsStr::new(IMPORTS), false)?;
        let staging = namespace.child(OsStr::new(STAGING), false)?;
        Ok(Self {
            namespace,
            sources: sources_dir,
            imports,
            staging,
            policy,
        })
    }

    fn lock(&self) -> Result<FileLock> {
        self.namespace
            .lock(
                OsStr::new(LOCK),
                LockOptions {
                    wait: Wait::Until {
                        deadline: std::time::Instant::now() + std::time::Duration::from_secs(5),
                        cancelled: None,
                    },
                    ..Default::default()
                },
            )
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::WouldBlock {
                    Error::Busy
                } else {
                    Error::Io(error)
                }
            })
    }

    fn read_metadata(
        directory: &Directory,
        project: &str,
        source_id: &str,
        policy: &Policy,
    ) -> Result<Metadata> {
        let file = directory.open_regular_read(OsStr::new(METADATA))?;
        ml_exp_io::directory::require_sealed(&file).map_err(|_| Error::Archive)?;
        let mut bytes = Vec::new();
        (&file)
            .take((sources::MAX_METADATA_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > sources::MAX_METADATA_BYTES {
            return Err(Error::Archive);
        }
        let value: Value = serde_json::from_slice(&bytes).map_err(why!(Error::Archive))?;
        Metadata::from_value(&value, project, source_id, policy).map_err(why!(Error::Archive))
    }

    fn verify_tree(
        &self,
        project: &str,
        source_id: &str,
        directory: &Directory,
    ) -> Result<Metadata> {
        let tree = directory.child(OsStr::new("tree"), false)?;
        seal_check(&tree)?;
        let (plan, files) = topology(&tree, "")?;
        let mut digest =
            TreeDigest::new(TreePlan::new(plan, &self.policy).map_err(why!(Error::Archive))?);
        while let Some(request) = digest.next_chunk() {
            let (parent, name) = files.get(request.path).ok_or(Error::Archive)?;
            let mut file = parent.open_regular_read(name)?;
            file.seek(SeekFrom::Start(request.offset))?;
            let mut bytes = vec![0u8; request.len];
            file.read_exact(&mut bytes)?;
            digest = digest.consume(&bytes).map_err(why!(Error::Archive))?;
        }
        let tree_digest = digest.finish().map_err(why!(Error::Archive))?;
        let metadata = Self::read_metadata(directory, project, source_id, &self.policy)?;
        if metadata.tree_digest() != tree_digest {
            return Err(Error::Archive);
        }
        Ok(metadata)
    }

    fn receipt(&self, project: &str, archive_sha: &str) -> Result<Option<(Directory, Metadata)>> {
        let project_dir = match self.imports.child(OsStr::new(project), false) {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(Error::Io(error)),
        };
        let file = match project_dir.open_regular_read(OsStr::new(archive_sha)) {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(Error::Io(error)),
        };
        ml_exp_io::directory::require_sealed(&file).map_err(|_| Error::Archive)?;
        let value: Value = serde_json::from_reader(file).map_err(why!(Error::Archive))?;
        if value.get("schema_version").and_then(Value::as_u64) != Some(1)
            || value.get("project").and_then(Value::as_str) != Some(project)
            || value.get("archive_sha256").and_then(Value::as_str) != Some(archive_sha)
        {
            return Err(Error::Archive);
        }
        let source_id = value
            .get("source_id")
            .and_then(Value::as_str)
            .ok_or(Error::Archive)?;
        sources::validate_source_id(source_id).map_err(why!(Error::Identity))?;
        let project_sources = self.sources.child(OsStr::new(project), false)?;
        let source = project_sources.child(OsStr::new(source_id), false)?;
        let metadata = self.verify_tree(project, source_id, &source)?;
        if value.get("tree_digest").and_then(Value::as_str) != Some(metadata.tree_digest()) {
            return Err(Error::Archive);
        }
        Ok(Some((source, metadata)))
    }

    pub fn find_archive(&self, project: &str, archive_sha: &str) -> Result<Option<Metadata>> {
        sources::validate_archive_digest(archive_sha).map_err(why!(Error::Identity))?;
        let _guard = self.lock()?;
        Ok(self
            .receipt(project, archive_sha)?
            .map(|(_, metadata)| metadata))
    }

    pub fn read(&self, project: &str, source_id: &str) -> Result<Metadata> {
        ml_exp_core::identity::validate_id(project).map_err(why!(Error::Identity))?;
        sources::validate_source_id(source_id).map_err(why!(Error::Identity))?;
        // No lock: a source appears by one atomic rename of a tree already sealed
        // and is never changed or removed afterwards. Receipts do need the lock,
        // because publish_new and seal_regular are two steps.
        let project_sources = self.sources.child(OsStr::new(project), false)?;
        let source = project_sources.child(OsStr::new(source_id), false)?;
        self.verify_tree(project, source_id, &source)
    }

    /// A verified source small enough to deliver to a job. The configured store limit
    /// is about what may be built into an image, and is not this limit.
    fn code_tree(&self, project: &str, source_id: &str) -> Result<(Vec<TreeEntry>, Files)> {
        ml_exp_core::identity::validate_id(project).map_err(why!(Error::Identity))?;
        sources::validate_source_id(source_id).map_err(why!(Error::Identity))?;
        let project_sources = self.sources.child(OsStr::new(project), false)?;
        let source = project_sources.child(OsStr::new(source_id), false)?;
        // Sizes and names first: one that is too large is refused without being read.
        let (entries, files) = topology(&source.child(OsStr::new("tree"), false)?, "")?;
        let bytes = entries
            .iter()
            .map(|entry| match entry.kind {
                TreeKind::File { size, .. } => size,
                TreeKind::Directory => 0,
            })
            .fold(0u64, u64::saturating_add);
        if bytes > sources::MAX_CODE_BYTES
            || entries.len() > ml_exp_core::artifacts::MAX_FILES
            || !listing_of(source_id, &entries).listing_fits()
        {
            return Err(Error::Limit);
        }
        self.verify_tree(project, source_id, &source)?;
        Ok((entries, files))
    }
    pub fn check_code(&self, project: &str, source_id: &str) -> Result<()> {
        self.code_tree(project, source_id).map(|_| ())
    }
    pub fn code_manifest(&self, project: &str, source_id: &str) -> Result<CodeManifest> {
        let (entries, mut opened) = self.code_tree(project, source_id)?;
        let mut directories = Vec::new();
        let mut files = Vec::new();
        for entry in entries {
            let TreeKind::File { size, executable } = entry.kind else {
                directories.push(entry.path);
                continue;
            };
            let (parent, name) = opened.remove(&entry.path).ok_or(Error::Archive)?;
            let (sha256, bytes) =
                ml_exp_io::artifacts::hash(&mut parent.open_regular_read(&name)?)?;
            if bytes != size {
                return Err(Error::Archive);
            }
            files.push(CodeFile {
                path: entry.path,
                bytes,
                sha256,
                executable,
            });
        }
        directories.sort();
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(CodeManifest {
            source_id: source_id.to_owned(),
            directories,
            files,
        })
    }
    pub fn code_content(&self, project: &str, source_id: &str) -> Result<CodeContent> {
        let (entries, files) = self.code_tree(project, source_id)?;
        let sizes: BTreeMap<String, u64> = entries
            .into_iter()
            .filter_map(|entry| match entry.kind {
                TreeKind::File { size, .. } => Some((entry.path, size)),
                TreeKind::Directory => None,
            })
            .collect();
        // Both maps are ordered by path, which is the order the manifest lists.
        let files = files
            .into_iter()
            .map(|(path, (parent, name))| -> Result<_> {
                let size = *sizes.get(&path).ok_or(Error::Archive)?;
                Ok((parent, name, size))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(CodeContent {
            bytes: sizes.values().sum(),
            files,
        })
    }

    pub fn import_archive<R: Read + Seek>(
        &self,
        project: &str,
        stream: &mut R,
        expected_sha: &str,
        created_at: &str,
    ) -> Result<Metadata> {
        ml_exp_core::identity::validate_id(project).map_err(why!(Error::Identity))?;
        sources::validate_archive_digest(expected_sha).map_err(why!(Error::Identity))?;
        if let Some(metadata) = self.find_archive(project, expected_sha)? {
            return Ok(metadata);
        }
        // Stream and stage without the lock: a slow upload must not turn other
        // requests into Busy, and each import owns a randomly named stage.
        stream.seek(SeekFrom::Start(0))?;
        let mut archive_sha = Sha256::new();
        let mut size = 0u64;
        let mut random = [0u8; 8];
        getrandom::fill(&mut random).map_err(why!(Error::Identity))?;
        let staging_name = format!(
            "import-{}-{:016x}",
            &expected_sha[..16],
            u64::from_be_bytes(random)
        );
        let stage = self.staging.create_new_child(OsStr::new(&staging_name))?;
        let mut published = false;
        let result = (|| {
            let archive_file = stage.create_new_regular(OsStr::new("archive.bin"))?;
            let mut archive_file = archive_file;
            let mut buffer = [0u8; 64 * 1024];
            loop {
                let n = stream.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                size = size.checked_add(n as u64).ok_or(Error::Archive)?;
                if size > self.policy.max_archive_bytes {
                    return Err(Error::Archive);
                }
                archive_sha.update(&buffer[..n]);
                archive_file.write_all(&buffer[..n])?;
            }
            archive_file.sync_all()?;
            if format!("{:x}", archive_sha.finalize()) != expected_sha {
                return Err(Error::Archive);
            }
            archive_file.seek(SeekFrom::Start(0))?;
            let tree = stage.create_new_child(OsStr::new("tree"))?;
            archive::extract(&mut archive_file, &tree, &self.policy)?;
            drop(archive_file);
            stage.remove_staged_regular(OsStr::new("archive.bin"))?;
            let (plan, files) = topology(&tree, "")?;
            let mut digest =
                TreeDigest::new(TreePlan::new(plan, &self.policy).map_err(why!(Error::Archive))?);
            while let Some(request) = digest.next_chunk() {
                let (parent, name) = files.get(request.path).ok_or(Error::Archive)?;
                let mut file = parent.open_regular_read(name)?;
                file.seek(SeekFrom::Start(request.offset))?;
                let mut bytes = vec![0u8; request.len];
                file.read_exact(&mut bytes)?;
                digest = digest.consume(&bytes).map_err(why!(Error::Archive))?;
            }
            let tree_digest = digest.finish().map_err(why!(Error::Archive))?;
            let observation = json!({"kind":"archive","sha256":expected_sha});
            let metadata = Metadata::new(
                project,
                &tree_digest,
                created_at,
                &observation,
                &self.policy,
            )
            .map_err(why!(Error::Archive))?;
            stage.publish_new(OsStr::new(METADATA), &encode(&metadata.to_value())?)?;
            let metadata_file = stage.open_regular_read(OsStr::new(METADATA))?;
            ml_exp_io::directory::seal_regular(&metadata_file, false)?;
            seal_tree(&tree)?;
            let source_id = metadata.source_id().to_owned();
            let _guard = self.lock()?;
            // An import that lost the race discards its stage below.
            if let Some((_, existing)) = self.receipt(project, expected_sha)? {
                return Ok(existing);
            }
            let project_sources = self.sources.child(OsStr::new(project), true)?;
            if project_sources.entry_exists(OsStr::new(&source_id))? {
                let existing = project_sources.child(OsStr::new(&source_id), false)?;
                let verified = self.verify_tree(project, &source_id, &existing)?;
                let imports_project = self.imports.child(OsStr::new(project), true)?;
                let receipt = json!({"schema_version":1,"project":project,"archive_sha256":expected_sha,"source_id":source_id,"tree_digest":verified.tree_digest()});
                imports_project.publish_new(OsStr::new(expected_sha), &encode(&receipt)?)?;
                let receipt_file = imports_project.open_regular_read(OsStr::new(expected_sha))?;
                ml_exp_io::directory::seal_regular(&receipt_file, false)?;
                return Ok(verified);
            }
            let published_directory =
                match stage.publish_directory(&project_sources, OsStr::new(&source_id)) {
                    Ok(directory) => directory,
                    Err(error) => {
                        // The atomic rename may already have happened when
                        // this API reports an error. Never delete the stage.
                        published = true;
                        return Err(Error::EffectUncertain(format!(
                            "publish_directory:{:?}",
                            error.kind()
                        )));
                    }
                };
            published = true;
            if published_directory.source_mode(true).is_err() {
                return Err(Error::EffectUncertain("source_root_seal".into()));
            }
            let verified = self.verify_tree(project, &source_id, &published_directory)?;
            let imports_project = self.imports.child(OsStr::new(project), true)?;
            let receipt = json!({"schema_version":1,"project":project,"archive_sha256":expected_sha,"source_id":source_id,"tree_digest":verified.tree_digest()});
            imports_project.publish_new(OsStr::new(expected_sha), &encode(&receipt)?)?;
            let receipt_file = imports_project.open_regular_read(OsStr::new(expected_sha))?;
            ml_exp_io::directory::seal_regular(&receipt_file, false)?;
            Ok(verified)
        })();
        if published {
            return result;
        }
        // The staged tree is unpublished and is never used as an authority.
        // Cleanup is part of the bounded operation: a successful receipt with
        // a leaked staging tree is still safe to query, but must be reported
        // as uncertain so an operator can inspect the exact archive receipt.
        match (result, cleanup_stage(&self.staging, &stage)) {
            (Ok(metadata), Ok(())) => Ok(metadata),
            (Ok(_), Err(_)) | (Err(_), Err(_)) => {
                Err(Error::EffectUncertain("staging_cleanup".into()))
            }
            (Err(error), Ok(())) => Err(error),
        }
    }
}

/// The listing's shape with placeholder hashes, which have the real ones' length.
fn listing_of(source_id: &str, entries: &[TreeEntry]) -> CodeManifest {
    let mut manifest = CodeManifest {
        source_id: source_id.to_owned(),
        directories: Vec::new(),
        files: Vec::new(),
    };
    for entry in entries {
        match entry.kind {
            TreeKind::Directory => manifest.directories.push(entry.path.clone()),
            TreeKind::File { size, executable } => manifest.files.push(CodeFile {
                path: entry.path.clone(),
                bytes: size,
                sha256: "0".repeat(64),
                executable,
            }),
        }
    }
    manifest
}
/// The files of a code source one after another, in the order of their manifest. The
/// manifest has the sizes, so this needs no framing and has no paths for a worker to trust.
pub struct CodeContent {
    pub bytes: u64,
    files: Vec<(Directory, OsString, u64)>,
}
impl CodeContent {
    pub fn each_chunk(self, mut send: impl FnMut(Vec<u8>) -> std::io::Result<()>) -> Result<()> {
        for (parent, name, size) in self.files {
            let mut file = parent.open_regular_read(&name)?.take(size);
            let mut sent = 0;
            loop {
                let mut chunk = vec![0; 256 * 1024];
                let count = file.read(&mut chunk)?;
                if count == 0 {
                    break;
                }
                chunk.truncate(count);
                send(chunk)?;
                sent += count as u64;
            }
            if sent != size {
                return Err(Error::Archive);
            }
        }
        Ok(())
    }
}

type Files = BTreeMap<String, (Directory, OsString)>;

fn topology(directory: &Directory, prefix: &str) -> Result<(Vec<TreeEntry>, Files)> {
    let mut entries = Vec::new();
    let mut files = BTreeMap::new();
    let names = directory.entry_names_limited(sources::MAX_TOPOLOGY_ENTRIES)?;
    for name in names {
        let name_str = name.to_str().ok_or(Error::Archive)?;
        let path = if prefix.is_empty() {
            name_str.to_owned()
        } else {
            format!("{prefix}/{name_str}")
        };
        if let Ok(child) = directory.child(&name, false) {
            entries.push(TreeEntry {
                path: path.clone(),
                kind: TreeKind::Directory,
            });
            let (mut nested, nested_files) = topology(&child, &path)?;
            entries.append(&mut nested);
            files.extend(nested_files);
        } else {
            let file = directory.open_regular_read(&name)?;
            let metadata = file.metadata()?;
            entries.push(TreeEntry {
                path: path.clone(),
                kind: TreeKind::File {
                    size: metadata.len(),
                    executable: metadata.mode() & 0o111 != 0,
                },
            });
            files.insert(path, (directory.clone(), name));
        }
    }
    Ok((entries, files))
}

fn seal_tree(directory: &Directory) -> Result<()> {
    for name in directory.entry_names_limited(sources::MAX_TOPOLOGY_ENTRIES)? {
        if let Ok(child) = directory.child(&name, false) {
            seal_tree(&child)?;
            child.source_mode(true)?;
        }
    }
    directory.source_mode(true)?;
    Ok(())
}

fn seal_check(directory: &Directory) -> Result<()> {
    if directory.mode()? & 0o222 != 0 {
        return Err(Error::Archive);
    }
    for name in directory.entry_names_limited(sources::MAX_TOPOLOGY_ENTRIES)? {
        if let Ok(child) = directory.child(&name, false) {
            seal_check(&child)?;
        } else {
            let file = directory.open_regular_read(&name)?;
            ml_exp_io::directory::require_sealed(&file)?;
        }
    }
    Ok(())
}

fn cleanup_stage(parent: &Directory, stage: &Directory) -> Result<()> {
    stage.source_mode(false)?;
    for name in stage.entry_names_limited(sources::MAX_TOPOLOGY_ENTRIES)? {
        if let Ok(child) = stage.child(&name, false) {
            cleanup_stage(stage, &child)?;
        } else {
            stage.remove_staged_regular(&name)?;
        }
    }
    parent.remove_staged_empty_child(stage)?;
    Ok(())
}
