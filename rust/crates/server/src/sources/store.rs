//! Durable immutable source trees and archive receipts.
//!
//! This is an isolated preview store. It never imports a Project implicitly,
//! executes repository code, or retries a publication whose effect is unclear.
use super::archive;
use fs2::FileExt;
use ml_exp_core::sources::{self, Metadata, Policy, TreeDigest, TreeEntry, TreeKind, TreePlan};
use ml_exp_io::directory::Directory;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    fs::File,
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
    #[error("source storage operation may have published an effect; inspect the exact receipt")]
    EffectUncertain,
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
    anchor: File,
    policy: Policy,
}

struct Guard<'a> {
    store: &'a Store,
    file: &'a File,
}
impl Guard<'_> {
    fn check(&self) -> Result<()> {
        self.store.namespace.verify_private_parent()?;
        self.store
            .namespace
            .validate_regular_entry(OsStr::new(LOCK), self.file)?;
        Ok(())
    }
}
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        let _ = FileExt::unlock(self.file);
    }
}

fn encode(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value).map_err(|_| Error::Identity)?;
    bytes.push(b'\n');
    if bytes.len() > sources::MAX_METADATA_BYTES {
        return Err(Error::Identity);
    }
    Ok(bytes)
}

fn policy_value(policy: &Policy) -> Result<Value> {
    serde_json::to_value(policy).map_err(|_| Error::Identity)
}

impl Store {
    pub fn initialize(workspace_root: &Path, policy: Policy) -> Result<()> {
        policy.validate().map_err(|_| Error::Identity)?;
        let root = Directory::open(workspace_root, false)?;
        let namespace = root.create_new_child(OsStr::new(NAMESPACE))?;
        namespace.publish_new(OsStr::new(POLICY), &encode(&policy_value(&policy)?)?)?;
        let policy_file = namespace.open_regular_read(OsStr::new(POLICY))?;
        namespace.seal_regular(OsStr::new(POLICY), &policy_file, false)?;
        let lock = namespace.create_new_regular(OsStr::new(LOCK))?;
        lock.sync_all()?;
        namespace.create_new_child(OsStr::new(SOURCES))?;
        namespace.create_new_child(OsStr::new(IMPORTS))?;
        namespace.create_new_child(OsStr::new(STAGING))?;
        namespace.verify_private_parent()?;
        Ok(())
    }

    pub fn open(workspace_root: &Path, expected: Policy) -> Result<Self> {
        expected.validate().map_err(|_| Error::Identity)?;
        let root = Directory::open(workspace_root, false)?;
        let namespace = root.child(OsStr::new(NAMESPACE), false)?;
        let policy = serde_json::from_slice::<Policy>(
            &namespace.read_regular(OsStr::new(POLICY), sources::MAX_METADATA_BYTES as u64)?,
        )
        .map_err(|_| Error::Identity)?;
        if policy.validate().is_err()
            || policy.max_archive_bytes != expected.max_archive_bytes
            || policy.max_source_bytes != expected.max_source_bytes
            || policy.max_source_files != expected.max_source_files
        {
            return Err(Error::Identity);
        }
        let anchor = namespace.open_existing_lock(OsStr::new(LOCK))?;
        let sources_dir = namespace.child(OsStr::new(SOURCES), false)?;
        let imports = namespace.child(OsStr::new(IMPORTS), false)?;
        let staging = namespace.child(OsStr::new(STAGING), false)?;
        Ok(Self {
            namespace,
            sources: sources_dir,
            imports,
            staging,
            anchor,
            policy,
        })
    }

    fn lock(&self) -> Result<Guard<'_>> {
        let end = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match self.anchor.try_lock_exclusive() {
                Ok(()) => break,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && std::time::Instant::now() < end =>
                {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    return Err(Error::Busy);
                }
                Err(error) => return Err(Error::Io(error)),
            }
        }
        let guard = Guard {
            store: self,
            file: &self.anchor,
        };
        guard.check()?;
        Ok(guard)
    }

    fn read_metadata(
        directory: &Directory,
        project: &str,
        source_id: &str,
        policy: &Policy,
    ) -> Result<Metadata> {
        let file = directory.open_regular_read(OsStr::new(METADATA))?;
        directory
            .validate_immutable_regular_entry(OsStr::new(METADATA), &file)
            .map_err(|_| Error::Archive)?;
        let mut bytes = Vec::new();
        (&file)
            .take((sources::MAX_METADATA_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > sources::MAX_METADATA_BYTES {
            return Err(Error::Archive);
        }
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| Error::Archive)?;
        Metadata::from_value(&value, project, source_id, policy).map_err(|_| Error::Archive)
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
            TreeDigest::new(TreePlan::new(plan, &self.policy).map_err(|_| Error::Archive)?);
        while let Some(request) = digest.next_chunk() {
            let (parent, name) = files.get(request.path).ok_or(Error::Archive)?;
            let mut file = parent.open_regular_read(name)?;
            file.seek(SeekFrom::Start(request.offset))?;
            let mut bytes = vec![0u8; request.len];
            file.read_exact(&mut bytes)?;
            digest = digest.consume(&bytes).map_err(|_| Error::Archive)?;
            parent.validate_regular_entry(name, &file)?;
        }
        let tree_digest = digest.finish().map_err(|_| Error::Archive)?;
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
        project_dir
            .validate_immutable_regular_entry(OsStr::new(archive_sha), &file)
            .map_err(|_| Error::Archive)?;
        let value: Value = serde_json::from_reader(file).map_err(|_| Error::Archive)?;
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
        sources::validate_source_id(source_id).map_err(|_| Error::Identity)?;
        let project_sources = self.sources.child(OsStr::new(project), false)?;
        let source = project_sources.child(OsStr::new(source_id), false)?;
        let metadata = self.verify_tree(project, source_id, &source)?;
        if value.get("tree_digest").and_then(Value::as_str) != Some(metadata.tree_digest()) {
            return Err(Error::Archive);
        }
        Ok(Some((source, metadata)))
    }

    pub fn find_archive(&self, project: &str, archive_sha: &str) -> Result<Option<Metadata>> {
        sources::validate_archive_digest(archive_sha).map_err(|_| Error::Identity)?;
        let _guard = self.lock()?;
        Ok(self
            .receipt(project, archive_sha)?
            .map(|(_, metadata)| metadata))
    }

    pub fn read(&self, project: &str, source_id: &str) -> Result<Metadata> {
        ml_exp_core::identity::validate_id(project).map_err(|_| Error::Identity)?;
        sources::validate_source_id(source_id).map_err(|_| Error::Identity)?;
        let _guard = self.lock()?;
        let project_sources = self.sources.child(OsStr::new(project), false)?;
        let source = project_sources.child(OsStr::new(source_id), false)?;
        self.verify_tree(project, source_id, &source)
    }

    pub fn import_archive<R: Read + Seek>(
        &self,
        project: &str,
        stream: &mut R,
        expected_sha: &str,
        created_at: &str,
    ) -> Result<Metadata> {
        ml_exp_core::identity::validate_id(project).map_err(|_| Error::Identity)?;
        sources::validate_archive_digest(expected_sha).map_err(|_| Error::Identity)?;
        let _guard = self.lock()?;
        if let Some((_, metadata)) = self.receipt(project, expected_sha)? {
            return Ok(metadata);
        }
        stream.seek(SeekFrom::Start(0))?;
        let mut archive_sha = Sha256::new();
        let mut size = 0u64;
        let mut random = [0u8; 8];
        getrandom::fill(&mut random).map_err(|_| Error::Identity)?;
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
            let archive_for_removal = stage.open_regular_read(OsStr::new("archive.bin"))?;
            stage.remove_staged_regular(OsStr::new("archive.bin"), &archive_for_removal)?;
            let (plan, files) = topology(&tree, "")?;
            let mut digest =
                TreeDigest::new(TreePlan::new(plan, &self.policy).map_err(|_| Error::Archive)?);
            while let Some(request) = digest.next_chunk() {
                let (parent, name) = files.get(request.path).ok_or(Error::Archive)?;
                let mut file = parent.open_regular_read(name)?;
                file.seek(SeekFrom::Start(request.offset))?;
                let mut bytes = vec![0u8; request.len];
                file.read_exact(&mut bytes)?;
                digest = digest.consume(&bytes).map_err(|_| Error::Archive)?;
            }
            let tree_digest = digest.finish().map_err(|_| Error::Archive)?;
            let observation = json!({"kind":"archive","sha256":expected_sha});
            let metadata = Metadata::new(
                project,
                &tree_digest,
                created_at,
                &observation,
                &self.policy,
            )
            .map_err(|_| Error::Archive)?;
            stage.publish_new(OsStr::new(METADATA), &encode(&metadata.to_value())?)?;
            let metadata_file = stage.open_regular_read(OsStr::new(METADATA))?;
            stage.seal_regular(OsStr::new(METADATA), &metadata_file, false)?;
            seal_tree(&tree)?;
            stage.source_mode(true)?;
            let source_id = metadata.source_id().to_owned();
            let project_sources = self.sources.child(OsStr::new(project), true)?;
            if project_sources.entry_exists(OsStr::new(&source_id))? {
                let existing = project_sources.child(OsStr::new(&source_id), false)?;
                let verified = self.verify_tree(project, &source_id, &existing)?;
                let imports_project = self.imports.child(OsStr::new(project), true)?;
                let receipt = json!({"schema_version":1,"project":project,"archive_sha256":expected_sha,"source_id":source_id,"tree_digest":verified.tree_digest()});
                imports_project.publish_new(OsStr::new(expected_sha), &encode(&receipt)?)?;
                let receipt_file = imports_project.open_regular_read(OsStr::new(expected_sha))?;
                imports_project.seal_regular(OsStr::new(expected_sha), &receipt_file, false)?;
                return Ok(verified);
            }
            let published_directory =
                match stage.publish_directory(&project_sources, OsStr::new(&source_id)) {
                    Ok(directory) => directory,
                    Err(_) => {
                        // The atomic rename may already have happened when
                        // this API reports an error. Never delete the stage.
                        published = true;
                        return Err(Error::EffectUncertain);
                    }
                };
            published = true;
            let verified = self.verify_tree(project, &source_id, &published_directory)?;
            let imports_project = self.imports.child(OsStr::new(project), true)?;
            let receipt = json!({"schema_version":1,"project":project,"archive_sha256":expected_sha,"source_id":source_id,"tree_digest":verified.tree_digest()});
            imports_project.publish_new(OsStr::new(expected_sha), &encode(&receipt)?)?;
            let receipt_file = imports_project.open_regular_read(OsStr::new(expected_sha))?;
            imports_project.seal_regular(OsStr::new(expected_sha), &receipt_file, false)?;
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
            (Ok(_), Err(_)) | (Err(_), Err(_)) => Err(Error::EffectUncertain),
            (Err(error), Ok(())) => Err(error),
        }
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
            directory.validate_immutable_regular_entry(&name, &file)?;
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
            let file = stage.open_regular_read(&name)?;
            stage.remove_staged_regular(&name, &file)?;
        }
    }
    parent.remove_staged_empty_child(stage)?;
    Ok(())
}
