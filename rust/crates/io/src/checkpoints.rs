//! Descriptor-safe verification and sealing for persistent checkpoint state.
//!
//! Checkpoint identity and manifest rules live in ml-exp-core. This module
//! supplies only the filesystem effect shell: bounded ready-marker reads,
//! exact generation traversal, stable SHA256 checks and optional read-only
//! sealing. No network registration or provider operation occurs.
use ml_exp_core::checkpoints::{self, FileRecord, Manifest};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
};

use crate::directory::Directory;

pub const READY_LIMIT: u64 = checkpoints::MANIFEST_LIMIT as u64;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("checkpoint filesystem operation failed")]
    Io(#[from] std::io::Error),
    #[error("checkpoint manifest is invalid")]
    Manifest(#[from] checkpoints::Error),
    #[error("checkpoint generation file set differs from its ready marker")]
    FileSet,
    #[error("checkpoint file changed while it was being verified")]
    Changed,
    #[error("checkpoint generation is not a regular immutable tree")]
    UnsafeTree,
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    pub manifest: Manifest,
    pub generation: String,
}

pub fn read_ready(root: &Directory) -> Result<Manifest> {
    let bytes = root.read_regular(OsStr::new("checkpoint.ready.json"), READY_LIMIT)?;
    let value = serde_json::Value::Object(
        ml_exp_core::json::mapping(&bytes).map_err(|_| checkpoints::Error::Manifest)?,
    );
    Ok(checkpoints::manifest(&value)?)
}

fn generation_directory(root: &Directory, generation: &str) -> Result<Directory> {
    let checkpoints = root.child(OsStr::new("checkpoints"), false)?;
    Ok(checkpoints.child(OsStr::new(generation), false)?)
}

fn walk(
    directory: &Directory,
    prefix: &str,
    found: &mut BTreeSet<String>,
    entries: &mut usize,
) -> Result<()> {
    if prefix.len() > checkpoints::MAX_PATH_BYTES || prefix.split('/').count() > 256 {
        return Err(Error::UnsafeTree);
    }
    for name in directory.entry_names_limited(checkpoints::MAX_FILES)? {
        *entries += 1;
        if *entries > checkpoints::MAX_FILES {
            return Err(Error::UnsafeTree);
        }
        let name = name.to_str().ok_or(Error::UnsafeTree)?;
        let path = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}/{name}")
        };
        if let Ok(child) = directory.child(OsStr::new(name), false) {
            walk(&child, &path, found, entries)?;
        } else {
            let file = directory.open_regular_read(OsStr::new(name))?;
            if !found.insert(path) {
                return Err(Error::FileSet);
            }
            if found.len() > checkpoints::MAX_FILES {
                return Err(Error::FileSet);
            }
            directory.validate_regular_entry(OsStr::new(name), &file)?;
        }
    }
    Ok(())
}

fn open_file(root: &Directory, path: &str) -> Result<(Directory, File)> {
    let mut parts: Vec<_> = path.split('/').collect();
    let name = parts.pop().ok_or(Error::UnsafeTree)?;
    let mut directory = root.clone();
    for part in parts {
        directory = directory.child(OsStr::new(part), false)?;
    }
    let file = directory.open_regular_read(OsStr::new(name))?;
    Ok((directory, file))
}

fn verify_file(root: &Directory, record: &FileRecord) -> Result<()> {
    let (parent, mut file) = open_file(root, &record.path)?;
    let before = file.metadata()?;
    if !before.is_file() {
        return Err(Error::UnsafeTree);
    }
    file.seek(SeekFrom::Start(0))?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes = bytes.checked_add(count as u64).ok_or(Error::Changed)?;
        digest.update(&buffer[..count]);
    }
    let after = file.metadata()?;
    if before.ino() != after.ino()
        || before.len() != after.len()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
        || before.ctime() != after.ctime()
        || before.ctime_nsec() != after.ctime_nsec()
        || bytes != record.bytes
        || format!("{:x}", digest.finalize()) != record.sha256
    {
        return Err(Error::Changed);
    }
    parent.validate_regular_entry(
        OsStr::new(record.path.rsplit('/').next().ok_or(Error::UnsafeTree)?),
        &file,
    )?;
    Ok(())
}

fn seal_directories(directory: &Directory, depth: usize, budget: &mut usize) -> Result<()> {
    if depth > 256 {
        return Err(Error::UnsafeTree);
    }
    for name in directory.entry_names_limited(checkpoints::MAX_FILES)? {
        *budget += 1;
        if *budget > checkpoints::MAX_FILES {
            return Err(Error::UnsafeTree);
        }
        if let Ok(child) = directory.child(&name, false) {
            seal_directories(&child, depth + 1, budget)?;
        }
    }
    directory.source_mode(true)?;
    Ok(())
}

/// A restored generation was already sealed by the original registration.
/// Checking those permissions again catches accidental writable replacements.
pub fn verify_sealed(root: &Directory, manifest: &Manifest) -> Result<Verified> {
    for record in &manifest.files {
        let (parent, file) = open_file(root, &record.path)?;
        parent
            .validate_immutable_regular_entry(
                OsStr::new(record.path.rsplit('/').next().ok_or(Error::UnsafeTree)?),
                &file,
            )
            .map_err(|_| Error::UnsafeTree)?;
    }
    verify(root, manifest, false)
}

/// Verify that exactly the manifest's generation files exist and are stable.
/// If seal is true, files and the generation directory become read-only.
pub fn verify(root: &Directory, manifest: &Manifest, seal: bool) -> Result<Verified> {
    let generation = generation_directory(root, &manifest.generation)?;
    let mut found = BTreeSet::new();
    walk(
        &generation,
        &format!("checkpoints/{}", manifest.generation),
        &mut found,
        &mut 0,
    )?;
    let expected: BTreeSet<_> = manifest
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect();
    if found != expected {
        return Err(Error::FileSet);
    }
    for record in &manifest.files {
        verify_file(root, record)?;
    }
    if seal {
        for record in &manifest.files {
            let (parent, file) = open_file(root, &record.path)?;
            parent.seal_regular(
                OsStr::new(record.path.rsplit('/').next().ok_or(Error::UnsafeTree)?),
                &file,
                false,
            )?;
        }
        seal_directories(&generation, 0, &mut 0)?;
    }
    // Detect added/replaced files across hashing and sealing, including nested
    // paths. A registration never trusts only the initial directory listing.
    let mut after = BTreeSet::new();
    walk(
        &generation,
        &format!("checkpoints/{}", manifest.generation),
        &mut after,
        &mut 0,
    )?;
    if after != expected {
        return Err(Error::FileSet);
    }
    for record in &manifest.files {
        verify_file(root, record)?;
    }
    generation.verify_private_parent()?;
    Ok(Verified {
        manifest: manifest.clone(),
        generation: manifest.generation.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_core::checkpoints;
    use serde_json::{Value, json};
    use std::{fs, os::unix::fs::PermissionsExt, path::Path};

    fn ready() -> Value {
        json!({"step":4,"files":[{"path":"checkpoints/step-0004/model.bin","bytes":7,"sha256":"9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"}]})
    }

    #[test]
    fn verifies_exact_generation_and_seals_files() {
        let temp = tempfile::tempdir().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let root = Directory::open(temp.path(), false).unwrap();
        let checkpoints_dir = root.child(OsStr::new("checkpoints"), true).unwrap();
        let generation = checkpoints_dir
            .child(OsStr::new("step-0004"), true)
            .unwrap();
        let file = generation
            .create_new_regular(OsStr::new("model.bin"))
            .unwrap();
        use std::io::Write;
        (&file).write_all(b"weights").unwrap();
        file.sync_all().unwrap();
        generation
            .seal_regular(OsStr::new("model.bin"), &file, false)
            .unwrap();
        root.publish_new(
            OsStr::new("checkpoint.ready.json"),
            serde_json::to_vec(&ready()).unwrap().as_slice(),
        )
        .unwrap();
        let manifest = checkpoints::manifest(&ready()).unwrap();
        let verified = verify(&root, &manifest, true).unwrap();
        assert_eq!(verified.generation, "step-0004");
        assert_eq!(
            fs::metadata(Path::new(&format!(
                "{}/checkpoints/step-0004/model.bin",
                temp.path().display()
            )))
            .unwrap()
            .permissions()
            .mode()
                & 0o777,
            0o400
        );
    }

    #[test]
    fn rejects_extra_files_and_changed_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let root = Directory::open(temp.path(), false).unwrap();
        let checkpoints_dir = root.child(OsStr::new("checkpoints"), true).unwrap();
        let generation = checkpoints_dir
            .child(OsStr::new("step-0004"), true)
            .unwrap();
        let file = generation
            .create_new_regular(OsStr::new("model.bin"))
            .unwrap();
        use std::io::Write;
        (&file).write_all(b"weights").unwrap();
        file.sync_all().unwrap();
        generation
            .seal_regular(OsStr::new("model.bin"), &file, false)
            .unwrap();
        let extra = generation
            .create_new_regular(OsStr::new("extra.bin"))
            .unwrap();
        (&extra).write_all(b"x").unwrap();
        extra.sync_all().unwrap();
        assert!(matches!(
            verify(&root, &checkpoints::manifest(&ready()).unwrap(), false),
            Err(Error::FileSet)
        ));
    }

    #[test]
    fn rejects_a_stable_file_with_the_wrong_digest_or_size() {
        let temp = tempfile::tempdir().unwrap();
        let root = Directory::open(temp.path(), false).unwrap();
        let checkpoints_dir = root.child(OsStr::new("checkpoints"), true).unwrap();
        let generation = checkpoints_dir
            .child(OsStr::new("step-0004"), true)
            .unwrap();
        let file = generation
            .create_new_regular(OsStr::new("model.bin"))
            .unwrap();
        use std::io::Write;
        (&file).write_all(b"changed").unwrap();
        file.sync_all().unwrap();
        assert!(matches!(
            verify(&root, &checkpoints::manifest(&ready()).unwrap(), false),
            Err(Error::Changed)
        ));
    }
}
