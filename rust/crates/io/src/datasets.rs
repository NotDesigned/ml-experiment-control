//! Descriptor-confined dataset extraction and verification. Payload buffers are
//! fixed; dataset identity and completeness rules live in the pure core.
use crate::{
    artifacts::{self, Kind},
    directory::{Directory, require_sealed, seal_regular},
};
use ml_exp_core::{artifacts::FileReceipt, datasets::Definition};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    os::unix::fs::MetadataExt,
};
fn invalid() -> io::Error {
    io::Error::other("dataset files are unsafe, incomplete or changed")
}
pub fn verify(root: &Directory, definition: &Definition, sealed: bool) -> io::Result<()> {
    definition.validate().map_err(|_| invalid())?;
    verify_inventory(root, &definition.files, sealed)
}
pub fn verify_inventory(root: &Directory, files: &[FileReceipt], sealed: bool) -> io::Result<()> {
    ml_exp_core::data_preparation::content_hash(files).map_err(|_| invalid())?;
    require_file_set(root, files, sealed)?;
    for record in files {
        let mut file = artifacts::open(root, &record.path, Kind::Data)?;
        if sealed {
            require_sealed(&file)?;
        }
        let before = file.metadata()?;
        let (digest, size) = artifacts::hash(&mut file)?;
        let after = file.metadata()?;
        if size != record.bytes
            || digest != record.sha256
            || before.len() != after.len()
            || before.mtime() != after.mtime()
            || before.mtime_nsec() != after.mtime_nsec()
            || before.ctime() != after.ctime()
            || before.ctime_nsec() != after.ctime_nsec()
        {
            return Err(invalid());
        }
    }
    Ok(())
}
/// A sealed tree with the expected files and sizes, without reading any content.
/// This is for a check that repeats; the job's `verify` is what proves the bytes.
pub fn verify_file_sizes(root: &Directory, definition: &Definition) -> io::Result<()> {
    definition.validate().map_err(|_| invalid())?;
    require_file_set(root, &definition.files, true)?;
    for record in &definition.files {
        let file = artifacts::open(root, &record.path, Kind::Data)?;
        require_sealed(&file)?;
        if file.metadata()?.len() != record.bytes {
            return Err(invalid());
        }
    }
    Ok(())
}
fn require_file_set(root: &Directory, files: &[FileReceipt], sealed: bool) -> io::Result<()> {
    if sealed {
        verify_sealed_directories(root, 0, &mut 0)?;
    }
    let mut paths = Vec::new();
    artifacts::walk(root, "", &mut paths, &mut 0, Kind::Data)?;
    paths.sort();
    if paths != files.iter().map(|f| f.path.clone()).collect::<Vec<_>>() {
        return Err(invalid());
    }
    Ok(())
}
fn verify_sealed_directories(root: &Directory, depth: usize, budget: &mut usize) -> io::Result<()> {
    if depth > 128 || root.mode()? & 0o222 != 0 {
        return Err(invalid());
    }
    for name in root.entry_names_limited(ml_exp_core::artifacts::MAX_FILES)? {
        *budget += 1;
        if *budget > ml_exp_core::artifacts::MAX_FILES {
            return Err(invalid());
        }
        if let Ok(child) = root.child(&name, false) {
            verify_sealed_directories(&child, depth + 1, budget)?;
        }
    }
    Ok(())
}
fn parent(root: &Directory, path: &str) -> io::Result<(Directory, String)> {
    let mut parts = path.split('/').collect::<Vec<_>>();
    let name = parts.pop().ok_or_else(invalid)?.to_owned();
    let mut directory = root.clone();
    for part in parts {
        directory = directory.child(OsStr::new(part), true)?;
    }
    Ok((directory, name))
}
pub fn extract(file: &mut File, root: &Directory, definition: &Definition) -> io::Result<()> {
    definition.validate().map_err(|_| invalid())?;
    file.seek(SeekFrom::Start(0))?;
    let mut archive = tar::Archive::new(&mut *file);
    let mut seen = BTreeSet::new();
    let expected_files: BTreeMap<_, _> = definition
        .files
        .iter()
        .map(|f| (f.path.as_str(), f))
        .collect();
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_str().ok_or_else(invalid)?.to_owned();
        if !entry.header().entry_type().is_file()
            || !ml_exp_core::datasets::path(&path)
            || !seen.insert(path.clone())
        {
            return Err(invalid());
        }
        let expected = expected_files.get(path.as_str()).ok_or_else(invalid)?;
        if entry.size() != expected.bytes {
            return Err(invalid());
        }
        let (parent, name) = parent(root, &path)?;
        let mut output = parent.create_new_regular(OsStr::new(&name))?;
        let mut remaining = expected.bytes;
        let mut buffer = [0u8; 65536];
        while remaining > 0 {
            let limit = remaining.min(buffer.len() as u64) as usize;
            let count = entry.read(&mut buffer[..limit])?;
            if count == 0 {
                return Err(invalid());
            }
            output.write_all(&buffer[..count])?;
            remaining -= count as u64;
        }
        seal_regular(&output, false)?;
    }
    let tail = archive.into_inner();
    let mut buffer = [0u8; 65536];
    loop {
        let n = tail.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        if buffer[..n].iter().any(|b| *b != 0) {
            return Err(invalid());
        }
    }
    verify(root, definition, false)
}
pub fn seal(root: &Directory) -> io::Result<()> {
    seal_tree(root, false)
}
/// Seal a completed script's newly generated files before publishing its
/// receipt. Ordinary archive sealing still requires presealed files.
pub fn seal_generated(root: &Directory) -> io::Result<()> {
    seal_tree(root, true)
}
fn seal_tree(root: &Directory, generated: bool) -> io::Result<()> {
    fn visit(
        directory: &Directory,
        depth: usize,
        budget: &mut usize,
        generated: bool,
    ) -> io::Result<()> {
        if depth > 128 {
            return Err(invalid());
        }
        for name in directory.entry_names_limited(ml_exp_core::artifacts::MAX_FILES)? {
            *budget += 1;
            if *budget > ml_exp_core::artifacts::MAX_FILES {
                return Err(invalid());
            }
            if let Ok(child) = directory.child(&name, false) {
                visit(&child, depth + 1, budget, generated)?;
            } else {
                let file = directory.open_regular_read(&name)?;
                if generated {
                    seal_regular(&file, false)?;
                } else {
                    require_sealed(&file)?;
                }
            }
        }
        directory.source_mode(true)
    }
    visit(root, 0, &mut 0, generated)
}
/// Cleanup is permitted only for the caller's own unpublished staging tree.
/// Traversal never follows a link; any unexpected type stops cleanup.
pub fn remove_staging(parent: &Directory, tree: &Directory) -> io::Result<()> {
    fn visit(
        parent: &Directory,
        tree: &Directory,
        depth: usize,
        budget: &mut usize,
    ) -> io::Result<()> {
        if depth > 128 {
            return Err(invalid());
        }
        tree.source_mode(false)?;
        for name in tree.entry_names_limited(ml_exp_core::artifacts::MAX_FILES)? {
            *budget += 1;
            if *budget > ml_exp_core::artifacts::MAX_FILES {
                return Err(invalid());
            }
            if let Ok(child) = tree.child(&name, false) {
                visit(tree, &child, depth + 1, budget)?;
            } else {
                tree.remove_staged_regular(&name)?;
            }
        }
        parent.remove_staged_empty_child(tree)
    }
    visit(parent, tree, 0, &mut 0)
}
/// Record all bounded regular files through retained directory descriptors.
pub fn inventory(tree: &Directory) -> std::io::Result<Vec<FileReceipt>> {
    let mut paths = Vec::new();
    artifacts::walk(tree, "", &mut paths, &mut 0, artifacts::Kind::Data)?;
    paths.sort();
    if paths.is_empty() {
        return Err(std::io::Error::other("empty prepared dataset"));
    }
    let mut files = Vec::new();
    for path in paths {
        let mut file = artifacts::open(tree, &path, artifacts::Kind::Data)?;
        let before = file.metadata()?;
        let (sha256, bytes) = artifacts::hash(&mut file)?;
        let after = file.metadata()?;
        if before.len() != after.len()
            || before.modified()? != after.modified()?
            || before.ctime() != after.ctime()
            || before.ctime_nsec() != after.ctime_nsec()
        {
            return Err(std::io::Error::other("prepared data changed"));
        }
        files.push(FileReceipt {
            path,
            bytes,
            sha256,
        });
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dataset_roundtrip_is_exact_immutable_and_rejects_tampering() {
        let source = tempfile::tempdir().unwrap();
        std::fs::create_dir(source.path().join("train")).unwrap();
        std::fs::write(source.path().join("train/tokens.bin"), [0u8, 1, 2, 255]).unwrap();
        std::fs::write(source.path().join("empty"), []).unwrap();
        let source = Directory::open(source.path(), false).unwrap();
        let mut archive = tempfile::tempfile().unwrap();
        let (sha256, archive_bytes, files) = artifacts::pack_data(&source, &mut archive).unwrap();
        let definition = Definition {
            sha256,
            archive_bytes,
            files,
        };
        let destination = tempfile::tempdir().unwrap();
        let destination = Directory::open(destination.path(), false).unwrap();
        let tree = destination.create_new_child(OsStr::new("tree")).unwrap();
        extract(&mut archive, &tree, &definition).unwrap();
        seal(&tree).unwrap();
        verify(&tree, &definition, true).unwrap();
        let file = tree
            .child(OsStr::new("train"), false)
            .unwrap()
            .open_regular_read(OsStr::new("tokens.bin"))
            .unwrap();
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .unwrap();
        assert!(verify(&tree, &definition, true).is_err());
        remove_staging(&destination, &tree).unwrap();
    }
    #[test]
    fn the_size_check_refuses_a_changed_shape_and_leaves_content_to_the_hash() {
        use std::os::unix::fs::PermissionsExt;
        let source = tempfile::tempdir().unwrap();
        std::fs::write(source.path().join("tokens.bin"), b"abcd").unwrap();
        let source = Directory::open(source.path(), false).unwrap();
        let mut archive = tempfile::tempfile().unwrap();
        let (sha256, archive_bytes, files) = artifacts::pack_data(&source, &mut archive).unwrap();
        let definition = Definition {
            sha256,
            archive_bytes,
            files,
        };
        let destination = tempfile::tempdir().unwrap();
        let path = destination.path().join("tree");
        let root = Directory::open(destination.path(), false).unwrap();
        let tree = root.create_new_child(OsStr::new("tree")).unwrap();
        extract(&mut archive, &tree, &definition).unwrap();
        seal(&tree).unwrap();
        verify_file_sizes(&tree, &definition).unwrap();

        let file = path.join("tokens.bin");
        let rewrite = |bytes: &[u8]| {
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
            std::fs::write(&file, bytes).unwrap();
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o400)).unwrap();
        };
        rewrite(b"abcx");
        verify_file_sizes(&tree, &definition).unwrap();
        assert!(verify(&tree, &definition, true).is_err());
        rewrite(b"abc");
        assert!(verify_file_sizes(&tree, &definition).is_err());
        rewrite(b"abcd");
        verify_file_sizes(&tree, &definition).unwrap();

        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(verify_file_sizes(&tree, &definition).is_err());
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o400)).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(path.join("extra"), b"").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o500)).unwrap();
        assert!(verify_file_sizes(&tree, &definition).is_err());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    #[test]
    fn extra_files_links_and_bad_member_hash_never_publish_a_dataset() {
        let source = tempfile::tempdir().unwrap();
        std::fs::write(source.path().join("data"), b"tokens").unwrap();
        let source = Directory::open(source.path(), false).unwrap();
        let mut archive = tempfile::tempfile().unwrap();
        let (sha256, archive_bytes, files) = artifacts::pack_data(&source, &mut archive).unwrap();
        let mut definition = Definition {
            sha256,
            archive_bytes,
            files,
        };
        definition.files[0].sha256 = "1".repeat(64);
        let target = tempfile::tempdir().unwrap();
        let target = Directory::open(target.path(), false).unwrap();
        assert!(extract(&mut archive, &target, &definition).is_err());
        std::os::unix::fs::symlink("/outside", source.path().join("link")).unwrap();
        assert!(artifacts::pack_data(&source, &mut tempfile::tempfile().unwrap()).is_err());
    }
}
