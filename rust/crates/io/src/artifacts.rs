//! Result archive effects on backend storage. Constant-size payload buffers;
//! checkpoints are excluded, no symlinks, no extra/missing/unstable files.
use crate::directory::Directory;
use ml_exp_core::artifacts::{
    self, FileReceipt,
    publication::{Policy, export_path},
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    os::unix::fs::MetadataExt,
};

fn invalid() -> io::Error {
    io::Error::other("result archive files are invalid or changed")
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Results,
    Data,
}
fn allowed(kind: Kind, path: &str) -> bool {
    match kind {
        Kind::Results => export_path(path),
        Kind::Data => ml_exp_core::datasets::path(path),
    }
}
pub(crate) fn walk(
    directory: &Directory,
    prefix: &str,
    found: &mut Vec<String>,
    budget: &mut usize,
    kind: Kind,
) -> io::Result<()> {
    if prefix.len() > artifacts::MAX_PATH || prefix.split('/').count() > 128 {
        return Err(invalid());
    }
    let mut names = directory.entry_names_limited(artifacts::MAX_FILES)?;
    names.sort();
    for name in names {
        *budget += 1;
        if *budget > artifacts::MAX_FILES {
            return Err(invalid());
        }
        let name = name.to_str().ok_or_else(invalid)?;
        if kind == Kind::Results
            && prefix.is_empty()
            && matches!(name, "checkpoints" | "checkpoint.ready.json")
        {
            continue;
        }
        let path = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}/{name}")
        };
        if !allowed(kind, &path) {
            return Err(invalid());
        }
        if let Ok(child) = directory.child(OsStr::new(name), false) {
            walk(&child, &path, found, budget, kind)?;
        } else {
            let file = directory.open_regular_read(OsStr::new(name))?;
            directory.validate_regular_entry(OsStr::new(name), &file)?;
            found.push(path);
        }
    }
    directory.verify_private_parent()?;
    Ok(())
}
pub(crate) fn open(root: &Directory, path: &str, kind: Kind) -> io::Result<(Directory, File)> {
    if !allowed(kind, path) {
        return Err(invalid());
    }
    let mut names = path.split('/').collect::<Vec<_>>();
    let name = names.pop().ok_or_else(invalid)?;
    let mut parent = root.clone();
    for n in names {
        parent = parent.child(OsStr::new(n), false)?;
    }
    let file = parent.open_regular_read(OsStr::new(name))?;
    Ok((parent, file))
}
struct HashWriter<W> {
    inner: W,
    hash: Sha256,
    bytes: u64,
}
impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(b)?;
        self.hash.update(&b[..n]);
        self.bytes = self.bytes.checked_add(n as u64).ok_or_else(invalid)?;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
struct HashReader<R> {
    inner: R,
    hash: Sha256,
    bytes: u64,
}
impl<R: Read> Read for HashReader<R> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(b)?;
        self.hash.update(&b[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
}
/// Stream a deterministic regular-file tar. The caller owns an unpublished
/// destination and removes it on failure; no API-host archive is necessary.
pub fn pack(
    root: &Directory,
    output: &mut File,
    policy: &Policy,
) -> io::Result<(String, u64, Vec<FileReceipt>)> {
    pack_kind(root, output, policy, Kind::Results)
}
pub fn pack_data(
    root: &Directory,
    output: &mut File,
) -> io::Result<(String, u64, Vec<FileReceipt>)> {
    pack_kind(root, output, &Policy::default(), Kind::Data)
}
fn pack_kind(
    root: &Directory,
    output: &mut File,
    policy: &Policy,
    kind: Kind,
) -> io::Result<(String, u64, Vec<FileReceipt>)> {
    policy.validate().map_err(|_| invalid())?;
    let mut paths = Vec::new();
    walk(root, "", &mut paths, &mut 0, kind)?;
    if kind == Kind::Data {
        paths.sort();
    }
    if paths.is_empty() {
        return Err(invalid());
    }
    policy
        .check(
            &paths
                .iter()
                .map(|p| FileReceipt {
                    path: p.clone(),
                    bytes: 0,
                    sha256: "0".repeat(64),
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|_| invalid())?;
    let writer = HashWriter {
        inner: output,
        hash: Sha256::new(),
        bytes: 0,
    };
    let mut archive = tar::Builder::new(writer);
    let mut files = Vec::new();
    for path in &paths {
        let (parent, mut file) = open(root, path, kind)?;
        let before = file.metadata()?;
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o600);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        header.set_size(before.len());
        header.set_cksum();
        let mut reader = HashReader {
            inner: &mut file,
            hash: Sha256::new(),
            bytes: 0,
        };
        archive.append_data(
            &mut header,
            if kind == Kind::Results {
                format!("outputs/{path}")
            } else {
                path.clone()
            },
            &mut reader,
        )?;
        let bytes = reader.bytes;
        let sha256 = format!("{:x}", reader.hash.finalize());
        let after = file.metadata()?;
        if before.len() != after.len()
            || before.mtime() != after.mtime()
            || before.mtime_nsec() != after.mtime_nsec()
            || before.ctime() != after.ctime()
            || before.ctime_nsec() != after.ctime_nsec()
            || bytes != before.len()
        {
            return Err(invalid());
        }
        parent.validate_regular_entry(
            OsStr::new(path.rsplit('/').next().ok_or_else(invalid)?),
            &file,
        )?;
        files.push(FileReceipt {
            path: path.clone(),
            bytes,
            sha256,
        });
    }
    archive.finish()?;
    let writer = archive.into_inner()?;
    writer.inner.sync_all()?;
    let mut after = Vec::new();
    walk(root, "", &mut after, &mut 0, kind)?;
    if kind == Kind::Data {
        after.sort();
    }
    if paths != after {
        return Err(invalid());
    }
    // A second hash pass also catches a file changed after its own first read.
    for record in &files {
        let (_, mut file) = open(root, &record.path, kind)?;
        let (h, n) = hash(&mut file)?;
        if h != record.sha256 || n != record.bytes {
            return Err(invalid());
        }
    }
    Ok((format!("{:x}", writer.hash.finalize()), writer.bytes, files))
}
pub fn hash(reader: &mut impl Read) -> io::Result<(String, u64)> {
    let mut h = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
        size = size.checked_add(n as u64).ok_or_else(invalid)?;
    }
    Ok((format!("{:x}", h.finalize()), size))
}
/// Validate the complete local archive and every regular member without
/// extracting untrusted paths. Missing values never turn into empty files.
pub fn verify(
    file: &mut File,
    receipt: &artifacts::ArchiveReceipt,
) -> io::Result<artifacts::DownloadProof> {
    artifacts::validate_receipt(receipt).map_err(|_| invalid())?;
    file.seek(SeekFrom::Start(0))?;
    let (digest, size) = hash(file)?;
    if digest != receipt.sha256 || size != receipt.bytes {
        return Err(invalid());
    }
    file.seek(SeekFrom::Start(0))?;
    let mut archive = tar::Archive::new(&mut *file);
    let mut seen = BTreeSet::new();
    let mut proofs = std::collections::BTreeMap::new();
    for row in archive.entries()? {
        let mut row = row?;
        if !row.header().entry_type().is_file() {
            return Err(invalid());
        }
        let path = row.path()?.to_str().ok_or_else(invalid)?.to_owned();
        let local = path.strip_prefix("outputs/").ok_or_else(invalid)?;
        let expected = receipt
            .files
            .iter()
            .find(|r| r.path == local)
            .ok_or_else(invalid)?;
        if !seen.insert(local.to_owned()) || row.size() != expected.bytes {
            return Err(invalid());
        }
        let (digest, size) = hash(&mut row)?;
        if digest != expected.sha256 || size != expected.bytes {
            return Err(invalid());
        }
        proofs.insert(path, expected.clone());
    }
    let proof = artifacts::DownloadProof {
        archive_sha256: receipt.sha256.clone(),
        archive_bytes: receipt.bytes,
        files: proofs,
    };
    artifacts::proof_matches(receipt, &proof).map_err(|_| invalid())?;
    Ok(proof)
}

/// Verify a remote archive without spooling it to the API disk. The reader is
/// bounded by the frozen byte count, including tar padding and trailing bytes.
pub fn verify_stream(
    reader: &mut impl Read,
    receipt: &artifacts::ArchiveReceipt,
) -> io::Result<artifacts::DownloadProof> {
    artifacts::validate_receipt(receipt).map_err(|_| invalid())?;
    let proofs = verify_members(
        reader,
        &receipt.sha256,
        receipt.bytes,
        &receipt.files,
        "outputs/",
        None,
    )?;
    let proof = artifacts::DownloadProof {
        archive_sha256: receipt.sha256.clone(),
        archive_bytes: receipt.bytes,
        files: proofs,
    };
    artifacts::proof_matches(receipt, &proof).map_err(|_| invalid())?;
    Ok(proof)
}
pub fn verify_data_stream(
    reader: &mut impl Read,
    definition: &ml_exp_core::datasets::Definition,
) -> io::Result<()> {
    definition.validate().map_err(|_| invalid())?;
    verify_members(
        reader,
        &definition.sha256,
        definition.archive_bytes,
        &definition.files,
        "",
        None,
    )?;
    Ok(())
}
fn verify_members(
    reader: &mut impl Read,
    sha256: &str,
    bytes: u64,
    files: &[FileReceipt],
    prefix: &str,
    mut selected: Option<(&str, &mut dyn Write)>,
) -> io::Result<std::collections::BTreeMap<String, FileReceipt>> {
    let mut reader = HashReader {
        inner: reader.take(bytes.checked_add(1).ok_or_else(invalid)?),
        hash: Sha256::new(),
        bytes: 0,
    };
    let mut proofs = std::collections::BTreeMap::new();
    {
        let mut archive = tar::Archive::new(&mut reader);
        for row in archive.entries()? {
            let mut row = row?;
            if !row.header().entry_type().is_file() {
                return Err(invalid());
            }
            let path = row.path()?.to_str().ok_or_else(invalid)?.to_owned();
            let local = path.strip_prefix(prefix).ok_or_else(invalid)?;
            let expected = files.iter().find(|f| f.path == local).ok_or_else(invalid)?;
            if proofs.contains_key(&path) || row.size() != expected.bytes {
                return Err(invalid());
            }
            let (digest, bytes) =
                if let Some((_, writer)) = selected.as_mut().filter(|(path, _)| *path == local) {
                    let mut reader = HashReader {
                        inner: &mut row,
                        hash: Sha256::new(),
                        bytes: 0,
                    };
                    io::copy(&mut reader, writer)?;
                    (format!("{:x}", reader.hash.finalize()), reader.bytes)
                } else {
                    hash(&mut row)?
                };
            if digest != expected.sha256 || bytes != expected.bytes {
                return Err(invalid());
            }
            proofs.insert(path, expected.clone());
        }
    }
    io::copy(&mut reader, &mut io::sink())?;
    let digest = format!("{:x}", reader.hash.finalize());
    if digest != sha256 || reader.bytes != bytes || proofs.len() != files.len() {
        return Err(invalid());
    }
    Ok(proofs)
}

/// Retain just one file locally, but verify all archive bytes and member hashes
/// before the caller publishes that file. Partial download grants no ACK.
pub fn copy_result_file(
    reader: &mut impl Read,
    receipt: &artifacts::ArchiveReceipt,
    path: &str,
    output: &mut impl Write,
) -> io::Result<FileReceipt> {
    artifacts::validate_receipt(receipt).map_err(|_| invalid())?;
    let record = receipt
        .files
        .iter()
        .find(|f| f.path == path)
        .ok_or_else(invalid)?;
    if !export_path(path) {
        return Err(invalid());
    }
    verify_members(
        reader,
        &receipt.sha256,
        receipt.bytes,
        &receipt.files,
        "outputs/",
        Some((path, output)),
    )?;
    Ok(record.clone())
}

pub fn estimate(root: &Directory) -> io::Result<u64> {
    let mut paths = Vec::new();
    walk(root, "", &mut paths, &mut 0, Kind::Results)?;
    let mut bytes = 16u64 * 1024 * 1024;
    for path in paths {
        let (_, file) = open(root, &path, Kind::Results)?;
        bytes = bytes
            .checked_add(file.metadata()?.len())
            .and_then(|v| v.checked_add(8192))
            .ok_or_else(invalid)?;
    }
    Ok(bytes)
}

/// Discover a bounded regular-file dataset inventory while verifying the entire
/// tar object. No extraction or local payload file is needed. Compressed input
/// is deliberately refused: native dataset definitions describe plain tar.
pub fn inspect_data_stream(
    reader: &mut impl Read,
    sha256: &str,
    bytes: u64,
) -> io::Result<ml_exp_core::datasets::Definition> {
    ml_exp_core::sources::validate_archive_digest(sha256).map_err(|_| invalid())?;
    ml_exp_core::multipart::part_count(bytes, ml_exp_core::multipart::Policy::default())
        .map_err(|_| invalid())?;
    let mut reader = HashReader {
        inner: reader.take(bytes.checked_add(1).ok_or_else(invalid)?),
        hash: Sha256::new(),
        bytes: 0,
    };
    let mut files = std::collections::BTreeMap::new();
    let mut members = 0usize;
    let mut expanded = 0u64;
    {
        let mut archive = tar::Archive::new(&mut reader);
        for entry in archive.entries()? {
            members += 1;
            if members > artifacts::MAX_FILES * 2 {
                return Err(invalid());
            }
            let mut entry = entry?;
            let path = entry.path()?.to_str().ok_or_else(invalid)?.to_owned();
            if !ml_exp_core::datasets::path(&path) {
                return Err(invalid());
            }
            if !entry.header().entry_type().is_file()
                || files.len() >= artifacts::MAX_FILES
                || files.contains_key(&path)
            {
                return Err(invalid());
            }
            expanded = expanded.checked_add(entry.size()).ok_or_else(invalid)?;
            if expanded > bytes {
                return Err(invalid());
            }
            let (sha256, size) = hash(&mut entry)?;
            if size != entry.size() {
                return Err(invalid());
            }
            files.insert(
                path.clone(),
                FileReceipt {
                    path,
                    bytes: size,
                    sha256,
                },
            );
        }
    }
    io::copy(&mut reader, &mut io::sink())?;
    if reader.bytes != bytes || format!("{:x}", reader.hash.finalize()) != sha256 {
        return Err(invalid());
    }
    let definition = ml_exp_core::datasets::Definition {
        sha256: sha256.into(),
        archive_bytes: bytes,
        files: files.into_values().collect(),
    };
    definition.validate().map_err(|_| invalid())?;
    Ok(definition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    #[test]
    fn imported_data_discovers_files_and_rejects_wrong_or_truncated_bytes() {
        let root = tempfile::tempdir().unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        std::fs::create_dir(root.path().join("nested")).unwrap();
        std::fs::write(root.path().join("nested/train.bin"), b"training bytes").unwrap();
        std::fs::write(root.path().join("empty"), []).unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let (sha, bytes, files) = pack_data(&directory, &mut file).unwrap();
        file.rewind().unwrap();
        let definition = inspect_data_stream(&mut file, &sha, bytes).unwrap();
        assert_eq!(definition.files, files);
        file.rewind().unwrap();
        verify_data_stream(&mut file, &definition).unwrap();
        file.rewind().unwrap();
        assert!(inspect_data_stream(&mut file, &"0".repeat(64), bytes).is_err());
        file.rewind().unwrap();
        assert!(inspect_data_stream(&mut file, &sha, bytes - 1).is_err());
        file.rewind().unwrap();
        assert!(inspect_data_stream(&mut file, &sha, bytes + 1).is_err());
    }
    fn receipt(sha256: String, bytes: u64, files: Vec<FileReceipt>) -> artifacts::ArchiveReceipt {
        artifacts::ArchiveReceipt {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
            object_key: format!("demo/train/attempt-001/{sha256}.tar"),
            sha256,
            bytes,
            files,
        }
    }
    #[test]
    fn streaming_roundtrip_verifies_every_file_and_leaves_checkpoints_out() {
        let root = tempfile::tempdir().unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        std::fs::create_dir(root.path().join("nested")).unwrap();
        std::fs::write(root.path().join("nested/weights.bin"), [0, 1, 2, 255]).unwrap();
        std::fs::write(root.path().join("empty.log"), []).unwrap();
        std::fs::create_dir(root.path().join("checkpoints")).unwrap();
        symlink("/outside", root.path().join("checkpoints/private")).unwrap();
        std::fs::write(
            root.path().join("checkpoint.ready.json"),
            b"reserved state marker",
        )
        .unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let (sha, bytes, files) = pack(
            &directory,
            &mut file,
            &Policy {
                required_files: vec!["nested/weights.bin".into()],
            },
        )
        .unwrap();
        assert_eq!(files.len(), 2);
        let receipt = receipt(sha, bytes, files);
        let proof = verify(&mut file, &receipt).unwrap();
        artifacts::proof_matches(&receipt, &proof).unwrap();
        file.seek(SeekFrom::Start(600)).unwrap();
        file.write_all(&[42]).unwrap();
        assert!(verify(&mut file, &receipt).is_err());
        assert!(root.path().join("checkpoints/private").is_symlink());
    }
    #[test]
    fn missing_required_files_symlinks_and_duplicate_members_fail() {
        let root = tempfile::tempdir().unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        std::fs::write(root.path().join("metrics.jsonl"), b"{}\n").unwrap();
        let mut file = tempfile::tempfile().unwrap();
        assert!(
            pack(
                &directory,
                &mut file,
                &Policy {
                    required_files: vec!["weights.bin".into()]
                }
            )
            .is_err()
        );
        symlink("metrics.jsonl", root.path().join("link")).unwrap();
        assert!(pack(&directory, &mut file, &Policy::default()).is_err());
        std::fs::remove_file(root.path().join("link")).unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let (sha, bytes, files) = pack(&directory, &mut file, &Policy::default()).unwrap();
        let mut wrong = receipt(sha, bytes, files);
        wrong.files[0].sha256 = "0".repeat(64);
        assert!(verify(&mut file, &wrong).is_err());
    }
}
