//! Result archive effects on backend storage. Constant-size payload buffers;
//! everything in OUTPUT_DIR is a result. What cannot be archived is recorded and
//! skipped, so one odd path cannot strand the rest. A failed read or a file that
//! changes while being packed still fails the archive.
use crate::directory::{Directory, EntryType};
use ml_exp_core::artifacts::{
    self, FileReceipt,
    publication::{Policy, SkipReason, Skipped, export_path},
};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
};

const MAX_DEPTH: usize = 128;
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
/// One pass over a result tree. Nothing it meets can fail the walk except an I/O error.
struct ResultWalk<'a> {
    policy: &'a Policy,
    found: BTreeSet<String>,
    skipped: Skipped,
    entries: usize,
    limit: usize,
    listed: usize,
}
impl ResultWalk<'_> {
    fn visit(&mut self, directory: &Directory, prefix: &str, depth: usize) -> io::Result<()> {
        let (names, more) = directory.first_entry_names(artifacts::MAX_FILES)?;
        if more > 0 {
            let shown = if prefix.is_empty() { "." } else { prefix };
            self.skipped.add(shown, SkipReason::FileLimit);
            self.skipped.unlisted = self.skipped.unlisted.saturating_add(more);
        }
        for name in names {
            let shown = name.to_string_lossy();
            let path = if prefix.is_empty() {
                shown.into_owned()
            } else {
                format!("{prefix}/{shown}")
            };
            let required = self.policy.required_files.contains(&path);
            let cost = artifacts::listed_cost(&path);
            self.entries += 1;
            if self.entries > self.limit || self.listed + cost > artifacts::MAX_LISTED_BYTES {
                // A required file has its room reserved and is added after the walk.
                if !required {
                    self.skipped.add(&path, SkipReason::FileLimit);
                }
                continue;
            }
            if name.to_str().is_none() || !export_path(&path) {
                self.skipped.add(&path, SkipReason::UnsupportedPath);
                continue;
            }
            match directory.entry_type(&name)?.ok_or_else(invalid)? {
                EntryType::SymbolicLink => self.skipped.add(&path, SkipReason::SymbolicLink),
                EntryType::Special => self.skipped.add(&path, SkipReason::SpecialFile),
                EntryType::Directory if depth >= MAX_DEPTH => {
                    self.skipped.add(&path, SkipReason::UnsupportedPath);
                }
                EntryType::Directory => {
                    self.visit(&directory.child(&name, false)?, &path, depth + 1)?;
                }
                EntryType::Regular => {
                    if !required {
                        self.listed += cost;
                    }
                    self.found.insert(path);
                }
            }
        }
        Ok(())
    }
}
fn selected_results(root: &Directory, policy: &Policy) -> io::Result<(Vec<String>, Skipped)> {
    let required = &policy.required_files;
    let mut walk = ResultWalk {
        policy,
        found: BTreeSet::new(),
        skipped: Skipped::default(),
        entries: 0,
        limit: artifacts::MAX_FILES.saturating_sub(required.len()),
        listed: required.iter().map(|p| artifacts::listed_cost(p)).sum(),
    };
    walk.visit(root, "", 0)?;
    // A required file past the limit still leaves. One that cannot be opened stays
    // absent here, and `Policy::check` then fails the archive.
    for path in required {
        if !walk.found.contains(path) && open(root, path, Kind::Results).is_ok() {
            walk.found.insert(path.clone());
        }
    }
    Ok((walk.found.into_iter().collect(), walk.skipped))
}
fn selected_paths(
    root: &Directory,
    kind: Kind,
    policy: &Policy,
) -> io::Result<(Vec<String>, Skipped)> {
    if kind == Kind::Results {
        return selected_results(root, policy);
    }
    let mut paths = Vec::new();
    walk(root, "", &mut paths, &mut 0)?;
    if paths.is_empty() {
        return Err(invalid());
    }
    paths.sort();
    Ok((paths, Skipped::default()))
}
/// A dataset tree is all or nothing: its identity is every file it holds.
pub(crate) fn walk(
    directory: &Directory,
    prefix: &str,
    found: &mut Vec<String>,
    budget: &mut usize,
) -> io::Result<()> {
    if prefix.len() > artifacts::MAX_PATH || prefix.split('/').count() > MAX_DEPTH {
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
        let path = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}/{name}")
        };
        if !allowed(Kind::Data, &path) {
            return Err(invalid());
        }
        if let Ok(child) = directory.child(OsStr::new(name), false) {
            walk(&child, &path, found, budget)?;
        } else {
            directory.open_regular_read(OsStr::new(name))?;
            found.push(path);
        }
    }
    Ok(())
}
pub(crate) fn open(root: &Directory, path: &str, kind: Kind) -> io::Result<File> {
    if !allowed(kind, path) {
        return Err(invalid());
    }
    let mut names = path.split('/').collect::<Vec<_>>();
    let name = names.pop().ok_or_else(invalid)?;
    let mut parent = root.clone();
    for n in names {
        parent = parent.child(OsStr::new(n), false)?;
    }
    parent.open_regular_read(OsStr::new(name))
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
/// destination and removes it on failure; no API-host archive is necessary. Only a
/// missing required file refuses the selection. The returned skip list says what the
/// archive leaves out, and may leave it with no members at all.
pub fn pack(
    root: &Directory,
    output: &mut File,
    policy: &Policy,
) -> io::Result<(String, u64, Vec<FileReceipt>, Skipped)> {
    pack_kind(root, output, policy, Kind::Results)
}
pub fn pack_data(
    root: &Directory,
    output: &mut File,
) -> io::Result<(String, u64, Vec<FileReceipt>)> {
    let (sha256, bytes, files, _) = pack_kind(root, output, &Policy::default(), Kind::Data)?;
    Ok((sha256, bytes, files))
}
fn pack_kind(
    root: &Directory,
    output: &mut File,
    policy: &Policy,
    kind: Kind,
) -> io::Result<(String, u64, Vec<FileReceipt>, Skipped)> {
    policy.validate().map_err(|_| invalid())?;
    let selected = selected_paths(root, kind, policy)?;
    let paths = &selected.0;
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
    for path in paths {
        let mut file = open(root, path, kind)?;
        let before = file.metadata()?;
        let mut header = member_header(before.len());
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
        let stable = before.len() == after.len() && before.modified()? == after.modified()?;
        #[cfg(unix)]
        let stable = stable
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec();
        if !stable || bytes != before.len() {
            return Err(invalid());
        }
        files.push(FileReceipt {
            path: path.clone(),
            bytes,
            sha256,
        });
    }
    archive.finish()?;
    let writer = archive.into_inner()?;
    writer.inner.sync_all()?;
    if selected != selected_paths(root, kind, policy)? {
        return Err(invalid());
    }
    // A second hash pass also catches a file changed after its own first read.
    for record in &files {
        let mut file = open(root, &record.path, kind)?;
        let (h, n) = hash(&mut file)?;
        if h != record.sha256 || n != record.bytes {
            return Err(invalid());
        }
    }
    Ok((
        format!("{:x}", writer.hash.finalize()),
        writer.bytes,
        files,
        selected.1,
    ))
}
/// The header `pack` gives every member before its path is set.
fn member_header(bytes: u64) -> tar::Header {
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Regular);
    header.set_mode(0o600);
    header.set_uid(0);
    header.set_gid(0);
    header.set_mtime(0);
    header.set_size(bytes);
    header.set_cksum();
    header
}
/// The tar name of a result member, as `pack` writes it.
fn result_name(path: &str) -> String {
    format!("outputs/{path}")
}
fn padded(bytes: u64) -> Option<u64> {
    bytes.div_ceil(512).checked_mul(512)
}
/// Where `pack` wrote the header of the result member `path`, from its receipt alone:
/// members are written in receipt order, each as one 512-byte header and its bytes
/// padded to 512, after a GNU long-name entry when the tar name is over 100 bytes.
/// `None` when the receipt does not list `path`. A caller reads the header there and
/// compares it with `result_member_header` before it trusts any byte after it.
pub fn result_member_offset(files: &[FileReceipt], path: &str) -> Option<u64> {
    let mut offset = 0u64;
    for file in files {
        if file.path == path {
            return Some(offset);
        }
        let name = result_name(&file.path).len() as u64;
        if name > 100 {
            // The long-name entry: its own header, then the name and a NUL, padded.
            offset = offset.checked_add(512)?.checked_add(padded(name + 1)?)?;
        }
        offset = offset.checked_add(512)?.checked_add(padded(file.bytes)?)?;
    }
    None
}
/// The exact header `pack` writes for a result member whose tar name fits its header.
pub fn result_member_header(path: &str, bytes: u64) -> io::Result<[u8; 512]> {
    let name = result_name(path);
    if name.len() > 100 {
        return Err(invalid());
    }
    let mut header = member_header(bytes);
    header.set_path(&name)?;
    header.set_cksum();
    Ok(*header.as_bytes())
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

pub fn estimate(root: &Directory, policy: &Policy) -> io::Result<u64> {
    policy.validate().map_err(|_| invalid())?;
    let (paths, _) = selected_results(root, policy)?;
    let mut bytes = 16u64 * 1024 * 1024;
    for path in paths {
        let file = open(root, &path, Kind::Results)?;
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
    ml_exp_core::multipart::part_count(bytes).map_err(|_| invalid())?;
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    #[test]
    fn a_fifo_in_a_packed_tree_is_refused_rather_than_read_as_empty() {
        let root = tempfile::tempdir().unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        std::fs::create_dir(root.path().join("nested")).unwrap();
        std::fs::write(root.path().join("nested/train.bin"), b"training bytes").unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(root.path().join("nested/pipe"))
                .status()
                .unwrap()
                .success()
        );
        assert!(pack_data(&directory, &mut tempfile::tempfile().unwrap()).is_err());
    }
    #[test]
    fn dataset_trees_stay_all_or_nothing_for_every_path_results_would_skip() {
        for odd_name in [true, false] {
            let root = tempfile::tempdir().unwrap();
            std::fs::write(root.path().join("kept"), b"x").unwrap();
            if odd_name {
                std::fs::write(root.path().join("line\nbreak"), b"x").unwrap();
            } else {
                symlink("kept", root.path().join("link")).unwrap();
            }
            let directory = Directory::open(root.path(), false).unwrap();
            assert!(pack_data(&directory, &mut tempfile::tempfile().unwrap()).is_err());
        }
        let empty = tempfile::tempdir().unwrap();
        let directory = Directory::open(empty.path(), false).unwrap();
        assert!(pack_data(&directory, &mut tempfile::tempfile().unwrap()).is_err());
    }
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
    fn streaming_roundtrip_verifies_every_file() {
        let root = tempfile::tempdir().unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        std::fs::create_dir(root.path().join("nested")).unwrap();
        std::fs::write(root.path().join("nested/weights.bin"), [0, 1, 2, 255]).unwrap();
        std::fs::write(root.path().join("empty.log"), []).unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let (sha, bytes, files, skipped) = pack(
            &directory,
            &mut file,
            &Policy {
                required_files: vec!["nested/weights.bin".into()],
            },
        )
        .unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(skipped, Skipped::default());
        let receipt = receipt(sha, bytes, files);
        let proof = verify(&mut file, &receipt).unwrap();
        artifacts::proof_matches(&receipt, &proof).unwrap();
        file.seek(SeekFrom::Start(600)).unwrap();
        file.write_all(&[42]).unwrap();
        assert!(verify(&mut file, &receipt).is_err());
    }
    /// The receipt alone places every member `pack` wrote, including after names that
    /// need a GNU long-name entry and at the exact 100-byte boundary of the header.
    #[test]
    fn a_member_is_found_from_its_receipt_where_pack_wrote_it() {
        let root = tempfile::tempdir().unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        let fits = "a".repeat(92);
        let over = "b".repeat(93);
        let deep = format!("c/{}/{}", "d".repeat(150), "e".repeat(200));
        std::fs::write(root.path().join(&fits), [1u8; 700]).unwrap();
        std::fs::write(root.path().join(&over), [2u8; 512]).unwrap();
        std::fs::create_dir_all(root.path().join(&deep).parent().unwrap()).unwrap();
        std::fs::write(root.path().join(&deep), b"deep").unwrap();
        std::fs::write(root.path().join("empty.log"), []).unwrap();
        let metrics = b"{\"step\":1,\"loss\":2.5}\n{\"step\":2,\"loss\":2.25}\n";
        std::fs::write(root.path().join("metrics.jsonl"), metrics).unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let (_, bytes, files, _) = pack(&directory, &mut file, &Policy::default()).unwrap();
        let mut tar = Vec::new();
        file.rewind().unwrap();
        file.read_to_end(&mut tar).unwrap();
        assert_eq!(tar.len() as u64, bytes);
        for record in &files {
            let mut offset = result_member_offset(&files, &record.path).unwrap() as usize;
            let name = format!("outputs/{}", record.path);
            if name.len() > 100 {
                assert_eq!(tar[offset + 156], b'L', "{name}");
                offset += 512 + (name.len() + 1).div_ceil(512) * 512;
            } else {
                assert_eq!(
                    tar[offset..offset + 512],
                    result_member_header(&record.path, record.bytes).unwrap()
                );
            }
            let header = tar::Header::from_byte_slice(&tar[offset..offset + 512]);
            assert_eq!(header.size().unwrap(), record.bytes, "{name}");
            let content = &tar[offset + 512..offset + 512 + record.bytes as usize];
            assert_eq!(format!("{:x}", Sha256::digest(content)), record.sha256);
        }
        let at = result_member_offset(&files, "metrics.jsonl").unwrap() as usize + 512;
        assert_eq!(&tar[at..at + metrics.len()], metrics);
        assert!(result_member_offset(&files, "absent.jsonl").is_none());
        assert!(result_member_header(&over, 512).is_err());
    }
    #[test]
    fn a_checkpoints_directory_in_outputs_is_an_ordinary_result() {
        let root = tempfile::tempdir().unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        std::fs::create_dir(root.path().join("checkpoints")).unwrap();
        std::fs::write(root.path().join("checkpoints/state.pt"), b"kept").unwrap();
        std::fs::write(root.path().join("checkpoint.ready.json"), b"{}").unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let (sha, bytes, files, _) = pack(&directory, &mut file, &Policy::default()).unwrap();
        let paths: Vec<_> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["checkpoint.ready.json", "checkpoints/state.pt"]);
        let receipt = receipt(sha, bytes, files);
        verify(&mut file, &receipt).unwrap();
        // Naming a path adds nothing: it only requires what the walk already found.
        let named = Policy {
            required_files: vec!["checkpoints/state.pt".into()],
        };
        assert_eq!(
            estimate(&directory, &named).unwrap(),
            estimate(&directory, &Policy::default()).unwrap()
        );
        let absent = Policy {
            required_files: vec!["checkpoints/absent.pt".into()],
        };
        assert!(pack(&directory, &mut tempfile::tempfile().unwrap(), &absent).is_err());
    }
    fn rows(skipped: &Skipped) -> Vec<(&str, SkipReason)> {
        skipped
            .paths
            .iter()
            .map(|row| (row.path.as_str(), row.reason))
            .collect()
    }
    #[test]
    fn hidden_names_and_an_outputs_entry_leave_with_the_rest_and_verify() {
        let root = tempfile::tempdir().unwrap();
        for dir in [".hydra", "outputs/nested"] {
            std::fs::create_dir_all(root.path().join(dir)).unwrap();
        }
        for file in [
            ".hydra/config.yaml",
            ".env",
            "outputs/nested/final.pt",
            "metrics.jsonl",
        ] {
            std::fs::write(root.path().join(file), file.as_bytes()).unwrap();
        }
        let directory = Directory::open(root.path(), false).unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let (sha, bytes, files, skipped) = pack(
            &directory,
            &mut file,
            &Policy {
                required_files: vec![".hydra/config.yaml".into()],
            },
        )
        .unwrap();
        assert_eq!(skipped, Skipped::default());
        let paths: Vec<_> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                ".env",
                ".hydra/config.yaml",
                "metrics.jsonl",
                "outputs/nested/final.pt"
            ]
        );
        let receipt = receipt(sha, bytes, files);
        let proof = verify(&mut file, &receipt).unwrap();
        assert!(proof.files.contains_key("outputs/outputs/nested/final.pt"));
        file.rewind().unwrap();
        let mut copied = Vec::new();
        copy_result_file(&mut file, &receipt, "outputs/nested/final.pt", &mut copied).unwrap();
        assert_eq!(copied, b"outputs/nested/final.pt");
    }
    #[test]
    fn links_special_files_and_unsupported_names_are_recorded_and_never_followed() {
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), b"never archived").unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("data")).unwrap();
        std::fs::write(root.path().join("data/kept.bin"), b"kept").unwrap();
        symlink(outside.path(), root.path().join("data/to-directory")).unwrap();
        symlink(outside.path().join("secret"), root.path().join("to-file")).unwrap();
        symlink("/absent", root.path().join("dangling")).unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(root.path().join("pipe"))
                .status()
                .unwrap()
                .success()
        );
        std::fs::write(root.path().join("line\nbreak"), b"x").unwrap();
        std::fs::write(root.path().join("back\\slash"), b"x").unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let (sha, size, files, skipped) = pack(&directory, &mut file, &Policy::default()).unwrap();
        let paths: Vec<_> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["data/kept.bin"]);
        assert_eq!(
            rows(&skipped),
            [
                ("back\\slash", SkipReason::UnsupportedPath),
                ("dangling", SkipReason::SymbolicLink),
                ("data/to-directory", SkipReason::SymbolicLink),
                ("line\\nbreak", SkipReason::UnsupportedPath),
                ("pipe", SkipReason::SpecialFile),
                ("to-file", SkipReason::SymbolicLink),
            ]
        );
        assert_eq!(skipped.unlisted, 0);
        verify(&mut file, &receipt(sha, size, files)).unwrap();
        // The size check covers only what is actually packed.
        assert_eq!(
            estimate(&directory, &Policy::default()).unwrap(),
            16 * 1024 * 1024 + 4 + 8192
        );
        // The same tree yields the same archive and the same list.
        let again = pack(
            &directory,
            &mut tempfile::tempfile().unwrap(),
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(again.3, skipped);
    }
    #[test]
    fn a_name_that_is_not_utf8_is_recorded_when_the_filesystem_holds_one() {
        use std::os::unix::ffi::OsStrExt;
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("kept"), b"x").unwrap();
        // Some filesystems refuse this name; then there is nothing to walk past.
        if std::fs::write(root.path().join(OsStr::from_bytes(b"odd-\xff")), b"x").is_err() {
            return;
        }
        let directory = Directory::open(root.path(), false).unwrap();
        let (_, _, files, skipped) = pack(
            &directory,
            &mut tempfile::tempfile().unwrap(),
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(
            rows(&skipped),
            [("odd-\u{fffd}", SkipReason::UnsupportedPath)]
        );
    }
    #[test]
    fn a_required_link_or_special_file_is_a_missing_file_and_fails() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("real"), b"x").unwrap();
        symlink("real", root.path().join("weights.bin")).unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        let policy = Policy {
            required_files: vec!["weights.bin".into()],
        };
        assert!(pack(&directory, &mut tempfile::tempfile().unwrap(), &policy).is_err());
        assert!(estimate(&directory, &Policy::default()).is_ok());
    }
    #[test]
    fn an_output_directory_with_nothing_to_archive_still_leaves_its_skip_list() {
        let root = tempfile::tempdir().unwrap();
        symlink("/absent", root.path().join("link")).unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let (sha, bytes, files, skipped) = pack(&directory, &mut file, &Policy::default()).unwrap();
        assert!(files.is_empty());
        assert_eq!(rows(&skipped), [("link", SkipReason::SymbolicLink)]);
        let receipt = receipt(sha, bytes, files);
        let proof = verify(&mut file, &receipt).unwrap();
        artifacts::proof_matches(&receipt, &proof).unwrap();
        file.rewind().unwrap();
        verify_stream(&mut file, &receipt).unwrap();
    }
    #[test]
    fn a_failed_read_fails_the_archive_instead_of_losing_the_directory_silently() {
        use std::os::unix::fs::PermissionsExt;
        if crate::directory::effective_uid() == 0 {
            // Root opens a mode 000 directory, so there is no failed read to observe.
            return;
        }
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("kept"), b"x").unwrap();
        let locked = root.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        std::fs::write(locked.join("weights.bin"), b"x").unwrap();
        let mode = |bits| std::fs::Permissions::from_mode(bits);
        std::fs::set_permissions(&locked, mode(0o000)).unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        let error = pack(
            &directory,
            &mut tempfile::tempfile().unwrap(),
            &Policy::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert!(estimate(&directory, &Policy::default()).is_err());
        std::fs::set_permissions(&locked, mode(0o700)).unwrap();
    }
    #[test]
    fn past_the_file_limit_the_skip_list_names_what_is_missing_and_a_required_file_still_leaves() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("a")).unwrap();
        for n in 0..artifacts::MAX_FILES {
            std::fs::write(root.path().join(format!("a/{n:05}")), []).unwrap();
        }
        std::fs::write(root.path().join("b-not-required"), b"x").unwrap();
        std::fs::create_dir(root.path().join("c")).unwrap();
        std::fs::write(root.path().join("c/under-a-skipped-directory"), b"x").unwrap();
        std::fs::write(root.path().join("weights.bin"), b"weights").unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        let policy = Policy {
            required_files: vec!["weights.bin".into(), "c/under-a-skipped-directory".into()],
        };
        let (sha, bytes, files, skipped) =
            pack(&directory, &mut tempfile::tempfile().unwrap(), &policy).unwrap();
        // The directory `a` itself takes one of the entries the two required files left.
        assert_eq!(files.len(), artifacts::MAX_FILES - 1);
        for path in ["weights.bin", "c/under-a-skipped-directory"] {
            assert!(files.iter().any(|f| f.path == path), "{path}");
        }
        assert!(!files.iter().any(|f| f.path == "b-not-required"));
        assert_eq!(
            rows(&skipped),
            [
                ("a/19997", SkipReason::FileLimit),
                ("a/19998", SkipReason::FileLimit),
                ("a/19999", SkipReason::FileLimit),
                ("b-not-required", SkipReason::FileLimit),
                ("c", SkipReason::FileLimit),
            ]
        );
        artifacts::validate_receipt(&receipt(sha, bytes, files)).unwrap();
    }
    #[test]
    fn one_directory_past_the_limit_is_read_in_sort_order_and_counted() {
        let root = tempfile::tempdir().unwrap();
        for n in 0..artifacts::MAX_FILES + 3 {
            std::fs::write(root.path().join(format!("{n:05}")), []).unwrap();
        }
        let directory = Directory::open(root.path(), false).unwrap();
        let (files, skipped) = selected_results(&directory, &Policy::default()).unwrap();
        assert_eq!(files.len(), artifacts::MAX_FILES);
        assert_eq!(files.last().unwrap(), "19999");
        assert_eq!(rows(&skipped), [(".", SkipReason::FileLimit)]);
        assert_eq!(skipped.unlisted, 3);
    }
    #[test]
    fn long_paths_stop_at_what_a_saved_state_can_hold() {
        let root = tempfile::tempdir().unwrap();
        let name = "n".repeat(240);
        let count = 8000;
        for n in 0..count {
            std::fs::write(root.path().join(format!("{name}{n:05}")), []).unwrap();
        }
        let directory = Directory::open(root.path(), false).unwrap();
        let (files, skipped) = selected_results(&directory, &Policy::default()).unwrap();
        assert!(files.len() < count && !skipped.paths.is_empty());
        assert_eq!(
            files.len() + skipped.paths.len() + skipped.unlisted as usize,
            count
        );
        // A receipt and a proof of these files fit the 8 MiB read limit on a saved State.
        let rows: Vec<_> = files
            .into_iter()
            .map(|path| FileReceipt {
                path,
                bytes: u64::MAX,
                sha256: "0".repeat(64),
            })
            .collect();
        let saved = serde_json::to_vec(&serde_json::json!({
            "receipt": rows,
            "proof": rows.iter().map(|f| (format!("outputs/{}", f.path), f)).collect::<Vec<_>>(),
            "skipped": skipped,
        }))
        .unwrap();
        assert!(saved.len() < 8 * 1024 * 1024, "{}", saved.len());
    }
    #[test]
    fn missing_required_files_and_changed_hashes_fail() {
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
        let mut file = tempfile::tempfile().unwrap();
        let (sha, bytes, files, _) = pack(&directory, &mut file, &Policy::default()).unwrap();
        let mut wrong = receipt(sha, bytes, files);
        wrong.files[0].sha256 = "0".repeat(64);
        assert!(verify(&mut file, &wrong).is_err());
    }
}
