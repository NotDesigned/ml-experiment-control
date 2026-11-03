//! Bounded tar/gzip decoding. No archive path is passed to `unpack` or a shell.
use super::store::Error;
use flate2::read::MultiGzDecoder;
use ml_exp_core::sources::{
    ArchiveState, CHUNK_BYTES, MAX_METADATA_BYTES, MAX_PATH_BYTES, MemberKind, Policy,
};
use ml_exp_io::directory::Directory;
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
};

struct Bounded<R> {
    reader: R,
    remaining: u64,
}
impl<R: Read> Read for Bounded<R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let maximum = bytes
            .len()
            .min(self.remaining.saturating_add(1).min(usize::MAX as u64) as usize);
        let n = self.reader.read(&mut bytes[..maximum])?;
        if n as u64 > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expanded archive exceeds its bound",
            ));
        }
        self.remaining -= n as u64;
        Ok(n)
    }
}
pub(super) fn directory(root: &Directory, path: &str, create: bool) -> Result<Directory, Error> {
    let mut current = root.clone();
    if !path.is_empty() {
        for part in path.split('/') {
            current = current.child(OsStr::new(part), create)?;
            current.verify_private_parent()?;
        }
    }
    Ok(current)
}
fn path_name(path: &str) -> (&str, &str) {
    path.rsplit_once('/').unwrap_or(("", path))
}
fn extension<R: Read>(entry: &mut tar::Entry<'_, R>, limit: usize) -> Result<Vec<u8>, Error> {
    if entry.size() > limit as u64 {
        return Err(Error::Limit);
    }
    let expected = entry.size() as usize;
    let mut bytes = Vec::with_capacity(expected);
    entry.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() != expected {
        return Err(Error::Archive);
    }
    Ok(bytes)
}

pub(super) fn extract(file: &mut File, tree: &Directory, policy: &Policy) -> Result<(), Error> {
    let mut magic = [0u8; 2];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut magic).map_err(|_| Error::Archive)?;
    file.seek(SeekFrom::Start(0))?;
    let expanded: Box<dyn Read + '_> = if magic == [0x1f, 0x8b] {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let reader = Bounded {
        reader: expanded,
        remaining: policy.expanded_archive_limit()?,
    };
    let mut archive = tar::Archive::new(reader);
    let mut state = ArchiveState::new(policy.clone())?;
    let mut pending_path = None::<String>;
    let mut pending_size = None::<u64>;
    let mut headers = 0usize;
    for entry in archive.entries().map_err(|_| Error::Archive)?.raw(true) {
        let mut entry = entry.map_err(|_| Error::Archive)?;
        headers = headers.checked_add(1).ok_or(Error::Limit)?;
        if headers > policy.max_source_files.saturating_mul(3).saturating_add(16) {
            return Err(Error::Limit);
        }
        let kind = entry.header().entry_type();
        if kind.is_pax_local_extensions() {
            let bytes = extension(&mut entry, MAX_METADATA_BYTES)?;
            let mut values = BTreeMap::new();
            for item in tar::PaxExtensions::new(&bytes) {
                let item = item.map_err(|_| Error::Archive)?;
                let key = item.key().map_err(|_| Error::Archive)?;
                let value = item.value().map_err(|_| Error::Archive)?;
                if values.insert(key.to_owned(), value.to_owned()).is_some() {
                    return Err(Error::Archive);
                }
            }
            for (key, value) in values {
                match key.as_str() {
                    "path" => {
                        if pending_path.replace(value).is_some() {
                            return Err(Error::Archive);
                        }
                    }
                    "size" => {
                        if pending_size
                            .replace(value.parse().map_err(|_| Error::Archive)?)
                            .is_some()
                        {
                            return Err(Error::Archive);
                        }
                    }
                    "mtime" | "atime" | "ctime" | "uid" | "gid" | "uname" | "gname" | "charset"
                    | "comment" => {}
                    _ => return Err(Error::Archive),
                }
            }
            continue;
        }
        if kind.is_gnu_longname() {
            let mut bytes = extension(&mut entry, MAX_PATH_BYTES + 1)?;
            if bytes.last() == Some(&0) {
                bytes.pop();
            }
            let path = String::from_utf8(bytes).map_err(|_| Error::Archive)?;
            if pending_path.replace(path).is_some() {
                return Err(Error::Archive);
            }
            continue;
        }
        let member_kind = if kind.is_dir() {
            MemberKind::Directory
        } else if kind.is_file() {
            MemberKind::File
        } else {
            MemberKind::Other
        };
        let raw = pending_path.take().unwrap_or(
            std::str::from_utf8(&entry.path_bytes())
                .map_err(|_| Error::Archive)?
                .to_owned(),
        );
        let size = entry.size();
        if pending_size.take().is_some_and(|expected| expected != size) {
            return Err(Error::Archive);
        }
        if member_kind == MemberKind::Directory && size != 0 {
            return Err(Error::Archive);
        }
        let (next, relative) = state.accept(&raw, member_kind, size)?;
        state = next;
        let Some(path) = relative else {
            continue;
        };
        if member_kind == MemberKind::Directory {
            directory(tree, &path, true)?;
            continue;
        }
        let (parent, name) = path_name(&path);
        let parent = directory(tree, parent, true)?;
        let mut output = parent.create_new_regular(OsStr::new(name))?;
        let mut remaining = size;
        let mut buffer = vec![0u8; CHUNK_BYTES];
        while remaining != 0 {
            let maximum = buffer.len().min(remaining as usize);
            let n = entry
                .read(&mut buffer[..maximum])
                .map_err(|_| Error::Archive)?;
            if n == 0 {
                return Err(Error::Archive);
            }
            output.write_all(&buffer[..n])?;
            remaining -= n as u64;
        }
        if output.metadata()?.len() != size {
            return Err(Error::Archive);
        }
        let executable = entry.header().mode().map_err(|_| Error::Archive)? & 0o111 != 0;
        parent.seal_regular(OsStr::new(name), &output, executable)?;
    }
    if pending_path.is_some() || pending_size.is_some() {
        return Err(Error::Archive);
    }
    state.finish()?;
    // Finish gzip CRC/trailer verification and bound trailing tar padding.
    // A second hidden nonzero archive after an end marker is not accepted.
    let mut reader = archive.into_inner();
    let mut buffer = [0u8; 8192];
    loop {
        let n = reader.read(&mut buffer).map_err(|_| Error::Archive)?;
        if n == 0 {
            break;
        }
        if buffer[..n].iter().any(|byte| *byte != 0) {
            return Err(Error::Archive);
        }
    }
    tree.verify_private_parent()?;
    Ok(())
}
