//! Shared filesystem boundary. No domain decisions, HTTP, provider calls or hidden configuration.
//! Durable writes and bounded private reads never silently replace an immutable record.
#[cfg(unix)]
pub mod checkpoints;
#[cfg(unix)]
pub mod directory;
#[cfg(unix)]
pub mod framing;
#[cfg(unix)]
pub mod process;

use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

pub fn read_private(path: &Path, limit: u64, require_owner: bool) -> io::Result<Vec<u8>> {
    let read_limit = limit
        .checked_add(1)
        .ok_or_else(|| io::Error::other("invalid private file bound"))?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.len() > limit {
        return Err(io::Error::other("private file must be regular and bounded"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // SAFETY: geteuid has no preconditions and does not access Rust memory.
        let uid = unsafe { libc::geteuid() };
        if meta.mode() & 0o077 != 0 || require_owner && meta.uid() != uid {
            return Err(io::Error::other(
                "private file ownership or permissions invalid",
            ));
        }
    }
    let mut bytes = Vec::new();
    file.take(read_limit).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(io::Error::other("private file exceeds limit"));
    }
    Ok(bytes)
}

pub fn write_atomic(path: &Path, bytes: &[u8], immutable: bool) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    if immutable {
        temporary.persist_noclobber(path).map_err(|e| e.error)?;
    } else {
        temporary.persist(path).map_err(|e| e.error)?;
    }
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let before = file.metadata()?;
    if !before.is_file() {
        return Err(io::Error::other("hash input must be a regular file"));
    }
    // Bound the read to the snapshot size: a growing log must not extend it forever.
    let mut reader = (&file).take(before.len());
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    let after = file.metadata()?;
    if before.len() != after.len() || before.modified()? != after.modified()? {
        return Err(io::Error::other("hash input changed while reading"));
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn create_private_dir(path: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immutable_publication_survives_duplicate_and_replacement_is_explicit() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("state.json");
        write_atomic(&path, b"first", true).unwrap();
        assert!(write_atomic(&path, b"second", true).is_err());
        assert_eq!(read_private(&path, 5, true).unwrap(), b"first");
        assert!(read_private(&path, 4, true).is_err());
        write_atomic(&path, b"second", false).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
    }
    #[cfg(unix)]
    #[test]
    fn reject_public_files_symlinks_and_fifos_without_blocking() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("token");
        fs::write(&path, b"test-only-secret").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read_private(&path, 100, true).is_err());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let link = temp.path().join("link");
        symlink(&path, &link).unwrap();
        assert!(read_private(&link, 100, true).is_err());
        let fifo = temp.path().join("fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(status.success());
        assert!(read_private(&fifo, 100, true).is_err());
        assert!(sha256_file(&fifo).is_err());
        assert!(sha256_file(&link).is_err());
    }
}
