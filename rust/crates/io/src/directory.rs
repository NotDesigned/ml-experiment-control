//! Unix directory capabilities. All leaf I/O is relative to retained directory
//! descriptors; path-based checks are never followed by path-based writes.
//!
//! Newly created directory entries and files are fsynced before success. Parent
//! attachment checks detect a renamed/replaced ancestor and fail closed. The
//! actual operation still uses retained descriptors, even if a rename races a
//! check. Atomic no-replace publication requires Linux/Android or Apple rename
//! support; an unsupported platform fails rather than emulating it unsafely.
#![forbid(unsafe_code)]

use rustix::fs::{self as rfs, AtFlags, FileType, Mode, OFlags};
use std::{
    ffi::{OsStr, OsString},
    fmt::Write as FmtWrite,
    fs::File,
    io::{self, Read, Write},
    os::unix::ffi::OsStrExt,
    path::{Component, Path},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub struct Directory {
    inner: Arc<Inner>,
}
#[derive(Debug)]
struct Inner {
    file: File,
    parent: Option<Directory>,
    name: OsString,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn leaf(name: &OsStr) -> io::Result<()> {
    let bytes = name.as_bytes();
    if bytes.is_empty()
        || bytes.contains(&b'/')
        || bytes.contains(&0)
        || matches!(bytes, b"." | b"..")
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "one relative directory entry is required",
        ));
    }
    let mut parts = Path::new(name).components();
    if !matches!(parts.next(), Some(Component::Normal(_))) || parts.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "one relative directory entry is required",
        ));
    }
    Ok(())
}
fn same_file(file: &File, stat: &rfs::Stat) -> io::Result<bool> {
    let metadata = rfs::fstat(file)?;
    Ok(metadata.st_dev == stat.st_dev && metadata.st_ino == stat.st_ino)
}
fn directory_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK
}

impl Directory {
    /// Walk an explicit absolute path one component at a time, never following
    /// any symlink. The retained ancestry detects replacement of an old root.
    pub fn open(path: &Path, create: bool) -> io::Result<Self> {
        if !path.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "directory capability needs an absolute path",
            ));
        }
        let file = File::from(rfs::open("/", directory_flags(), Mode::empty())?);
        let mut directory = Self {
            inner: Arc::new(Inner {
                file,
                parent: None,
                name: OsString::from("/"),
            }),
        };
        for part in path.components() {
            match part {
                Component::RootDir => {}
                Component::Normal(name) => directory = directory.child(name, create)?,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "dot/parent directory traversal is not allowed",
                    ));
                }
            }
        }
        directory.verify_attached()?;
        Ok(directory)
    }

    pub fn verify_attached(&self) -> io::Result<()> {
        if let Some(parent) = &self.inner.parent {
            parent.verify_attached()?;
            let entry = rfs::statat(
                &parent.inner.file,
                &self.inner.name,
                AtFlags::SYMLINK_NOFOLLOW,
            )
            .map_err(|error| {
                if error == rustix::io::Errno::NOENT {
                    invalid("anchored directory was removed")
                } else {
                    error.into()
                }
            })?;
            if FileType::from_raw_mode(entry.st_mode) != FileType::Directory
                || !same_file(&self.inner.file, &entry)?
            {
                return Err(invalid("directory was renamed or replaced"));
            }
        }
        Ok(())
    }

    /// A created directory is private from its first instant, and its parent
    /// directory entry is synced before returning its fixed child descriptor.
    pub fn child(&self, name: &OsStr, create: bool) -> io::Result<Self> {
        leaf(name)?;
        self.verify_attached()?;
        let created = if create {
            match rfs::mkdirat(&self.inner.file, name, Mode::RWXU) {
                Ok(()) => {
                    rfs::fsync(&self.inner.file)?;
                    true
                }
                Err(rustix::io::Errno::EXIST) => false,
                Err(error) => return Err(error.into()),
            }
        } else {
            false
        };
        let file = match rfs::openat(&self.inner.file, name, directory_flags(), Mode::empty()) {
            Ok(file) => File::from(file),
            Err(error) => {
                self.verify_attached()?;
                return Err(error.into());
            }
        };
        if created {
            rfs::fsync(&file)?;
        }
        let result = Self {
            inner: Arc::new(Inner {
                file,
                parent: Some(self.clone()),
                name: name.to_owned(),
            }),
        };
        result.verify_attached()?;
        Ok(result)
    }

    /// Any entry counts, including a directory, dangling symlink or FIFO. This
    /// is deliberately different from opening/validating a regular payload.
    pub fn entry_exists(&self, name: &OsStr) -> io::Result<bool> {
        leaf(name)?;
        self.verify_attached()?;
        let result = match rfs::statat(&self.inner.file, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(_) => true,
            Err(rustix::io::Errno::NOENT) => false,
            Err(error) => return Err(error.into()),
        };
        self.verify_attached()?;
        Ok(result)
    }

    pub fn validate_regular_entry(&self, name: &OsStr, file: &File) -> io::Result<()> {
        leaf(name)?;
        self.verify_attached()?;
        let entry = rfs::statat(&self.inner.file, name, AtFlags::SYMLINK_NOFOLLOW)?;
        if FileType::from_raw_mode(entry.st_mode) != FileType::RegularFile
            || !same_file(file, &entry)?
        {
            return Err(invalid("regular entry was replaced"));
        }
        Ok(())
    }

    /// A stable, non-truncating lock leaf. The caller must also validate its
    /// binding after acquiring its advisory lock and before guarded effects.
    pub fn open_lock(&self, name: &OsStr) -> io::Result<File> {
        leaf(name)?;
        self.verify_attached()?;
        let file = File::from(rfs::openat(
            &self.inner.file,
            name,
            OFlags::RDWR | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::RUSR | Mode::WUSR,
        )?);
        self.validate_regular_entry(name, &file)?;
        rfs::fsync(&self.inner.file)?;
        Ok(file)
    }

    /// Open a pre-existing lock for shared observation, without creating a leaf,
    /// directory, temporary or journal. Missing or invalid entries are errors.
    pub fn open_existing_lock(&self, name: &OsStr) -> io::Result<File> {
        leaf(name)?;
        self.verify_attached()?;
        let file = File::from(rfs::openat(
            &self.inner.file,
            name,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )?);
        self.validate_regular_entry(name, &file)?;
        Ok(file)
    }

    pub fn read_regular(&self, name: &OsStr, maximum: u64) -> io::Result<Vec<u8>> {
        let bound = maximum
            .checked_add(1)
            .ok_or_else(|| invalid("invalid read bound"))?;
        leaf(name)?;
        self.verify_attached()?;
        let file = File::from(rfs::openat(
            &self.inner.file,
            name,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )?);
        self.validate_regular_entry(name, &file)?;
        if file.metadata()?.len() > maximum {
            return Err(invalid("regular file exceeds limit"));
        }
        let mut bytes = Vec::new();
        (&file).take(bound).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > maximum {
            return Err(invalid("regular file exceeds limit"));
        }
        self.validate_regular_entry(name, &file)?;
        Ok(bytes)
    }

    /// Publish immutable bytes with a same-directory unpredictable temporary,
    /// retained file identity, fsync, atomic RENAME_NOREPLACE and directory fsync.
    /// There is no check-then-replace fallback and no path traversal during I/O.
    pub fn publish_new(&self, name: &OsStr, bytes: &[u8]) -> io::Result<()> {
        leaf(name)?;
        self.verify_attached()?;
        let (temporary, mut file) = self.temporary()?;
        let result = (|| {
            file.write_all(bytes)?;
            file.sync_all()?;
            self.validate_regular_entry(&temporary, &file)?;
            rename_new(&self.inner.file, &temporary, name)?;
            rfs::fsync(&self.inner.file)?;
            self.verify_attached()?;
            Ok(())
        })();
        if result.is_err() && self.validate_regular_entry(&temporary, &file).is_ok() {
            let _ = rfs::unlinkat(&self.inner.file, &temporary, AtFlags::empty());
            let _ = rfs::fsync(&self.inner.file);
        }
        result
    }

    fn temporary(&self) -> io::Result<(OsString, File)> {
        for _ in 0..16 {
            let mut random = [0u8; 16];
            getrandom::fill(&mut random)
                .map_err(|_| invalid("cannot obtain temporary-file randomness"))?;
            let mut suffix = String::with_capacity(32);
            for byte in random {
                write!(&mut suffix, "{byte:02x}")
                    .map_err(|_| invalid("temporary-name formatting failed"))?;
            }
            let name = OsString::from(format!(".ml-expd-tmp-{suffix}"));
            match rfs::openat(
                &self.inner.file,
                &name,
                OFlags::WRONLY
                    | OFlags::CREATE
                    | OFlags::EXCL
                    | OFlags::CLOEXEC
                    | OFlags::NOFOLLOW
                    | OFlags::NONBLOCK,
                Mode::RUSR | Mode::WUSR,
            ) {
                Ok(file) => return Ok((name, File::from(file))),
                Err(rustix::io::Errno::EXIST) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "cannot allocate a unique temporary entry",
        ))
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
))]
fn rename_new(directory: &File, from: &OsStr, to: &OsStr) -> io::Result<()> {
    rfs::renameat_with(directory, from, directory, to, rfs::RenameFlags::NOREPLACE)
        .map_err(Into::into)
}
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
)))]
fn rename_new(_directory: &File, _from: &OsStr, _to: &OsStr) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace publication is unavailable on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::symlink};

    #[test]
    fn capabilities_publish_once_and_never_follow_intermediate_or_leaf_links() {
        let temp = tempfile::tempdir().unwrap();
        let root = Directory::open(&temp.path().join("a/b"), true).unwrap();
        root.publish_new(OsStr::new("receipt"), b"first").unwrap();
        assert_eq!(
            root.publish_new(OsStr::new("receipt"), b"second")
                .unwrap_err()
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(
            root.read_regular(OsStr::new("receipt"), 5).unwrap(),
            b"first"
        );
        assert!(root.read_regular(OsStr::new("receipt"), 4).is_err());
        symlink(temp.path().join("outside"), temp.path().join("a/link")).unwrap();
        assert!(Directory::open(&temp.path().join("a/link/child"), true).is_err());
        symlink(temp.path().join("absent"), temp.path().join("a/b/broken")).unwrap();
        assert!(root.entry_exists(OsStr::new("broken")).unwrap());
        assert!(root.read_regular(OsStr::new("broken"), 10).is_err());
        assert_eq!(
            root.publish_new(OsStr::new("broken"), b"bad")
                .unwrap_err()
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert!(root.publish_new(OsStr::new("../escape"), b"bad").is_err());
        assert!(!temp.path().join("outside").exists());
    }

    #[test]
    fn renamed_parent_and_replaced_directory_are_detected_without_redirecting_writes() {
        let temp = tempfile::tempdir().unwrap();
        let old = temp.path().join("root");
        let outside = temp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        let root = Directory::open(&old, true).unwrap();
        let child = root.child(OsStr::new("project"), true).unwrap();
        fs::rename(old.join("project"), old.join("moved")).unwrap();
        symlink(&outside, old.join("project")).unwrap();
        assert!(child.publish_new(OsStr::new("receipt"), b"bad").is_err());
        assert!(child.entry_exists(OsStr::new("receipt")).is_err());
        assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
        assert_eq!(fs::read_dir(old.join("moved")).unwrap().count(), 0);
        fs::rename(&old, temp.path().join("old-root")).unwrap();
        fs::create_dir(&old).unwrap();
        assert!(root.open_lock(OsStr::new("lock")).is_err());
        assert_eq!(fs::read_dir(&old).unwrap().count(), 0);
    }

    #[test]
    fn fifo_and_directory_leaf_never_block_or_become_regular_files() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let path = temp.path().join("fifo");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        assert!(directory.open_lock(OsStr::new("fifo")).is_err());
        assert!(directory.read_regular(OsStr::new("fifo"), 10).is_err());
        fs::create_dir(temp.path().join("folder")).unwrap();
        assert!(directory.open_lock(OsStr::new("folder")).is_err());
        assert!(directory.entry_exists(OsStr::new("fifo")).unwrap());
    }

    #[test]
    fn concurrent_publications_have_one_winner_and_keep_the_original_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let threads: Vec<_> = (0..4)
            .map(|number| {
                let directory = directory.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let bytes = format!("receipt-{number}").into_bytes();
                    barrier.wait();
                    let result = directory.publish_new(OsStr::new("receipt"), &bytes);
                    match result {
                        Ok(()) => Some(bytes),
                        Err(error) => {
                            assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
                            None
                        }
                    }
                })
            })
            .collect();
        let winners: Vec<_> = threads
            .into_iter()
            .filter_map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(winners.len(), 1);
        assert_eq!(
            directory.read_regular(OsStr::new("receipt"), 1024).unwrap(),
            winners[0]
        );
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
    }

    #[test]
    fn existing_lock_open_never_creates_or_repairs_a_leaf() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        assert_eq!(
            directory
                .open_existing_lock(OsStr::new("missing"))
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
        fs::write(temp.path().join("lock"), b"original").unwrap();
        let file = directory.open_existing_lock(OsStr::new("lock")).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 8);
        assert_eq!(fs::read(temp.path().join("lock")).unwrap(), b"original");
        symlink(temp.path().join("lock"), temp.path().join("alias")).unwrap();
        assert!(directory.open_existing_lock(OsStr::new("alias")).is_err());
        assert!(
            std::process::Command::new("mkfifo")
                .arg(temp.path().join("fifo"))
                .status()
                .unwrap()
                .success()
        );
        assert!(directory.open_existing_lock(OsStr::new("fifo")).is_err());
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 3);
    }
}
