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
    os::unix::{
        ffi::{OsStrExt, OsStringExt},
        fs::PermissionsExt,
    },
    path::{Component, Path},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub struct Directory {
    inner: Arc<Inner>,
}

/// A pathname observation, never a claim about an independently opened handle.
/// SQLite owners use stat rather than extra database FDs: closing any such FD
/// can release POSIX locks belonging to another SQLite connection in-process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegularIdentity {
    pub device: u64,
    pub inode: u64,
    pub links: u64,
    pub mode: u32,
    pub owner: u32,
}

/// Advisory capacity of the filesystem containing this retained directory.
/// Other processes can consume it; a successful check is never a reservation.
#[derive(Clone, Copy, Debug)]
pub struct Capacity {
    pub available_bytes: u64,
    pub available_inodes: u64,
}

pub fn effective_uid() -> u32 {
    rustix::process::geteuid().as_raw()
}

/// Root is the trusted administrator; arbitrary other file owners are not.
pub fn trusted_storage_owner(owner: u32, effective: u32) -> bool {
    owner == 0 || owner == effective
}

impl RegularIdentity {
    pub fn is_private_for(self, effective: u32) -> bool {
        self.links == 1 && self.mode & 0o022 == 0 && trusted_storage_owner(self.owner, effective)
    }
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
    /// Original spelling assembled from retained components, for provenance or
    /// a library that only accepts pathnames. This is NOT a dirfd I/O primitive.
    pub fn path(&self) -> std::path::PathBuf {
        self.inner.parent.as_ref().map_or_else(
            || std::path::PathBuf::from("/"),
            |parent| parent.path().join(&self.inner.name),
        )
    }

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

    /// The leaf containing a SQLite database must not be writable by another
    /// group/user. This complements attachment checks; it cannot restrain root
    /// or another process acting as the same account.
    pub fn verify_private_parent(&self) -> io::Result<()> {
        self.verify_attached()?;
        let stat = rfs::fstat(&self.inner.file)?;
        if stat.st_mode & 0o022 != 0 || !trusted_storage_owner(stat.st_uid, effective_uid()) {
            return Err(invalid(
                "storage directory is writable by another group or user",
            ));
        }
        Ok(())
    }

    /// Stable identity of this retained directory, for binding a trusted root
    /// across process restarts. This is not an independently reopened pathname.
    #[allow(clippy::unnecessary_cast)]
    pub fn identity(&self) -> io::Result<(u64, u64)> {
        self.verify_attached()?;
        let stat = rfs::fstat(&self.inner.file)?;
        Ok((stat.st_dev as u64, stat.st_ino as u64))
    }

    /// Inspect a leaf through the directory FD without opening/closing it.
    /// Absence is distinct from a symlink, FIFO, device or directory.
    #[allow(clippy::unnecessary_cast)] // Stat field widths differ across Unix.
    pub fn regular_identity(&self, name: &OsStr) -> io::Result<Option<RegularIdentity>> {
        leaf(name)?;
        self.verify_attached()?;
        let identity = match rfs::statat(&self.inner.file, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) if FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile => {
                Some(RegularIdentity {
                    device: stat.st_dev as u64,
                    inode: stat.st_ino as u64,
                    links: stat.st_nlink as u64,
                    mode: stat.st_mode as u32,
                    owner: stat.st_uid,
                })
            }
            Ok(_) => return Err(invalid("regular storage entry required")),
            Err(rustix::io::Errno::NOENT) => None,
            Err(error) => return Err(error.into()),
        };
        self.verify_attached()?;
        Ok(identity)
    }

    /// Enumerate names from a fresh directory stream tied to this descriptor.
    /// Callers still validate/open each required child using directory methods.
    pub fn entry_names(&self) -> io::Result<Vec<OsString>> {
        self.verify_attached()?;
        let mut names = Vec::new();
        for entry in rfs::Dir::read_from(&self.inner.file)? {
            let entry = entry?;
            let name = entry.file_name().to_bytes();
            if !matches!(name, b"." | b"..") {
                names.push(OsString::from_vec(name.to_vec()));
            }
        }
        names.sort();
        self.verify_attached()?;
        Ok(names)
    }

    /// Read at most `maximum` directory entries, failing before collecting an
    /// unbounded name list. This neither creates nor skips invalid entries.
    pub fn entry_names_limited(&self, maximum: usize) -> io::Result<Vec<OsString>> {
        self.verify_attached()?;
        let mut names = Vec::new();
        for entry in rfs::Dir::read_from(&self.inner.file)? {
            let entry = entry?;
            let name = entry.file_name().to_bytes();
            if !matches!(name, b"." | b"..") {
                if names.len() >= maximum {
                    return Err(invalid("directory entry limit exceeded"));
                }
                names.push(OsString::from_vec(name.to_vec()));
            }
        }
        names.sort();
        self.verify_attached()?;
        Ok(names)
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

    /// Explicit fresh namespace allocation. Unlike `child(..., true)`, an
    /// existing entry always fails, including an empty directory that appeared
    /// concurrently. Intended for reviewed operator bootstrap, not observation.
    pub fn create_new_child(&self, name: &OsStr) -> io::Result<Self> {
        leaf(name)?;
        self.verify_private_parent()?;
        rfs::mkdirat(&self.inner.file, name, Mode::RWXU)?;
        rfs::fsync(&self.inner.file)?;
        let directory = self.child(name, false)?;
        rfs::fsync(&directory.inner.file)?;
        directory.verify_attached()?;
        Ok(directory)
    }

    /// Stream only into a newly allocated private regular file. This does not
    /// publish immutable content; the caller owns the containing staging tree,
    /// bounds writes, fsyncs and seals before any directory publication.
    pub fn create_new_regular(&self, name: &OsStr) -> io::Result<File> {
        leaf(name)?;
        self.verify_private_parent()?;
        let file = File::from(rfs::openat(
            &self.inner.file,
            name,
            OFlags::RDWR
                | OFlags::CREATE
                | OFlags::EXCL
                | OFlags::CLOEXEC
                | OFlags::NOFOLLOW
                | OFlags::NONBLOCK,
            Mode::RUSR | Mode::WUSR,
        )?);
        self.validate_regular_entry(name, &file)?;
        rfs::fsync(&self.inner.file)?;
        Ok(file)
    }

    /// Seal a retained regular file without following its name. Only the two
    /// source-file modes are supported; aliases and replaced bindings fail.
    pub fn seal_regular(&self, name: &OsStr, file: &File, executable: bool) -> io::Result<()> {
        self.verify_private_parent()?;
        self.validate_regular_entry(name, file)?;
        file.sync_all()?;
        file.set_permissions(std::fs::Permissions::from_mode(if executable {
            0o500
        } else {
            0o400
        }))?;
        file.sync_all()?;
        self.validate_regular_entry(name, file)?;
        self.verify_private_parent()
    }

    /// Change only this fixed directory's owner mode. Unsealing is for an
    /// operation's own unpublished staging tree, never an existing source.
    pub fn source_mode(&self, sealed: bool) -> io::Result<()> {
        self.verify_private_parent()?;
        self.inner
            .file
            .set_permissions(std::fs::Permissions::from_mode(if sealed {
                0o500
            } else {
                0o700
            }))?;
        self.inner.file.sync_all()?;
        self.verify_private_parent()
    }

    pub fn mode(&self) -> io::Result<u32> {
        self.verify_attached()?;
        let mode = self.inner.file.metadata()?.permissions().mode();
        self.verify_attached()?;
        Ok(mode)
    }

    /// Atomically publish this retained child under another retained parent.
    /// Existing destinations are never replaced. Any error after rename can
    /// mean publication happened; callers must not delete either destination.
    pub fn publish_directory(&self, target: &Directory, name: &OsStr) -> io::Result<Directory> {
        leaf(name)?;
        self.verify_private_parent()?;
        let parent = self
            .inner
            .parent
            .as_ref()
            .ok_or_else(|| invalid("cannot publish filesystem root"))?;
        parent.verify_private_parent()?;
        target.verify_private_parent()?;
        rename_between_new(
            &parent.inner.file,
            &self.inner.name,
            &target.inner.file,
            name,
        )?;
        rfs::fsync(&parent.inner.file)?;
        rfs::fsync(&target.inner.file)?;
        let published = target.child(name, false)?;
        let old = rfs::fstat(&self.inner.file)?;
        if !same_file(&published.inner.file, &old)? {
            return Err(invalid("published directory binding changed"));
        }
        published.verify_private_parent()?;
        Ok(published)
    }

    /// Remove a known regular leaf only inside an operation-owned unpublished
    /// staging tree, under the caller's domain lock. No symlink is followed.
    /// This cannot prevent an administrator/same-UID adversary replacing a leaf
    /// between checks; it is not an arbitrary-tree deletion primitive.
    pub fn remove_staged_regular(&self, name: &OsStr, file: &File) -> io::Result<()> {
        self.verify_private_parent()?;
        self.validate_regular_entry(name, file)?;
        rfs::unlinkat(&self.inner.file, name, AtFlags::empty())?;
        rfs::fsync(&self.inner.file)?;
        self.verify_private_parent()
    }

    /// Remove one retained empty child of this exact parent. The caller owns
    /// the unpublished staging namespace; populated/published trees are refused.
    pub fn remove_staged_empty_child(&self, child: &Directory) -> io::Result<()> {
        self.verify_private_parent()?;
        child.verify_private_parent()?;
        let parent = child
            .inner
            .parent
            .as_ref()
            .ok_or_else(|| invalid("child parent is unavailable"))?;
        if !same_file(&parent.inner.file, &rfs::fstat(&self.inner.file)?)? {
            return Err(invalid("staging parent binding differs"));
        }
        if !child.entry_names_limited(1)?.is_empty() {
            return Err(invalid("staging directory is not empty"));
        }
        rfs::unlinkat(&self.inner.file, &child.inner.name, AtFlags::REMOVEDIR)?;
        rfs::fsync(&self.inner.file)?;
        self.verify_private_parent()
    }

    pub fn capacity(&self) -> io::Result<Capacity> {
        self.verify_attached()?;
        let stat = rfs::fstatvfs(&self.inner.file)?;
        let available_bytes = stat
            .f_bavail
            .checked_mul(stat.f_frsize)
            .ok_or_else(|| invalid("filesystem capacity overflow"))?;
        let result = Capacity {
            available_bytes,
            available_inodes: stat.f_favail,
        };
        self.verify_attached()?;
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
            || entry.st_nlink != 1
            || entry.st_mode & 0o022 != 0
            || !trusted_storage_owner(entry.st_uid, effective_uid())
        {
            return Err(invalid("regular entry binding or ownership became unsafe"));
        }
        Ok(())
    }

    /// Validate a regular file that is intended to be immutable metadata.
    /// The owner-write bit is rejected as well as group/other writes; callers
    /// should publish with `seal_regular` and re-check this binding on reads.
    pub fn validate_immutable_regular_entry(&self, name: &OsStr, file: &File) -> io::Result<()> {
        self.validate_regular_entry(name, file)?;
        if file.metadata()?.permissions().mode() & 0o222 != 0 {
            return Err(invalid("immutable metadata is writable"));
        }
        Ok(())
    }

    /// A stable, non-truncating lock leaf. The caller must also validate its
    /// binding after acquiring its advisory lock and before guarded effects.
    pub fn open_lock(&self, name: &OsStr) -> io::Result<File> {
        leaf(name)?;
        self.verify_private_parent()?;
        if self
            .regular_identity(name)?
            .is_some_and(|identity| !identity.is_private_for(effective_uid()))
        {
            return Err(invalid("private metadata lock required"));
        }
        let file = File::from(rfs::openat(
            &self.inner.file,
            name,
            OFlags::RDWR | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::RUSR | Mode::WUSR,
        )?);
        self.validate_regular_entry(name, &file)?;
        if !self
            .regular_identity(name)?
            .is_some_and(|identity| identity.is_private_for(effective_uid()))
        {
            return Err(invalid("private metadata lock required"));
        }
        rfs::fsync(&self.inner.file)?;
        Ok(file)
    }

    /// Open a pre-existing lock for shared observation, without creating a leaf,
    /// directory, temporary or journal. Missing or invalid entries are errors.
    pub fn open_existing_lock(&self, name: &OsStr) -> io::Result<File> {
        self.existing_lock(name, OFlags::RDONLY)
    }

    /// An existing mutable lock/ledger. Never creates or replaces the inode;
    /// callers serialize, revalidate and fsync their finite state writes.
    pub fn open_existing_lock_rw(&self, name: &OsStr) -> io::Result<File> {
        self.existing_lock(name, OFlags::RDWR)
    }

    fn existing_lock(&self, name: &OsStr, access: OFlags) -> io::Result<File> {
        leaf(name)?;
        self.verify_private_parent()?;
        if self
            .regular_identity(name)?
            .is_some_and(|identity| !identity.is_private_for(effective_uid()))
        {
            return Err(invalid("private metadata lock required"));
        }
        let file = File::from(rfs::openat(
            &self.inner.file,
            name,
            access | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )?);
        self.validate_regular_entry(name, &file)?;
        if !self
            .regular_identity(name)?
            .is_some_and(|identity| identity.is_private_for(effective_uid()))
        {
            return Err(invalid("private metadata lock required"));
        }
        Ok(file)
    }

    /// Append bounded metadata to an existing private file. The caller must
    /// hold its domain lock; this is not a compare-and-set operation. An error
    /// after write starts can mean a partial/durable effect and needs recovery.
    pub fn append_regular(&self, name: &OsStr, bytes: &[u8], maximum: u64) -> io::Result<()> {
        leaf(name)?;
        self.verify_private_parent()?;
        if !self
            .regular_identity(name)?
            .is_some_and(|id| id.is_private_for(effective_uid()))
        {
            return Err(invalid("private existing append file required"));
        }
        let mut file = File::from(rfs::openat(
            &self.inner.file,
            name,
            OFlags::WRONLY | OFlags::APPEND | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )?);
        self.validate_regular_entry(name, &file)?;
        if file
            .metadata()?
            .len()
            .checked_add(bytes.len() as u64)
            .is_none_or(|size| size > maximum)
        {
            return Err(invalid("append file exceeds its budget"));
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        self.validate_regular_entry(name, &file)?;
        self.verify_private_parent()
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

    /// Open a regular metadata/log file through this directory capability.
    /// Callers performing bounded tail reads must validate this exact entry
    /// again after I/O and retain a fixed initial length. Never use an extra
    /// SQLite database/sidecar descriptor to verify SQLite's own POSIX locks.
    pub fn open_regular_read(&self, name: &OsStr) -> io::Result<File> {
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

    /// Replace mutable metadata through this directory FD. Never follows an
    /// existing leaf; unsupported leaf types fail instead of being repaired.
    /// The caller holds its metadata lock and treats a post-publication error
    /// as an uncertain effect, rather than assuming the replacement rolled back.
    pub fn replace_regular(&self, name: &OsStr, bytes: &[u8]) -> io::Result<()> {
        leaf(name)?;
        self.regular_identity(name)?;
        let (temporary, mut file) = self.temporary()?;
        let result = (|| {
            file.write_all(bytes)?;
            file.sync_all()?;
            self.validate_regular_entry(&temporary, &file)?;
            self.regular_identity(name)?;
            rfs::renameat(&self.inner.file, &temporary, &self.inner.file, name)?;
            rfs::fsync(&self.inner.file)?;
            self.validate_regular_entry(name, &file)?;
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
    rename_between_new(directory, from, directory, to)
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

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
))]
fn rename_between_new(
    from_directory: &File,
    from: &OsStr,
    to_directory: &File,
    to: &OsStr,
) -> io::Result<()> {
    rfs::renameat_with(
        from_directory,
        from,
        to_directory,
        to,
        rfs::RenameFlags::NOREPLACE,
    )
    .map_err(Into::into)
}
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
)))]
fn rename_between_new(
    _from_directory: &File,
    _from: &OsStr,
    _to_directory: &File,
    _to: &OsStr,
) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace directory publication is unavailable",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{PermissionsExt, symlink},
    };

    #[test]
    fn bounded_names_and_read_handle_do_not_create_or_follow_entries() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        directory
            .publish_new(OsStr::new("a.log"), b"one\ntwo\n")
            .unwrap();
        directory
            .publish_new(OsStr::new("b.log"), b"three\n")
            .unwrap();
        assert!(directory.entry_names_limited(1).is_err());
        assert_eq!(directory.entry_names_limited(2).unwrap().len(), 2);
        let mut file = directory.open_regular_read(OsStr::new("a.log")).unwrap();
        let mut content = String::new();
        file.read_to_string(&mut content).unwrap();
        assert_eq!(content, "one\ntwo\n");
        assert!(directory.open_regular_read(OsStr::new("missing")).is_err());
        assert!(!temp.path().join("missing").exists());
        symlink("a.log", temp.path().join("alias")).unwrap();
        assert!(directory.open_regular_read(OsStr::new("alias")).is_err());
        fs::rename(temp.path().join("a.log"), temp.path().join("old.log")).unwrap();
        directory
            .publish_new(OsStr::new("a.log"), b"different")
            .unwrap();
        assert!(
            directory
                .validate_regular_entry(OsStr::new("a.log"), &file)
                .is_err()
        );
    }

    #[test]
    fn explicit_new_child_never_accepts_an_existing_empty_namespace() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let child = directory.create_new_child(OsStr::new("fresh")).unwrap();
        assert!(child.entry_names().unwrap().is_empty());
        let error = directory.create_new_child(OsStr::new("fresh")).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert!(child.entry_names().unwrap().is_empty());
        assert_eq!(
            fs::metadata(temp.path().join("fresh"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
    }

    #[test]
    fn ownership_policy_accepts_only_service_or_root_and_private_single_links() {
        assert!(trusted_storage_owner(0, 995));
        assert!(trusted_storage_owner(995, 995));
        assert!(!trusted_storage_owner(1000, 995));
        let identity = RegularIdentity {
            device: 1,
            inode: 2,
            links: 1,
            mode: 0o100640,
            owner: 995,
        };
        assert!(identity.is_private_for(995));
        assert!(
            !RegularIdentity {
                owner: 1000,
                ..identity
            }
            .is_private_for(995)
        );
        assert!(
            !RegularIdentity {
                links: 2,
                ..identity
            }
            .is_private_for(995)
        );
        assert!(
            !RegularIdentity {
                mode: 0o100660,
                ..identity
            }
            .is_private_for(995)
        );
    }

    #[test]
    fn stat_enumeration_and_mutable_metadata_use_the_retained_directory() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("storage");
        let directory = Directory::open(&path, true).unwrap();
        directory
            .publish_new(OsStr::new("record.json"), b"old")
            .unwrap();
        let identity = directory
            .regular_identity(OsStr::new("record.json"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.links, 1);
        assert_eq!(identity.mode & 0o777, 0o600);
        let names = vec![OsString::from("record.json")];
        assert_eq!(directory.entry_names().unwrap(), names);
        assert_eq!(directory.entry_names().unwrap(), names);
        directory
            .replace_regular(OsStr::new("record.json"), b"new")
            .unwrap();
        assert_eq!(
            directory
                .read_regular(OsStr::new("record.json"), 10)
                .unwrap(),
            b"new"
        );
        assert_ne!(
            directory
                .regular_identity(OsStr::new("record.json"))
                .unwrap()
                .unwrap()
                .inode,
            identity.inode
        );
        symlink("record.json", path.join("link")).unwrap();
        assert!(directory.regular_identity(OsStr::new("link")).is_err());
        assert!(
            directory
                .replace_regular(OsStr::new("link"), b"bad")
                .is_err()
        );
        fs::rename(&path, temp.path().join("detached")).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(
            directory
                .replace_regular(OsStr::new("record.json"), b"bad")
                .is_err()
        );
        assert_eq!(fs::read_dir(&path).unwrap().count(), 0);
        assert!(directory.entry_names().is_err());
        assert!(
            directory
                .regular_identity(OsStr::new("record.json"))
                .is_err()
        );
    }

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

    #[test]
    fn mutable_existing_lock_and_append_preserve_inode_and_bound_writes() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let name = OsStr::new("ledger");
        assert!(directory.open_existing_lock_rw(name).is_err());
        assert!(directory.append_regular(name, b"x", 10).is_err());
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
        directory.publish_new(name, b"old").unwrap();
        let before = directory.regular_identity(name).unwrap();
        let mut lock = directory.open_existing_lock_rw(name).unwrap();
        lock.write_all(b"new").unwrap();
        lock.sync_all().unwrap();
        directory.append_regular(name, b"\n", 4).unwrap();
        assert_eq!(fs::read(temp.path().join(name)).unwrap(), b"new\n");
        assert_eq!(directory.regular_identity(name).unwrap(), before);
        assert!(directory.append_regular(name, b"x", 4).is_err());
        assert_eq!(fs::read(temp.path().join(name)).unwrap(), b"new\n");
        symlink("ledger", temp.path().join("symbolic")).unwrap();
        for alias in ["symbolic", "missing"] {
            assert!(directory.open_existing_lock_rw(OsStr::new(alias)).is_err());
            assert!(
                directory
                    .append_regular(OsStr::new(alias), b"x", 10)
                    .is_err()
            );
        }
        fs::hard_link(temp.path().join(name), temp.path().join("hard")).unwrap();
        assert!(directory.open_existing_lock_rw(name).is_err());
        assert!(directory.append_regular(name, b"x", 10).is_err());
        assert_eq!(fs::read(temp.path().join(name)).unwrap(), b"new\n");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(temp.path().join("fifo"))
                .status()
                .unwrap()
                .success()
        );
        assert!(directory.open_existing_lock_rw(OsStr::new("fifo")).is_err());
        assert!(
            directory
                .append_regular(OsStr::new("fifo"), b"x", 10)
                .is_err()
        );
    }
}
