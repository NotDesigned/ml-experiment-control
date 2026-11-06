//! Unix directory capabilities. All leaf I/O is relative to retained directory
//! descriptors; path-based checks are never followed by path-based writes.
//!
//! Newly created directory entries and files are fsynced before success. Regular
//! files can use atomic hard-link publication on filesystems without no-replace
//! rename. Directory moves require native no-replace rename.
#![forbid(unsafe_code)]

use rustix::fs::{self as rfs, AtFlags, FileType, Mode, OFlags};
use std::{
    collections::BTreeSet,
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

use crate::file_lock::acquire;
pub use crate::file_lock::{FileLock, LockOptions, Wait};

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
    pub mode: u32,
    pub owner: u32,
}

/// What a directory entry is, read without following a link or opening it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryType {
    Directory,
    Regular,
    SymbolicLink,
    /// A FIFO, socket or device.
    Special,
}

/// Advisory capacity of the filesystem containing this retained directory.
/// Other processes can consume it; a successful check is never a reservation.
#[derive(Clone, Copy, Debug)]
pub struct Capacity {
    pub available_bytes: u64,
    /// None when the filesystem does not report an inode pool (e.g. CephFS).
    /// This is unknown capacity, never an invented zero or reservation.
    pub available_inodes: Option<u64>,
}
impl Capacity {
    fn reported(available_bytes: u64, total_inodes: u64, available_inodes: u64) -> Self {
        Self {
            available_bytes,
            available_inodes: (total_inodes != 0).then_some(available_inodes),
        }
    }
    /// Reject a known shortage. Unsupported inode counts cannot establish a
    /// shortage or guarantee a write; ordinary filesystem errors still apply.
    pub fn insufficient(self, bytes: u64, inodes: u64) -> bool {
        self.available_bytes < bytes || self.available_inodes.is_some_and(|n| n < inodes)
    }
}

pub fn effective_uid() -> u32 {
    rustix::process::geteuid().as_raw()
}

/// Root is the trusted administrator; arbitrary other file owners are not.
pub fn trusted_storage_owner(owner: u32, effective: u32) -> bool {
    owner == 0 || owner == effective
}

impl RegularIdentity {
    /// Not writable by another group or user. Link counts are not compared: a
    /// CPFS client can briefly report a stale value.
    pub fn is_private_for(self, effective: u32) -> bool {
        self.mode & 0o022 == 0 && trusted_storage_owner(self.owner, effective)
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
/// Inode only. A shared filesystem does not promise a stable st_dev, and
/// content integrity comes from the hashes verified at publication.
fn same_file(file: &File, stat: &rfs::Stat) -> io::Result<bool> {
    let metadata = rfs::fstat(file)?;
    Ok(metadata.st_ino == stat.st_ino)
}
/// O_RDONLY|O_NONBLOCK opens a FIFO, which would then read as empty, and opens
/// a directory too. Every opener of an existing leaf therefore checks the type.
fn regular_file(file: File) -> io::Result<File> {
    if FileType::from_raw_mode(rfs::fstat(&file)?.st_mode) != FileType::RegularFile {
        return Err(invalid("regular file required"));
    }
    Ok(file)
}
fn directory_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK
}

/// Seal a retained regular file. Only the two source-file modes are supported.
pub fn seal_regular(file: &File, executable: bool) -> io::Result<()> {
    file.sync_all()?;
    file.set_permissions(std::fs::Permissions::from_mode(if executable {
        0o500
    } else {
        0o400
    }))?;
    file.sync_all()
}

/// Sealed content must still be read-only. This catches an accidental chmod or
/// an unsealed copy; the hashes verify the content itself.
pub fn require_sealed(file: &File) -> io::Result<()> {
    if file.metadata()?.permissions().mode() & 0o222 != 0 {
        return Err(invalid("sealed file is writable"));
    }
    Ok(())
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
    /// any symlink.
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
        Ok(directory)
    }

    /// Whether this directory still sits at the name it was created under. A
    /// rename is what publishes a staged tree, so `false` means it may already
    /// be published and cleanup must leave it alone. This separates a refused
    /// rename from an fsync failure after a rename that happened, which an
    /// error alone cannot. One level only; it says nothing about ancestors.
    pub fn still_at_staged_name(&self) -> io::Result<bool> {
        let Some(parent) = &self.inner.parent else {
            return Ok(false);
        };
        match rfs::statat(
            &parent.inner.file,
            &self.inner.name,
            AtFlags::SYMLINK_NOFOLLOW,
        ) {
            Ok(entry) => Ok(
                FileType::from_raw_mode(entry.st_mode) == FileType::Directory
                    && same_file(&self.inner.file, &entry)?,
            ),
            Err(rustix::io::Errno::NOENT) => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    /// A directory holding credentials or bootstrap state must not be writable
    /// by another group or user. Mode and owner of this one directory, with no
    /// walk of its ancestors. It cannot restrain root or the same account.
    pub fn require_private(&self) -> io::Result<()> {
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
        let stat = rfs::fstat(&self.inner.file)?;
        Ok((stat.st_dev as u64, stat.st_ino as u64))
    }

    /// Inspect a leaf through the directory FD without opening/closing it.
    /// Absence is distinct from a symlink, FIFO, device or directory.
    #[allow(clippy::unnecessary_cast)] // Stat field widths differ across Unix.
    pub fn regular_identity(&self, name: &OsStr) -> io::Result<Option<RegularIdentity>> {
        leaf(name)?;
        match rfs::statat(&self.inner.file, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) if FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile => {
                Ok(Some(RegularIdentity {
                    device: stat.st_dev as u64,
                    inode: stat.st_ino as u64,
                    mode: stat.st_mode as u32,
                    owner: stat.st_uid,
                }))
            }
            Ok(_) => Err(invalid("regular storage entry required")),
            Err(rustix::io::Errno::NOENT) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Enumerate names from a fresh directory stream tied to this descriptor.
    /// Callers still validate/open each required child using directory methods.
    pub fn entry_names(&self) -> io::Result<Vec<OsString>> {
        let mut names = Vec::new();
        for entry in rfs::Dir::read_from(&self.inner.file)? {
            let entry = entry?;
            let name = entry.file_name().to_bytes();
            if !matches!(name, b"." | b"..") {
                names.push(OsString::from_vec(name.to_vec()));
            }
        }
        names.sort();
        Ok(names)
    }

    /// Read at most `maximum` directory entries, failing before collecting an
    /// unbounded name list. This neither creates nor skips invalid entries.
    pub fn entry_names_limited(&self, maximum: usize) -> io::Result<Vec<OsString>> {
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
        Ok(names)
    }

    /// The first `maximum` names in sort order and how many more there are. Memory
    /// stays bounded however large the directory is, and the same names come back
    /// whatever order the filesystem lists them in.
    pub fn first_entry_names(&self, maximum: usize) -> io::Result<(Vec<OsString>, u64)> {
        let mut names = BTreeSet::new();
        let mut more = 0u64;
        for entry in rfs::Dir::read_from(&self.inner.file)? {
            let entry = entry?;
            let name = entry.file_name().to_bytes();
            if !matches!(name, b"." | b"..") {
                names.insert(OsString::from_vec(name.to_vec()));
                if names.len() > maximum {
                    names.pop_last();
                    more += 1;
                }
            }
        }
        Ok((names.into_iter().collect(), more))
    }

    /// None when the entry is absent. Every other failure stays an error, so a read
    /// that fails cannot be mistaken for a type or for a missing file.
    pub fn entry_type(&self, name: &OsStr) -> io::Result<Option<EntryType>> {
        leaf(name)?;
        match rfs::statat(&self.inner.file, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => Ok(Some(match FileType::from_raw_mode(stat.st_mode) {
                FileType::Directory => EntryType::Directory,
                FileType::RegularFile => EntryType::Regular,
                FileType::Symlink => EntryType::SymbolicLink,
                _ => EntryType::Special,
            })),
            Err(rustix::io::Errno::NOENT) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// A created directory is private from its first instant, and its parent
    /// directory entry is synced before returning its fixed child descriptor.
    pub fn child(&self, name: &OsStr, create: bool) -> io::Result<Self> {
        leaf(name)?;
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
        let file = File::from(rfs::openat(
            &self.inner.file,
            name,
            directory_flags(),
            Mode::empty(),
        )?);
        if created {
            rfs::fsync(&file)?;
        }
        Ok(Self {
            inner: Arc::new(Inner {
                file,
                parent: Some(self.clone()),
                name: name.to_owned(),
            }),
        })
    }

    /// Explicit fresh namespace allocation. Unlike `child(..., true)`, an
    /// existing entry always fails, including an empty directory that appeared
    /// concurrently. Intended for reviewed operator bootstrap, not observation.
    pub fn create_new_child(&self, name: &OsStr) -> io::Result<Self> {
        leaf(name)?;
        rfs::mkdirat(&self.inner.file, name, Mode::RWXU)?;
        rfs::fsync(&self.inner.file)?;
        let directory = self.child(name, false)?;
        rfs::fsync(&directory.inner.file)?;
        Ok(directory)
    }

    /// Stream only into a newly allocated private regular file. This does not
    /// publish immutable content; the caller owns the containing staging tree,
    /// bounds writes, fsyncs and seals before any directory publication.
    pub fn create_new_regular(&self, name: &OsStr) -> io::Result<File> {
        leaf(name)?;
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
        rfs::fsync(&self.inner.file)?;
        Ok(file)
    }

    /// Change only this fixed directory's owner mode. Unsealing is for an
    /// operation's own unpublished staging tree, never an existing source.
    pub fn source_mode(&self, sealed: bool) -> io::Result<()> {
        self.inner
            .file
            .set_permissions(std::fs::Permissions::from_mode(if sealed {
                0o500
            } else {
                0o700
            }))?;
        self.inner.file.sync_all()
    }

    pub fn mode(&self) -> io::Result<u32> {
        Ok(self.inner.file.metadata()?.permissions().mode())
    }

    /// Atomically publish this retained child under another retained parent.
    /// Existing destinations are never replaced. Any error after rename can
    /// mean publication happened; callers must not delete either destination.
    pub fn publish_directory(&self, target: &Directory, name: &OsStr) -> io::Result<Directory> {
        leaf(name)?;
        let parent = self
            .inner
            .parent
            .as_ref()
            .ok_or_else(|| invalid("cannot publish filesystem root"))?;
        rename_between_new(
            &parent.inner.file,
            &self.inner.name,
            &target.inner.file,
            name,
        )?;
        rfs::fsync(&parent.inner.file)?;
        rfs::fsync(&target.inner.file)?;
        target.child(name, false)
    }

    /// Remove a known regular leaf only inside an operation-owned unpublished
    /// staging tree, under the caller's domain lock. No symlink is followed and
    /// an unexpected type stops the cleanup. This is not an arbitrary-tree
    /// deletion primitive.
    pub fn remove_staged_regular(&self, name: &OsStr) -> io::Result<()> {
        self.regular_identity(name)?;
        rfs::unlinkat(&self.inner.file, name, AtFlags::empty())?;
        rfs::fsync(&self.inner.file)?;
        Ok(())
    }

    /// Remove any entry but a directory: a link goes and its target stays. Only for
    /// a reviewed removal of a whole tree, under the caller's domain lock.
    pub fn remove_non_directory(&self, name: &OsStr) -> io::Result<()> {
        leaf(name)?;
        rfs::unlinkat(&self.inner.file, name, AtFlags::empty())?;
        rfs::fsync(&self.inner.file)?;
        Ok(())
    }

    /// No-replace publication of an operation-owned completed regular file.
    pub fn publish_regular_as(&self, source: &OsStr, target: &OsStr) -> io::Result<()> {
        leaf(source)?;
        leaf(target)?;
        rename_regular_new(&self.inner.file, source, target)?;
        rfs::fsync(&self.inner.file)?;
        Ok(())
    }

    /// Remove one retained empty child of this exact parent. The caller owns
    /// the unpublished staging namespace; populated/published trees are refused.
    pub fn remove_staged_empty_child(&self, child: &Directory) -> io::Result<()> {
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
        Ok(())
    }

    pub fn capacity(&self) -> io::Result<Capacity> {
        let stat = rfs::fstatvfs(&self.inner.file)?;
        let available_bytes = stat
            .f_bavail
            .checked_mul(stat.f_frsize)
            .ok_or_else(|| invalid("filesystem capacity overflow"))?;
        Ok(Capacity::reported(
            available_bytes,
            stat.f_files,
            stat.f_favail,
        ))
    }

    /// Any entry counts, including a directory, dangling symlink or FIFO. This
    /// is deliberately different from opening/validating a regular payload.
    pub fn entry_exists(&self, name: &OsStr) -> io::Result<bool> {
        leaf(name)?;
        match rfs::statat(&self.inner.file, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(_) => Ok(true),
            Err(rustix::io::Errno::NOENT) => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    /// Take an advisory lock on `name` through a fresh description. Contention
    /// is returned as `WouldBlock`, which the caller maps to its own busy error.
    pub fn lock(&self, name: &OsStr, options: LockOptions<'_>) -> io::Result<FileLock> {
        let file = self.lock_leaf(name, options.create, options.write)?;
        acquire(file, options.shared, options.wait)
    }

    /// Bootstrap a lock leaf without taking the lock, so that it cannot lose to
    /// whoever holds the lock at that moment.
    pub fn create_lock_file(&self, name: &OsStr) -> io::Result<()> {
        self.lock_leaf(name, true, true)?;
        Ok(())
    }

    fn lock_leaf(&self, name: &OsStr, create: bool, write: bool) -> io::Result<File> {
        leaf(name)?;
        let access = if create || write {
            OFlags::RDWR
        } else {
            OFlags::RDONLY
        };
        let (creation, mode) = if create {
            (OFlags::CREATE, Mode::RUSR | Mode::WUSR)
        } else {
            (OFlags::empty(), Mode::empty())
        };
        // NONBLOCK keeps a FIFO left at this name from hanging the open.
        let file = File::from(rfs::openat(
            &self.inner.file,
            name,
            access | creation | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            mode,
        )?);
        if FileType::from_raw_mode(rfs::fstat(&file)?.st_mode) != FileType::RegularFile {
            return Err(invalid("regular lock file required"));
        }
        if create {
            rfs::fsync(&self.inner.file)?;
        }
        Ok(file)
    }

    /// Append bounded metadata to an existing private file. The caller must
    /// hold its domain lock; this is not a compare-and-set operation. An error
    /// after write starts can mean a partial/durable effect and needs recovery.
    pub fn append_regular(&self, name: &OsStr, bytes: &[u8]) -> io::Result<()> {
        leaf(name)?;
        if !self
            .regular_identity(name)?
            .is_some_and(|id| id.is_private_for(effective_uid()))
        {
            return Err(invalid("private existing append file required"));
        }
        let mut file = regular_file(File::from(rfs::openat(
            &self.inner.file,
            name,
            OFlags::WRONLY | OFlags::APPEND | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )?))?;
        file.write_all(bytes)?;
        file.sync_all()
    }

    /// Read a whole regular file of the service's own state. Anything a job,
    /// a peer or a credential owner can write keeps a `read_regular` limit.
    pub fn read_whole(&self, name: &OsStr) -> io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.open_regular_read(name)?.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    pub fn read_regular(&self, name: &OsStr, maximum: u64) -> io::Result<Vec<u8>> {
        let bound = maximum
            .checked_add(1)
            .ok_or_else(|| invalid("invalid read bound"))?;
        leaf(name)?;
        let file = regular_file(File::from(rfs::openat(
            &self.inner.file,
            name,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )?))?;
        if file.metadata()?.len() > maximum {
            return Err(invalid("regular file exceeds limit"));
        }
        let mut bytes = Vec::new();
        (&file).take(bound).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > maximum {
            return Err(invalid("regular file exceeds limit"));
        }
        Ok(bytes)
    }

    /// Open a regular metadata/log file through this directory capability.
    /// Callers performing bounded tail reads retain a fixed initial length.
    /// Never use an extra SQLite database/sidecar descriptor to verify
    /// SQLite's own POSIX locks.
    pub fn open_regular_read(&self, name: &OsStr) -> io::Result<File> {
        leaf(name)?;
        let file = regular_file(File::from(rfs::openat(
            &self.inner.file,
            name,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )?))?;
        Ok(file)
    }

    /// Publish immutable bytes with a same-directory unpredictable temporary,
    /// fsync, atomic no-replace publication and directory fsync.
    /// There is no check-then-replace fallback and no path traversal during I/O.
    pub fn publish_new(&self, name: &OsStr, bytes: &[u8]) -> io::Result<()> {
        leaf(name)?;
        let (temporary, mut file) = self.temporary()?;
        let result = (|| {
            file.write_all(bytes)?;
            file.sync_all()?;
            rename_regular_new(&self.inner.file, &temporary, name)?;
            rfs::fsync(&self.inner.file)?;
            Ok(())
        })();
        if result.is_err() {
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
            rfs::renameat(&self.inner.file, &temporary, &self.inner.file, name)?;
            rfs::fsync(&self.inner.file)?;
            Ok(())
        })();
        if result.is_err() {
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

/// linkat is an atomic no-clobber operation, including on CPFS. A failed
/// unlink leaves ambiguous publication, never permission to overwrite a target.
fn link_regular_new(directory: &File, from: &OsStr, to: &OsStr) -> io::Result<()> {
    let stat = rfs::statat(directory, from, AtFlags::SYMLINK_NOFOLLOW)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(invalid("regular publication source required"));
    }
    rfs::linkat(directory, from, directory, to, AtFlags::empty())?;
    rfs::fsync(directory)?;
    rfs::unlinkat(directory, from, AtFlags::empty())?;
    Ok(())
}

fn rename_regular_new(directory: &File, from: &OsStr, to: &OsStr) -> io::Result<()> {
    match rename_new(directory, from, to) {
        Err(error)
            if matches!(error.raw_os_error(), Some(22 | 38 | 95))
                || error.kind() == io::ErrorKind::Unsupported =>
        {
            link_regular_new(directory, from, to)
        }
        result => result,
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
    #[test]
    fn unsupported_inode_pool_differs_from_known_exhaustion_and_keeps_byte_guard() {
        // The real shared CephFS reports f_files=f_favail=0 with ample bytes.
        let unknown = Capacity::reported(32, 0, 0);
        assert_eq!(unknown.available_inodes, None);
        assert!(!unknown.insufficient(16, 1024));
        assert!(unknown.insufficient(33, 1));
        // A filesystem that reports an inode pool can genuinely be exhausted.
        let exhausted = Capacity::reported(32, 100, 0);
        assert_eq!(exhausted.available_inodes, Some(0));
        assert!(exhausted.insufficient(16, 1));
        let known = Capacity::reported(32, 100, 2);
        assert!(!known.insufficient(32, 2));
        assert!(known.insufficient(32, 3));
        assert!(known.insufficient(33, 1));
    }
    use std::{
        fs,
        os::unix::fs::{PermissionsExt, symlink},
    };

    #[test]
    fn hard_link_publication_is_no_clobber_and_removes_the_source_name() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let mut first = directory.create_new_regular(OsStr::new("first")).unwrap();
        first.write_all(b"original").unwrap();
        first.sync_all().unwrap();
        link_regular_new(
            &directory.inner.file,
            OsStr::new("first"),
            OsStr::new("target"),
        )
        .unwrap();
        assert!(!directory.entry_exists(OsStr::new("first")).unwrap());
        let mut second = directory.create_new_regular(OsStr::new("second")).unwrap();
        second.write_all(b"different").unwrap();
        assert_eq!(
            link_regular_new(
                &directory.inner.file,
                OsStr::new("second"),
                OsStr::new("target")
            )
            .unwrap_err()
            .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(
            directory.read_regular(OsStr::new("target"), 100).unwrap(),
            b"original"
        );
        assert!(directory.entry_exists(OsStr::new("second")).unwrap());
        symlink("target", temp.path().join("alias")).unwrap();
        assert!(
            link_regular_new(
                &directory.inner.file,
                OsStr::new("alias"),
                OsStr::new("unsafe")
            )
            .is_err()
        );
        assert!(!directory.entry_exists(OsStr::new("unsafe")).unwrap());
    }

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
    }

    #[test]
    fn failed_publication_leaves_no_temporary_and_keeps_the_existing_name() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        directory.publish_new(OsStr::new("once"), b"first").unwrap();
        assert!(
            directory
                .publish_new(OsStr::new("once"), b"second")
                .is_err()
        );
        assert!(
            directory
                .replace_regular(OsStr::new("once"), b"third")
                .is_ok()
        );
        fs::create_dir(temp.path().join("folder")).unwrap();
        assert!(
            directory
                .replace_regular(OsStr::new("folder"), b"bad")
                .is_err()
        );
        let names = directory.entry_names().unwrap();
        assert_eq!(names, ["folder", "once"].map(OsString::from));
        assert_eq!(fs::read(temp.path().join("once")).unwrap(), b"third");
    }

    #[test]
    fn sealing_is_read_only_and_a_writable_file_is_not_sealed() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let file = directory.create_new_regular(OsStr::new("data")).unwrap();
        assert!(require_sealed(&file).is_err());
        seal_regular(&file, false).unwrap();
        require_sealed(&file).unwrap();
        file.set_permissions(fs::Permissions::from_mode(0o600))
            .unwrap();
        assert!(require_sealed(&file).is_err());
        seal_regular(&file, true).unwrap();
        assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o500);
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
    fn ownership_policy_accepts_only_service_or_root_and_refuses_writable_files() {
        assert!(trusted_storage_owner(0, 995));
        assert!(trusted_storage_owner(995, 995));
        assert!(!trusted_storage_owner(1000, 995));
        let identity = RegularIdentity {
            device: 1,
            inode: 2,
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
    fn a_published_stage_is_no_longer_at_its_staged_name() {
        let temp = tempfile::tempdir().unwrap();
        let slots = Directory::open(&temp.path().join("slots"), true).unwrap();
        let runs = Directory::open(&temp.path().join("runs"), true).unwrap();
        let stage = slots.create_new_child(OsStr::new("stage")).unwrap();
        assert!(stage.still_at_staged_name().unwrap());
        // A refused rename leaves the stage in place for its owner to clean up.
        runs.create_new_child(OsStr::new("taken")).unwrap();
        assert!(stage.publish_directory(&runs, OsStr::new("taken")).is_err());
        assert!(stage.still_at_staged_name().unwrap());
        // Whatever follows the rename, cleanup must not touch the published tree.
        stage
            .publish_directory(&runs, OsStr::new("published"))
            .unwrap();
        assert!(!stage.still_at_staged_name().unwrap());
        slots.create_new_child(OsStr::new("stage")).unwrap();
        assert!(!stage.still_at_staged_name().unwrap());
        assert!(
            !Directory::open(Path::new("/"), false)
                .unwrap()
                .still_at_staged_name()
                .unwrap()
        );
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
        let create = LockOptions {
            create: true,
            ..Default::default()
        };
        assert!(directory.lock(OsStr::new("fifo"), create).is_err());
        assert!(directory.create_lock_file(OsStr::new("fifo")).is_err());
        assert!(directory.read_regular(OsStr::new("fifo"), 10).is_err());
        assert!(directory.open_regular_read(OsStr::new("fifo")).is_err());
        fs::create_dir(temp.path().join("folder")).unwrap();
        assert!(directory.lock(OsStr::new("folder"), create).is_err());
        assert!(directory.read_regular(OsStr::new("folder"), 10).is_err());
        assert!(directory.open_regular_read(OsStr::new("folder")).is_err());
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
    fn locking_without_create_never_creates_or_repairs_a_leaf() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let options = LockOptions::default();
        assert_eq!(
            directory
                .lock(OsStr::new("missing"), options)
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
        fs::write(temp.path().join("lock"), b"original").unwrap();
        let file = directory.lock(OsStr::new("lock"), options).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 8);
        assert_eq!(fs::read(temp.path().join("lock")).unwrap(), b"original");
        symlink(temp.path().join("lock"), temp.path().join("alias")).unwrap();
        assert!(directory.lock(OsStr::new("alias"), options).is_err());
        assert!(
            std::process::Command::new("mkfifo")
                .arg(temp.path().join("fifo"))
                .status()
                .unwrap()
                .success()
        );
        assert!(directory.lock(OsStr::new("fifo"), options).is_err());
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 3);
    }

    #[test]
    fn mutable_lock_and_append_preserve_inode_and_bound_writes() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let name = OsStr::new("ledger");
        let options = LockOptions {
            write: true,
            ..Default::default()
        };
        assert!(directory.lock(name, options).is_err());
        assert!(directory.append_regular(name, b"x").is_err());
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
        directory.publish_new(name, b"old").unwrap();
        let before = directory.regular_identity(name).unwrap();
        let mut lock = directory.lock(name, options).unwrap();
        lock.write_all(b"new").unwrap();
        lock.sync_all().unwrap();
        directory.append_regular(name, b"\n").unwrap();
        assert_eq!(fs::read(temp.path().join(name)).unwrap(), b"new\n");
        assert_eq!(directory.regular_identity(name).unwrap(), before);
        assert_eq!(directory.read_whole(name).unwrap(), b"new\n");
        symlink("ledger", temp.path().join("symbolic")).unwrap();
        for alias in ["symbolic", "missing"] {
            assert!(directory.lock(OsStr::new(alias), options).is_err());
            assert!(directory.append_regular(OsStr::new(alias), b"x").is_err());
        }
        assert!(
            std::process::Command::new("mkfifo")
                .arg(temp.path().join("fifo"))
                .status()
                .unwrap()
                .success()
        );
        assert!(directory.lock(OsStr::new("fifo"), options).is_err());
        assert!(directory.append_regular(OsStr::new("fifo"), b"x").is_err());
        assert!(directory.read_whole(OsStr::new("fifo")).is_err());
    }

    #[test]
    fn two_locks_in_one_process_exclude_each_other() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let name = OsStr::new("lock");
        let exclusive = LockOptions {
            create: true,
            ..Default::default()
        };
        let shared = LockOptions {
            shared: true,
            ..Default::default()
        };
        let first = directory.lock(name, exclusive).unwrap();
        for options in [exclusive, shared] {
            assert_eq!(
                directory.lock(name, options).unwrap_err().kind(),
                io::ErrorKind::WouldBlock
            );
        }
        drop(first);
        let reader = directory.lock(name, shared).unwrap();
        let other = directory.lock(name, shared).unwrap();
        assert_eq!(
            directory.lock(name, exclusive).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        drop(reader);
        drop(other);
        assert!(directory.lock(name, exclusive).is_ok());
    }

    #[test]
    fn dropping_a_lock_releases_it_while_an_inherited_description_remains_open() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let name = OsStr::new("lock");
        let options = LockOptions {
            create: true,
            ..Default::default()
        };
        let lock = directory.lock(name, options).unwrap();
        // try_clone shares the kernel open file description, as fork does.
        let inherited = lock.try_clone().unwrap();
        drop(lock);
        assert!(directory.lock(name, options).is_ok());
        drop(inherited);
    }

    #[test]
    fn waits_end_at_the_deadline_or_on_cancellation_and_succeed_after_release() {
        use std::{
            sync::atomic::AtomicBool,
            time::{Duration, Instant},
        };
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let name = OsStr::new("lock");
        let first = directory
            .lock(
                name,
                LockOptions {
                    create: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let until = |deadline, cancelled| LockOptions {
            wait: Wait::Until {
                deadline,
                cancelled,
            },
            ..Default::default()
        };
        let start = Instant::now();
        let wait = Duration::from_millis(30);
        assert_eq!(
            directory
                .lock(name, until(start + wait, None))
                .unwrap_err()
                .kind(),
            io::ErrorKind::WouldBlock
        );
        assert!(start.elapsed() >= wait);
        let cancelled = AtomicBool::new(true);
        let far = Instant::now() + Duration::from_secs(60);
        assert_eq!(
            directory
                .lock(name, until(far, Some(&cancelled)))
                .unwrap_err()
                .kind(),
            io::ErrorKind::WouldBlock
        );
        assert!(start.elapsed() < Duration::from_secs(30));
        std::thread::scope(|scope| {
            let blocked = scope.spawn(|| {
                directory
                    .lock(
                        name,
                        LockOptions {
                            wait: Wait::Block,
                            ..Default::default()
                        },
                    )
                    .is_ok()
            });
            let polling = scope.spawn(|| {
                directory
                    .lock(name, until(Instant::now() + Duration::from_secs(30), None))
                    .is_ok()
            });
            std::thread::sleep(Duration::from_millis(20));
            drop(first);
            assert!(blocked.join().unwrap());
            assert!(polling.join().unwrap());
        });
    }
}
