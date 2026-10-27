//! Linux SQLite pathname boundary, shared by metadata repositories.
//!
//! JSON/lock owners use real directory-relative I/O. Native SQLite instead
//! resolves an ordinary pathname through its own VFS. Retained directory and
//! stat identities detect ordinary removal/replacement before and after work;
//! they do NOT prove which inode SQLite opened during a hostile ABA rename.
//! The directory must be private to the service's trusted account. A same-UID
//! adversary/root, an uncooperative writer, and arbitrary namespace mutation
//! are outside this boundary. No fake `/proc/self/fd` anchoring or custom VFS.
//!
//! Never open/close additional main/WAL/SHM file descriptors for checking:
//! POSIX close can discard another SQLite connection's process-level locks.
#![forbid(unsafe_code)]

use ml_exp_io::directory::{Directory, RegularIdentity, effective_uid};
use rusqlite::{Connection, OpenFlags, Transaction, TransactionBehavior, config::DbConfig};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::OsString,
    io,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::Duration,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("SQLite storage I/O failed")]
    Io(#[from] io::Error),
    #[error("SQLite operation failed")]
    Sqlite(#[from] rusqlite::Error),
    #[error("existing SQLite database is missing")]
    Missing,
    #[error("invalid SQLite storage: {0}")]
    Invalid(&'static str),
    #[error("SQLite pathname binding changed; operation unavailable")]
    Detached,
    #[error("SQLite storage may have changed; reconcile durable state before retry")]
    EffectUncertain,
}

impl Error {
    pub fn effects_may_have_applied(&self) -> bool {
        matches!(self, Self::EffectUncertain)
    }
}

#[derive(Debug, Default)]
struct Anchors {
    parent: OnceLock<Directory>,
    main: OnceLock<RegularIdentity>,
}

#[derive(Clone, Debug)]
pub struct SqliteDatabase {
    path: PathBuf,
    anchors: Arc<Anchors>,
    wal: bool,
}

struct Binding {
    parent: Directory,
    name: OsString,
    main: RegularIdentity,
    sidecars: RefCell<BTreeMap<OsString, Option<RegularIdentity>>>,
}

fn allowed(identity: RegularIdentity) -> Result<RegularIdentity, Error> {
    if !identity.is_private_for(effective_uid()) {
        return Err(Error::Invalid("private single-link regular file required"));
    }
    Ok(identity)
}

fn uri_path(path: &Path) -> String {
    use std::fmt::Write;
    let mut encoded = String::new();
    for byte in path.as_os_str().as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'-' | b'_' | b'~') {
            encoded.push(char::from(*byte));
        } else {
            write!(&mut encoded, "%{byte:02X}").expect("writing to String is infallible");
        }
    }
    encoded
}

impl Binding {
    fn validate(&self, effects: bool) -> Result<(), Error> {
        let checked = (|| {
            self.parent.verify_private_parent()?;
            if self.parent.regular_identity(&self.name)? != Some(self.main) {
                return Err(Error::Detached);
            }
            for (name, previous) in self.sidecars.borrow_mut().iter_mut() {
                let current = self
                    .parent
                    .regular_identity(name)?
                    .map(allowed)
                    .transpose()?;
                // DELETE-mode repositories legitimately remove this journal
                // at commit. Still reject symlinks, FIFOs and writable aliases.
                if name.as_bytes().ends_with(b"-journal") {
                    *previous = current;
                    continue;
                }
                match (*previous, current) {
                    (Some(old), Some(current)) if old == current => {}
                    (None, current) => *previous = current,
                    _ => return Err(Error::Detached),
                }
            }
            Ok(())
        })();
        checked.map_err(|_| {
            if effects {
                Error::EffectUncertain
            } else {
                Error::Detached
            }
        })
    }
}

impl SqliteDatabase {
    /// Side-effect free. An explicit initialize call alone may create storage.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            anchors: Arc::new(Anchors::default()),
            wal: false,
        }
    }

    /// Reuse a metadata owner's already fixed parent capability. SQLite still
    /// receives its original pathname; this never claims a dirfd-native VFS.
    pub fn in_directory(parent: Directory, name: &std::ffi::OsStr) -> Result<Self, Error> {
        parent.regular_identity(name)?; // also validates one safe leaf name
        let database = Self::new(parent.path().join(name));
        let _ = database.anchors.parent.set(parent);
        Ok(database)
    }

    /// Explicit repository policy. Default preserves the existing mode,
    /// including legacy DELETE-mode records databases.
    pub fn with_wal(mut self) -> Self {
        self.wal = true;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn parent(&self, create: bool) -> Result<Directory, Error> {
        if let Some(parent) = self.anchors.parent.get() {
            parent.verify_private_parent()?;
            return Ok(parent.clone());
        }
        let path = self
            .path
            .parent()
            .ok_or(Error::Invalid("absolute database path required"))?;
        let candidate = Directory::open(path, create)?;
        candidate.verify_private_parent()?;
        let _ = self.anchors.parent.set(candidate);
        let parent = self.anchors.parent.get().ok_or(Error::Detached)?.clone();
        parent.verify_private_parent()?;
        Ok(parent)
    }

    fn open(&self, create: bool, read_only: bool) -> Result<(Connection, Binding), Error> {
        // The bundled WAL-reset fix is a persistence requirement, not merely a
        // test dependency preference. 3.51.3 == 3_051_003.
        if rusqlite::version_number() < 3_051_003 {
            return Err(Error::Invalid("SQLite 3.51.3 or newer required"));
        }
        let parent = self.parent(create)?;
        let name = self
            .path
            .file_name()
            .ok_or(Error::Invalid("database leaf required"))?
            .to_owned();
        let before = parent.regular_identity(&name)?.map(allowed).transpose()?;
        if before.is_none() && (!create || self.anchors.main.get().is_some()) {
            return Err(Error::Missing);
        }
        if let Some(expected) = self.anchors.main.get() {
            if before != Some(*expected) {
                return Err(Error::Detached);
            }
        }
        let sidecars = ["-wal", "-shm", "-journal"]
            .into_iter()
            .map(|suffix| {
                let mut leaf = name.clone();
                leaf.push(suffix);
                let identity = parent.regular_identity(&leaf)?.map(allowed).transpose()?;
                Ok((leaf, identity))
            })
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        let mut flags = OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_NOFOLLOW;
        flags |= if read_only {
            OpenFlags::SQLITE_OPEN_READ_ONLY
        } else {
            OpenFlags::SQLITE_OPEN_READ_WRITE
        };
        if create {
            flags |= OpenFlags::SQLITE_OPEN_CREATE;
        }
        // Use SQLite's documented modeof URI on first creation, copying
        // permissions from a separate immutable 0600 metadata leaf. Do not
        // open/chmod/close the DB behind SQLite's POSIX locking machinery.
        let mode_template = if before.is_none() {
            let leaf = std::ffi::OsStr::new(".sqlite-create-mode-0600");
            match parent.publish_new(leaf, b"private SQLite creation permissions\n") {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
            let identity = parent
                .regular_identity(leaf)?
                .map(allowed)
                .transpose()?
                .ok_or(Error::Detached)?;
            if identity.mode & 0o777 != 0o600 {
                return Err(Error::Invalid("private SQLite creation template required"));
            }
            Some(parent.path().join(leaf))
        } else {
            None
        };
        let connection = if read_only || mode_template.is_some() {
            let mut uri = format!("file:{}", uri_path(&self.path));
            if read_only {
                // Native READ_ONLY alone may still create/repair SHM. The
                // bundled Unix VFS option refuses that observation-side write.
                uri.push_str("?mode=ro&readonly_shm=1");
            } else if let Some(template) = mode_template {
                uri.push_str("?modeof=");
                uri.push_str(&uri_path(&template));
            }
            Connection::open_with_flags(uri, flags | OpenFlags::SQLITE_OPEN_URI)?
        } else {
            Connection::open_with_flags(&self.path, flags)?
        };
        // Disable close-time checkpoint/deletion before any SQL. Otherwise Drop
        // might unlink names under a detached parent after the final check.
        connection.set_db_config(DbConfig::SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, true)?;
        let after = parent
            .regular_identity(&name)?
            .map(allowed)
            .transpose()?
            .ok_or(Error::Detached)?;
        if before.is_none() && after.mode & 0o777 != 0o600 {
            return Err(Error::Invalid(
                "SQLite created database with unsafe permissions",
            ));
        }
        if before.is_some_and(|before| before != after) {
            return Err(Error::Detached);
        }
        let _ = self.anchors.main.set(after);
        if self.anchors.main.get() != Some(&after) {
            return Err(Error::Detached);
        }
        let binding = Binding {
            parent,
            name,
            main: after,
            sidecars: RefCell::new(sidecars),
        };
        binding.validate(false)?;
        connection.busy_timeout(Duration::from_secs(30))?;
        if !read_only {
            connection.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")?;
        }
        binding.validate(false)?;
        Ok((connection, binding))
    }

    fn finish<T, E: From<Error>>(
        connection: Connection,
        binding: &Binding,
        result: Result<T, E>,
        effects: bool,
    ) -> Result<T, E> {
        let before = binding.validate(effects);
        let closed = connection
            .close()
            .map_err(|(_, error)| Error::Sqlite(error));
        let after = binding.validate(effects);
        before.map_err(E::from)?;
        if let Err(error) = closed {
            return Err(E::from(if effects {
                Error::EffectUncertain
            } else {
                error
            }));
        }
        after.map_err(E::from)?;
        result
    }

    /// Existing storage only. No schema migration, CREATE flag or directory
    /// creation. SQLite's native WAL reader may require an already valid SHM;
    /// this is observation, not an immutable snapshot of a changing database.
    pub fn read_only<T, E: From<Error>>(
        &self,
        operation: impl FnOnce(&Connection) -> Result<T, E>,
    ) -> Result<T, E> {
        let (connection, binding) = self.open(false, true).map_err(E::from)?;
        let result = operation(&connection);
        Self::finish(connection, &binding, result, false)
    }

    /// Existing read/write storage without schema initialization.
    pub fn read<T, E: From<Error>>(
        &self,
        operation: impl FnOnce(&Connection) -> Result<T, E>,
    ) -> Result<T, E> {
        let (connection, binding) = self.open(false, false).map_err(E::from)?;
        let result = operation(&connection);
        Self::finish(connection, &binding, result, false)
    }

    pub fn initialize<T, E: From<Error>>(
        &self,
        validate: impl Fn() -> Result<(), E>,
        operation: impl FnOnce(&Transaction<'_>) -> Result<T, E>,
    ) -> Result<T, E> {
        self.write(true, validate, operation)
    }

    pub fn transaction<T, E: From<Error>>(
        &self,
        validate: impl Fn() -> Result<(), E>,
        operation: impl FnOnce(&Transaction<'_>) -> Result<T, E>,
    ) -> Result<T, E> {
        self.write(false, validate, operation)
    }

    fn write<T, E: From<Error>>(
        &self,
        create: bool,
        validate: impl Fn() -> Result<(), E>,
        operation: impl FnOnce(&Transaction<'_>) -> Result<T, E>,
    ) -> Result<T, E> {
        validate()?;
        let (mut connection, binding) = self.open(create, false).map_err(E::from)?;
        let result = (|| {
            validate()?;
            if create && self.wal {
                connection
                    .execute_batch("PRAGMA journal_mode=WAL;")
                    .map_err(Error::from)
                    .map_err(E::from)?;
                binding.validate(true).map_err(E::from)?;
                if validate().is_err() {
                    return Err(E::from(Error::EffectUncertain));
                }
            }
            let transaction = connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(Error::from)
                .map_err(E::from)?;
            binding.validate(false).map_err(E::from)?;
            validate()?;
            let value = operation(&transaction)?;
            binding.validate(true).map_err(E::from)?;
            if validate().is_err() {
                return Err(E::from(Error::EffectUncertain));
            }
            transaction
                .commit()
                .map_err(|_| E::from(Error::EffectUncertain))?;
            binding.validate(true).map_err(E::from)?;
            if validate().is_err() {
                return Err(E::from(Error::EffectUncertain));
            }
            Ok(value)
        })();
        let result = Self::finish(connection, &binding, result, true);
        if validate().is_err() {
            return Err(E::from(Error::EffectUncertain));
        }
        result
    }
}

#[cfg(test)]
mod tests;
