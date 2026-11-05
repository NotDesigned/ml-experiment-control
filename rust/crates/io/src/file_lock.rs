//! Advisory whole-file locks taken through a fresh open file description.
//!
//! flock belongs to the description, so a fresh open per acquire makes two
//! acquisitions in one process exclude each other exactly as two processes do.
//! Never point this at a SQLite file: closing any descriptor of that file drops
//! the POSIX locks SQLite holds on it.
use std::{
    fs::File,
    io,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const POLL: Duration = Duration::from_millis(5);

/// What to do when the lock is held. A wait that ends without the lock returns
/// `WouldBlock`, the same as `Try`, so callers keep one mapping for contention.
#[derive(Clone, Copy, Debug, Default)]
pub enum Wait<'a> {
    #[default]
    Try,
    Block,
    Until {
        deadline: Instant,
        cancelled: Option<&'a AtomicBool>,
    },
}

/// The default is an exclusive `Try` on an existing leaf, opened read-only.
#[derive(Clone, Copy, Debug, Default)]
pub struct LockOptions<'a> {
    /// Create the leaf when missing. Read and RPC paths must leave this false,
    /// because they may not repair state, and a ledger lock must too, because
    /// a recreated ledger reads as unstarted.
    pub create: bool,
    pub shared: bool,
    /// Open read-write for a lock file that also holds data.
    pub write: bool,
    pub wait: Wait<'a>,
}

/// A held lock. It reads as the locked `File`.
#[derive(Debug)]
pub struct FileLock(File);

impl Deref for FileLock {
    type Target = File;
    fn deref(&self) -> &File {
        &self.0
    }
}
impl DerefMut for FileLock {
    fn deref_mut(&mut self) -> &mut File {
        &mut self.0
    }
}
impl Drop for FileLock {
    fn drop(&mut self) {
        // A concurrent Command fork can briefly inherit this description before
        // CLOEXEC closes it. Closing only our descriptor would leave the lock
        // with that child, so the release is explicit.
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

fn attempt(file: &File, shared: bool) -> io::Result<()> {
    let result = if shared {
        fs2::FileExt::try_lock_shared(file)
    } else {
        fs2::FileExt::try_lock_exclusive(file)
    };
    // Windows reports contention as ERROR_LOCK_VIOLATION, without this kind.
    result.map_err(|error| {
        if error.raw_os_error() == fs2::lock_contended_error().raw_os_error() {
            io::Error::from(io::ErrorKind::WouldBlock)
        } else {
            error
        }
    })
}

fn block(file: &File, shared: bool) -> io::Result<()> {
    loop {
        let result = if shared {
            fs2::FileExt::lock_shared(file)
        } else {
            fs2::FileExt::lock_exclusive(file)
        };
        match result {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

fn until(
    file: &File,
    shared: bool,
    deadline: Instant,
    cancelled: Option<&AtomicBool>,
) -> io::Result<()> {
    loop {
        match attempt(file, shared) {
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if cancelled.is_some_and(|flag| flag.load(Ordering::Relaxed))
                    || Instant::now() >= deadline
                {
                    return Err(error);
                }
                std::thread::sleep(POLL);
            }
            other => return other,
        }
    }
}

pub(crate) fn acquire(file: File, shared: bool, wait: Wait<'_>) -> io::Result<FileLock> {
    match wait {
        Wait::Try => attempt(&file, shared)?,
        Wait::Block => block(&file, shared)?,
        Wait::Until {
            deadline,
            cancelled,
        } => until(&file, shared, deadline, cancelled)?,
    }
    Ok(FileLock(file))
}
