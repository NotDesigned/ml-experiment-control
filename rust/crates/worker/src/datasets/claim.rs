//! Who may build one asset's tree. flock does not exclude across nodes on every shared
//! filesystem, while creating a name exclusively does on those measured. No marker can
//! prove its owner dead, so an owner proves itself alive by changing the marker, and a
//! waiter takes over only after it stayed unchanged for a lease on the waiter's own
//! clock. A paused owner could outlive that, so it rechecks before making anything final.
use ml_exp_io::directory::Directory;
use std::{
    ffi::OsStr,
    io,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const NAME: &str = "building";

#[derive(Clone, Copy)]
pub(super) struct Timing {
    pub beat: Duration,
    /// Longer than a new name can take to appear on another node, about 4.5 minutes as
    /// measured on LeoFS, so that a slow view is not taken for a dead owner.
    pub lease: Duration,
    pub poll: Duration,
    pub limit: Duration,
}
impl Default for Timing {
    fn default() -> Self {
        Self {
            beat: Duration::from_secs(20),
            lease: Duration::from_secs(360),
            poll: Duration::from_secs(5),
            limit: Duration::from_secs(3600),
        }
    }
}

pub(super) struct Claim<'a> {
    directory: &'a Directory,
    owner: String,
    timing: Timing,
}

fn name() -> &'static OsStr {
    OsStr::new(NAME)
}

fn owner() -> String {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}:{}:{now}", host.trim(), std::process::id())
}

impl<'a> Claim<'a> {
    /// Waits for another build to end. The caller must read the state again afterwards.
    pub(super) fn acquire(
        directory: &'a Directory,
        timing: Timing,
        cancelled: &AtomicBool,
    ) -> io::Result<Self> {
        let owner = owner();
        let start = Instant::now();
        let mut seen: Option<Vec<u8>> = None;
        let mut since = Instant::now();
        loop {
            if cancelled.load(Ordering::Relaxed) || start.elapsed() >= timing.limit {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "another build of this dataset is still running",
                ));
            }
            match directory.publish_new(name(), owner.as_bytes()) {
                Ok(()) => {
                    return Ok(Self {
                        directory,
                        owner,
                        timing,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
            match directory.read_regular(name(), 4096) {
                Ok(current) if seen.as_ref() != Some(&current) => {
                    seen = Some(current);
                    since = Instant::now();
                }
                Ok(_) if since.elapsed() >= timing.lease => {
                    eprintln!("ML_EXPD_DATA_BUILD_TAKEOVER the marker stopped changing");
                    match directory.remove_staged_regular(name()) {
                        Err(error) if error.kind() != io::ErrorKind::NotFound => {
                            return Err(error);
                        }
                        _ => {
                            seen = None;
                            continue;
                        }
                    }
                }
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            std::thread::sleep(timing.poll);
        }
    }
    fn held(&self) -> bool {
        self.directory
            .read_regular(name(), 4096)
            .is_ok_and(|bytes| bytes.starts_with(self.owner.as_bytes()))
    }
    /// Before anything final: another node may have taken over from a paused owner.
    pub(super) fn still_held(&self) -> io::Result<()> {
        if self.held() {
            Ok(())
        } else {
            Err(io::Error::other("another build took over this dataset"))
        }
    }
    /// Runs `work` while showing that this owner is alive.
    pub(super) fn beating<T>(&self, work: impl FnOnce() -> T) -> T {
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut count = 0u64;
                let mut last = Instant::now();
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(self.timing.beat.min(Duration::from_millis(200)));
                    if last.elapsed() >= self.timing.beat {
                        last = Instant::now();
                        count += 1;
                        // A failed beat only risks a takeover, which `still_held` catches.
                        if self.held() {
                            let text = format!("{}\n{count}", self.owner);
                            let _ = self.directory.replace_regular(name(), text.as_bytes());
                        }
                    }
                }
            });
            let result = work();
            stop.store(true, Ordering::Relaxed);
            result
        })
    }
    /// A marker left behind only delays the next build by one lease.
    pub(super) fn release(self) {
        if self.held() {
            let _ = self.directory.remove_staged_regular(name());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const QUICK: Timing = Timing {
        beat: Duration::from_millis(20),
        lease: Duration::from_millis(150),
        poll: Duration::from_millis(10),
        limit: Duration::from_secs(5),
    };
    fn directory(root: &tempfile::TempDir) -> Directory {
        Directory::open(root.path(), false).unwrap()
    }
    #[test]
    fn one_owner_at_a_time_and_release_lets_the_next_in() {
        let root = tempfile::tempdir().unwrap();
        let directory = directory(&root);
        let cancelled = AtomicBool::new(false);
        let first = Claim::acquire(&directory, QUICK, &cancelled).unwrap();
        first.still_held().unwrap();
        first.release();
        assert!(!directory.entry_exists(name()).unwrap());
        Claim::acquire(&directory, QUICK, &cancelled)
            .unwrap()
            .release();
    }
    #[test]
    fn a_live_owner_is_never_taken_over() {
        let root = tempfile::tempdir().unwrap();
        let directory = directory(&root);
        let cancelled = AtomicBool::new(false);
        let first = Claim::acquire(&directory, QUICK, &cancelled).unwrap();
        let short = Timing {
            limit: Duration::from_millis(600),
            ..QUICK
        };
        first.beating(|| {
            let error = Claim::acquire(&directory, short, &cancelled).err().unwrap();
            assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
        });
        first.still_held().unwrap();
    }
    #[test]
    fn a_silent_owner_is_taken_over_and_then_knows_it() {
        let root = tempfile::tempdir().unwrap();
        let directory = directory(&root);
        let cancelled = AtomicBool::new(false);
        let silent = Claim::acquire(&directory, QUICK, &cancelled).unwrap();
        let next = Claim::acquire(&directory, QUICK, &cancelled).unwrap();
        assert!(silent.still_held().is_err());
        next.still_held().unwrap();
        silent.release();
        next.still_held()
            .expect("a late release leaves the new owner alone");
    }
    #[test]
    fn cancelling_stops_the_wait() {
        let root = tempfile::tempdir().unwrap();
        let directory = directory(&root);
        let _first = Claim::acquire(&directory, QUICK, &AtomicBool::new(false)).unwrap();
        let error = Claim::acquire(&directory, QUICK, &AtomicBool::new(true))
            .err()
            .unwrap();
        assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    }
}
