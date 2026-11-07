//! Which operation holds a cloud file lock (the run lock or the effects lock),
//! for diagnostics only.
//!
//! The file lock stays the authority: this slot only remembers the holder in this
//! process, so a holder in another process is reported as unknown.
use ml_exp_io::directory::FileLock;
use std::{
    io::Write,
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, Instant},
};

/// A hold at least this long is logged when it ends.
const SLOW: Duration = Duration::from_secs(2);

type Held = Option<(&'static str, Instant)>;

/// `name` starts every line, so each lock's lines stay apart.
#[derive(Clone)]
pub(crate) struct Holder {
    name: &'static str,
    held: Arc<Mutex<Held>>,
}

impl Holder {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            held: Arc::default(),
        }
    }
    fn slot(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(PoisonError::into_inner)
    }
    pub(crate) fn acquired(&self, lock: FileLock, label: &'static str) -> Guard {
        let at = Instant::now();
        *self.slot() = Some((label, at));
        Guard {
            _lock: lock,
            holder: self.clone(),
            label,
            at,
        }
    }
    fn current(&self, now: Instant) -> Option<(&'static str, Duration)> {
        self.slot()
            .map(|(label, at)| (label, now.saturating_duration_since(at)))
    }
    /// One line for a waiter that gave up with a busy error.
    pub(crate) fn busy(&self, waiter: &'static str) {
        emit(&busy_line(self.name, waiter, self.current(Instant::now())));
    }
}

/// A cloud file lock, with its holder recorded until it is released.
pub(crate) struct Guard {
    _lock: FileLock,
    holder: Holder,
    label: &'static str,
    at: Instant,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let held = self.at.elapsed();
        {
            let mut slot = self.holder.slot();
            if slot.is_some_and(|(label, at)| label == self.label && at == self.at) {
                *slot = None;
            }
        }
        if held >= SLOW {
            emit(&held_line(self.holder.name, self.label, held));
        }
    }
}

fn held_line(name: &str, label: &str, held: Duration) -> String {
    format!("{name}: held {label} {}ms", held.as_millis())
}

fn busy_line(name: &str, waiter: &str, holder: Option<(&str, Duration)>) -> String {
    match holder {
        Some((label, held)) => format!(
            "{name}: busy waiter={waiter} holder={label} held_for={}ms",
            held.as_millis()
        ),
        None => format!("{name}: busy waiter={waiter} holder=other process held_for=unknown"),
    }
}

fn emit(line: &str) {
    let _ = writeln!(std::io::stderr().lock(), "{line}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_name_the_holder() {
        assert_eq!(
            held_line("cloud-lock", "retry.retry", Duration::from_millis(2345)),
            "cloud-lock: held retry.retry 2345ms"
        );
        assert_eq!(
            busy_line(
                "cloud-lock",
                "actions.cancel",
                Some(("exchange.receive", Duration::from_millis(4100)))
            ),
            "cloud-lock: busy waiter=actions.cancel holder=exchange.receive held_for=4100ms"
        );
        assert_eq!(
            busy_line("cloud-lock", "actions.cancel", None),
            "cloud-lock: busy waiter=actions.cancel holder=other process held_for=unknown"
        );
        assert_eq!(
            busy_line(
                "cloud-effects-lock",
                "outbox.status",
                Some(("outbox.execute", Duration::from_millis(900)))
            ),
            "cloud-effects-lock: busy waiter=outbox.status holder=outbox.execute held_for=900ms"
        );
    }

    #[test]
    fn slot_follows_the_guard() {
        let root = tempfile::tempdir().unwrap();
        let directory = ml_exp_io::directory::Directory::open(root.path(), false).unwrap();
        let holder = Holder::new("cloud-lock");
        assert!(holder.current(Instant::now()).is_none());
        let lock = directory
            .lock(
                std::ffi::OsStr::new("l"),
                ml_exp_io::directory::LockOptions {
                    create: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let guard = holder.acquired(lock, "retry.retry");
        assert_eq!(
            holder.current(Instant::now()).map(|(label, _)| label),
            Some("retry.retry")
        );
        drop(guard);
        assert!(holder.current(Instant::now()).is_none());
    }
}
