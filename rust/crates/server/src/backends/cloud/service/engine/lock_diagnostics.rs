//! Which operation holds the cloud run lock, for diagnostics only.
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

#[derive(Clone, Default)]
pub(super) struct Holder(Arc<Mutex<Held>>);

impl Holder {
    fn slot(&self) -> std::sync::MutexGuard<'_, Held> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
    pub(super) fn acquired(&self, lock: FileLock, label: &'static str) -> Guard {
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
    /// One line for a waiter that gave up with CLOUD_RUN_BUSY.
    pub(super) fn busy(&self, waiter: &'static str) {
        emit(&busy_line(waiter, self.current(Instant::now())));
    }
}

/// The cloud run lock, with its holder recorded until it is released.
pub(super) struct Guard {
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
            emit(&held_line(self.label, held));
        }
    }
}

fn held_line(label: &str, held: Duration) -> String {
    format!("cloud-lock: held {label} {}ms", held.as_millis())
}

fn busy_line(waiter: &str, holder: Option<(&str, Duration)>) -> String {
    match holder {
        Some((label, held)) => format!(
            "cloud-lock: busy waiter={waiter} holder={label} held_for={}ms",
            held.as_millis()
        ),
        None => format!("cloud-lock: busy waiter={waiter} holder=other process held_for=unknown"),
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
            held_line("retry.retry", Duration::from_millis(2345)),
            "cloud-lock: held retry.retry 2345ms"
        );
        assert_eq!(
            busy_line(
                "actions.cancel",
                Some(("exchange.receive", Duration::from_millis(4100)))
            ),
            "cloud-lock: busy waiter=actions.cancel holder=exchange.receive held_for=4100ms"
        );
        assert_eq!(
            busy_line("actions.cancel", None),
            "cloud-lock: busy waiter=actions.cancel holder=other process held_for=unknown"
        );
    }

    #[test]
    fn slot_follows_the_guard() {
        let root = tempfile::tempdir().unwrap();
        let directory = ml_exp_io::directory::Directory::open(root.path(), false).unwrap();
        let holder = Holder::default();
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
