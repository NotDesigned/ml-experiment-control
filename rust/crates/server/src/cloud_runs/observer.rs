//! One bounded read-only scheduler poll at a time. Discovery reads original
//! Actions; neither polling nor restart ever submits a missing cloud job.
use super::{Service, service::Prepared};
use fs2::FileExt;
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
pub struct Observer {
    stop: Arc<AtomicBool>,
    wake: Arc<(Mutex<()>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}
impl Observer {
    pub fn start(service: Arc<Service>) -> crate::backends::cloud::Result<Self> {
        let leader = service
            .directory
            .open_lock(OsStr::new(".observer.lock"))
            .map_err(super::service::io)?;
        leader
            .try_lock_exclusive()
            .map_err(|_| crate::backends::cloud::Error::evidence())?;
        let stop = Arc::new(AtomicBool::new(false));
        let wake = Arc::new((Mutex::new(()), Condvar::new()));
        let flag = stop.clone();
        let sleep = wake.clone();
        let handle = thread::Builder::new()
            .name("native-cloud-observer".into())
            .spawn(move || {
                let start = Instant::now();
                let mut schedule = ml_exp_core::polling::Schedule::default();
                let policy = service.polling.clone();
                let mut candidates = BTreeSet::new();
                let mut discover_at = 0;
                while !flag.load(Ordering::Relaxed) {
                    if service
                        .directory
                        .validate_regular_entry(OsStr::new(".observer.lock"), &leader)
                        .is_err()
                    {
                        break;
                    }
                    let elapsed = start.elapsed().as_secs();
                    if elapsed >= discover_at {
                        if let Ok(found) = service.observer_candidates() {
                            candidates = found;
                        }
                        if let Ok((now, _)) = crate::projects::transition_context() {
                            let _ = service.debug_expire(&now);
                        }
                        discover_at = elapsed.saturating_add(30);
                    }
                    let (next, selected) = schedule.select(&candidates, elapsed, &policy);
                    schedule = next;
                    if let Some((action, _)) = selected {
                        let result = crate::projects::transition_context()
                            .map_err(|_| ())
                            .and_then(|(now, _)| {
                                service.refresh(&action, &now, &flag).map_err(|_| ())
                            });
                        schedule = schedule.outcome(
                            &action,
                            start.elapsed().as_secs(),
                            &policy,
                            result.is_ok(),
                            false,
                        );
                    }
                    let Ok(guard) = sleep.0.lock() else {
                        break;
                    };
                    if flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let _ = sleep.1.wait_timeout(guard, Duration::from_secs(1));
                }
            })
            .map_err(|_| crate::backends::cloud::Error::configuration())?;
        Ok(Self {
            stop,
            wake,
            thread: Some(handle),
        })
    }
}
impl Drop for Observer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Ok(_guard) = self.wake.0.lock() {
            self.wake.1.notify_all();
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}
impl Service {
    fn observer_candidates(&self) -> crate::backends::cloud::Result<BTreeSet<String>> {
        let mut result = BTreeSet::new();
        for name in self
            .slots
            .entry_names_limited(16384)
            .map_err(super::service::io)?
        {
            let Some(name) = name.to_str() else {
                continue;
            };
            // Preparation slot names are a single raw SHA256 plus .json;
            // progress/history/caches cannot be mistaken for an Action.
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            if ml_exp_core::sources::validate_archive_digest(id).is_err() {
                continue;
            }
            let prepared: Prepared =
                serde_json::from_value(super::service::read(&self.slots, name)?)
                    .map_err(|_| crate::backends::cloud::Error::evidence())?;
            let action = prepared.plan["action_id"]
                .as_str()
                .ok_or_else(crate::backends::cloud::Error::evidence)?;
            self.check(&prepared)?;
            if self.actions.observe(action).map_err(super::service::op)?["execution"]["status"]
                != "VERIFIED"
            {
                continue;
            }
            let state = self.outbox.optional_status(&prepared.execution.job)?;
            let terminal = state
                .as_ref()
                .and_then(|s| s["observation"]["state"].as_str())
                .is_some_and(|s| {
                    matches!(
                        s,
                        "SUCCEEDED" | "FAILED" | "CANCELLED" | "TIMEOUT" | "PREEMPTED"
                    )
                });
            if !terminal {
                result.insert(action.into());
            }
            if result.len() > 4096 {
                return Err(crate::backends::cloud::Error::evidence());
            }
        }
        Ok(result)
    }
}
