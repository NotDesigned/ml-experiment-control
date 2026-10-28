//! One cancellable observer per isolated native namespace. No scheduler
//! mutation, checkpoint selection, automatic retry or result release path.
use crate::submissions::{Error, Service};
use ml_exp_core::polling::{Policy, Schedule};
use std::{
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
    pub fn start(service: Arc<Service>, policy: Policy) -> Result<Self, Error> {
        if !policy.enabled || !policy.validate() {
            return Err(Error::Configuration);
        }
        let leader = service.background_leader()?;
        let stop = Arc::new(AtomicBool::new(false));
        let wake = Arc::new((Mutex::new(()), Condvar::new()));
        let flag = stop.clone();
        let sleep = wake.clone();
        let handle = thread::Builder::new()
            .name("native-observer".into())
            .spawn(move || {
                let start = Instant::now();
                let mut schedule = Schedule::default();
                let mut candidates = Default::default();
                let mut discover_at = 0;
                let _ = service.background_report(None, None, Ok(()));
                while !flag.load(Ordering::Relaxed) {
                    if service.validate_background_leader(&leader).is_err() {
                        break;
                    }
                    let elapsed = start.elapsed().as_secs();
                    if elapsed >= discover_at {
                        match service.background_candidates(policy.max_candidates) {
                            Ok(found) => {
                                candidates = found;
                                if candidates.is_empty() {
                                    let _ = service.background_report(None, None, Ok(()));
                                }
                            }
                            Err(error) => {
                                let _ = service.background_report(None, None, Err(&error));
                            }
                        }
                        discover_at = elapsed.saturating_add(30);
                    }
                    let (next, selected) = schedule.select(&candidates, elapsed, &policy);
                    schedule = next;
                    if let Some((action, phase)) = selected {
                        if let Ok((now, _)) = crate::projects::transition_context() {
                            let outcome = service.background_sample(
                                &action,
                                phase,
                                &now,
                                &flag,
                                Duration::from_secs(policy.rpc_timeout_seconds),
                            );
                            let busy = matches!(outcome, Err(Error::Busy));
                            schedule = schedule.outcome(
                                &action,
                                start.elapsed().as_secs(),
                                &policy,
                                outcome.is_ok(),
                                busy,
                            );
                            let _ = service.background_report(
                                Some(&action),
                                Some(phase),
                                outcome.as_ref().map(|_| ()),
                            );
                        }
                    }
                    let Ok(guard) = sleep.0.lock() else {
                        break;
                    };
                    if flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let _ = sleep.1.wait_timeout(guard, Duration::from_secs(1));
                }
                let _ = service.background_stopped();
            })
            .map_err(|_| Error::Configuration)?;
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
