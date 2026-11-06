//! One bounded read-only scheduler poll at a time. Discovery reads original
//! Actions; neither polling nor restart ever submits a missing cloud job.
use super::{Service, engine::Prepared};
use crate::background::{Polled, Sample, Worker};
use ml_exp_core::polling::Phase;
use ml_exp_io::directory::LockOptions;
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    sync::{Arc, atomic::AtomicBool},
};

/// Cloud's side of the observer. A sample only reaches `outbox.reconcile`, which
/// never enters create, and no status is written. Sessions expire during discovery.
struct Polling(Arc<Service>);
impl Polled for Polling {
    fn discover(&self) -> Option<BTreeSet<String>> {
        let found = self.0.observer_candidates().ok();
        if let Ok((now, _)) = crate::projects::transition_context() {
            let _ = self.0.debug_expire(&now);
        }
        found
    }
    fn sample(&self, action: &str, phase: Phase, stop: &AtomicBool) -> Sample {
        // Worker evidence and history arrive through the exchange. Only the
        // scheduler phase has a provider read, so an Action costs one per cycle.
        if phase != Phase::Scheduler {
            return Sample::Skipped;
        }
        let refreshed = crate::projects::transition_context()
            .map_err(why!(()))
            .and_then(|(now, _)| self.0.refresh(action, &now, stop).map_err(why!(())));
        if refreshed.is_ok() {
            Sample::Done
        } else {
            Sample::Failed
        }
    }
}
impl Service {
    /// The adapter alone, so a test can drive it without the thread.
    #[doc(hidden)]
    pub fn observer_polling(self: Arc<Self>) -> impl Polled {
        Polling(self)
    }
    pub fn start_observer(self: Arc<Self>) -> crate::backends::cloud::Result<Worker> {
        let leader = self
            .directory
            .lock(
                OsStr::new(".observer.lock"),
                LockOptions {
                    create: true,
                    ..Default::default()
                },
            )
            .map_err(why_lock!(crate::backends::cloud::Error::evidence()))?;
        let policy = self.polling.clone();
        crate::background::observe("native-cloud-observer", Polling(self), policy, leader)
            .map_err(why!(crate::backends::cloud::Error::configuration()))
    }
}
impl Service {
    fn observer_candidates(&self) -> crate::backends::cloud::Result<BTreeSet<String>> {
        let mut result = BTreeSet::new();
        for name in self.slots.entry_names().map_err(super::engine::io)? {
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
                serde_json::from_value(super::engine::read(&self.slots, name)?)
                    .map_err(why!(crate::backends::cloud::Error::evidence()))?;
            let action = prepared.plan["action_id"]
                .as_str()
                .ok_or_else(crate::backends::cloud::Error::evidence)?;
            if self.actions.observe(action).map_err(super::engine::op)?["execution"]["status"]
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
        }
        Ok(result)
    }
}
