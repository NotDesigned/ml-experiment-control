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
            if !preparation_slot(name) {
                continue;
            }
            let prepared: Prepared =
                serde_json::from_value(super::engine::read(&self.slots, name)?)
                    .map_err(why!(crate::backends::cloud::Error::evidence()))?;
            // A slot holds only the Attempt its name was derived from.
            let binding = &prepared.execution.binding;
            if ml_exp_core::first_submission::native::attempt_slot(
                &binding.project,
                &binding.run_id,
                &binding.attempt_id,
            )
            .ok()
            .is_none_or(|slot| format!("{slot}.json") != name)
            {
                continue;
            }
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
/// A first Attempt's preparation is a raw SHA256 plus `.json`; a retry's adds its
/// Attempt, as in `<sha256>.attempt-002.json`. Progress, history, exchange and other
/// files beside them cannot be mistaken for an Action.
fn preparation_slot(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".json") else {
        return false;
    };
    let (digest, attempt) = match stem.split_once('.') {
        Some((digest, attempt)) => (digest, Some(attempt)),
        None => (stem, None),
    };
    ml_exp_core::sources::validate_archive_digest(digest).is_ok()
        && attempt.is_none_or(|attempt| {
            attempt != "attempt-001"
                && ml_exp_core::first_submission::native::retry::number(attempt).is_ok()
        })
}
#[cfg(test)]
mod tests {
    use super::preparation_slot;
    #[test]
    fn retry_preparations_are_candidates_and_their_companions_are_not() {
        let digest = "a".repeat(64);
        assert!(preparation_slot(&format!("{digest}.json")));
        assert!(preparation_slot(&format!("{digest}.attempt-002.json")));
        assert!(preparation_slot(&format!("{digest}.attempt-120.json")));
        for name in [
            format!("{digest}.attempt-001.json"),
            format!("{digest}.attempt-2.json"),
            format!("{digest}.attempt-002.json.exchange.json"),
            format!("{digest}.attempt-002.startup-failure.json"),
            format!("{digest}.json.artifacts.json"),
            format!("{digest}.startup-failure.json"),
            "prepared-action-0123456789abcdef.json".into(),
            format!("dataset-{digest}.json"),
            format!("runtime.{digest}.runtime.json"),
            "capacity-pai-h200.json".into(),
        ] {
            assert!(!preparation_slot(&name), "{name}");
        }
    }
}
