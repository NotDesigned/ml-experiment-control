//! Fair bounded background observation scheduling. Time is supplied by the
//! caller; these decisions never authorize a provider mutation or a retry.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A refresh cannot change job ownership or erase an accepted terminal result.
pub fn scheduler_follows(
    previous: &crate::slurm::JobStatus,
    next: &crate::slurm::StatusObservation,
) -> bool {
    let status = &next.status;
    previous.validate().is_ok()
        && status.validate().is_ok()
        && !next.retained_previous
        && status.ssh_alias == previous.ssh_alias
        && status.job_id == previous.job_id
        && status.scheduler_name == previous.scheduler_name
        && status.partition == previous.partition
        && status.state != crate::state::RunState::Unknown
        && (!previous.state.is_terminal() || status.state == previous.state)
        && !(previous.state == crate::state::RunState::Running
            && status.state == crate::state::RunState::Queued)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Policy {
    pub enabled: bool,
    pub interval_seconds: u64,
    pub max_candidates: usize,
    pub rpc_timeout_seconds: u64,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_seconds: 10,
            max_candidates: 4096,
            rpc_timeout_seconds: 5,
        }
    }
}
impl Policy {
    pub fn validate(&self) -> bool {
        (2..=300).contains(&self.interval_seconds)
            && (1..=4096).contains(&self.max_candidates)
            && (1..=30).contains(&self.rpc_timeout_seconds)
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Scheduler,
    Worker,
    History,
}
impl Phase {
    fn next(self) -> Self {
        match self {
            Self::Scheduler => Self::Worker,
            Self::Worker => Self::History,
            Self::History => Self::Scheduler,
        }
    }
}
#[derive(Clone, Debug, Default)]
struct Entry {
    due: u64,
    failures: u32,
    phase: Phase,
}
#[derive(Clone, Debug, Default)]
pub struct Schedule {
    entries: BTreeMap<String, Entry>,
    after: String,
}
impl Schedule {
    pub fn select(
        mut self,
        candidates: &BTreeSet<String>,
        now: u64,
        policy: &Policy,
    ) -> (Self, Option<(String, Phase)>) {
        if !policy.enabled || !policy.validate() || candidates.len() > policy.max_candidates {
            return (self, None);
        }
        self.entries.retain(|key, _| candidates.contains(key));
        for key in candidates {
            self.entries.entry(key.clone()).or_default();
        }
        let ready = |key: &&String| self.entries[*key].due <= now;
        let selected = candidates
            .iter()
            .filter(ready)
            .find(|key| *key > &self.after)
            .or_else(|| candidates.iter().find(ready))
            .cloned();
        let Some(key) = selected else {
            return (self, None);
        };
        self.after = key.clone();
        let phase = self.entries[&key].phase;
        (self, Some((key, phase)))
    }
    pub fn outcome(
        mut self,
        key: &str,
        now: u64,
        policy: &Policy,
        success: bool,
        busy: bool,
    ) -> Self {
        if let Some(entry) = self.entries.get_mut(key) {
            if busy {
                entry.due = now.saturating_add(1);
                return self;
            }
            entry.failures = if success {
                0
            } else {
                entry.failures.saturating_add(1).min(5)
            };
            entry.phase = entry.phase.next();
            let delay = policy
                .interval_seconds
                .saturating_mul(1 << entry.failures)
                .min(300);
            entry.due = now.saturating_add(if success && entry.phase != Phase::Scheduler {
                1
            } else {
                delay
            });
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fair_rotation_backoff_and_busy_do_not_discard_pending_phases() {
        let policy = Policy {
            enabled: true,
            ..Policy::default()
        };
        let candidates = BTreeSet::from(["a".into(), "b".into()]);
        let (state, selected) = Schedule::default().select(&candidates, 0, &policy);
        assert_eq!(selected, Some(("a".into(), Phase::Scheduler)));
        let state = state.outcome("a", 0, &policy, false, true);
        let (state, selected) = state.select(&candidates, 0, &policy);
        assert_eq!(selected, Some(("b".into(), Phase::Scheduler)));
        let state = state.outcome("b", 0, &policy, true, false);
        let (state, selected) = state.select(&candidates, 1, &policy);
        assert_eq!(selected, Some(("a".into(), Phase::Scheduler)));
        let state = state.outcome("a", 1, &policy, false, false);
        let (state, selected) = state.select(&BTreeSet::from(["a".into()]), 20, &policy);
        assert_eq!(selected, None);
        assert_eq!(
            state.select(&BTreeSet::from(["a".into()]), 21, &policy).1,
            Some(("a".into(), Phase::Worker))
        );
    }
}
