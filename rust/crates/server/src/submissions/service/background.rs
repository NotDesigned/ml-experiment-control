//! Opt-in background sampling shares the explicit read-only effect paths.
//! Each tick performs at most one bounded RPC and releases the execution lock.
use super::*;
use ml_exp_core::polling::Phase;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Samples {
    request_sha256: String,
    phases: BTreeMap<String, Sample>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    at: String,
    error: Option<String>,
    failures: u32,
    last_success_at: Option<String>,
}

impl Service {
    pub fn background_leader(&self) -> Result<File> {
        let file = self
            .directory
            .open_lock(OsStr::new(".observer.lock"))
            .map_err(|_| Error::Authority)?;
        file.try_lock_exclusive().map_err(|_| Error::Busy)?;
        self.directory
            .validate_regular_entry(OsStr::new(".observer.lock"), &file)
            .map_err(|_| Error::Authority)?;
        Ok(file)
    }
    pub fn validate_background_leader(&self, file: &File) -> Result<()> {
        self.directory
            .validate_regular_entry(OsStr::new(".observer.lock"), file)
            .map_err(|_| Error::Authority)
    }
    pub fn background_report(
        &self,
        action: Option<&str>,
        phase: Option<Phase>,
        result: std::result::Result<(), &Error>,
    ) -> Result<()> {
        let (at, _) = crate::projects::transition_context().map_err(|_| Error::Configuration)?;
        let code = result.err().map(|error| match error {
            Error::Busy => "BUSY",
            Error::ObservationUnavailable => "OBSERVATION_UNAVAILABLE",
            Error::ObservationConflict => "OBSERVATION_CONFLICT",
            _ => "OBSERVATION_REJECTED",
        });
        if let (Some(action), Some(phase)) = (action, phase) {
            let prepared = self.load(action)?;
            let name = Self::polling_name(&prepared)?;
            let mut samples = self.polling_samples(&prepared)?;
            let phase = format!("{phase:?}");
            let previous = samples.phases.get(&phase);
            let failures = if code.is_none() {
                0
            } else {
                previous.map_or(1, |sample| sample.failures.saturating_add(1))
            };
            let last_success_at = if code.is_none() {
                Some(at.clone())
            } else {
                previous.and_then(|sample| sample.last_success_at.clone())
            };
            samples.phases.insert(
                phase,
                Sample {
                    at: at.clone(),
                    error: code.map(str::to_owned),
                    failures,
                    last_success_at,
                },
            );
            self.slots
                .replace_regular(
                    OsStr::new(&name),
                    &serde_json::to_vec(&samples).map_err(|_| Error::Configuration)?,
                )
                .map_err(|_| Error::Uncertain)?;
        }
        self.directory.replace_regular(OsStr::new("observer.json"),&serde_json::to_vec(&json!({"status":if code.is_some(){"RETRY_PENDING"}else if action.is_some(){"OBSERVING"}else{"IDLE"},"at":at,"submission_id":action,"phase":phase.map(|p|format!("{p:?}")),"error":code,"provider_mutations":false,"heartbeat_proven":false})).map_err(|_|Error::Configuration)?).map_err(|_|Error::Uncertain)
    }
    pub fn background_stopped(&self) -> Result<()> {
        let (at, _) = crate::projects::transition_context().map_err(|_| Error::Configuration)?;
        self.directory.replace_regular(OsStr::new("observer.json"),&serde_json::to_vec(&json!({"status":"STOPPED","at":at,"provider_mutations":false,"heartbeat_proven":false})).map_err(|_|Error::Configuration)?).map_err(|_|Error::Uncertain)
    }
    fn polling_name(prepared: &Prepared) -> Result<String> {
        Ok(format!(
            "{}.polling.json",
            native::attempt_slot(
                &prepared.binding.project,
                &prepared.binding.run_id,
                &prepared.binding.attempt_id
            )?
        ))
    }
    fn polling_samples(&self, prepared: &Prepared) -> Result<Samples> {
        let name = Self::polling_name(prepared)?;
        let request_sha256 = prepared
            .request
            .identity()
            .map_err(|_| Error::Authority)?
            .request_sha256;
        if !self
            .slots
            .entry_exists(OsStr::new(&name))
            .map_err(|_| Error::Authority)?
        {
            return Ok(Samples {
                request_sha256,
                ..Default::default()
            });
        }
        let bytes = self
            .slots
            .read_regular(OsStr::new(&name), 16384)
            .map_err(|_| Error::Authority)?;
        let samples: Samples = serde_json::from_value(Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Authority)?,
        ))
        .map_err(|_| Error::Authority)?;
        if samples.request_sha256 != request_sha256
            || samples.phases.len() > 3
            || samples
                .phases
                .keys()
                .any(|key| !matches!(key.as_str(), "Scheduler" | "Worker" | "History"))
        {
            return Err(Error::Authority);
        }
        Ok(samples)
    }
    pub(super) fn polling_status(&self, prepared: &Prepared) -> Result<Value> {
        Ok(json!({"phases":self.polling_samples(prepared)?.phases,"provider_mutations":false}))
    }
    pub fn background_status(&self) -> Result<Value> {
        let name = OsStr::new("observer.json");
        if !self
            .directory
            .entry_exists(name)
            .map_err(|_| Error::Authority)?
        {
            return Ok(json!({"status":"NOT_STARTED"}));
        }
        let bytes = self
            .directory
            .read_regular(name, 65536)
            .map_err(|_| Error::Authority)?;
        Ok(Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Authority)?,
        ))
    }
    pub fn background_candidates(&self, maximum: usize) -> Result<BTreeSet<String>> {
        let mut actions = BTreeSet::new();
        for project in &self.settings.projects {
            if !project.discovery_enabled() {
                continue;
            }
            if self.lifecycle.as_ref().is_some_and(|service| {
                service
                    .active_projects()
                    .is_ok_and(|active| !active.contains(&project.project))
            }) {
                continue;
            }
            let discovery = self
                .evidence
                .discover(&project.project, &experiments::DiscoveryLimits::default())
                .map_err(|_| Error::Authority)?;
            for run in discovery.runs {
                let snapshot = self
                    .evidence
                    .snapshot_metadata(&run.reference)
                    .map_err(|_| Error::Authority)?;
                for attempt in snapshot.attempts {
                    let Some(prepared) = self.saved_attempt(
                        &project.project,
                        &snapshot.run_id,
                        &attempt.attempt_id,
                    )?
                    else {
                        continue;
                    };
                    let action = prepared.plan["action_id"]
                        .as_str()
                        .ok_or(Error::Authority)?;
                    if self.actions.observe(action)?["execution"]["status"] != "VERIFIED" {
                        continue;
                    }
                    // A terminal scheduler receipt is independent of process and
                    // history completion. All three must be complete to retire.
                    let terminal = self
                        .scheduler_observation(&prepared)?
                        .is_some_and(|saved| saved.observation.status.state.is_terminal());
                    let managed = Self::background_managed(&prepared)?;
                    if terminal && !managed {
                        continue;
                    }
                    let finished = self.observation(&prepared)?.is_some_and(|saved| {
                        saved.evidence.worker.phase
                            == ml_exp_core::managed::evidence::Phase::Finished
                    });
                    if terminal
                        && finished
                        && self.history_complete(&prepared)?
                        && self.journal_terminal_recorded(&prepared)
                    {
                        continue;
                    }
                    actions.insert(action.into());
                    if actions.len() > maximum {
                        return Err(Error::Configuration);
                    }
                }
            }
        }
        Ok(actions)
    }
    pub fn background_sample(
        &self,
        action: &str,
        phase: Phase,
        now: &str,
        cancelled: &AtomicBool,
        timeout: std::time::Duration,
    ) -> Result<Value> {
        let prepared = self.load(action)?;
        let phase = if Self::background_managed(&prepared)? {
            phase
        } else {
            Phase::Scheduler
        };
        match phase {
            Phase::Scheduler => {
                self.refresh_scheduler_with_limit(action, now, cancelled, Some(timeout))
            }
            Phase::Worker => self.refresh_with_limit(action, now, cancelled, Some(timeout)),
            Phase::History => self.ingest_history_with_limit(action, cancelled, Some(timeout)),
        }
    }
    fn background_managed(prepared: &Prepared) -> Result<bool> {
        let (manifest, _) = prepared
            .request
            .prepared_content()
            .ok_or(Error::Authority)?;
        let plan =
            ml_exp_core::slurm::staging::plan(prepared.request.intent(), manifest.as_bytes())
                .map_err(|_| Error::Authority)?;
        Ok(plan.is_some_and(|plan| plan.managed.is_some()))
    }
    pub(super) fn observation_ssh(
        backend: &Backend,
        limit: Option<std::time::Duration>,
    ) -> SshConfig {
        let mut ssh = backend.control_ssh();
        if let Some(limit) = limit {
            ssh.budget.timeout = ssh.budget.timeout.min(limit);
            ssh.budget.connect_timeout_seconds = ssh
                .budget
                .connect_timeout_seconds
                .min(ssh.budget.timeout.as_secs().max(1) as u32);
        }
        ssh
    }
}
