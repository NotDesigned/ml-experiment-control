//! The purge of an ended Attempt's `state/` on backend storage, once every checkpoint it
//! registered has a verified object-store copy, which is then that checkpoint's
//! authority. Nothing is removed here: the reviewed storage-maintenance plan for `state`
//! runs as the backend's CPU job with the Attempt's own capability, and its receipt is
//! kept beside the Attempt. The plan's identity decides the job, so a repeated request or
//! a restart observes the same job and never makes a second one.
use super::*;
use ml_exp_core::maintenance::{Plan, Selection};

const CONTRACT: &str = "native-state-purge.v1";
const LEADER: &str = ".state-purge.lock";
/// How often the automatic purge looks for Attempts that became eligible.
const PASS_SECONDS: u64 = 60;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Purge {
    contract: String,
    /// Starts at 0. Only an explicit request after a job that is proven to have failed
    /// makes the next one, with a new plan and so a new job.
    generation: u32,
    plan: Plan,
    /// `automatic` or `explicit`: who asked for this generation.
    trigger: String,
    /// When every condition was first seen to hold.
    eligible_at: String,
    /// The checkpoints whose verified copies made this purge safe.
    checkpoints: Vec<String>,
    /// Written before the job is asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    submitted_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    receipt: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completed_at: Option<String>,
}

fn io_json(_: serde_json::Error) -> cloud::Error {
    cloud::Error::evidence()
}
fn seconds(now: &str) -> cloud::Result<i64> {
    Ok(
        time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
            .map_err(why!(cloud::Error::configuration()))?
            .unix_timestamp(),
    )
}

impl Service {
    fn purge_name(prepared: &Prepared) -> cloud::Result<String> {
        let job = &prepared.execution.job;
        Ok(format!(
            "{}.state-purge.json",
            Self::attempt_slot(&job.project, &job.run_id, &job.attempt_id)?
        ))
    }
    pub(super) fn purge_saved(&self, prepared: &Prepared) -> cloud::Result<Option<Purge>> {
        let name = Self::purge_name(prepared)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(None);
        }
        let saved: Purge = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(why!(cloud::Error::evidence()))?;
        let package = prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        if saved.contract != CONTRACT
            || saved.plan != self.purge_plan(prepared, saved.generation)?
            || saved.plan.config != package.config
            || saved
                .receipt
                .as_ref()
                .is_some_and(|r| saved.plan.receipt().ok().as_ref() != Some(r))
        {
            return Err(cloud::Error::evidence());
        }
        Ok(Some(saved))
    }
    fn purge_save(&self, prepared: &Prepared, saved: &Purge) -> cloud::Result<()> {
        self.slots
            .replace_regular(OsStr::new(&Self::purge_name(prepared)?), &encode(saved)?)
            .map_err(why!(cloud::Error {
                code: "STATE_PURGE_COMMIT_UNCERTAIN".into(),
                http_status: None,
                uncertain: true,
            }))
    }
    fn purge_plan(&self, prepared: &Prepared, generation: u32) -> cloud::Result<Plan> {
        let job = &prepared.execution.job;
        let request = canonical::digest(&json!([
            CONTRACT,
            job.project,
            job.run_id,
            job.attempt_id,
            generation
        ]))
        .map_err(why!(cloud::Error::evidence()))?;
        let plan = Plan {
            contract: "native-storage-maintenance.v1".into(),
            config: prepared
                .execution
                .package()
                .map_err(why!(cloud::Error::evidence()))?
                .config,
            selection: Selection::State,
            request_id: request[..32].into(),
        };
        plan.validate().map_err(why!(cloud::Error::evidence()))?;
        Ok(plan)
    }
    /// Everything that still stands in the way, and the checkpoints whose verified
    /// copies the purge relies on. Only local evidence is read: the observer keeps the
    /// scheduler observation current, and the worker reports its own receipts.
    fn purge_blocked(
        &self,
        prepared: &Prepared,
        explicit: bool,
        accept_unfinished: bool,
    ) -> cloud::Result<(Vec<&'static str>, Vec<String>)> {
        let mut blocked = Vec::new();
        let package = prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        if !package.checkpoint_archive && !explicit {
            blocked.push("CHECKPOINT_ARCHIVE_NOT_ENABLED");
        }
        let terminal = match self.outbox.optional_status(&prepared.execution.job)? {
            Some(state) => super::sync_jobs::proven_terminal(&state)?.is_some(),
            None => false,
        };
        if !terminal {
            blocked.push("ATTEMPT_NOT_TERMINAL");
        }
        let exchange = self.exchange_saved(prepared)?;
        let worker = exchange.as_ref().map(|s| &s.evidence.worker);
        let finished = worker.is_some_and(|w| {
            w.phase == ml_exp_core::managed::evidence::Phase::Finished && w.receipt.is_some()
        });
        // A worker that never reported its end, as after a preemption, may have written
        // a generation it never registered. Only a request that says so accepts that.
        if !finished && !(explicit && accept_unfinished) {
            blocked.push("WORKER_NOT_FINISHED");
        }
        if exchange
            .as_ref()
            .is_some_and(|s| s.checkpoint_error.is_some())
        {
            blocked.push("LAST_CHECKPOINT_UNVERIFIED");
        }
        let completed = worker
            .and_then(|w| w.receipt.as_ref())
            .is_some_and(|r| r.training == ml_exp_core::managed::evidence::Training::Completed);
        if package.result_upload && completed {
            let published = self
                .artifact_saved(prepared)?
                .is_some_and(|s| s.state.is_some() || s.administrative_release.is_some());
            if !published {
                blocked.push("RESULTS_NOT_PUBLISHED");
            }
        }
        let rows = if package.checkpoint_context.is_some() {
            self.archive_rows(prepared)?
        } else {
            Vec::new()
        };
        if rows.iter().any(|(_, receipt)| receipt.is_none()) {
            blocked.push("CHECKPOINT_NOT_ARCHIVED");
        }
        let checkpoints = rows
            .iter()
            .map(|(document, _)| {
                document["checkpoint_id"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(cloud::Error::evidence)
            })
            .collect::<cloud::Result<Vec<_>>>()?;
        Ok((blocked, checkpoints))
    }
    fn purge_view(&self, prepared: &Prepared, blocked: &[&str]) -> cloud::Result<Value> {
        let job = &prepared.execution.job;
        let saved = self.purge_saved(prepared)?;
        let status = match &saved {
            Some(s) if s.receipt.is_some() => "PURGED",
            Some(s) if s.submitted_at.is_some() => {
                match self
                    .sync_outbox
                    .optional_status(&self.maintenance_job(&prepared.execution, &s.plan)?)?
                {
                    Some(state)
                        if super::sync_jobs::proven_terminal(&state)?.is_some_and(|o| {
                            o.state != ml_exp_core::state::RunState::Succeeded
                        }) =>
                    {
                        "PURGE_JOB_FAILED"
                    }
                    _ => "PURGING",
                }
            }
            Some(_) if blocked.is_empty() => "WAITING",
            _ if blocked.is_empty() => "ELIGIBLE",
            _ => "BLOCKED",
        };
        Ok(json!({
            "project": job.project,
            "run_id": job.run_id,
            "attempt_id": job.attempt_id,
            "status": status,
            "blocked_by": blocked,
            "purge_delay_seconds": self.archive.purge_delay_seconds,
            "path": "state",
            "plan_id": saved.as_ref().map(|s| s.plan.id()).transpose().map_err(why!(cloud::Error::evidence()))?,
            "generation": saved.as_ref().map(|s| s.generation),
            "trigger": saved.as_ref().map(|s| s.trigger.clone()),
            "eligible_at": saved.as_ref().map(|s| s.eligible_at.clone()),
            "submitted_at": saved.as_ref().and_then(|s| s.submitted_at.clone()),
            "completed_at": saved.as_ref().and_then(|s| s.completed_at.clone()),
            "checkpoints_on_object_store": saved.as_ref().map(|s| s.checkpoints.clone()),
            "receipt": saved.as_ref().and_then(|s| s.receipt.clone()),
            "outputs_removed": false,
            "checkpoint_records_preserved": true,
        }))
    }
    /// The purge record of one Attempt, if it has one, for its checkpoint listing.
    pub(super) fn purge_summary(&self, prepared: &Prepared) -> cloud::Result<Value> {
        Ok(match self.purge_saved(prepared)? {
            None => Value::Null,
            Some(s) => json!({
                "status": if s.receipt.is_some() { "PURGED" } else if s.submitted_at.is_some() { "PURGING" } else { "WAITING" },
                "plan_id": s.plan.id().map_err(why!(cloud::Error::evidence()))?,
                "trigger": s.trigger,
                "eligible_at": s.eligible_at,
                "submitted_at": s.submitted_at,
                "completed_at": s.completed_at,
            }),
        })
    }
    /// Ask for the job once the record that names it is durable. A repeated call
    /// observes the job the plan already has.
    fn purge_submit(&self, prepared: &Prepared, saved: &Purge) -> cloud::Result<()> {
        let action = prepared.plan["action_id"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        self.maintenance_execute(action, &saved.plan).map(|_| ())
    }
    /// `ml-exp run purge-state`: the same conditions as the automatic purge, without its
    /// delay. A blocked request changes nothing and says what blocks it. Only
    /// `accept_unfinished` lets it purge an Attempt whose worker never reported its end.
    pub fn purge_state(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        accept_unfinished: bool,
        now: &str,
    ) -> cloud::Result<Value> {
        if !self.enabled {
            return Err(cloud::Error::configuration());
        }
        let lock = self.lock("state_purge.purge_state")?;
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(missing)?;
        self.require_live(project, run)?;
        let (blocked, checkpoints) = self.purge_blocked(&prepared, true, accept_unfinished)?;
        let saved = self.purge_saved(&prepared)?;
        let next = match saved {
            Some(s) if s.receipt.is_some() => {
                drop(lock);
                return self.purge_view(&prepared, &[]);
            }
            Some(s) if s.submitted_at.is_some() => {
                let state = self
                    .sync_outbox
                    .optional_status(&self.maintenance_job(&prepared.execution, &s.plan)?)?;
                let failed = match &state {
                    Some(state) => super::sync_jobs::proven_terminal(state)?
                        .is_some_and(|o| o.state != ml_exp_core::state::RunState::Succeeded),
                    None => false,
                };
                if !failed {
                    // The same job again: an interrupted request finishes, an existing
                    // job is observed.
                    drop(lock);
                    self.purge_submit(&prepared, &s)?;
                    return self.purge_view(&prepared, &[]);
                }
                s.generation
                    .checked_add(1)
                    .ok_or_else(cloud::Error::evidence)?
            }
            Some(s) => s.generation,
            None => 0,
        };
        if !blocked.is_empty() {
            drop(lock);
            return self.purge_view(&prepared, &blocked);
        }
        let record = Purge {
            contract: CONTRACT.into(),
            generation: next,
            plan: self.purge_plan(&prepared, next)?,
            trigger: "explicit".into(),
            eligible_at: now.into(),
            checkpoints,
            submitted_at: Some(now.into()),
            receipt: None,
            completed_at: None,
        };
        self.purge_save(&prepared, &record)?;
        drop(lock);
        self.purge_submit(&prepared, &record)?;
        self.purge_view(&prepared, &[])
    }
    /// One look at one Attempt by the automatic purge.
    fn purge_automatically(&self, prepared: &Prepared, now: &str) -> cloud::Result<()> {
        let lock = self.lock("state_purge.purge_automatically")?;
        let saved = self.purge_saved(prepared)?;
        if saved.is_some_and(|s| s.submitted_at.is_some()) {
            return Ok(());
        }
        let (blocked, checkpoints) = self.purge_blocked(prepared, false, false)?;
        if !blocked.is_empty() {
            return Ok(());
        }
        let mut record = match self.purge_saved(prepared)? {
            Some(record) => record,
            None => {
                let record = Purge {
                    contract: CONTRACT.into(),
                    generation: 0,
                    plan: self.purge_plan(prepared, 0)?,
                    trigger: "automatic".into(),
                    eligible_at: now.into(),
                    checkpoints: checkpoints.clone(),
                    submitted_at: None,
                    receipt: None,
                    completed_at: None,
                };
                self.purge_save(prepared, &record)?;
                record
            }
        };
        let due = seconds(&record.eligible_at)?
            .saturating_add(self.archive.purge_delay_seconds.min(i64::MAX as u64) as i64);
        if seconds(now)? < due {
            return Ok(());
        }
        record.checkpoints = checkpoints;
        record.submitted_at = Some(now.into());
        self.purge_save(prepared, &record)?;
        drop(lock);
        self.purge_submit(prepared, &record)
    }
    /// Every Attempt whose Run copies checkpoints, looked at once. One Attempt's error
    /// never stops the others; its record says how far it got.
    pub fn purge_pass(&self, now: &str) -> cloud::Result<()> {
        if !self.enabled || !self.archive.automatic_purge || !self.has_collectors() {
            return Ok(());
        }
        for name in self.slots.entry_names().map_err(io)? {
            let Some(name) = name.to_str() else {
                continue;
            };
            if !super::super::observer::preparation_slot(name) {
                continue;
            }
            let Ok(prepared) = read(&self.slots, name)
                .and_then(|v| serde_json::from_value::<Prepared>(v).map_err(io_json))
            else {
                continue;
            };
            let b = &prepared.execution.binding;
            let Ok(Some(prepared)) = self.saved_attempt(&b.project, &b.run_id, &b.attempt_id)
            else {
                continue;
            };
            if !prepared
                .execution
                .package()
                .is_ok_and(|p| p.checkpoint_archive)
                || self.require_live(&b.project, &b.run_id).is_err()
            {
                continue;
            }
            if let Err(error) = self.purge_automatically(&prepared, now) {
                eprintln!(
                    "automatic state purge of {}/{}/{} deferred: {}",
                    b.project, b.run_id, b.attempt_id, error.code
                );
            }
        }
        Ok(())
    }
    /// The receipt of a purge job, if this Attempt's purge plan is the one it names.
    pub fn receive_purge(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
        receipt: &Value,
        now: &str,
    ) -> cloud::Result<Option<Value>> {
        let _lock = self.lock("state_purge.receive_purge")?;
        let Some(prepared) = self.saved_attempt(project, run, attempt)? else {
            return Ok(None);
        };
        let Some(mut saved) = self.purge_saved(&prepared)? else {
            return Ok(None);
        };
        let id = saved.plan.id().map_err(why!(cloud::Error::evidence()))?;
        if receipt["plan_id"] != id.as_str() {
            return Ok(None);
        }
        let action = prepared.plan["action_id"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        self.maintenance_auth(action, &saved.plan, token)?;
        if saved
            .plan
            .receipt()
            .map_err(why!(cloud::Error::evidence()))?
            != *receipt
        {
            return Err(cloud::Error::evidence());
        }
        if saved.receipt.is_none() {
            saved.receipt = Some(receipt.clone());
            saved.completed_at = Some(now.into());
            self.purge_save(&prepared, &saved)?;
        }
        Ok(Some(json!({"plan_id": id, "status": "RECEIVED"})))
    }
    /// One thread that looks for eligible Attempts, when the automatic purge is on.
    /// Another server holding the leader lock is left to do it.
    pub fn start_purger(self: Arc<Self>) -> Option<crate::background::Worker> {
        if !self.enabled || !self.archive.automatic_purge || !self.has_collectors() {
            return None;
        }
        let leader = self
            .directory
            .lock(
                OsStr::new(LEADER),
                LockOptions {
                    create: true,
                    ..Default::default()
                },
            )
            .ok()?;
        crate::background::Worker::spawn("native-state-purge", move |stop| {
            let _leader = leader;
            while !stop.requested() {
                if let Ok((now, _)) = crate::projects::transition_context() {
                    let _ = self.purge_pass(&now);
                }
                for _ in 0..PASS_SECONDS {
                    if !stop.pause() {
                        return;
                    }
                }
            }
        })
        .ok()
    }
}
