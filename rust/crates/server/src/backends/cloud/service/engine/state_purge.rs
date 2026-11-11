//! The purge of an ended Attempt's `state/` on backend storage, once every checkpoint it
//! registered has a verified object-store copy, which is then that checkpoint's
//! authority. Nothing is removed here: the reviewed storage-maintenance plan for `state`
//! runs as the backend's CPU job with the Attempt's own capability, and its receipt is
//! kept beside the Attempt. The plan's identity decides the job, so a repeated request or
//! a restart observes the same job and never makes a second one. A further job runs the
//! same plan only after the last is proven to have failed: the worker then resumes the
//! targets it retained, or returns the receipt it already has.
use super::*;
use ml_exp_core::{
    cloud::Job,
    maintenance::{Plan, Selection},
    state::RunState,
};

const CONTRACT: &str = "native-state-purge.v1";
const LEADER: &str = ".state-purge.lock";
/// How often the automatic purge looks for Attempts that became eligible.
const PASS_SECONDS: u64 = 60;
/// Jobs the automatic purge runs for one plan; an explicit request may run more.
const AUTOMATIC_JOBS: u32 = 3;
const EXPLICIT_JOBS: u32 = 10;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Purge {
    contract: String,
    plan: Plan,
    /// `automatic` or `explicit`: who asked first.
    trigger: String,
    /// When every condition was seen to hold, until one stops holding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eligible_at: Option<String>,
    /// The checkpoints whose verified copies made this purge safe.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    checkpoints: Vec<String>,
    /// Written before the job is asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    submitted_at: Option<String>,
    /// The job running the plan: 0 for its first job.
    #[serde(default, skip_serializing_if = "is_zero")]
    job_attempt: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    receipt: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completed_at: Option<String>,
}
fn is_zero(value: &u32) -> bool {
    *value == 0
}

/// The sync jobs asked for an Attempt, and the sync that delivered an unfinished
/// worker's state, beside the Attempt.
#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Syncs {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    jobs: Vec<Job>,
    /// When a sync job of an Attempt whose worker never reported its end said it
    /// delivered the current state with every copy verified, after training ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unfinished_delivered_at: Option<String>,
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
/// The job's state, if proven ended.
fn ended(state: Option<Value>) -> cloud::Result<Option<RunState>> {
    Ok(match state {
        // An entry not observed yet proves nothing; proven_terminal returns None.
        Some(state) => super::sync_jobs::proven_terminal(&state)?.map(|o| o.state),
        None => None,
    })
}

/// What a purge decision reads, gathered from local evidence only.
#[derive(Clone, Copy, Debug)]
struct Evidence {
    /// The Run copies checkpoints to the object store.
    archive_enabled: bool,
    /// The training job is proven to have ended.
    training_ended: bool,
    /// A recorded sync job of the Attempt is not proven to have ended.
    sync_running: bool,
    /// The worker reported its end with a receipt.
    worker_finished: bool,
    /// A sync of an unfinished worker delivered its state after training ended.
    unfinished_sync_delivered: bool,
    /// The last ready marker failed its verification.
    last_checkpoint_unverified: bool,
    /// Completed training whose results are not yet published.
    results_pending: bool,
    /// A registered checkpoint has no verified copy.
    checkpoint_unarchived: bool,
}
impl Evidence {
    /// Facts not read yet, taken as holding, so only the ones read can block.
    fn first(archive_enabled: bool, training_ended: bool) -> Self {
        Self {
            archive_enabled,
            training_ended,
            sync_running: false,
            worker_finished: true,
            unfinished_sync_delivered: true,
            last_checkpoint_unverified: false,
            results_pending: false,
            checkpoint_unarchived: false,
        }
    }
}
/// Everything that stands in the way of a purge, in a fixed order. The same rules for
/// both paths, except that only an explicit request may accept an unfinished worker.
fn blocked_by(e: &Evidence, explicit: bool, accept_unfinished: bool) -> Vec<&'static str> {
    let mut blocked = Vec::new();
    if !e.archive_enabled {
        blocked.push("CHECKPOINT_ARCHIVE_NOT_ENABLED");
    }
    if !e.training_ended {
        blocked.push("ATTEMPT_NOT_TERMINAL");
    }
    if e.sync_running {
        blocked.push("SYNC_IN_PROGRESS");
    }
    if !e.worker_finished {
        if !(explicit && accept_unfinished) {
            blocked.push("WORKER_NOT_FINISHED");
        } else if !e.unfinished_sync_delivered {
            blocked.push("SYNC_REQUIRED");
        }
    }
    if e.last_checkpoint_unverified {
        blocked.push("LAST_CHECKPOINT_UNVERIFIED");
    }
    if e.results_pending {
        blocked.push("RESULTS_NOT_PUBLISHED");
    }
    if e.checkpoint_unarchived {
        blocked.push("CHECKPOINT_NOT_ARCHIVED");
    }
    blocked
}
/// A `synced` report is kept only once training has ended and every copy exists.
fn accepts_unfinished_sync(training_ended: bool, every_copy_verified: bool) -> bool {
    training_ended && every_copy_verified
}
#[derive(Debug, PartialEq, Eq)]
enum AfterJob {
    /// Observe the job again on the next look; it is running, succeeded or unknown.
    Observe,
    /// The job is proven to have failed: run the same plan in a further job.
    Further,
    /// It failed and the limit of jobs for this plan is reached.
    GiveUp,
}
fn after_job(state: Option<RunState>, job_attempt: u32, limit: u32) -> AfterJob {
    match state {
        None | Some(RunState::Succeeded) => AfterJob::Observe,
        // Never a further job beside one that may still run.
        Some(state) if !state.is_terminal() => AfterJob::Observe,
        Some(_) if job_attempt.saturating_add(1) >= limit => AfterJob::GiveUp,
        Some(_) => AfterJob::Further,
    }
}
#[derive(Debug, PartialEq, Eq)]
enum Automatic {
    Nothing,
    /// A condition stopped holding before the purge started.
    ResetEligible,
    /// Every condition holds for the first time: the delay starts now.
    StartDelay,
    Submit,
}
fn automatic(blocked: bool, eligible_at: Option<i64>, now: i64, delay: u64) -> Automatic {
    let delay = delay.min(i64::MAX as u64) as i64;
    match (blocked, eligible_at) {
        (true, Some(_)) => Automatic::ResetEligible,
        (true, None) => Automatic::Nothing,
        (false, None) if delay == 0 => Automatic::Submit,
        (false, None) => Automatic::StartDelay,
        (false, Some(at)) if now >= at.saturating_add(delay) => Automatic::Submit,
        (false, Some(_)) => Automatic::Nothing,
    }
}
/// A new Run that reads a generation in place may not read one from a Run whose
/// Attempts' state/ is purged once copied.
pub(super) fn resume_refused(new_run: bool, copies: bool, source_copies: bool) -> bool {
    new_run && !copies && source_copies
}

impl Service {
    fn purge_name(prepared: &Prepared) -> cloud::Result<String> {
        let job = &prepared.execution.job;
        Ok(format!(
            "{}.state-purge.json",
            Self::attempt_slot(&job.project, &job.run_id, &job.attempt_id)?
        ))
    }
    fn syncs_name(prepared: &Prepared) -> cloud::Result<String> {
        let job = &prepared.execution.job;
        Ok(format!(
            "{}.syncs.json",
            Self::attempt_slot(&job.project, &job.run_id, &job.attempt_id)?
        ))
    }
    fn syncs_saved(&self, prepared: &Prepared) -> cloud::Result<Syncs> {
        let name = Self::syncs_name(prepared)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(Syncs::default());
        }
        let saved: Syncs = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(why!(cloud::Error::evidence()))?;
        let parent = &prepared.execution.job;
        if saved.jobs.iter().any(|job| {
            job.project != parent.project
                || job.attempt_id != parent.attempt_id
                || !job.run_id.starts_with("sync-")
        }) {
            return Err(cloud::Error::evidence());
        }
        Ok(saved)
    }
    fn syncs_save(&self, prepared: &Prepared, saved: &Syncs) -> cloud::Result<()> {
        self.slots
            .replace_regular(OsStr::new(&Self::syncs_name(prepared)?), &encode(saved)?)
            .map_err(io)
    }
    /// Kept before a sync job is asked for, under the service lock, so a purge can see
    /// every sync of the Attempt.
    pub(super) fn record_sync_job(&self, prepared: &Prepared, job: &Job) -> cloud::Result<()> {
        let mut saved = self.syncs_saved(prepared)?;
        if !saved.jobs.contains(job) {
            saved.jobs.push(job.clone());
            self.syncs_save(prepared, &saved)?;
        }
        Ok(())
    }
    /// A sync of an Attempt whose worker never reported its end delivered the current
    /// state and has a verified copy of every checkpoint. Kept only after training ended.
    pub(super) fn record_unfinished_sync(
        &self,
        prepared: &Prepared,
        now: &str,
    ) -> cloud::Result<Value> {
        let _lock = self.lock("state_purge.record_unfinished_sync")?;
        let terminal = ended(self.outbox.optional_status(&prepared.execution.job)?)?.is_some();
        let archived = self
            .archive_rows(prepared)?
            .iter()
            .all(|(_, receipt)| receipt.is_some());
        if !accepts_unfinished_sync(terminal, archived) {
            return Ok(json!({"recorded": false}));
        }
        let mut saved = self.syncs_saved(prepared)?;
        saved.unfinished_delivered_at = Some(now.into());
        self.syncs_save(prepared, &saved)?;
        Ok(json!({"recorded": true}))
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
            || saved.plan != self.purge_plan(prepared)?
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
    fn purge_plan(&self, prepared: &Prepared) -> cloud::Result<Plan> {
        let job = &prepared.execution.job;
        let request =
            canonical::digest(&json!([CONTRACT, job.project, job.run_id, job.attempt_id]))
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
    fn purge_job_state(&self, prepared: &Prepared, saved: &Purge) -> cloud::Result<Option<Value>> {
        self.sync_outbox.optional_status(&self.maintenance_job(
            &prepared.execution,
            &saved.plan,
            saved.job_attempt,
        )?)
    }
    /// Everything that still stands in the way, and the checkpoints whose verified
    /// copies the purge relies on. Only local evidence is read. The automatic purge
    /// stops at the first obstacle, so an Attempt still training costs one read.
    fn purge_blocked(
        &self,
        prepared: &Prepared,
        explicit: bool,
        accept_unfinished: bool,
    ) -> cloud::Result<(Vec<&'static str>, Vec<String>)> {
        let package = prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        // Only a Run that copies checkpoints is purged, by either path.
        let archive_enabled = package.checkpoint_archive && package.checkpoint_context.is_some();
        let training_ended =
            ended(self.outbox.optional_status(&prepared.execution.job)?)?.is_some();
        let first = blocked_by(
            &Evidence::first(archive_enabled, training_ended),
            explicit,
            accept_unfinished,
        );
        if !explicit && !first.is_empty() {
            return Ok((first, Vec::new()));
        }
        // A sync may still register and copy the last ready generation.
        let syncs = self.syncs_saved(prepared)?;
        let mut sync_running = false;
        for job in &syncs.jobs {
            if ended(self.sync_outbox.optional_status(job)?)?.is_none() {
                sync_running = true;
                break;
            }
        }
        let exchange = self.exchange_saved(prepared)?;
        let worker = exchange.as_ref().map(|s| &s.evidence.worker);
        // A worker that never reported its end, as after a preemption, may have written
        // a generation it never registered.
        let worker_finished = worker.is_some_and(|w| {
            w.phase == ml_exp_core::managed::evidence::Phase::Finished && w.receipt.is_some()
        });
        let completed = worker
            .and_then(|w| w.receipt.as_ref())
            .is_some_and(|r| r.training == ml_exp_core::managed::evidence::Training::Completed);
        let results_pending = package.result_upload
            && completed
            && !self
                .artifact_saved(prepared)?
                .is_some_and(|s| s.state.is_some() || s.administrative_release.is_some());
        let mut evidence = Evidence {
            archive_enabled,
            training_ended,
            sync_running,
            worker_finished,
            unfinished_sync_delivered: syncs.unfinished_delivered_at.is_some(),
            last_checkpoint_unverified: exchange
                .as_ref()
                .is_some_and(|s| s.checkpoint_error.is_some()),
            results_pending,
            checkpoint_unarchived: false,
        };
        let blocked = blocked_by(&evidence, explicit, accept_unfinished);
        if !explicit && !blocked.is_empty() {
            return Ok((blocked, Vec::new()));
        }
        let rows = if package.checkpoint_context.is_some() {
            self.archive_rows(prepared)?
        } else {
            Vec::new()
        };
        evidence.checkpoint_unarchived = rows.iter().any(|(_, receipt)| receipt.is_none());
        let blocked = blocked_by(&evidence, explicit, accept_unfinished);
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
                match ended(self.purge_job_state(prepared, s)?)? {
                    Some(state) if state != RunState::Succeeded => "PURGE_JOB_FAILED",
                    _ => "PURGING",
                }
            }
            Some(s) if blocked.is_empty() && s.eligible_at.is_some() => "WAITING",
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
            "job_attempt": saved.as_ref().map(|s| s.job_attempt),
            "trigger": saved.as_ref().map(|s| s.trigger.clone()),
            "eligible_at": saved.as_ref().and_then(|s| s.eligible_at.clone()),
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
                "status": if s.receipt.is_some() { "PURGED" } else if s.submitted_at.is_some() { "PURGING" } else if s.eligible_at.is_some() { "WAITING" } else { "BLOCKED" },
                "plan_id": s.plan.id().map_err(why!(cloud::Error::evidence()))?,
                "trigger": s.trigger,
                "eligible_at": s.eligible_at,
                "submitted_at": s.submitted_at,
                "job_attempt": s.job_attempt,
                "completed_at": s.completed_at,
            }),
        })
    }
    /// Ask for the record's job, or observe it. Repeating it is safe: the outbox sends
    /// a job that was never sent and otherwise reconciles the one there is.
    fn purge_submit(&self, prepared: &Prepared, saved: &Purge) -> cloud::Result<()> {
        let action = prepared.plan["action_id"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        self.maintenance_execute_attempt(action, &saved.plan, saved.job_attempt)
            .map(|_| ())
    }
    /// A record that asked for a job and has no receipt: observe the job, and after one
    /// that is proven to have failed, run the same plan in a further job.
    fn purge_follow_up(&self, prepared: &Prepared, limit: u32) -> cloud::Result<()> {
        let Some(saved) = self.purge_saved(prepared)? else {
            return Ok(());
        };
        if saved.receipt.is_some() || saved.submitted_at.is_none() {
            return Ok(());
        }
        // A crash between the record and the request leaves no job: this sends it.
        self.purge_submit(prepared, &saved)?;
        match after_job(
            ended(self.purge_job_state(prepared, &saved)?)?,
            saved.job_attempt,
            limit,
        ) {
            AfterJob::Observe | AfterJob::GiveUp => Ok(()),
            AfterJob::Further => {
                let lock = self.lock("state_purge.purge_follow_up")?;
                let Some(mut current) = self.purge_saved(prepared)? else {
                    return Ok(());
                };
                if current.job_attempt != saved.job_attempt || current.receipt.is_some() {
                    return Ok(());
                }
                current.job_attempt += 1;
                self.purge_save(prepared, &current)?;
                drop(lock);
                self.purge_submit(prepared, &current)
            }
        }
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
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(missing)?;
        self.require_live(project, run)?;
        if let Some(saved) = self.purge_saved(&prepared)? {
            if saved.receipt.is_some() {
                return self.purge_view(&prepared, &[]);
            }
            if saved.submitted_at.is_some() {
                self.purge_follow_up(&prepared, EXPLICIT_JOBS)?;
                return self.purge_view(&prepared, &[]);
            }
        }
        let lock = self.lock("state_purge.purge_state")?;
        let (blocked, checkpoints) = self.purge_blocked(&prepared, true, accept_unfinished)?;
        if !blocked.is_empty() {
            drop(lock);
            return self.purge_view(&prepared, &blocked);
        }
        let mut record = match self.purge_saved(&prepared)? {
            Some(record) if record.submitted_at.is_some() => {
                drop(lock);
                return self.purge_view(&prepared, &[]);
            }
            Some(record) => record,
            None => Purge {
                contract: CONTRACT.into(),
                plan: self.purge_plan(&prepared)?,
                trigger: "explicit".into(),
                eligible_at: None,
                checkpoints: Vec::new(),
                submitted_at: None,
                job_attempt: 0,
                receipt: None,
                completed_at: None,
            },
        };
        record.eligible_at.get_or_insert_with(|| now.into());
        record.checkpoints = checkpoints;
        record.submitted_at = Some(now.into());
        self.purge_save(&prepared, &record)?;
        drop(lock);
        self.purge_submit(&prepared, &record)?;
        self.purge_view(&prepared, &[])
    }
    /// One look at one Attempt by the automatic purge. Conditions are read without the
    /// service lock; it is taken only to write the record.
    fn purge_automatically(&self, prepared: &Prepared, now: &str) -> cloud::Result<()> {
        let saved = self.purge_saved(prepared)?;
        if saved.as_ref().is_some_and(|s| s.submitted_at.is_some()) {
            return self.purge_follow_up(prepared, AUTOMATIC_JOBS);
        }
        let (blocked, checkpoints) = self.purge_blocked(prepared, false, false)?;
        let eligible_at = saved
            .as_ref()
            .and_then(|s| s.eligible_at.as_deref())
            .map(seconds)
            .transpose()?;
        match automatic(
            !blocked.is_empty(),
            eligible_at,
            seconds(now)?,
            self.archive.purge_delay_seconds,
        ) {
            Automatic::Nothing => Ok(()),
            Automatic::ResetEligible => {
                let _lock = self.lock("state_purge.purge_automatically")?;
                if let Some(mut current) = self.purge_saved(prepared)? {
                    if current.submitted_at.is_none() {
                        current.eligible_at = None;
                        self.purge_save(prepared, &current)?;
                    }
                }
                Ok(())
            }
            step @ (Automatic::StartDelay | Automatic::Submit) => {
                let lock = self.lock("state_purge.purge_automatically")?;
                let mut record = match self.purge_saved(prepared)? {
                    Some(record) if record.submitted_at.is_some() => return Ok(()),
                    Some(record) => record,
                    None => Purge {
                        contract: CONTRACT.into(),
                        plan: self.purge_plan(prepared)?,
                        trigger: "automatic".into(),
                        eligible_at: None,
                        checkpoints: Vec::new(),
                        submitted_at: None,
                        job_attempt: 0,
                        receipt: None,
                        completed_at: None,
                    },
                };
                record.eligible_at.get_or_insert_with(|| now.into());
                record.checkpoints = checkpoints;
                if step == Automatic::StartDelay {
                    return self.purge_save(prepared, &record);
                }
                record.submitted_at = Some(now.into());
                self.purge_save(prepared, &record)?;
                drop(lock);
                self.purge_submit(prepared, &record)
            }
        }
    }
    /// Every Attempt whose Run copies checkpoints, looked at once. One Attempt's error
    /// never stops the others; its record says how far it got. `done` keeps the slots
    /// that need no further look: a Run that does not copy, or a purge with its receipt.
    pub fn purge_pass(&self, now: &str, done: &mut std::collections::BTreeSet<String>) {
        if !self.enabled || !self.archive.automatic_purge || !self.has_collectors() {
            return;
        }
        let Ok(names) = self.slots.entry_names() else {
            return;
        };
        for name in names {
            let Some(name) = name.to_str() else {
                continue;
            };
            if done.contains(name) || !super::super::observer::preparation_slot(name) {
                continue;
            }
            let Ok(prepared) = read(&self.slots, name)
                .and_then(|v| serde_json::from_value::<Prepared>(v).map_err(io_json))
            else {
                continue;
            };
            let b = &prepared.execution.binding;
            // The same checks `saved_attempt` makes, without reading the slot again.
            if Self::attempt_slot(&b.project, &b.run_id, &b.attempt_id)
                .ok()
                .as_deref()
                != Some(name)
                || action::validate(&prepared.plan, &prepared.execution).is_err()
                || prepared.plan["native_cloud"]["approval"] != json!(prepared.approval)
            {
                continue;
            }
            if !prepared
                .execution
                .package()
                .is_ok_and(|p| p.checkpoint_archive)
            {
                done.insert(name.to_owned());
                continue;
            }
            if self.require_live(&b.project, &b.run_id).is_err() {
                continue;
            }
            match self.purge_automatically(&prepared, now) {
                Ok(()) => {
                    if self
                        .purge_saved(&prepared)
                        .is_ok_and(|s| s.is_some_and(|s| s.receipt.is_some()))
                    {
                        done.insert(name.to_owned());
                    }
                }
                Err(error) => eprintln!(
                    "automatic state purge of {}/{}/{} deferred: {}",
                    b.project, b.run_id, b.attempt_id, error.code
                ),
            }
        }
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
        let Some(prepared) = self.saved_attempt(project, run, attempt)? else {
            return Ok(None);
        };
        let Some(saved) = self.purge_saved(&prepared)? else {
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
        let _lock = self.lock("state_purge.receive_purge")?;
        let mut saved = self
            .purge_saved(&prepared)?
            .ok_or_else(cloud::Error::evidence)?;
        if saved.receipt.is_none() {
            saved.receipt = Some(receipt.clone());
            saved.completed_at = Some(now.into());
            self.purge_save(&prepared, &saved)?;
        }
        Ok(Some(json!({"plan_id": id, "status": "RECEIVED"})))
    }
    /// One thread that looks for eligible Attempts, when the automatic purge is on. The
    /// leader lock is tried on every pass, so a second server takes over when the first
    /// stops, and only the holder purges.
    pub fn start_purger(self: Arc<Self>) -> Option<crate::background::Worker> {
        if !self.enabled || !self.archive.automatic_purge || !self.has_collectors() {
            return None;
        }
        crate::background::Worker::spawn("native-state-purge", move |stop| {
            let mut leader = None;
            let mut done = std::collections::BTreeSet::new();
            while !stop.requested() {
                if leader.is_none() {
                    leader = self
                        .directory
                        .lock(
                            OsStr::new(LEADER),
                            LockOptions {
                                create: true,
                                ..Default::default()
                            },
                        )
                        .ok();
                }
                if leader.is_some() {
                    if let Ok((now, _)) = crate::projects::transition_context() {
                        self.purge_pass(&now, &mut done);
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Evidence under which every condition holds.
    fn clear() -> Evidence {
        Evidence::first(true, true)
    }
    fn both(e: Evidence) -> (Vec<&'static str>, Vec<&'static str>) {
        (blocked_by(&e, false, false), blocked_by(&e, true, false))
    }

    #[test]
    fn every_condition_holding_blocks_nothing_on_either_path() {
        assert_eq!(both(clear()), (vec![], vec![]));
        assert!(blocked_by(&clear(), true, true).is_empty());
    }

    #[test]
    fn a_sync_not_proven_ended_blocks_both_paths_until_it_is() {
        let running = Evidence {
            sync_running: true,
            ..clear()
        };
        assert_eq!(
            both(running),
            (vec!["SYNC_IN_PROGRESS"], vec!["SYNC_IN_PROGRESS"])
        );
        assert_eq!(blocked_by(&running, true, true), vec!["SYNC_IN_PROGRESS"]);
        assert_eq!(both(clear()), (vec![], vec![]));
    }

    #[test]
    fn an_unfinished_worker_needs_an_explicit_request_and_an_accepted_sync() {
        let unfinished = Evidence {
            worker_finished: false,
            unfinished_sync_delivered: false,
            ..clear()
        };
        // The automatic purge never accepts it, nor a request that does not say so.
        assert_eq!(
            both(unfinished),
            (vec!["WORKER_NOT_FINISHED"], vec!["WORKER_NOT_FINISHED"])
        );
        assert_eq!(
            blocked_by(&unfinished, false, true),
            vec!["WORKER_NOT_FINISHED"]
        );
        // PURGE UNFINISHED waits for a sync that delivered the state.
        assert_eq!(blocked_by(&unfinished, true, true), vec!["SYNC_REQUIRED"]);
        let delivered = Evidence {
            unfinished_sync_delivered: true,
            ..unfinished
        };
        assert!(blocked_by(&delivered, true, true).is_empty());
        assert_eq!(
            blocked_by(&delivered, false, false),
            vec!["WORKER_NOT_FINISHED"]
        );
        // A sync still running blocks even then.
        let running = Evidence {
            sync_running: true,
            ..delivered
        };
        assert_eq!(blocked_by(&running, true, true), vec!["SYNC_IN_PROGRESS"]);
    }

    #[test]
    fn a_synced_report_is_kept_only_after_training_with_every_copy() {
        assert!(accepts_unfinished_sync(true, true));
        assert!(!accepts_unfinished_sync(false, true));
        assert!(!accepts_unfinished_sync(true, false));
        assert!(!accepts_unfinished_sync(false, false));
    }

    #[test]
    fn a_run_that_does_not_copy_is_never_purged_by_either_path() {
        let off = Evidence {
            archive_enabled: false,
            ..clear()
        };
        for (explicit, unfinished) in [(false, false), (true, false), (true, true)] {
            assert_eq!(
                blocked_by(&off, explicit, unfinished),
                vec!["CHECKPOINT_ARCHIVE_NOT_ENABLED"]
            );
        }
        // Whatever else holds, nothing tells the automatic purge to start.
        assert_eq!(automatic(true, None, 10_000, 0), Automatic::Nothing);
    }

    #[test]
    fn every_other_condition_blocks_with_its_own_code() {
        for (evidence, code) in [
            (
                Evidence {
                    training_ended: false,
                    ..clear()
                },
                "ATTEMPT_NOT_TERMINAL",
            ),
            (
                Evidence {
                    last_checkpoint_unverified: true,
                    ..clear()
                },
                "LAST_CHECKPOINT_UNVERIFIED",
            ),
            (
                Evidence {
                    results_pending: true,
                    ..clear()
                },
                "RESULTS_NOT_PUBLISHED",
            ),
            (
                Evidence {
                    checkpoint_unarchived: true,
                    ..clear()
                },
                "CHECKPOINT_NOT_ARCHIVED",
            ),
        ] {
            assert_eq!(both(evidence), (vec![code], vec![code]), "{code}");
        }
    }

    #[test]
    fn the_delay_starts_when_all_hold_and_restarts_after_a_condition_fails() {
        let delay = 900;
        assert_eq!(automatic(false, None, 1_000, delay), Automatic::StartDelay);
        assert_eq!(
            automatic(false, Some(1_000), 1_899, delay),
            Automatic::Nothing
        );
        assert_eq!(
            automatic(false, Some(1_000), 1_900, delay),
            Automatic::Submit
        );
        // A condition stops holding before the purge: the start is forgotten ...
        assert_eq!(
            automatic(true, Some(1_000), 1_500, delay),
            Automatic::ResetEligible
        );
        assert_eq!(automatic(true, None, 1_600, delay), Automatic::Nothing);
        // ... so holding again waits the whole delay from then.
        assert_eq!(automatic(false, None, 1_700, delay), Automatic::StartDelay);
        assert_eq!(
            automatic(false, Some(1_700), 1_900, delay),
            Automatic::Nothing
        );
        assert_eq!(
            automatic(false, Some(1_700), 2_600, delay),
            Automatic::Submit
        );
        // 0 purges at once.
        assert_eq!(automatic(false, None, 1_000, 0), Automatic::Submit);
    }

    #[test]
    fn only_a_job_proven_failed_leads_to_one_further_job_and_the_automatic_limit_is_three() {
        for state in [None, Some(RunState::Running), Some(RunState::Succeeded)] {
            assert_eq!(after_job(state, 0, AUTOMATIC_JOBS), AfterJob::Observe);
        }
        for state in [RunState::Failed, RunState::Cancelled, RunState::Preempted] {
            assert_eq!(after_job(Some(state), 0, AUTOMATIC_JOBS), AfterJob::Further);
        }
        let failed = Some(RunState::Failed);
        assert_eq!(AUTOMATIC_JOBS, 3);
        assert_eq!(after_job(failed, 1, AUTOMATIC_JOBS), AfterJob::Further);
        // Jobs 0, 1 and 2 have run: no fourth.
        assert_eq!(after_job(failed, 2, AUTOMATIC_JOBS), AfterJob::GiveUp);
        assert_eq!(after_job(failed, 2, EXPLICIT_JOBS), AfterJob::Further);
        assert_eq!(after_job(failed, 9, EXPLICIT_JOBS), AfterJob::GiveUp);
    }

    #[test]
    fn a_new_run_that_reads_in_place_cannot_resume_from_a_run_that_copies() {
        assert!(resume_refused(true, false, true));
        assert!(!resume_refused(true, true, true));
        assert!(!resume_refused(true, false, false));
        assert!(!resume_refused(true, true, false));
        // A Run that exists keeps what it was created with.
        assert!(!resume_refused(false, false, true));
    }
}
