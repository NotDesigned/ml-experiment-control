use super::{Error, Result, Runtime, transport::Sdk};
use crate::background::Worker;
use crate::records::{EventRow, RecordGuard, RemoteHistoryEvidence, RemoteObservation, Store};
use ml_exp_core::{
    records::{RunKind, Scope},
    tracking::{
        self,
        publication::{
            self, HistoryEvent, PublicationAction, PublicationState, PublicationStatus,
            RemoteEvidence,
        },
    },
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
mod repair;
mod session;
pub use repair::repair;
use session::{Next, Session};

/// Seconds before the next look at a scope waiting for remote history. W&B is
/// read at most this often per scope, so it also bounds the cost of waiting.
const POLL_SECONDS: u64 = 5;
/// An idle scope costs two local reads and no cloud request.
const IDLE_SECONDS: u64 = 2;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryRepair {
    events: BTreeMap<i64, String>,
    attempted: bool,
    remote_confirmed: bool,
    // States saved before automatic recovery have none of these.
    // One is what `attempted` already says, so an older server can still load it.
    #[serde(default, skip_serializing_if = "at_most_one")]
    attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    at: Option<String>,
}

/// What remains of a confirmed repair once a later one replaces its record.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RepairAudit {
    first: i64,
    last: i64,
    count: usize,
    attempts: u32,
    source: Option<String>,
    at: Option<String>,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    state: PublicationState,
    finish_requested: bool,
    remote_finished: bool,
    ever_opened: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    history_repair: Option<HistoryRepair>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    repairs: Vec<RepairAudit>,
    /// The most rows W&B has shown repeating a record: a repair crossed a late delivery.
    /// Nothing was lost, and a W&B row cannot be removed, so this only reports it.
    #[serde(default, skip_serializing_if = "is_zero")]
    duplicate_rows: u64,
    /// W&B no longer has a Run this scope had opened: a person deleted it. It is
    /// never recreated, so the scope stops here and shows why.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    remote_missing: bool,
}
fn at_most_one(value: &u32) -> bool {
    *value <= 1
}
fn is_zero(value: &u64) -> bool {
    *value == 0
}
pub struct Publisher {
    _worker: Worker,
}
/// Publication depends on frozen Run authority and a local journal, never on
/// a particular scheduler or a worker's ability to reach W&B directly.
pub trait Source: Send + Sync {
    fn publisher_leader(&self) -> Result<ml_exp_io::directory::FileLock>;
    fn publisher_store(&self) -> Result<Store>;
    fn tracking_journal(&self, project: &str, run: &str) -> Result<Store>;
    fn publication_scopes(&self, store: &Store) -> Result<BTreeSet<String>> {
        candidate_scopes(store)
    }
    /// The terminal state the provider has proven for this Attempt, so its worker
    /// can no longer write. None while it may still run, or with no provider.
    fn provider_terminal(
        &self,
        _project: &str,
        _run: &str,
        _attempt: &str,
    ) -> Result<Option<String>> {
        Ok(None)
    }
}
impl Source for crate::backends::slurm::service::Service {
    fn publisher_leader(&self) -> Result<ml_exp_io::directory::FileLock> {
        self.publisher_leader().map_err(why!(Error::Storage))
    }
    fn publisher_store(&self) -> Result<Store> {
        self.publisher_store().map_err(why!(Error::Storage))
    }
    fn tracking_journal(&self, project: &str, run: &str) -> Result<Store> {
        self.tracking_journal(project, run)
            .map_err(why!(Error::Identity))
    }
}
impl Source for crate::api::Api {
    fn publisher_leader(&self) -> Result<ml_exp_io::directory::FileLock> {
        if let Some(service) = self.submissions() {
            Source::publisher_leader(&*service)
        } else if let Some(service) = self.cloud() {
            Source::publisher_leader(&*service)
        } else {
            self.local_publisher_leader()
        }
    }
    fn publisher_store(&self) -> Result<Store> {
        if let Some(service) = self.submissions() {
            Source::publisher_store(&*service)
        } else if let Some(service) = self.cloud() {
            Source::publisher_store(&*service)
        } else {
            self.local_publisher_store()
        }
    }
    fn tracking_journal(&self, project: &str, run: &str) -> Result<Store> {
        if self.run_kind(project, run).map_err(why!(Error::Identity))? == RunKind::Cloud {
            let service = self.cloud().ok_or(Error::Configuration)?;
            service
                .validate_run(project, run)
                .map_err(why!(Error::Identity))?;
            return Source::tracking_journal(&*service, project, run);
        }
        if let Some(store) = self
            .local_journal(project, run)
            .map_err(why!(Error::Identity))?
        {
            return Ok(store);
        }
        self.submissions()
            .ok_or(Error::Configuration)
            .and_then(|s| Source::tracking_journal(&*s, project, run))
    }
    fn provider_terminal(&self, project: &str, run: &str, attempt: &str) -> Result<Option<String>> {
        if self.run_kind(project, run).map_err(why!(Error::Identity))? != RunKind::Cloud {
            return Ok(None);
        }
        let service = self.cloud().ok_or(Error::Configuration)?;
        Source::provider_terminal(&*service, project, run, attempt)
    }
}
impl Publisher {
    pub fn start(service: Arc<dyn Source>, runtime: Runtime) -> Result<Self> {
        if !runtime.config.enabled {
            return Err(Error::Configuration);
        }
        let leader = service.publisher_leader().map_err(why!(Error::Storage))?;
        let store = service.publisher_store().map_err(why!(Error::Storage))?;
        let worker = Worker::spawn("native-wandb", move |stop| {
            let flag = stop.flag();
            // Moved in by name: the flock must live as long as this thread.
            let _leader = leader;
            let start = Instant::now();
            let policy = ml_exp_core::polling::Policy {
                enabled: true,
                ..Default::default()
            };
            let mut schedule = ml_exp_core::polling::Schedule::default();
            let mut session = Session::default();
            let mut candidates = BTreeSet::new();
            let mut discover_at = 0;
            while !stop.requested() {
                let elapsed = start.elapsed().as_secs();
                if elapsed >= discover_at {
                    if let Ok(scopes) = service.publication_scopes(&store) {
                        session.retain(&scopes);
                        candidates = scopes;
                    }
                    discover_at = elapsed.saturating_add(30);
                }
                let (next, selected) = schedule.select(&candidates, elapsed, &policy);
                schedule = next;
                // Sleep only when no scope is due: the rotation inside `select`
                // is what shares the thread between Runs, one batch at a time.
                let Some((scope, _)) = selected else {
                    if !stop.pause() {
                        break;
                    }
                    continue;
                };
                let outcome =
                    publish_scope(&*service, &store, &runtime, &mut session, &scope, flag);
                let now = start.elapsed().as_secs();
                schedule = match &outcome {
                    Ok(Next::Poll) => schedule.due_at(&scope, now.saturating_add(POLL_SECONDS)),
                    Ok(Next::Idle) => schedule.due_at(&scope, now.saturating_add(IDLE_SECONDS)),
                    Err(_) => schedule.outcome(&scope, now, &policy, false, false),
                };
                match outcome {
                    Err(error) if !stop.requested() => {
                        record_failure(&store, &scope, &mut session, &error);
                    }
                    _ => {}
                }
            }
            session.shutdown();
        })
        .map_err(why!(Error::Transport))?;
        Ok(Self { _worker: worker })
    }
}
fn candidate_scopes(store: &Store) -> Result<BTreeSet<String>> {
    let mut scopes = BTreeSet::new();
    for (project, run) in store.publication_runs().map_err(why!(Error::Storage))? {
        let Ok(guard) = store.guard(&project, &run) else {
            continue;
        };
        for scope in store.run_scopes(&guard).map_err(why!(Error::Storage))? {
            let id = scope["id"].as_str().ok_or(Error::Identity)?;
            let saved = store
                .publication_state(&guard, id)
                .map_err(why!(Error::Storage))?;
            let confirmed = scope["confirmed"].as_i64().ok_or(Error::Identity)?;
            if saved.as_ref().is_some_and(|s| s["remote_missing"] == true) {
                continue;
            }
            if !saved.is_some_and(|s| s["remote_finished"] == true)
                || store
                    .derived_after(&guard, id, confirmed)
                    .map_err(why!(Error::Storage))?
            {
                scopes.insert(id.into());
            }
        }
    }
    Ok(scopes)
}
fn record_failure(store: &Store, scope: &str, session: &mut Session, error: &Error) {
    // A complete per-Run error frame leaves the pipe usable. A broken pipe does
    // not, and losing the SDK strands the other Runs' buffered rows: those stay
    // reserved, and only automatic recovery may append them, once proven absent.
    if matches!(error, Error::Transport | Error::SdkMissing) {
        session.lost();
    }
    // Rows may have arrived unseen during a failed read, so the next scan starts over.
    session.forget_watch(scope);
    let _ = failure(store, scope, error);
}
fn failure(store: &Store, scope: &str, error: &Error) -> Result<()> {
    let parsed = Scope::parse(scope).map_err(why!(Error::Identity))?;
    let guard = store
        .guard(&parsed.project, &parsed.run)
        .map_err(why!(Error::Storage))?;
    let (at, _) = crate::projects::transition_context().map_err(why!(Error::Storage))?;
    let code = match error {
        Error::SdkMissing => "WANDB_SDK_NOT_INSTALLED",
        Error::Identity => "PUBLICATION_RECONCILE_REQUIRED",
        Error::SdkRequest(code) => code,
        Error::Publication(code) => code,
        _ => "PUBLISHER_UNAVAILABLE",
    };
    store
        .publication_failure(&guard, scope, code, &at)
        .map_err(why!(Error::Storage))
}
fn publish_scope(
    service: &dyn Source,
    store: &Store,
    runtime: &Runtime,
    session: &mut Session,
    scope: &str,
    cancelled: &AtomicBool,
) -> Result<Next> {
    let parsed = Scope::parse(scope).map_err(why!(Error::Identity))?;
    let _authority = service
        .tracking_journal(&parsed.project, &parsed.run)
        .map_err(why!(Error::Publication("JOURNAL_AUTHORITY_UNAVAILABLE")))?;
    // Only a gate on finishing: an unreadable provider record leaves rows publishing.
    let ended = parsed.attempt.as_str().and_then(|attempt| {
        service
            .provider_terminal(&parsed.project, &parsed.run, attempt)
            .ok()
            .flatten()
    });
    publish_ended(store, runtime, session, scope, ended.as_deref(), cancelled)
}
#[cfg(test)]
fn publish_bound(
    store: &Store,
    runtime: &Runtime,
    session: &mut Session,
    scope: &str,
    cancelled: &AtomicBool,
) -> Result<Next> {
    publish_ended(store, runtime, session, scope, None, cancelled)
}
/// `ended` is the provider's proven terminal state of the scope's Attempt. A job the
/// provider stopped (preempted, killed, timed out) never writes METRICS_DRAINED, yet
/// its worker can no longer write either, so its W&B Run is finished as failed once
/// every journal row is confirmed. No lifecycle event is invented for it.
fn publish_ended(
    store: &Store,
    runtime: &Runtime,
    session: &mut Session,
    scope: &str,
    ended: Option<&str>,
    cancelled: &AtomicBool,
) -> Result<Next> {
    let parsed = Scope::parse(scope).map_err(why!(Error::Identity))?;
    let guard = store
        .guard(&parsed.project, &parsed.run)
        .map_err(why!(Error::Storage))?;
    let binding = store
        .binding(&guard)
        .map_err(why!(Error::Storage))?
        .ok_or(Error::Identity)?;
    if binding["route"]["enabled"] != true {
        return Err(Error::Identity);
    }
    let status = store
        .scope_status(&guard, scope)
        .map_err(why!(Error::Storage))?
        .ok_or(Error::Identity)?;
    let confirmed = status["confirmed"].as_i64().ok_or(Error::Identity)?;
    let total = store.total(&guard, scope).map_err(why!(Error::Storage))?;
    // An ended Attempt is looked at until it is finished, though its journal is still.
    if ended.is_none() && session.is_idle(scope, total) {
        return Ok(Next::Idle);
    }
    let mut saved: Saved = store
        .publication_state(&guard, scope)
        .map_err(why!(Error::Storage))?
        .map(serde_json::from_value)
        .transpose()
        .map_err(why!(Error::Identity))?
        .unwrap_or_default();
    if saved.remote_missing {
        session.mark_idle(scope, total);
        return Ok(Next::Idle);
    }
    if saved.remote_finished {
        // A finished W&B Run is resumed only for derived rows added after the Run
        // ended. It is then published, proven and finished again like any other.
        if !store
            .derived_after(&guard, scope, confirmed)
            .map_err(why!(Error::Storage))?
        {
            return Ok(Next::Idle);
        }
        saved.remote_finished = false;
        saved.finish_requested = false;
        saved.state.finished = false;
    }
    if saved.state.confirmed > confirmed {
        return Err(Error::Identity);
    }
    let display_version = tracking::DISPLAY_VERSION;
    saved.state.confirmed = confirmed;
    saved.state.sent.retain(|sequence, _| *sequence > confirmed);
    let highest = saved
        .state
        .sent
        .keys()
        .next_back()
        .map_or(confirmed, |s| *s);
    let mut events = store
        .events_bounded(
            &guard,
            scope,
            confirmed,
            session.share(scope).max(highest - confirmed),
            session::WINDOW_BYTES,
        )
        .map_err(why!(Error::Storage))?;
    // `sent` is saved as one bounded document, so event IDs bound a window too.
    let mut reserved = 0;
    if let Some(cut) = events.iter().position(|event| {
        reserved += event.event_id.len() + 24;
        reserved > session::RESERVATION_BYTES && event.seq > highest
    }) {
        events.truncate(cut);
    }
    let through = events.last().map_or(confirmed, |e| e.seq);
    // Only the journal's end can be terminal, and this query walks back through
    // every record since the last lifecycle event.
    let lifecycle = if through == total {
        store
            .latest_lifecycle(&guard, scope)
            .map_err(why!(Error::Storage))?
    } else {
        json!({})
    };
    let drained = lifecycle["phase"] == "METRICS_DRAINED"
        && matches!(
            lifecycle["state"].as_str(),
            Some("SUCCEEDED" | "FAILED" | "CANCELLED" | "TIMEOUT")
        );
    let terminal = (drained || ended.is_some()) && through == total;
    let mut job = json!({"operation":"observe","api_key":runtime.key,"route":binding["route"],"wandb_id":status["wandb_id"],"scope":status,"config":binding["config"],"confirmed":confirmed,"through":through,
        "open_session":!saved.finish_requested,"resume":if saved.ever_opened||confirmed>0||!saved.state.sent.is_empty(){"must"}else{"allow"}});
    let remote = match session.open(runtime)?.call(&job, cancelled) {
        // A Run this scope opened before is gone. Recreating it would publish a
        // partial copy under the same name, so the scope stops and says why.
        Err(Error::SdkRequest(code @ "WANDB_REMOTE_RUN_MISSING")) if job["resume"] == "must" => {
            saved.remote_missing = true;
            let (at, _) = crate::projects::transition_context().map_err(why!(Error::Storage))?;
            save(
                store,
                &guard,
                scope,
                &saved,
                "REMOTE_RUN_MISSING",
                Some(code),
                &at,
            )?;
            session.forget_watch(scope);
            session.mark_idle(scope, total);
            return Ok(Next::Idle);
        }
        result => result?,
    };
    if remote["wandb_id"] != status["wandb_id"] || remote["route"] != binding["route"] {
        return Err(Error::Publication("REMOTE_RUN_IDENTITY_MISMATCH"));
    }
    let evidence: RemoteEvidence = serde_json::from_value(remote["evidence"].clone())
        .map_err(why!(Error::Publication("REMOTE_HISTORY_EVIDENCE_INVALID")))?;
    let local: Vec<_> = events
        .iter()
        .map(|event| HistoryEvent {
            sequence: event.seq,
            event_id: event.event_id.clone(),
        })
        .collect();
    let expected = store
        .event_id(&guard, scope, confirmed)
        .map_err(why!(Error::Storage))?;
    let plan = publication::plan(
        &saved.state,
        &evidence,
        &local,
        expected.as_deref(),
        terminal,
        display_version,
    )
    .map_err(why!(Error::Publication("PUBLICATION_PLAN_INVALID")))?;
    match &plan.status {
        // A hole among our own reservations is what recovery looks at below.
        PublicationStatus::ReconcileRequired { .. } if saved.state.sent.is_empty() => {
            return Err(Error::Publication("REMOTE_HISTORY_RECONCILE_REQUIRED"));
        }
        PublicationStatus::IdentityConflict { .. } => {
            return Err(Error::Publication("REMOTE_EVENT_IDENTITY_CONFLICT"));
        }
        _ => {}
    }
    let (at, _) = crate::projects::transition_context().map_err(why!(Error::Storage))?;
    let display_ack = evidence.display_version == Some(display_version)
        && evidence.display_through == Some(plan.state.confirmed)
        && plan.state.confirmed == through;
    let proof = RemoteHistoryEvidence {
        route: binding["route"].clone(),
        wandb_id: status["wandb_id"].as_str().ok_or(Error::Identity)?.into(),
        through: plan.state.confirmed,
        observed: evidence
            .history
            .iter()
            .filter(|item| {
                item.sequence >= confirmed.max(1) && item.sequence <= plan.state.confirmed
            })
            .map(|item| RemoteObservation {
                sequence: item.sequence,
                event_id: item.event_id.clone(),
            })
            .collect(),
        display_version: display_ack.then_some(display_version),
        display_through: display_ack.then_some(plan.state.confirmed),
    };
    if plan.state.confirmed > confirmed || display_ack {
        store
            .confirm_publication(&guard, scope, &proof, None, &at)
            .map_err(why!(Error::Publication("PUBLICATION_ACK_EVIDENCE_INVALID")))?;
    }
    saved.ever_opened = true;
    saved.state = plan.state;
    saved.duplicate_rows = saved
        .duplicate_rows
        .max(remote["evidence"]["duplicate_rows"].as_u64().unwrap_or(0));
    if let Some(repair) = saved.history_repair.as_mut() {
        if repair.attempted
            && repair
                .events
                .keys()
                .all(|seq| *seq <= saved.state.confirmed)
        {
            repair.remote_confirmed = true;
        }
    }
    session.note_backlog(scope, total > saved.state.confirmed);
    let finish = plan
        .actions
        .iter()
        .any(|action| matches!(action, PublicationAction::Finish { .. }));
    let exit_code = if drained && lifecycle["state"] == "SUCCEEDED" {
        0
    } else {
        1
    };
    if saved.finish_requested
        && terminal
        && finish
        && remote["state"] == if exit_code == 0 { "finished" } else { "failed" }
    {
        saved.remote_finished = true;
        saved.state.finished = true;
        save(store, &guard, scope, &saved, "FINISHED", None, &at)?;
        return Ok(Next::Idle);
    }
    saved.state.finished = false;
    saved.finish_requested |= finish;
    // Reserve all selected sends durably BEFORE any SDK call. A crash between
    // reservation and transport remains uncertain; it never permits a resend.
    let (phase, error) = match plan.status {
        PublicationStatus::Synced { .. } => ("SYNCED", None),
        PublicationStatus::DisplayAckPending { .. } => {
            ("DISPLAY_ACK_PENDING", Some("REMOTE_DISPLAY_ACK_PENDING"))
        }
        PublicationStatus::Finished { .. } => {
            ("FINISH_ACK_PENDING", Some("REMOTE_FINISH_ACK_PENDING"))
        }
        _ => ("ACK_PENDING", Some("REMOTE_ACK_PENDING")),
    };
    let visit = repair::Visit {
        store,
        guard: &guard,
        scope,
        events: &events,
        at: &at,
        through,
        cancelled,
    };
    if plan.actions.is_empty() && !saved.state.sent.is_empty() {
        return repair::await_reservations(
            &visit,
            session,
            &mut saved,
            &remote,
            &job,
            (phase, error),
        );
    }
    session.forget_watch(scope);
    if plan.actions.is_empty() {
        save(store, &guard, scope, &saved, phase, error, &at)?;
        if phase == "SYNCED" && through == total {
            session.mark_idle(scope, total);
        }
        return Ok(Next::Idle);
    }
    let display = session
        .display(store, &guard, scope, through)
        .map_err(why!(Error::Storage))?;
    let keys: Vec<String> =
        serde_json::from_value(remote["summary_keys"].clone()).map_err(why!(Error::Identity))?;
    let mut effects = Vec::new();
    for action in plan.actions {
        effects.push(match action {
            PublicationAction::EmitHistory{sequence,event_id}=>{
                let event=events.iter().find(|event|event.seq==sequence&&event.event_id==event_id).ok_or(Error::Identity)?;
                json!({"operation":"history","sequence":sequence,"record":tracking::history_record(&event.document(),&display).map_err(history_error)?})
            },
            PublicationAction::PublishDisplay{..}=>json!({"operation":"display","patch":tracking::summary_patch(&display,&keys,"worker JSONL; accepted observations").map_err(why!(Error::Identity))?}),
            PublicationAction::Finish{..}=>json!({"operation":"finish","exit_code":exit_code}),
        });
    }
    job["operation"] = json!("apply");
    job["physical_next_step"] = remote["physical_next_step"].clone();
    job["resume"] = json!("must");
    job["definitions"] = json!(tracking::definition_specs(&display).map_err(why!(
        Error::Publication("JOURNAL_METRIC_DEFINITIONS_INVALID")
    ))?);
    // Reject an oversized/invalid effect before reserving any new sequence.
    // Transport loss after the durable reservation still requires remote ACK.
    let frames = Sdk::frames(&job, effects)?;
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Transport);
    }
    save(store, &guard, scope, &saved, phase, error, &at)?;
    session.send(scope, &job, frames, cancelled)?;
    Ok(Next::Poll)
}
fn history_error(error: tracking::Error) -> Error {
    match error {
        tracking::Error::Display(code @ "DISPLAY_NAME_CONFLICT") => Error::Publication(code),
        _ => Error::Publication("JOURNAL_HISTORY_PROJECTION_INVALID"),
    }
}
fn save(
    store: &Store,
    guard: &RecordGuard<'_>,
    scope: &str,
    saved: &Saved,
    status: &str,
    error: Option<&str>,
    at: &str,
) -> Result<()> {
    store
        .save_publication_state(
            guard,
            scope,
            &serde_json::to_value(saved).map_err(why!(Error::Identity))?,
            status,
            error,
            at,
        )
        .map_err(why!(Error::Storage))
}

#[cfg(test)]
mod tests;
