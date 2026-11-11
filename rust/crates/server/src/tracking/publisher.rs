//! W&B as a best-effort mirror of the journal. Each Attempt's rows are sent to
//! its one W&B Run in order, the cursor advances when the SDK accepted them,
//! and a failed send is retried from the cursor. W&B is never read back: the
//! journal, result archives and MLflow are the authoritative record.
use super::{Error, Result, Runtime, transport::Sdk};
use crate::background::Worker;
use crate::records::{RecordGuard, Store};
use ml_exp_core::{
    records::{RunKind, Scope},
    tracking::{
        self,
        publication::{self, Progress, PublicationState, Step},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
mod session;
use session::{Next, Session};

/// An idle scope costs two local reads and no cloud request.
const IDLE_SECONDS: u64 = 2;

/// Unknown fields are ignored, so the reservations, repair records and
/// duplicate counts an earlier release saved still load; they are dropped on
/// the next save.
#[derive(Default, Serialize, Deserialize)]
struct Saved {
    #[serde(default)]
    state: PublicationState,
    #[serde(default)]
    remote_finished: bool,
    #[serde(default)]
    ever_opened: bool,
    /// W&B no longer has a Run this scope had opened: a person deleted it. It is
    /// never recreated, so the scope stops here and shows why.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    remote_missing: bool,
    /// The SHA256 of the exclusion list this state publishes a rebuilt W&B Run
    /// for. A server that predates rebuilds cannot load it, so it never sends the
    /// excluded names to the new id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    wandb_rebuild: Option<String>,
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
                    Ok(Next::More) => schedule.due_at(&scope, now),
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
    // not: the child is dropped, and every scope resends from its cursor.
    if matches!(error, Error::Transport | Error::SdkMissing) {
        session.lost();
    }
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
        Error::Identity => "PUBLICATION_STATE_INVALID",
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
/// One scope and what this visit knows about it.
struct Visit<'a, 'g> {
    store: &'a Store,
    guard: &'a RecordGuard<'g>,
    scope: &'a str,
    wandb_id: &'a str,
    at: &'a str,
    total: i64,
}
/// `ended` is the provider's proven terminal state of the scope's Attempt. A job the
/// provider stopped (preempted, killed, timed out) never writes METRICS_DRAINED, yet
/// its worker can no longer write either, so its W&B Run is finished as failed once
/// every journal row is sent. No lifecycle event is invented for it.
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
    let mut confirmed = status["confirmed"].as_i64().ok_or(Error::Identity)?;
    let wandb_id = status["wandb_id"].as_str().ok_or(Error::Identity)?;
    // What this thread remembers is about one W&B Run: a rebuild starts afresh.
    session.rebind(scope, wandb_id);
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
        // ended. They are then sent and the Run finished again like any other.
        if !store
            .derived_after(&guard, scope, confirmed)
            .map_err(why!(Error::Storage))?
        {
            return Ok(Next::Idle);
        }
        saved.remote_finished = false;
        saved.state.finished = false;
    }
    let (at, _) = crate::projects::transition_context().map_err(why!(Error::Storage))?;
    let visit = Visit {
        store,
        guard: &guard,
        scope,
        wandb_id,
        at: &at,
        total,
    };
    // An earlier release waited to see reserved rows in W&B history. They were
    // handed to the SDK, so they count as sent once and are not sent again.
    if !saved.state.sent.is_empty() {
        saved.ever_opened = true;
        confirmed = carry_legacy(&visit, &saved.state, confirmed)?;
        saved.state.sent.clear();
        let phase = if confirmed >= total {
            "SYNCED"
        } else {
            "PENDING"
        };
        save(&visit, &saved, phase, None)?;
    }
    let display_current = status["display_version"] == tracking::DISPLAY_VERSION
        && status["display_confirmed"] == confirmed;
    // Only the journal's end can be terminal, and this query walks back through
    // every record since the last lifecycle event.
    let lifecycle = if confirmed == total {
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
    let step = publication::next(&Progress {
        cursor: confirmed,
        total,
        display_current,
        terminal: drained || ended.is_some(),
        finished: saved.remote_finished,
    });
    let job = json!({"api_key":runtime.key,"route":binding["route"],"wandb_id":wandb_id,
        "scope":status,"config":binding["config"],
        "resume":if saved.ever_opened || confirmed > 0 {"must"} else {"allow"}});
    let sent = match step {
        Step::Idle => {
            // A status an earlier release left, such as ACK_PENDING, moves on.
            if status["status"] != "SYNCED" {
                save(&visit, &saved, "SYNCED", None)?;
            }
            session.mark_idle(scope, total);
            return Ok(Next::Idle);
        }
        Step::Send | Step::Display => send(
            &visit, runtime, session, &mut saved, job, confirmed, cancelled,
        ),
        Step::Finish => {
            let exit_code = i32::from(!(drained && lifecycle["state"] == "SUCCEEDED"));
            finish(
                &visit, runtime, session, &mut saved, job, exit_code, cancelled,
            )
        }
    };
    match sent {
        // A Run this scope opened before is gone. Recreating it would publish a
        // partial copy under the same name, so the scope stops and says why.
        Err(Error::SdkRequest(code @ "WANDB_REMOTE_RUN_MISSING")) => {
            saved.remote_missing = true;
            save(&visit, &saved, "REMOTE_RUN_MISSING", Some(code))?;
            session.mark_idle(scope, total);
            Ok(Next::Idle)
        }
        result => result,
    }
}
/// Advance the cursor through the rows an earlier release reserved.
fn carry_legacy(visit: &Visit<'_, '_>, state: &PublicationState, confirmed: i64) -> Result<i64> {
    let highest = state.sent.keys().next_back().copied().unwrap_or(confirmed);
    if highest <= confirmed {
        return Ok(confirmed);
    }
    let rows = visit
        .store
        .events(visit.guard, visit.scope, confirmed, highest - confirmed)
        .map_err(why!(Error::Storage))?;
    let ids: Vec<_> = rows
        .iter()
        .map(|row| (row.seq, row.event_id.as_str()))
        .collect();
    let through = publication::legacy_cursor(state, confirmed, &ids);
    if through > confirmed {
        advance(visit, through, false)?;
    }
    Ok(through)
}
/// Send the rows after `confirmed`, with the summary for their end, and move the
/// cursor as each request is accepted. With no rows left it sends the summary.
fn send(
    visit: &Visit<'_, '_>,
    runtime: &Runtime,
    session: &mut Session,
    saved: &mut Saved,
    mut job: Value,
    confirmed: i64,
    cancelled: &AtomicBool,
) -> Result<Next> {
    let &Visit {
        store,
        guard,
        scope,
        total,
        ..
    } = visit;
    let events = store
        .events_bounded(
            guard,
            scope,
            confirmed,
            session.batch,
            session::WINDOW_BYTES,
        )
        .map_err(why!(Error::Storage))?;
    let through = events.last().map_or(confirmed, |e| e.seq);
    let display = session
        .display(store, guard, scope, through)
        .map_err(why!(Error::Storage))?;
    let history = events
        .iter()
        .map(|event| tracking::history_record(&event.document(), &display).map_err(history_error))
        .collect::<Result<Vec<_>>>()?;
    let patch = tracking::summary_patch(
        &display,
        &summary_names(&display),
        "worker JSONL; accepted observations",
    )
    .map_err(why!(Error::Identity))?;
    job["operation"] = json!("apply");
    job["definitions"] = json!(tracking::definition_specs(&display).map_err(why!(
        Error::Publication("JOURNAL_METRIC_DEFINITIONS_INVALID")
    ))?);
    let frames = Sdk::frames(&job, history, json!(patch))?;
    let mut cursor = confirmed;
    for mut frame in frames {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Transport);
        }
        let rows = frame["history"].as_array().map_or(0, Vec::len);
        if saved.ever_opened {
            frame["resume"] = json!("must");
        }
        let applied = session.call(runtime, &frame, cancelled)?;
        if applied["applied"] != true || applied["rows"] != rows {
            return Err(Error::Transport);
        }
        cursor += i64::try_from(rows).map_err(why!(Error::Identity))?;
        advance(visit, cursor, !frame["display"].is_null())?;
        if !saved.ever_opened {
            saved.ever_opened = true;
            let phase = if cursor >= total { "SYNCED" } else { "PENDING" };
            save(visit, saved, phase, None)?;
        }
    }
    Ok(Next::More)
}
/// Finish the W&B Run: `finished` for a Run that succeeded, `failed` otherwise.
fn finish(
    visit: &Visit<'_, '_>,
    runtime: &Runtime,
    session: &mut Session,
    saved: &mut Saved,
    mut job: Value,
    exit_code: i32,
    cancelled: &AtomicBool,
) -> Result<Next> {
    job["operation"] = json!("finish");
    job["exit_code"] = json!(exit_code);
    let finished = session.call(runtime, &job, cancelled)?;
    if finished["finished"] != true {
        return Err(Error::Transport);
    }
    saved.ever_opened = true;
    saved.remote_finished = true;
    saved.state.finished = true;
    save(visit, saved, "FINISHED", None)?;
    Ok(Next::Idle)
}
/// Every name this journal summarizes. The SDK deletes only those it holds, so
/// a name with no valid value loses its stale summary and other keys stay.
fn summary_names(display: &Value) -> Vec<String> {
    let names: BTreeSet<_> = display["series"]
        .as_object()
        .into_iter()
        .flat_map(|series| series.values())
        .filter_map(|row| row["result"].as_str().map(String::from))
        .collect();
    names.into_iter().collect()
}
fn history_error(error: tracking::Error) -> Error {
    match error {
        tracking::Error::Display(code @ "DISPLAY_NAME_CONFLICT") => Error::Publication(code),
        _ => Error::Publication("JOURNAL_HISTORY_PROJECTION_INVALID"),
    }
}
fn advance(visit: &Visit<'_, '_>, through: i64, display: bool) -> Result<()> {
    visit
        .store
        .advance_publication(
            visit.guard,
            visit.scope,
            visit.wandb_id,
            through,
            display,
            visit.at,
        )
        .map_err(why!(Error::Storage))
}
fn save(visit: &Visit<'_, '_>, saved: &Saved, status: &str, error: Option<&str>) -> Result<()> {
    visit
        .store
        .save_publication_state(
            visit.guard,
            visit.scope,
            &serde_json::to_value(saved).map_err(why!(Error::Identity))?,
            status,
            error,
            visit.at,
        )
        .map_err(why!(Error::Storage))
}

#[cfg(test)]
mod tests;
