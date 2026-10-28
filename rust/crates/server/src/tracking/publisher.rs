use super::{Error, Result, Runtime, transport::Sdk};
use crate::{
    records::{RecordGuard, RemoteHistoryEvidence, RemoteObservation, Store},
    submissions::Service,
};
use ml_exp_core::{
    records::Scope,
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
    collections::BTreeSet,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    state: PublicationState,
    finish_requested: bool,
    remote_finished: bool,
    ever_opened: bool,
}
pub struct Publisher {
    stop: Arc<AtomicBool>,
    wake: Arc<(Mutex<()>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}
impl Publisher {
    pub fn start(service: Arc<Service>, runtime: Runtime) -> Result<Self> {
        if !runtime.config.enabled {
            return Err(Error::Configuration);
        }
        let leader = service.publisher_leader().map_err(|_| Error::Storage)?;
        let store = service.publisher_store().map_err(|_| Error::Storage)?;
        let stop = Arc::new(AtomicBool::new(false));
        let wake = Arc::new((Mutex::new(()), Condvar::new()));
        let flag = stop.clone();
        let sleep = wake.clone();
        let handle = thread::Builder::new()
            .name("native-wandb".into())
            .spawn(move || {
                let start = Instant::now();
                let policy = ml_exp_core::polling::Policy {
                    enabled: true,
                    ..Default::default()
                };
                let mut schedule = ml_exp_core::polling::Schedule::default();
                let mut sdk = None;
                let mut candidates = BTreeSet::new();
                let mut discover_at = 0;
                while !flag.load(Ordering::Relaxed) {
                    if service.validate_publisher_leader(&leader).is_err() {
                        break;
                    }
                    let elapsed = start.elapsed().as_secs();
                    if elapsed >= discover_at {
                        if let Ok(scopes) = candidate_scopes(&store) {
                            candidates = scopes;
                        }
                        discover_at = elapsed.saturating_add(30);
                    }
                    let (next, selected) = schedule.select(&candidates, elapsed, &policy);
                    schedule = next;
                    if let Some((scope, _)) = selected {
                        let outcome = publish_scope(
                            &service, &store, &runtime, &mut sdk, &scope, &flag, &leader,
                        );
                        schedule = schedule.outcome(
                            &scope,
                            start.elapsed().as_secs(),
                            &policy,
                            outcome.is_ok(),
                            false,
                        );
                        if let Err(error) = outcome {
                            sdk = None;
                            let _ = failure(&store, &scope, &error);
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
                drop(sdk);
            })
            .map_err(|_| Error::Transport)?;
        Ok(Self {
            stop,
            wake,
            thread: Some(handle),
        })
    }
}
impl Drop for Publisher {
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
fn candidate_scopes(store: &Store) -> Result<BTreeSet<String>> {
    let mut scopes = BTreeSet::new();
    for (project, run) in store.publication_runs(4096).map_err(|_| Error::Storage)? {
        let Ok(guard) = store.guard(&project, &run) else {
            continue;
        };
        for scope in store.run_scopes(&guard).map_err(|_| Error::Storage)? {
            let id = scope["id"].as_str().ok_or(Error::Identity)?;
            let saved = store
                .publication_state(&guard, id)
                .map_err(|_| Error::Storage)?;
            if !saved.is_some_and(|s| s["remote_finished"] == true) {
                scopes.insert(id.into());
            }
            if scopes.len() > 4096 {
                return Err(Error::Storage);
            }
        }
    }
    Ok(scopes)
}
fn failure(store: &Store, scope: &str, error: &Error) -> Result<()> {
    let parsed = Scope::parse(scope).map_err(|_| Error::Identity)?;
    let guard = store
        .guard(&parsed.project, &parsed.run)
        .map_err(|_| Error::Storage)?;
    let (at, _) = crate::projects::transition_context().map_err(|_| Error::Storage)?;
    let code = match error {
        Error::SdkMissing => "WANDB_SDK_NOT_INSTALLED",
        Error::Identity => "PUBLICATION_RECONCILE_REQUIRED",
        _ => "PUBLISHER_UNAVAILABLE",
    };
    store
        .publication_failure(&guard, scope, code, &at)
        .map_err(|_| Error::Storage)
}
fn publish_scope(
    service: &Service,
    store: &Store,
    runtime: &Runtime,
    sdk: &mut Option<Sdk>,
    scope: &str,
    cancelled: &AtomicBool,
    leader: &std::fs::File,
) -> Result<()> {
    let parsed = Scope::parse(scope).map_err(|_| Error::Identity)?;
    let _authority = service
        .tracking_journal(&parsed.project, &parsed.run)
        .map_err(|_| Error::Identity)?;
    publish_bound(store, runtime, sdk, scope, cancelled, &|| {
        service
            .validate_publisher_leader(leader)
            .map_err(|_| Error::Identity)
    })
}
fn publish_bound(
    store: &Store,
    runtime: &Runtime,
    sdk: &mut Option<Sdk>,
    scope: &str,
    cancelled: &AtomicBool,
    validate: &dyn Fn() -> Result<()>,
) -> Result<()> {
    validate()?;
    let parsed = Scope::parse(scope).map_err(|_| Error::Identity)?;
    let guard = store
        .guard(&parsed.project, &parsed.run)
        .map_err(|_| Error::Storage)?;
    let binding = store
        .binding(&guard)
        .map_err(|_| Error::Storage)?
        .ok_or(Error::Identity)?;
    if binding["route"]["enabled"] != true {
        return Err(Error::Identity);
    }
    let status = store
        .scope_status(&guard, scope)
        .map_err(|_| Error::Storage)?
        .ok_or(Error::Identity)?;
    let confirmed = status["confirmed"].as_i64().ok_or(Error::Identity)?;
    let total = store.total(&guard, scope).map_err(|_| Error::Storage)?;
    let mut saved: Saved = store
        .publication_state(&guard, scope)
        .map_err(|_| Error::Storage)?
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| Error::Identity)?
        .unwrap_or_default();
    if saved.remote_finished {
        return Ok(());
    }
    if saved.state.confirmed > confirmed {
        return Err(Error::Identity);
    }
    saved.state.confirmed = confirmed;
    saved.state.sent.retain(|sequence, _| *sequence > confirmed);
    let mut events = store
        .events(&guard, scope, confirmed, 128)
        .map_err(|_| Error::Storage)?;
    let mut bytes = 0;
    let mut count = 0;
    for event in &events {
        let size = serde_json::to_vec(&event.document())
            .map_err(|_| Error::Identity)?
            .len();
        if bytes + size > 2 * 1024 * 1024 {
            break;
        }
        bytes += size;
        count += 1;
    }
    if !events.is_empty() && count == 0 {
        return Err(Error::Storage);
    }
    events.truncate(count);
    let through = events.last().map_or(confirmed, |e| e.seq);
    let lifecycle = store
        .latest_lifecycle(&guard, scope)
        .map_err(|_| Error::Storage)?;
    let terminal = lifecycle["phase"] == "METRICS_DRAINED"
        && matches!(
            lifecycle["state"].as_str(),
            Some("SUCCEEDED" | "FAILED" | "CANCELLED" | "TIMEOUT")
        )
        && through == total;
    let mut job = json!({"operation":"observe","api_key":runtime.key,"route":binding["route"],"wandb_id":status["wandb_id"],"scope":status,"config":binding["config"],"confirmed":confirmed,"through":through,
        "open_session":!saved.finish_requested,"resume":if saved.ever_opened||confirmed>0||!saved.state.sent.is_empty(){"must"}else{"allow"}});
    if sdk.is_none() {
        *sdk = Some(Sdk::open(&runtime.config.python)?);
    }
    let remote = sdk
        .as_mut()
        .ok_or(Error::Transport)?
        .call(&job, cancelled)?;
    validate()?;
    if remote["wandb_id"] != status["wandb_id"] || remote["route"] != binding["route"] {
        return Err(Error::Identity);
    }
    let evidence: RemoteEvidence =
        serde_json::from_value(remote["evidence"].clone()).map_err(|_| Error::Identity)?;
    let local: Vec<_> = events
        .iter()
        .map(|event| HistoryEvent {
            sequence: event.seq,
            event_id: event.event_id.clone(),
        })
        .collect();
    let expected = store
        .event_id(&guard, scope, confirmed)
        .map_err(|_| Error::Storage)?;
    let plan = publication::plan(
        &saved.state,
        &evidence,
        &local,
        expected.as_deref(),
        terminal,
        2,
    )
    .map_err(|_| Error::Identity)?;
    if matches!(
        plan.status,
        PublicationStatus::ReconcileRequired { .. } | PublicationStatus::IdentityConflict { .. }
    ) {
        return Err(Error::Identity);
    }
    let (at, _) = crate::projects::transition_context().map_err(|_| Error::Storage)?;
    let display_ack = evidence.display_version == Some(2)
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
        display_version: display_ack.then_some(2),
        display_through: display_ack.then_some(plan.state.confirmed),
    };
    store
        .confirm_publication(&guard, scope, &proof, None, &at)
        .map_err(|_| Error::Identity)?;
    saved.ever_opened = true;
    saved.state = plan.state;
    let finish = plan
        .actions
        .iter()
        .any(|action| matches!(action, PublicationAction::Finish { .. }));
    let exit_code = if lifecycle["state"] == "SUCCEEDED" {
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
        return save(store, &guard, scope, &saved, "FINISHED", None, &at);
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
    if plan.actions.is_empty() {
        validate()?;
        return save(store, &guard, scope, &saved, phase, error, &at);
    }
    let display = store
        .display(&guard, scope, through)
        .map_err(|_| Error::Storage)?;
    let keys: Vec<String> =
        serde_json::from_value(remote["summary_keys"].clone()).map_err(|_| Error::Identity)?;
    let mut effects = Vec::new();
    for action in plan.actions {
        effects.push(match action {
            PublicationAction::EmitHistory{sequence,event_id}=>{
                let event=events.iter().find(|event|event.seq==sequence&&event.event_id==event_id).ok_or(Error::Identity)?;
                json!({"operation":"history","sequence":sequence,"record":tracking::history_record(&event.document(),&display).map_err(|_|Error::Identity)?})
            },
            PublicationAction::PublishDisplay{..}=>json!({"operation":"display","patch":tracking::summary_patch(&display,&keys,terminal,"worker JSONL; accepted observations").map_err(|_|Error::Identity)?}),
            PublicationAction::Finish{..}=>json!({"operation":"finish","exit_code":exit_code}),
        });
    }
    job["operation"] = json!("apply");
    job["resume"] = json!("must");
    job["definitions"] = json!(tracking::definition_specs(&display).map_err(|_| Error::Identity)?);
    job["effects"] = json!(effects);
    // Reject an oversized/invalid effect before reserving any new sequence.
    // Transport loss after the durable reservation still requires remote ACK.
    Sdk::validate_request(&job)?;
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Transport);
    }
    validate()?;
    save(store, &guard, scope, &saved, phase, error, &at)?;
    validate()?;
    let applied = sdk
        .as_mut()
        .ok_or(Error::Transport)?
        .call(&job, cancelled)?;
    if applied["applied"] != true {
        return Err(Error::Transport);
    }
    Ok(())
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
            &serde_json::to_value(saved).map_err(|_| Error::Identity)?,
            status,
            error,
            at,
        )
        .map_err(|_| Error::Storage)
}

#[cfg(test)]
mod tests;
