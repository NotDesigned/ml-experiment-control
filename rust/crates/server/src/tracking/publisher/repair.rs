//! Recovery of reservations the remote lacks: automatic after they settle, or by
//! an operator with the API stopped. Same proof either way, and neither replays
//! a row it has not first observed to be absent.
use super::*;

/// Each attempt waits twice as long as the last, up to 64 settle delays. Nothing stops
/// for good: an attempt can fail before sending a row, and no person is watching.
fn settle_for(settle: Duration, attempts: u32) -> Duration {
    settle.saturating_mul(1 << attempts.min(6))
}

pub(super) struct Visit<'a, 'g> {
    pub(super) store: &'a Store,
    pub(super) guard: &'a RecordGuard<'g>,
    pub(super) scope: &'a str,
    pub(super) events: &'a [EventRow],
    pub(super) at: &'a str,
    pub(super) through: i64,
    pub(super) cancelled: &'a AtomicBool,
}

/// Reservations are outstanding and nothing can be sent past them. While the
/// running SDK child holds them, wait for it. Otherwise nothing is still sending
/// them, so once history has stayed unchanged for the settle delay, compare it
/// with the journal and the reservations and append only what is proven absent.
pub(super) fn await_reservations(
    visit: &Visit<'_, '_>,
    session: &mut Session,
    saved: &mut Saved,
    remote: &serde_json::Value,
    job: &serde_json::Value,
    pending: (&str, Option<&str>),
) -> Result<Next> {
    let &Visit {
        store,
        guard,
        scope,
        events,
        at,
        through,
        cancelled,
    } = visit;
    let stable = session.stable_for(scope, &remote["evidence"]);
    if session.holds(scope) {
        if stable.is_some_and(|elapsed| elapsed >= session::STALL) {
            session.abandon(scope, job, cancelled);
        }
        save(store, guard, scope, saved, pending.0, pending.1, at)?;
        return Ok(Next::Poll);
    }
    let attempts = saved
        .history_repair
        .as_ref()
        .filter(|repair| repair.attempted && !repair.remote_confirmed)
        .map_or(0, |repair| repair.attempts.max(1));
    let repair_pending = ("REPAIR_ACK_PENDING", Some("REMOTE_REPAIR_ACK_PENDING"));
    let settled = stable.is_some_and(|elapsed| elapsed >= settle_for(session.settle, attempts));
    if !settled {
        let (phase, error) = if attempts > 0 {
            repair_pending
        } else {
            ("RECOVERY_SETTLING", Some("REMOTE_ACK_PENDING"))
        };
        save(store, guard, scope, saved, phase, error, at)?;
        return Ok(Next::Poll);
    }
    let evidence: RemoteEvidence = serde_json::from_value(remote["evidence"].clone())
        .map_err(why!(Error::Publication("REMOTE_HISTORY_EVIDENCE_INVALID")))?;
    let local = identities(events);
    let expected = store
        .event_id(guard, scope, saved.state.confirmed)
        .map_err(why!(Error::Storage))?;
    let missing =
        publication::missing_reserved(&saved.state, &evidence, &local, expected.as_deref())
            .map_err(why!(Error::Publication(
                "REMOTE_HISTORY_RECONCILE_REQUIRED"
            )))?;
    if missing.is_empty() {
        save(store, guard, scope, saved, pending.0, pending.1, at)?;
        return Ok(Next::Poll);
    }
    let display = session
        .display(store, guard, scope, through)
        .map_err(why!(Error::Storage))?;
    let frames = repair_frames(events, &missing, &display, job, remote)?;
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Transport);
    }
    record_repair(saved, &missing, "automatic", at);
    save(
        store,
        guard,
        scope,
        saved,
        repair_pending.0,
        repair_pending.1,
        at,
    )?;
    session.forget_watch(scope);
    session.send(scope, job, frames, cancelled)?;
    Ok(Next::Poll)
}

fn identities(events: &[EventRow]) -> Vec<HistoryEvent> {
    events
        .iter()
        .map(|event| HistoryEvent {
            sequence: event.seq,
            event_id: event.event_id.clone(),
        })
        .collect()
}

/// The original records of `missing`, unchanged, as frames for a new SDK apply.
fn repair_frames(
    events: &[EventRow],
    missing: &[HistoryEvent],
    display: &serde_json::Value,
    job: &serde_json::Value,
    observed: &serde_json::Value,
) -> Result<Vec<serde_json::Value>> {
    let effects = missing
        .iter()
        .map(|item| {
            let event = events
                .iter()
                .find(|e| e.seq == item.sequence && e.event_id == item.event_id)
                .ok_or(Error::Identity)?;
            Ok(json!({"operation":"history","sequence":item.sequence,
                "record":tracking::history_record(&event.document(),display).map_err(history_error)?}))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut job = job.clone();
    job["operation"] = json!("apply");
    job["physical_next_step"] = observed["physical_next_step"].clone();
    job["resume"] = json!("must");
    job["definitions"] = json!(tracking::definition_specs(display).map_err(why!(
        Error::Publication("JOURNAL_METRIC_DEFINITIONS_INVALID")
    ))?);
    Sdk::frames(&job, effects)
}

/// The auditable record, saved before any row is sent. A new repair after a
/// confirmed one keeps a summary of it; a repeat counts as another attempt.
fn record_repair(saved: &mut Saved, missing: &[HistoryEvent], source: &str, at: &str) {
    let attempts = match saved.history_repair.take() {
        Some(done) if done.remote_confirmed => {
            saved.repairs.push(RepairAudit {
                first: done.events.keys().next().copied().unwrap_or_default(),
                last: done.events.keys().next_back().copied().unwrap_or_default(),
                count: done.events.len(),
                attempts: done.attempts.max(1),
                source: done.source,
                at: done.at,
            });
            if saved.repairs.len() > 32 {
                saved.repairs.remove(0);
            }
            1
        }
        Some(open) if open.attempted => open.attempts.max(1) + 1,
        _ => 1,
    };
    saved.history_repair = Some(HistoryRepair {
        events: missing
            .iter()
            .map(|e| (e.sequence, e.event_id.clone()))
            .collect(),
        attempted: true,
        remote_confirmed: false,
        attempts,
        source: Some(source.into()),
        at: Some(at.into()),
    });
}

pub fn repair(
    service: Arc<dyn Source>,
    runtime: Runtime,
    scope: &str,
) -> Result<serde_json::Value> {
    let _leader = service.publisher_leader()?;
    let parsed = Scope::parse(scope).map_err(why!(Error::Identity))?;
    let store = service.tracking_journal(&parsed.project, &parsed.run)?;
    let guard = store
        .guard(&parsed.project, &parsed.run)
        .map_err(why!(Error::Storage))?;
    let binding = store
        .binding(&guard)
        .map_err(why!(Error::Storage))?
        .ok_or(Error::Identity)?;
    let status = store
        .scope_status(&guard, scope)
        .map_err(why!(Error::Storage))?
        .ok_or(Error::Identity)?;
    let confirmed = status["confirmed"].as_i64().ok_or(Error::Identity)?;
    let mut saved: Saved = serde_json::from_value(
        store
            .publication_state(&guard, scope)
            .map_err(why!(Error::Storage))?
            .ok_or(Error::Identity)?,
    )
    .map_err(why!(Error::Identity))?;
    if saved.remote_finished || binding["route"]["enabled"] != true {
        return Err(Error::Identity);
    }
    let highest = saved
        .state
        .sent
        .keys()
        .next_back()
        .map_or(confirmed, |s| *s);
    let events = store
        .events(&guard, scope, confirmed, (highest - confirmed).max(128))
        .map_err(why!(Error::Storage))?;
    let through = events.last().map_or(confirmed, |e| e.seq);
    let mut sdk = Sdk::open(&runtime.config.python)?;
    let flag = AtomicBool::new(false);
    let job = json!({"operation":"observe","api_key":runtime.key,"route":binding["route"],
        "wandb_id":status["wandb_id"],"scope":status,"config":binding["config"],"confirmed":confirmed,
        "through":through,"open_session":false,"resume":"must"});
    let first = sdk.call(&job, &flag)?;
    let observed = sdk.call(&job, &flag)?;
    if first["evidence"] != observed["evidence"]
        || observed["wandb_id"] != status["wandb_id"]
        || observed["route"] != binding["route"]
    {
        return Err(Error::Publication("REMOTE_REPAIR_EVIDENCE_CHANGED"));
    }
    let evidence: RemoteEvidence =
        serde_json::from_value(observed["evidence"].clone()).map_err(why!(Error::Identity))?;
    let local = identities(&events);
    let expected = store
        .event_id(&guard, scope, confirmed)
        .map_err(why!(Error::Storage))?;
    let missing =
        publication::missing_reserved(&saved.state, &evidence, &local, expected.as_deref())
            .map_err(why!(Error::Identity))?;
    let proof_through = evidence
        .last_history_step
        .unwrap_or(-1)
        .saturating_add(1)
        .max(missing.last().map_or(confirmed, |event| event.sequence))
        .min(through);
    if missing.is_empty() {
        return Ok(
            json!({"scope":scope,"status":"NO_RESERVED_GAP","confirmed":confirmed,"writes":0}),
        );
    }
    // A confirmed earlier repair does not stop a new gap being repaired.
    if saved
        .history_repair
        .as_ref()
        .is_some_and(|r| r.attempted && !r.remote_confirmed)
    {
        return Ok(
            json!({"scope":scope,"status":"REPAIR_ACK_PENDING","confirmed":confirmed,"writes":0}),
        );
    }
    let display = store
        .display(&guard, scope, through)
        .map_err(why!(Error::Identity))?;
    let frames = repair_frames(&events, &missing, &display, &job, &observed)?;
    let (at, _) = crate::projects::transition_context().map_err(why!(Error::Storage))?;
    record_repair(&mut saved, &missing, "operator", &at);
    save(
        &store,
        &guard,
        scope,
        &saved,
        "REPAIR_ACK_PENDING",
        Some("REMOTE_REPAIR_ACK_PENDING"),
        &at,
    )?;
    // The original sent map and confirmed cursor are unchanged. A lost reply
    // leaves this new repair reservation uncertain; repeating only observes.
    for frame in &frames {
        let applied = sdk.call(frame, &flag)?;
        if applied["applied"] != true {
            return Err(Error::Transport);
        }
    }
    sdk.shutdown()?;
    let mut remote = sdk.call(&job, &flag)?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let evidence: RemoteEvidence =
            serde_json::from_value(remote["evidence"].clone()).map_err(why!(Error::Identity))?;
        let proof = RemoteHistoryEvidence {
            route: binding["route"].clone(),
            wandb_id: status["wandb_id"].as_str().ok_or(Error::Identity)?.into(),
            through: proof_through,
            observed: evidence
                .history
                .iter()
                .filter(|e| e.sequence <= proof_through)
                .map(|e| RemoteObservation {
                    sequence: e.sequence,
                    event_id: e.event_id.clone(),
                })
                .collect(),
            display_version: None,
            display_through: None,
        };
        if publication::missing_reserved(&saved.state, &evidence, &local, expected.as_deref())
            .map_err(why!(Error::Identity))?
            .is_empty()
        {
            let (at, _) = crate::projects::transition_context().map_err(why!(Error::Storage))?;
            store
                .confirm_publication(&guard, scope, &proof, None, &at)
                .map_err(why!(Error::Publication("PUBLICATION_ACK_EVIDENCE_INVALID")))?;
            saved.state.confirmed = proof.through;
            saved.state.sent.retain(|seq, _| *seq > proof.through);
            saved
                .history_repair
                .as_mut()
                .ok_or(Error::Identity)?
                .remote_confirmed = true;
            save(&store, &guard, scope, &saved, "SYNCED", None, &at)?;
            return Ok(
                json!({"scope":scope,"status":"REPAIRED","confirmed":proof.through,"appended":missing.len(),"journal_unchanged":true}),
            );
        }
        if Instant::now() >= deadline {
            return Ok(
                json!({"scope":scope,"status":"REPAIR_ACK_PENDING","writes":missing.len(),"confirmed":confirmed}),
            );
        }
        thread::sleep(Duration::from_secs(2));
        remote = sdk.call(&job, &flag)?;
    }
}

#[cfg(test)]
mod settle_tests {
    use super::*;

    #[test]
    fn each_attempt_waits_longer_up_to_a_limit() {
        let settle = Duration::from_secs(180);
        assert_eq!(settle_for(settle, 0), settle);
        assert_eq!(settle_for(settle, 1), settle * 2);
        assert_eq!(settle_for(settle, 3), settle * 8);
        assert_eq!(settle_for(settle, 6), settle * 64);
        assert_eq!(settle_for(settle, 500), settle * 64);
    }
}
