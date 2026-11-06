//! Explicit operator recovery; normal publication never replays unknown sends.
use super::*;

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
    let events = store
        .events(&guard, scope, confirmed, 128)
        .map_err(why!(Error::Storage))?;
    let through = events.last().map_or(confirmed, |e| e.seq);
    let mut sdk = Sdk::open(&runtime.config.python)?;
    let flag = AtomicBool::new(false);
    let mut job = json!({"operation":"observe","api_key":runtime.key,"route":binding["route"],
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
    let local: Vec<_> = events
        .iter()
        .map(|e| HistoryEvent {
            sequence: e.seq,
            event_id: e.event_id.clone(),
        })
        .collect();
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
    if saved.history_repair.as_ref().is_some_and(|r| r.attempted) {
        return Ok(
            json!({"scope":scope,"status":"REPAIR_ACK_PENDING","confirmed":confirmed,"writes":0}),
        );
    }
    let display = store
        .display(&guard, scope, through)
        .map_err(why!(Error::Identity))?;
    let effects: Vec<_> = missing.iter().map(|item| {
        let event = events.iter().find(|e|e.seq==item.sequence && e.event_id==item.event_id).ok_or(Error::Identity)?;
        Ok(json!({"operation":"history","sequence":item.sequence,
            "record":tracking::history_record(&event.document(),&display).map_err(why!(Error::Identity))?}))
    }).collect::<Result<_>>()?;
    job["operation"] = json!("apply");
    job["physical_next_step"] = observed["physical_next_step"].clone();
    job["effects"] = json!(effects);
    job["definitions"] =
        json!(tracking::definition_specs(&display).map_err(why!(Error::Identity))?);
    Sdk::validate_request(&job)?;
    saved.history_repair = Some(HistoryRepair {
        events: missing
            .iter()
            .map(|e| (e.sequence, e.event_id.clone()))
            .collect(),
        attempted: true,
        remote_confirmed: false,
    });
    let (at, _) = crate::projects::transition_context().map_err(why!(Error::Storage))?;
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
    let applied = sdk.call(&job, &flag)?;
    if applied["applied"] != true {
        return Err(Error::Transport);
    }
    sdk.shutdown()?;
    job["operation"] = json!("observe");
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
