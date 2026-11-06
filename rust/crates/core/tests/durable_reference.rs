//! Journal recovery and chaining over the recoverable head and transition log.
use ml_exp_core::durable::{self, Snapshot};
use serde_json::{Value, json};

/// Commit `count` successive changes, returning the final head and its journal.
fn chain(count: u64) -> (Value, Vec<Value>) {
    let mut head = json!({"schema_version":1,"projects":[]});
    let mut events: Vec<Value> = Vec::new();
    for revision in 1..=count {
        let state = ["PAUSED", "ARCHIVED", "ACTIVE"][(revision as usize - 1) % 3];
        let payload = json!({"schema_version":1,"projects":[{"project":"demo","state":state}],
            "extra":{"n":revision,"score":0.1,"unicode":"科学"}});
        let at = format!("2026-10-27T00:00:{revision:02}Z");
        let event = json!({"event":"CHANGE","project":"demo","at":at,"detail":{"n":revision}});
        let identity = format!("{revision:032x}");
        let snapshot = Snapshot::from_value(&head).unwrap();
        assert_eq!(snapshot.revision, revision - 1);
        let plan = durable::commit(
            &snapshot,
            &events,
            &payload,
            &event,
            Some(revision - 1),
            &identity,
        )
        .unwrap();
        let transition = json!({"event":"CHANGE","project":"demo","at":at,"detail":{"n":revision},
            "revision":revision,"transition_id":identity});
        assert_eq!(plan.revision, revision);
        assert_eq!(plan.transition, transition);
        let mut expected = payload;
        expected["_durability"] = json!({"revision":revision,"last_transition":transition});
        assert_eq!(plan.head, expected);
        // A crash after publishing the head and before the append.
        let next = Snapshot::from_value(&plan.head).unwrap();
        assert_eq!(
            durable::missing_transition(&next, &events).unwrap(),
            Some(plan.transition.clone())
        );
        events.push(plan.transition);
        assert_eq!(durable::missing_transition(&next, &events).unwrap(), None);
        head = plan.head;
    }
    (head, events)
}

#[test]
fn each_commit_chains_from_the_last_and_keeps_unicode_and_floats_in_the_payload() {
    let (head, events) = chain(8);
    assert_eq!(events.len(), 8);
    assert_eq!(head["extra"], json!({"n":8,"score":0.1,"unicode":"科学"}));
    assert_eq!(head["_durability"]["revision"], 8);
    assert_eq!(head["_durability"]["last_transition"], events[7]);
    assert_eq!(durable::validate_journal(&events).unwrap(), 8);
}

#[test]
fn a_stale_revision_repeated_identity_or_malformed_identity_never_commits() {
    let (head, events) = chain(2);
    let snapshot = Snapshot::from_value(&head).unwrap();
    let payload = json!({"schema_version":1,"projects":[]});
    let event = json!({"event":"CHANGE"});
    let fresh = format!("{:032x}", 3);
    assert!(durable::commit(&snapshot, &events, &payload, &event, Some(2), &fresh).is_ok());
    assert!(durable::commit(&snapshot, &events, &payload, &event, None, &fresh).is_ok());
    assert!(durable::commit(&snapshot, &events, &payload, &event, Some(1), &fresh).is_err());
    let used = format!("{:032x}", 1);
    assert!(durable::commit(&snapshot, &events, &payload, &event, Some(2), &used).is_err());
    for identity in [
        "",
        "short",
        &"G".repeat(32),
        &"A".repeat(32),
        &"a".repeat(33),
    ] {
        assert!(
            durable::commit(&snapshot, &events, &payload, &event, Some(2), identity).is_err(),
            "{identity:?}"
        );
    }
    // A journal missing the head's transition must be reconciled before any commit.
    assert!(durable::commit(&snapshot, &events[..1], &payload, &event, Some(2), &fresh).is_err());
}

#[test]
fn only_the_final_transition_is_recoverable() {
    let (head, events) = chain(3);
    let snapshot = Snapshot::from_value(&head).unwrap();
    assert_eq!(
        durable::missing_transition(&snapshot, &events[..2]).unwrap(),
        Some(events[2].clone())
    );
    // Older history is never rebuilt.
    assert!(durable::missing_transition(&snapshot, &events[..1]).is_err());
    assert!(durable::missing_transition(&snapshot, &[]).is_err());
    assert!(durable::missing_transition(&snapshot, &events[1..]).is_err());
    let duplicated = [
        events[0].clone(),
        events[0].clone(),
        events[1].clone(),
        events[2].clone(),
    ];
    assert!(durable::missing_transition(&snapshot, &duplicated).is_err());
    // The head and the journal disagree about the same transition.
    let mut changed = events.clone();
    changed[2]["detail"] = json!({"n":"different evidence"});
    assert!(durable::missing_transition(&snapshot, &changed).is_err());
}

#[test]
fn an_event_without_a_transition_id_keeps_its_place_without_advancing_the_revision() {
    let (head, mut events) = chain(2);
    let note = json!({"event":"NOTE","detail":"科学"});
    events.insert(1, note.clone());
    events.push(note);
    assert_eq!(durable::validate_journal(&events).unwrap(), 2);
    let snapshot = Snapshot::from_value(&head).unwrap();
    assert_eq!(
        durable::missing_transition(&snapshot, &events).unwrap(),
        None
    );
    assert!(durable::validate_journal(&[json!([])]).is_err());
}
