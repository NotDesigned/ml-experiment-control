//! Pure W&B publication planning.
//!
//! The publisher shell owns the SDK session, HTTP reads/writes, credentials,
//! durable outbox and clocks.  This module only consumes an explicit snapshot
//! of local events, the local session state and remote evidence.  It decides
//! whether the shell may emit a history row, must reconcile an uncertain
//! remote history, should publish display metadata, or may finish a terminal
//! run.  A missing or ambiguous remote observation never becomes an ACK.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid publication state: {0}")]
    State(&'static str),
    #[error("invalid publication event: {0}")]
    Event(&'static str),
    #[error("invalid remote history: {0}")]
    Remote(&'static str),
}
pub type Result<T> = std::result::Result<T, Error>;

/// W&B flattens config dictionaries and omits empty mappings, including parents
/// whose only children were empty mappings. Accept only that observed encoding;
/// nonempty values, nulls, arrays and unexpected keys retain exact identity.
pub fn config_matches(expected: &serde_json::Value, observed: &serde_json::Value) -> bool {
    fn empty_mapping(value: &serde_json::Value) -> bool {
        value
            .as_object()
            .is_some_and(|m| m.values().all(empty_mapping))
    }
    match (expected.as_object(), observed.as_object()) {
        (Some(expected), Some(observed)) => {
            observed.keys().all(|key| expected.contains_key(key))
                && expected.iter().all(|(key, value)| match observed.get(key) {
                    Some(observed) => config_matches(value, observed),
                    None => empty_mapping(value),
                })
        }
        _ => expected == observed,
    }
}

/// One accepted journal event that may be sent to W&B history.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEvent {
    pub sequence: i64,
    pub event_id: String,
}

/// A sequence/event identity observed in W&B history.  The shell obtains this
/// from `scan_history`; the reducer does not trust an integer cursor alone.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteHistoryEvent {
    pub sequence: i64,
    pub event_id: String,
}

/// The exact remote evidence needed for one publication decision.  `None`
/// means the provider did not return that observation; it is never interpreted
/// as an empty or successful value.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteEvidence {
    pub history: Vec<RemoteHistoryEvent>,
    /// W&B's zero-based `lastHistoryStep`; `None` means no usable value was
    /// returned.  A value of -1 denotes an empty history.
    pub last_history_step: Option<i64>,
    pub display_version: Option<i64>,
    pub display_through: Option<i64>,
}

/// Ephemeral/durable state known before this reducer call.  `sent` is the
/// publisher-session reservation: it records rows handed to the SDK but not
/// yet proven by a subsequent remote history read.  The shell may persist it
/// in its outbox, but it must not advance `confirmed` from this field.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationState {
    pub confirmed: i64,
    pub sent: BTreeMap<i64, String>,
    pub display_version: Option<i64>,
    pub display_through: Option<i64>,
    pub finished: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationAction {
    EmitHistory { sequence: i64, event_id: String },
    PublishDisplay { through: i64, version: i64 },
    Finish { through: i64 },
}

/// Statuses intentionally distinguish a local pending SDK write from a write
/// that was attempted and now awaits remote history ACK.  This keeps a lost
/// SDK response from looking like an exact remote confirmation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PublicationStatus {
    Pending {
        through: i64,
        emits: usize,
    },
    AckPending {
        confirmed: i64,
        through: i64,
    },
    DisplayAckPending {
        through: i64,
        version: i64,
    },
    Synced {
        through: i64,
    },
    Finished {
        through: i64,
    },
    ReconcileRequired {
        reason: String,
    },
    IdentityConflict {
        sequence: i64,
        expected: String,
        observed: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationPlan {
    pub state: PublicationState,
    pub through: i64,
    pub actions: Vec<PublicationAction>,
    pub status: PublicationStatus,
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && value.bytes().all(|byte| !byte.is_ascii_control())
}

fn validate_event(event: &HistoryEvent) -> Result<()> {
    if event.sequence <= 0 {
        return Err(Error::Event("positive sequence"));
    }
    if !valid_id(&event.event_id) {
        return Err(Error::Event("event_id"));
    }
    Ok(())
}

fn validate_remote_event(event: &RemoteHistoryEvent) -> Result<()> {
    if event.sequence <= 0 {
        return Err(Error::Remote("positive sequence"));
    }
    if !valid_id(&event.event_id) {
        return Err(Error::Remote("event_id"));
    }
    Ok(())
}

fn validate_state(state: &PublicationState) -> Result<()> {
    if state.confirmed < 0 {
        return Err(Error::State("non-negative confirmed cursor"));
    }
    if state.display_version.is_some_and(|version| version <= 0) {
        return Err(Error::State("positive display version"));
    }
    if state.display_through.is_some_and(|through| through < 0) {
        return Err(Error::State("non-negative display cursor"));
    }
    for (&sequence, event_id) in &state.sent {
        if sequence <= 0 || !valid_id(event_id) {
            return Err(Error::State("sent history reservation"));
        }
    }
    Ok(())
}

fn remote_index(remote: &RemoteEvidence) -> Result<BTreeMap<i64, String>> {
    if remote.last_history_step.is_some_and(|step| step < -1) {
        return Err(Error::Remote("last_history_step"));
    }
    let mut index = BTreeMap::new();
    for event in &remote.history {
        validate_remote_event(event)?;
        match index.insert(event.sequence, event.event_id.clone()) {
            None => {}
            Some(previous) if previous == event.event_id => {}
            Some(_) => return Err(Error::Remote("conflicting duplicate sequence")),
        }
    }
    Ok(index)
}

fn reconcile(reason: &str) -> PublicationStatus {
    PublicationStatus::ReconcileRequired {
        reason: reason.into(),
    }
}

/// Reduce one publisher cycle to deterministic actions and a next state.
///
/// `events` must be the exact local journal suffix after `state.confirmed`,
/// although replaying already-confirmed events is accepted when remote history
/// proves their identities.  The shell must execute `EmitHistory` and
/// `PublishDisplay`, then call this reducer again with fresh remote evidence.
/// It may call `Finish` only after the returned status is `Finished`.
pub fn plan(
    state: &PublicationState,
    remote: &RemoteEvidence,
    events: &[HistoryEvent],
    expected_event_id: Option<&str>,
    terminal: bool,
    display_version: i64,
) -> Result<PublicationPlan> {
    validate_state(state)?;
    if display_version <= 0 {
        return Err(Error::State("positive display version"));
    }
    if expected_event_id.is_some_and(|event_id| !valid_id(event_id)) {
        return Err(Error::Event("expected event_id"));
    }
    let mut previous = None;
    for event in events {
        validate_event(event)?;
        if previous.is_some_and(|sequence| event.sequence <= sequence) {
            return Err(Error::Event("strictly increasing event sequence"));
        }
        previous = Some(event.sequence);
    }
    let seen = remote_index(remote)?;
    let remote_next = match remote.last_history_step {
        Some(step) => Some(
            step.checked_add(1)
                .ok_or(Error::Remote("last_history_step overflow"))?,
        ),
        None => None,
    };
    let maximum = seen
        .keys()
        .next_back()
        .copied()
        .unwrap_or(state.confirmed)
        .max(remote_next.unwrap_or(state.confirmed));

    // The expected identity is a narrow proof for the durable cursor.  A
    // missing or mismatching row means the provider read is incomplete or
    // points at another Run; neither case is safe to repair by writing.
    if state.confirmed > 0
        && expected_event_id.is_some()
        && seen.get(&state.confirmed).map(String::as_str) != expected_event_id
    {
        return Ok(PublicationPlan {
            state: state.clone(),
            through: state.confirmed,
            actions: Vec::new(),
            status: reconcile("REMOTE_HISTORY_RECONCILE_REQUIRED"),
        });
    }

    let mut next = state.clone();
    let mut actions = Vec::new();
    let mut confirmed = state.confirmed;
    for event in events {
        if let Some(remote_id) = seen.get(&event.sequence) {
            if remote_id != &event.event_id {
                return Ok(PublicationPlan {
                    state: state.clone(),
                    through: events.last().map_or(state.confirmed, |item| item.sequence),
                    actions: Vec::new(),
                    status: PublicationStatus::IdentityConflict {
                        sequence: event.sequence,
                        expected: event.event_id.clone(),
                        observed: remote_id.clone(),
                    },
                });
            }
            confirmed = confirmed.max(event.sequence);
            next.sent.remove(&event.sequence);
            continue;
        }
        if event.sequence <= maximum {
            return Ok(PublicationPlan {
                state: state.clone(),
                through: events.last().map_or(state.confirmed, |item| item.sequence),
                actions: Vec::new(),
                status: reconcile("REMOTE_HISTORY_RECONCILE_REQUIRED"),
            });
        }
        match next.sent.get(&event.sequence) {
            Some(sent_id) if sent_id != &event.event_id => {
                return Ok(PublicationPlan {
                    state: state.clone(),
                    through: events.last().map_or(state.confirmed, |item| item.sequence),
                    actions: Vec::new(),
                    status: PublicationStatus::IdentityConflict {
                        sequence: event.sequence,
                        expected: event.event_id.clone(),
                        observed: sent_id.clone(),
                    },
                });
            }
            Some(_) => {}
            None => {
                next.sent.insert(event.sequence, event.event_id.clone());
                actions.push(PublicationAction::EmitHistory {
                    sequence: event.sequence,
                    event_id: event.event_id.clone(),
                });
            }
        }
    }
    next.confirmed = confirmed;
    let through = events.last().map_or(confirmed, |item| item.sequence);

    // A local SDK send is not an ACK.  Every event has to be present in the
    // remote identity prefix before display metadata or terminal finish may
    // advance.
    if !actions.is_empty() {
        return Ok(PublicationPlan {
            state: next,
            through,
            status: PublicationStatus::Pending {
                through,
                emits: actions.len(),
            },
            actions,
        });
    }
    if confirmed < through {
        return Ok(PublicationPlan {
            state: next,
            through,
            actions,
            status: PublicationStatus::AckPending { confirmed, through },
        });
    }

    let display_ack =
        remote.display_version == Some(display_version) && remote.display_through == Some(through);
    if !display_ack {
        actions.push(PublicationAction::PublishDisplay {
            through,
            version: display_version,
        });
        return Ok(PublicationPlan {
            state: next,
            through,
            actions,
            status: PublicationStatus::DisplayAckPending {
                through,
                version: display_version,
            },
        });
    }

    next.display_version = Some(display_version);
    next.display_through = Some(through);
    if terminal {
        next.finished = true;
        actions.push(PublicationAction::Finish { through });
        Ok(PublicationPlan {
            state: next,
            through,
            actions,
            status: PublicationStatus::Finished { through },
        })
    } else {
        Ok(PublicationPlan {
            state: next,
            through,
            actions,
            status: PublicationStatus::Synced { through },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_config_accepts_only_wandb_empty_mapping_elision() {
        use serde_json::json;
        let expected = json!({"run_id":"train","parameters":{},"protocol":{"definitions":{},"id":"p1"},"null":null,"list":[{}]});
        let visible = json!({"run_id":"train","protocol":{"id":"p1"},"null":null,"list":[{}]});
        assert!(config_matches(&expected, &expected));
        assert!(config_matches(&expected, &visible));
        for different in [
            json!({"run_id":"other","protocol":{"id":"p1"},"null":null,"list":[{}]}),
            json!({"run_id":"train","protocol":{},"null":null,"list":[{}]}),
            json!({"run_id":"train","protocol":{"id":"p1"},"list":[{}]}),
            json!({"run_id":"train","protocol":{"id":"p1"},"null":null,"list":[]}),
            json!({"run_id":"train","protocol":{"id":"p1"},"null":null,"list":[{}],"foreign":{}}),
        ] {
            assert!(!config_matches(&expected, &different));
        }
        assert!(config_matches(&json!({"nested":{"empty":{}}}), &json!({})));
    }

    fn event(sequence: i64, event_id: &str) -> HistoryEvent {
        HistoryEvent {
            sequence,
            event_id: event_id.into(),
        }
    }
    fn remote(history: &[(i64, &str)]) -> RemoteEvidence {
        RemoteEvidence {
            history: history
                .iter()
                .map(|(sequence, event_id)| RemoteHistoryEvent {
                    sequence: *sequence,
                    event_id: (*event_id).into(),
                })
                .collect(),
            ..RemoteEvidence::default()
        }
    }

    fn reduce(
        state: &PublicationState,
        remote: &RemoteEvidence,
        events: &[HistoryEvent],
        expected_event_id: Option<&str>,
        terminal: bool,
        display_version: i64,
    ) -> Result<PublicationPlan> {
        super::plan(
            state,
            remote,
            events,
            expected_event_id,
            terminal,
            display_version,
        )
    }

    #[test]
    fn first_cycle_emits_without_calling_it_an_ack() {
        let plan = reduce(
            &PublicationState::default(),
            &RemoteEvidence::default(),
            &[event(1, "e1"), event(2, "e2")],
            None,
            false,
            2,
        )
        .unwrap();
        assert_eq!(
            plan.status,
            PublicationStatus::Pending {
                through: 2,
                emits: 2
            }
        );
        assert_eq!(plan.state.confirmed, 0);
        assert_eq!(plan.state.sent.len(), 2);
        assert_eq!(plan.actions.len(), 2);
    }

    #[test]
    fn sent_rows_wait_for_remote_history_ack() {
        let mut state = PublicationState::default();
        state.sent.insert(1, "e1".into());
        state.sent.insert(2, "e2".into());
        let plan = reduce(
            &state,
            &RemoteEvidence::default(),
            &[event(1, "e1"), event(2, "e2")],
            None,
            false,
            2,
        )
        .unwrap();
        assert_eq!(
            plan.status,
            PublicationStatus::AckPending {
                confirmed: 0,
                through: 2
            }
        );
        assert!(plan.actions.is_empty());
    }

    #[test]
    fn remote_ack_is_identity_based_and_clears_reservations() {
        let mut state = PublicationState::default();
        state.sent.insert(1, "e1".into());
        state.sent.insert(2, "e2".into());
        let mut remote = remote(&[(1, "e1"), (2, "e2")]);
        remote.display_version = Some(2);
        remote.display_through = Some(2);
        let plan = reduce(
            &state,
            &remote,
            &[event(1, "e1"), event(2, "e2")],
            Some("e1"),
            false,
            2,
        )
        .unwrap();
        assert_eq!(plan.status, PublicationStatus::Synced { through: 2 });
        assert_eq!(plan.state.confirmed, 2);
        assert!(plan.state.sent.is_empty());
        assert_eq!(plan.state.display_version, Some(2));
    }

    #[test]
    fn same_sequence_with_different_event_id_is_conflict() {
        let plan = reduce(
            &PublicationState::default(),
            &remote(&[(1, "remote")]),
            &[event(1, "local")],
            None,
            false,
            2,
        )
        .unwrap();
        assert_eq!(
            plan.status,
            PublicationStatus::IdentityConflict {
                sequence: 1,
                expected: "local".into(),
                observed: "remote".into()
            }
        );
        assert!(plan.actions.is_empty());
    }

    #[test]
    fn a_gap_or_ahead_remote_cursor_requires_reconcile() {
        let plan = reduce(
            &PublicationState::default(),
            &remote(&[(1, "e1"), (3, "e3")]),
            &[event(1, "e1"), event(2, "e2")],
            None,
            false,
            2,
        )
        .unwrap();
        assert_eq!(
            plan.status,
            PublicationStatus::ReconcileRequired {
                reason: "REMOTE_HISTORY_RECONCILE_REQUIRED".into()
            }
        );
        let ahead = RemoteEvidence {
            last_history_step: Some(4),
            ..RemoteEvidence::default()
        };
        let plan = reduce(
            &PublicationState::default(),
            &ahead,
            &[event(1, "e1")],
            None,
            false,
            2,
        )
        .unwrap();
        assert!(matches!(
            plan.status,
            PublicationStatus::ReconcileRequired { .. }
        ));
    }

    #[test]
    fn display_ack_is_separate_from_history_ack() {
        let remote = remote(&[(1, "e1")]);
        let plan = reduce(
            &PublicationState::default(),
            &remote,
            &[event(1, "e1")],
            None,
            false,
            2,
        )
        .unwrap();
        assert_eq!(
            plan.status,
            PublicationStatus::DisplayAckPending {
                through: 1,
                version: 2
            }
        );
        assert_eq!(
            plan.actions,
            vec![PublicationAction::PublishDisplay {
                through: 1,
                version: 2
            }]
        );
    }

    #[test]
    fn terminal_finish_requires_display_ack() {
        let mut remote = remote(&[(1, "e1")]);
        remote.display_version = Some(2);
        remote.display_through = Some(1);
        let plan = reduce(
            &PublicationState::default(),
            &remote,
            &[event(1, "e1")],
            None,
            true,
            2,
        )
        .unwrap();
        assert_eq!(plan.status, PublicationStatus::Finished { through: 1 });
        assert_eq!(plan.actions, vec![PublicationAction::Finish { through: 1 }]);
        assert!(plan.state.finished);
    }

    #[test]
    fn expected_cursor_identity_missing_from_remote_is_not_ack() {
        let plan = reduce(
            &PublicationState {
                confirmed: 2,
                ..PublicationState::default()
            },
            &RemoteEvidence::default(),
            &[],
            Some("e2"),
            false,
            2,
        )
        .unwrap();
        assert!(matches!(
            plan.status,
            PublicationStatus::ReconcileRequired { .. }
        ));
        assert_eq!(plan.state.confirmed, 2);
    }

    #[test]
    fn invalid_duplicate_remote_sequence_is_rejected() {
        let remote = remote(&[(1, "e1"), (1, "different")]);
        assert!(matches!(
            reduce(&PublicationState::default(), &remote, &[], None, false, 2),
            Err(Error::Remote("conflicting duplicate sequence"))
        ));
    }
}
