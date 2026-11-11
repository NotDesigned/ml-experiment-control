//! Pure W&B publication planning for a best-effort mirror.
//!
//! The journal, result archives and MLflow are the authoritative record. W&B is
//! sent each journal row once, in order, and the cursor advances when the SDK
//! accepted a batch; a failed send is retried from the cursor. Nothing here
//! reads W&B back. The publisher shell owns the SDK, the store and clocks.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The publication state kept per scope beside the journal's cursor.
///
/// Every field has a default and unknown fields are ignored, so a state saved
/// by an earlier release (with its reservations, repair records and duplicate
/// counts) still loads.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationState {
    #[serde(default)]
    pub finished: bool,
    /// Only read, never written: rows an earlier release handed to the SDK and
    /// was waiting to see in W&B history. They count as sent once.
    #[serde(default, skip_serializing)]
    pub sent: BTreeMap<i64, String>,
}

/// What one visit to a scope does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Send the rows after the cursor, with the display for their end.
    Send,
    /// Every row is sent but the summary for the cursor is not.
    Display,
    /// Every row and the display are sent and the Attempt ended.
    Finish,
    /// Nothing to send until the journal grows.
    Idle,
}

/// One scope as the shell observed it locally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// Rows the SDK accepted.
    pub cursor: i64,
    /// Rows in the journal.
    pub total: i64,
    /// The summary was sent for exactly `cursor` at the current display version.
    pub display_current: bool,
    /// The journal or the provider proved the Attempt ended.
    pub terminal: bool,
    /// The W&B Run is already finished for `total` rows.
    pub finished: bool,
}

pub fn next(progress: &Progress) -> Step {
    if progress.cursor < progress.total {
        Step::Send
    } else if !progress.display_current {
        Step::Display
    } else if progress.terminal && !progress.finished {
        Step::Finish
    } else {
        Step::Idle
    }
}

/// How far rows an earlier release reserved carry `cursor`: the reserved rows
/// that follow it without a gap and match the journal's identities. `events`
/// are the journal's `(sequence, event_id)` after `cursor`, in order.
pub fn legacy_cursor(state: &PublicationState, cursor: i64, events: &[(i64, &str)]) -> i64 {
    let mut through = cursor;
    for &(sequence, event_id) in events {
        if sequence != through + 1
            || state.sent.get(&sequence).map(String::as_str) != Some(event_id)
        {
            break;
        }
        through = sequence;
    }
    through
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(cursor: i64, total: i64) -> Progress {
        Progress {
            cursor,
            total,
            display_current: true,
            terminal: false,
            finished: false,
        }
    }

    #[test]
    fn rows_then_display_then_finish() {
        assert_eq!(next(&progress(0, 3)), Step::Send);
        let shown = Progress {
            display_current: false,
            ..progress(3, 3)
        };
        assert_eq!(next(&shown), Step::Display);
        assert_eq!(next(&progress(3, 3)), Step::Idle);
        let ended = Progress {
            terminal: true,
            ..progress(3, 3)
        };
        assert_eq!(next(&ended), Step::Finish);
        assert_eq!(
            next(&Progress {
                finished: true,
                ..ended
            }),
            Step::Idle
        );
        // An ended Attempt still sends its last rows before it finishes.
        assert_eq!(
            next(&Progress {
                terminal: true,
                ..progress(2, 3)
            }),
            Step::Send
        );
    }

    #[test]
    fn an_earlier_releases_state_loads_and_its_reservations_count_as_sent() {
        let legacy = serde_json::json!({
            "confirmed": 1,
            "sent": {"2": "e2", "3": "e3", "5": "e5"},
            "display_version": 4,
            "display_through": 1,
            "finished": false,
        });
        let state: PublicationState = serde_json::from_value(legacy).unwrap();
        let events = [(2, "e2"), (3, "e3"), (4, "e4"), (5, "e5")];
        assert_eq!(legacy_cursor(&state, 1, &events), 3, "stops at the gap");
        assert_eq!(legacy_cursor(&state, 1, &[(2, "other")]), 1);
        assert_eq!(legacy_cursor(&state, 3, &events[2..]), 3);
        let saved = serde_json::to_value(&state).unwrap();
        assert_eq!(saved, serde_json::json!({"finished": false}));
        let empty: PublicationState = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(empty, PublicationState::default());
    }
}
