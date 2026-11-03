//! Recoverable JSON head + contiguous transition journal. All values, time and
//! transition identities are explicit; this module never repairs a file.
use crate::observation::python_equal;
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid durable state or transition history")]
pub struct Error;

#[derive(Clone)]
pub struct Snapshot {
    pub value: Value,
    pub revision: u64,
    pub last_transition: Option<Value>,
}

impl Snapshot {
    pub fn from_value(raw: &Value) -> Result<Self, Error> {
        let mut value = raw.as_object().cloned().ok_or(Error)?;
        let Some(metadata) = value.shift_remove("_durability") else {
            return Ok(Self {
                value: Value::Object(value),
                revision: 0,
                last_transition: None,
            });
        };
        if metadata.is_null() {
            return Ok(Self {
                value: Value::Object(value),
                revision: 0,
                last_transition: None,
            });
        }
        let revision = metadata
            .get("revision")
            .and_then(Value::as_u64)
            .filter(|n| *n > 0)
            .ok_or(Error)?;
        let transition = metadata
            .get("last_transition")
            .filter(|v| v.is_object())
            .ok_or(Error)?;
        if transition.get("revision").and_then(Value::as_u64) != Some(revision) {
            return Err(Error);
        }
        transition_id(transition)?;
        Ok(Self {
            value: Value::Object(value),
            revision,
            last_transition: Some(transition.clone()),
        })
    }
}

fn transition_id(value: &Value) -> Result<&str, Error> {
    value
        .get("transition_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty() && s.len() <= 1024)
        .ok_or(Error)
}

/// Check every transition, including history before the current last event.
/// Non-state legacy events retain their order without advancing the revision.
pub fn validate_journal(events: &[Value]) -> Result<u64, Error> {
    let mut revision = 0u64;
    let mut identities = BTreeSet::new();
    for event in events {
        if !event.is_object() {
            return Err(Error);
        }
        if event.get("transition_id").is_none() {
            continue;
        }
        let id = transition_id(event)?;
        revision = revision.checked_add(1).ok_or(Error)?;
        if !identities.insert(id) || event.get("revision").and_then(Value::as_u64) != Some(revision)
        {
            return Err(Error);
        }
    }
    Ok(revision)
}

/// The sole recoverable omission is the head's complete final transition after
/// an intact immediate predecessor. A missing older prefix is never rebuilt.
pub fn missing_transition(snapshot: &Snapshot, events: &[Value]) -> Result<Option<Value>, Error> {
    let revision = validate_journal(events)?;
    let last = events
        .iter()
        .rev()
        .find(|event| event.get("transition_id").is_some());
    match (&snapshot.last_transition, last) {
        (None, None) if snapshot.revision == 0 => Ok(None),
        (Some(head), Some(last)) if transition_id(head)? == transition_id(last)? => {
            if revision == snapshot.revision && python_equal(head, last) {
                Ok(None)
            } else {
                Err(Error)
            }
        }
        (Some(head), _) if snapshot.revision == revision.checked_add(1).ok_or(Error)? => {
            let id = transition_id(head)?;
            if events
                .iter()
                .any(|event| event.get("transition_id").and_then(Value::as_str) == Some(id))
            {
                return Err(Error);
            }
            Ok(Some(head.clone()))
        }
        _ => Err(Error),
    }
}

pub struct Commit {
    pub head: Value,
    pub transition: Value,
    pub revision: u64,
}

/// Requires an already reconciled journal. The shell publishes `head` first,
/// then appends `transition`; it must report uncertainty after a possible write.
pub fn commit(
    snapshot: &Snapshot,
    events: &[Value],
    value: &Value,
    event: &Value,
    expected_revision: Option<u64>,
    identity: &str,
) -> Result<Commit, Error> {
    if missing_transition(snapshot, events)?.is_some()
        || expected_revision.is_some_and(|expected| expected != snapshot.revision)
        || identity.len() != 32
        || !identity
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || events
            .iter()
            .any(|v| v.get("transition_id").and_then(Value::as_str) == Some(identity))
    {
        return Err(Error);
    }
    let mut payload = value.as_object().cloned().ok_or(Error)?;
    if payload.contains_key("_durability") {
        return Err(Error);
    }
    let mut transition = event.as_object().cloned().ok_or(Error)?;
    if transition.contains_key("revision") || transition.contains_key("transition_id") {
        return Err(Error);
    }
    let revision = snapshot.revision.checked_add(1).ok_or(Error)?;
    transition.insert("revision".into(), json!(revision));
    transition.insert("transition_id".into(), json!(identity));
    let transition = Value::Object(transition);
    payload.insert(
        "_durability".into(),
        json!({"revision":revision,"last_transition":transition}),
    );
    Ok(Commit {
        head: Value::Object(payload),
        transition,
        revision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contiguous_history_only_recovers_the_missing_last_transition() {
        let zero = Snapshot::from_value(&json!({"schema_version":1})).unwrap();
        let first = commit(
            &zero,
            &[],
            &zero.value,
            &json!({"event":"BOOTSTRAP"}),
            Some(0),
            &"1".repeat(32),
        )
        .unwrap();
        let head = Snapshot::from_value(&first.head).unwrap();
        assert_eq!(
            missing_transition(&head, &[]).unwrap(),
            Some(first.transition.clone())
        );
        assert!(
            missing_transition(&head, std::slice::from_ref(&first.transition))
                .unwrap()
                .is_none()
        );
        let second = commit(
            &head,
            std::slice::from_ref(&first.transition),
            &head.value,
            &json!({"event":"REGISTER"}),
            Some(1),
            &"2".repeat(32),
        )
        .unwrap();
        let next = Snapshot::from_value(&second.head).unwrap();
        assert!(missing_transition(&next, &[]).is_err());
        assert!(missing_transition(&next, &[first.transition.clone(), first.transition]).is_err());
        assert!(missing_transition(&head, &[second.transition]).is_err());
    }
    #[test]
    fn invalid_metadata_cas_duplicates_and_reserved_keys_never_produce_a_commit() {
        let zero = Snapshot::from_value(&json!({})).unwrap();
        assert!(
            Snapshot::from_value(&json!({"_durability":{"revision":true,"last_transition":{}}}))
                .is_err()
        );
        for event in [
            json!([]),
            json!({"revision":1}),
            json!({"transition_id":"x"}),
        ] {
            assert!(commit(&zero, &[], &json!({}), &event, None, &"a".repeat(32)).is_err());
        }
        assert!(commit(&zero, &[], &json!({}), &json!({}), Some(1), &"a".repeat(32)).is_err());
        assert!(
            commit(
                &zero,
                &[],
                &json!({"_durability":{}}),
                &json!({}),
                None,
                &"a".repeat(32)
            )
            .is_err()
        );
        let first = commit(&zero, &[], &json!({}), &json!({}), None, &"a".repeat(32)).unwrap();
        let head = Snapshot::from_value(&first.head).unwrap();
        assert!(
            commit(
                &head,
                std::slice::from_ref(&first.transition),
                &json!({}),
                &json!({}),
                None,
                &"a".repeat(32)
            )
            .is_err()
        );
    }
}
