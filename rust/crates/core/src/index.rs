//! Pure decisions for the disposable Run query index. Files remain canonical.
use crate::observation::{merge_terminal_observation, python_equal};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid Run index row: {0}")]
    Invalid(&'static str),
    #[error("Run index payload does not match its exact project/Run key")]
    Identity,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    pub project: String,
    pub run_id: String,
}
impl Key {
    pub fn from_row(row: &Value) -> Result<Self, Error> {
        Ok(Self {
            project: string(row, "project")?.into(),
            run_id: string(row, "run_id")?.into(),
        })
    }
}
fn string<'a>(value: &'a Value, name: &'static str) -> Result<&'a str, Error> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or(Error::Invalid(name))
}

/// Accept legacy observation IDs as data, never as filesystem path segments.
/// Preserve all unknown metadata; only key/SQL projection fields are inspected.
pub fn validate_row(row: &Value, expected: Option<&Key>) -> Result<Key, Error> {
    if !row.is_object() {
        return Err(Error::Invalid("JSON object required"));
    }
    let key = Key::from_row(row)?;
    if expected.is_some_and(|expected| *expected != key) {
        return Err(Error::Identity);
    }
    string(row, "run_dir")?;
    for name in ["campaign", "role", "scheduler_state"] {
        if row
            .get(name)
            .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            return Err(Error::Invalid(name));
        }
    }
    if let Some(value) = row.get("scanned_at").filter(|value| !value.is_null()) {
        if !value.as_f64().is_some_and(f64::is_finite) {
            return Err(Error::Invalid("scanned_at"));
        }
    }
    Ok(key)
}

/// A scan timestamp alone never signals a scientific or operational update.
/// Python's deep number/bool equality is deliberate compatibility behavior.
pub fn changed(previous: Option<&Value>, current: &Value) -> Result<bool, Error> {
    validate_row(current, None)?;
    let Some(previous) = previous else {
        return Ok(true);
    };
    let mut previous = previous
        .as_object()
        .cloned()
        .ok_or(Error::Invalid("stored row"))?;
    let mut current = current
        .as_object()
        .cloned()
        .ok_or(Error::Invalid("current row"))?;
    previous.shift_remove("scanned_at");
    current.shift_remove("scanned_at");
    Ok(!python_equal(
        &Value::Object(previous),
        &Value::Object(current),
    ))
}

pub fn matches_campaign(row: &Value, campaign: &str) -> bool {
    row.get("campaign").and_then(Value::as_str) == Some(campaign)
        || row
            .get("campaign_memberships")
            .and_then(Value::as_array)
            .is_some_and(|members| {
                members
                    .iter()
                    .any(|member| member.get("campaign").and_then(Value::as_str) == Some(campaign))
            })
}

fn truth(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Number(number) => number.as_f64().is_none_or(|n| n != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
        Value::Bool(true) => true,
    }
}

/// Merge the current verb into collector status without clearing another verb's
/// failure. No current time, transport refresh or retry decision is invented.
pub fn poll(
    previous: &Value,
    verb: &str,
    error: Option<&str>,
    now: f64,
    outcome: Option<&str>,
) -> Result<Value, Error> {
    if !now.is_finite() {
        return Err(Error::Invalid("poll timestamp"));
    }
    let mut results = previous.as_object().cloned().unwrap_or_default();
    let resolved = outcome.filter(|value| !value.is_empty()).unwrap_or(
        if error.is_some_and(|s| !s.is_empty()) {
            "failed"
        } else {
            "succeeded"
        },
    );
    results.insert(
        verb.into(),
        json!({"outcome":resolved,"at":now,"error":error}),
    );
    let errors: Vec<String> = results
        .values()
        .filter_map(|record| record.get("error"))
        .filter(|error| truth(error))
        .map(crate::metrics::python_string)
        .collect();
    Ok(
        json!({"last_poll_at":now,"last_verb":verb,"last_error":if errors.is_empty(){None}else{Some(errors.join("; "))},"verb_results":results}),
    )
}

/// This checked entry is stricter than the historical field-merging helper:
/// evidence is carried forward only when both inputs name the same complete
/// project/Run/Attempt. Missing identity means retain current evidence only.
pub fn merge_attempt(previous: Option<&Map<String, Value>>, current: &Map<String, Value>) -> Value {
    let exact = previous.is_some_and(|previous| {
        ["project", "run_id", "attempt_id"].iter().all(|name| {
            current
                .get(*name)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .is_some_and(|expected| {
                    previous.get(*name).and_then(Value::as_str) == Some(expected)
                })
        })
    });
    if exact {
        merge_terminal_observation(previous, current)
    } else {
        Value::Object(current.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn change_detection_ignores_only_scan_time_and_preserves_unknown_science() {
        let old = json!({"project":"p","run_id":"r","run_dir":"/r","scanned_at":1,"future":{"loss":1,"count":true}});
        let mut next = old.clone();
        next["scanned_at"] = json!(2);
        next["future"] = json!({"loss":1.0,"count":1});
        assert!(!changed(Some(&old), &next).unwrap());
        next["future"]["loss"] = json!(2);
        assert!(changed(Some(&old), &next).unwrap());
        assert!(
            validate_row(
                &old,
                Some(&Key {
                    project: "foreign".into(),
                    run_id: "r".into()
                })
            )
            .is_err()
        );
    }
    #[test]
    fn terminal_merge_requires_exact_attempt_even_for_the_same_run() {
        let old = json!({"project":"p","run_id":"r","attempt_id":"attempt-001","scheduler_state":"RUNNING","latest_metric":{"step":10,"loss":2.5},"process_evidence":{"stdout_tail":"training"}});
        let mut terminal = json!({"project":"p","run_id":"r","attempt_id":"attempt-001","scheduler_state":"FAILED","process_evidence":{"state":"EXITED"}});
        let merged = merge_attempt(old.as_object(), terminal.as_object().unwrap());
        assert_eq!(merged["latest_metric"]["step"], 10);
        assert_eq!(merged["scheduler_state"], "FAILED");
        assert_eq!(merged["process_evidence"]["state"], "EXITED");
        terminal["attempt_id"] = json!("attempt-002");
        assert_eq!(
            merge_attempt(old.as_object(), terminal.as_object().unwrap()),
            terminal
        );
        terminal.as_object_mut().unwrap().shift_remove("attempt_id");
        assert_eq!(
            merge_attempt(old.as_object(), terminal.as_object().unwrap()),
            terminal
        );
    }
}
