//! Backend-neutral observational projections. Never refresh a provider or grant retries.
use serde_json::{Map, Value, json};

fn present(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::String(s) => !s.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        _ => true,
    }
}
fn truth(value: &Value) -> bool {
    match value {
        Value::Number(n) => n.as_f64().is_none_or(|n| n != 0.0),
        _ => present(value),
    }
}
/// Python JSON equality is deliberately separate from frozen byte identity.
pub fn python_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(_) | Value::Bool(_), Value::Number(_) | Value::Bool(_)) => {
            crate::metrics::numeric_equal(left, right)
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|other| python_equal(v, other)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| python_equal(a, b))
        }
        _ => left == right,
    }
}

pub fn operational_decision(value: &Value) -> Value {
    let mut result = Map::new();
    for key in [
        "action",
        "reason",
        "resume_checkpoint",
        "retries_allowed",
        "retries_used",
    ] {
        if let Some(value) = value.get(key).filter(|v| !v.is_null()) {
            result.insert(key.into(), value.clone());
        }
    }
    Value::Object(result)
}

/// Remove producer failure classifications before attaching daemon assessments.
pub fn sanitized_outward(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(key, _)| key.as_str() != "failure_class")
                .map(|(key, value)| {
                    (
                        key.clone(),
                        if key == "decision" {
                            operational_decision(value)
                        } else {
                            sanitized_outward(value)
                        },
                    )
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(sanitized_outward).collect()),
        _ => value.clone(),
    }
}

pub fn record_progress(
    previous: &Value,
    phase: &str,
    message: &str,
    at: &str,
    updated_unix: f64,
    timeout: Option<&Value>,
) -> Result<Value, &'static str> {
    if !updated_unix.is_finite() {
        return Err("invalid progress time");
    }
    let event = json!({"phase":phase,"message":message,"at":at});
    let mut events = previous
        .get("events")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    events.push(event.clone());
    let from = events.len().saturating_sub(64);
    Ok(
        json!({"phase":phase,"message":message,"at":at,"updated_unix":updated_unix,"timeout_seconds":timeout,"events":events[from..]}),
    )
}

pub fn progress_view(
    value: &Value,
    status: &str,
    active: bool,
    last_activity: Option<f64>,
    now: f64,
) -> Result<Value, &'static str> {
    if !now.is_finite() || last_activity.is_some_and(|n| !n.is_finite()) {
        return Err("invalid progress time");
    }
    let old = value.get("updated_unix").filter(|v| !v.is_null());
    let mut updated = old.cloned();
    if let Some(activity) = last_activity {
        let previous = old.and_then(Value::as_f64).unwrap_or(0.0);
        if !previous.is_finite() {
            return Err("invalid stored progress time");
        }
        updated = Some(if previous >= activity {
            old.filter(|v| truth(v)).cloned().unwrap_or(json!(0))
        } else {
            json!(activity)
        });
    }
    let age = match updated.as_ref() {
        Some(value) => {
            let at = value.as_f64().ok_or("invalid stored progress time")?;
            if !at.is_finite() {
                return Err("invalid stored progress time");
            }
            let age = now - at;
            Some(if age <= 0.0 { json!(0) } else { json!(age) })
        }
        None => None,
    };
    let stalled = active
        && age
            .as_ref()
            .and_then(Value::as_f64)
            .is_some_and(|n| n >= 120.0);
    Ok(
        json!({"status":status,"phase":if active {value.get("phase").cloned().unwrap_or(json!(status))} else {json!(status)},
        "message":value.get("message"),"phase_started_at":value.get("at"),"last_progress_unix":updated,
        "seconds_since_progress":age,"no_progress_warning":stalled,
        "diagnostic":if stalled {Some("No recorded progress for at least 120 seconds; inspect logs. This does not prove failure; do not resubmit.")} else {None},
        "phase_timeout_seconds":value.get("timeout_seconds"),"events":value.get("events").cloned().unwrap_or(json!([]))}),
    )
}
