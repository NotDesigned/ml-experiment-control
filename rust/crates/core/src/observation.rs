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
fn model_field(name: &str) -> bool {
    matches!(
        name,
        "latest_metric"
            | "metric_log_lines"
            | "model_observed"
            | "step"
            | "optimizer_step"
            | "latest_completed_checkpoint"
            | "latest_completed_checkpoint_step"
            | "artifacts"
            | "evaluation_variants"
    ) || name != "model_state"
        && [
            "latest_metric_",
            "metric_",
            "checkpoint_",
            "artifact_",
            "evaluation_",
            "model_evidence_",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}
fn metric_step(value: &Value) -> Option<f64> {
    let map = value.as_object()?;
    match map.get("step").or_else(|| map.get("optimizer_step"))? {
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// Retain omitted model evidence only after an exact Run becomes terminal.
/// Scheduler/worker/process lifecycle fields from the current observation win.
pub fn merge_terminal_observation(
    previous: Option<&Map<String, Value>>,
    current: &Map<String, Value>,
) -> Value {
    let mut result = current.clone();
    let Some(previous) = previous.filter(|m| !m.is_empty()) else {
        return Value::Object(result);
    };
    let terminal = current
        .get("scheduler_state")
        .and_then(Value::as_str)
        .is_some_and(|s| {
            matches!(
                s.to_ascii_uppercase().as_str(),
                "SUCCEEDED" | "FAILED" | "PREEMPTED" | "CANCELLED"
            )
        });
    if !terminal {
        return Value::Object(result);
    }
    let none = Value::Null;
    let old_run = previous.get("run_id").unwrap_or(&none);
    let new_run = current.get("run_id").unwrap_or(&none);
    if truth(old_run) && truth(new_run) && !python_equal(old_run, new_run) {
        return Value::Object(result);
    }
    let mut retained = Vec::new();
    for (name, value) in previous {
        if !model_field(name) || !present(value) {
            continue;
        }
        let current_value = result.get(name).unwrap_or(&none);
        if name == "latest_metric" {
            let stronger = present(current_value)
                && match (metric_step(value), metric_step(current_value)) {
                    (Some(a), Some(b)) => b >= a,
                    _ => true,
                };
            if stronger {
                continue;
            }
        } else if name == "artifacts" {
            if let (Some(old), Some(new)) = (value.as_object(), current_value.as_object()) {
                let mut merged = old.clone();
                merged.extend(new.clone());
                if !python_equal(&Value::Object(merged.clone()), current_value) {
                    result.insert(name.clone(), Value::Object(merged));
                    retained.push(name.clone());
                }
                continue;
            }
            if present(current_value) {
                continue;
            }
        } else if present(current_value) {
            continue;
        }
        result.insert(name.clone(), value.clone());
        retained.push(name.clone());
    }
    if let Some(previous_process) = previous.get("process_evidence").and_then(Value::as_object) {
        let mut process = result
            .get("process_evidence")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let mut streams = Vec::new();
        for name in ["stdout_tail", "stderr_tail"] {
            let old = previous_process.get(name).unwrap_or(&none);
            if present(process.get(name).unwrap_or(&none)) || !present(old) {
                continue;
            }
            process.insert(name.into(), old.clone());
            streams.push(name);
        }
        if !streams.is_empty() {
            let mut sources = previous_process
                .get("sources")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if let Some(new) = process.get("sources").and_then(Value::as_object) {
                sources.extend(new.clone());
            }
            process.insert("observed".into(), json!(true));
            process.insert("sources".into(), Value::Object(sources));
            process.insert("retained".into(), json!(true));
            process.insert("retained_streams".into(), json!(streams));
            result.insert("process_evidence".into(), Value::Object(process));
            retained.extend(
                streams
                    .iter()
                    .map(|name| format!("process_evidence.{name}")),
            );
        }
    }
    if retained
        .iter()
        .any(|name| !name.starts_with("process_evidence."))
    {
        result.insert("model_observed".into(), json!(true));
        result.insert("model_state".into(), json!("OBSERVED"));
    }
    if !retained.is_empty() {
        retained.sort();
        result.insert("evidence_unavailable_reason".into(), Value::Null);
        result.insert("evidence_outcome".into(), json!("OBSERVED"));
        result.insert("retained_evidence".into(),json!({"retained":true,"fields":retained,"reason":"terminal observation omitted previously observed evidence","source_scheduler_state":previous.get("scheduler_state")}));
    }
    Value::Object(result)
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
