//! Backend-neutral observational projections. Never refresh a provider or grant retries.
use serde_json::{Map, Value};

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
