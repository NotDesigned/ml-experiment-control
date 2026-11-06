//! Backend-neutral observational projections. Never refresh a provider or grant retries.
use serde_json::{Map, Value};

/// JSON equality where numbers and bools compare by numeric value, so step 1 and
/// 1.0 are one point. For scientific logic only: own records and identities use
/// `canonical::equal`.
pub fn numeric_tolerant_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(_) | Value::Bool(_), Value::Number(_) | Value::Bool(_)) => {
            crate::metrics::numeric_equal(left, right)
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(k, v)| {
                    b.get(k)
                        .is_some_and(|other| numeric_tolerant_equal(v, other))
                })
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| numeric_tolerant_equal(a, b))
        }
        _ => left == right,
    }
}

/// Remove producer failure classifications before attaching daemon assessments.
pub fn sanitized_outward(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(key, _)| key.as_str() != "failure_class")
                .map(|(key, value)| (key.clone(), sanitized_outward(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(sanitized_outward).collect()),
        _ => value.clone(),
    }
}
