//! Pure JSONL normalization. Invalid values retain failure evidence, never zero.
use serde_json::{Map, Number, Value, json};

use super::MetricError;

#[derive(Clone, Debug)]
pub enum MetricValue {
    Json(Value),
    /// Standard JSON cannot represent NaN/infinity. Typed producers use this
    /// variant so normalization can preserve the exact nonfinite diagnosis.
    Float(f64),
}

#[derive(Clone, Debug)]
pub struct ObservationInput {
    fields: Vec<(String, MetricValue)>,
}

impl ObservationInput {
    pub fn from_json(value: &Value) -> Result<Self, MetricError> {
        Ok(Self {
            fields: value
                .as_object()
                .ok_or(MetricError::Record)?
                .iter()
                .map(|(key, value)| (key.clone(), MetricValue::Json(value.clone())))
                .collect(),
        })
    }

    pub fn with_number(mut self, name: impl Into<String>, value: f64) -> Self {
        let name = name.into();
        if let Some((_, field)) = self.fields.iter_mut().find(|(key, _)| key == &name) {
            *field = MetricValue::Float(value);
        } else {
            self.fields.push((name, MetricValue::Float(value)));
        }
        self
    }

    fn get(&self, key: &str) -> Option<&MetricValue> {
        self.fields
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }
}

impl MetricValue {
    fn json(&self) -> Value {
        match self {
            Self::Json(value) => value.clone(),
            Self::Float(number) => Number::from_f64(*number)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        }
    }
    fn string(&self) -> Option<&str> {
        match self {
            Self::Json(Value::String(value)) => Some(value),
            _ => None,
        }
    }
    fn is_number(&self) -> bool {
        matches!(self, Self::Json(Value::Number(_)) | Self::Float(_))
    }
    fn finite(&self) -> bool {
        match self {
            Self::Json(value) => finite(value),
            Self::Float(value) => value.is_finite(),
        }
    }
    fn is_null(&self) -> bool {
        matches!(self, Self::Json(Value::Null))
    }
    fn python_string(&self) -> String {
        match self {
            Self::Json(value) => python_string(value),
            Self::Float(value) if value.is_nan() => "nan".to_owned(),
            Self::Float(value) if value.is_infinite() => if value.is_sign_negative() {
                "-inf"
            } else {
                "inf"
            }
            .to_owned(),
            Self::Float(value) => crate::canonical::python_float(*value).unwrap(),
        }
    }
}

pub fn finite(value: &Value) -> bool {
    value
        .as_number()
        .and_then(Number::as_f64)
        .is_some_and(f64::is_finite)
}

fn truncate(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}

fn repr_string(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut result = String::from(quote);
    for c in value.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c == quote => {
                result.push('\\');
                result.push(c);
            }
            c if !super::python_printability::is_printable(c) => {
                let code = u32::from(c);
                if code <= 255 {
                    result.push_str(&format!("\\x{code:02x}"));
                } else if code <= 65535 {
                    result.push_str(&format!("\\u{code:04x}"));
                } else {
                    result.push_str(&format!("\\U{code:08x}"));
                }
            }
            c => result.push(c),
        }
    }
    result.push(quote);
    result
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => repr_string(value),
        Value::Number(number) => {
            let raw = number.to_string();
            if raw.contains(['.', 'e', 'E']) {
                if let Some(number) = number.as_f64().filter(|number| number.is_finite()) {
                    crate::canonical::python_float(number).unwrap()
                } else {
                    if raw.starts_with('-') { "-inf" } else { "inf" }.to_owned()
                }
            } else {
                raw
            }
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{}: {}", repr_string(key), python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub(crate) fn python_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        _ => python_repr(value),
    }
}

pub(crate) fn printable(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.chars().count() <= max && !value.chars().any(|c| c < ' ')
}

fn value(record: &ObservationInput, key: &str) -> Value {
    record
        .get(key)
        .map(MetricValue::json)
        .unwrap_or(Value::Null)
}

pub fn observations(
    record: &Value,
    schema: &Value,
    protocol_id: Option<&str>,
    source: Option<&str>,
) -> Result<Vec<Value>, MetricError> {
    observations_typed(
        &ObservationInput::from_json(record)?,
        schema,
        protocol_id,
        source,
    )
}

pub fn observations_typed(
    record: &ObservationInput,
    schema: &Value,
    protocol_id: Option<&str>,
    source: Option<&str>,
) -> Result<Vec<Value>, MetricError> {
    let schema = schema.as_object().ok_or(MetricError::Schema)?;
    let empty = Map::new();
    let definitions = match schema.get("definitions") {
        None => &empty,
        Some(Value::Object(definitions)) => definitions,
        _ => return Err(MetricError::Schema),
    };
    let mut common = Map::new();
    for key in ["step", "epoch"] {
        if record.get(key).is_some_and(MetricValue::finite) {
            common.insert(key.to_owned(), value(record, key));
        }
    }
    for key in ["timestamp", "checkpoint_id", "dataset_id"] {
        if record.get(key).is_some_and(|value| {
            value
                .string()
                .is_some_and(|value| value.chars().count() <= 2000)
                || value.finite()
        }) {
            common.insert(key.to_owned(), value(record, key));
        }
    }
    let mut context = record
        .get("protocol_id")
        .map(MetricValue::json)
        .unwrap_or_else(|| {
            protocol_id
                .map(|v| Value::String(v.to_owned()))
                .unwrap_or(Value::Null)
        });
    let bad_context = record.get("protocol_id").is_some_and(|value| {
        !value.is_null()
            && value
                .string()
                .is_none_or(|value| value.chars().count() > 2000)
    });
    if bad_context {
        context = Value::Null;
    }
    let variant = value(record, "variant_id");
    let bad_variant = record.get("variant_id").is_some_and(|value| {
        !value.is_null() && !value.string().is_some_and(|value| printable(value, 2000))
    });
    let names = if let Some(name) = record.get("name") {
        let Some(name) = name.string().filter(|name| printable(name, 128)) else {
            return Ok(vec![
                json!({"name":null,"value":null,"unit":null,"status":"FAILED","error":"INVALID_METRIC_NAME","errors":["INVALID_METRIC_NAME"],"source":source}),
            ]);
        };
        vec![(name.to_owned(), record.get("value"), Some(record))]
    } else {
        record
            .fields
            .iter()
            .filter(|(key, value)| {
                ![
                    "step",
                    "epoch",
                    "timestamp",
                    "checkpoint_id",
                    "dataset_id",
                    "protocol_id",
                    "variant_id",
                ]
                .contains(&key.as_str())
                    && value.is_number()
            })
            .map(|(key, value)| (key.clone(), Some(value), None))
            .collect()
    };
    let mut results = Vec::with_capacity(names.len());
    for (name, raw, item) in names {
        let definition = match definitions.get(&name) {
            None => &empty,
            Some(Value::Object(definition)) => definition,
            _ => return Err(MetricError::Schema),
        };
        let mut unit = item
            .and_then(|item| item.get("unit"))
            .map(MetricValue::json)
            .unwrap_or_else(|| definition.get("unit").cloned().unwrap_or(Value::Null));
        let mut state = item
            .and_then(|item| item.get("status"))
            .map(MetricValue::json)
            .unwrap_or(json!("VALID"));
        let mut error = item
            .and_then(|item| item.get("error"))
            .map(|value| match value {
                MetricValue::Float(number) if !number.is_finite() => json!(value.python_string()),
                value => value.json(),
            })
            .unwrap_or(Value::Null);
        let mut errors = Vec::<Value>::new();
        let mut fail =
            |state_ref: &mut Value, error_ref: &mut Value, status: &str, message: &str| {
                *state_ref = json!(status);
                *error_ref = json!(message);
                errors.push(json!(message));
            };
        if !state
            .as_str()
            .is_some_and(|value| ["VALID", "PARTIAL", "FAILED", "MISSING"].contains(&value))
        {
            fail(&mut state, &mut error, "FAILED", "INVALID_STATUS");
        }
        let valid_number = raw.is_some_and(MetricValue::finite);
        if state == "VALID" && !valid_number {
            fail(
                &mut state,
                &mut error,
                "FAILED",
                "NONFINITE_OR_MISSING_VALUE",
            );
        }
        if !unit.as_str().is_some_and(|unit| printable(unit, 128)) {
            let status = if state == "FAILED" {
                "FAILED"
            } else {
                "UNDECLARED"
            };
            fail(&mut state, &mut error, status, "UNIT_NOT_DECLARED");
            unit = Value::Null;
        } else if definition
            .get("unit")
            .is_some_and(|expected| !expected.is_null() && expected != &unit)
        {
            fail(&mut state, &mut error, "FAILED", "UNIT_MISMATCH");
        } else if !definitions.is_empty() && !definitions.contains_key(&name) {
            let status = if state == "FAILED" {
                "FAILED"
            } else {
                "UNDECLARED"
            };
            fail(&mut state, &mut error, status, "METRIC_NOT_DECLARED");
        }
        if bad_context {
            fail(&mut state, &mut error, "FAILED", "INVALID_PROTOCOL_ID");
        }
        if bad_variant {
            fail(&mut state, &mut error, "FAILED", "INVALID_VARIANT_ID");
        }
        let summary = item.and_then(|item| item.get("summary"));
        if summary.is_some_and(|summary| !matches!(summary, MetricValue::Json(Value::Bool(_)))) {
            fail(&mut state, &mut error, "FAILED", "INVALID_SUMMARY_FLAG");
        }
        if protocol_id.is_some_and(|expected| context.as_str() != Some(expected)) {
            fail(&mut state, &mut error, "FAILED", "PROTOCOL_MISMATCH");
        }
        let mut sums = Map::new();
        for key in ["numerator", "denominator"] {
            if let Some(raw) = item
                .and_then(|item| item.get(key))
                .filter(|raw| !raw.is_null())
            {
                sums.insert(
                    key.to_owned(),
                    if raw.finite() {
                        raw.json()
                    } else {
                        Value::Null
                    },
                );
                if !raw.finite() {
                    fail(
                        &mut state,
                        &mut error,
                        "FAILED",
                        "NONFINITE_AGGREGATION_COUNT",
                    );
                }
            }
        }
        if definition.get("aggregation").and_then(Value::as_str) == Some("ratio_of_sums")
            && (sums.get("numerator").is_none_or(Value::is_null)
                || sums.get("denominator").is_none_or(Value::is_null)
                || sums
                    .get("denominator")
                    .and_then(Value::as_f64)
                    .is_some_and(|value| value <= 0.0))
        {
            fail(
                &mut state,
                &mut error,
                "FAILED",
                "INVALID_AGGREGATION_COUNTS",
            );
        }
        let invalid = raw
            .filter(|raw| !raw.is_null() && !valid_number)
            .map(|raw| json!(truncate(&raw.python_string(), 128)))
            .unwrap_or(Value::Null);
        let mut result = common.clone();
        result.extend([
            ("name".to_owned(), json!(name)),
            (
                "value".to_owned(),
                if valid_number {
                    raw.unwrap().json()
                } else {
                    Value::Null
                },
            ),
            ("unit".to_owned(), unit),
            ("invalid_value".to_owned(), invalid),
            (
                "variant_id".to_owned(),
                if bad_variant {
                    Value::Null
                } else {
                    variant.clone()
                },
            ),
            ("protocol_id".to_owned(), context.clone()),
            ("status".to_owned(), state),
            (
                "error".to_owned(),
                if error.is_null() {
                    Value::Null
                } else {
                    json!(truncate(&python_string(&error), 2000))
                },
            ),
            (
                "source".to_owned(),
                source
                    .filter(|source| !source.is_empty())
                    .map(|source| json!(source))
                    .unwrap_or_else(|| value(record, "_source_path")),
            ),
            ("errors".to_owned(), Value::Array(errors)),
        ]);
        result.extend(sums);
        if let Some(summary) = summary {
            result.insert(
                "summary".to_owned(),
                json!(matches!(summary, MetricValue::Json(Value::Bool(true)))),
            );
        }
        results.push(Value::Object(result));
    }
    Ok(results)
}
