//! One JSONL line: UTF-8 JSON, plus the bare NaN, Infinity and -Infinity a
//! diverging Python `json.dumps` script writes. Those stay typed validation
//! failures, never null or a successful numeric zero. A byte order mark or any
//! other encoding does not parse, so the line becomes a diagnostic.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Number, Value, json};

use crate::metrics::{ObservationInput, float_token, json_text};

/// A parsed line with each nonfinite number replaced by its text token.
pub(super) struct MetricLine {
    value: Value,
    /// Top-level fields holding a nonfinite number, directly or in a container.
    nonfinite: Vec<String>,
}
impl MetricLine {
    pub(super) fn has_field(&self, key: &str) -> bool {
        self.value.get(key).is_some()
    }
    pub(super) fn observation(&self) -> Option<ObservationInput> {
        let mut input = ObservationInput::from_json(&self.value).ok()?;
        for name in &self.nonfinite {
            input = match &self.value[name] {
                // A string here is a token, which parses back to its number.
                Value::String(token) => input.with_number(name, token.parse::<f64>().ok()?),
                container => input.with_invalid_text(name, json_text(container)),
            };
        }
        Some(input)
    }
}

/// Normalize number spelling and turn each marker into its token. `found`
/// reports whether any nonfinite number was seen.
fn normalized(value: Value, markers: &BTreeMap<String, f64>, found: &mut bool) -> Option<Value> {
    Some(match value {
        Value::String(ref text) if markers.contains_key(text) => {
            *found = true;
            json!(float_token(markers[text]))
        }
        Value::Number(number) => {
            let text = number.to_string();
            if text.contains(['.', 'e', 'E']) {
                let number = text.parse::<f64>().ok()?;
                Number::from_f64(number)
                    .map(Value::Number)
                    .unwrap_or_else(|| {
                        *found = true;
                        json!(float_token(number))
                    })
            } else if text == "-0" {
                json!(0)
            } else {
                Value::Number(number)
            }
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| normalized(value, markers, found))
                .collect::<Option<_>>()?,
        ),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| Some((key, normalized(value, markers, found)?)))
                .collect::<Option<_>>()?,
        ),
        value => value,
    })
}

pub(super) fn parse(bytes: &[u8]) -> Option<MetricLine> {
    let text = std::str::from_utf8(bytes).ok()?;
    // Record every actual string (including escaped strings) before choosing
    // markers. User-controlled text can never be confused with a number token.
    let mut strings = BTreeSet::new();
    let mut spans = Vec::new();
    let mut offset = 0;
    let raw = text.as_bytes();
    while offset < raw.len() {
        if raw[offset] != b'"' {
            offset += 1;
            continue;
        }
        let start = offset;
        offset += 1;
        while offset < raw.len() {
            match raw[offset] {
                b'\\' => offset += 2,
                b'"' => {
                    offset += 1;
                    break;
                }
                _ => offset += 1,
            }
        }
        let literal = text.get(start..offset)?;
        strings.insert(serde_json::from_str::<String>(literal).ok()?);
        spans.push((start, offset));
    }
    let mut nonce = 0;
    let prefix = loop {
        let prefix = format!("\0ml-exp-nonfinite:{nonce}:");
        if !strings.iter().any(|value| value.starts_with(&prefix)) {
            break prefix;
        }
        nonce += 1;
    };
    let mut result = String::new();
    let mut markers = BTreeMap::new();
    offset = 0;
    let mut span = 0;
    while offset < raw.len() {
        if spans.get(span).is_some_and(|(start, _)| *start == offset) {
            let end = spans[span].1;
            result.push_str(&text[offset..end]);
            offset = end;
            span += 1;
            continue;
        }
        let token = [
            ("-Infinity", f64::NEG_INFINITY),
            ("Infinity", f64::INFINITY),
            ("NaN", f64::NAN),
        ]
        .into_iter()
        .find(|(token, _)| text[offset..].starts_with(token));
        if let Some((token, number)) = token {
            // Quoting a value token must never turn an invalid object key into
            // valid JSON. A bare number/name before ':' is a key, including in
            // nested objects; quoted user keys were copied above.
            if text[offset + token.len()..]
                .trim_start_matches([' ', '\t', '\r', '\n'])
                .starts_with(':')
            {
                return None;
            }
            let key = format!("{prefix}{}", markers.len());
            result.push_str(&serde_json::to_string(&key).ok()?);
            markers.insert(key, number);
            offset += token.len();
        } else {
            let ch = text[offset..].chars().next()?;
            result.push(ch);
            offset += ch.len_utf8();
        }
    }
    let mut nonfinite = Vec::new();
    let value = match serde_json::from_str::<Value>(&result).ok()? {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| {
                    let mut found = false;
                    let value = normalized(value, &markers, &mut found)?;
                    if found {
                        nonfinite.push(key.clone());
                    }
                    Some((key, value))
                })
                .collect::<Option<Map<_, _>>>()?,
        ),
        value => normalized(value, &markers, &mut false)?,
    };
    Some(MetricLine { value, nonfinite })
}
