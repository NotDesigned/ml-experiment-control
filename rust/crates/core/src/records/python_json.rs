//! Python's JSONL reader accepts encoding detection and nonfinite numbers.
//! Preserve those values as typed validation failures, rather than converting
//! them to null or replacing an invalid result with a successful numeric zero.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Number, Value};

use crate::metrics::{ObservationInput, python_repr};

#[derive(Clone, Debug)]
pub(super) enum PythonValue {
    Json(Value),
    Float(f64),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}
impl PythonValue {
    pub(super) fn json(&self) -> Value {
        match self {
            Self::Json(value) => value.clone(),
            Self::Float(_) => Value::Null,
            Self::Array(values) => Value::Array(values.iter().map(Self::json).collect()),
            Self::Object(fields) => Value::Object(
                fields
                    .iter()
                    .map(|(key, value)| (key.clone(), value.json()))
                    .collect(),
            ),
        }
    }
    fn repr(&self) -> String {
        match self {
            Self::Json(value) => python_repr(value),
            Self::Float(value) if value.is_nan() => "nan".into(),
            Self::Float(value) if value.is_sign_negative() => "-inf".into(),
            Self::Float(_) => "inf".into(),
            Self::Array(values) => format!(
                "[{}]",
                values.iter().map(Self::repr).collect::<Vec<_>>().join(", ")
            ),
            Self::Object(values) => format!(
                "{{{}}}",
                values
                    .iter()
                    .map(|(key, value)| format!(
                        "{}: {}",
                        python_repr(&Value::String(key.clone())),
                        value.repr()
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
    pub(super) fn observation(&self) -> Option<ObservationInput> {
        match self {
            Self::Json(value) => ObservationInput::from_json(value).ok(),
            Self::Object(fields) => {
                let mut result = ObservationInput::from_json(&self.json()).ok()?;
                for (name, value) in fields {
                    result = match value {
                        Self::Json(_) => result,
                        Self::Float(number) => result.with_number(name, *number),
                        Self::Array(_) | Self::Object(_) => {
                            result.with_invalid_repr(name, value.repr())
                        }
                    };
                }
                Some(result)
            }
            _ => None,
        }
    }
    pub(super) fn get(&self, key: &str) -> Option<Self> {
        match self {
            Self::Json(Value::Object(value)) => value.get(key).cloned().map(Self::Json),
            Self::Object(fields) => fields
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.clone()),
            _ => None,
        }
    }
    pub(super) fn array(&self) -> Option<Vec<Self>> {
        match self {
            Self::Json(Value::Array(values)) => {
                Some(values.iter().cloned().map(Self::Json).collect())
            }
            Self::Array(values) => Some(values.clone()),
            _ => None,
        }
    }
    pub(super) fn has_nonfinite(&self) -> bool {
        !matches!(self, Self::Json(_))
    }
}

fn decode(bytes: &[u8]) -> Option<String> {
    enum Encoding {
        Utf8(usize),
        Utf16(bool, usize),
        Utf32(bool, usize),
    }
    let encoding = if bytes.starts_with(&[0, 0, 0xfe, 0xff]) {
        Encoding::Utf32(false, 4)
    } else if bytes.starts_with(&[0xff, 0xfe, 0, 0]) {
        Encoding::Utf32(true, 4)
    } else if bytes.starts_with(&[0xfe, 0xff]) {
        Encoding::Utf16(false, 2)
    } else if bytes.starts_with(&[0xff, 0xfe]) {
        Encoding::Utf16(true, 2)
    } else if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        Encoding::Utf8(3)
    } else if bytes.len() >= 4 {
        if bytes[0] == 0 {
            if bytes[1] == 0 {
                Encoding::Utf32(false, 0)
            } else {
                Encoding::Utf16(false, 0)
            }
        } else if bytes[1] == 0 {
            if bytes[2] != 0 || bytes[3] != 0 {
                Encoding::Utf16(true, 0)
            } else {
                Encoding::Utf32(true, 0)
            }
        } else {
            Encoding::Utf8(0)
        }
    } else if bytes.len() == 2 && bytes[0] == 0 {
        Encoding::Utf16(false, 0)
    } else if bytes.len() == 2 && bytes[1] == 0 {
        Encoding::Utf16(true, 0)
    } else {
        Encoding::Utf8(0)
    };
    match encoding {
        Encoding::Utf8(skip) => String::from_utf8(bytes[skip..].to_vec()).ok(),
        Encoding::Utf16(little, skip) => {
            let bytes = &bytes[skip..];
            if bytes.len() % 2 != 0 {
                return None;
            }
            let words = bytes.chunks_exact(2).map(|v| {
                if little {
                    u16::from_le_bytes([v[0], v[1]])
                } else {
                    u16::from_be_bytes([v[0], v[1]])
                }
            });
            std::char::decode_utf16(words)
                .collect::<std::result::Result<String, _>>()
                .ok()
        }
        Encoding::Utf32(little, skip) => {
            let bytes = &bytes[skip..];
            if bytes.len() % 4 != 0 {
                return None;
            }
            bytes
                .chunks_exact(4)
                .map(|v| {
                    char::from_u32(if little {
                        u32::from_le_bytes([v[0], v[1], v[2], v[3]])
                    } else {
                        u32::from_be_bytes([v[0], v[1], v[2], v[3]])
                    })
                })
                .collect()
        }
    }
}

fn convert(value: Value, replacements: &BTreeMap<String, f64>) -> Option<PythonValue> {
    Some(match value {
        Value::String(ref value) if replacements.contains_key(value) => {
            PythonValue::Float(replacements[value])
        }
        Value::Number(number) => {
            let text = number.to_string();
            if text.contains(['.', 'e', 'E']) {
                let number = text.parse::<f64>().ok()?;
                if number.is_finite() {
                    PythonValue::Json(Value::Number(Number::from_f64(number)?))
                } else {
                    PythonValue::Float(number)
                }
            } else if text == "-0" {
                PythonValue::Json(Value::Number(Number::from(0)))
            } else {
                PythonValue::Json(Value::Number(number))
            }
        }
        Value::Array(values) => {
            let values = values
                .into_iter()
                .map(|value| convert(value, replacements))
                .collect::<Option<Vec<_>>>()?;
            if values.iter().all(|value| !value.has_nonfinite()) {
                PythonValue::Json(Value::Array(
                    values.into_iter().map(|value| value.json()).collect(),
                ))
            } else {
                PythonValue::Array(values)
            }
        }
        Value::Object(values) => {
            let values = values
                .into_iter()
                .map(|(key, value)| Some((key, convert(value, replacements)?)))
                .collect::<Option<Vec<_>>>()?;
            if values.iter().all(|(_, value)| !value.has_nonfinite()) {
                PythonValue::Json(Value::Object(
                    values
                        .into_iter()
                        .map(|(key, value)| (key, value.json()))
                        .collect::<Map<_, _>>(),
                ))
            } else {
                PythonValue::Object(values)
            }
        }
        value => PythonValue::Json(value),
    })
}

pub(super) fn parse(bytes: &[u8]) -> Option<PythonValue> {
    let text = decode(bytes)?;
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
    let mut replacements = BTreeMap::new();
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
            // Quoting a value token must never turn a Python-invalid object
            // key into valid JSON. A bare number/name before ':' is a key,
            // including nested objects; quoted user keys were copied above.
            if text[offset + token.len()..]
                .trim_start_matches([' ', '\t', '\r', '\n'])
                .starts_with(':')
            {
                return None;
            }
            let key = format!("{prefix}{}", replacements.len());
            result.push_str(&serde_json::to_string(&key).ok()?);
            replacements.insert(key, number);
            offset += token.len();
        } else {
            let ch = text[offset..].chars().next()?;
            result.push(ch);
            offset += ch.len_utf8();
        }
    }
    convert(serde_json::from_str(&result).ok()?, &replacements)
}
