//! One deterministic JSON encoding for digests, identities and delivered payloads:
//! sorted keys, compact separators, raw UTF-8 and exact number text. Pure.
use serde_json::{Number, Value};
use sha2::{Digest, Sha256};

const NONFINITE: CanonicalError = CanonicalError("nonfinite number has no canonical form");

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("canonical JSON: {0}")]
pub struct CanonicalError(&'static str);

pub fn bytes(value: &Value) -> Result<Vec<u8>, CanonicalError> {
    let mut output = Vec::new();
    write_value(value, &mut output)?;
    Ok(output)
}

pub fn text(value: &Value) -> Result<String, CanonicalError> {
    String::from_utf8(bytes(value)?).map_err(|_| CanonicalError("output is not UTF-8"))
}

pub fn digest(value: &Value) -> Result<String, CanonicalError> {
    Ok(format!("{:x}", Sha256::digest(bytes(value)?)))
}

pub fn protocol_identity(value: &Value) -> Result<String, CanonicalError> {
    Ok(format!("sha256:{}", digest(value)?))
}

/// One finite float's shortest round-trip text. Both zeros compare equal, so
/// they share the spelling `0.0`.
pub fn float_text(value: f64) -> Result<String, CanonicalError> {
    if !value.is_finite() {
        return Err(NONFINITE);
    }
    let value = if value == 0.0 { 0.0 } else { value };
    Ok(ryu::Buffer::new().format_finite(value).to_owned())
}

/// An integer keeps its exact decimal text at any length. A float stays a float,
/// because canonical bytes are also delivered: lr=1.0 must not reach a script as 1.
fn number_text(number: &Number) -> Result<String, CanonicalError> {
    let raw = number.to_string();
    if raw.contains(['.', 'e', 'E']) {
        float_text(number.as_f64().ok_or(NONFINITE)?)
    } else if raw == "-0" {
        Ok("0".into())
    } else {
        Ok(raw)
    }
}

fn write_string(value: &str, output: &mut Vec<u8>) -> Result<(), CanonicalError> {
    serde_json::to_writer(output, value).map_err(|_| CanonicalError("string cannot be written"))
}

fn write_value(value: &Value, output: &mut Vec<u8>) -> Result<(), CanonicalError> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) => output.extend_from_slice(number_text(number)?.as_bytes()),
        Value::String(value) => write_string(value, output)?,
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_value(value, output)?;
            }
            output.push(b']');
        }
        Value::Object(fields) => {
            let mut members: Vec<_> = fields.iter().collect();
            members.sort_unstable_by(|left, right| left.0.cmp(right.0));
            output.push(b'{');
            for (index, (key, value)) in members.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_string(key, output)?;
                output.push(b':');
                write_value(value, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}
