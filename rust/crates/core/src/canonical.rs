//! Python-compatible frozen JSON bytes. Pure: no files, environment or runtime interpreter.
//! The raw entry also preserves lone UTF-16 surrogates accepted in legacy JSON.
mod parser;

use parser::Node;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonOptions {
    pub sort_keys: bool,
    pub ensure_ascii: bool,
    /// Python's default separators include spaces; frozen digests usually do not.
    pub compact: bool,
}
impl Default for JsonOptions {
    fn default() -> Self {
        Self {
            sort_keys: true,
            ensure_ascii: true,
            compact: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("canonical JSON: {0}")]
pub struct CanonicalError(pub(crate) &'static str);

pub fn python_json_bytes(value: &Value) -> Result<Vec<u8>, CanonicalError> {
    encode(value, JsonOptions::default())
}

pub fn encode(value: &Value, options: JsonOptions) -> Result<Vec<u8>, CanonicalError> {
    let raw = serde_json::to_vec(value).map_err(|_| CanonicalError("invalid JSON value"))?;
    python_json_from_slice(&raw, options)
}

pub fn python_json_from_slice(
    input: &[u8],
    options: JsonOptions,
) -> Result<Vec<u8>, CanonicalError> {
    let node = parser::parse(input)?;
    let mut output = String::new();
    encode_node(&node, options, &mut output)?;
    Ok(output.into_bytes())
}

pub fn digest(value: &Value) -> Result<String, CanonicalError> {
    Ok(format!("{:x}", Sha256::digest(python_json_bytes(value)?)))
}

pub fn protocol_identity(value: &Value) -> Result<String, CanonicalError> {
    Ok(format!("sha256:{}", digest(value)?))
}

fn encode_node(
    node: &Node,
    options: JsonOptions,
    output: &mut String,
) -> Result<(), CanonicalError> {
    match node {
        Node::Null => output.push_str("null"),
        Node::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Node::Number(raw) => output.push_str(&python_number(raw)?),
        Node::String(value) => encode_string(value, options.ensure_ascii, output)?,
        Node::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push_str(if options.compact { "," } else { ", " });
                }
                encode_node(value, options, output)?;
            }
            output.push(']');
        }
        Node::Object(values) => {
            let mut members: Vec<_> = values.iter().collect();
            if options.sort_keys {
                members.sort_by(|a, b| a.0.cmp(&b.0));
            }
            output.push('{');
            for (index, (key, value)) in members.into_iter().enumerate() {
                if index > 0 {
                    output.push_str(if options.compact { "," } else { ", " });
                }
                encode_string(key, options.ensure_ascii, output)?;
                output.push_str(if options.compact { ":" } else { ": " });
                encode_node(value, options, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn encode_string(value: &[u32], ascii: bool, output: &mut String) -> Result<(), CanonicalError> {
    use std::fmt::Write;
    output.push('"');
    for &code in value {
        match code {
            0x22 => output.push_str("\\\""),
            0x5c => output.push_str("\\\\"),
            8 => output.push_str("\\b"),
            12 => output.push_str("\\f"),
            10 => output.push_str("\\n"),
            13 => output.push_str("\\r"),
            9 => output.push_str("\\t"),
            c if c < 0x20 || ascii && c >= 0x7f => {
                if c <= 0xffff {
                    write!(output, "\\u{c:04x}").expect("String writes cannot fail");
                } else {
                    let c = c - 0x10000;
                    write!(
                        output,
                        "\\u{:04x}\\u{:04x}",
                        0xd800 + (c >> 10),
                        0xdc00 + (c & 0x3ff)
                    )
                    .expect("String writes cannot fail");
                }
            }
            c => output.push(
                char::from_u32(c)
                    .ok_or(CanonicalError("lone surrogate cannot be encoded as UTF-8"))?,
            ),
        }
    }
    output.push('"');
    Ok(())
}

fn python_number(raw: &str) -> Result<String, CanonicalError> {
    if !raw.contains(['.', 'e', 'E']) {
        if raw == "-0" {
            return Ok("0".into());
        }
        return Ok(raw.into());
    }
    let number = raw
        .parse::<f64>()
        .map_err(|_| CanonicalError("invalid floating point number"))?;
    if !number.is_finite() {
        return Err(CanonicalError(
            "nonfinite values have no frozen JSON identity",
        ));
    }
    python_float(number)
}

/// CPython repr's shortest digits, exponent threshold and signed zero spelling.
pub fn python_float(number: f64) -> Result<String, CanonicalError> {
    if !number.is_finite() {
        return Err(CanonicalError(
            "nonfinite values have no frozen JSON identity",
        ));
    }
    if number == 0.0 {
        return Ok(if number.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into());
    }
    let mut buffer = ryu::Buffer::new();
    let raw = buffer.format_finite(number);
    let (sign, raw) = raw.strip_prefix('-').map_or(("", raw), |s| ("-", s));
    let (mantissa, extra) = match raw.split_once('e') {
        Some((m, e)) => (
            m,
            e.parse::<i32>()
                .map_err(|_| CanonicalError("invalid exponent"))?,
        ),
        None => (raw, 0),
    };
    let point = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let all = mantissa.replace('.', "");
    let leading = all.bytes().take_while(|c| *c == b'0').count();
    let digits = all[leading..].trim_end_matches('0');
    let exponent = extra + point - leading as i32 - 1;
    if !(-4..16).contains(&exponent) {
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        return Ok(format!(
            "{sign}{}{fraction}e{}{exponent:02}",
            &digits[..1],
            if exponent < 0 { "-" } else { "+" },
            exponent = exponent.abs()
        ));
    }
    let decimal = exponent + 1;
    let body = if decimal <= 0 {
        format!("0.{}{}", "0".repeat((-decimal) as usize), digits)
    } else if decimal as usize >= digits.len() {
        format!(
            "{}{}.0",
            digits,
            "0".repeat(decimal as usize - digits.len())
        )
    } else {
        format!(
            "{}.{}",
            &digits[..decimal as usize],
            &digits[decimal as usize..]
        )
    };
    Ok(format!("{sign}{body}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_bytes_keep_non_ascii_float_kinds_and_big_integers() {
        let value = serde_json::from_str(
            r#"{"z":1e-7,"a":-0.0,"unicode":"模型😀","big":123456789012345678901234567890}"#,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(python_json_bytes(&value).unwrap()).unwrap(),
            r#"{"a":-0.0,"big":123456789012345678901234567890,"unicode":"\u6a21\u578b\ud83d\ude00","z":1e-07}"#
        );
        assert_eq!(python_float(1e16).unwrap(), "1e+16");
        assert_eq!(python_float(1e15).unwrap(), "1000000000000000.0");
        assert_eq!(python_float(1e-4).unwrap(), "0.0001");
        assert_eq!(python_float(1e-5).unwrap(), "1e-05");
    }
    #[test]
    fn legacy_surrogates_duplicates_and_python_key_order_are_preserved() {
        let raw = r#"{"\ud800":1,"😀":2,"\ue000":3,"a":0,"a":4}"#.as_bytes();
        let encoded = python_json_from_slice(raw, JsonOptions::default()).unwrap();
        assert_eq!(
            String::from_utf8(encoded).unwrap(),
            r#"{"a":4,"\ud800":1,"\ue000":3,"\ud83d\ude00":2}"#
        );
        assert!(
            python_json_from_slice(
                raw,
                JsonOptions {
                    ensure_ascii: false,
                    ..Default::default()
                }
            )
            .is_err()
        );
        for raw in [
            b"01".as_slice(),
            b"1e309",
            b"[1,]",
            b"{\"a\":1,}",
            b"{\"a\":NaN}",
        ] {
            assert!(python_json_from_slice(raw, JsonOptions::default()).is_err());
        }
    }
}
