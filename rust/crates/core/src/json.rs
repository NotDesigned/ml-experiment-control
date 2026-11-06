//! Strict JSON with duplicate-key, depth and node refusal.
pub const MAX_DEPTH: usize = 64;
/// An edge bounds bytes, and parsing multiplies them: 8 MiB of `[0,0,...]` is four
/// million values, reached before any token check on the exchange routes. This
/// stays above the largest valid message, about 530,000 nodes in a READY dataset
/// State at `datasets::MAX_METADATA`, so a size check decides that case first.
const MAX_NODES: usize = 600_000;
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("JSON is malformed or not one mapping")]
    Malformed,
    #[error("JSON exceeds its depth or node bound")]
    Complexity,
    #[error("JSON contains duplicate or invalid mapping keys")]
    MappingKey,
    #[error("JSON contains an unsupported non-finite number")]
    Number,
}
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, value::RawValue};
use std::fmt;

/// One mapping from anywhere outside this service's own state.
pub fn mapping(bytes: &[u8]) -> Result<Map<String, Value>, Error> {
    parse(bytes, MAX_NODES)
}
/// One mapping from a file this service wrote and reads with `read_whole`. It has
/// no node budget because a refusal here would strand a record already committed.
pub fn own_mapping(bytes: &[u8]) -> Result<Map<String, Value>, Error> {
    parse(bytes, usize::MAX)
}
fn parse(bytes: &[u8], mut remaining: usize) -> Result<Map<String, Value>, Error> {
    let text = std::str::from_utf8(bytes).map_err(|_| Error::Malformed)?;
    match value(text, 0, &mut remaining)? {
        Value::Object(map) => Ok(map),
        _ => Err(Error::Malformed),
    }
}
fn consume(remaining: &mut usize) -> Result<(), Error> {
    *remaining = remaining.checked_sub(1).ok_or(Error::Complexity)?;
    Ok(())
}
fn value(text: &str, depth: usize, remaining: &mut usize) -> Result<Value, Error> {
    consume(remaining)?;
    if depth > MAX_DEPTH {
        return Err(Error::Complexity);
    }
    let first = text.trim_start().as_bytes().first();
    if matches!(first, Some(b'{') | Some(b'[')) {
        let mut deserializer = serde_json::Deserializer::from_str(text);
        let mut failure = None;
        let visitor = Container {
            depth,
            remaining,
            failure: &mut failure,
        };
        let result = if first == Some(&b'{') {
            deserializer.deserialize_map(visitor)
        } else {
            deserializer.deserialize_seq(visitor)
        };
        let result = result.map_err(|_| failure.unwrap_or(Error::Malformed))?;
        deserializer.end().map_err(|_| Error::Malformed)?;
        return Ok(result);
    }
    let result: Value = serde_json::from_str(text).map_err(|_| Error::Malformed)?;
    if let Value::Number(number) = &result {
        if text.contains(['.', 'e', 'E']) && !number.as_f64().is_some_and(f64::is_finite) {
            return Err(Error::Number);
        }
    }
    Ok(result)
}
struct Container<'a> {
    depth: usize,
    remaining: &'a mut usize,
    failure: &'a mut Option<Error>,
}
impl<'de> Visitor<'de> for Container<'_> {
    type Value = Value;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("bounded JSON metadata")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Value, A::Error> {
        let mut result = Vec::new();
        while let Some(raw) = sequence.next_element::<&RawValue>()? {
            let item = value(raw.get(), self.depth + 1, self.remaining).map_err(|error| {
                *self.failure = Some(error);
                de::Error::custom("metadata policy refused value")
            })?;
            result.push(item);
        }
        Ok(Value::Array(result))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut mapping: A) -> Result<Value, A::Error> {
        let mut result = Map::new();
        while let Some(key) = mapping.next_key::<String>()? {
            consume(self.remaining).map_err(|error| {
                *self.failure = Some(error);
                de::Error::custom("metadata node limit")
            })?;
            if result.contains_key(&key) {
                *self.failure = Some(Error::MappingKey);
                return Err(de::Error::custom("duplicate metadata key"));
            }
            let raw = mapping.next_value::<&RawValue>()?;
            let item = value(raw.get(), self.depth + 1, self.remaining).map_err(|error| {
                *self.failure = Some(error);
                de::Error::custom("metadata policy refused value")
            })?;
            result.insert(key, item);
        }
        Ok(Value::Object(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The root, its key and the array are three nodes.
    fn wide(values: usize) -> Vec<u8> {
        let mut body = b"{\"a\":[0".to_vec();
        for _ in 1..values {
            body.extend_from_slice(b",0");
        }
        body.extend_from_slice(b"]}");
        body
    }
    #[test]
    fn the_node_budget_bounds_width_that_the_depth_check_does_not() {
        assert!(mapping(&wide(MAX_NODES - 3)).is_ok());
        assert_eq!(mapping(&wide(MAX_NODES - 2)), Err(Error::Complexity));
        // The 8 MiB an edge admits, as small numbers.
        assert_eq!(mapping(&wide(4 * 1024 * 1024)), Err(Error::Complexity));
    }
    #[test]
    fn the_budget_counts_mapping_keys_and_values() {
        let mut body = b"{\"a\":0".to_vec();
        for key in 1..MAX_NODES / 2 {
            body.extend_from_slice(format!(",\"{key}\":0").as_bytes());
        }
        body.push(b'}');
        assert_eq!(mapping(&body), Err(Error::Complexity));
        assert_eq!(own_mapping(&body).unwrap().len(), MAX_NODES / 2);
    }
    #[test]
    fn an_own_record_has_no_budget_and_keeps_every_other_refusal() {
        let map = own_mapping(&wide(MAX_NODES)).unwrap();
        assert_eq!(map["a"].as_array().unwrap().len(), MAX_NODES);
        assert_eq!(own_mapping(b"[]"), Err(Error::Malformed));
        assert_eq!(own_mapping(br#"{"a":1,"a":2}"#), Err(Error::MappingKey));
        assert_eq!(own_mapping(br#"{"a":1e999}"#), Err(Error::Number));
        let deep = format!(
            "{{\"a\":{}{}}}",
            "[".repeat(MAX_DEPTH + 1),
            "]".repeat(MAX_DEPTH + 1)
        );
        assert_eq!(own_mapping(deep.as_bytes()), Err(Error::Complexity));
    }
}
