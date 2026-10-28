//! Strict JSON remains JSON: YAML style and scalar resolution are irrelevant.
const MAX_DEPTH: usize = 64;
const MAX_NODES: usize = 100_000;
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

pub fn mapping(bytes: &[u8]) -> Result<Map<String, Value>, Error> {
    if bytes.len() > 4 * 1024 * 1024 {
        return Err(Error::Complexity);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| Error::Malformed)?;
    let mut remaining = MAX_NODES;
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
