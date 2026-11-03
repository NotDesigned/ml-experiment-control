//! Finite references in original YAML observations. Cycles, forward references,
//! duplicate anchor names, and excessive expansion are errors before cloning.
use super::{DocumentError, MAX_BYTES, MAX_DEPTH, MAX_NODES};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use yaml_rust2::scanner::{Scanner, Token, TokenType};

pub(super) fn unique_names(text: &str) -> Result<(), DocumentError> {
    let mut names = BTreeSet::new();
    for (count, Token(_, token)) in Scanner::new(text.chars()).enumerate() {
        if count >= MAX_NODES * 4 {
            return Err(DocumentError::Complexity);
        }
        if let TokenType::Anchor(name) = token {
            if !names.insert(name) {
                return Err(DocumentError::UnsupportedYaml);
            }
        }
    }
    Ok(())
}

struct Expansion {
    value: Value,
    bytes: usize,
    nodes: usize,
    depth: usize,
}

#[derive(Default)]
pub(super) struct References {
    values: BTreeMap<usize, Option<Expansion>>,
    cached_bytes: usize,
    expanded_bytes: usize,
    expanded_nodes: usize,
}
fn add(a: usize, b: usize, limit: usize) -> Result<usize, DocumentError> {
    a.checked_add(b)
        .filter(|n| *n <= limit)
        .ok_or(DocumentError::Complexity)
}
impl References {
    pub(super) fn events(&self, count: usize) -> Result<(), DocumentError> {
        add(count, self.expanded_nodes, MAX_NODES - 1).map(|_| ())
    }
    pub(super) fn reserve(&mut self, id: usize) -> Result<(), DocumentError> {
        if id != 0 && self.values.insert(id, None).is_some() {
            return Err(DocumentError::UnsupportedYaml);
        }
        Ok(())
    }
    pub(super) fn complete(&mut self, id: usize, value: &Value) -> Result<(), DocumentError> {
        if id == 0 {
            return Ok(());
        }
        if !matches!(self.values.get(&id), Some(None)) {
            return Err(DocumentError::UnsupportedYaml);
        }
        let bytes = serde_json::to_vec(value)
            .map_err(|_| DocumentError::Malformed)?
            .len();
        self.cached_bytes = add(self.cached_bytes, bytes, MAX_BYTES)?;
        let mut pending = vec![(value, 1)];
        let mut nodes = 0;
        let mut depth = 0;
        while let Some((value, level)) = pending.pop() {
            nodes = add(nodes, 1, MAX_NODES)?;
            depth = depth.max(level);
            if depth > MAX_DEPTH {
                return Err(DocumentError::Complexity);
            }
            match value {
                Value::Array(items) => pending.extend(items.iter().map(|v| (v, level + 1))),
                Value::Object(items) => {
                    nodes = add(nodes, items.len(), MAX_NODES)?;
                    pending.extend(items.values().map(|v| (v, level + 1)));
                }
                _ => {}
            }
        }
        self.values.insert(
            id,
            Some(Expansion {
                value: value.clone(),
                bytes,
                nodes,
                depth,
            }),
        );
        Ok(())
    }
    pub(super) fn expand(&mut self, id: usize, depth: usize) -> Result<Value, DocumentError> {
        let value = self
            .values
            .get(&id)
            .and_then(Option::as_ref)
            .ok_or(DocumentError::UnsupportedYaml)?;
        add(depth, value.depth, MAX_DEPTH)?;
        self.expanded_bytes = add(self.expanded_bytes, value.bytes, MAX_BYTES)?;
        self.expanded_nodes = add(self.expanded_nodes, value.nodes, MAX_NODES)?;
        Ok(value.value.clone())
    }
}
