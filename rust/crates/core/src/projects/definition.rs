//! The authored Project file. This validates its shape and authorizes no I/O.
use super::{Error, Result};
use crate::identity;
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectDefinition {
    pub schema_version: u32,
    pub project: String,
    pub title: String,
    pub run_roots: Vec<String>,
}
impl ProjectDefinition {
    pub fn from_value(value: &Value) -> Result<Self> {
        let definition: Self =
            serde_json::from_value(value.clone()).map_err(|_| Error::Definition)?;
        if definition.schema_version != 1 {
            return Err(Error::Definition);
        }
        identity::validate_id(&definition.project).map_err(|_| Error::Definition)?;
        Ok(definition)
    }
    pub fn project(&self) -> &str {
        &self.project
    }
}
