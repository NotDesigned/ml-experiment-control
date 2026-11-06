use ml_exp_core::{identity, observation::numeric_tolerant_equal};
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestSource {
    ProjectJson,
    RunJson,
    AttemptJson,
}
impl ManifestSource {
    pub(super) fn filename(self) -> &'static str {
        match self {
            Self::ProjectJson => "project.json",
            Self::RunJson => "manifest.json",
            Self::AttemptJson => "attempt.json",
        }
    }
}

// Manifests can contain user environment values. Do not Debug-format them.
#[derive(Clone, Serialize)]
pub struct Document {
    pub source: ManifestSource,
    pub raw_sha256: String,
    pub metadata: Map<String, Value>,
}
impl Document {
    pub(super) fn decode(bytes: &[u8], source: ManifestSource) -> Result<Self, DocumentError> {
        let metadata = ml_exp_core::json::mapping(bytes).map_err(|error| match error {
            ml_exp_core::json::Error::Malformed => DocumentError::Malformed,
            ml_exp_core::json::Error::Complexity => DocumentError::Complexity,
            ml_exp_core::json::Error::MappingKey => DocumentError::MappingKey,
            ml_exp_core::json::Error::Number => DocumentError::Number,
        })?;
        Ok(Self {
            source,
            raw_sha256: format!("sha256:{:x}", Sha256::digest(bytes)),
            metadata,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DocumentError {
    #[error("metadata is malformed or not one mapping document")]
    Malformed,
    #[error("metadata exceeds the nesting or node limit")]
    Complexity,
    #[error("metadata contains duplicate or non-string mapping keys")]
    MappingKey,
    #[error("metadata contains an unsupported non-finite numeric value")]
    Number,
    #[error("metadata identity is missing or conflicts with the requested object")]
    Identity,
    #[error("metadata schema version is unsupported")]
    Version,
    #[error("Attempt changes immutable Run provenance")]
    Provenance,
}

fn field<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    map.get(key).filter(|v| !v.is_null())
}
fn exact(map: &Map<String, Value>, key: &str, expected: &str) -> Result<(), DocumentError> {
    if field(map, key).and_then(Value::as_str) == Some(expected) {
        Ok(())
    } else {
        Err(DocumentError::Identity)
    }
}
fn version(map: &Map<String, Value>) -> Result<(), DocumentError> {
    if map
        .get("schema_version")
        .is_some_and(|v| v.as_u64() != Some(1))
    {
        return Err(DocumentError::Version);
    }
    Ok(())
}
pub fn validate_project(map: &Map<String, Value>, project: &str) -> Result<(), DocumentError> {
    identity::validate_id(project).map_err(why!(DocumentError::Identity))?;
    version(map)?;
    exact(map, "project", project)
}
pub fn validate_run(
    map: &Map<String, Value>,
    project: &str,
    run: &str,
) -> Result<(), DocumentError> {
    identity::validate_observation_id(run).map_err(why!(DocumentError::Identity))?;
    validate_project(map, project)?;
    exact(map, "run_id", run)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Provenance {
    /// Only completeness of these named comparisons, not all scientific fields.
    pub compared_fields_complete: bool,
    pub compared_fields: Vec<String>,
    pub missing_fields: Vec<String>,
}
const COMPARED_FIELDS: [&str; 5] = ["source_id", "image_id", "config_path", "seed", "campaign"];
pub fn validate_attempt(
    run: &Document,
    attempt: &Document,
    project: &str,
    run_id: &str,
    attempt_id: &str,
) -> Result<Provenance, DocumentError> {
    identity::validate_id(attempt_id).map_err(why!(DocumentError::Identity))?;
    validate_run(&run.metadata, project, run_id)?;
    version(&attempt.metadata)?;
    exact(&attempt.metadata, "project", project)?;
    exact(&attempt.metadata, "run_id", run_id)?;
    exact(&attempt.metadata, "attempt_id", attempt_id)?;
    let mut missing_fields = Vec::new();
    for key in COMPARED_FIELDS {
        match (field(&run.metadata, key), field(&attempt.metadata, key)) {
            (Some(left), Some(right)) if !numeric_tolerant_equal(left, right) => {
                return Err(DocumentError::Provenance);
            }
            (None, _) | (_, None) => missing_fields.push(key.into()),
            _ => {}
        }
    }
    Ok(Provenance {
        compared_fields_complete: missing_fields.is_empty(),
        compared_fields: COMPARED_FIELDS.into_iter().map(str::to_owned).collect(),
        missing_fields,
    })
}
