use ml_exp_core::{identity, observation::python_equal};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::{str::FromStr, sync::OnceLock};
use yaml_rust2::{
    parser::{Event, Parser},
    scanner::TScalarStyle,
};

mod aliases;

pub const MAX_BYTES: usize = 4 * 1024 * 1024;
const MAX_DEPTH: usize = 64;
const MAX_NODES: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Json,
    Yaml,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestSource {
    ProjectYaml,
    RunYaml,
    RunJson,
    LegacyRunYaml,
    AttemptYaml,
    AttemptJson,
    LegacyAttemptYaml,
}
impl ManifestSource {
    pub(super) fn filename(self) -> &'static str {
        match self {
            Self::ProjectYaml => "research_project.yaml",
            Self::RunYaml => "manifest.yaml",
            Self::RunJson => "manifest.json",
            Self::LegacyRunYaml => "control_manifest.yaml",
            Self::AttemptYaml => "attempt.yaml",
            Self::AttemptJson => "attempt.json",
            Self::LegacyAttemptYaml => "control_attempt.yaml",
        }
    }
    pub(super) fn format(self) -> Format {
        if matches!(self, Self::RunJson | Self::AttemptJson) {
            Format::Json
        } else {
            Format::Yaml
        }
    }
}

// Manifests can contain user environment values. Do not Debug-format them.
#[derive(Clone, Serialize)]
pub struct Document {
    pub source: ManifestSource,
    pub raw_sha256: String,
    pub metadata: Map<String, Value>,
    pub legacy_observation_only: bool,
}
impl Document {
    pub(super) fn decode(bytes: &[u8], source: ManifestSource) -> Result<Self, DocumentError> {
        let metadata = match source.format() {
            Format::Yaml => parse_yaml(bytes, true)?,
            Format::Json => parse_mapping(bytes, Format::Json)?,
        };
        let legacy_observation_only = matches!(
            source,
            ManifestSource::LegacyRunYaml | ManifestSource::LegacyAttemptYaml
        ) || metadata.get("identity_version").and_then(Value::as_u64)
            != Some(2);
        Ok(Self {
            source,
            raw_sha256: format!("sha256:{:x}", Sha256::digest(bytes)),
            metadata,
            legacy_observation_only,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DocumentError {
    #[error("metadata exceeds the bounded document limit")]
    TooLarge,
    #[error("metadata is malformed or not one mapping document")]
    Malformed,
    #[error("metadata uses unsupported YAML references, tags or ambiguous plain scalars")]
    UnsupportedYaml,
    #[error("metadata exceeds the nesting or node limit")]
    Complexity,
    #[error("metadata contains duplicate or non-string mapping keys")]
    MappingKey,
    #[error("metadata contains an unsupported non-finite numeric value")]
    Number,
    #[error("metadata identity is missing or conflicts with the requested object")]
    Identity,
    #[error("metadata schema or identity version is unsupported")]
    Version,
    #[error("Attempt changes immutable Run provenance")]
    Provenance,
}

enum Frame {
    Sequence(Vec<Value>),
    Mapping(Map<String, Value>, Option<String>),
}
fn append(
    stack: &mut [Frame],
    root: &mut Option<Value>,
    value: Value,
) -> Result<(), DocumentError> {
    match stack.last_mut() {
        Some(Frame::Sequence(items)) => items.push(value),
        Some(Frame::Mapping(items, pending)) => {
            if let Some(key) = pending.take() {
                items.insert(key, value);
            } else {
                let Value::String(key) = value else {
                    return Err(DocumentError::MappingKey);
                };
                if items.contains_key(&key) {
                    return Err(DocumentError::MappingKey);
                }
                *pending = Some(key);
            }
        }
        None if root.is_none() => *root = Some(value),
        None => return Err(DocumentError::Malformed),
    }
    Ok(())
}

// PyYAML's non-JSON scalar forms are rejected, not silently interpreted using
// YAML 1.2 semantics. Quoted strings retain their exact string meaning. These
// are scalar-policy recognizers; YAML syntax is parsed by yaml-rust2.
fn legacy_scalar(value: &str) -> bool {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN
        .get_or_init(|| {
            Regex::new(concat!(
                r"(?x)^(?:",
                r"[-+]?0[bB][01_]+|[-+]?0[0-7_]+|[-+]?0[xX][0-9a-fA-F_]+|",
                r"[-+]?[0-9][0-9_]*|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+|",
                r"[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+][0-9]+)?|",
                r"\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|",
                r"[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|",
                r"[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)|",
                r"[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt\x20\t].*)?|",
                r"yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF|=|<<",
                r")$"
            ))
            .expect("fixed scalar policy regex")
        })
        .is_match(value)
}
fn scalar(value: String, style: TScalarStyle) -> Result<Value, DocumentError> {
    if style != TScalarStyle::Plain {
        return Ok(Value::String(value));
    }
    match value.as_str() {
        "" | "~" | "null" | "Null" | "NULL" => return Ok(Value::Null),
        "true" | "True" | "TRUE" => return Ok(Value::Bool(true)),
        "false" | "False" | "FALSE" => return Ok(Value::Bool(false)),
        _ => {}
    }
    if let Ok(number) = Number::from_str(&value) {
        let integer = !value.contains(['.', 'e', 'E']);
        let legacy_float = value.contains('.') && legacy_scalar(&value);
        if integer {
            return Ok(Value::Number(number));
        }
        if legacy_float {
            return value
                .parse::<f64>()
                .ok()
                .and_then(Number::from_f64)
                .map(Value::Number)
                .ok_or(DocumentError::UnsupportedYaml);
        }
        return Err(DocumentError::UnsupportedYaml);
    }
    if legacy_scalar(&value) {
        return Err(DocumentError::UnsupportedYaml);
    }
    Ok(Value::String(value))
}

/// Accepted YAML is a bounded JSON-compatible subset with quoted ambiguous
/// scalars. No alias expansion, tags, merge keys, duplicate keys or streams.
/// This is not a complete PyYAML 1.1 deserializer or scientific hash codec.
pub fn parse_mapping(bytes: &[u8], format: Format) -> Result<Map<String, Value>, DocumentError> {
    if bytes.len() > MAX_BYTES {
        return Err(DocumentError::TooLarge);
    }
    if format == Format::Json {
        return ml_exp_core::json::mapping(bytes).map_err(|error| match error {
            ml_exp_core::json::Error::Malformed => DocumentError::Malformed,
            ml_exp_core::json::Error::Complexity => DocumentError::Complexity,
            ml_exp_core::json::Error::MappingKey => DocumentError::MappingKey,
            ml_exp_core::json::Error::Number => DocumentError::Number,
        });
    }
    parse_yaml(bytes, false)
}

// Authored inputs keep the strict policy above. Observation of frozen original
// bytes accepts finite references, with separate cached and expanded budgets.
fn parse_yaml(
    bytes: &[u8],
    historical_references: bool,
) -> Result<Map<String, Value>, DocumentError> {
    if bytes.len() > MAX_BYTES {
        return Err(DocumentError::TooLarge);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| DocumentError::Malformed)?;
    // YAML 1.1 treats these raw characters as line breaks, while YAML 1.2
    // treats them as string characters. Escaped literals remain supported.
    if text.contains(['\u{85}', '\u{2028}', '\u{2029}']) {
        return Err(DocumentError::UnsupportedYaml);
    }
    if historical_references {
        aliases::unique_names(text)?;
    }
    let mut parser = Parser::new_from_str(text);
    let mut stack = Vec::new();
    let mut frame_anchors = Vec::new();
    let mut references = aliases::References::default();
    let mut root = None;
    let mut documents = 0;
    for count in 0..=MAX_NODES {
        references.events(count)?;
        let (event, _) = parser.next_token().map_err(|_| DocumentError::Malformed)?;
        match event {
            Event::StreamStart | Event::DocumentEnd => {}
            Event::DocumentStart => {
                documents += 1;
                if documents != 1 {
                    return Err(DocumentError::Malformed);
                }
            }
            Event::StreamEnd => break,
            Event::Alias(id) => {
                if !historical_references {
                    return Err(DocumentError::UnsupportedYaml);
                }
                let value = references.expand(id, stack.len())?;
                append(&mut stack, &mut root, value)?;
            }
            Event::Scalar(value, style, anchor, tag) => {
                if anchor != 0 && !historical_references || tag.is_some() {
                    return Err(DocumentError::UnsupportedYaml);
                }
                let value = scalar(value, style)?;
                references.reserve(anchor)?;
                references.complete(anchor, &value)?;
                append(&mut stack, &mut root, value)?;
            }
            Event::SequenceStart(anchor, ref tag) | Event::MappingStart(anchor, ref tag) => {
                if anchor != 0 && !historical_references || tag.is_some() {
                    return Err(DocumentError::UnsupportedYaml);
                }
                if stack.len() >= MAX_DEPTH {
                    return Err(DocumentError::Complexity);
                }
                references.reserve(anchor)?;
                frame_anchors.push(anchor);
                stack.push(if matches!(event, Event::SequenceStart(..)) {
                    Frame::Sequence(Vec::new())
                } else {
                    Frame::Mapping(Map::new(), None)
                });
            }
            Event::SequenceEnd => {
                let Some(Frame::Sequence(items)) = stack.pop() else {
                    return Err(DocumentError::Malformed);
                };
                let value = Value::Array(items);
                references
                    .complete(frame_anchors.pop().ok_or(DocumentError::Malformed)?, &value)?;
                append(&mut stack, &mut root, value)?;
            }
            Event::MappingEnd => {
                let Some(Frame::Mapping(items, None)) = stack.pop() else {
                    return Err(DocumentError::Malformed);
                };
                let value = Value::Object(items);
                references
                    .complete(frame_anchors.pop().ok_or(DocumentError::Malformed)?, &value)?;
                append(&mut stack, &mut root, value)?;
            }
            Event::Nothing => return Err(DocumentError::Malformed),
        }
    }
    if documents != 1 || !stack.is_empty() {
        return Err(DocumentError::Malformed);
    }
    match root {
        Some(Value::Object(mapping)) => Ok(mapping),
        _ => Err(DocumentError::Malformed),
    }
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
        || map
            .get("identity_version")
            .is_some_and(|v| !matches!(v.as_u64(), Some(1 | 2)))
    {
        return Err(DocumentError::Version);
    }
    Ok(())
}
pub fn validate_project(map: &Map<String, Value>, project: &str) -> Result<(), DocumentError> {
    identity::validate_id(project).map_err(|_| DocumentError::Identity)?;
    version(map)?;
    exact(map, "project", project)
}
pub fn validate_run(
    map: &Map<String, Value>,
    project: &str,
    run: &str,
) -> Result<(), DocumentError> {
    identity::validate_observation_id(run).map_err(|_| DocumentError::Identity)?;
    validate_project(map, project)?;
    exact(map, "run_id", run)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Provenance {
    /// Only completeness of these named comparisons, not all scientific fields.
    pub compared_fields_complete: bool,
    pub compared_fields: Vec<String>,
    pub missing_fields: Vec<String>,
    pub project_inherited_from_json_run: bool,
}
pub fn validate_attempt(
    run: &Document,
    attempt: &Document,
    project: &str,
    run_id: &str,
    attempt_id: &str,
) -> Result<Provenance, DocumentError> {
    identity::validate_id(attempt_id).map_err(|_| DocumentError::Identity)?;
    validate_run(&run.metadata, project, run_id)?;
    version(&attempt.metadata)?;
    let inherited = run.source == ManifestSource::RunJson
        && attempt.source == ManifestSource::AttemptJson
        && field(&attempt.metadata, "project").is_none();
    if !inherited {
        exact(&attempt.metadata, "project", project)?;
    }
    exact(&attempt.metadata, "run_id", run_id)?;
    exact(&attempt.metadata, "attempt_id", attempt_id)?;
    let mut missing_fields = Vec::new();
    for key in ["source_id", "image_id", "config_path", "seed", "campaign"] {
        let seed_value = |map: &Map<String, Value>| {
            field(map, "seed").cloned().or_else(|| {
                map.get("resolved_config")
                    .and_then(|v| v.get("seed"))
                    .filter(|v| !v.is_null())
                    .cloned()
            })
        };
        let (mut left, mut right) = if key == "seed" {
            (seed_value(&run.metadata), seed_value(&attempt.metadata))
        } else {
            (
                field(&run.metadata, key).cloned(),
                field(&attempt.metadata, key).cloned(),
            )
        };
        if key == "seed" && left.is_none() && right.is_none() {
            left = run
                .metadata
                .get("resolved_config")
                .and_then(|v| v.get("seeds"))
                .filter(|v| !v.is_null())
                .cloned();
            right = attempt
                .metadata
                .get("resolved_config")
                .and_then(|v| v.get("seeds"))
                .filter(|v| !v.is_null())
                .cloned();
        }
        match (left, right) {
            (Some(left), Some(right)) if !python_equal(&left, &right) => {
                return Err(DocumentError::Provenance);
            }
            (None, _) | (_, None) => missing_fields.push(key.into()),
            _ => {}
        }
    }
    Ok(Provenance {
        compared_fields_complete: missing_fields.is_empty(),
        compared_fields: ["source_id", "image_id", "config_path", "seed", "campaign"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        missing_fields,
        project_inherited_from_json_run: inherited,
    })
}
