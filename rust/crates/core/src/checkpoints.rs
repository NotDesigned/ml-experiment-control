//! Pure persistent-checkpoint manifests, identities and restore plans.
//!
//! Checkpoint bytes remain on backend storage. This module only validates the
//! worker's ready manifest, derives its immutable identity, binds it to the
//! exact Run/Attempt and returns a restore plan. Filesystem verification,
//! sealing and HTTP registration belong to an outer effect shell.
use crate::{canonical, identity, sources};
use serde_json::{Map, Value, json};
use sha2::Digest;
use std::collections::BTreeSet;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

pub const MANIFEST_LIMIT: usize = 256 * 1024;
pub const MAX_FILES: usize = 20_000;
pub const MAX_PATH_BYTES: usize = 4096;
pub const MAX_GENERATION_BYTES: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid persistent checkpoint manifest")]
    Manifest,
    #[error("invalid checkpoint identity or provenance")]
    Identity,
    #[error("checkpoint generation is already bound to another manifest")]
    GenerationConflict,
    #[error("registered checkpoint identity differs")]
    Document,
    #[error("invalid checkpoint registration timestamp")]
    Timestamp,
    #[error("checkpoint storage scope differs")]
    StorageScope,
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRecord {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub step: u64,
    pub files: Vec<FileRecord>,
    pub generation: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Context {
    pub project: String,
    pub run_id: String,
    pub attempt_id: String,
    pub source_id: String,
    pub image_id: String,
    pub storage_scope: String,
    pub state_root: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestorePlan {
    pub checkpoint_id: String,
    pub step: u64,
    pub generation: String,
    pub state_root: String,
    pub files: Vec<FileRecord>,
}

fn object(value: &Value) -> Result<&Map<String, Value>> {
    value.as_object().ok_or(Error::Manifest)
}

fn exact_keys<'a>(value: &'a Value, keys: &[&str]) -> Result<&'a Map<String, Value>> {
    let fields = object(value)?;
    if fields.len() != keys.len() || keys.iter().any(|key| !fields.contains_key(*key)) {
        return Err(Error::Manifest);
    }
    Ok(fields)
}

fn string<'a>(fields: &'a Map<String, Value>, key: &str, maximum: usize) -> Result<&'a str> {
    let value = fields
        .get(key)
        .and_then(Value::as_str)
        .ok_or(Error::Manifest)?;
    if value.is_empty() || value.len() > maximum || value.contains(['\\', '\0']) {
        return Err(Error::Manifest);
    }
    Ok(value)
}

fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn generation(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_GENERATION_BYTES
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric() || (index > 0 && b"_.-".contains(&byte))
        })
}

fn checkpoint_path(value: &str) -> Result<(&str, &str)> {
    if value.is_empty() || value.len() > MAX_PATH_BYTES || value.contains(['\\', '\0']) {
        return Err(Error::Manifest);
    }
    let parts: Vec<_> = value.split('/').collect();
    if parts.len() < 3
        || parts
            .iter()
            .any(|part| part.is_empty() || *part == "." || *part == ".." || part.starts_with('.'))
        || parts[0] != "checkpoints"
        || !generation(parts[1])
    {
        return Err(Error::Manifest);
    }
    Ok((parts[1], value))
}

fn bounded_manifest(value: &Value) -> Result<()> {
    let encoded = serde_json::to_vec(value).map_err(|_| Error::Manifest)?;
    if encoded.len() > MANIFEST_LIMIT {
        return Err(Error::Manifest);
    }
    Ok(())
}

/// Validate the worker's `{step, files}` ready marker and canonicalize file order.
pub fn manifest(value: &Value) -> Result<Manifest> {
    let fields = exact_keys(value, &["step", "files"])?;
    bounded_manifest(value)?;
    let step = fields
        .get("step")
        .and_then(Value::as_u64)
        .ok_or(Error::Manifest)?;
    let rows = fields
        .get("files")
        .and_then(Value::as_array)
        .ok_or(Error::Manifest)?;
    if rows.is_empty() || rows.len() > MAX_FILES {
        return Err(Error::Manifest);
    }
    let mut files = Vec::with_capacity(rows.len());
    let mut seen = BTreeSet::new();
    let mut generations = BTreeSet::new();
    for row in rows {
        let fields = exact_keys(row, &["path", "bytes", "sha256"])?;
        let path = string(fields, "path", MAX_PATH_BYTES)?;
        let (generation, _) = checkpoint_path(path)?;
        let bytes = fields
            .get("bytes")
            .and_then(Value::as_u64)
            .filter(|value| *value < (1u64 << 63))
            .ok_or(Error::Manifest)?;
        let sha = string(fields, "sha256", 64)?;
        if !sha256(sha) || !seen.insert(path.to_owned()) {
            return Err(Error::Manifest);
        }
        generations.insert(generation.to_owned());
        files.push(FileRecord {
            path: path.to_owned(),
            bytes,
            sha256: sha.to_owned(),
        });
    }
    if generations.len() != 1 {
        return Err(Error::Manifest);
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(Manifest {
        step,
        files,
        generation: generations.into_iter().next().ok_or(Error::Manifest)?,
    })
}

fn context_string(value: &Value, key: &str, maximum: usize) -> Result<String> {
    let fields = object(value)?;
    Ok(string(fields, key, maximum)?.to_owned())
}

pub fn context(value: &Value) -> Result<Context> {
    let fields = object(value)?;
    if fields.len() != 7 {
        return Err(Error::Identity);
    }
    let project = context_string(value, "project", 128)?;
    let run_id = context_string(value, "run_id", 128)?;
    let attempt_id = context_string(value, "attempt_id", 128)?;
    identity::validate_id(&project).map_err(|_| Error::Identity)?;
    identity::validate_id(&run_id).map_err(|_| Error::Identity)?;
    identity::validate_attempt(&attempt_id).map_err(|_| Error::Identity)?;
    let source_id = context_string(value, "source_id", 128)?;
    sources::validate_source_id(&source_id).map_err(|_| Error::Identity)?;
    let image_id = context_string(value, "image_id", 512)?;
    let storage_scope = context_string(value, "storage_scope", 256)?;
    let state_root = context_string(value, "state_root", 4096)?;
    if !state_root.starts_with('/') || state_root.split('/').any(|part| part == "..") {
        return Err(Error::Identity);
    }
    Ok(Context {
        project,
        run_id,
        attempt_id,
        source_id,
        image_id,
        storage_scope,
        state_root,
    })
}

fn file_values(files: &[FileRecord]) -> Vec<Value> {
    files
        .iter()
        .map(|file| json!({"path":file.path,"bytes":file.bytes,"sha256":file.sha256}))
        .collect()
}

fn context_value(context: &Context) -> Value {
    json!({
        "project": context.project,
        "run_id": context.run_id,
        "attempt_id": context.attempt_id,
        "source_id": context.source_id,
        "image_id": context.image_id,
        "storage_scope": context.storage_scope,
        "state_root": context.state_root,
    })
}

fn identity_digest(value: &Value) -> Result<String> {
    let bytes = canonical::encode(
        value,
        canonical::JsonOptions {
            sort_keys: true,
            ensure_ascii: false,
            compact: true,
        },
    )
    .map_err(|_| Error::Document)?;
    Ok(format!("checkpoint.{:x}", sha2::Sha256::digest(bytes)))
}

/// Build the immutable checkpoint document and derive its exact identity.
pub fn document(context: &Context, ready: &Value) -> Result<Value> {
    let manifest = manifest(ready)?;
    let mut fields = context_value(context)
        .as_object()
        .cloned()
        .ok_or(Error::Document)?;
    fields.insert("schema_version".into(), Value::from(1));
    fields.insert("step".into(), Value::from(manifest.step));
    fields.insert("files".into(), Value::Array(file_values(&manifest.files)));
    let base = Value::Object(fields);
    let checkpoint_id = identity_digest(&base)?;
    let mut result = base.as_object().cloned().ok_or(Error::Document)?;
    result.insert("checkpoint_id".into(), Value::String(checkpoint_id));
    Ok(Value::Object(result))
}

pub fn validate_checkpoint_id(value: &str) -> Result<&str> {
    if !value.starts_with("checkpoint.") || !value.strip_prefix("checkpoint.").is_some_and(sha256) {
        return Err(Error::Identity);
    }
    Ok(value)
}

fn valid_checkpoint_id(value: &str) -> bool {
    value.strip_prefix("checkpoint.").is_some_and(sha256)
}

/// Validate a persisted document against its exact context and optional ID.
pub fn validate_document(
    value: &Value,
    expected_context: Option<&Context>,
    expected_id: Option<&str>,
) -> Result<(Context, Manifest)> {
    let fields = object(value)?;
    if fields.get("schema_version").and_then(Value::as_u64) != Some(1)
        || !valid_checkpoint_id(
            fields
                .get("checkpoint_id")
                .and_then(Value::as_str)
                .unwrap_or(""),
        )
    {
        return Err(Error::Document);
    }
    let mut context_fields = Map::new();
    for key in [
        "project",
        "run_id",
        "attempt_id",
        "source_id",
        "image_id",
        "storage_scope",
        "state_root",
    ] {
        context_fields.insert(key.into(), fields.get(key).cloned().ok_or(Error::Document)?);
    }
    let context_value = Value::Object(context_fields);
    let actual_context = context(&context_value)?;
    if expected_context.is_some_and(|expected| expected != &actual_context) {
        return Err(Error::Document);
    }
    if expected_id.is_some_and(|expected| expected != fields["checkpoint_id"].as_str().unwrap()) {
        return Err(Error::Document);
    }
    let ready = json!({
        "step": fields.get("step").cloned().ok_or(Error::Document)?,
        "files": fields.get("files").cloned().ok_or(Error::Document)?,
    });
    let actual_manifest = manifest(&ready)?;
    let expected = document(&actual_context, &ready)?;
    if expected["checkpoint_id"] != fields["checkpoint_id"] {
        return Err(Error::Document);
    }
    Ok((actual_context, actual_manifest))
}

pub fn register(value: &Value, registered_at: &str) -> Result<Value> {
    let (context, _) = validate_document(value, None, None)?;
    let _ = context;
    if OffsetDateTime::parse(registered_at, &Rfc3339).is_err() {
        return Err(Error::Timestamp);
    }
    let mut result = object(value)?.clone();
    result.insert("status".into(), Value::String("REGISTERED".into()));
    result.insert("verification".into(), Value::String("worker-sha256".into()));
    result.insert("registered_at".into(), Value::String(registered_at.into()));
    Ok(Value::Object(result))
}

pub fn restore_plan(value: &Value, expected_scope: Option<&str>) -> Result<RestorePlan> {
    let (context, manifest) = validate_document(value, None, None)?;
    if expected_scope.is_some_and(|scope| scope != context.storage_scope) {
        return Err(Error::StorageScope);
    }
    Ok(RestorePlan {
        checkpoint_id: value["checkpoint_id"]
            .as_str()
            .ok_or(Error::Document)?
            .to_owned(),
        step: manifest.step,
        generation: manifest.generation,
        state_root: context.state_root,
        files: manifest.files,
    })
}

pub fn ensure_generation_compatible(existing: &[Value], candidate: &Value) -> Result<()> {
    let (_, candidate_manifest) = validate_document(candidate, None, None)?;
    for value in existing {
        let (_, existing_manifest) = validate_document(value, None, None)?;
        if existing_manifest.generation == candidate_manifest.generation
            && value["checkpoint_id"] != candidate["checkpoint_id"]
        {
            return Err(Error::GenerationConflict);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context_value() -> Value {
        json!({
            "project":"demo",
            "run_id":"trial",
            "attempt_id":"attempt-001",
            "source_id":format!("source.{}", "a".repeat(64)),
            "image_id":format!("sha256:{}", "b".repeat(64)),
            "storage_scope":format!("storage.{}", "c".repeat(64)),
            "state_root":"/data/demo/trial/attempt-001/state"
        })
    }

    fn ready() -> Value {
        json!({
            "step":42,
            "files":[
                {"path":"checkpoints/step-0042/model.pt","bytes":7,"sha256":"9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"},
                {"path":"checkpoints/step-0042/optimizer.bin","bytes":3,"sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}
            ]
        })
    }

    #[test]
    fn python_checkpoint_identity_and_sorted_manifest_are_exact() {
        let context = context(&context_value()).unwrap();
        let value = document(&context, &ready()).unwrap();
        assert_eq!(
            value["checkpoint_id"],
            "checkpoint.8eba860d7883dfe578d7e998cf184ef752493fe0b041b6c9c3ebfb3acfac67fd"
        );
        let (_, manifest) = validate_document(&value, Some(&context), None).unwrap();
        assert_eq!(manifest.files[0].path, "checkpoints/step-0042/model.pt");
    }

    #[test]
    fn mixed_generations_paths_and_identity_changes_fail_closed() {
        let context = context(&context_value()).unwrap();
        let value = document(&context, &ready()).unwrap();
        let mut mixed = ready();
        mixed["files"]
            .as_array_mut()
            .unwrap()
            .push(json!({"path":"checkpoints/other/state.bin","bytes":1,"sha256":"a".repeat(64)}));
        assert_eq!(manifest(&mixed), Err(Error::Manifest));
        let mut changed = value.clone();
        changed["step"] = 43.into();
        assert_eq!(
            validate_document(&changed, None, None),
            Err(Error::Document)
        );
        assert_eq!(
            restore_plan(&value, Some("storage.bad")),
            Err(Error::StorageScope)
        );
    }

    #[test]
    fn registration_and_generation_conflicts_are_explicit() {
        let context = context(&context_value()).unwrap();
        let value = document(&context, &ready()).unwrap();
        let registered = register(&value, "2026-10-28T00:00:00Z").unwrap();
        assert_eq!(registered["status"], "REGISTERED");
        assert!(ensure_generation_compatible(&[registered], &value).is_ok());
        let mut different = ready();
        different["files"][0]["bytes"] = 8.into();
        let different = document(&context, &different).unwrap();
        assert_eq!(
            ensure_generation_compatible(&[value], &different),
            Err(Error::GenerationConflict)
        );
    }
}
