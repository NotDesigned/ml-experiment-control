//! Read historical checkpoint receipts without a native submission, lock
//! creation, sealing, remote payload access or execution authority. The caller
//! holds the Run deletion fence and pins the exact Run/Attempt manifests.
use super::*;
use crate::experiments::AttemptRecord;
use ml_exp_core::checkpoints::Reference;
use serde_json::json;

fn child(root: &Directory, parts: &[&str]) -> Result<Option<Directory>> {
    let mut directory = root.clone();
    for part in parts {
        if !directory.entry_exists(OsStr::new(part))? {
            return Ok(None);
        }
        directory = directory.child(OsStr::new(part), false)?;
        directory.verify_private_parent()?;
    }
    Ok(Some(directory))
}
fn read(directory: &Directory, name: &str) -> Result<Option<Vec<u8>>> {
    let name = OsStr::new(name);
    if !directory.entry_exists(name)? {
        return Ok(None);
    }
    Ok(Some(directory.read_regular(name, MAX_DOCUMENT_BYTES)?))
}
fn mapping(bytes: &[u8]) -> Result<Value> {
    ml_exp_core::json::mapping(bytes)
        .map(Value::Object)
        .map_err(|_| Error::Document)
}
fn names(directory: &Directory) -> Result<Vec<std::ffi::OsString>> {
    let mut names = directory.entry_names_limited(checkpoints::MAX_FILES * 2 + 1)?;
    if names.len() > checkpoints::MAX_FILES * 2 + 1 {
        return Err(Error::Document);
    }
    names.sort();
    for name in &names {
        let name = name.to_str().ok_or(Error::Document)?;
        if name == LOCK {
            continue;
        }
        let id = name
            .strip_prefix("withdrawn.")
            .unwrap_or(name)
            .strip_suffix(".json")
            .ok_or(Error::Document)?;
        checkpoints::validate_checkpoint_id(id).map_err(|_| Error::Document)?;
    }
    Ok(names)
}
fn context(transfer: &[u8], record: &AttemptRecord, attempt: &str) -> Result<Context> {
    let transfer = mapping(transfer)?;
    let context =
        checkpoints::context(&transfer["checkpoint_state"]).map_err(|_| Error::Document)?;
    for (key, expected) in [
        ("project", context.project.as_str()),
        ("run_id", context.run_id.as_str()),
        ("attempt_id", attempt),
    ] {
        if transfer[key].as_str() != Some(expected) {
            return Err(Error::Document);
        }
    }
    if context.attempt_id != attempt
        || record.run.metadata.get("project").and_then(Value::as_str) != Some(&context.project)
        || record.run.metadata.get("run_id").and_then(Value::as_str) != Some(&context.run_id)
        || [(&record.run, false), (&record.attempt, true)]
            .into_iter()
            .any(|(doc, is_attempt)| {
                doc.metadata.get("source_id").and_then(Value::as_str) != Some(&context.source_id)
                    || doc.metadata.get("image_id").and_then(Value::as_str)
                        != Some(&context.image_id)
                    || is_attempt
                        && doc.metadata.get("attempt_id").and_then(Value::as_str) != Some(attempt)
            })
    {
        return Err(Error::Document);
    }
    Ok(context)
}
fn projected(bytes: &[u8], context: &Context, id: &str) -> Result<Value> {
    let value = mapping(bytes)?;
    let reference = Reference {
        run_id: context.run_id.clone(),
        attempt_id: context.attempt_id.clone(),
        checkpoint_id: id.into(),
    };
    checkpoints::validate_document(&value, Some(context), Some(id)).map_err(|_| Error::Document)?;
    checkpoints::selected_at(
        &reference,
        &value,
        &context.project,
        &context.storage_scope,
        &context.state_root,
    )
    .map_err(|_| Error::Document)?;
    // Preserve the standard receipt; never return transfer credentials or
    // arbitrary historical extension fields in the public checkpoint API.
    let document = checkpoints::document(
        context,
        &json!({"step":value["step"],"files":value["files"]}),
    )
    .map_err(|_| Error::Document)?;
    checkpoints::register(
        &document,
        value["registered_at"].as_str().ok_or(Error::Document)?,
    )
    .map_err(|_| Error::Document)
}

pub(crate) fn observe(
    registry: &Directory,
    record: &AttemptRecord,
    attempt: &str,
    id: Option<&str>,
) -> Result<Vec<Value>> {
    let project = record.run.metadata["project"]
        .as_str()
        .ok_or(Error::Document)?;
    let run = record.run.metadata["run_id"]
        .as_str()
        .ok_or(Error::Document)?;
    ml_exp_core::identity::validate_id(project).map_err(|_| Error::Document)?;
    ml_exp_core::identity::validate_id(run).map_err(|_| Error::Document)?;
    ml_exp_core::identity::validate_attempt(attempt).map_err(|_| Error::Document)?;
    if let Some(id) = id {
        checkpoints::validate_checkpoint_id(id).map_err(|_| Error::Document)?;
    }
    let Some(directory) = child(registry, &[NAMESPACE, project, run, attempt])? else {
        return Ok(Vec::new());
    };
    let before = names(&directory)?;
    let selected = before
        .iter()
        .filter_map(|name| {
            let name = name.to_str()?;
            let value = name.strip_suffix(".json")?;
            (value.starts_with("checkpoint.") && id.is_none_or(|id| id == value)).then_some(value)
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    let transfer_dir =
        child(registry, &["artifact-transfers", project, run])?.ok_or(Error::Document)?;
    let transfer_name = format!("{attempt}.json");
    let transfer = read(&transfer_dir, &transfer_name)?.ok_or(Error::Document)?;
    let context = context(&transfer, record, attempt)?;
    let mut values = Vec::new();
    let mut originals = Vec::new();
    let mut total_bytes = 0usize;
    for id in selected {
        if directory.entry_exists(OsStr::new(&format!("withdrawn.{id}.json")))? {
            continue;
        }
        let name = format!("{id}.json");
        let bytes = read(&directory, &name)?.ok_or(Error::Document)?;
        total_bytes = total_bytes
            .checked_add(bytes.len())
            .filter(|v| *v <= crate::read_model::MAX_RESPONSE_BYTES)
            .ok_or(Error::Document)?;
        values.push(projected(&bytes, &context, id)?);
        originals.push((name, bytes));
    }
    if before != names(&directory)?
        || read(&transfer_dir, &transfer_name)?.as_ref() != Some(&transfer)
    {
        return Err(Error::Document);
    }
    for (name, bytes) in originals {
        if read(&directory, &name)?.as_ref() != Some(&bytes) {
            return Err(Error::Document);
        }
    }
    directory.verify_private_parent()?;
    transfer_dir.verify_private_parent()?;
    registry.verify_private_parent()?;
    Ok(values)
}
