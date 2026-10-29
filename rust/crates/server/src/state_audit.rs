//! Offline state observation. Opens existing capabilities only, without the
//! native workspace marker, initialization, provider calls or payload reads.
use crate::{
    deletion::RunFence,
    experiments::{self, EvidenceDocument},
    read_config, records,
};
use ml_exp_core::state_audit::{self as domain, Fingerprint, Issue, Key, Snapshot};
use ml_exp_io::directory::Directory;
use serde_json::{Value, json};
use std::{ffi::OsStr, path::Path};

mod catalog;
mod database;
mod isolation;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("AUDIT_READ_ONLY_ISOLATION_UNAVAILABLE")]
    Isolation,
    #[error("AUDIT_METADATA_UNAVAILABLE")]
    Evidence,
    #[error("AUDIT_DATABASE_UNAVAILABLE")]
    Database,
    #[error("AUDIT_DATABASE_BUSY")]
    DatabaseBusy,
    #[error("AUDIT_DATABASE_BYTE_LIMIT")]
    DatabaseBytes,
    #[error("AUDIT_DATABASE_ROW_LIMIT")]
    DatabaseRows,
    #[error("AUDIT_DATABASE_SHAPE_INVALID")]
    DatabaseShape,
    #[error("AUDIT_DATABASE_VALUE_LIMIT")]
    DatabaseValue,
    #[error("AUDIT_DATABASE_BINDING_UNAVAILABLE")]
    DatabaseBinding,
    #[error("AUDIT_DATABASE_SQL_VALUE_INVALID")]
    DatabaseSqlValue,
    #[error("AUDIT_DATABASE_READ_ONLY_UNAVAILABLE")]
    DatabaseReadOnly,
}
impl Error {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Isolation => "AUDIT_READ_ONLY_ISOLATION_UNAVAILABLE",
            Self::Evidence => "AUDIT_METADATA_UNAVAILABLE",
            Self::Database => "AUDIT_DATABASE_UNAVAILABLE",
            Self::DatabaseBusy => "AUDIT_DATABASE_BUSY",
            Self::DatabaseBytes => "AUDIT_DATABASE_BYTE_LIMIT",
            Self::DatabaseRows => "AUDIT_DATABASE_ROW_LIMIT",
            Self::DatabaseShape => "AUDIT_DATABASE_SHAPE_INVALID",
            Self::DatabaseValue => "AUDIT_DATABASE_VALUE_LIMIT",
            Self::DatabaseBinding => "AUDIT_DATABASE_BINDING_UNAVAILABLE",
            Self::DatabaseSqlValue => "AUDIT_DATABASE_SQL_VALUE_INVALID",
            Self::DatabaseReadOnly => "AUDIT_DATABASE_READ_ONLY_UNAVAILABLE",
        }
    }
}
type Result<T> = std::result::Result<T, Error>;
fn key(
    project: &str,
    run: Option<&str>,
    attempt: Option<&str>,
    kind: &str,
    item: Option<&str>,
) -> Key {
    Key {
        project: project.into(),
        run: run.map(str::to_owned),
        attempt: attempt.map(str::to_owned),
        kind: kind.into(),
        item: item.map(str::to_owned),
    }
}
struct Audit {
    entries: Vec<Fingerprint>,
    issues: Vec<Issue>,
}
impl Audit {
    fn push(&mut self, key: Key, value: &Value, records: u64) -> Result<()> {
        if self.entries.len() >= domain::MAX_ENTRIES {
            return Err(Error::Evidence);
        }
        self.entries
            .push(domain::fingerprint(key, value, records).map_err(|_| Error::Evidence)?);
        Ok(())
    }
    fn issue(&mut self, key: Key, code: &str) -> Result<()> {
        if self.issues.len() >= domain::MAX_ENTRIES {
            return Err(Error::Evidence);
        }
        self.issues.push(Issue {
            key,
            code: code.into(),
        });
        Ok(())
    }
    fn evidence(&mut self, key: Key, evidence: &EvidenceDocument) -> Result<()> {
        let value = match evidence {
            EvidenceDocument::Missing => json!({"availability":"MISSING"}),
            EvidenceDocument::Present { raw_sha256, .. } => {
                json!({"availability":"PRESENT","raw_sha256":raw_sha256})
            }
            EvidenceDocument::Unavailable { code, .. } => {
                self.issue(key.clone(), "EVIDENCE_UNAVAILABLE")?;
                json!({"availability":"UNAVAILABLE","code":code})
            }
        };
        self.push(key, &value, 1)
    }
}
fn optional_child(parent: &Directory, name: &str) -> Result<Option<Directory>> {
    if !parent
        .entry_exists(OsStr::new(name))
        .map_err(|_| Error::Evidence)?
    {
        return Ok(None);
    }
    let child = parent
        .child(OsStr::new(name), false)
        .map_err(|_| Error::Evidence)?;
    child.verify_private_parent().map_err(|_| Error::Evidence)?;
    Ok(Some(child))
}
fn nested(root: &Directory, names: &[&str]) -> Result<Option<Directory>> {
    let mut directory = root.clone();
    for name in names {
        let Some(child) = optional_child(&directory, name)? else {
            return Ok(None);
        };
        directory = child;
    }
    Ok(Some(directory))
}
fn mapping(directory: &Directory, name: &str) -> Result<Option<(Value, String)>> {
    if !directory
        .entry_exists(OsStr::new(name))
        .map_err(|_| Error::Evidence)?
    {
        return Ok(None);
    }
    let bytes = directory
        .read_regular(OsStr::new(name), 4 * 1024 * 1024)
        .map_err(|_| Error::Evidence)?;
    let value = experiments::parse_mapping(&bytes, experiments::Format::Json)
        .map_err(|_| Error::Evidence)?;
    Ok(Some((
        Value::Object(value),
        ml_exp_core::slurm::helper::sha256_bytes(&bytes),
    )))
}
fn transfer(
    root: &Directory,
    project: &str,
    run: &str,
    attempt: &str,
    audit: &mut Audit,
) -> Result<Option<Value>> {
    let result = nested(root, &["artifact-transfers", project, run])?
        .map(|directory| mapping(&directory, &format!("{attempt}.json")))
        .transpose()?
        .flatten();
    let k = key(project, Some(run), Some(attempt), "result-transfer", None);
    match result {
        Some((value, sha)) => {
            if [
                ("project", project),
                ("run_id", run),
                ("attempt_id", attempt),
            ]
            .iter()
            .any(|(k, v)| value.get(*k).is_some_and(|x| x.as_str() != Some(*v)))
            {
                audit.issue(k.clone(), "TRANSFER_IDENTITY_CONFLICT")?;
            }
            audit.push(k, &json!({"raw_sha256":sha}), 1)?;
            Ok(Some(value))
        }
        None => {
            audit.push(k, &Value::Null, 0)?;
            Ok(None)
        }
    }
}
fn checkpoints(
    root: &Directory,
    snapshot: &experiments::Snapshot,
    attempt: &str,
    transfer: Option<&Value>,
    audit: &mut Audit,
) -> Result<()> {
    let project = &snapshot.project;
    let run = &snapshot.run_id;
    let prefix = key(project, Some(run), Some(attempt), "checkpoints", None);
    let Some(directory) = nested(root, &["persistent-checkpoints", project, run, attempt])? else {
        return audit.push(prefix, &json!([]), 0);
    };
    let mut names = directory
        .entry_names_limited(20_001)
        .map_err(|_| Error::Evidence)?;
    if names.len() > 20_000 {
        return Err(Error::Evidence);
    }
    names.sort();
    let mut inventory = Vec::new();
    for name in &names {
        let name = name.to_str().ok_or(Error::Evidence)?;
        if name == ".checkpoints.lock" {
            continue;
        }
        let Some(id) = name.strip_suffix(".json") else {
            return Err(Error::Evidence);
        };
        let (value, sha) = mapping(&directory, name)?.ok_or(Error::Evidence)?;
        if id.starts_with("withdrawn.") {
            ml_exp_core::checkpoints::validate_checkpoint_id(
                id.strip_prefix("withdrawn.").ok_or(Error::Evidence)?,
            )
            .map_err(|_| Error::Evidence)?;
            inventory.push(json!({"name":name,"sha256":sha}));
            continue;
        }
        let k = key(project, Some(run), Some(attempt), "checkpoint", Some(id));
        let expected = transfer
            .and_then(|t| t.get("checkpoint_state"))
            .filter(|v| !v.is_null())
            .map(|v| serde_json::from_value::<ml_exp_core::checkpoints::Context>(v.clone()))
            .transpose()
            .map_err(|_| Error::Evidence)?;
        match ml_exp_core::checkpoints::validate_document(&value, expected.as_ref(), Some(id)) {
            Ok((context, _))
                if context.project == *project
                    && context.run_id == *run
                    && context.attempt_id == attempt
                    && snapshot
                        .run
                        .metadata
                        .get("source_id")
                        .and_then(Value::as_str)
                        == Some(&context.source_id)
                    && snapshot
                        .run
                        .metadata
                        .get("image_id")
                        .and_then(Value::as_str)
                        == Some(&context.image_id) =>
            {
                if expected.is_none() {
                    audit.issue(k.clone(), "CHECKPOINT_STORAGE_BINDING_UNAVAILABLE")?;
                }
            }
            _ => audit.issue(k.clone(), "CHECKPOINT_IDENTITY_CONFLICT")?,
        }
        audit.push(k, &json!({"raw_sha256":sha}), 1)?;
        inventory.push(json!({"name":name,"sha256":sha}));
    }
    if names
        != directory
            .entry_names_limited(20_001)
            .map_err(|_| Error::Evidence)?
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    {
        return Err(Error::Evidence);
    }
    audit.push(prefix, &Value::Array(inventory), names.len() as u64)
}
fn journal(
    settings: &read_config::Settings,
    root: &Directory,
    snapshot: &experiments::Snapshot,
    audit: &mut Audit,
) -> Result<()> {
    let project = &snapshot.project;
    let run = &snapshot.run_id;
    let k = key(project, Some(run), None, "journal", None);
    let present = optional_child(root, "tracking")?
        .is_some_and(|d| d.entry_exists(OsStr::new("outbox.sqlite")).unwrap_or(true));
    if !present {
        return audit.push(k, &Value::Null, 0);
    }
    let store = records::Store::read_only(&settings.registry_root);
    let guard = store.guard(project, run).map_err(|_| Error::Evidence)?;
    let binding = store.binding(&guard).map_err(|_| Error::Evidence)?;
    let mut scopes = store.run_scopes(&guard).map_err(|_| Error::Evidence)?;
    scopes.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    if scopes.len() > 257 {
        return Err(Error::Evidence);
    }
    if binding.is_some() && !crate::read_api::journal_binding_matches(snapshot, binding.as_ref()) {
        audit.issue(k.clone(), "JOURNAL_IDENTITY_CONFLICT")?;
    }
    audit.push(k, &json!({"workspace":store.workspace_id(&guard).map_err(|_| Error::Evidence)?,"binding":binding,"scopes":scopes}), scopes.len() as u64)?;
    for attempt in &snapshot.attempts {
        let k = key(
            project,
            Some(run),
            Some(&attempt.attempt_id),
            "metrics",
            None,
        );
        let values = store
            .metric_records_bounded(&guard, &attempt.attempt_id, 8 * 1024 * 1024, 100_000)
            .map_err(|_| Error::Evidence)?;
        if (!values.is_empty() && binding.is_none())
            || crate::read_model::validate_journal_observations(
                &values,
                snapshot
                    .run
                    .metadata
                    .get("evaluation")
                    .unwrap_or(&Value::Null),
            )
            .is_err()
        {
            audit.issue(k.clone(), "METRIC_IDENTITY_OR_SCHEMA_CONFLICT")?;
        }
        audit.push(k, &json!(values), values.len() as u64)?;
    }
    Ok(())
}
fn controls(root: &Directory, actions: Option<&Directory>, audit: &mut Audit) -> Result<()> {
    let lifecycle = if root
        .entry_exists(OsStr::new("registry.json"))
        .map_err(|_| Error::Evidence)?
    {
        Some(root.clone())
    } else {
        optional_child(root, "projects")?
    };
    for name in ["registry.json", "events.jsonl"] {
        let bytes = lifecycle
            .as_ref()
            .map(|directory| -> Result<_> {
                if !directory
                    .entry_exists(OsStr::new(name))
                    .map_err(|_| Error::Evidence)?
                {
                    return Ok(None);
                }
                directory
                    .read_regular(OsStr::new(name), 8 * 1024 * 1024)
                    .map(Some)
                    .map_err(|_| Error::Evidence)
            })
            .transpose()?
            .flatten();
        let value = bytes
            .as_ref()
            .map(|b| json!({"raw_sha256":ml_exp_core::slurm::helper::sha256_bytes(b)}))
            .unwrap_or(Value::Null);
        audit.push(
            key("native-state", None, None, "project-lifecycle", Some(name)),
            &value,
            u64::from(bytes.is_some()),
        )?;
    }
    if let Some(deleted) = optional_child(root, "deleted-runs")? {
        for project in deleted
            .entry_names_limited(257)
            .map_err(|_| Error::Evidence)?
        {
            let project = project.to_str().ok_or(Error::Evidence)?;
            ml_exp_core::identity::validate_id(project).map_err(|_| Error::Evidence)?;
            let directory = deleted
                .child(OsStr::new(project), false)
                .map_err(|_| Error::Evidence)?;
            for leaf in directory
                .entry_names_limited(4097)
                .map_err(|_| Error::Evidence)?
            {
                let leaf = leaf.to_str().ok_or(Error::Evidence)?;
                let run = leaf.strip_suffix(".json").ok_or(Error::Evidence)?;
                let (value, sha) = mapping(&directory, leaf)?.ok_or(Error::Evidence)?;
                if value
                    .get("project")
                    .is_some_and(|v| v.as_str() != Some(project))
                    || value.get("run_id").is_some_and(|v| v.as_str() != Some(run))
                {
                    return Err(Error::Evidence);
                }
                audit.push(
                    key(project, Some(run), None, "deletion-tombstone", None),
                    &json!({"raw_sha256":sha}),
                    1,
                )?;
            }
        }
    }
    if let Some(actions) = actions {
        for leaf in actions
            .entry_names_limited(100_001)
            .map_err(|_| Error::Evidence)?
        {
            let name = leaf.to_str().ok_or(Error::Evidence)?;
            if !name.starts_with("action-") {
                continue;
            }
            ml_exp_core::operations::validate_action_id(name).map_err(|_| Error::Evidence)?;
            let directory = actions.child(&leaf, false).map_err(|_| Error::Evidence)?;
            let (value, sha) = mapping(&directory, "plan.json")?.ok_or(Error::Evidence)?;
            if value["action_id"] != name {
                return Err(Error::Evidence);
            }
            audit.push(
                key("native-state", None, None, "action-plan", Some(name)),
                &json!({"raw_sha256":sha}),
                1,
            )?;
        }
    }
    Ok(())
}
/// Capture only existing metadata and database rows. No initializer, W&B SDK,
/// network, source-tree traversal, checkpoint payload or result download occurs.
/// Two complete observations must agree; concurrent source changes fail closed.
pub fn capture(settings: &read_config::Settings, action_root: Option<&Path>) -> Result<Snapshot> {
    let settings = settings.clone();
    let action_root = action_root.map(Path::to_path_buf);
    // Restriction is irreversible and thread-local. Keep the caller's normal
    // permissions and apply it before this thread opens any SQLite connection.
    // READ_ONLY SQLite alone can still try to create an empty WAL file.
    std::thread::Builder::new()
        .name("state-audit-read-only".into())
        .spawn(move || {
            isolation::restrict()?;
            capture_isolated(&settings, action_root.as_deref())
        })
        .map_err(|_| Error::Isolation)?
        .join()
        .map_err(|_| Error::Isolation)?
}
fn capture_isolated(
    settings: &read_config::Settings,
    action_root: Option<&Path>,
) -> Result<Snapshot> {
    let first = capture_once(settings, action_root)?;
    let after = capture_once(settings, action_root)?;
    if first.entries != after.entries || first.issues != after.issues {
        return Err(Error::Evidence);
    }
    Ok(first)
}
fn capture_once(settings: &read_config::Settings, action_root: Option<&Path>) -> Result<Snapshot> {
    settings.validate().map_err(|_| Error::Evidence)?;
    let root = Directory::open(&settings.registry_root, false).map_err(|_| Error::Evidence)?;
    root.verify_private_parent().map_err(|_| Error::Evidence)?;
    let fence = RunFence::new(&settings.registry_root);
    let guard = fence.shared_existing().map_err(|_| Error::Evidence)?;
    let store = experiments::Store::open(settings.bindings(), fence.clone())
        .map_err(|_| Error::Evidence)?;
    let mut audit = Audit {
        entries: Vec::new(),
        issues: Vec::new(),
    };
    let mut journals = Audit {
        entries: Vec::new(),
        issues: Vec::new(),
    };
    database::capture(
        &root,
        "tracking",
        "outbox.sqlite",
        "record-journal",
        &mut journals,
    )?;
    let actions = action_root
        .map(|path| Directory::open(path, false).map_err(|_| Error::Evidence))
        .transpose()?;
    if let Some(actions) = &actions {
        actions
            .verify_private_parent()
            .map_err(|_| Error::Evidence)?;
        database::capture_direct(actions, "actions.sqlite3", "action-journal", &mut journals)?;
    }
    catalog::capture(settings, &root, &mut audit)?;
    for project in &settings.projects {
        let doc = store
            .project(&project.project, None)
            .map_err(|_| Error::Evidence)?;
        audit.push(
            key(&project.project, None, None, "project", None),
            &json!({"raw_sha256":doc.raw_sha256}),
            1,
        )?;
        let discovery = if project.discovery_enabled() {
            Some(
                store
                    .discover(&project.project, &Default::default())
                    .map_err(|_| Error::Evidence)?,
            )
        } else {
            None
        };
        let references = match &discovery {
            Some(d) => d.runs.iter().map(|r| r.reference.clone()).collect(),
            None => settings
                .runs()
                .into_iter()
                .filter(|r| r.project == project.project)
                .collect::<Vec<_>>(),
        };
        if let Some(d) = &discovery {
            for deleted in &d.deleted {
                audit.push(
                    key(
                        &project.project,
                        Some(&deleted.reference.run_id),
                        None,
                        "deleted-run",
                        None,
                    ),
                    &json!({"raw_sha256":deleted.raw_sha256}),
                    1,
                )?;
            }
        }
        for reference in references {
            let first = store
                .snapshot_metadata(&reference)
                .map_err(|_| Error::Evidence)?;
            audit.push(key(&first.project, Some(&first.run_id), None, "run", None), &json!({"raw_sha256":first.run.raw_sha256,"legacy_observation_only":first.run.legacy_observation_only,"current_attempt":first.current_attempt_id}), 1)?;
            for (name, value) in [
                ("status", &first.current_status),
                ("collection", &first.current_collection),
                ("decision", &first.current_decision),
            ] {
                audit.evidence(
                    key(&first.project, Some(&first.run_id), None, name, None),
                    value,
                )?;
            }
            for attempt in &first.attempts {
                let k = key(
                    &first.project,
                    Some(&first.run_id),
                    Some(&attempt.attempt_id),
                    "attempt",
                    None,
                );
                audit.push(k.clone(), &json!({"raw_sha256":attempt.manifest.raw_sha256,"provenance":attempt.provenance,"legacy_observation_only":attempt.manifest.legacy_observation_only}), 1)?;
                // Historical absence is part of the fingerprint, not damage
                // introduced by migration. It remains observation-only and
                // cannot establish native submission authority.
                for (name, value) in [
                    ("status", &attempt.status),
                    ("backend", &attempt.backend),
                    ("collection", &attempt.collection),
                    ("decision", &attempt.decision),
                    ("submission", &attempt.submission),
                ] {
                    audit.evidence(
                        key(
                            &first.project,
                            Some(&first.run_id),
                            Some(&attempt.attempt_id),
                            name,
                            None,
                        ),
                        value,
                    )?;
                }
                let transfer = transfer(
                    &root,
                    &first.project,
                    &first.run_id,
                    &attempt.attempt_id,
                    &mut audit,
                )?;
                checkpoints(
                    &root,
                    &first,
                    &attempt.attempt_id,
                    transfer.as_ref(),
                    &mut audit,
                )?;
            }
            journal(settings, &root, &first, &mut audit)?;
            let after = store
                .snapshot_metadata(&reference)
                .map_err(|_| Error::Evidence)?;
            if serde_json::to_value(&first).map_err(|_| Error::Evidence)?
                != serde_json::to_value(&after).map_err(|_| Error::Evidence)?
            {
                return Err(Error::Evidence);
            }
        }
        if let Some(before) = discovery {
            let after = store
                .discover(&project.project, &Default::default())
                .map_err(|_| Error::Evidence)?;
            if !before.same_bindings(&after) {
                return Err(Error::Evidence);
            }
        }
        store
            .project(&project.project, Some(&doc.raw_sha256))
            .map_err(|_| Error::Evidence)?;
    }
    let mut after = Audit {
        entries: Vec::new(),
        issues: Vec::new(),
    };
    database::capture(
        &root,
        "tracking",
        "outbox.sqlite",
        "record-journal",
        &mut after,
    )?;
    if let Some(actions) = &actions {
        database::capture_direct(actions, "actions.sqlite3", "action-journal", &mut after)?;
    }
    if journals.entries != after.entries {
        return Err(Error::Evidence);
    }
    audit.entries.extend(journals.entries);
    controls(&root, actions.as_ref(), &mut audit)?;
    guard.validate().map_err(|_| Error::Evidence)?;
    root.verify_private_parent().map_err(|_| Error::Evidence)?;
    Snapshot::new(audit.entries, audit.issues).map_err(|_| Error::Evidence)
}
