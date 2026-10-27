use crate::experiments::{Format, parse_mapping};
use fs2::FileExt;
use ml_exp_core::{
    canonical::python_json_bytes,
    durable,
    projects::{self, LifecycleRecord, PathEvidence, RegistrationSource, Registry, Seed},
};
use ml_exp_io::directory::Directory;
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

const HEAD: &str = "registry.json";
const EVENTS: &str = "events.jsonl";
const LOCK: &str = ".registry.lock";
const MAX_HEAD: u64 = 4 * 1024 * 1024;
const MAX_JOURNAL: u64 = 8 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("project storage I/O failed")]
    Io(#[from] std::io::Error),
    #[error("project registry or journal is invalid")]
    Invalid,
    #[error("project registry source binding differs from approved configuration")]
    Binding,
    #[error(
        "native Project registration requires empty campaigns; Campaign loading is not implemented"
    )]
    UnsupportedDefinition,
    #[error("project registry transition is unavailable")]
    Transition(#[from] projects::Error),
    #[error("project registry is busy")]
    Busy,
    #[error("project registry may have changed; inspect its durable state before retry")]
    EffectUncertain,
}
impl From<durable::Error> for Error {
    fn from(_: durable::Error) -> Self {
        Self::Invalid
    }
}

pub enum Operation {
    Register {
        project: String,
        path: PathEvidence,
    },
    Action {
        project: String,
        action: String,
        reason: String,
    },
    Unregister {
        project: String,
        reason: String,
    },
    UnregisterAll {
        reason: String,
    },
}
#[derive(Serialize)]
pub struct Snapshot {
    pub projects: Vec<LifecycleRecord>,
    pub events: Vec<Value>,
    pub revision: u64,
    pub journal_pending: bool,
}
struct Loaded {
    state: durable::Snapshot,
    registry: Registry,
    events: Vec<Value>,
    normalized_journal: Option<Vec<u8>>,
    missing: Option<Value>,
}
struct Lock<'a> {
    store: &'a Store,
    file: File,
}
impl Lock<'_> {
    fn validate(&self) -> Result<(), Error> {
        self.store.directory.verify_private_parent()?;
        self.store
            .directory
            .validate_regular_entry(OsStr::new(LOCK), &self.store.anchor)?;
        self.store
            .directory
            .validate_regular_entry(OsStr::new(LOCK), &self.file)?;
        Ok(())
    }
}
impl Drop for Lock<'_> {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

#[derive(Clone)]
pub struct Store {
    directory: Directory,
    anchor: Arc<File>,
}

/// Stored native paths were already resolved by trusted configuration. Reading
/// the ledger never expands a home or follows an authored path. Legacy aliases
/// need an operator-provided path resolver before importing that namespace.
pub fn canonical_path(path: &str) -> Result<PathEvidence, Error> {
    let p = Path::new(path);
    if !p.is_absolute()
        || p.to_str().is_none()
        || p.components().any(|c| {
            !matches!(
                c,
                std::path::Component::RootDir | std::path::Component::Normal(_)
            )
        })
    {
        return Err(Error::Invalid);
    }
    Ok(PathEvidence {
        provided: path.into(),
        expanded_absolute: path.into(),
        canonical_absolute: path.into(),
    })
}
fn decode(bytes: &[u8]) -> Result<Value, Error> {
    parse_mapping(bytes, Format::Json)
        .map(Value::Object)
        .map_err(|_| Error::Invalid)
}
fn registry(payload: &Value) -> Result<Registry, Error> {
    let paths = Registry::required_paths(payload)?
        .iter()
        .map(|path| canonical_path(path))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Registry::from_value(payload, &paths)?)
}
/// Parse complete framing without repairing it. Only a syntactically partial
/// final record is recoverable. Valid-but-ambiguous JSON is never discarded.
fn journal(bytes: &[u8]) -> Result<(Vec<Value>, Option<Vec<u8>>), Error> {
    let end = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
    let mut events = Vec::new();
    if end > 0 {
        for line in bytes[..end - 1].split(|b| *b == b'\n') {
            events.push(decode(line)?);
        }
    }
    let tail = &bytes[end..];
    if tail.is_empty() {
        return Ok((events, None));
    }
    match decode(tail) {
        Ok(value) => {
            events.push(value);
            let mut normalized = bytes.to_vec();
            normalized.push(b'\n');
            Ok((events, Some(normalized)))
        }
        Err(_) => match serde_json::from_slice::<Value>(tail) {
            Err(error)
                if error.is_eof()
                    && std::str::from_utf8(tail)
                        .err()
                        .is_none_or(|error| error.error_len().is_none()) =>
            {
                Ok((events, Some(bytes[..end].to_vec())))
            }
            _ => Err(Error::Invalid),
        },
    }
}
fn encoded(value: &Value) -> Result<Vec<u8>, Error> {
    let mut bytes = python_json_bytes(value).map_err(|_| Error::Invalid)?;
    bytes.push(b'\n');
    Ok(bytes)
}

impl Store {
    pub fn initialize(
        path: &Path,
        seeds: &[Seed],
        now: &str,
        transition_id: &str,
    ) -> Result<Self, Error> {
        let change = projects::bootstrap(None, seeds, now)?;
        if let Some(commit) = &change.commit {
            let empty = durable::Snapshot::from_value(&json!({"schema_version":1,"projects":[]}))?;
            let plan = durable::commit(
                &empty,
                &[],
                &commit.payload,
                &commit.event,
                Some(0),
                transition_id,
            )?;
            if encoded(&plan.head)?.len() as u64 > MAX_HEAD
                || encoded(&plan.transition)?.len() as u64 > MAX_JOURNAL
            {
                return Err(Error::Invalid);
            }
        }
        let parent = Directory::open(path.parent().ok_or(Error::Invalid)?, false)?;
        let name = path.file_name().ok_or(Error::Invalid)?;
        if parent.entry_exists(name)? {
            return Err(Error::Invalid);
        }
        let directory = parent.create_new_child(name).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                Error::Io(error)
            } else {
                Error::EffectUncertain
            }
        })?;
        (|| {
            directory.verify_private_parent()?;
            directory.publish_new(OsStr::new(LOCK), b"")?;
            directory.publish_new(OsStr::new(EVENTS), b"")?;
            directory.publish_new(
                OsStr::new(HEAD),
                &encoded(&json!({"schema_version":1,"projects":[]}))?,
            )?;
            let store = Self::open(path)?;
            let lock = store.lock(true)?;
            let loaded = store.load(&lock)?;
            if let Some(commit) = change.commit {
                store.persist(&lock, loaded, commit, Some(0), transition_id)?;
            }
            drop(lock);
            Ok(store)
        })()
        .map_err(|_: Error| Error::EffectUncertain)
    }
    pub fn open(path: &Path) -> Result<Self, Error> {
        let directory = Directory::open(path, false)?;
        directory.verify_private_parent()?;
        let anchor = Arc::new(directory.open_existing_lock(OsStr::new(LOCK))?);
        let store = Self { directory, anchor };
        {
            let lock = store.lock(false)?;
            store.load(&lock)?;
        }
        Ok(store)
    }
    fn lock(&self, exclusive: bool) -> Result<Lock<'_>, Error> {
        let file = self.directory.open_existing_lock(OsStr::new(LOCK))?;
        let end = Instant::now() + Duration::from_secs(5);
        loop {
            let result = if exclusive {
                FileExt::try_lock_exclusive(&file)
            } else {
                FileExt::try_lock_shared(&file)
            };
            match result {
                Ok(()) => break,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < end => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Err(Error::Busy),
                Err(e) => return Err(e.into()),
            }
        }
        let lock = Lock { store: self, file };
        lock.validate()?;
        Ok(lock)
    }
    fn load(&self, lock: &Lock<'_>) -> Result<Loaded, Error> {
        lock.validate()?;
        let head = self.directory.read_regular(OsStr::new(HEAD), MAX_HEAD)?;
        let state = durable::Snapshot::from_value(&decode(&head)?)?;
        let registry = registry(&state.value)?;
        let (events, normalized_journal) = journal(
            &self
                .directory
                .read_regular(OsStr::new(EVENTS), MAX_JOURNAL)?,
        )?;
        let missing = durable::missing_transition(&state, &events)?;
        if self.directory.read_regular(OsStr::new(HEAD), MAX_HEAD)? != head {
            return Err(Error::Invalid);
        }
        lock.validate()?;
        Ok(Loaded {
            state,
            registry,
            events,
            normalized_journal,
            missing,
        })
    }
    pub fn snapshot(&self, limit: usize) -> Result<Snapshot, Error> {
        let lock = self.lock(false)?;
        let loaded = self.load(&lock)?;
        let start = loaded.events.len().saturating_sub(limit.min(1000));
        let result = Snapshot {
            projects: loaded.registry.records().to_vec(),
            events: loaded.events[start..].to_vec(),
            revision: loaded.state.revision,
            journal_pending: loaded.missing.is_some() || loaded.normalized_journal.is_some(),
        };
        lock.validate()?;
        Ok(result)
    }
    fn repair(&self, lock: &Lock<'_>, loaded: Loaded) -> Result<Loaded, Error> {
        if loaded.normalized_journal.is_none() && loaded.missing.is_none() {
            return Ok(loaded);
        }
        lock.validate()?;
        let result = (|| {
            if let Some(bytes) = loaded.normalized_journal {
                self.directory.replace_regular(OsStr::new(EVENTS), &bytes)?;
            }
            if let Some(transition) = loaded.missing {
                self.directory.append_regular(
                    OsStr::new(EVENTS),
                    &encoded(&transition)?,
                    MAX_JOURNAL,
                )?;
            }
            let repaired = self.load(lock)?;
            if repaired.normalized_journal.is_some() || repaired.missing.is_some() {
                return Err(Error::Invalid);
            }
            Ok(repaired)
        })();
        result.map_err(|_: Error| Error::EffectUncertain)
    }
    fn persist(
        &self,
        lock: &Lock<'_>,
        loaded: Loaded,
        change: projects::Commit,
        expected: Option<u64>,
        identity: &str,
    ) -> Result<(), Error> {
        // Compute and bound the entire proposal before any journal repair.
        let mut reconciled = loaded.events.clone();
        if let Some(missing) = &loaded.missing {
            reconciled.push(missing.clone());
        }
        let plan = durable::commit(
            &loaded.state,
            &reconciled,
            &change.payload,
            &change.event,
            expected,
            identity,
        )?;
        let bytes = encoded(&plan.head)?;
        let event = encoded(&plan.transition)?;
        let mut journal_size = match &loaded.normalized_journal {
            Some(bytes) => bytes.len() as u64,
            None => self
                .directory
                .open_regular_read(OsStr::new(EVENTS))?
                .metadata()?
                .len(),
        };
        if let Some(missing) = &loaded.missing {
            journal_size = journal_size
                .checked_add(encoded(missing)?.len() as u64)
                .ok_or(Error::Invalid)?;
        }
        if bytes.len() as u64 > MAX_HEAD
            || journal_size
                .checked_add(event.len() as u64)
                .is_none_or(|n| n > MAX_JOURNAL)
        {
            return Err(Error::Invalid);
        }
        lock.validate()?;
        (|| {
            let repaired = self.repair(lock, loaded)?;
            if repaired.state.revision + 1 != plan.revision {
                return Err(Error::Invalid);
            }
            self.directory.replace_regular(OsStr::new(HEAD), &bytes)?;
            if self.directory.read_regular(OsStr::new(HEAD), MAX_HEAD)? != bytes {
                return Err(Error::Invalid);
            }
            self.directory
                .append_regular(OsStr::new(EVENTS), &event, MAX_JOURNAL)?;
            let final_state = self.load(lock)?;
            if final_state.state.revision != plan.revision || final_state.missing.is_some() {
                return Err(Error::Invalid);
            }
            lock.validate()
        })()
        .map_err(|_: Error| Error::EffectUncertain)
    }
    pub fn execute(
        &self,
        operation: &Operation,
        now: &str,
        identity: &str,
        expected_revision: Option<u64>,
        approved: Option<&BTreeMap<String, PathBuf>>,
    ) -> Result<Value, Error> {
        let lock = self.lock(true)?;
        let loaded = self.load(&lock)?;
        if let Some(approved) = approved {
            for record in loaded.registry.records() {
                if approved
                    .get(&record.project)
                    .is_some_and(|path| path.to_str() != Some(record.project_file.as_str()))
                {
                    return Err(Error::Binding);
                }
            }
        }
        if expected_revision.is_some_and(|revision| revision != loaded.state.revision) {
            return Err(Error::Invalid);
        }
        let (result, change) = plan(&loaded.registry, operation, now)?;
        let effects =
            change.is_some() || loaded.missing.is_some() || loaded.normalized_journal.is_some();
        if let Some(change) = change {
            self.persist(&lock, loaded, change, expected_revision, identity)?;
        } else {
            self.repair(&lock, loaded)?;
        }
        lock.validate()
            .map_err(|e| if effects { Error::EffectUncertain } else { e })?;
        Ok(result)
    }
}
fn converted<T: Serialize>(
    change: projects::Change<T>,
) -> Result<(Value, Option<projects::Commit>), Error> {
    Ok((
        serde_json::to_value(change.result).map_err(|_| Error::Invalid)?,
        change.commit,
    ))
}
fn plan(
    registry: &Registry,
    operation: &Operation,
    now: &str,
) -> Result<(Value, Option<projects::Commit>), Error> {
    match operation {
        Operation::Register { project, path } => {
            converted(registry.register(project, path, RegistrationSource::Manual, now)?)
        }
        Operation::Action {
            project,
            action,
            reason,
        } => converted(registry.action(project, action, reason, now)?),
        Operation::Unregister { project, reason } => {
            converted(registry.unregister(project, reason, now)?)
        }
        Operation::UnregisterAll { reason } => converted(registry.unregister_all(reason, now)?),
    }
}
