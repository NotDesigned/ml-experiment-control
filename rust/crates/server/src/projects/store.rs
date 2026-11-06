use ml_exp_core::{
    canonical, durable,
    projects::{self, LifecycleRecord, PathEvidence, RegistrationSource, Registry, Seed},
};
use ml_exp_io::directory::{Directory, FileLock, LockOptions, Wait};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const HEAD: &str = "registry.json";
const EVENTS: &str = "events.jsonl";
const LOCK: &str = ".registry.lock";

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
/// Proof that the caller holds `.registry.lock`.
type Lock = FileLock;

#[derive(Clone)]
pub struct Store {
    directory: Directory,
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
    ml_exp_core::json::own_mapping(bytes)
        .map(Value::Object)
        .map_err(why!(Error::Invalid))
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
    let mut bytes = canonical::bytes(value).map_err(why!(Error::Invalid))?;
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
            durable::commit(
                &empty,
                &[],
                &commit.payload,
                &commit.event,
                Some(0),
                transition_id,
            )?;
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
        let store = Self { directory };
        {
            let lock = store.lock(false)?;
            store.load(&lock)?;
        }
        Ok(store)
    }
    fn lock(&self, exclusive: bool) -> Result<Lock, Error> {
        self.directory
            .lock(
                OsStr::new(LOCK),
                LockOptions {
                    shared: !exclusive,
                    wait: Wait::Until {
                        deadline: Instant::now() + Duration::from_secs(5),
                        cancelled: None,
                    },
                    ..Default::default()
                },
            )
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::WouldBlock {
                    Error::Busy
                } else {
                    Error::from(e)
                }
            })
    }
    fn load(&self, _lock: &Lock) -> Result<Loaded, Error> {
        let head = self.directory.read_whole(OsStr::new(HEAD))?;
        let state = durable::Snapshot::from_value(&decode(&head)?)?;
        let registry = registry(&state.value)?;
        let (events, normalized_journal) =
            journal(&self.directory.read_whole(OsStr::new(EVENTS))?)?;
        let missing = durable::missing_transition(&state, &events)?;
        if self.directory.read_whole(OsStr::new(HEAD))? != head {
            return Err(Error::Invalid);
        }
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
        Ok(result)
    }
    fn repair(&self, lock: &Lock, loaded: Loaded) -> Result<Loaded, Error> {
        if loaded.normalized_journal.is_none() && loaded.missing.is_none() {
            return Ok(loaded);
        }
        let result = (|| {
            if let Some(bytes) = loaded.normalized_journal {
                self.directory.replace_regular(OsStr::new(EVENTS), &bytes)?;
            }
            if let Some(transition) = loaded.missing {
                self.directory
                    .append_regular(OsStr::new(EVENTS), &encoded(&transition)?)?;
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
        lock: &Lock,
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
        (|| {
            let repaired = self.repair(lock, loaded)?;
            if repaired.state.revision + 1 != plan.revision {
                return Err(Error::Invalid);
            }
            self.directory.replace_regular(OsStr::new(HEAD), &bytes)?;
            if self.directory.read_whole(OsStr::new(HEAD))? != bytes {
                return Err(Error::Invalid);
            }
            self.directory.append_regular(OsStr::new(EVENTS), &event)?;
            let final_state = self.load(lock)?;
            if final_state.state.revision != plan.revision || final_state.missing.is_some() {
                return Err(Error::Invalid);
            }
            Ok(())
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
        if let Some(change) = change {
            self.persist(&lock, loaded, change, expected_revision, identity)?;
        } else {
            self.repair(&lock, loaded)?;
        }
        Ok(result)
    }
}
fn converted<T: Serialize>(
    change: projects::Change<T>,
) -> Result<(Value, Option<projects::Commit>), Error> {
    Ok((
        serde_json::to_value(change.result).map_err(why!(Error::Invalid))?,
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
