use super::{Database, Error, Result};
use fs2::FileExt;
use ml_exp_core::{
    canonical::python_json_bytes,
    operations::{
        self, Claim, Dispatch, Evidence, FrozenFiles, Policy, Reconcile, Scope, Transition,
    },
};
use ml_exp_io::{create_private_dir, write_atomic};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug)]
pub enum CommandPhase {
    Authorize,
    Execute,
}
impl CommandPhase {
    fn name(self) -> &'static str {
        match self {
            Self::Authorize => "authorize",
            Self::Execute => "execute",
        }
    }
}
#[derive(Debug)]
pub struct Claimed {
    pub snapshot: Value,
    pub dispatch: Option<Dispatch>,
}

#[derive(Clone, Copy, Debug)]
pub struct ClaimContext<'a> {
    pub now: &'a str,
    pub transition_id: &'a str,
    pub pid: u32,
}

struct Lock(File);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

/// The filesystem lock covers immutable plan/command writes and their database
/// publication. SQLite itself owns execution CAS and state+audit atomicity.
#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
    database: Database,
}
impl Store {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        create_private_dir(&root)?;
        if fs::symlink_metadata(&root)?.file_type().is_symlink() {
            return Err(Error::Invalid("Action root must not be a symlink"));
        }
        Ok(Self {
            database: Database::new(&root),
            root,
        })
    }
    pub fn database(&self) -> &Database {
        &self.database
    }
    pub fn directory(&self, action_id: &str) -> Result<PathBuf> {
        operations::validate_action_id(action_id)?;
        let path = self.root.join(action_id);
        if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(Error::Invalid("Action directory must not be a symlink"));
        }
        Ok(path)
    }
    fn lock(&self) -> Result<Lock> {
        let path = self.root.join(".actions.lock");
        if fs::symlink_metadata(&path).is_ok_and(|m| !m.is_file()) {
            return Err(Error::Invalid("Action lock must be a regular file"));
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = options.open(path)?;
        if !file.metadata()?.is_file() {
            return Err(Error::Invalid("Action lock must be a regular file"));
        }
        file.lock_exclusive()?;
        Ok(Lock(file))
    }
    fn read_json(path: &Path) -> Result<Option<Value>> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_file() || metadata.len() > 64 * 1024 * 1024 {
            return Err(Error::Invalid("bounded regular JSON file required"));
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = options.open(path)?;
        if !file.metadata()?.is_file() {
            return Err(Error::Invalid("JSON file must be regular"));
        }
        let mut bytes = Vec::new();
        file.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(Error::Invalid("JSON file too large"));
        }
        let value: Value = serde_json::from_slice(&bytes)?;
        if !value.is_object() {
            return Err(Error::Invalid("JSON object required"));
        }
        Ok(Some(value))
    }
    fn plan(&self, action_id: &str) -> Result<Value> {
        let plan = Self::read_json(&self.directory(action_id)?.join("plan.json"))?
            .ok_or(Error::NotFound)?;
        if plan.get("action_id").and_then(Value::as_str) != Some(action_id) {
            return Err(Error::Invalid("plan Action identity differs"));
        }
        Ok(plan)
    }
    fn execution(
        &self,
        action_id: &str,
        plan: &Value,
        now: &str,
        transition_id: &str,
    ) -> Result<Value> {
        let execution = self.database.read(action_id)?;
        if execution.as_object().is_some_and(|v| !v.is_empty()) {
            return Ok(execution);
        }
        self.database.commit(action_id,&operations::initial_execution(plan),&json!({"timestamp":now,"event":"action_prepared","payload":{"ready":plan.get("ready"),"operation":plan.get("operation")}}),None,transition_id)
    }
    fn snapshot_unlocked(&self, action_id: &str, now: &str, transition_id: &str) -> Result<Value> {
        let plan = self.plan(action_id)?;
        let execution = self.execution(action_id, &plan, now, transition_id)?;
        let journal = self.database.journal(action_id, 100)?;
        let mut snapshot = plan.as_object().cloned().ok_or(Error::Invalid("plan"))?;
        snapshot.insert("execution".into(), execution);
        snapshot.insert("journal".into(), json!(journal));
        Ok(Value::Object(snapshot))
    }
    pub fn snapshot(&self, action_id: &str, now: &str, transition_id: &str) -> Result<Value> {
        let _lock = self.lock()?;
        self.snapshot_unlocked(action_id, now, transition_id)
    }
    pub fn save_plan(&self, plan: &Value, now: &str, transition_id: &str) -> Result<Value> {
        let _lock = self.lock()?;
        let action_id = plan
            .get("action_id")
            .and_then(Value::as_str)
            .ok_or(Error::Invalid("action_id"))?;
        let directory = self.directory(action_id)?;
        create_private_dir(&directory)?;
        let path = directory.join("plan.json");
        if let Some(existing) = Self::read_json(&path)? {
            if existing.get("request_digest") != plan.get("request_digest") {
                return Err(Error::IdempotencyConflict);
            }
        } else {
            write_atomic(&path, &serde_json::to_vec_pretty(plan)?, true)?;
        }
        self.snapshot_unlocked(action_id, now, transition_id)
    }
    fn apply(&self, action_id: &str, transition: &Transition, transition_id: &str) -> Result<()> {
        self.database.commit(
            action_id,
            &transition.execution,
            &transition.event,
            transition.expected_status.as_deref(),
            transition_id,
        )?;
        Ok(())
    }
    pub fn authorize(
        &self,
        action_id: &str,
        actor: &str,
        note: &str,
        now: &str,
        transition_id: &str,
    ) -> Result<Value> {
        let _lock = self.lock()?;
        // A crash may have published plan.json before its initial database
        // row. Bootstrap and authorization are distinct audit transitions.
        let bootstrap = format!(
            "{:x}",
            Sha256::digest(format!("bootstrap:{action_id}:{transition_id}"))
        );
        let snapshot = self.snapshot_unlocked(action_id, now, &bootstrap[..32])?;
        let transition =
            operations::authorize(&snapshot, &snapshot["execution"], actor, note, now)?;
        self.apply(action_id, &transition, transition_id)?;
        self.snapshot_unlocked(action_id, now, transition_id)
    }
    /// Persist intent before returning dispatch information. The caller must
    /// enqueue at most once; existing EXECUTING/unknown effects never replay.
    pub fn claim(
        &self,
        action_id: &str,
        confirmation: &str,
        policy: &Policy,
        files: &FrozenFiles,
        context: ClaimContext<'_>,
    ) -> Result<Claimed> {
        let ClaimContext {
            now,
            transition_id,
            pid,
        } = context;
        let _lock = self.lock()?;
        let snapshot = self.snapshot_unlocked(action_id, now, transition_id)?;
        match operations::claim(
            &snapshot,
            &snapshot["execution"],
            confirmation,
            policy,
            files,
            now,
        )? {
            Claim::AlreadyVerified => Ok(Claimed {
                snapshot,
                dispatch: None,
            }),
            Claim::Execute {
                transition,
                dispatch,
            } => {
                self.apply(action_id, &transition, transition_id)?;
                // Auxiliary evidence only: failure here cannot undo or replay
                // the authoritative SQLite execution claim and audit event.
                let claim =
                    json!({"claimed_at":now,"pid":pid,"intent_digest":snapshot["intent_digest"]});
                let _ = write_atomic(
                    &self.directory(action_id)?.join("execution.claim"),
                    &serde_json::to_vec_pretty(&claim)?,
                    false,
                );
                Ok(Claimed {
                    snapshot: self.snapshot_unlocked(action_id, now, transition_id)?,
                    dispatch: Some(dispatch),
                })
            }
        }
    }
    /// Commit an executor's already planned result with revision CAS. This does
    /// not dispatch or make an uncertain submission safe for automatic retry.
    pub fn commit(
        &self,
        action_id: &str,
        transition: &Transition,
        transition_id: &str,
        now: &str,
    ) -> Result<Value> {
        let _lock = self.lock()?;
        self.apply(action_id, transition, transition_id)?;
        self.snapshot_unlocked(action_id, now, transition_id)
    }
    pub fn reconcile(
        &self,
        action_id: &str,
        policy: &Policy,
        evidence: &Evidence,
        now: &str,
        transition_id: &str,
    ) -> Result<(Value, Option<Dispatch>)> {
        let _lock = self.lock()?;
        let snapshot = self.snapshot_unlocked(action_id, now, transition_id)?;
        match operations::reconcile(&snapshot, &snapshot["execution"], policy, evidence, now)? {
            Reconcile::Unchanged => Ok((snapshot, None)),
            Reconcile::Commit(transition) => {
                self.apply(action_id, &transition, transition_id)?;
                Ok((self.snapshot_unlocked(action_id, now, transition_id)?, None))
            }
            Reconcile::RecoverTransaction(dispatch) => Ok((snapshot, Some(dispatch))),
        }
    }
    fn list_unlocked(
        &self,
        scope: Option<&Scope>,
        now: &str,
        transition_id: &str,
    ) -> Result<Vec<Value>> {
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let Some(id) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if !id.starts_with("action-") {
                continue;
            }
            if operations::validate_action_id(&id).is_err() {
                continue;
            }
            ids.push(id);
        }
        ids.sort();
        let expected = scope.map(serde_json::to_value).transpose()?;
        let mut items = Vec::new();
        for id in ids {
            let plan = match self.plan(&id) {
                Ok(plan) => plan,
                Err(Error::NotFound) => continue,
                Err(error) => return Err(error),
            };
            if expected
                .as_ref()
                .is_some_and(|expected| plan.get("scope") != Some(expected))
            {
                continue;
            }
            items.push(self.snapshot_unlocked(&id, now, transition_id)?);
        }
        Ok(items)
    }
    pub fn list(
        &self,
        scope: Option<&Scope>,
        now: &str,
        transition_id: &str,
    ) -> Result<Vec<Value>> {
        let _lock = self.lock()?;
        self.list_unlocked(scope, now, transition_id)
    }
    /// Startup must run before the daemon serves or resumes any queue worker.
    /// Merely records abandoned claims; it never issues provider submit calls.
    pub fn recover_interrupted(&self, now: &str, transition_id: &str) -> Result<Vec<Value>> {
        let _lock = self.lock()?;
        let mut recovered = Vec::new();
        for snapshot in self.list_unlocked(None, now, transition_id)? {
            if let Some(transition) = operations::interrupted(&snapshot["execution"], now)? {
                let id = snapshot["action_id"]
                    .as_str()
                    .ok_or(Error::Invalid("action_id"))?;
                self.apply(id, &transition, transition_id)?;
                recovered.push(self.snapshot_unlocked(id, now, transition_id)?);
            }
        }
        Ok(recovered)
    }
    pub fn append_journal(
        &self,
        action_id: &str,
        event: &str,
        payload: &Value,
        event_id: &str,
        now: &str,
    ) -> Result<()> {
        let _lock = self.lock()?;
        self.database.append_event(
            action_id,
            &json!({"timestamp":now,"event":event,"payload":payload}),
            event_id,
        )
    }
    /// Sensitive notes and confirmation commands stay outside immutable plans.
    pub fn write_command(
        &self,
        action_id: &str,
        phase: CommandPhase,
        payload: &Value,
        event_id: &str,
        now: &str,
    ) -> Result<Value> {
        if !payload.is_object() {
            return Err(Error::Invalid("command payload"));
        }
        let _lock = self.lock()?;
        let directory = self.directory(action_id)?.join("commands");
        create_private_dir(&directory)?;
        let path = directory.join(format!("{}.json", phase.name()));
        let record = if let Some(existing) = Self::read_json(&path)? {
            if existing.get("payload") != Some(payload) {
                return Err(Error::CommandConflict);
            }
            existing
        } else {
            let record = json!({"action_id":action_id,"phase":phase.name(),"payload":payload,"created_at":now,"journal_event_id":event_id});
            write_atomic(&path, &serde_json::to_vec_pretty(&record)?, true)?;
            record
        };
        let event_id = match record
            .get("journal_event_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            Some(id) => id.to_owned(),
            None => format!(
                "{:x}",
                Sha256::digest(
                    python_json_bytes(&record)
                        .map_err(|_| Error::Invalid("legacy command encoding"))?
                )
            ),
        };
        self.database.append_event(action_id,&json!({"timestamp":now,"event":format!("{}_command_recorded",phase.name()),"payload":{}}),&event_id)?;
        Ok(record)
    }
    pub fn read_command(&self, action_id: &str, phase: CommandPhase) -> Result<Value> {
        let _lock = self.lock()?;
        let record = Self::read_json(
            &self
                .directory(action_id)?
                .join("commands")
                .join(format!("{}.json", phase.name())),
        )?
        .ok_or(Error::NotFound)?;
        Ok(record
            .get("payload")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(|| json!({})))
    }
    pub fn write_activity_error(
        &self,
        action_id: &str,
        phase: &str,
        message: &str,
        category: &str,
        now: &str,
    ) -> Result<()> {
        if !matches!(phase, "prepare" | "authorize" | "execute" | "reconcile") {
            return Err(Error::Invalid("activity phase"));
        }
        let _lock = self.lock()?;
        let directory = self.directory(action_id)?.join("activity_errors");
        create_private_dir(&directory)?;
        let message: String = message.chars().take(1000).collect();
        write_atomic(
            &directory.join(format!("{phase}.json")),
            &serde_json::to_vec_pretty(
                &json!({"action_id":action_id,"phase":phase,"message":message,"category":category,"created_at":now}),
            )?,
            false,
        )?;
        Ok(())
    }
    pub fn activity_error(&self, action_id: &str, phase: &str) -> Result<Value> {
        if !matches!(phase, "prepare" | "authorize" | "execute" | "reconcile") {
            return Err(Error::Invalid("activity phase"));
        }
        let _lock = self.lock()?;
        Ok(Self::read_json(
            &self
                .directory(action_id)?
                .join("activity_errors")
                .join(format!("{phase}.json")),
        )?
        .unwrap_or_else(|| json!({})))
    }
}
