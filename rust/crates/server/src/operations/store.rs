use super::{Database, Error, Result};
use ml_exp_core::{
    canonical::python_json_bytes,
    operations::{
        self, Claim, Dispatch, Evidence, FrozenFiles, Policy, Reconcile, Scope, Transition,
    },
};
use ml_exp_io::directory::{Directory, FileLock, LockOptions, Wait, effective_uid};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{ffi::OsStr, path::PathBuf};

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

/// Proof that the caller holds `.actions.lock`.
type Lock = FileLock;

/// The filesystem lock covers immutable plan/command writes and their database
/// publication. SQLite itself owns execution CAS and state+audit atomicity.
#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
    directory: Directory,
    database: Database,
}
impl Store {
    /// Existing storage only. Unlike `open`, this never bootstraps a lock or
    /// creates the root. Reads also use `observe`, not snapshot recovery.
    pub fn open_existing(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let directory = Directory::open(&root, false)?;
        directory.open_regular_read(OsStr::new(".actions.lock"))?;
        Ok(Self {
            database: Database::in_directory(directory.clone())?,
            root,
            directory,
        })
    }
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let directory = Directory::open(&root, true)?;
        // Only a completely fresh dedicated root may bootstrap its lock.
        // Existing storage without it requires explicit offline recovery;
        // ordinary reopen/read must not create a second lock domain.
        let create = directory.entry_names()?.is_empty();
        drop(directory.lock(
            OsStr::new(".actions.lock"),
            LockOptions {
                create,
                wait: Wait::Block,
                ..Default::default()
            },
        )?);
        Ok(Self {
            database: Database::in_directory(directory.clone())?,
            root,
            directory,
        })
    }
    pub fn database(&self) -> &Database {
        &self.database
    }
    /// Display/test provenance only; repository I/O goes through `Directory`.
    pub fn directory(&self, action_id: &str) -> Result<PathBuf> {
        operations::validate_action_id(action_id)?;
        Ok(self.root.join(action_id))
    }
    fn action_directory(&self, action_id: &str, create: bool, _lock: &Lock) -> Result<Directory> {
        operations::validate_action_id(action_id)?;
        self.directory
            .child(OsStr::new(action_id), create)
            .map_err(|error| {
                if !create && error.kind() == std::io::ErrorKind::NotFound {
                    Error::NotFound
                } else {
                    error.into()
                }
            })
    }
    fn subdirectory(
        &self,
        action_id: &str,
        name: &str,
        create: bool,
        lock: &Lock,
    ) -> Result<Directory> {
        let action = self.action_directory(action_id, false, lock)?;
        Ok(action.child(OsStr::new(name), create)?)
    }
    fn lock(&self) -> Result<Lock> {
        Ok(self.directory.lock(
            OsStr::new(".actions.lock"),
            LockOptions {
                wait: Wait::Block,
                ..Default::default()
            },
        )?)
    }
    fn locked<T>(&self, operation: impl FnOnce(&Lock) -> Result<T>) -> Result<T> {
        let lock = self.lock()?;
        operation(&lock)
    }
    fn read_json(directory: &Directory, name: &OsStr) -> Result<Option<Value>> {
        match directory.regular_identity(name)? {
            None => return Ok(None),
            Some(identity) if identity.is_private_for(effective_uid()) => {}
            Some(_) => return Err(Error::Invalid("private JSON file required")),
        }
        let value: Value =
            serde_json::from_slice(&directory.read_regular(name, 64 * 1024 * 1024)?)?;
        if !value.is_object() {
            return Err(Error::Invalid("JSON object required"));
        }
        Ok(Some(value))
    }
    fn plan(&self, action_id: &str, lock: &Lock) -> Result<Value> {
        let plan = Self::read_json(
            &self.action_directory(action_id, false, lock)?,
            OsStr::new("plan.json"),
        )?
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
        _lock: &Lock,
    ) -> Result<Value> {
        let execution = self.database.read(action_id)?;
        if execution.as_object().is_some_and(|v| !v.is_empty()) {
            return Ok(execution);
        }
        self.database.commit(action_id,&operations::initial_execution(plan),&json!({"timestamp":now,"event":"action_prepared","payload":{"ready":plan.get("ready"),"operation":plan.get("operation")}}),None,transition_id)
    }
    fn snapshot_unlocked(
        &self,
        action_id: &str,
        now: &str,
        transition_id: &str,
        lock: &Lock,
    ) -> Result<Value> {
        let plan = self.plan(action_id, lock)?;
        let execution = self.execution(action_id, &plan, now, transition_id, lock)?;
        let journal = self.database.journal(action_id, 100)?;
        let mut snapshot = plan.as_object().cloned().ok_or(Error::Invalid("plan"))?;
        snapshot.insert("execution".into(), execution);
        snapshot.insert("journal".into(), json!(journal));
        Ok(Value::Object(snapshot))
    }
    pub fn snapshot(&self, action_id: &str, now: &str, transition_id: &str) -> Result<Value> {
        self.locked(|lock| self.snapshot_unlocked(action_id, now, transition_id, lock))
    }
    /// Immutable plan plus one committed execution revision. No journal repair,
    /// generated transition ID, clock read or provider effect accompanies this.
    pub fn observe(&self, action_id: &str) -> Result<Value> {
        self.locked(|lock| {
            let plan = self.plan(action_id, lock)?;
            if plan.get("action_id").and_then(Value::as_str) != Some(action_id) {
                return Err(Error::Invalid("plan Action identity mismatch"));
            }
            let execution = self.database.observe(action_id)?;
            let mut value = plan.as_object().cloned().ok_or(Error::Invalid("plan"))?;
            value.insert("execution".into(), execution);
            Ok(Value::Object(value))
        })
    }
    /// Private, immutable execution input; never part of public snapshots or
    /// journals. It must be durable before claiming a native scheduler Action.
    pub fn bind_slurm_request(
        &self,
        action_id: &str,
        request: &ml_exp_core::slurm::helper::Request,
    ) -> Result<()> {
        use ml_exp_core::slurm::{action, helper};
        let frame = helper::encode_request(request)
            .map_err(why!(Error::Invalid("native Slurm request")))?;
        self.locked(|lock| {
            let plan = self.plan(action_id, lock)?;
            action::validate(&plan, request)
                .map_err(why!(Error::Invalid("native Slurm Action binding")))?;
            let directory = self.action_directory(action_id, false, lock)?;
            let name = OsStr::new("native-slurm-request.frame");
            if directory.entry_exists(name)? {
                let stored = self.slurm_request_unlocked(action_id, lock)?;
                if &stored != request {
                    return Err(Error::IdempotencyConflict);
                }
            } else {
                let execution = self.database.observe(action_id)?;
                if !matches!(
                    execution["status"].as_str(),
                    Some("PREPARED" | "AUTHORIZED")
                ) {
                    return Err(Error::Invalid(
                        "missing native Slurm request requires offline investigation",
                    ));
                }
                directory.publish_new(name, &frame)?;
            }
            Ok(())
        })
    }
    fn slurm_request_unlocked(
        &self,
        action_id: &str,
        lock: &Lock,
    ) -> Result<ml_exp_core::slurm::helper::Request> {
        use ml_exp_core::slurm::{action, helper};
        let directory = self.action_directory(action_id, false, lock)?;
        let name = OsStr::new("native-slurm-request.frame");
        let identity = directory.regular_identity(name)?.ok_or(Error::NotFound)?;
        if !identity.is_private_for(effective_uid()) || identity.mode & 0o077 != 0 {
            return Err(Error::Invalid("private native Slurm request file required"));
        }
        let frame = directory.read_regular(name, helper::MAX_FRAME_BYTES as u64)?;
        let request = helper::decode_request(&frame)
            .map_err(why!(Error::Invalid("native Slurm request frame")))?;
        action::validate(&self.plan(action_id, lock)?, &request)
            .map_err(why!(Error::Invalid("native Slurm Action binding")))?;
        Ok(request)
    }
    /// Existing private input only. Missing/corrupt state is never reconstructed
    /// from a client request after execution may already have been claimed.
    pub fn slurm_request(&self, action_id: &str) -> Result<ml_exp_core::slurm::helper::Request> {
        self.locked(|lock| self.slurm_request_unlocked(action_id, lock))
    }
    /// The cloud adapter uses the same Action authorization/CAS as Slurm.
    /// Private request bytes are immutable and must exist before execution.
    pub fn bind_cloud_request(
        &self,
        action_id: &str,
        request: &ml_exp_core::cloud::action::Execution,
    ) -> Result<()> {
        use ml_exp_core::cloud::action;
        request
            .validate()
            .map_err(why!(Error::Invalid("native cloud request")))?;
        self.locked(|lock| {
            action::validate(&self.plan(action_id, lock)?, request)
                .map_err(why!(Error::Invalid("native cloud binding")))?;
            let directory = self.action_directory(action_id, false, lock)?;
            let name = OsStr::new("native-cloud-request.json");
            if directory.entry_exists(name)? {
                if self.cloud_request_unlocked(action_id, lock)? != *request {
                    return Err(Error::IdempotencyConflict);
                }
            } else {
                let execution = self.database.observe(action_id)?;
                if !matches!(
                    execution["status"].as_str(),
                    Some("PREPARED" | "AUTHORIZED")
                ) {
                    return Err(Error::Invalid(
                        "missing native cloud input requires offline investigation",
                    ));
                }
                directory.publish_new(name, &serde_json::to_vec(request)?)?;
            }
            Ok(())
        })
    }
    fn cloud_request_unlocked(
        &self,
        action_id: &str,
        lock: &Lock,
    ) -> Result<ml_exp_core::cloud::action::Execution> {
        let directory = self.action_directory(action_id, false, lock)?;
        let name = OsStr::new("native-cloud-request.json");
        let identity = directory.regular_identity(name)?.ok_or(Error::NotFound)?;
        if !identity.is_private_for(effective_uid()) || identity.mode & 0o077 != 0 {
            return Err(Error::Invalid("private native cloud request required"));
        }
        let bytes = directory.read_regular(name, 2 * 1024 * 1024)?;
        let request = serde_json::from_value(Value::Object(
            ml_exp_core::json::mapping(&bytes)
                .map_err(why!(Error::Invalid("native cloud input")))?,
        ))?;
        ml_exp_core::cloud::action::validate(&self.plan(action_id, lock)?, &request)
            .map_err(why!(Error::Invalid("native cloud binding")))?;
        Ok(request)
    }
    pub fn cloud_request(&self, action_id: &str) -> Result<ml_exp_core::cloud::action::Execution> {
        self.locked(|lock| self.cloud_request_unlocked(action_id, lock))
    }
    pub fn save_plan(&self, plan: &Value, now: &str, transition_id: &str) -> Result<Value> {
        self.locked(|lock| {
            let action_id = plan
                .get("action_id")
                .and_then(Value::as_str)
                .ok_or(Error::Invalid("action_id"))?;
            let directory = self.action_directory(action_id, true, lock)?;
            if let Some(existing) = Self::read_json(&directory, OsStr::new("plan.json"))? {
                if existing.get("request_digest") != plan.get("request_digest") {
                    return Err(Error::IdempotencyConflict);
                }
            } else {
                directory
                    .publish_new(OsStr::new("plan.json"), &serde_json::to_vec_pretty(plan)?)?;
            }
            self.snapshot_unlocked(action_id, now, transition_id, lock)
        })
    }
    fn apply(
        &self,
        action_id: &str,
        transition: &Transition,
        transition_id: &str,
        _lock: &Lock,
    ) -> Result<()> {
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
        self.locked(|lock| {
            // A crash may have published plan.json before its initial database
            // row. Bootstrap and authorization are distinct audit transitions.
            let bootstrap = format!(
                "{:x}",
                Sha256::digest(format!("bootstrap:{action_id}:{transition_id}"))
            );
            let snapshot = self.snapshot_unlocked(action_id, now, &bootstrap[..32], lock)?;
            let transition =
                operations::authorize(&snapshot, &snapshot["execution"], actor, note, now)?;
            self.apply(action_id, &transition, transition_id, lock)?;
            self.snapshot_unlocked(action_id, now, transition_id, lock)
        })
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
        self.locked(|lock| {
        let snapshot = self.snapshot_unlocked(action_id, now, transition_id,lock)?;
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
                self.apply(action_id, &transition, transition_id,lock)?;
                // Auxiliary evidence only: failure here cannot undo or replay
                // the authoritative SQLite execution claim and audit event.
                let claim =
                    json!({"claimed_at":now,"pid":pid,"intent_digest":snapshot["intent_digest"]});
                let _ = self.action_directory(action_id,false,lock)?.replace_regular(OsStr::new("execution.claim"), &serde_json::to_vec_pretty(&claim)?);
                Ok(Claimed {
                    snapshot: self.snapshot_unlocked(action_id, now, transition_id,lock)?,
                    dispatch: Some(dispatch),
                })
            }
        }
        })
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
        self.locked(|lock| {
            self.apply(action_id, transition, transition_id, lock)?;
            self.snapshot_unlocked(action_id, now, transition_id, lock)
        })
    }
    pub fn reconcile(
        &self,
        action_id: &str,
        policy: &Policy,
        evidence: &Evidence,
        now: &str,
        transition_id: &str,
    ) -> Result<(Value, Option<Dispatch>)> {
        self.locked(|lock| {
            let snapshot = self.snapshot_unlocked(action_id, now, transition_id, lock)?;
            match operations::reconcile(&snapshot, &snapshot["execution"], policy, evidence, now)? {
                Reconcile::Unchanged => Ok((snapshot, None)),
                Reconcile::Commit(transition) => {
                    self.apply(action_id, &transition, transition_id, lock)?;
                    Ok((
                        self.snapshot_unlocked(action_id, now, transition_id, lock)?,
                        None,
                    ))
                }
                Reconcile::RecoverTransaction(dispatch) => Ok((snapshot, Some(dispatch))),
            }
        })
    }
    fn list_unlocked(
        &self,
        scope: Option<&Scope>,
        now: &str,
        transition_id: &str,
        lock: &Lock,
    ) -> Result<Vec<Value>> {
        let mut ids = Vec::new();
        for name in self.directory.entry_names()? {
            let Some(id) = name.to_str() else {
                continue;
            };
            if operations::validate_action_id(id).is_err() {
                continue;
            }
            match self.action_directory(id, false, lock) {
                Ok(_) => ids.push(id.to_owned()),
                Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotADirectory => {
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        ids.sort();
        let expected = scope.map(serde_json::to_value).transpose()?;
        let mut items = Vec::new();
        for id in ids {
            let plan = match self.plan(&id, lock) {
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
            items.push(self.snapshot_unlocked(&id, now, transition_id, lock)?);
        }
        Ok(items)
    }
    pub fn list(
        &self,
        scope: Option<&Scope>,
        now: &str,
        transition_id: &str,
    ) -> Result<Vec<Value>> {
        self.locked(|lock| self.list_unlocked(scope, now, transition_id, lock))
    }
    /// Prove no prior plan/committed execution belongs to this Run. An orphaned
    /// committed row cannot be interpreted as absence merely because cleanup
    /// removed its plan directory. Called only by explicit native admission.
    pub fn has_scope_history(&self, scope: &Scope) -> Result<bool> {
        self.locked(|lock| {
            let committed = self.database.admission_action_ids()?;
            let mut ids = std::collections::BTreeSet::from_iter(committed.iter().cloned());
            for name in self.directory.entry_names()? {
                if let Some(id) = name
                    .to_str()
                    .filter(|id| operations::validate_action_id(id).is_ok())
                {
                    ids.insert(id.to_owned());
                }
            }
            let expected = serde_json::to_value(scope)?;
            let mut found = false;
            for id in ids {
                let plan = self.plan(&id, lock)?;
                if plan.get("scope") == Some(&expected) {
                    found = true;
                }
            }
            Ok(found)
        })
    }
    /// Startup must run before the daemon serves or resumes any queue worker.
    /// Merely records abandoned claims; it never issues provider submit calls.
    pub fn recover_interrupted(&self, now: &str, transition_id: &str) -> Result<Vec<Value>> {
        self.locked(|lock| {
            let mut recovered = Vec::new();
            for snapshot in self.list_unlocked(None, now, transition_id, lock)? {
                if let Some(transition) = operations::interrupted(&snapshot["execution"], now)? {
                    let id = snapshot["action_id"]
                        .as_str()
                        .ok_or(Error::Invalid("action_id"))?;
                    self.apply(id, &transition, transition_id, lock)?;
                    recovered.push(self.snapshot_unlocked(id, now, transition_id, lock)?);
                }
            }
            Ok(recovered)
        })
    }
    pub fn append_journal(
        &self,
        action_id: &str,
        event: &str,
        payload: &Value,
        event_id: &str,
        now: &str,
    ) -> Result<()> {
        self.locked(|_lock| {
            self.database.append_event(
                action_id,
                &json!({"timestamp":now,"event":event,"payload":payload}),
                event_id,
            )
        })
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
        self.locked(|lock| {
            let directory = self.subdirectory(action_id, "commands", true, lock)?;
            let name = format!("{}.json", phase.name());
            let record = if let Some(existing) = Self::read_json(&directory, OsStr::new(&name))? {
                // Free-form command replay follows the old Python dictionary
                // equality. Frozen intent/digest identities remain byte-bound.
                if !existing.get("payload").is_some_and(|previous| {
                    ml_exp_core::observation::python_equal(previous, payload)
                }) {
                    return Err(Error::CommandConflict);
                }
                existing
            } else {
                let record = json!({
                    "action_id": action_id,
                    "phase": phase.name(),
                    "payload": payload,
                    "created_at": now,
                    "journal_event_id": event_id,
                });
                directory.publish_new(OsStr::new(&name), &serde_json::to_vec_pretty(&record)?)?;
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
                            .map_err(why!(Error::Invalid("legacy command encoding")))?
                    )
                ),
            };
            let event = json!({
                "timestamp": now,
                "event": format!("{}_command_recorded", phase.name()),
                "payload": {},
            });
            self.database.append_event(action_id, &event, &event_id)?;
            Ok(record)
        })
    }
    pub fn read_command(&self, action_id: &str, phase: CommandPhase) -> Result<Value> {
        self.locked(|lock| {
            let directory = self.subdirectory(action_id, "commands", false, lock)?;
            let record =
                Self::read_json(&directory, OsStr::new(&format!("{}.json", phase.name())))?
                    .ok_or(Error::NotFound)?;
            Ok(record
                .get("payload")
                .filter(|v| v.is_object())
                .cloned()
                .unwrap_or_else(|| json!({})))
        })
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
        self.locked(|lock| {
        let directory = self.subdirectory(action_id,"activity_errors",true,lock)?;
        let message: String = message.chars().take(1000).collect();
        directory.replace_regular(
            OsStr::new(&format!("{phase}.json")),
            &serde_json::to_vec_pretty(
                &json!({"action_id":action_id,"phase":phase,"message":message,"category":category,"created_at":now}),
            )?,
        )?;
        Ok(())
        })
    }
    pub fn activity_error(&self, action_id: &str, phase: &str) -> Result<Value> {
        if !matches!(phase, "prepare" | "authorize" | "execute" | "reconcile") {
            return Err(Error::Invalid("activity phase"));
        }
        self.locked(|lock| {
            let directory = match self.subdirectory(action_id, "activity_errors", false, lock) {
                Ok(directory) => directory,
                Err(Error::NotFound) => return Ok(json!({})),
                Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(json!({}));
                }
                Err(error) => return Err(error),
            };
            Ok(
                Self::read_json(&directory, OsStr::new(&format!("{phase}.json")))?
                    .unwrap_or_else(|| json!({})),
            )
        })
    }
}
