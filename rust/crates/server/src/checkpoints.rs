//! Native durable registry for backend-resident persistent checkpoints.
//!
//! Authentication, worker capability tokens and provider storage remain
//! outside this module. The registry owns only exact identity binding,
//! generation uniqueness, immutable JSON receipts and safe reads.
use ml_exp_core::checkpoints::{self, Context};
use ml_exp_io::directory::{Directory, FileLock, LockOptions};
use serde_json::Value;
use std::{ffi::OsStr, path::Path};

const NAMESPACE: &str = "persistent-checkpoints";
const LOCK: &str = ".checkpoints.lock";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("checkpoint registry filesystem operation failed")]
    Io(#[from] std::io::Error),
    #[error("checkpoint registry document is invalid")]
    Document,
    #[error("checkpoint registry operation is busy")]
    Busy,
    #[error("checkpoint generation is already bound to another manifest")]
    GenerationConflict,
}
pub type Result<T> = std::result::Result<T, Error>;

pub struct Store {
    namespace: Directory,
}

impl Store {
    pub fn open(registry_root: &Path) -> Result<Self> {
        let root = Directory::open(registry_root, true)?;
        let namespace = root.child(OsStr::new(NAMESPACE), true)?;
        namespace.create_lock_file(OsStr::new(LOCK))?;
        Ok(Self { namespace })
    }

    pub fn open_existing(registry_root: &Path) -> Result<Self> {
        let root = Directory::open(registry_root, false)?;
        let namespace = root.child(OsStr::new(NAMESPACE), false)?;
        namespace.open_regular_read(OsStr::new(LOCK))?;
        Ok(Self { namespace })
    }

    fn lock(&self) -> Result<FileLock> {
        self.namespace
            .lock(OsStr::new(LOCK), LockOptions::default())
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::WouldBlock {
                    Error::Busy
                } else {
                    Error::Io(error)
                }
            })
    }

    fn attempt_directory(&self, context: &Context, create: bool) -> Result<Option<Directory>> {
        checkpoints::context(&serde_json::to_value(context).map_err(why!(Error::Document))?)
            .map_err(why!(Error::Document))?;
        let project = match self.namespace.child(OsStr::new(&context.project), create) {
            Ok(directory) => directory,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(Error::Io(error)),
        };
        let run = match project.child(OsStr::new(&context.run_id), create) {
            Ok(directory) => directory,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(Error::Io(error)),
        };
        match run.child(OsStr::new(&context.attempt_id), create) {
            Ok(directory) => Ok(Some(directory)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(Error::Io(error)),
        }
    }

    fn encode(value: &Value) -> Result<Vec<u8>> {
        let mut bytes = serde_json::to_vec(value).map_err(why!(Error::Document))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    fn read_value(directory: &Directory, name: &OsStr) -> Result<Value> {
        let file = directory.open_regular_read(name)?;
        ml_exp_io::directory::require_sealed(&file).map_err(|_| Error::Document)?;
        let bytes = directory.read_whole(name)?;
        serde_json::from_slice(&bytes).map_err(why!(Error::Document))
    }

    fn names(directory: &Directory) -> Result<Vec<std::ffi::OsString>> {
        let mut names = directory
            .entry_names_limited(checkpoints::MAX_FILES)?
            .into_iter()
            .filter(|name| {
                name.to_str().is_some_and(|value| {
                    value.starts_with("checkpoint.") && value.ends_with(".json")
                })
            })
            .collect::<Vec<_>>();
        names.sort();
        Ok(names)
    }

    fn read_locked(&self, context: &Context, checkpoint_id: Option<&str>) -> Result<Vec<Value>> {
        let Some(directory) = self.attempt_directory(context, false)? else {
            return Ok(Vec::new());
        };
        if let Some(checkpoint_id) = checkpoint_id {
            checkpoints::validate_checkpoint_id(checkpoint_id).map_err(why!(Error::Document))?;
            let name = OsString::from(format!("{checkpoint_id}.json"));
            if !directory.entry_exists(&name)?
                || directory.entry_exists(OsStr::new(&format!("withdrawn.{checkpoint_id}.json")))?
            {
                return Ok(Vec::new());
            }
            let value = Self::read_value(&directory, &name)?;
            checkpoints::validate_document(&value, Some(context), Some(checkpoint_id))
                .map_err(why!(Error::Document))?;
            return Ok(vec![value]);
        }
        let mut values = Vec::new();
        for name in Self::names(&directory)? {
            if directory
                .entry_exists(OsStr::new(&format!("withdrawn.{}", name.to_string_lossy())))?
            {
                continue;
            }
            let value = Self::read_value(&directory, &name)?;
            checkpoints::validate_document(
                &value,
                Some(context),
                name.to_str().and_then(|name| name.strip_suffix(".json")),
            )
            .map_err(why!(Error::Document))?;
            values.push(value);
        }
        Ok(values)
    }

    pub fn list(&self, context: &Context) -> Result<Vec<Value>> {
        let _guard = self.lock()?;
        self.read_locked(context, None)
    }

    pub fn read(&self, context: &Context, checkpoint_id: &str) -> Result<Option<Value>> {
        let _guard = self.lock()?;
        Ok(self.read_locked(context, Some(checkpoint_id))?.pop())
    }

    pub fn register(&self, context: &Context, ready: &Value, registered_at: &str) -> Result<Value> {
        let _guard = self.lock()?;
        let candidate = checkpoints::document(context, ready).map_err(why!(Error::Document))?;
        let checkpoint_id = candidate["checkpoint_id"]
            .as_str()
            .ok_or(Error::Document)?
            .to_owned();
        if let Some(existing) = self.read_locked(context, Some(&checkpoint_id))?.pop() {
            return Ok(existing);
        }
        if self.attempt_directory(context, false)?.is_some_and(|d| {
            d.entry_exists(OsStr::new(&format!("withdrawn.{checkpoint_id}.json")))
                .unwrap_or(true)
        }) {
            return Err(Error::Document);
        }
        let existing = self.read_locked(context, None)?;
        checkpoints::ensure_generation_compatible(&existing, &candidate).map_err(|error| {
            match error {
                checkpoints::Error::GenerationConflict => Error::GenerationConflict,
                _ => Error::Document,
            }
        })?;
        let registered =
            checkpoints::register(&candidate, registered_at).map_err(why!(Error::Document))?;
        let directory = self
            .attempt_directory(context, true)?
            .ok_or(Error::Document)?;
        let name = OsString::from(format!("{checkpoint_id}.json"));
        directory.publish_new(&name, &Self::encode(&registered)?)?;
        let file = directory.open_regular_read(&name)?;
        ml_exp_io::directory::seal_regular(&file, false)?;
        Ok(registered)
    }
}

use std::ffi::OsString;

/// Which Attempt registered a checkpoint, for a request that names only the Run.
impl Store {
    /// The Attempt of this Run holding this registered, not withdrawn checkpoint.
    /// A checkpoint identity covers its Attempt, so a second holder is refused.
    pub fn attempt_of(
        &self,
        project: &str,
        run: &str,
        checkpoint_id: &str,
    ) -> Result<Option<String>> {
        ml_exp_core::identity::validate_id(project).map_err(why!(Error::Document))?;
        ml_exp_core::identity::validate_id(run).map_err(why!(Error::Document))?;
        checkpoints::validate_checkpoint_id(checkpoint_id).map_err(why!(Error::Document))?;
        let child =
            |directory: &Directory, name: &str| match directory.child(OsStr::new(name), false) {
                Ok(directory) => Ok(Some(directory)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(error) => Err(Error::Io(error)),
            };
        let _guard = self.lock()?;
        let Some(project) = child(&self.namespace, project)? else {
            return Ok(None);
        };
        let Some(run) = child(&project, run)? else {
            return Ok(None);
        };
        let name = OsString::from(format!("{checkpoint_id}.json"));
        let withdrawn = OsString::from(format!("withdrawn.{checkpoint_id}.json"));
        let mut found = None;
        for attempt in run.entry_names_limited(checkpoints::MAX_FILES)? {
            let Some(attempt) = attempt
                .to_str()
                .filter(|a| ml_exp_core::identity::validate_attempt(a).is_ok())
            else {
                continue;
            };
            let Some(directory) = child(&run, attempt)? else {
                continue;
            };
            if directory.entry_exists(&name)? && !directory.entry_exists(&withdrawn)? {
                if found.is_some() {
                    return Err(Error::Document);
                }
                found = Some(attempt.to_owned());
            }
        }
        Ok(found)
    }
}

/// Fill the Attempt a reference left out from the registry. `Ok(false)` means no
/// Attempt of that Run registered the checkpoint. A named Attempt is kept as given,
/// and the exact read that follows refuses it when the checkpoint is not its own.
pub fn resolve(
    registry_root: &Path,
    project: &str,
    reference: &mut checkpoints::Reference,
) -> Result<bool> {
    if !reference.unresolved() {
        return Ok(true);
    }
    let store = match Store::open_existing(registry_root) {
        Ok(store) => store,
        Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(false);
        }
        Err(error) => return Err(error),
    };
    match store.attempt_of(project, &reference.run_id, &reference.checkpoint_id)? {
        Some(attempt) => {
            reference.attempt_id = attempt;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// A retry may leave its Attempt out: a restart is from the retried Attempt, and a
/// checkpoint belongs to the Attempt that registered it. `Ok(false)` as [`resolve`].
pub fn resolve_retry(
    registry_root: &Path,
    project: &str,
    retried: &str,
    request: &mut ml_exp_core::first_submission::native::retry::Request,
) -> Result<bool> {
    if let Some(start) = request
        .restart_from
        .as_mut()
        .filter(|start| start.attempt_id.is_empty())
    {
        start.attempt_id = retried.into();
    }
    match request.resume_from.as_mut() {
        Some(reference) => resolve(registry_root, project, reference),
        None => Ok(true),
    }
}

/// Object-store copies. A receipt lives beside the document it describes, sealed,
/// and is written only after the server has read every object back.
impl Store {
    fn archive_name(checkpoint_id: &str) -> Result<OsString> {
        checkpoints::validate_checkpoint_id(checkpoint_id).map_err(why!(Error::Document))?;
        Ok(OsString::from(format!("archive.{checkpoint_id}.json")))
    }
    fn archived_locked(
        &self,
        context: &Context,
        document: &Value,
    ) -> Result<Option<checkpoints::archive::Receipt>> {
        let id = document["checkpoint_id"].as_str().ok_or(Error::Document)?;
        let Some(directory) = self.attempt_directory(context, false)? else {
            return Ok(None);
        };
        let name = Self::archive_name(id)?;
        if !directory.entry_exists(&name)? {
            return Ok(None);
        }
        let file = directory.open_regular_read(&name)?;
        let receipt: checkpoints::archive::Receipt =
            serde_json::from_slice(&directory.read_whole(&name)?).map_err(why!(Error::Document))?;
        receipt.validate(document).map_err(why!(Error::Document))?;
        // Publication is followed by sealing. A stop between the two leaves this exact,
        // valid receipt writable, and sealing it now finishes the same write.
        if ml_exp_io::directory::require_sealed(&file).is_err() {
            ml_exp_io::directory::seal_regular(&file, false)?;
        }
        Ok(Some(receipt))
    }
    /// The verified copy of one registered, not withdrawn checkpoint, if there is one.
    pub fn archived(
        &self,
        context: &Context,
        checkpoint_id: &str,
    ) -> Result<Option<(Value, Option<checkpoints::archive::Receipt>)>> {
        let _guard = self.lock()?;
        let Some(document) = self.read_locked(context, Some(checkpoint_id))?.pop() else {
            return Ok(None);
        };
        let receipt = self.archived_locked(context, &document)?;
        Ok(Some((document, receipt)))
    }
    /// Every registered document of this Attempt with its verified copy, if any.
    pub fn list_archived(
        &self,
        context: &Context,
    ) -> Result<Vec<(Value, Option<checkpoints::archive::Receipt>)>> {
        let _guard = self.lock()?;
        let mut result = Vec::new();
        for document in self.read_locked(context, None)? {
            let receipt = self.archived_locked(context, &document)?;
            result.push((document, receipt));
        }
        Ok(result)
    }
    /// Keep the first receipt. A second one for the same document is the same
    /// statement made later, so the first is returned.
    pub fn record_archive(
        &self,
        context: &Context,
        receipt: &checkpoints::archive::Receipt,
    ) -> Result<checkpoints::archive::Receipt> {
        let _guard = self.lock()?;
        let document = self
            .read_locked(context, Some(&receipt.checkpoint_id))?
            .pop()
            .ok_or(Error::Document)?;
        receipt.validate(&document).map_err(why!(Error::Document))?;
        if let Some(existing) = self.archived_locked(context, &document)? {
            return Ok(existing);
        }
        let directory = self
            .attempt_directory(context, false)?
            .ok_or(Error::Document)?;
        let name = Self::archive_name(&receipt.checkpoint_id)?;
        let value = serde_json::to_value(receipt).map_err(why!(Error::Document))?;
        directory.publish_new(&name, &Self::encode(&value)?)?;
        let file = directory.open_regular_read(&name)?;
        ml_exp_io::directory::seal_regular(&file, false)?;
        Ok(receipt.clone())
    }
}

impl Store {
    pub fn withdraw(&self, context: &Context, id: &str, plan_id: &str) -> Result<()> {
        checkpoints::validate_checkpoint_id(id).map_err(why!(Error::Document))?;
        ml_exp_core::sources::validate_archive_digest(plan_id).map_err(why!(Error::Document))?;
        let _guard = self.lock()?;
        let directory = self
            .attempt_directory(context, false)?
            .ok_or(Error::Document)?;
        let original = Self::read_value(&directory, OsStr::new(&format!("{id}.json")))?;
        checkpoints::validate_document(&original, Some(context), Some(id))
            .map_err(why!(Error::Document))?;
        let name = format!("withdrawn.{id}.json");
        let value =
            serde_json::json!({"checkpoint_id":id,"plan_id":plan_id,"status":"REMOVAL_REQUESTED"});
        if directory.entry_exists(OsStr::new(&name))? {
            if directory.read_whole(OsStr::new(&name))? != Self::encode(&value)? {
                return Err(Error::Document);
            }
        } else {
            directory.publish_new(OsStr::new(&name), &Self::encode(&value)?)?;
        }
        let file = directory.open_regular_read(OsStr::new(&name))?;
        ml_exp_io::directory::seal_regular(&file, false)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{fs, os::unix::fs::PermissionsExt};

    fn context() -> Context {
        Context {
            project: "demo".into(),
            run_id: "trial".into(),
            attempt_id: "attempt-001".into(),
            source_id: format!("source.{}", "a".repeat(64)),
            image_id: format!("sha256:{}", "b".repeat(64)),
            storage_scope: format!("storage.{}", "c".repeat(64)),
            state_root: "/data/demo/trial/attempt-001/state".into(),
        }
    }

    fn ready(step: u64, generation: &str, bytes: u64) -> Value {
        json!({
            "step":step,
            "files":[{"path":format!("checkpoints/{generation}/state.bin"),"bytes":bytes,"sha256":"d".repeat(64)}]
        })
    }

    #[test]
    fn registration_is_idempotent_and_reads_exact_attempt() {
        let temp = tempfile::tempdir().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let store = Store::open(temp.path()).unwrap();
        let context = context();
        let first = store
            .register(&context, &ready(4, "step-0004", 7), "2026-10-28T00:00:00Z")
            .unwrap();
        assert_eq!(
            store
                .register(&context, &ready(4, "step-0004", 7), "2026-10-28T00:01:00Z")
                .unwrap(),
            first
        );
        assert_eq!(store.list(&context).unwrap(), vec![first.clone()]);
        assert_eq!(
            store
                .read(&context, first["checkpoint_id"].as_str().unwrap())
                .unwrap(),
            Some(first)
        );
    }

    #[test]
    fn a_checkpoint_names_its_attempt_and_a_named_one_is_kept() {
        let temp = tempfile::tempdir().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let store = Store::open(temp.path()).unwrap();
        let first = context();
        let second = Context {
            attempt_id: "attempt-002".into(),
            state_root: "/data/demo/trial/attempt-002/state".into(),
            ..context()
        };
        let one = store
            .register(&first, &ready(4, "step-0004", 7), "2026-10-28T00:00:00Z")
            .unwrap();
        let two = store
            .register(&second, &ready(8, "step-0008", 7), "2026-10-28T00:00:00Z")
            .unwrap();
        for (document, attempt) in [(&one, "attempt-001"), (&two, "attempt-002")] {
            let id = document["checkpoint_id"].as_str().unwrap();
            let mut received = checkpoints::Reference {
                run_id: "trial".into(),
                attempt_id: String::new(),
                checkpoint_id: id.into(),
            };
            assert!(resolve(temp.path(), "demo", &mut received).unwrap());
            assert_eq!(received.attempt_id, attempt);
            let mut explicit = received.clone();
            assert!(resolve(temp.path(), "demo", &mut explicit).unwrap());
            assert_eq!(explicit, received);
            // A named Attempt is never replaced: the exact read refuses a wrong one.
            let mut wrong = checkpoints::Reference {
                attempt_id: "attempt-009".into(),
                ..received.clone()
            };
            assert!(resolve(temp.path(), "demo", &mut wrong).unwrap());
            assert_eq!(wrong.attempt_id, "attempt-009");
        }
        let unknown = format!("checkpoint.{}", "f".repeat(64));
        for (project, run) in [("demo", "trial"), ("demo", "other"), ("else", "trial")] {
            let mut reference = checkpoints::Reference {
                run_id: run.into(),
                attempt_id: String::new(),
                checkpoint_id: unknown.clone(),
            };
            assert!(!resolve(temp.path(), project, &mut reference).unwrap());
            assert!(reference.unresolved());
        }
        let mut restart: ml_exp_core::first_submission::native::retry::Request =
            serde_json::from_value(json!({"restart_from":{"run_id":"trial"}})).unwrap();
        assert!(resolve_retry(temp.path(), "demo", "attempt-002", &mut restart).unwrap());
        assert_eq!(restart.restart_from.unwrap().attempt_id, "attempt-002");
        let empty = tempfile::tempdir().unwrap();
        fs::set_permissions(empty.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let mut reference = checkpoints::Reference {
            run_id: "trial".into(),
            attempt_id: String::new(),
            checkpoint_id: unknown,
        };
        assert!(!resolve(empty.path(), "demo", &mut reference).unwrap());
    }

    #[test]
    fn an_archive_receipt_is_kept_once_beside_its_document() {
        let temp = tempfile::tempdir().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let store = Store::open(temp.path()).unwrap();
        let context = context();
        let document = store
            .register(&context, &ready(4, "step-0004", 7), "2026-10-28T00:00:00Z")
            .unwrap();
        let id = document["checkpoint_id"].as_str().unwrap().to_owned();
        let id = id.as_str();
        assert_eq!(
            store.archived(&context, id).unwrap(),
            Some((document.clone(), None))
        );
        let receipt =
            checkpoints::archive::Receipt::new(&document, "2026-11-08T00:00:00Z", vec![7]).unwrap();
        assert_eq!(store.record_archive(&context, &receipt).unwrap(), receipt);
        let later =
            checkpoints::archive::Receipt::new(&document, "2026-11-09T00:00:00Z", vec![7]).unwrap();
        assert_eq!(store.record_archive(&context, &later).unwrap(), receipt);
        assert_eq!(
            store.list_archived(&context).unwrap(),
            vec![(document.clone(), Some(receipt.clone()))]
        );
        // The receipt is not a checkpoint document and is never listed as one.
        assert_eq!(store.list(&context).unwrap(), vec![document]);
        let mut wrong = receipt.clone();
        wrong.bytes = 8;
        assert!(store.record_archive(&context, &wrong).is_err());
        // A stop between publication and sealing: the receipt is sealed on the next read.
        let path = temp.path().join(format!(
            "persistent-checkpoints/demo/trial/attempt-001/archive.{id}.json"
        ));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(
            store.archived(&context, id).unwrap().unwrap().1,
            Some(receipt)
        );
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o400
        );
    }

    #[test]
    fn generation_collision_and_corrupt_receipt_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path()).unwrap();
        let context = context();
        let first = store
            .register(&context, &ready(4, "step-0004", 7), "2026-10-28T00:00:00Z")
            .unwrap();
        assert!(matches!(
            store.register(&context, &ready(5, "step-0004", 8), "2026-10-28T00:00:00Z"),
            Err(Error::GenerationConflict)
        ));
        let path = temp.path().join(format!(
            "persistent-checkpoints/demo/trial/attempt-001/{}.json",
            first["checkpoint_id"].as_str().unwrap()
        ));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&path, b"{}").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o400)).unwrap();
        assert!(matches!(store.list(&context), Err(Error::Document)));
    }
}
