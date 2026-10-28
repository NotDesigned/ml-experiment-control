//! Native durable registry for backend-resident persistent checkpoints.
//!
//! Authentication, worker capability tokens and provider storage remain
//! outside this module. The registry owns only exact identity binding,
//! generation uniqueness, immutable JSON receipts and safe reads.
use fs2::FileExt;
use ml_exp_core::checkpoints::{self, Context};
use ml_exp_io::directory::Directory;
use serde_json::Value;
use std::{ffi::OsStr, fs::File, path::Path};

const NAMESPACE: &str = "persistent-checkpoints";
const LOCK: &str = ".checkpoints.lock";
const MAX_DOCUMENT_BYTES: u64 = checkpoints::MANIFEST_LIMIT as u64 * 2;

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

struct Guard {
    file: File,
}
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

pub struct Store {
    namespace: Directory,
}

impl Store {
    pub fn open(registry_root: &Path) -> Result<Self> {
        let root = Directory::open(registry_root, true)?;
        let namespace = root.child(OsStr::new(NAMESPACE), true)?;
        namespace.open_lock(OsStr::new(LOCK))?;
        Ok(Self { namespace })
    }

    pub fn open_existing(registry_root: &Path) -> Result<Self> {
        let root = Directory::open(registry_root, false)?;
        let namespace = root.child(OsStr::new(NAMESPACE), false)?;
        namespace.verify_private_parent()?;
        namespace.open_existing_lock(OsStr::new(LOCK))?;
        Ok(Self { namespace })
    }

    fn lock(&self) -> Result<Guard> {
        let file = self.namespace.open_existing_lock(OsStr::new(LOCK))?;
        file.try_lock_exclusive().map_err(|error| {
            if error.kind() == std::io::ErrorKind::WouldBlock {
                Error::Busy
            } else {
                Error::Io(error)
            }
        })?;
        Ok(Guard { file })
    }

    fn attempt_directory(&self, context: &Context, create: bool) -> Result<Option<Directory>> {
        checkpoints::context(&serde_json::to_value(context).map_err(|_| Error::Document)?)
            .map_err(|_| Error::Document)?;
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
        let mut bytes = serde_json::to_vec(value).map_err(|_| Error::Document)?;
        bytes.push(b'\n');
        if bytes.len() as u64 > MAX_DOCUMENT_BYTES {
            return Err(Error::Document);
        }
        Ok(bytes)
    }

    fn read_value(directory: &Directory, name: &OsStr) -> Result<Value> {
        let file = directory.open_regular_read(name)?;
        directory
            .validate_immutable_regular_entry(name, &file)
            .map_err(|_| Error::Document)?;
        let bytes = directory.read_regular(name, MAX_DOCUMENT_BYTES)?;
        serde_json::from_slice(&bytes).map_err(|_| Error::Document)
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
            checkpoints::validate_checkpoint_id(checkpoint_id).map_err(|_| Error::Document)?;
            let name = OsString::from(format!("{checkpoint_id}.json"));
            if !directory.entry_exists(&name)? {
                return Ok(Vec::new());
            }
            let value = Self::read_value(&directory, &name)?;
            checkpoints::validate_document(&value, Some(context), Some(checkpoint_id))
                .map_err(|_| Error::Document)?;
            return Ok(vec![value]);
        }
        Self::names(&directory)?
            .into_iter()
            .map(|name| {
                let value = Self::read_value(&directory, &name)?;
                checkpoints::validate_document(
                    &value,
                    Some(context),
                    name.to_str().and_then(|name| name.strip_suffix(".json")),
                )
                .map_err(|_| Error::Document)?;
                Ok(value)
            })
            .collect()
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
        let candidate = checkpoints::document(context, ready).map_err(|_| Error::Document)?;
        let checkpoint_id = candidate["checkpoint_id"]
            .as_str()
            .ok_or(Error::Document)?
            .to_owned();
        if let Some(existing) = self.read_locked(context, Some(&checkpoint_id))?.pop() {
            return Ok(existing);
        }
        let existing = self.read_locked(context, None)?;
        checkpoints::ensure_generation_compatible(&existing, &candidate).map_err(|error| {
            match error {
                checkpoints::Error::GenerationConflict => Error::GenerationConflict,
                _ => Error::Document,
            }
        })?;
        let registered =
            checkpoints::register(&candidate, registered_at).map_err(|_| Error::Document)?;
        let directory = self
            .attempt_directory(context, true)?
            .ok_or(Error::Document)?;
        let name = OsString::from(format!("{checkpoint_id}.json"));
        directory.publish_new(&name, &Self::encode(&registered)?)?;
        let file = directory.open_regular_read(&name)?;
        directory.seal_regular(&name, &file, false)?;
        Ok(registered)
    }
}

use std::ffi::OsString;

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
