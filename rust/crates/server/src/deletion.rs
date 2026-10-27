//! Unix deletion fences over retained directory capabilities. No research asset
//! removal or provider effect occurs here. Other platforms are not accepted as
//! a server replacement by this branch.
#![cfg(unix)]
#![forbid(unsafe_code)]

use fs2::FileExt;
use ml_exp_core::identity::{validate_id, validate_observation_id};
use ml_exp_io::directory::Directory;
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fs::File,
    io,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

type Result<T> = std::result::Result<T, Error>;
const LOCK: &str = ".run-deletion.lock";
const MAX_TOMBSTONE: u64 = 4 * 1024 * 1024;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid Run deletion identity")]
    InvalidIdentity,
    #[error("Run is busy; retry deletion")]
    Busy,
    #[error("Run was deleted by an administrator")]
    Deleted,
    #[error("exclusive Run deletion guard required")]
    ExclusiveRequired,
    #[error("Run deletion storage I/O failed")]
    Io(#[from] io::Error),
}

#[derive(Debug, Default)]
struct Anchors {
    root: OnceLock<Directory>,
    deleted: OnceLock<Directory>,
    projects: Mutex<BTreeMap<String, Directory>>,
}
#[derive(Clone, Debug)]
pub struct RunFence {
    root: PathBuf,
    anchors: Arc<Anchors>,
}
pub struct Guard<'a> {
    fence: &'a RunFence,
    directory: Directory,
    file: File,
    exclusive: bool,
}

fn fixed(cell: &OnceLock<Directory>, candidate: Directory) -> io::Result<Directory> {
    let _ = cell.set(candidate);
    let directory = cell
        .get()
        .ok_or_else(|| io::Error::other("directory anchor unavailable"))?
        .clone();
    directory.verify_attached()?;
    Ok(directory)
}

impl RunFence {
    /// Construction is side-effect free. Descriptors are pinned on first use;
    /// clones share the same root/deleted/project capabilities thereafter.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            anchors: Arc::new(Anchors::default()),
        }
    }

    /// Display/provenance only. Actual filesystem operations never use this path.
    pub fn path(&self, project: &str, run: &str) -> Result<PathBuf> {
        self.leaf(project, run)?;
        Ok(self
            .root
            .join("deleted-runs")
            .join(project)
            .join(format!("{run}.json")))
    }
    fn leaf(&self, project: &str, run: &str) -> Result<String> {
        validate_id(project).map_err(|_| Error::InvalidIdentity)?;
        validate_observation_id(run).map_err(|_| Error::InvalidIdentity)?;
        Ok(format!("{run}.json"))
    }
    fn directory(&self, create: bool) -> io::Result<Directory> {
        if let Some(directory) = self.anchors.root.get() {
            directory.verify_attached()?;
            return Ok(directory.clone());
        }
        fixed(&self.anchors.root, Directory::open(&self.root, create)?)
    }
    fn deleted_directory(&self, create: bool) -> io::Result<Directory> {
        if let Some(directory) = self.anchors.deleted.get() {
            directory.verify_attached()?;
            return Ok(directory.clone());
        }
        fixed(
            &self.anchors.deleted,
            self.directory(create)?
                .child(OsStr::new("deleted-runs"), create)?,
        )
    }
    fn project_directory(&self, project: &str, create: bool) -> Result<Option<Directory>> {
        let mut projects = self
            .anchors
            .projects
            .lock()
            .map_err(|_| io::Error::other("directory anchor lock poisoned"))?;
        if let Some(directory) = projects.get(project) {
            directory.verify_attached()?;
            return Ok(Some(directory.clone()));
        }
        let result = self
            .deleted_directory(create)
            .and_then(|root| root.child(OsStr::new(project), create));
        match result {
            Ok(directory) => {
                projects.insert(project.into(), directory.clone());
                Ok(Some(directory))
            }
            Err(error) if !create && error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
    fn lock(&self, exclusive: bool, blocking: bool, initialize: bool) -> Result<Guard<'_>> {
        let directory = self.directory(initialize)?;
        let file = if initialize {
            directory.open_lock(OsStr::new(LOCK))?
        } else {
            directory.open_existing_lock(OsStr::new(LOCK))?
        };
        let result = match (exclusive, blocking) {
            (true, true) => FileExt::lock_exclusive(&file),
            (true, false) => FileExt::try_lock_exclusive(&file),
            (false, true) => FileExt::lock_shared(&file),
            (false, false) => FileExt::try_lock_shared(&file),
        };
        match result {
            Ok(()) => {
                directory.validate_regular_entry(OsStr::new(LOCK), &file)?;
                // Writable initialization pins/creates the metadata root. A
                // read-only observer must never repair missing filesystem state.
                if initialize {
                    self.deleted_directory(true)?;
                }
                Ok(Guard {
                    fence: self,
                    directory,
                    file,
                    exclusive,
                })
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Err(Error::Busy),
            Err(error) => Err(error.into()),
        }
    }
    pub fn shared(&self) -> Result<Guard<'_>> {
        self.lock(false, true, true)
    }
    /// Shared observation over an existing root and regular lock. Never creates
    /// directories, lock files or metadata; absence/corruption fails closed.
    pub fn shared_existing(&self) -> Result<Guard<'_>> {
        self.lock(false, true, false)
    }
    pub fn exclusive(&self, blocking: bool) -> Result<Guard<'_>> {
        self.lock(true, blocking, true)
    }
    pub fn guard(&self, project: &str, run: &str) -> Result<Guard<'_>> {
        self.leaf(project, run)?;
        self.observation_guard(project, run)
    }
    pub fn observation_guard(&self, project: &str, run: &str) -> Result<Guard<'_>> {
        let guard = self.shared()?;
        guard.require_observed_live(project, run)?;
        Ok(guard)
    }
    /// Compatibility observation without initializing any filesystem object.
    /// A missing lock is unavailable evidence and is never silently rebuilt.
    pub fn existing_observation_guard(&self, project: &str, run: &str) -> Result<Guard<'_>> {
        let guard = self.shared_existing()?;
        guard.require_observed_live(project, run)?;
        Ok(guard)
    }
    pub fn is_deleted(&self, project: &str, run: &str) -> Result<bool> {
        let name = match self.leaf(project, run) {
            Ok(name) => name,
            Err(Error::InvalidIdentity) => return Ok(false),
            Err(error) => return Err(error),
        };
        match self.project_directory(project, false)? {
            Some(directory) => Ok(directory.entry_exists(OsStr::new(&name))?),
            None => Ok(false),
        }
    }
    pub fn mark(
        &self,
        project: &str,
        run: &str,
        metadata: &Map<String, Value>,
        at: &str,
    ) -> Result<Value> {
        self.leaf(project, run)?;
        self.exclusive(false)?.mark(project, run, metadata, at)
    }
}

impl Guard<'_> {
    /// Revalidate the retained directory ancestry and exact lock-file binding.
    /// Store owners must call this after blocking acquisition, before commit,
    /// and before returning a read or mutation result; acquiring the guard once
    /// cannot make later external filesystem replacement safe.
    pub fn validate(&self) -> Result<()> {
        Ok(self
            .directory
            .validate_regular_entry(OsStr::new(LOCK), &self.file)?)
    }
    pub fn require_observed_live(&self, project: &str, run: &str) -> Result<()> {
        self.validate()?;
        if self.fence.is_deleted(project, run)? {
            Err(Error::Deleted)
        } else {
            Ok(())
        }
    }
    pub fn require_live(&self, project: &str, run: &str) -> Result<()> {
        self.fence.leaf(project, run)?;
        self.require_observed_live(project, run)
    }
    pub fn is_deleted(&self, project: &str, run: &str) -> Result<bool> {
        self.validate()?;
        self.fence.is_deleted(project, run)
    }
    pub fn mark(
        &self,
        project: &str,
        run: &str,
        metadata: &Map<String, Value>,
        at: &str,
    ) -> Result<Value> {
        if !self.exclusive {
            return Err(Error::ExclusiveRequired);
        }
        self.validate()?;
        let name = self.fence.leaf(project, run)?;
        let directory = self
            .fence
            .project_directory(project, true)?
            .ok_or_else(|| io::Error::other("project directory unavailable"))?;
        if directory.entry_exists(OsStr::new(&name))? {
            let value = read_existing(&directory, OsStr::new(&name)).unwrap_or(json!({}));
            directory.verify_attached()?;
            self.validate()?;
            return Ok(value);
        }
        let mut record = metadata.clone();
        record.insert("project".into(), json!(project));
        record.insert("run_id".into(), json!(run));
        record.insert("status".into(), json!("DELETED"));
        record.insert("deleted_at".into(), json!(at));
        let value = Value::Object(record);
        let bytes = serde_json::to_vec(&value).map_err(io::Error::other)?;
        match directory.publish_new(OsStr::new(&name), &bytes) {
            Ok(()) => {
                self.validate()?;
                Ok(value)
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let previous = read_existing(&directory, OsStr::new(&name)).unwrap_or(json!({}));
                directory.verify_attached()?;
                self.validate()?;
                Ok(previous)
            }
            Err(error) => Err(error.into()),
        }
    }
}
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}
fn read_existing(directory: &Directory, name: &OsStr) -> io::Result<Value> {
    serde_json::from_slice(&directory.read_regular(name, MAX_TOMBSTONE)?).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_io::create_private_dir;
    use std::fs;

    #[test]
    fn deletion_is_permanent_idempotent_and_cannot_upgrade_shared_guard() {
        let temp = tempfile::tempdir().unwrap();
        let fence = RunFence::new(temp.path());
        let read = fence.guard("p", "r").unwrap();
        assert!(matches!(
            fence.mark("p", "r", &Map::new(), "first"),
            Err(Error::Busy)
        ));
        assert!(matches!(
            read.mark("p", "r", &Map::new(), "first"),
            Err(Error::ExclusiveRequired)
        ));
        drop(read);
        let value = fence.mark("p", "r", &Map::new(), "first").unwrap();
        assert_eq!(
            value,
            json!({"project":"p","run_id":"r","status":"DELETED","deleted_at":"first"})
        );
        assert_eq!(value, fence.mark("p", "r", &Map::new(), "second").unwrap());
        assert!(matches!(fence.guard("p", "r"), Err(Error::Deleted)));
        assert!(fence.guard("p", "other").is_ok());
    }

    #[test]
    fn historical_unsafe_ids_do_not_construct_paths_and_companion_ids_are_fenced() {
        let temp = tempfile::tempdir().unwrap();
        let fence = RunFence::new(temp.path());
        for id in ["../x", "x/y", "", ".dot", "a\n", "模型"] {
            assert!(fence.path("p", id).is_err());
            assert!(!fence.is_deleted("p", id).unwrap());
            assert!(fence.observation_guard("p", id).is_ok());
        }
        let companion = format!("{}--preparation", "x".repeat(128));
        assert!(fence.path("p", &companion).is_ok());
        fence.mark("p", &companion, &Map::new(), "now").unwrap();
        assert!(fence.is_deleted("p", &companion).unwrap());
    }

    #[test]
    fn damaged_tombstone_remains_fenced_without_being_rewritten() {
        let temp = tempfile::tempdir().unwrap();
        let fence = RunFence::new(temp.path());
        let path = fence.path("p", "r").unwrap();
        create_private_dir(path.parent().unwrap()).unwrap();
        fs::write(&path, b"bad JSON").unwrap();
        assert!(matches!(fence.guard("p", "r"), Err(Error::Deleted)));
        assert_eq!(fence.mark("p", "r", &Map::new(), "now").unwrap(), json!({}));
        assert_eq!(fs::read(path).unwrap(), b"bad JSON");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_and_fifo_cannot_be_followed_or_block_the_lock() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let fence = RunFence::new(temp.path());
        let path = fence.path("p", "r").unwrap();
        create_private_dir(path.parent().unwrap()).unwrap();
        symlink(temp.path().join("absent"), &path).unwrap();
        assert!(fence.is_deleted("p", "r").unwrap());
        assert_eq!(fence.mark("p", "r", &Map::new(), "now").unwrap(), json!({}));
        let lock = temp.path().join(".run-deletion.lock");
        fs::remove_file(&lock).unwrap();
        let status = std::process::Command::new("mkfifo")
            .arg(&lock)
            .status()
            .unwrap();
        assert!(status.success());
        assert!(fence.shared().is_err());
    }

    #[test]
    fn cross_process_guards_use_the_same_exclusive_lock() {
        const CHILD_ROOT: &str = "ML_EXPD_SYNTHETIC_FENCE_CHILD_ROOT";
        const CHILD_BUSY: &str = "ML_EXPD_SYNTHETIC_FENCE_EXPECT_BUSY";
        if let Some(root) = std::env::var_os(CHILD_ROOT) {
            let fence = RunFence::new(root);
            if std::env::var_os(CHILD_BUSY).is_some() {
                assert!(matches!(fence.exclusive(false), Err(Error::Busy)));
            } else {
                assert!(fence.exclusive(false).is_ok());
            }
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let fence = RunFence::new(temp.path());
        let guard = fence.shared().unwrap();
        let run = |busy| {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "deletion::tests::cross_process_guards_use_the_same_exclusive_lock",
                    "--test-threads=1",
                ])
                .env(CHILD_ROOT, temp.path());
            if busy {
                command.env(CHILD_BUSY, "1");
            } else {
                command.env_remove(CHILD_BUSY);
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        };
        run(true);
        drop(guard);
        run(false);
    }

    #[test]
    fn intermediate_symlinks_never_receive_tombstones_or_lock_files() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let outside = temp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        for suffix in ["root-link", "metadata-link", "project-link"] {
            let root = temp.path().join(suffix);
            if suffix == "root-link" {
                symlink(&outside, &root).unwrap();
            } else {
                fs::create_dir(&root).unwrap();
                if suffix == "metadata-link" {
                    symlink(&outside, root.join("deleted-runs")).unwrap();
                } else {
                    fs::create_dir(root.join("deleted-runs")).unwrap();
                    symlink(&outside, root.join("deleted-runs/p")).unwrap();
                }
            }
            let fence = RunFence::new(root);
            assert!(fence.mark("p", "r", &Map::new(), "now").is_err());
            assert!(fence.is_deleted("p", "r").is_err());
            assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
        }
    }

    #[test]
    fn cached_parent_capabilities_fail_closed_when_directories_are_replaced() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        let outside = temp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        let fence = RunFence::new(&root);
        fence.mark("p", "r", &Map::new(), "first").unwrap();
        fs::rename(root.join("deleted-runs/p"), root.join("deleted-runs/old-p")).unwrap();
        assert!(fence.is_deleted("p", "r").is_err());
        symlink(&outside, root.join("deleted-runs/p")).unwrap();
        assert!(fence.is_deleted("p", "r").is_err());
        assert!(fence.mark("p", "new", &Map::new(), "now").is_err());
        assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
        fs::remove_file(root.join("deleted-runs/p")).unwrap();
        fs::create_dir(root.join("deleted-runs/p")).unwrap();
        assert!(fence.is_deleted("p", "r").is_err());
        assert!(root.join("deleted-runs/old-p/r.json").is_file());
        // Replacing a higher metadata parent must also retain the old fence as
        // unavailable, never report the newly empty directory as a live identity.
        fs::rename(root.join("deleted-runs"), root.join("old-deleted")).unwrap();
        fs::create_dir(root.join("deleted-runs")).unwrap();
        assert!(fence.is_deleted("p", "r").is_err());
    }

    #[test]
    fn root_replacement_during_an_exclusive_guard_never_redirects_publication() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        let outside = temp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        let fence = RunFence::new(&root);
        let guard = fence.exclusive(false).unwrap();
        fs::rename(&root, temp.path().join("old-root")).unwrap();
        symlink(&outside, &root).unwrap();
        assert!(guard.mark("p", "r", &Map::new(), "now").is_err());
        assert!(guard.require_live("p", "r").is_err());
        assert!(fence.is_deleted("p", "r").is_err());
        assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
        assert!(!temp.path().join("old-root/deleted-runs/p/r.json").exists());
    }

    #[test]
    fn replacing_a_lock_entry_invalidates_an_existing_guard() {
        let temp = tempfile::tempdir().unwrap();
        let fence = RunFence::new(temp.path());
        let guard = fence.exclusive(false).unwrap();
        fs::rename(temp.path().join(LOCK), temp.path().join("old-lock")).unwrap();
        fs::write(temp.path().join(LOCK), b"").unwrap();
        assert!(guard.require_live("p", "r").is_err());
        assert!(guard.mark("p", "r", &Map::new(), "now").is_err());
        assert!(!fence.path("p", "r").unwrap().exists());
    }

    #[test]
    fn existing_guards_never_create_missing_roots_locks_or_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let absent = temp.path().join("missing-root");
        let fence = RunFence::new(&absent);
        assert!(fence.shared_existing().is_err());
        assert!(fence.existing_observation_guard("p", "r").is_err());
        assert!(!absent.exists());
        let fence = RunFence::new(temp.path());
        assert!(fence.shared_existing().is_err());
        assert!(fence.existing_observation_guard("p", "r").is_err());
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);

        // A historical workspace can have a lock but no tombstone directory.
        // Reading it must not manufacture an empty deleted-runs hierarchy.
        fs::write(temp.path().join(LOCK), b"unchanged-lock").unwrap();
        let before = fs::metadata(temp.path().join(LOCK))
            .unwrap()
            .modified()
            .unwrap();
        let guard = fence.existing_observation_guard("p", "r").unwrap();
        guard.validate().unwrap();
        assert!(!guard.is_deleted("p", "r").unwrap());
        drop(guard);
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
        assert!(!temp.path().join("deleted-runs").exists());
        assert_eq!(fs::read(temp.path().join(LOCK)).unwrap(), b"unchanged-lock");
        assert_eq!(
            fs::metadata(temp.path().join(LOCK))
                .unwrap()
                .modified()
                .unwrap(),
            before
        );
    }

    #[test]
    fn existing_guards_reject_symlink_fifo_and_replaced_lock_without_repair() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        fs::create_dir(&root).unwrap();
        let target = temp.path().join("outside-lock");
        fs::write(&target, b"private-original").unwrap();
        symlink(&target, root.join(LOCK)).unwrap();
        let fence = RunFence::new(&root);
        assert!(fence.shared_existing().is_err());
        assert_eq!(fs::read(&target).unwrap(), b"private-original");
        assert!(!root.join("deleted-runs").exists());
        fs::remove_file(root.join(LOCK)).unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(root.join(LOCK))
                .status()
                .unwrap()
                .success()
        );
        assert!(fence.existing_observation_guard("p", "r").is_err());
        assert!(!root.join("deleted-runs").exists());
        fs::remove_file(root.join(LOCK)).unwrap();
        fs::write(root.join(LOCK), b"").unwrap();
        let guard = fence.shared_existing().unwrap();
        fs::rename(root.join(LOCK), root.join("old-lock")).unwrap();
        fs::write(root.join(LOCK), b"replacement").unwrap();
        assert!(guard.validate().is_err());
        assert_eq!(fs::read(root.join(LOCK)).unwrap(), b"replacement");
        assert!(!root.join("deleted-runs").exists());
    }

    #[test]
    fn existing_observers_respect_permanent_tombstones_without_writing() {
        let temp = tempfile::tempdir().unwrap();
        let writer = RunFence::new(temp.path());
        writer.mark("p", "r", &Map::new(), "first").unwrap();
        let path = writer.path("p", "r").unwrap();
        let original = fs::read(&path).unwrap();
        let before = fs::metadata(&path).unwrap().modified().unwrap();
        let reader = RunFence::new(temp.path());
        assert!(matches!(
            reader.existing_observation_guard("p", "r"),
            Err(Error::Deleted)
        ));
        reader
            .existing_observation_guard("p", "other")
            .unwrap()
            .validate()
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), before);
    }
}
