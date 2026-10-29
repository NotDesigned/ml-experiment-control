//! Provision only a frozen Run's directories below an existing approved mount.
//! Retained directory descriptors reject links and replaced ancestors. No chmod,
//! adoption of a dataset store, mount creation or provider effects occur here.
use crate::managed::{Error, Result};
use ml_exp_core::cloud::launch::Package;
use ml_exp_io::directory::Directory;
use std::path::{Component, Path};

fn descendant(anchor: &Directory, mount: &Path, path: &Path) -> Result<()> {
    let mut directory = anchor.clone();
    for component in path
        .strip_prefix(mount)
        .map_err(|_| Error::Configuration)?
        .components()
    {
        let Component::Normal(name) = component else {
            return Err(Error::Configuration);
        };
        directory = directory.child(name, true).map_err(|_| Error::Storage)?;
        directory
            .verify_private_parent()
            .map_err(|_| Error::Storage)?;
    }
    directory.verify_attached().map_err(|_| Error::Storage)
}

pub(crate) fn prepare(package: &Package) -> Result<()> {
    let Some(storage) = &package.storage else {
        return Ok(());
    };
    package.validate().map_err(|_| Error::Configuration)?;
    let mount = Directory::open(&storage.mount, false).map_err(|_| Error::Storage)?;
    descendant(&mount, &storage.mount, &storage.runs)?;
    // The dataset store creates and atomically publishes its own marker-bearing
    // root. Creating an empty root here would make that initialization ambiguous.
    prepare_data(storage)
}
pub(crate) fn prepare_data(storage: &ml_exp_core::cloud::launch::StorageLayout) -> Result<()> {
    storage.validate().map_err(|_| Error::Configuration)?;
    let mount = Directory::open(&storage.mount, false).map_err(|_| Error::Storage)?;
    descendant(
        &mount,
        &storage.mount,
        storage.datasets.parent().ok_or(Error::Configuration)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_core::cloud::launch::{self, StorageLayout};
    use std::{
        collections::BTreeMap,
        fs,
        os::unix::fs::{PermissionsExt, symlink},
        sync::atomic::AtomicBool,
    };

    fn package(mount: &Path) -> Package {
        let root = mount.join("native/runs");
        let mut value = launch::compile(launch::Input {
            root: root.to_str().unwrap(),
            project: "demo",
            run: "first",
            attempt: "attempt-001",
            workdir: mount.to_str().unwrap(),
            argv: &["/bin/true".into()],
            timeout_seconds: 3,
            worker_sha256: &format!("sha256:{}", "a".repeat(64)),
        })
        .unwrap();
        value.storage = Some(StorageLayout {
            mount: mount.into(),
            runs: root,
            datasets: mount.join("native/data/cache"),
        });
        value
    }

    #[test]
    fn fresh_mounted_workspace_provisions_runs_and_cache_parent_then_executes_once() {
        let temp = tempfile::tempdir().unwrap();
        let value = package(temp.path());
        let original = fs::metadata(temp.path()).unwrap().permissions().mode();
        let receipt = crate::cloud::run(&value, &BTreeMap::new(), &AtomicBool::new(false)).unwrap();
        assert_eq!(receipt.phase, crate::managed::Phase::Finished);
        assert!(value.root.is_dir());
        assert!(temp.path().join("native/data").is_dir());
        assert!(!value.storage.as_ref().unwrap().datasets.exists());
        assert_eq!(
            fs::metadata(temp.path()).unwrap().permissions().mode(),
            original
        );
        assert_eq!(
            crate::cloud::run(&value, &BTreeMap::new(), &AtomicBool::new(false)).unwrap(),
            receipt
        );
    }

    #[test]
    fn refuses_absent_mount_links_and_shared_writable_descendants_without_repair() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("missing-mount");
        assert!(prepare(&package(&missing)).is_err());
        assert!(!missing.exists());
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), temp.path().join("native")).unwrap();
        assert!(prepare(&package(temp.path())).is_err());
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
        fs::remove_file(temp.path().join("native")).unwrap();
        fs::create_dir(temp.path().join("native")).unwrap();
        fs::set_permissions(
            temp.path().join("native"),
            fs::Permissions::from_mode(0o777),
        )
        .unwrap();
        assert!(prepare(&package(temp.path())).is_err());
        assert!(!temp.path().join("native/runs").exists());
        assert_eq!(
            fs::metadata(temp.path().join("native"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o777
        );
    }

    #[test]
    fn old_package_keeps_existing_root_requirement_and_has_no_storage_field() {
        let temp = tempfile::tempdir().unwrap();
        let mut value = package(temp.path());
        value.storage = None;
        assert!(
            serde_json::to_value(&value)
                .unwrap()
                .get("storage")
                .is_none()
        );
        assert!(crate::cloud::prepare(&value, &BTreeMap::new()).is_err());
        assert!(!value.root.exists());
        let mut invalid = package(temp.path());
        invalid.storage.as_mut().unwrap().datasets = invalid.root.join("data");
        assert!(invalid.validate().is_err());
        invalid.storage.as_mut().unwrap().datasets = temp.path().join("../escape");
        assert!(invalid.validate().is_err());
    }
}
