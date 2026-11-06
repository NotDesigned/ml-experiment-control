use super::*;
use ml_exp_core::{datasets::Definition, slurm::helper::Target};
use sha2::{Digest, Sha256};
use std::fs;

struct Fixture {
    _temporary: tempfile::TempDir,
    config: Config,
    definition: Definition,
    payload: Vec<u8>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        use std::os::unix::fs::PermissionsExt;
        fn unseal(path: &std::path::Path) {
            let metadata = fs::symlink_metadata(path).unwrap();
            if metadata.file_type().is_symlink() {
                return;
            }
            fs::set_permissions(
                path,
                fs::Permissions::from_mode(if metadata.is_dir() { 0o700 } else { 0o600 }),
            )
            .unwrap();
            if metadata.is_dir() {
                for child in fs::read_dir(path).unwrap() {
                    unseal(&child.unwrap().path());
                }
            }
        }
        unseal(self._temporary.path());
    }
}
fn packed(contents: &[u8]) -> (Definition, Vec<u8>) {
    let source = tempfile::tempdir().unwrap();
    fs::write(source.path().join("tokens.bin"), contents).unwrap();
    let source = Directory::open(source.path(), false).unwrap();
    let mut file = tempfile::tempfile().unwrap();
    let (sha256, archive_bytes, files) =
        ml_exp_io::artifacts::pack_data(&source, &mut file).unwrap();
    file.rewind().unwrap();
    let mut payload = Vec::new();
    file.read_to_end(&mut payload).unwrap();
    (
        Definition {
            sha256,
            archive_bytes,
            files,
        },
        payload,
    )
}
impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("data");
        fs::create_dir(&root).unwrap();
        let (definition, payload) = packed(b"owned dataset fixture");
        let config = Config {
            root,
            target: Target {
                backend_id: "test-slurm".into(),
                profile_sha256: format!("sha256:{}", "1".repeat(64)),
                storage_scope: format!("sha256:{}", "2".repeat(64)),
            },
        };
        Store::initialize(config.clone()).unwrap();
        Self {
            _temporary: temporary,
            config,
            definition,
            payload,
        }
    }
    fn store(&self) -> Store {
        Store::open(self.config.clone()).unwrap()
    }
    fn request(&self, command: Command) -> Request {
        Request {
            version: 1,
            project: "trial".into(),
            target: self.config.target.clone(),
            command,
        }
    }
    fn invoke(&self, command: Command, payload: &[u8]) -> io::Result<Reply> {
        self.store()
            .handle(&self.request(command), payload, &AtomicBool::new(false))
    }
    fn begin(&self) -> State {
        self.invoke(
            Command::Begin {
                definition: self.definition.clone(),
            },
            &[],
        )
        .unwrap()
        .state
    }
    fn status(&self) -> State {
        self.invoke(
            Command::Status {
                asset_id: self.definition.asset_id(),
            },
            &[],
        )
        .unwrap()
        .state
    }
    fn upload(&self) -> State {
        self.invoke(
            Command::Part {
                asset_id: self.definition.asset_id(),
                number: 0,
                sha256: format!("{:x}", Sha256::digest(&self.payload)),
                bytes: self.payload.len() as u64,
            },
            &self.payload,
        )
        .unwrap()
        .state
    }
    fn complete(&self) -> io::Result<Reply> {
        self.invoke(
            Command::Complete {
                asset_id: self.definition.asset_id(),
            },
            &[],
        )
    }
    fn directory(&self) -> PathBuf {
        self.config
            .root
            .join("trial")
            .join(self.definition.asset_id())
    }
}

#[test]
fn resumable_parts_and_ready_receipt_survive_reopen_without_retaining_archives() {
    let f = Fixture::new();
    let empty = f.begin();
    assert!(f.complete().is_err());
    assert_eq!(f.status(), empty);
    let uploaded = f.upload();
    assert_eq!(f.upload(), uploaded);
    assert_eq!(f.begin(), uploaded);
    let ready = f.complete().unwrap().state;
    uploaded.validate_successor(&ready).unwrap();
    assert!(ready.ready.is_some());
    assert!(!f.directory().join("archive.tar").exists());
    assert_eq!(
        fs::read_dir(f.directory().join("parts")).unwrap().count(),
        0
    );
    let tree = f.directory().join("tree");
    assert_eq!(
        fs::read(tree.join("tokens.bin")).unwrap(),
        b"owned dataset fixture"
    );
    assert_eq!(f.complete().unwrap().state, ready);
    assert_eq!(f.upload(), ready);
    assert_eq!(f.begin(), ready);
    assert_eq!(f.status(), ready);
}

#[test]
fn invalid_payload_or_foreign_scope_does_not_record_parts() {
    let f = Fixture::new();
    let original = f.begin();
    let mut command = Command::Part {
        asset_id: f.definition.asset_id(),
        number: 0,
        sha256: "0".repeat(64),
        bytes: f.payload.len() as u64,
    };
    assert!(f.invoke(command.clone(), &f.payload).is_err());
    if let Command::Part { sha256, .. } = &mut command {
        *sha256 = format!("{:x}", Sha256::digest(&f.payload));
    }
    assert!(f.invoke(command.clone(), &f.payload[..1]).is_err());
    let mut request = f.request(command);
    request.target.storage_scope = format!("sha256:{}", "3".repeat(64));
    assert!(
        f.store()
            .handle(&request, &f.payload, &AtomicBool::new(false))
            .is_err()
    );
    let mut request = f.request(Command::Status {
        asset_id: f.definition.asset_id(),
    });
    request.project = "foreign".into();
    assert!(
        f.store()
            .handle(&request, &[], &AtomicBool::new(false))
            .is_err()
    );
    assert_eq!(f.status(), original);
}

#[test]
fn durable_part_without_state_commit_is_reconciled_and_changed_part_is_refused() {
    let f = Fixture::new();
    f.begin();
    fs::write(f.directory().join("parts/0"), &f.payload).unwrap();
    assert_eq!(f.upload().parts.len(), 1);
    let old = f.status();
    let mut changed = f.payload.clone();
    changed[0] ^= 1;
    assert!(
        f.invoke(
            Command::Part {
                asset_id: f.definition.asset_id(),
                number: 0,
                sha256: format!("{:x}", Sha256::digest(&changed)),
                bytes: changed.len() as u64,
            },
            &changed
        )
        .is_err()
    );
    assert_eq!(f.status(), old);
}

#[test]
fn whole_archive_or_member_digest_failure_never_publishes_ready() {
    for change_archive in [false, true] {
        let mut f = Fixture::new();
        if change_archive {
            f.definition.sha256 = "9".repeat(64);
        } else {
            f.definition.files[0].sha256 = "9".repeat(64);
        }
        f.begin();
        f.upload();
        assert!(f.complete().is_err());
        assert!(f.status().ready.is_none());
        assert!(!f.directory().join("tree").exists());
        assert!(f.directory().join("parts/0").is_file());
    }
}

#[test]
fn interrupted_extraction_and_lost_ready_receipt_reconcile_original_identity() {
    let f = Fixture::new();
    f.begin();
    let uploaded = f.upload();
    fs::create_dir(f.directory().join("unpacked")).unwrap();
    fs::write(f.directory().join("unpacked/incomplete"), b"owned scratch").unwrap();
    fs::write(f.directory().join("archive.tar"), b"incomplete tar").unwrap();
    let ready = f.complete().unwrap().state;
    // Simulate successful tree publication followed by loss of the state commit.
    let directory = Directory::open(&f.directory(), false).unwrap();
    Store::save(&directory, &uploaded).unwrap();
    assert_eq!(f.complete().unwrap().state, ready);
    assert!(!f.directory().join("unpacked").exists());
}

#[test]
fn ready_tree_is_never_replaced_when_files_or_permissions_change() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new();
    f.begin();
    f.upload();
    let original = f.complete().unwrap().state;
    let root = f.directory().join("tree");
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    assert!(f.complete().is_err());
    fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).unwrap();
    let file = root.join("tokens.bin");
    fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&file, b"changed dataset").unwrap();
    fs::set_permissions(&file, fs::Permissions::from_mode(0o400)).unwrap();
    assert!(f.complete().is_err());
    assert_eq!(f.status(), original);
    assert_eq!(fs::read(&file).unwrap(), b"changed dataset");
}

#[test]
fn explicit_initialization_and_root_marker_do_not_reinterpret_existing_storage() {
    let f = Fixture::new();
    assert!(Store::initialize(f.config.clone()).is_err());
    let mut foreign = f.config.clone();
    foreign.target.backend_id = "foreign".into();
    assert!(Store::open(foreign).is_err());
    let marker = f.config.root.join(MARKER);
    fs::remove_file(&marker).unwrap();
    std::os::unix::fs::symlink("/etc/passwd", &marker).unwrap();
    assert!(Store::open(f.config.clone()).is_err());
}

#[test]
fn shared_mount_device_changes_preserve_identity_but_not_replaced_roots() {
    let f = Fixture::new();
    f.begin();
    f.upload();
    let ready = f.complete().unwrap().state;
    let path = f.config.root.join(MARKER);
    let original = fs::read(&path).unwrap();
    let mut marker: Marker = serde_json::from_slice(&original).unwrap();
    marker.identity.0 ^= 1;
    fs::write(&path, json(&marker).unwrap()).unwrap();
    assert_eq!(f.complete().unwrap().state, ready);
    // A different root inode or logical storage authority remains invalid.
    marker.identity.1 ^= 1;
    fs::write(&path, json(&marker).unwrap()).unwrap();
    assert!(Store::open(f.config.clone()).is_err());
    fs::write(&path, original).unwrap();
    assert_eq!(f.complete().unwrap().state, ready);
}

#[test]
fn cancelled_delivery_does_not_create_an_asset() {
    let f = Fixture::new();
    let request = f.request(Command::Begin {
        definition: f.definition.clone(),
    });
    assert!(
        f.store()
            .handle(&request, &[], &AtomicBool::new(true))
            .is_err()
    );
    assert!(!f.config.root.join("trial").exists());
}

fn unseal(path: &std::path::Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}

#[test]
fn a_tree_a_killed_job_left_without_a_receipt_is_rebuilt_from_verified_parts() {
    let sealed = |tree: &std::path::Path, bytes: &[u8]| {
        fs::write(tree.join("tokens.bin"), bytes).unwrap();
        unseal(&tree.join("tokens.bin"), 0o400);
    };
    let leftovers: [&dyn Fn(&std::path::Path); 5] = [
        // Cut short by the kill: nothing yet, then part of a file.
        &|_| {},
        &|tree| fs::write(tree.join("tokens.bin"), b"owned").unwrap(),
        // Every byte is there, but sealing never ran.
        &|tree| fs::write(tree.join("tokens.bin"), b"owned dataset fixture").unwrap(),
        // Sealed, and still not what the definition says.
        &|tree| {
            sealed(tree, b"owned dataset fixturE");
            unseal(tree, 0o500);
        },
        // A file nothing in the definition names, inside a sealed directory.
        &|tree| {
            sealed(tree, b"owned dataset fixture");
            fs::create_dir(tree.join("extra")).unwrap();
            fs::write(tree.join("extra/left"), b"x").unwrap();
            unseal(&tree.join("extra"), 0o500);
            unseal(tree, 0o500);
        },
    ];
    for leave in leftovers {
        let f = Fixture::new();
        f.begin();
        let uploaded = f.upload();
        let tree = f.directory().join("tree");
        fs::create_dir(&tree).unwrap();
        leave(&tree);
        let ready = f.complete().unwrap().state;
        uploaded.validate_successor(&ready).unwrap();
        assert!(ready.ready.is_some());
        assert_eq!(
            fs::read(tree.join("tokens.bin")).unwrap(),
            b"owned dataset fixture"
        );
        assert!(!tree.join("extra").exists());
        assert_eq!(
            fs::read_dir(f.directory().join("parts")).unwrap().count(),
            0
        );
        assert_eq!(f.complete().unwrap().state, ready);
    }
}

#[test]
fn a_leftover_tree_stays_until_the_parts_prove_the_archive_and_the_parts_outlive_a_failure() {
    let f = Fixture::new();
    f.begin();
    let uploaded = f.upload();
    let tree = f.directory().join("tree");
    fs::create_dir(&tree).unwrap();
    fs::write(tree.join("partial"), b"left by a kill").unwrap();
    let part = f.directory().join("parts/0");
    let original = fs::read(&part).unwrap();
    unseal(&part, 0o600);
    let mut changed = original.clone();
    changed[0] ^= 1;
    fs::write(&part, changed).unwrap();
    assert!(f.complete().is_err());
    assert_eq!(fs::read(tree.join("partial")).unwrap(), b"left by a kill");
    assert_eq!(f.status(), uploaded);
    // With the right bytes back, the same Complete proves them and rebuilds.
    fs::write(&part, original).unwrap();
    assert!(f.complete().unwrap().state.ready.is_some());
    assert!(!tree.join("partial").exists());
}

#[test]
fn a_rebuild_that_fails_verification_keeps_its_parts_and_publishes_nothing() {
    let mut f = Fixture::new();
    f.definition.files[0].sha256 = "9".repeat(64);
    f.begin();
    f.upload();
    let tree = f.directory().join("tree");
    fs::create_dir(&tree).unwrap();
    fs::write(tree.join("partial"), b"left by a kill").unwrap();
    assert!(f.complete().is_err());
    assert!(f.status().ready.is_none());
    assert!(!tree.exists());
    assert!(f.directory().join("parts/0").is_file());
    assert!(f.complete().is_err());
    assert!(f.directory().join("parts/0").is_file());
}

#[test]
fn only_a_tree_without_a_receipt_can_be_discarded() {
    let f = Fixture::new();
    f.begin();
    f.upload();
    let request = f.request(Command::Status {
        asset_id: f.definition.asset_id(),
    });
    let directory = Directory::open(&f.directory(), false).unwrap();
    fs::create_dir(f.directory().join("tree")).unwrap();
    f.store().discard_unpublished(&request, &directory).unwrap();
    assert!(!f.directory().join("tree").exists());
    let ready = f.complete().unwrap().state;
    assert!(ready.ready.is_some());
    assert!(f.store().discard_unpublished(&request, &directory).is_err());
    assert_eq!(
        fs::read(f.directory().join("tree/tokens.bin")).unwrap(),
        b"owned dataset fixture"
    );
}

/// What another node could do to a finished tree if flock did not exclude it.
#[test]
fn a_tree_removed_or_replaced_after_extraction_is_not_accepted() {
    let f = Fixture::new();
    f.begin();
    f.upload();
    f.complete().unwrap();
    let directory = Directory::open(&f.directory(), false).unwrap();
    let tree = directory.child(OsStr::new("tree"), false).unwrap();
    let path = f.directory().join("tree");
    Store::still_published(&directory, &tree, &f.definition).unwrap();

    let moved = f.directory().join("moved");
    fs::rename(&path, &moved).unwrap();
    assert!(Store::still_published(&directory, &tree, &f.definition).is_err());
    fs::create_dir(&path).unwrap();
    assert!(Store::still_published(&directory, &tree, &f.definition).is_err());
    fs::remove_dir(&path).unwrap();
    fs::rename(&moved, &path).unwrap();
    Store::still_published(&directory, &tree, &f.definition).unwrap();

    unseal(&path, 0o700);
    fs::remove_file(path.join("tokens.bin")).unwrap();
    unseal(&path, 0o500);
    assert!(Store::still_published(&directory, &tree, &f.definition).is_err());
}

#[test]
fn a_long_import_of_one_asset_does_not_hold_up_another() {
    let f = Fixture::new();
    f.begin();
    assert!(!f.config.root.join(".native-data.lock").exists());
    let project = Directory::open(&f.config.root.join("trial"), false).unwrap();
    let held = format!("{}.lock", f.definition.asset_id());
    let _held = project
        .lock(OsStr::new(&held), LockOptions::default())
        .unwrap();
    let (other, payload) = packed(b"another dataset");
    let start = Instant::now();
    assert_eq!(
        f.invoke(
            Command::Begin {
                definition: other.clone()
            },
            &[]
        )
        .unwrap()
        .state
        .definition,
        other
    );
    assert!(
        f.invoke(
            Command::Part {
                asset_id: other.asset_id(),
                number: 0,
                sha256: format!("{:x}", Sha256::digest(&payload)),
                bytes: payload.len() as u64,
            },
            &payload
        )
        .is_ok()
    );
    let ready = f
        .invoke(
            Command::Complete {
                asset_id: other.asset_id(),
            },
            &[],
        )
        .unwrap();
    assert!(ready.state.ready.is_some());
    assert!(start.elapsed() < Duration::from_secs(5));
    // The same asset does wait; ending the wait early keeps this test short.
    let cancelled = AtomicBool::new(false);
    let request = f.request(Command::Status {
        asset_id: f.definition.asset_id(),
    });
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(100));
            cancelled.store(true, Ordering::Relaxed);
        });
        assert!(f.store().handle(&request, &[], &cancelled).is_err());
    });
    // A query for an asset nobody began allocates nothing.
    let (unknown, _) = packed(b"never uploaded");
    assert!(
        f.invoke(
            Command::Status {
                asset_id: unknown.asset_id()
            },
            &[]
        )
        .is_err()
    );
    let name = format!("{}.lock", unknown.asset_id());
    assert!(!f.config.root.join("trial").join(name).exists());
}

/// Otherwise a batch of Runs reusing one READY asset time each other out on the hash.
#[test]
fn a_published_tree_is_hashed_after_the_lock_is_released() {
    let f = Fixture::new();
    f.begin();
    f.upload();
    let ready = f.complete().unwrap().state;
    fs::write(f.directory().join("archive.tar"), b"left by a kill").unwrap();
    let request = f.request(Command::Complete {
        asset_id: f.definition.asset_id(),
    });
    let file = f.directory().join("tree/tokens.bin");
    unseal(&file, 0o600);
    fs::write(&file, b"owned dataset fixturE").unwrap();
    unseal(&file, 0o400);
    let (state, published) = f
        .store()
        .exclusive(&request, &[], &AtomicBool::new(false))
        .unwrap();
    assert_eq!(state, ready);
    assert!(published.is_some());
    assert!(!f.directory().join("archive.tar").exists());
    assert!(f.complete().is_err());
    assert_eq!(f.status(), ready);
    assert_eq!(fs::read(&file).unwrap(), b"owned dataset fixturE");
}
