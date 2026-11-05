//! Filesystem acceptance for the native immutable source store.
use flate2::{Compression, write::GzEncoder};
use ml_exp_core::sources::Policy;
use ml_exp_server::sources::{Error, Store};
use sha2::{Digest, Sha256};
use std::{fs, io::Cursor, os::unix::fs::PermissionsExt, path::Path, sync::Barrier, thread};
use tar::{Builder, Header};

fn archive(include_secret: bool) -> (Vec<u8>, String) {
    let mut compressed = GzEncoder::new(Vec::new(), Compression::default());
    {
        let mut builder = Builder::new(&mut compressed);
        let mut add = |path: &str, body: &[u8], mode: u32| {
            let mut header = Header::new_gnu();
            header.set_path(path).unwrap();
            header.set_size(body.len() as u64);
            header.set_mode(mode);
            header.set_cksum();
            builder.append(&header, body).unwrap();
        };
        add("train.py", b"print('native')\n", 0o500);
        if include_secret {
            add(".env", b"secret", 0o400);
        }
        builder.finish().unwrap();
    }
    let bytes = compressed.finish().unwrap();
    let digest = format!("{:x}", Sha256::digest(&bytes));
    (bytes, digest)
}

#[test]
fn publish_read_reuse_and_archive_receipt_are_durable() {
    let temp = tempfile::tempdir().unwrap();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
    Store::initialize(temp.path(), Policy::default()).unwrap();
    let store = Store::open(temp.path(), Policy::default()).unwrap();
    let (bytes, digest) = archive(false);
    let mut first = Cursor::new(bytes.clone());
    let metadata = store
        .import_archive("demo", &mut first, &digest, "2026-10-28T00:00:00Z")
        .unwrap();
    assert!(metadata.source_id().starts_with("source."));
    assert_eq!(
        store
            .find_archive("demo", &digest)
            .unwrap()
            .unwrap()
            .source_id(),
        metadata.source_id()
    );
    let mut second = Cursor::new(bytes);
    let reused = store
        .import_archive("demo", &mut second, &digest, "2026-10-28T00:01:00Z")
        .unwrap();
    assert_eq!(reused.to_value(), metadata.to_value());
    assert!(
        fs::read_dir(temp.path().join("source-revisions/staging"))
            .unwrap()
            .next()
            .is_none()
    );
    assert!(
        !temp
            .path()
            .join(format!(
                "source-revisions/sources/demo/{}/archive.bin",
                metadata.source_id()
            ))
            .exists()
    );
    let source_path = temp.path().join(format!(
        "source-revisions/sources/demo/{}",
        metadata.source_id()
    ));
    let names: Vec<_> = fs::read_dir(&source_path)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert!(
        names
            .iter()
            .any(|name| name == Path::new("tree").as_os_str())
    );
    assert!(
        names
            .iter()
            .any(|name| name == Path::new("source.json").as_os_str())
    );
    assert_eq!(
        fs::metadata(source_path.join("source.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o400
    );
    let mut other_project = Cursor::new(archive(false).0);
    let other = store
        .import_archive("other", &mut other_project, &digest, "2026-10-28T00:02:00Z")
        .unwrap();
    assert_eq!(other.source_id(), metadata.source_id());
    assert!(store.read("other", other.source_id()).is_ok());
    let reopened = Store::open(temp.path(), Policy::default()).unwrap();
    assert_eq!(
        reopened
            .read("demo", metadata.source_id())
            .unwrap()
            .to_value(),
        metadata.to_value()
    );
}

#[test]
fn concurrent_imports_of_one_archive_publish_once_and_leave_no_stage() {
    let temp = tempfile::tempdir().unwrap();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
    Store::initialize(temp.path(), Policy::default()).unwrap();
    let store = Store::open(temp.path(), Policy::default()).unwrap();
    let (bytes, digest) = archive(false);
    let start = Barrier::new(8);
    let identities: Vec<String> = thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    let mut stream = Cursor::new(bytes.clone());
                    start.wait();
                    store
                        .import_archive("demo", &mut stream, &digest, "2026-10-28T00:00:00Z")
                        .unwrap()
                        .source_id()
                        .to_owned()
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    assert!(identities.iter().all(|identity| identity == &identities[0]));
    let published = fs::read_dir(temp.path().join("source-revisions/sources/demo")).unwrap();
    assert_eq!(published.count(), 1);
    let staged = fs::read_dir(temp.path().join("source-revisions/staging")).unwrap();
    assert_eq!(staged.count(), 0);
    let found = store.find_archive("demo", &digest).unwrap().unwrap();
    assert_eq!(found.source_id(), identities[0]);
}

#[test]
fn only_receipts_and_publication_wait_for_the_source_lock() {
    let temp = tempfile::tempdir().unwrap();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
    Store::initialize(temp.path(), Policy::default()).unwrap();
    let store = Store::open(temp.path(), Policy::default()).unwrap();
    let (bytes, digest) = archive(false);
    let source_id = store
        .import_archive(
            "demo",
            &mut Cursor::new(bytes),
            &digest,
            "2026-10-28T00:00:00Z",
        )
        .unwrap()
        .source_id()
        .to_owned();
    let held = fs::File::open(temp.path().join("source-revisions/.sources.lock")).unwrap();
    fs2::FileExt::lock_exclusive(&held).unwrap();
    assert!(store.read("demo", &source_id).is_ok());
    assert!(matches!(
        store.find_archive("demo", &digest),
        Err(Error::Busy)
    ));
    fs2::FileExt::unlock(&held).unwrap();
    assert!(store.find_archive("demo", &digest).unwrap().is_some());
}

#[test]
fn explicit_filename_choice_is_published_without_secret_heuristics() {
    let temp = tempfile::tempdir().unwrap();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
    Store::initialize(temp.path(), Policy::default()).unwrap();
    let store = Store::open(temp.path(), Policy::default()).unwrap();
    let (bytes, digest) = archive(true);
    let mut input = Cursor::new(bytes);
    assert!(
        store
            .import_archive("demo", &mut input, &digest, "2026-10-28T00:00:00Z")
            .is_ok()
    );
    assert!(store.find_archive("demo", &digest).unwrap().is_some());
}

#[test]
fn writable_or_mismatched_receipts_are_not_accepted() {
    let temp = tempfile::tempdir().unwrap();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
    Store::initialize(temp.path(), Policy::default()).unwrap();
    let store = Store::open(temp.path(), Policy::default()).unwrap();
    let (bytes, digest) = archive(false);
    let mut input = Cursor::new(bytes);
    let metadata = store
        .import_archive("demo", &mut input, &digest, "2026-10-28T00:00:00Z")
        .unwrap();
    let receipt_path = temp
        .path()
        .join(format!("source-revisions/imports/demo/{digest}"));
    fs::set_permissions(&receipt_path, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(store.find_archive("demo", &digest).is_err());
    fs::set_permissions(&receipt_path, fs::Permissions::from_mode(0o400)).unwrap();
    let mut receipt =
        serde_json::from_slice::<serde_json::Value>(&fs::read(&receipt_path).unwrap()).unwrap();
    receipt["tree_digest"] = serde_json::Value::String("sha256:00".into());
    fs::set_permissions(&receipt_path, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
    fs::set_permissions(&receipt_path, fs::Permissions::from_mode(0o400)).unwrap();
    assert!(store.find_archive("demo", &digest).is_err());
    assert!(store.read("demo", metadata.source_id()).is_ok());
}
