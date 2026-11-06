use super::*;
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
};

fn initialize(db: &SqliteDatabase) {
    db.initialize(|| Ok::<_, Error>(()), |tx| {
        tx.execute_batch("CREATE TABLE IF NOT EXISTS values_(id INTEGER PRIMARY KEY,value TEXT); INSERT OR IGNORE INTO values_ VALUES(1,'initial');")?;
        Ok::<_, Error>(())
    }).unwrap();
}

#[test]
fn fixed_sqlite_and_wal_survive_reopen_without_readonly_namespace_writes() {
    assert!(rusqlite::version_number() >= 3_051_003);
    let temp = tempfile::tempdir().unwrap();
    let db = SqliteDatabase::new(temp.path().join("a ?mode=rw#.sqlite")).with_wal();
    initialize(&db);
    let names = fs::read_dir(temp.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    assert!(names.len() >= 3, "WAL and SHM stay with the database");
    let value = db
        .read_only(|connection| {
            Ok::<_, Error>(
                connection.query_row("SELECT value FROM values_", [], |row| {
                    row.get::<_, String>(0)
                })?,
            )
        })
        .unwrap();
    assert_eq!(value, "initial");
    let after = fs::read_dir(temp.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    assert_eq!(names, after);
    let fresh = SqliteDatabase::new(db.path());
    let value = fresh
        .read(|connection| {
            Ok::<_, Error>(
                connection.query_row("SELECT value FROM values_", [], |row| {
                    row.get::<_, String>(0)
                })?,
            )
        })
        .unwrap();
    assert_eq!(value, "initial");
}

#[test]
fn missing_storage_is_not_repaired_and_unsafe_sidecars_fail_promptly() {
    let temp = tempfile::tempdir().unwrap();
    let absent = temp.path().join("absent/db.sqlite");
    assert!(
        SqliteDatabase::new(&absent)
            .read_only(|_| Ok::<_, Error>(()))
            .is_err()
    );
    assert!(!temp.path().join("absent").exists());
    for suffix in ["", "-wal", "-shm", "-journal"] {
        let directory = tempfile::tempdir().unwrap();
        let db = SqliteDatabase::new(directory.path().join("db.sqlite"));
        initialize(&db);
        let target = directory.path().join(format!("db.sqlite{suffix}"));
        if target.exists() {
            fs::rename(&target, directory.path().join("original")).unwrap();
        }
        symlink("missing", &target).unwrap();
        assert!(db.read(|_| Ok::<_, Error>(())).is_err());
    }
    let directory = tempfile::tempdir().unwrap();
    let db = SqliteDatabase::new(directory.path().join("db.sqlite"));
    initialize(&db);
    fs::remove_file(db.path()).unwrap();
    assert!(matches!(
        db.initialize(|| Ok::<_, Error>(()), |_| Ok::<_, Error>(())),
        Err(Error::Missing)
    ));
    assert!(!db.path().exists());
}

#[test]
fn readonly_missing_shm_refuses_repair_and_creation_modes_are_explicit() {
    let temp = tempfile::tempdir().unwrap();
    let db = SqliteDatabase::new(temp.path().join("wal.sqlite")).with_wal();
    initialize(&db);
    assert_eq!(
        fs::metadata(db.path()).unwrap().permissions().mode() & 0o777,
        0o600
    );
    fs::remove_file(temp.path().join("wal.sqlite-shm")).unwrap();
    let result = db.read_only(|connection| {
        Ok::<_, Error>(
            connection.query_row("SELECT value FROM values_", [], |row| {
                row.get::<_, String>(0)
            })?,
        )
    });
    assert!(result.is_err());
    assert!(!temp.path().join("wal.sqlite-shm").exists());
    let delete_journal = SqliteDatabase::new(temp.path().join("delete.sqlite"));
    initialize(&delete_journal);
    assert_eq!(
        delete_journal
            .read(|connection| Ok::<_, Error>(connection.query_row(
                "PRAGMA journal_mode",
                [],
                |row| row.get::<_, String>(0)
            )?))
            .unwrap(),
        "delete"
    );
    assert!(!temp.path().join("delete.sqlite-wal").exists());
}

#[test]
fn pathname_identity_checks_do_not_release_sqlite_posix_locks() {
    const CHILD: &str = "ML_EXPD_SYNTHETIC_STORAGE_LOCK_PROBE";
    if let Some(path) = std::env::var_os(CHILD) {
        let connection = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )
        .unwrap();
        connection.busy_timeout(Duration::ZERO).unwrap();
        assert!(
            matches!(connection.execute_batch("BEGIN IMMEDIATE;"),Err(rusqlite::Error::SqliteFailure(error,_)) if error.code==rusqlite::ErrorCode::DatabaseBusy)
        );
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let db = SqliteDatabase::new(temp.path().join("db.sqlite"));
    initialize(&db);
    let connection = Connection::open(db.path()).unwrap();
    connection.execute_batch("BEGIN EXCLUSIVE;").unwrap();
    let directory = Directory::open(temp.path(), false).unwrap();
    for _ in 0..100 {
        assert!(
            directory
                .regular_identity(std::ffi::OsStr::new("db.sqlite"))
                .unwrap()
                .is_some()
        );
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "storage::tests::pathname_identity_checks_do_not_release_sqlite_posix_locks",
            "--nocapture",
        ])
        .env(CHILD, db.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    connection.execute_batch("ROLLBACK;").unwrap();
}
