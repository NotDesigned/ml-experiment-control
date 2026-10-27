use super::*;
use ml_exp_core::operations::{self, Dispatch, Policy, Scope};
use serde_json::{Value, json};
use std::{
    fs,
    sync::{Arc, Barrier},
};
const NOW: &str = "2026-10-27T00:00:00Z";
fn uid(value: u32) -> String {
    format!("{value:032x}")
}
fn golden() -> Value {
    serde_json::from_str(include_str!("fixtures/python-actions-v1.json")).unwrap()
}
fn policy() -> Policy {
    Policy {
        allow_project_writes: true,
        ..Default::default()
    }
}

#[test]
fn python_prepared_authorized_claimed_snapshots_are_identical() {
    let golden = golden();
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let plan = &golden["plan"];
    let id = plan["action_id"].as_str().unwrap();
    assert_eq!(
        store.save_plan(plan, NOW, &uid(1)).unwrap(),
        golden["prepared"]
    );
    assert_eq!(
        store
            .authorize(id, "actor:test", "approved explicitly", NOW, &uid(2))
            .unwrap(),
        golden["authorized"]
    );
    let claimed = store
        .claim(
            id,
            &format!("EXECUTE {id}"),
            &policy(),
            &Default::default(),
            ClaimContext {
                now: NOW,
                transition_id: &uid(3),
                pid: 123,
            },
        )
        .unwrap();
    assert_eq!(claimed.dispatch, Some(Dispatch::ProjectWrite));
    assert_eq!(claimed.snapshot, golden["claimed"]);
    for case in golden["action_id_cases"].as_array().unwrap() {
        let scope: Scope = serde_json::from_value(case["scope"].clone()).unwrap();
        assert_eq!(
            operations::action_id(&scope, case["intent_key"].as_str().unwrap()),
            case["action_id"].as_str().unwrap()
        );
    }
    let saved = fs::read(store.directory(id).unwrap().join("plan.json")).unwrap();
    let mut changed = plan.clone();
    changed["request_digest"] = json!("sha256:different");
    assert!(matches!(
        store.save_plan(&changed, NOW, &uid(4)),
        Err(Error::IdempotencyConflict)
    ));
    assert_eq!(
        fs::read(store.directory(id).unwrap().join("plan.json")).unwrap(),
        saved
    );
}

#[test]
fn competing_claims_dispatch_exactly_once_and_restart_only_reconciles() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let golden = golden();
    let plan = &golden["plan"];
    let id = plan["action_id"].as_str().unwrap();
    store.save_plan(plan, NOW, &uid(1)).unwrap();
    store
        .authorize(id, "actor:test", "approved", NOW, &uid(2))
        .unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let mut threads = Vec::new();
    for index in 0..2 {
        let root = root.path().to_owned();
        let barrier = barrier.clone();
        let id = id.to_owned();
        threads.push(std::thread::spawn(move || {
            let store = Store::open(root).unwrap();
            barrier.wait();
            store.claim(
                &id,
                &format!("EXECUTE {id}"),
                &policy(),
                &Default::default(),
                ClaimContext {
                    now: NOW,
                    transition_id: &uid(index + 3),
                    pid: 123,
                },
            )
        }));
    }
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        store.database().journal(&|| Ok(()), id, 100).unwrap().len(),
        3
    );
    let reopened = Store::open(root.path()).unwrap();
    let recovered = reopened.recover_interrupted(NOW, &uid(6)).unwrap();
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0]["execution"]["status"], "RECONCILE_REQUIRED");
    assert_eq!(recovered[0]["execution"]["safe_to_retry"], false);
    assert!(
        reopened
            .recover_interrupted(NOW, &uid(7))
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        reopened.claim(
            id,
            &format!("EXECUTE {id}"),
            &policy(),
            &Default::default(),
            ClaimContext {
                now: NOW,
                transition_id: &uid(8),
                pid: 123
            }
        ),
        Err(Error::Decision(operations::Error::State(_)))
    ));
}

#[test]
fn sqlite_cas_and_failed_audit_insert_leave_state_and_events_atomic() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let db = store.database();
    let id = "action-0123456789abcdef";
    let first = db
        .commit(
            &|| Ok(()),
            id,
            &json!({"revision":0,"status":"PREPARED"}),
            &json!({"event":"prepared"}),
            None,
            &uid(1),
        )
        .unwrap();
    let mut authorized = first.clone();
    authorized["status"] = json!("AUTHORIZED");
    // The duplicate audit identity fails *after* the state UPSERT; transaction
    // rollback must leave neither a newer revision nor a partial audit event.
    assert!(matches!(
        db.commit(
            &|| Ok(()),
            id,
            &authorized,
            &json!({"event":"authorized"}),
            Some("PREPARED"),
            &uid(1)
        ),
        Err(Error::Sqlite(_))
    ));
    assert_eq!(db.read(&|| Ok(()), id).unwrap(), first);
    assert_eq!(db.journal(&|| Ok(()), id, 100).unwrap().len(), 1);
    let barrier = Arc::new(Barrier::new(2));
    let mut threads = Vec::new();
    for index in 0..2 {
        let db = db.clone();
        let authorized = authorized.clone();
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            db.commit(
                &|| Ok(()),
                id,
                &authorized,
                &json!({"event":"authorized"}),
                Some("PREPARED"),
                &uid(index + 2),
            )
        }));
    }
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(db.read(&|| Ok(()), id).unwrap()["revision"], 2);
    assert_eq!(db.journal(&|| Ok(()), id, 100).unwrap().len(), 2);
}

#[test]
fn blocked_action_authorization_or_claim_cannot_outlive_its_lock_or_directories() {
    use fs2::FileExt;
    use std::{
        fs::OpenOptions,
        time::{Duration, Instant},
    };
    for phase in ["authorize", "claim"] {
        for replacement in ["lock", "root", "action"] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().join("actions");
            let store = Store::open(&root).unwrap();
            let plan = golden()["plan"].clone();
            let id = plan["action_id"].as_str().unwrap().to_owned();
            store.save_plan(&plan, NOW, &uid(1)).unwrap();
            if phase == "claim" {
                store
                    .authorize(&id, "actor:test", "approved", NOW, &uid(2))
                    .unwrap();
            }
            let connection = rusqlite::Connection::open(store.database().path()).unwrap();
            let before: String = connection
                .query_row(
                    "SELECT payload FROM executions WHERE action_id=?",
                    [&id],
                    |row| row.get(0),
                )
                .unwrap();
            connection.execute_batch("BEGIN IMMEDIATE;").unwrap();
            let worker = store.clone();
            let action = id.clone();
            let confirmation = format!("EXECUTE {id}");
            let task = std::thread::spawn(move || {
                if phase == "authorize" {
                    worker
                        .authorize(&action, "actor:test", "approved", NOW, &uid(3))
                        .map(|_| ())
                } else {
                    worker
                        .claim(
                            &action,
                            &confirmation,
                            &policy(),
                            &Default::default(),
                            ClaimContext {
                                now: NOW,
                                transition_id: &uid(3),
                                pid: 123,
                            },
                        )
                        .map(|_| ())
                }
            });
            let start = Instant::now();
            loop {
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(root.join(".actions.lock"))
                    .unwrap();
                match file.try_lock_exclusive() {
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                    Ok(()) => drop(file),
                    Err(error) => panic!("unexpected lock error: {error}"),
                }
                assert!(start.elapsed() < Duration::from_secs(5));
                std::thread::sleep(Duration::from_millis(5));
            }
            std::thread::sleep(Duration::from_millis(20));
            match replacement {
                "lock" => {
                    fs::rename(root.join(".actions.lock"), root.join("old-lock")).unwrap();
                    fs::write(root.join(".actions.lock"), b"").unwrap();
                }
                "root" => {
                    fs::rename(&root, temp.path().join("detached")).unwrap();
                    fs::create_dir(&root).unwrap();
                }
                "action" => {
                    fs::rename(root.join(&id), root.join("detached-action")).unwrap();
                    fs::create_dir(root.join(&id)).unwrap();
                }
                _ => unreachable!(),
            }
            connection.execute_batch("ROLLBACK;").unwrap();
            assert!(
                task.join().unwrap().is_err(),
                "{phase}/{replacement} returned success"
            );
            let after: String = connection
                .query_row(
                    "SELECT payload FROM executions WHERE action_id=?",
                    [&id],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(
                before, after,
                "{phase}/{replacement} changed durable execution"
            );
            if replacement == "root" {
                assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
            }
        }
    }
}

#[test]
fn fixed_command_and_activity_directories_never_adopt_a_replacement() {
    for name in ["commands", "activity_errors"] {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path()).unwrap();
        let plan = golden()["plan"].clone();
        let id = plan["action_id"].as_str().unwrap();
        store.save_plan(&plan, NOW, &uid(1)).unwrap();
        if name == "commands" {
            store
                .write_command(
                    id,
                    CommandPhase::Authorize,
                    &json!({"note":"original"}),
                    &uid(2),
                    NOW,
                )
                .unwrap();
        } else {
            store
                .write_activity_error(id, "prepare", "original", "test", NOW)
                .unwrap();
        }
        let path = store.directory(id).unwrap().join(name);
        fs::rename(&path, temp.path().join("detached")).unwrap();
        fs::create_dir(&path).unwrap();
        if name == "commands" {
            fs::write(
                path.join("authorize.json"),
                "{\"payload\":{\"note\":\"replacement\"}}",
            )
            .unwrap();
            assert!(store.read_command(id, CommandPhase::Authorize).is_err());
        } else {
            fs::write(path.join("prepare.json"), "{\"message\":\"replacement\"}").unwrap();
            assert!(store.activity_error(id, "prepare").is_err());
        }
    }
}

#[test]
fn missing_action_lock_is_never_recreated_after_bootstrap_or_on_reopen() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let plan = golden()["plan"].clone();
    let id = plan["action_id"].as_str().unwrap();
    store.save_plan(&plan, NOW, &uid(1)).unwrap();
    store
        .authorize(id, "actor:test", "approved", NOW, &uid(2))
        .unwrap();
    let lock = root.path().join(".actions.lock");
    fs::remove_file(&lock).unwrap();
    assert!(store.snapshot(id, NOW, &uid(3)).is_err());
    assert!(store.save_plan(&plan, NOW, &uid(3)).is_err());
    assert!(
        store
            .claim(
                id,
                &format!("EXECUTE {id}"),
                &policy(),
                &Default::default(),
                ClaimContext {
                    now: NOW,
                    transition_id: &uid(3),
                    pid: 123
                }
            )
            .is_err()
    );
    assert!(Store::open(root.path()).is_err());
    assert!(!lock.exists());
}

#[test]
fn a_store_lifetime_pins_the_original_metadata_lock_inode() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let plan = golden()["plan"].clone();
    let id = plan["action_id"].as_str().unwrap();
    store.save_plan(&plan, NOW, &uid(1)).unwrap();
    fs::rename(
        root.path().join(".actions.lock"),
        root.path().join("old-lock"),
    )
    .unwrap();
    fs::write(root.path().join(".actions.lock"), b"").unwrap();
    assert!(store.snapshot(id, NOW, &uid(2)).is_err());
    assert!(store.save_plan(&plan, NOW, &uid(2)).is_err());
    assert!(
        store
            .authorize(id, "actor:test", "approved", NOW, &uid(2))
            .is_err()
    );
    let payload: String = rusqlite::Connection::open(store.database().path())
        .unwrap()
        .query_row(
            "SELECT payload FROM executions WHERE action_id=?",
            [id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&payload).unwrap()["status"],
        "PREPARED"
    );
}

#[test]
fn journal_deduplicates_and_commands_never_modify_immutable_plan() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let golden = golden();
    let plan = &golden["plan"];
    let id = plan["action_id"].as_str().unwrap();
    store.save_plan(plan, NOW, &uid(1)).unwrap();
    let original = fs::read(store.directory(id).unwrap().join("plan.json")).unwrap();
    let payload = json!({"note":"private freeform note"});
    let command = store
        .write_command(id, CommandPhase::Authorize, &payload, "command-1", NOW)
        .unwrap();
    assert_eq!(
        store
            .write_command(id, CommandPhase::Authorize, &payload, "ignored-new-id", NOW)
            .unwrap(),
        command
    );
    assert_eq!(
        store.read_command(id, CommandPhase::Authorize).unwrap(),
        payload
    );
    assert!(matches!(
        store.write_command(
            id,
            CommandPhase::Authorize,
            &json!({"note":"changed"}),
            "command-2",
            NOW
        ),
        Err(Error::CommandConflict)
    ));
    assert_eq!(
        store.database().journal(&|| Ok(()), id, 100).unwrap().len(),
        2
    );
    assert_eq!(
        fs::read(store.directory(id).unwrap().join("plan.json")).unwrap(),
        original
    );
    assert!(
        !String::from_utf8(original)
            .unwrap()
            .contains("private freeform note")
    );
    store
        .append_journal(
            id,
            "progress",
            &json!({"phase":"STAGING"}),
            "progress-1",
            NOW,
        )
        .unwrap();
    store
        .append_journal(
            id,
            "progress",
            &json!({"phase":"different retry"}),
            "progress-1",
            NOW,
        )
        .unwrap();
    let journal = store.database().journal(&|| Ok(()), id, 1).unwrap();
    assert_eq!(journal.len(), 1);
    assert_eq!(journal[0]["payload"]["phase"], "STAGING");
    store
        .write_activity_error(id, "execute", &"x".repeat(2000), "TRANSPORT", NOW)
        .unwrap();
    assert_eq!(
        store.activity_error(id, "execute").unwrap()["message"]
            .as_str()
            .unwrap()
            .len(),
        1000
    );
    assert!(store.activity_error(id, "../leak").is_err());
}

#[test]
fn command_replays_follow_python_numeric_equality_without_rewriting_frozen_bytes() {
    for (original, equivalent) in [
        (json!({"value":1}), json!({"value":1.0})),
        (
            json!({"value":true,"nested":[false,{"x":1}]}),
            json!({"nested":[0.0,{"x":true}],"value":1.0}),
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path()).unwrap();
        let plan = golden()["plan"].clone();
        let id = plan["action_id"].as_str().unwrap();
        store.save_plan(&plan, NOW, &uid(1)).unwrap();
        let plan_path = store.directory(id).unwrap().join("plan.json");
        let frozen_plan = fs::read(&plan_path).unwrap();
        let first = store
            .write_command(id, CommandPhase::Authorize, &original, "first-command", NOW)
            .unwrap();
        let command_path = store.directory(id).unwrap().join("commands/authorize.json");
        let frozen_command = fs::read(&command_path).unwrap();
        assert_eq!(
            store
                .write_command(
                    id,
                    CommandPhase::Authorize,
                    &equivalent,
                    "ignored-replay-id",
                    "2026-10-27T00:01:00Z"
                )
                .unwrap(),
            first,
        );
        assert_eq!(
            store.read_command(id, CommandPhase::Authorize).unwrap(),
            original
        );
        let mut different = equivalent;
        different["value"] = json!(2);
        assert!(matches!(
            store.write_command(
                id,
                CommandPhase::Authorize,
                &different,
                "different-command",
                NOW
            ),
            Err(Error::CommandConflict),
        ));
        assert_eq!(
            store.database().journal(&|| Ok(()), id, 100).unwrap().len(),
            2
        );
        assert_eq!(fs::read(&command_path).unwrap(), frozen_command);
        assert_eq!(fs::read(&plan_path).unwrap(), frozen_plan);
    }
}

#[test]
fn marker_missing_database_legacy_state_and_schema_drift_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let id = "action-0123456789abcdef";
    store.database().read(&|| Ok(()), id).unwrap();
    fs::remove_file(store.database().path()).unwrap();
    assert!(matches!(
        store.database().read(&|| Ok(()), id),
        Err(Error::MissingAuthoritative)
    ));
    assert!(!store.database().path().exists());
    let other = tempfile::tempdir().unwrap();
    let store = Store::open(other.path()).unwrap();
    fs::create_dir(store.directory(id).unwrap()).unwrap();
    fs::write(
        store.directory(id).unwrap().join("execution.json"),
        "{\"status\":\"AUTHORIZED\"}",
    )
    .unwrap();
    assert!(matches!(
        store.database().read(&|| Ok(()), id),
        Err(Error::LegacyMigration)
    ));
    let dbpath = store.database().path();
    let connection = rusqlite::Connection::open(dbpath).unwrap();
    connection.execute_batch("PRAGMA user_version=9;").unwrap();
    drop(connection);
    assert!(matches!(
        store.database().read(&|| Ok(()), id),
        Err(Error::UnsupportedSchema)
    ));
}

#[test]
fn revision_mismatch_and_corrupted_json_are_never_repaired_silently() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let id = "action-0123456789abcdef";
    store.database().read(&|| Ok(()), id).unwrap();
    let connection = rusqlite::Connection::open(store.database().path()).unwrap();
    connection
        .execute(
            "INSERT INTO executions VALUES (?,2,?)",
            rusqlite::params![id, "{\"revision\":1,\"status\":\"EXECUTING\"}"],
        )
        .unwrap();
    drop(connection);
    assert!(matches!(
        store.database().read(&|| Ok(()), id),
        Err(Error::InvalidRevision)
    ));
}

#[test]
fn reads_python_sqlite_rows_without_rewriting_plans_or_existing_payload_bytes() {
    let root = tempfile::tempdir().unwrap();
    let golden = golden();
    let id = golden["plan"]["action_id"].as_str().unwrap();
    // The old Python service created its metadata flock before publishing
    // plans/SQLite. Reconstruct that empty non-payload leaf with the fixture.
    drop(
        ml_exp_io::directory::Directory::open(root.path(), false)
            .unwrap()
            .open_lock(std::ffi::OsStr::new(".actions.lock"))
            .unwrap(),
    );
    fs::create_dir(root.path().join(id)).unwrap();
    let plan_bytes = serde_json::to_vec_pretty(&golden["plan"]).unwrap();
    fs::write(root.path().join(id).join("plan.json"), &plan_bytes).unwrap();
    let connection = rusqlite::Connection::open(root.path().join("actions.sqlite3")).unwrap();
    // SQLite iterdump orders table names, so FK targets may be created after
    // their referencing rows. Disable checks only for fixture import; the
    // production repository enables FK checks on every new connection.
    connection
        .execute_batch("PRAGMA foreign_keys=OFF;")
        .unwrap();
    connection
        .execute_batch(include_str!("fixtures/python-actions-v1.sql"))
        .unwrap();
    let before: String = connection
        .query_row(
            "SELECT payload FROM executions WHERE action_id=?",
            [id],
            |row| row.get(0),
        )
        .unwrap();
    drop(connection);
    let store = Store::open(root.path()).unwrap();
    assert_eq!(
        store.snapshot(id, NOW, &uid(40)).unwrap(),
        golden["claimed"]
    );
    let connection = rusqlite::Connection::open(store.database().path()).unwrap();
    let after: String = connection
        .query_row(
            "SELECT payload FROM executions WHERE action_id=?",
            [id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(before, after);
    assert_eq!(
        fs::read(root.path().join(id).join("plan.json")).unwrap(),
        plan_bytes
    );
    let recovered = store.recover_interrupted(NOW, &uid(41)).unwrap();
    assert_eq!(recovered[0]["execution"]["revision"], 4);
    assert_eq!(recovered[0]["journal"].as_array().unwrap().len(), 4);
}

#[test]
fn published_plan_without_initial_execution_can_authorize_after_restart() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let golden = golden();
    let plan = &golden["plan"];
    let id = plan["action_id"].as_str().unwrap();
    fs::create_dir(store.directory(id).unwrap()).unwrap();
    fs::write(
        store.directory(id).unwrap().join("plan.json"),
        serde_json::to_vec(plan).unwrap(),
    )
    .unwrap();
    let authorized = store
        .authorize(id, "actor:test", "approved after restart", NOW, &uid(1))
        .unwrap();
    assert_eq!(authorized["execution"]["status"], "AUTHORIZED");
    assert_eq!(authorized["execution"]["revision"], 2);
    assert_eq!(authorized["journal"].as_array().unwrap().len(), 2);
}

#[test]
fn sqlite_crash_fixture_child() {
    let Ok(root) = std::env::var("ML_EXPD_TEST_CRASH_ACTION_DB") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let mut connection = rusqlite::Connection::open(root.join("actions.sqlite3")).unwrap();
    let transaction = connection
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    transaction
        .execute("UPDATE executions SET revision=999,payload='{}'", [])
        .unwrap();
    fs::write(root.join("transaction-started"), b"ready").unwrap();
    loop {
        std::thread::park();
    }
}

#[test]
fn killed_sqlite_writer_rolls_back_uncommitted_state_without_replaying() {
    use std::{
        process::{Child, Command, Stdio},
        time::{Duration, Instant},
    };
    struct Guard(Child);
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let golden = golden();
    let id = golden["plan"]["action_id"].as_str().unwrap();
    let before = store.save_plan(&golden["plan"], NOW, &uid(1)).unwrap();
    let mut child = Guard(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "operations::tests::sqlite_crash_fixture_child",
                "--nocapture",
            ])
            .env("ML_EXPD_TEST_CRASH_ACTION_DB", root.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    while !root.path().join("transaction-started").exists() {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "crash fixture exited before starting transaction"
        );
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "crash fixture never began transaction"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    child.0.kill().unwrap();
    assert!(!child.0.wait().unwrap().success());
    let reopened = Store::open(root.path()).unwrap();
    assert_eq!(reopened.snapshot(id, NOW, &uid(2)).unwrap(), before);
    assert!(
        reopened
            .recover_interrupted(NOW, &uid(3))
            .unwrap()
            .is_empty()
    );
}

#[cfg(unix)]
#[test]
fn database_marker_and_json_symlinks_or_fifos_are_rejected_without_blocking() {
    use std::{
        os::unix::fs::symlink,
        process::Command,
        time::{Duration, Instant},
    };
    for name in ["actions.sqlite3", ".sqlite-authoritative"] {
        for fifo in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let store = Store::open(root.path()).unwrap();
            let path = root.path().join(name);
            if fifo {
                assert!(
                    Command::new("mkfifo")
                        .arg(&path)
                        .status()
                        .unwrap()
                        .success()
                );
            } else {
                let target = root.path().join("target");
                fs::write(&target, "private target").unwrap();
                symlink(&target, &path).unwrap();
            }
            let start = Instant::now();
            assert!(
                store
                    .database()
                    .read(&|| Ok(()), "action-0123456789abcdef")
                    .is_err()
            );
            assert!(start.elapsed() < Duration::from_secs(2));
        }
    }
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(root.path()).unwrap();
    let id = "action-0123456789abcdef";
    fs::create_dir(store.directory(id).unwrap()).unwrap();
    let target = root.path().join("foreign.json");
    fs::write(&target, serde_json::to_vec(&golden()["plan"]).unwrap()).unwrap();
    symlink(&target, store.directory(id).unwrap().join("plan.json")).unwrap();
    assert!(store.snapshot(id, NOW, &uid(1)).is_err());
}
