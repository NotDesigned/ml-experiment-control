use super::*;
use crate::deletion::{Error as FenceError, RunFence};
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeSet,
    fs,
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/python-run-index-v1.json")).unwrap()
}
fn row(project: &str, run: &str) -> Value {
    json!({"project":project,"run_id":run,"run_dir":format!("/canonical/{project}/{run}"),"scheduler_state":"RUNNING","scanned_at":1})
}
fn setup(root: &std::path::Path) -> RunIndex {
    RunIndex::open(
        root.join("index.sqlite"),
        RunFence::new(root.join("registry")),
    )
    .unwrap()
}

#[test]
fn python_index_roundtrip_change_detection_memberships_and_polling_match() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    let fixture = fixture();
    for case in fixture["upserts"].as_array().unwrap() {
        assert_eq!(
            index.upsert_run(&case["row"]).unwrap(),
            case["changed"].as_bool().unwrap()
        );
    }
    assert_eq!(
        index.list_runs(None, None).unwrap(),
        fixture["runs"].as_array().unwrap().clone()
    );
    let origin: Vec<_> = index
        .list_runs(Some("demo"), Some("origin"))
        .unwrap()
        .iter()
        .map(|row| row["run_id"].clone())
        .collect();
    assert_eq!(origin, fixture["origin_runs"].as_array().unwrap().clone());
    for case in fixture["polls"].as_array().unwrap() {
        index
            .record_poll(
                "demo",
                "run-a",
                case["verb"].as_str().unwrap(),
                case["error"].as_str(),
                case["now"].as_f64().unwrap(),
                case["outcome"].as_str(),
            )
            .unwrap();
        assert_eq!(
            index.collector_statuses(Some("demo")).unwrap(),
            vec![case["expected"].clone()]
        );
    }
    assert!(index.get_run("foreign", "run-a").unwrap().is_none());
    assert!(index.list_runs(Some("foreign"), None).unwrap().is_empty());
}

#[test]
fn tombstones_hide_all_run_and_poll_reads_and_block_reinsertion() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    let fence = RunFence::new(root.path().join("registry"));
    index.upsert_run(&row("p", "r")).unwrap();
    index
        .record_poll("p", "r", "status", None, 1.0, None)
        .unwrap();
    fence
        .mark("p", "r", &Map::new(), "2026-10-27T00:00:00Z")
        .unwrap();
    assert!(index.get_run("p", "r").unwrap().is_none());
    assert!(index.list_runs(None, None).unwrap().is_empty());
    assert!(index.collector_statuses(None).unwrap().is_empty());
    assert!(!index.upsert_run(&row("p", "r")).unwrap());
    assert!(
        !index
            .record_poll("p", "r", "status", None, 2.0, None)
            .unwrap()
    );
    assert!(index.delete_run("p", "r").unwrap());
    assert!(!index.delete_run("p", "r").unwrap());
    assert!(fence.is_deleted("p", "r").unwrap());
}

#[test]
fn unknown_metadata_is_preserved_but_foreign_payload_binding_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    let mut original = row("p", "r");
    original["future_schema"] = json!({"protocol":"unfamiliar","checkpoint":{"sha":"immutable"}});
    index.upsert_run(&original).unwrap();
    assert_eq!(index.get_run("p", "r").unwrap(), Some(original.clone()));
    let connection = rusqlite::Connection::open(index.path()).unwrap();
    let mut foreign = original;
    foreign["project"] = json!("other");
    connection
        .execute(
            "UPDATE runs SET row_json=? WHERE project='p' AND run_id='r'",
            [serde_json::to_string(&foreign).unwrap()],
        )
        .unwrap();
    drop(connection);
    assert!(matches!(
        index.get_run("p", "r"),
        Err(Error::Row(ml_exp_core::index::Error::Identity))
    ));
    assert!(matches!(
        index.list_runs(None, None),
        Err(Error::Row(ml_exp_core::index::Error::Identity))
    ));
}

#[test]
fn legacy_collector_schema_is_upgraded_without_losing_prior_status() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.sqlite");
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection.execute_batch("CREATE TABLE collector_status(project TEXT NOT NULL,run_id TEXT NOT NULL,last_poll_at REAL,last_verb TEXT,last_error TEXT,PRIMARY KEY(project,run_id)); INSERT INTO collector_status VALUES ('p','r',1,'status','legacy failure');").unwrap();
    drop(connection);
    let index = setup(root.path());
    let statuses = index.collector_statuses(None).unwrap();
    assert_eq!(statuses[0]["last_error"], "legacy failure");
    assert_eq!(statuses[0]["verb_results"], json!({}));
    index
        .record_poll("p", "r", "collect", None, 2.0, None)
        .unwrap();
    assert_eq!(
        index.collector_statuses(None).unwrap()[0]["verb_results"]["collect"]["outcome"],
        "succeeded"
    );
}

#[test]
fn invalid_legacy_poll_json_recovers_without_clearing_other_verbs_failures() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    index
        .record_poll("p", "r", "status", Some("status failed"), 1.0, None)
        .unwrap();
    index
        .record_poll("p", "r", "collect", None, 2.0, None)
        .unwrap();
    assert_eq!(
        index.collector_statuses(None).unwrap()[0]["last_error"],
        "status failed"
    );
    let connection = rusqlite::Connection::open(index.path()).unwrap();
    connection
        .execute("UPDATE collector_status SET verb_results='not-json'", [])
        .unwrap();
    drop(connection);
    index
        .record_poll("p", "r", "status", None, 3.0, None)
        .unwrap();
    assert!(index.collector_statuses(None).unwrap()[0]["last_error"].is_null());
    assert!(
        index
            .record_poll("p", "r", "status", None, f64::NAN, None)
            .is_err()
    );
}

#[test]
fn pruning_requires_complete_scan_and_cannot_affect_another_project() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    for (project, run) in [("p", "a"), ("p", "b"), ("other", "b")] {
        index.upsert_run(&row(project, run)).unwrap();
        index
            .record_poll(project, run, "status", None, 1.0, None)
            .unwrap();
    }
    assert!(
        index
            .prune_project_runs("p", &BTreeSet::new(), false)
            .unwrap()
            .is_empty()
    );
    assert_eq!(index.list_runs(None, None).unwrap().len(), 3);
    assert_eq!(
        index
            .prune_project_runs("p", &BTreeSet::from(["a".into()]), true)
            .unwrap(),
        vec!["b"]
    );
    assert!(index.get_run("other", "b").unwrap().is_some());
    assert_eq!(index.collector_statuses(Some("p")).unwrap().len(), 1);
    index.set_meta("last_scan", "explicit-time").unwrap();
    assert_eq!(
        index.get_meta("last_scan").unwrap().as_deref(),
        Some("explicit-time")
    );
}

#[test]
fn concurrent_upsert_events_publish_only_one_initial_change() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    let barrier = Arc::new(Barrier::new(2));
    let mut threads = Vec::new();
    for _ in 0..2 {
        let index = index.clone();
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            index.upsert_run(&row("p", "r")).unwrap()
        }));
    }
    assert_eq!(
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .filter(|changed| *changed)
            .count(),
        1
    );
}

#[test]
fn reads_hold_shared_fence_and_database_symlinks_or_fifo_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    let fence = RunFence::new(root.path().join("registry"));
    let guard = fence.shared().unwrap();
    assert!(matches!(
        fence.mark("p", "r", &Map::new(), "now"),
        Err(FenceError::Busy)
    ));
    drop(guard);
    index.upsert_run(&row("p", "r")).unwrap();
    #[cfg(unix)]
    {
        use std::{os::unix::fs::symlink, process::Command};
        for fifo in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("index.sqlite");
            if fifo {
                assert!(
                    Command::new("mkfifo")
                        .arg(&path)
                        .status()
                        .unwrap()
                        .success()
                );
            } else {
                symlink(index.path(), &path).unwrap();
            }
            let start = Instant::now();
            assert!(
                RunIndex::open(&path, RunFence::new(directory.path().join("registry"))).is_err()
            );
            assert!(start.elapsed() < Duration::from_secs(2));
        }
    }
    // A caller must explicitly reopen/rebuild a missing disposable index; an
    // ordinary query never silently creates an empty replacement database.
    fs::remove_file(index.path()).unwrap();
    assert!(index.get_run("p", "r").is_err());
    assert!(!index.path().exists());
}

#[test]
fn blocked_database_writer_keeps_fence_until_transaction_finishes() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    let fence = RunFence::new(root.path().join("registry"));
    let connection = rusqlite::Connection::open(index.path()).unwrap();
    connection.execute_batch("BEGIN IMMEDIATE;").unwrap();
    let writer = index.clone();
    let task = std::thread::spawn(move || writer.upsert_run(&row("p", "r")));
    let start = Instant::now();
    loop {
        match fence.exclusive(false) {
            Err(FenceError::Busy) => break,
            Ok(guard) => drop(guard),
            Err(error) => panic!("unexpected fence error: {error}"),
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "index writer never took shared fence"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(matches!(
        fence.mark("p", "r", &Map::new(), "blocked"),
        Err(FenceError::Busy)
    ));
    connection.execute_batch("ROLLBACK;").unwrap();
    assert!(task.join().unwrap().unwrap());
    fence.mark("p", "r", &Map::new(), "after-commit").unwrap();
    assert!(index.get_run("p", "r").unwrap().is_none());
}

#[test]
fn blocked_mutations_reject_replaced_fence_and_leave_sqlite_unchanged() {
    // A shared flock on an unlinked lock remains held, but no longer excludes
    // deletion through its replacement. Exercise real blocked SQLite writers,
    // including projection-only methods that intentionally permit tombstones.
    for replace_parent in [false, true] {
        for operation in ["upsert", "poll", "meta", "delete", "prune"] {
            let root = tempfile::tempdir().unwrap();
            let index = setup(root.path());
            let registry = root.path().join("registry");
            let fence = RunFence::new(&registry);
            let original = row("p", "r");
            index.upsert_run(&original).unwrap();
            index
                .record_poll("p", "r", "status", None, 1.0, None)
                .unwrap();
            index.set_meta("scan", "before").unwrap();
            let connection = rusqlite::Connection::open(index.path()).unwrap();
            connection.execute_batch("BEGIN IMMEDIATE;").unwrap();
            let writer = index.clone();
            let task = std::thread::spawn(move || match operation {
                "upsert" => {
                    let mut updated = row("p", "r");
                    updated["scheduler_state"] = json!("FAILED");
                    writer.upsert_run(&updated).map(|_| ())
                }
                "poll" => writer
                    .record_poll("p", "r", "status", None, 2.0, None)
                    .map(|_| ()),
                "meta" => writer.set_meta("scan", "after"),
                "delete" => writer.delete_run("p", "r").map(|_| ()),
                "prune" => writer
                    .prune_project_runs("p", &BTreeSet::new(), true)
                    .map(|_| ()),
                _ => unreachable!(),
            });
            let start = Instant::now();
            loop {
                match fence.exclusive(false) {
                    Err(FenceError::Busy) => break,
                    Ok(guard) => drop(guard),
                    Err(error) => panic!("unexpected fence error: {error}"),
                }
                assert!(start.elapsed() < Duration::from_secs(5));
                std::thread::sleep(Duration::from_millis(5));
            }
            // The database lock stays held throughout this interval; even if
            // the writer has not yet reached BEGIN, it cannot finish a write.
            std::thread::sleep(Duration::from_millis(20));
            if replace_parent {
                fs::rename(&registry, root.path().join("detached-registry")).unwrap();
                fs::create_dir(&registry).unwrap();
            } else {
                fs::rename(
                    registry.join(".run-deletion.lock"),
                    registry.join("detached-lock"),
                )
                .unwrap();
                fs::write(registry.join(".run-deletion.lock"), b"").unwrap();
            }
            // A fresh process can now acquire the replacement lock and mark
            // this same Run deleted while the old writer still holds its FD.
            RunFence::new(&registry)
                .mark("p", "r", &Map::new(), "replacement")
                .unwrap();
            connection.execute_batch("ROLLBACK;").unwrap();
            assert!(
                matches!(task.join().unwrap(), Err(Error::Fence(_))),
                "{operation}, replace_parent={replace_parent}"
            );
            let stored: String = connection
                .query_row(
                    "SELECT row_json FROM runs WHERE project='p' AND run_id='r'",
                    [],
                    |record| record.get(0),
                )
                .unwrap();
            assert_eq!(serde_json::from_str::<Value>(&stored).unwrap(), original);
            let poll: f64 = connection
                .query_row(
                    "SELECT last_poll_at FROM collector_status WHERE project='p' AND run_id='r'",
                    [],
                    |record| record.get(0),
                )
                .unwrap();
            assert_eq!(poll, 1.0);
            let meta: String = connection
                .query_row("SELECT value FROM meta WHERE key='scan'", [], |record| {
                    record.get(0)
                })
                .unwrap();
            assert_eq!(meta, "before");
        }
    }
}

#[test]
fn missing_fence_is_not_recreated_by_queries_or_mutations() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    index.upsert_run(&row("p", "r")).unwrap();
    let lock = root.path().join("registry/.run-deletion.lock");
    fs::remove_file(&lock).unwrap();
    assert!(matches!(index.get_run("p", "r"), Err(Error::Fence(_))));
    assert!(matches!(index.list_runs(None, None), Err(Error::Fence(_))));
    assert!(matches!(
        index.collector_statuses(None),
        Err(Error::Fence(_))
    ));
    assert!(matches!(index.get_meta("scan"), Err(Error::Fence(_))));
    assert!(matches!(
        index.upsert_run(&row("p", "r")),
        Err(Error::Fence(_))
    ));
    assert!(matches!(
        index.record_poll("p", "r", "status", None, 1.0, None),
        Err(Error::Fence(_))
    ));
    assert!(matches!(
        index.set_meta("scan", "after"),
        Err(Error::Fence(_))
    ));
    assert!(matches!(index.delete_run("p", "r"), Err(Error::Fence(_))));
    assert!(matches!(
        index.prune_project_runs("p", &BTreeSet::new(), true),
        Err(Error::Fence(_))
    ));
    assert!(!lock.exists());
}

#[test]
fn nullable_legacy_meta_stays_absent_without_being_rewritten() {
    let root = tempfile::tempdir().unwrap();
    let index = setup(root.path());
    let connection = rusqlite::Connection::open(index.path()).unwrap();
    connection
        .execute(
            "INSERT INTO meta(key,value) VALUES ('legacy-null',NULL)",
            [],
        )
        .unwrap();
    drop(connection);
    assert!(index.get_meta("legacy-null").unwrap().is_none());
    assert!(index.get_meta("missing").unwrap().is_none());
}
