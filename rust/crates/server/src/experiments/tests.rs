use super::*;
use crate::deletion::RunFence;
use ml_exp_io::directory::Directory;
use serde_json::{Map, Value, json};
use std::{ffi::OsStr, fs, os::unix::fs::symlink, path::Path};

const PROJECT: &[u8] = br#"{"schema_version":1,"project":"demo","title":"Synthetic readonly reader","run_roots":["runs"]}"#;
fn native_run() -> Value {
    json!({"schema_version":1,"project":"demo","run_id":"train",
        "created_at":"2026-10-27T01:02:03Z","config_path":"train.json",
        "source_id":format!("source.{}","1".repeat(64)),"campaign":"baseline",
        "image_id":format!("sha256:{}","2".repeat(64)),"seed":42,
        "backend":{"kind":"slurm"},"resources":{"gpus":1},"storage":{"run_dir":"/synthetic/run"},
        "command":["python","train.py"],"execution":{},"assets":[],"evaluation":{},
        "future_metadata":{"arbitrary_field":[1,true,null,"on"]}})
}
fn native_attempt() -> Value {
    let mut attempt = native_run();
    attempt["created_at"] = json!("2026-10-27T02:03:04Z");
    attempt["attempt_id"] = json!("attempt-001");
    attempt["resume_from"] = json!("checkpoint.example");
    attempt
}
fn bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}
fn publish(directory: &Directory, filename: &str, bytes: &[u8]) {
    directory.publish_new(OsStr::new(filename), bytes).unwrap();
}
struct Fixture {
    root: tempfile::TempDir,
    store: Store,
    fence: RunFence,
    reference: RunRef,
}
fn fixture() -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let project = Directory::open(&root.path().join("project"), true).unwrap();
    publish(&project, "project.json", PROJECT);
    let runs = project.child(OsStr::new("runs"), true).unwrap();
    let run = runs.child(OsStr::new("train"), true).unwrap();
    publish(&run, "manifest.json", &bytes(&native_run()));
    let attempt = run
        .child(OsStr::new("attempts"), true)
        .unwrap()
        .child(OsStr::new("attempt-001"), true)
        .unwrap();
    publish(&attempt, "attempt.json", &bytes(&native_attempt()));
    let fence = RunFence::new(root.path().join("registry"));
    drop(fence.shared().unwrap()); // Explicit synthetic operator initialization.
    let store = Store::open(
        vec![ProjectBinding {
            project: "demo".into(),
            project_file: root.path().join("project/project.json"),
            run_roots: vec![root.path().join("project/runs")],
        }],
        fence.clone(),
    )
    .unwrap();
    Fixture {
        root,
        store,
        fence,
        reference: RunRef {
            project: "demo".into(),
            run_id: "train".into(),
            root_index: 0,
            relative_components: vec!["train".into()],
        },
    }
}
fn run_path(root: &Path) -> std::path::PathBuf {
    root.join("project/runs/train")
}

#[test]
fn documents_require_strict_json_within_the_edge_budget() {
    for invalid in [
        "{x: 1}",
        "{\"x\": 1,}",
        "#comment\n{\"x\": 1}",
        "{\"x\": ~}",
        "{\"x\": 'a'}",
        "{\"x\":1,\"x\":2}",
        "x: 1\n",
        "[]",
    ] {
        assert!(
            ml_exp_core::json::mapping(invalid.as_bytes()).is_err(),
            "accepted {invalid:?}"
        );
    }
    let map = ml_exp_core::json::mapping(
        br#"{"large":123456789012345678901234567890,"float":1e+20,"text":"yes"}"#,
    )
    .unwrap();
    assert_eq!(map["large"].to_string(), "123456789012345678901234567890");
    let map =
        ml_exp_core::json::mapping(br#"{"emoji":"\uD83D\uDE00","nul":"\u0000","nel":"\u0085"}"#)
            .unwrap();
    assert_eq!(map["emoji"], "😀");
    assert_eq!(map["nul"], "\0");
    assert_eq!(map["nel"], "\u{85}");
    assert!(ml_exp_core::json::mapping(br#"{"number":1e999}"#).is_err());
    let deep = format!("{{\"x\":{}0{}}}", "[".repeat(80), "]".repeat(80));
    assert!(matches!(
        Document::decode(deep.as_bytes(), ManifestSource::RunJson),
        Err(DocumentError::Complexity)
    ));
}

#[test]
fn exact_run_attempt_and_byte_expectations_are_read_without_mutation_or_index() {
    let f = fixture();
    let before = fs::read(run_path(f.root.path()).join("manifest.json")).unwrap();
    let project = f.store.project("demo", None).unwrap();
    assert_eq!(project.metadata["title"], "Synthetic readonly reader");
    let run = f.store.run(&f.reference, None).unwrap();
    assert!(run.metadata.get("future_metadata").is_some());
    let first = f
        .store
        .attempt(&f.reference, "attempt-001", &ExpectedHashes::default())
        .unwrap();
    assert!(first.provenance.compared_fields_complete);
    assert_eq!(first.provenance.compared_fields.len(), 5);
    assert_ne!(
        first.run.metadata["created_at"],
        first.attempt.metadata["created_at"]
    );
    let expected = ExpectedHashes {
        run: Some(run.raw_sha256),
        attempt: Some(first.attempt.raw_sha256),
    };
    assert!(
        f.store
            .attempt(&f.reference, "attempt-001", &expected)
            .is_ok()
    );
    assert!(matches!(
        f.store
            .run(&f.reference, Some(&format!("sha256:{}", "0".repeat(64)))),
        Err(Error::BytesChanged)
    ));
    assert_eq!(
        fs::read(run_path(f.root.path()).join("manifest.json")).unwrap(),
        before
    );
    assert!(!f.root.path().join("index.sqlite").exists());
    assert!(
        f.store
            .attempt(&f.reference, "attempt-002", &ExpectedHashes::default())
            .is_err()
    );
    assert!(
        !run_path(f.root.path())
            .join("attempts/attempt-002")
            .exists()
    );
}

#[test]
fn retired_names_corrupt_files_and_collected_copies_are_not_a_fallback() {
    let f = fixture();
    let path = run_path(f.root.path());
    fs::rename(
        path.join("manifest.json"),
        path.join("control_manifest.json"),
    )
    .unwrap();
    assert!(matches!(
        f.store.run(&f.reference, None),
        Err(Error::Missing)
    ));
    // A document under an earlier name is left on disk and never read.
    fs::write(path.join("manifest.yaml"), bytes(&native_run())).unwrap();
    assert!(matches!(
        f.store.run(&f.reference, None),
        Err(Error::Missing)
    ));
    fs::remove_file(path.join("manifest.yaml")).unwrap();
    fs::write(path.join("manifest.json"), b"broken: [").unwrap();
    assert!(f.store.run(&f.reference, None).is_err());
    fs::remove_file(path.join("manifest.json")).unwrap();
    fs::remove_file(path.join("control_manifest.json")).unwrap();
    fs::create_dir(path.join("collected_run")).unwrap();
    fs::write(
        path.join("collected_run/manifest.json"),
        bytes(&native_run()),
    )
    .unwrap();
    assert!(matches!(
        f.store.run(&f.reference, None),
        Err(Error::Missing)
    ));
}

#[test]
fn identity_conflicts_paths_and_cross_attempt_evidence_fail_closed() {
    let f = fixture();
    let path = run_path(f.root.path());
    for (filename, field, value) in [
        ("manifest.json", "project", json!("foreign")),
        ("manifest.json", "run_id", json!("other")),
        ("manifest.json", "schema_version", json!(2)),
        (
            "attempts/attempt-001/attempt.json",
            "attempt_id",
            json!("attempt-002"),
        ),
        (
            "attempts/attempt-001/attempt.json",
            "source_id",
            json!("source.foreign"),
        ),
        ("attempts/attempt-001/attempt.json", "seed", json!(99)),
        ("attempts/attempt-001/attempt.json", "project", json!(null)),
    ] {
        let file = path.join(filename);
        let before = fs::read(&file).unwrap();
        let mut map = ml_exp_core::json::mapping(&before).unwrap();
        map.insert(field.into(), value);
        fs::write(&file, serde_json::to_vec(&map).unwrap()).unwrap();
        assert!(
            f.store
                .attempt(&f.reference, "attempt-001", &ExpectedHashes::default())
                .is_err(),
            "accepted conflict {field}"
        );
        fs::write(file, before).unwrap();
    }
    let mut bad = f.reference.clone();
    bad.relative_components = vec!["..".into()];
    assert!(f.store.run(&bad, None).is_err());
    bad = f.reference.clone();
    bad.run_id = "../train".into();
    assert!(f.store.run(&bad, None).is_err());
    bad = f.reference.clone();
    bad.root_index = 99;
    assert!(f.store.run(&bad, None).is_err());
    assert!(
        f.store
            .attempt(&f.reference, "../attempt-001", &ExpectedHashes::default())
            .is_err()
    );
}

#[test]
fn existing_fence_is_mandatory_deleted_runs_are_unreadable_and_no_missing_lock_is_repaired() {
    let empty = tempfile::tempdir().unwrap();
    assert!(Store::open(vec![], RunFence::new(empty.path().join("missing"))).is_err());
    assert!(!empty.path().join("missing").exists());
    let f = fixture();
    f.fence
        .mark("demo", "train", &Map::new(), "synthetic")
        .unwrap();
    assert!(matches!(
        f.store.run(&f.reference, None),
        Err(Error::Fence(crate::deletion::Error::Deleted))
    ));
    assert!(matches!(
        f.store
            .attempt(&f.reference, "attempt-001", &ExpectedHashes::default()),
        Err(Error::Fence(crate::deletion::Error::Deleted))
    ));
    fs::remove_file(f.root.path().join("registry/.run-deletion.lock")).unwrap();
    assert!(f.store.project("demo", None).is_err());
    assert!(!f.root.path().join("registry/.run-deletion.lock").exists());
}

#[test]
fn symlinks_and_nonregular_manifests_are_not_followed() {
    let f = fixture();
    let path = run_path(f.root.path());
    fs::remove_file(path.join("manifest.json")).unwrap();
    symlink(
        path.join("attempts/attempt-001/attempt.json"),
        path.join("manifest.json"),
    )
    .unwrap();
    assert!(f.store.run(&f.reference, None).is_err());
    fs::remove_file(path.join("manifest.json")).unwrap();
    fs::create_dir(path.join("manifest.json")).unwrap();
    assert!(f.store.run(&f.reference, None).is_err());
    fs::remove_dir(path.join("manifest.json")).unwrap();
}

fn evidence_file(root: &Path, file: &str, value: Value) {
    fs::write(
        run_path(root).join(file),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
}
fn extra_attempt(root: &Path, attempt: &str) {
    let directory = run_path(root).join("attempts").join(attempt);
    fs::create_dir(&directory).unwrap();
    let mut manifest = native_attempt();
    manifest["attempt_id"] = json!(attempt);
    fs::write(directory.join("attempt.json"), bytes(&manifest)).unwrap();
}
fn set_current(root: &Path, attempt: &str) {
    evidence_file(
        root,
        "status.json",
        json!({"project":"demo","run_id":"train","attempt_id":attempt,"state":"RUNNING","updated_at":"2026-10-27T03:04:05Z"}),
    );
}

#[test]
fn metadata_snapshots_keep_current_identity_but_never_open_current_payload_tails() {
    let f = fixture();
    set_current(f.root.path(), "attempt-001");
    let path = run_path(f.root.path()).join("attempts/attempt-001");
    symlink("/never-open-status-list-payload", path.join("stdout.log")).unwrap();
    symlink(
        "/never-open-status-list-payload",
        path.join("train_metrics.jsonl"),
    )
    .unwrap();
    let snapshot = f.store.snapshot_metadata(&f.reference).unwrap();
    assert_eq!(snapshot.current_attempt_id.as_deref(), Some("attempt-001"));
    let attempt = &snapshot.attempts[0];
    assert_eq!(attempt.attempt_id, "attempt-001");
    assert!(attempt.stdout.source.is_none());
    assert!(attempt.stderr.source.is_none());
    assert!(attempt.metrics.source.is_none());
    assert!(
        attempt
            .stdout
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::NotLoaded)
    );
    assert!(
        attempt
            .stderr
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::NotLoaded)
    );
    assert!(
        attempt
            .metrics
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::NotLoaded)
    );
    let full = f.store.snapshot(&f.reference).unwrap();
    assert!(
        full.attempts[0]
            .stdout
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::Io)
    );
    assert!(
        full.attempts[0]
            .metrics
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::Io)
    );
}

#[test]
fn snapshots_read_exact_current_evidence_and_only_one_attempt_payload() {
    let f = fixture();
    let path = run_path(f.root.path());
    extra_attempt(f.root.path(), "attempt-002");
    set_current(f.root.path(), "attempt-001");
    for file in [
        "status.json",
        "backend.json",
        "collection.json",
        "decision.json",
        "submission.json",
    ] {
        evidence_file(
            f.root.path(),
            &format!("attempts/attempt-001/{file}"),
            json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","state":"RUNNING","backend_job_id":"12345"}),
        );
    }
    fs::write(
        path.join("attempts/attempt-001/stdout.log"),
        "first\nsecond\n",
    )
    .unwrap();
    fs::write(
        path.join("attempts/attempt-001/train_metrics.jsonl"),
        "{\"step\":1,\"loss\":2.5}\n",
    )
    .unwrap();
    symlink(
        "/never-open-historical-payload",
        path.join("attempts/attempt-002/stdout.log"),
    )
    .unwrap();
    let snapshot = f.store.snapshot(&f.reference).unwrap();
    assert_eq!(snapshot.current_attempt_id.as_deref(), Some("attempt-001"));
    assert_eq!(
        snapshot.current_selection_source.as_deref(),
        Some("status.json")
    );
    assert_eq!(snapshot.attempts.len(), 2);
    let current = &snapshot.attempts[0];
    assert_eq!(current.stdout.lines, ["first", "second"]);
    assert_eq!(
        current.stdout.source.as_deref(),
        Some("attempts/attempt-001/stdout.log")
    );
    assert_eq!(current.metrics.records, [json!({"step":1,"loss":2.5})]);
    assert_eq!(
        current.metrics.source_attempt_id.as_deref(),
        Some("attempt-001")
    );
    assert!(
        matches!(&current.submission, EvidenceDocument::Present { metadata, .. } if metadata["backend_job_id"] == "12345")
    );
    assert!(snapshot.attempts[1].stdout.source.is_none());
    assert!(
        snapshot.attempts[1]
            .stdout
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::NotLoaded)
    );
    // Exact historic reads do not inherit current metrics or logs.
    let historical = f
        .store
        .snapshot_attempt(&f.reference, "attempt-002")
        .unwrap();
    assert_eq!(historical.attempts.len(), 1);
    assert_eq!(historical.attempts[0].attempt_id, "attempt-002");
    assert!(historical.current_attempt_id.is_none());
    assert!(historical.attempts[0].metrics.records.is_empty());
    assert!(
        historical.attempts[0]
            .stdout
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::Io)
    );
    assert!(matches!(
        f.store.snapshot_attempt(&f.reference, "attempt-999"),
        Err(Error::Missing)
    ));
}

#[test]
fn explicit_mirrors_select_current_but_corruption_or_foreign_binding_never_selects_an_older_one() {
    let f = fixture();
    let path = run_path(f.root.path());
    extra_attempt(f.root.path(), "attempt-002");
    let unbound = f.store.snapshot(&f.reference).unwrap();
    assert!(unbound.current_attempt_id.is_none());
    assert!(
        unbound
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::NoExplicitCurrent)
    );
    evidence_file(
        f.root.path(),
        "collection.json",
        json!({"attempt_id":"attempt-002"}),
    );
    assert_eq!(
        f.store
            .snapshot(&f.reference)
            .unwrap()
            .current_attempt_id
            .as_deref(),
        Some("attempt-002")
    );
    fs::write(path.join("status.json"), "malformed").unwrap();
    let bad = f.store.snapshot(&f.reference).unwrap();
    assert!(bad.current_attempt_id.is_none());
    assert!(matches!(
        bad.current_status,
        EvidenceDocument::Unavailable {
            code: EvidenceProblem::InvalidJson,
            ..
        }
    ));
    set_current(f.root.path(), "attempt-999");
    let bad = f.store.snapshot(&f.reference).unwrap();
    assert!(bad.current_attempt_id.is_none());
    assert!(
        bad.diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::CurrentAttemptUnavailable)
    );
    evidence_file(
        f.root.path(),
        "status.json",
        json!({"project":"foreign","attempt_id":"attempt-001"}),
    );
    let bad = f.store.snapshot(&f.reference).unwrap();
    assert!(bad.current_attempt_id.is_none());
    assert!(matches!(
        bad.current_status,
        EvidenceDocument::Unavailable {
            code: EvidenceProblem::IdentityConflict,
            ..
        }
    ));
}

#[test]
fn tails_are_bounded_and_jsonl_partial_corrupt_and_foreign_rows_remain_diagnostic() {
    let f = fixture();
    let path = run_path(f.root.path()).join("attempts/attempt-001");
    let mut log = vec![b'x'; 2 * 1024 * 1024];
    log.extend_from_slice(b"\nlast line\n");
    fs::write(path.join("stdout.log"), log).unwrap();
    fs::write(path.join("stderr.log"), b"bad utf8 \xff\n").unwrap();
    fs::write(path.join("train_metrics.jsonl"), b"{\"step\":1,\"loss\":2.5}\nnot json\n{\"attempt_id\":\"attempt-002\",\"step\":2,\"loss\":1.0}\n{\"step\":3,\"loss\":0.5}").unwrap();
    let snapshot = f
        .store
        .snapshot_attempt(&f.reference, "attempt-001")
        .unwrap();
    let attempt = &snapshot.attempts[0];
    assert_eq!(attempt.stdout.lines, ["last line"]);
    assert!(attempt.stdout.truncated);
    assert!(attempt.stdout.sampled_bytes <= 1024 * 1024);
    assert!(attempt.stdout.offset_bytes > 0);
    assert!(
        attempt
            .stderr
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::InvalidUtf8)
    );
    assert_eq!(attempt.metrics.records, [json!({"step":1,"loss":2.5})]);
    for code in [
        EvidenceProblem::MalformedRecord,
        EvidenceProblem::ForeignRecord,
        EvidenceProblem::PartialFinalLine,
    ] {
        assert!(attempt.metrics.diagnostics.iter().any(|d| d.code == code));
    }
    assert!(attempt.stdout.modified_at_unix.is_some());
    assert!(!attempt.manifest.metadata.contains_key("sampled_at"));
}

#[test]
fn log_tails_split_on_unicode_line_separators_without_changing_jsonl_framing() {
    let f = fixture();
    let path = run_path(f.root.path()).join("attempts/attempt-001");
    fs::write(
        path.join("stdout.log"),
        "a\r旧\u{85}新\u{2028}尾\r\nb\x0bc\x0cd\x1ce\x1df\x1eg\u{2029}",
    )
    .unwrap();
    let mut large = vec![b'x'; 2 * 1024 * 1024];
    large.extend_from_slice("\u{2028}retained\r\n".as_bytes());
    fs::write(path.join("stderr.log"), large).unwrap();
    fs::write(path.join("metrics.jsonl"), "{\"value\":1}\r{\"value\":2}\n").unwrap();
    let snapshot = f
        .store
        .snapshot_attempt(&f.reference, "attempt-001")
        .unwrap();
    let attempt = &snapshot.attempts[0];
    assert_eq!(
        attempt.stdout.lines,
        ["a", "旧", "新", "尾", "b", "c", "d", "e", "f", "g"]
    );
    assert_eq!(attempt.stdout.available_lines, 10);
    assert_eq!(attempt.stderr.lines, ["retained"]);
    assert!(attempt.stderr.truncated);
    assert!(attempt.stderr.sampled_bytes <= 1024 * 1024);
    assert!(attempt.metrics.records.is_empty());
    assert!(
        attempt
            .metrics
            .diagnostics
            .iter()
            .any(|d| d.code == EvidenceProblem::MalformedRecord)
    );
    fs::write(path.join("stdout.log"), "\r\n\n").unwrap();
    let snapshot = f
        .store
        .snapshot_attempt(&f.reference, "attempt-001")
        .unwrap();
    assert_eq!(snapshot.attempts[0].stdout.lines, ["", ""]);
}

#[test]
fn exact_snapshots_do_not_read_or_hide_bad_sibling_manifests_and_evidence_errors_are_not_missing() {
    let f = fixture();
    let path = run_path(f.root.path());
    extra_attempt(f.root.path(), "attempt-002");
    fs::write(path.join("attempts/attempt-002/attempt.json"), "broken: [").unwrap();
    evidence_file(
        f.root.path(),
        "attempts/attempt-001/backend.json",
        json!({"attempt_id":"attempt-999"}),
    );
    fs::write(
        path.join("attempts/attempt-001/status.json"),
        "invalid json",
    )
    .unwrap();
    let exact = f
        .store
        .snapshot_attempt(&f.reference, "attempt-001")
        .unwrap();
    assert!(matches!(
        exact.attempts[0].backend,
        EvidenceDocument::Unavailable {
            code: EvidenceProblem::IdentityConflict,
            ..
        }
    ));
    assert!(matches!(
        exact.attempts[0].status,
        EvidenceDocument::Unavailable {
            code: EvidenceProblem::InvalidJson,
            ..
        }
    ));
    assert!(matches!(
        exact.attempts[0].decision,
        EvidenceDocument::Missing
    ));
    assert!(f.store.snapshot(&f.reference).is_err());
    // The actual payload source is always an approved fixed name. A producer
    // path cannot redirect the reader into a dataset, credential or checkpoint.
    evidence_file(
        f.root.path(),
        "attempts/attempt-001/collection.json",
        json!({"metrics_path":"/must-not-read","stdout_path":"../../forbidden"}),
    );
    assert!(
        f.store
            .snapshot_attempt(&f.reference, "attempt-001")
            .unwrap()
            .attempts[0]
            .metrics
            .source
            .is_none()
    );
}

#[test]
fn deleted_fence_fails_a_snapshot_without_repair() {
    let f = fixture();
    assert!(
        f.store
            .snapshot_attempt(&f.reference, "attempt-001")
            .is_ok()
    );
    f.fence
        .mark("demo", "train", &Map::new(), "synthetic")
        .unwrap();
    assert!(matches!(
        f.store.snapshot_attempt(&f.reference, "attempt-001"),
        Err(Error::Fence(crate::deletion::Error::Deleted))
    ));
}

#[test]
fn metrics_tail_record_limit_does_not_count_the_terminal_empty_split() {
    let f = fixture();
    let path = run_path(f.root.path()).join("attempts/attempt-001/train_metrics.jsonl");
    fs::write(&path, b"{\"step\":1,\"loss\":2.5}\n".repeat(10_000)).unwrap();
    let exact = f
        .store
        .snapshot_attempt(&f.reference, "attempt-001")
        .unwrap();
    assert_eq!(exact.attempts[0].metrics.records.len(), 10_000);
    assert!(!exact.attempts[0].metrics.truncated);
    assert!(exact.attempts[0].metrics.diagnostics.is_empty());
    fs::write(&path, b"{\"step\":1,\"loss\":2.5}\n".repeat(10_001)).unwrap();
    let over = f
        .store
        .snapshot_attempt(&f.reference, "attempt-001")
        .unwrap();
    assert_eq!(over.attempts[0].metrics.records.len(), 10_000);
    assert!(over.attempts[0].metrics.truncated);
}
