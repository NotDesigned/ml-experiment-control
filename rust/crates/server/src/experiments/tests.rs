use super::*;
use crate::deletion::RunFence;
use ml_exp_io::directory::Directory;
use serde_json::{Map, Value, json};
use std::{
    ffi::OsStr,
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::Path,
};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("fixtures/python-manifests-v1.json")).unwrap()["cases"].as_array().unwrap().clone()
}
fn raw(name: &str) -> String {
    cases().into_iter().find(|v| v["name"] == name).unwrap()["raw"]
        .as_str()
        .unwrap()
        .into()
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
    publish(&project, "research_project.yaml", raw("project").as_bytes());
    let runs = project.child(OsStr::new("runs"), true).unwrap();
    let run = runs.child(OsStr::new("train"), true).unwrap();
    publish(&run, "manifest.yaml", raw("native-run").as_bytes());
    let attempt = run
        .child(OsStr::new("attempts"), true)
        .unwrap()
        .child(OsStr::new("attempt-001"), true)
        .unwrap();
    publish(&attempt, "attempt.yaml", raw("native-attempt").as_bytes());
    let fence = RunFence::new(root.path().join("registry"));
    drop(fence.shared().unwrap()); // Explicit synthetic operator initialization.
    let store = Store::open(
        vec![ProjectBinding {
            project: "demo".into(),
            project_file: root.path().join("project/research_project.yaml"),
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
fn actual_python_safe_dump_and_checked_in_legacy_metadata_have_identical_values_and_raw_sha() {
    for case in cases() {
        let bytes = case["raw"].as_str().unwrap().as_bytes();
        let map = parse_mapping(bytes, Format::Yaml).unwrap();
        assert!(
            ml_exp_core::observation::python_equal(&Value::Object(map), &case["metadata"]),
            "{}",
            case["name"]
        );
        let source = if case["name"] == "legacy-control" {
            ManifestSource::LegacyRunYaml
        } else {
            ManifestSource::RunYaml
        };
        let document = super::domain::Document::decode(bytes, source).unwrap();
        assert_eq!(document.raw_sha256, case["raw_sha256"]);
    }
}

#[test]
fn yaml_policy_rejects_ambiguous_or_expansive_input_and_json_requires_json_syntax() {
    for text in [
        "x: yes\n",
        "x: NO\n",
        "x: On\n",
        "x: 012\n",
        "x: +12\n",
        "x: 0x10\n",
        "x: 1_000\n",
        "x: .5\n",
        "x: 1.0e3\n",
        "x: 1e+20\n",
        "x: 12:34\n",
        "x: 2026-10-27\n",
        "x: .nan\n",
        "x: -.inf\n",
        "x: !!str 3\n",
        "x: &a [1, 2]\ny: *a\n",
        "x: 1\nx: 2\n",
        "1: value\n",
        "? [x, y]\n: value\n",
        "x: 1\n---\ny: 2\n",
        "[]\n",
        "x: {true: value}\n",
        "x: {a: 1, a: 2}\n",
    ] {
        assert!(
            parse_mapping(text.as_bytes(), Format::Yaml).is_err(),
            "accepted {text:?}"
        );
    }
    let valid = b"x: 'yes'\ny: '012'\nz: '2026-10-27'\nvalue: '<<'\nversion: 2.10.0\ncwd: .\n";
    let map = parse_mapping(valid, Format::Yaml).unwrap();
    assert_eq!(map["x"], "yes");
    assert_eq!(map["version"], "2.10.0");
    assert_eq!(map["cwd"], ".");
    for invalid_json in [
        "{x: 1}",
        "{\"x\": 1,}",
        "#comment\n{\"x\": 1}",
        "{\"x\": ~}",
        "{\"x\": 'a'}",
        "{\"x\":1,\"x\":2}",
    ] {
        assert!(parse_mapping(invalid_json.as_bytes(), Format::Json).is_err());
    }
    let map = parse_mapping(
        br#"{"large":123456789012345678901234567890,"float":1e+20,"text":"yes"}"#,
        Format::Json,
    )
    .unwrap();
    assert_eq!(map["large"].to_string(), "123456789012345678901234567890");
    let map = parse_mapping(
        br#"{"emoji":"\uD83D\uDE00","nul":"\u0000","nel":"\u0085"}"#,
        Format::Json,
    )
    .unwrap();
    assert_eq!(map["emoji"], "😀");
    assert_eq!(map["nul"], "\0");
    assert_eq!(map["nel"], "\u{85}");
    assert!(parse_mapping(br#"{"number":1e999}"#, Format::Json).is_err());
    for separator in ['\u{85}', '\u{2028}', '\u{2029}'] {
        for text in [format!("x: a{separator}b"), format!("x: \"a{separator}b\"")] {
            assert!(matches!(
                parse_mapping(text.as_bytes(), Format::Yaml),
                Err(DocumentError::UnsupportedYaml)
            ));
        }
    }
    assert_eq!(
        parse_mapping(b"x: \"a\\u0085b\"", Format::Yaml).unwrap()["x"],
        "a\u{85}b"
    );
    assert!(parse_mapping(&vec![b' '; super::domain::MAX_BYTES + 1], Format::Yaml).is_err());
    let deep = format!("x: {}0{}", "[".repeat(80), "]".repeat(80));
    assert!(matches!(
        parse_mapping(deep.as_bytes(), Format::Yaml),
        Err(DocumentError::Complexity)
    ));
    let wide = format!("x: [{}]", "0,".repeat(110_000));
    assert!(matches!(
        parse_mapping(wide.as_bytes(), Format::Yaml),
        Err(DocumentError::Complexity)
    ));
}

#[test]
fn exact_run_attempt_and_byte_expectations_are_read_without_mutation_or_index() {
    let f = fixture();
    let before = fs::read(run_path(f.root.path()).join("manifest.yaml")).unwrap();
    let project = f.store.project("demo", None).unwrap();
    assert_eq!(project.metadata["title"], "Synthetic readonly reader");
    let run = f.store.run(&f.reference, None).unwrap();
    assert!(!run.legacy_observation_only);
    assert!(run.metadata.get("future_metadata").is_some());
    let first = f
        .store
        .attempt(&f.reference, "attempt-001", &ExpectedHashes::default())
        .unwrap();
    assert!(first.provenance.compared_fields_complete);
    assert_eq!(first.provenance.compared_fields.len(), 5);
    assert!(!first.provenance.project_inherited_from_json_run);
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
        fs::read(run_path(f.root.path()).join("manifest.yaml")).unwrap(),
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
fn missing_canonical_files_allow_only_explicit_legacy_not_corrupt_or_collected_fallback() {
    let f = fixture();
    let path = run_path(f.root.path());
    fs::rename(
        path.join("manifest.yaml"),
        path.join("control_manifest.yaml"),
    )
    .unwrap();
    let legacy = f.store.run(&f.reference, None).unwrap();
    assert_eq!(legacy.source, ManifestSource::LegacyRunYaml);
    assert!(legacy.legacy_observation_only);
    fs::write(path.join("manifest.yaml"), b"broken: [").unwrap();
    assert!(f.store.run(&f.reference, None).is_err());
    fs::remove_file(path.join("manifest.yaml")).unwrap();
    fs::remove_file(path.join("control_manifest.yaml")).unwrap();
    fs::create_dir(path.join("collected_run")).unwrap();
    fs::write(path.join("collected_run/manifest.yaml"), raw("native-run")).unwrap();
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
        ("manifest.yaml", "project", json!("foreign")),
        ("manifest.yaml", "run_id", json!("other")),
        ("manifest.yaml", "schema_version", json!(2)),
        (
            "attempts/attempt-001/attempt.yaml",
            "attempt_id",
            json!("attempt-002"),
        ),
        (
            "attempts/attempt-001/attempt.yaml",
            "source_id",
            json!("source.foreign"),
        ),
        ("attempts/attempt-001/attempt.yaml", "seed", json!(99)),
    ] {
        let file = path.join(filename);
        let before = fs::read(&file).unwrap();
        let mut map = parse_mapping(&before, Format::Yaml).unwrap();
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
fn json_harness_project_inheritance_is_explicit_and_not_applied_to_yaml() {
    let f = fixture();
    let path = run_path(f.root.path());
    let run = parse_mapping(&fs::read(path.join("manifest.yaml")).unwrap(), Format::Yaml).unwrap();
    fs::remove_file(path.join("manifest.yaml")).unwrap();
    fs::write(
        path.join("manifest.json"),
        serde_json::to_vec(&run).unwrap(),
    )
    .unwrap();
    let file = path.join("attempts/attempt-001/attempt.yaml");
    let mut attempt = parse_mapping(&fs::read(&file).unwrap(), Format::Yaml).unwrap();
    attempt.shift_remove("project");
    fs::write(&file, serde_json::to_vec(&attempt).unwrap()).unwrap();
    assert!(
        f.store
            .attempt(&f.reference, "attempt-001", &ExpectedHashes::default())
            .is_err()
    );
    fs::remove_file(&file).unwrap();
    fs::write(
        file.with_file_name("attempt.json"),
        serde_json::to_vec(&attempt).unwrap(),
    )
    .unwrap();
    let read = f
        .store
        .attempt(&f.reference, "attempt-001", &ExpectedHashes::default())
        .unwrap();
    assert!(read.provenance.project_inherited_from_json_run);
    assert!(!read.attempt.metadata.contains_key("project"));
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
fn detached_namespaces_symlinks_and_nonregular_manifests_are_not_followed() {
    let f = fixture();
    let path = run_path(f.root.path());
    let original = fs::read(path.join("manifest.yaml")).unwrap();
    fs::remove_file(path.join("manifest.yaml")).unwrap();
    symlink(
        path.join("attempts/attempt-001/attempt.yaml"),
        path.join("manifest.yaml"),
    )
    .unwrap();
    assert!(f.store.run(&f.reference, None).is_err());
    fs::remove_file(path.join("manifest.yaml")).unwrap();
    fs::create_dir(path.join("manifest.yaml")).unwrap();
    assert!(f.store.run(&f.reference, None).is_err());
    fs::remove_dir(path.join("manifest.yaml")).unwrap();
    fs::write(path.join("manifest.yaml"), &original).unwrap();
    let runs = f.root.path().join("project/runs");
    fs::rename(&runs, f.root.path().join("project/old-runs")).unwrap();
    fs::create_dir(&runs).unwrap();
    fs::set_permissions(&runs, fs::Permissions::from_mode(0o700)).unwrap();
    assert!(f.store.run(&f.reference, None).is_err());
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
    let mut manifest = parse_mapping(raw("native-attempt").as_bytes(), Format::Yaml).unwrap();
    manifest.insert("attempt_id".into(), json!(attempt));
    fs::write(
        directory.join("attempt.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}
fn set_current(root: &Path, attempt: &str) {
    evidence_file(
        root,
        "status.json",
        json!({"project":"demo","run_id":"train","attempt_id":attempt,"state":"RUNNING","updated_at":"2026-10-27T03:04:05Z"}),
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
fn log_tails_preserve_python_text_separators_without_changing_jsonl_framing() {
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
    fs::write(path.join("attempts/attempt-002/attempt.yaml"), "broken: [").unwrap();
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
fn snapshot_manifest_budget_and_deleted_fence_fail_without_repair() {
    let f = fixture();
    let mut manifest = parse_mapping(raw("native-attempt").as_bytes(), Format::Yaml).unwrap();
    manifest.insert(
        "large_metadata".into(),
        Value::String("x".repeat(4 * 1024 * 1024 - 4096)),
    );
    for number in 2..=10 {
        let attempt = format!("attempt-{number:03}");
        extra_attempt(f.root.path(), &attempt);
        manifest.insert("attempt_id".into(), json!(attempt));
        fs::write(
            run_path(f.root.path())
                .join("attempts")
                .join(&attempt)
                .join("attempt.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }
    assert!(matches!(
        f.store.snapshot(&f.reference),
        Err(Error::SnapshotLimit)
    ));
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
