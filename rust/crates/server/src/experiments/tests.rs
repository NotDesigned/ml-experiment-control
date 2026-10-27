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
