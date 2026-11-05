use super::*;
use crate::experiments::{ProjectBinding, Store};
use serde_json::{Map, Value, json};
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
};

struct Fixture {
    temp: tempfile::TempDir,
    store: Store,
    fence: RunFence,
}
fn write(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn native(id: &str) -> Vec<u8> {
    serde_json::to_vec(
        &json!({"schema_version":1,"project":"demo","run_id":id,"source_id":"source.fixture"}),
    )
    .unwrap()
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("project/runs")).unwrap();
        write(
            &temp.path().join("project/research_project.yaml"),
            b"project: demo\ntitle: Synthetic\n",
        );
        let fence = RunFence::new(temp.path().join("registry"));
        drop(fence.shared().unwrap()); // Explicit synthetic test-only bootstrap.
        let store = Self::store(&temp, &fence, vec![temp.path().join("project/runs")]);
        Self { temp, store, fence }
    }
    fn store(temp: &tempfile::TempDir, fence: &RunFence, roots: Vec<PathBuf>) -> Store {
        Store::open(
            vec![ProjectBinding {
                project: "demo".into(),
                project_file: temp.path().join("project/research_project.yaml"),
                run_roots: roots,
            }],
            fence.clone(),
        )
        .unwrap()
    }
    fn runs(&self) -> PathBuf {
        self.temp.path().join("project/runs")
    }
    fn add(&self, relative: &str, id: &str) {
        write(
            &self.runs().join(relative).join("manifest.json"),
            &native(id),
        );
    }
    fn discover(&self) -> Result<Discovery, DiscoveryError> {
        self.store.discover("demo", &DiscoveryLimits::default())
    }
}

#[test]
fn canonical_and_instance_scoped_runs_are_unique_and_payloads_are_not_scanned() {
    let fixture = Fixture::new();
    // Keep the checked-in legacy bytes unchanged and bind their actual Project.
    write(
        &fixture.temp.path().join("project/research_project.yaml"),
        b"project: elf\n",
    );
    write(
        &fixture.runs().join("first/manifest.json"),
        b"{\"project\":\"elf\",\"run_id\":\"first\"}",
    );
    let store = Store::open(
        vec![ProjectBinding {
            project: "elf".into(),
            project_file: fixture.temp.path().join("project/research_project.yaml"),
            run_roots: vec![fixture.runs()],
        }],
        fixture.fence.clone(),
    )
    .unwrap();
    // Actual baseline discover_run_dirs supports controller state/instance/campaign.
    let scoped = fixture.runs().join("state/instance-1/campaign-1/second");
    let source: Value =
        serde_json::from_str(include_str!("../fixtures/python-manifests-v1.json")).unwrap();
    let legacy = source["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "legacy-control")
        .unwrap();
    write(
        &scoped.join("manifest.yaml"),
        legacy["raw"].as_str().unwrap().as_bytes(),
    );
    // Stop at the authoritative parent, including when a science copy has a
    // different ID or a payload entry would be unsafe outside a Run.
    write(
        &scoped.join("collected_run/manifest.yaml"),
        b"project: demo\nrun_id: mirror-copy\n",
    );
    symlink("/never-follow-training-data", scoped.join("dataset-link")).unwrap();
    let found = store.discover("elf", &DiscoveryLimits::default()).unwrap();
    assert_eq!(found.runs.len(), 2);
    assert!(found.deleted.is_empty());
    let legacy_id = legacy["metadata"]["run_id"].as_str().unwrap();
    let legacy_run = found
        .runs
        .iter()
        .find(|located| located.reference.run_id == legacy_id)
        .unwrap();
    assert_eq!(
        legacy_run.reference.relative_components,
        ["state", "instance-1", "campaign-1", "second"]
    );
    assert_eq!(legacy_run.raw_sha256, legacy["raw_sha256"]);
    assert!(
        found
            .runs
            .iter()
            .all(|located| located.reference.run_id != "mirror-copy")
    );
}

#[test]
fn duplicate_run_ids_are_rejected_even_for_identical_bytes_and_across_roots() {
    for different in [false, true] {
        let fixture = Fixture::new();
        fixture.add("one", "same");
        fixture.add("namespace/two", "same");
        if different {
            write(
                &fixture.runs().join("namespace/two/manifest.json"),
                b"{\"project\":\"demo\",\"run_id\":\"same\",\"seed\":2}",
            );
        }
        assert!(matches!(
            fixture.discover(),
            Err(DiscoveryError::DuplicateRun)
        ));
    }
    let fixture = Fixture::new();
    fixture.add("one", "same");
    let second = fixture.temp.path().join("second-root");
    write(&second.join("other/manifest.json"), &native("same"));
    let store = Fixture::store(&fixture.temp, &fixture.fence, vec![fixture.runs(), second]);
    assert!(matches!(
        store.discover("demo", &DiscoveryLimits::default()),
        Err(DiscoveryError::DuplicateRun)
    ));
}

#[test]
fn canonical_corruption_foreign_identity_and_science_only_copies_are_not_absence_or_fallback() {
    for case in ["corrupt", "foreign", "science-only"] {
        let fixture = Fixture::new();
        let run = fixture.runs().join("trial");
        match case {
            "corrupt" => {
                write(&run.join("manifest.yaml"), b"broken: [");
            }
            "foreign" => write(
                &run.join("manifest.json"),
                b"{\"project\":\"other\",\"run_id\":\"trial\"}",
            ),
            "science-only" => write(
                &run.join("collected_run/manifest.yaml"),
                b"project: demo\nrun_id: copied\n",
            ),
            _ => unreachable!(),
        }
        let result = fixture.discover();
        if case == "science-only" {
            assert!(matches!(result, Err(DiscoveryError::UnsupportedLayout)));
        } else {
            assert!(matches!(result, Err(DiscoveryError::Document(_))));
        }
    }
}

#[test]
fn linked_manifests_unsafe_permissions_and_names_fail_closed() {
    for case in ["linked-manifest", "writable-directory", "unsafe-name"] {
        let fixture = Fixture::new();
        match case {
            "linked-manifest" => {
                fixture.add("good", "trial");
                fs::hard_link(
                    fixture.runs().join("good/manifest.json"),
                    fixture.temp.path().join("copy.json"),
                )
                .unwrap();
            }
            "writable-directory" => {
                fixture.add("group", "trial");
                fs::set_permissions(
                    fixture.runs().join("group"),
                    fs::Permissions::from_mode(0o777),
                )
                .unwrap();
            }
            "unsafe-name" => fixture.add("invalid component", "trial"),
            _ => unreachable!(),
        }
        assert!(fixture.discover().is_err(), "{case}");
    }
}

#[test]
fn links_and_files_beside_runs_are_ignored_and_never_followed() {
    let fixture = Fixture::new();
    fixture.add("good", "trial");
    symlink("good", fixture.runs().join("directory-alias")).unwrap();
    symlink(
        "/never-follow-training-data",
        fixture.runs().join("dangling"),
    )
    .unwrap();
    write(&fixture.runs().join("note"), b"opaque");
    let found = fixture.discover().unwrap();
    assert_eq!(found.runs.len(), 1);
    assert_eq!(found.runs[0].reference.relative_components, ["good"]);
}

#[test]
fn discovery_preserves_deleted_identity_and_never_bootstraps_or_repairs_fences() {
    let fixture = Fixture::new();
    fixture.add("live", "live");
    fixture.add("removed", "removed");
    fixture
        .fence
        .mark("demo", "removed", &Map::new(), "synthetic")
        .unwrap();
    let discovery = fixture.discover().unwrap();
    assert_eq!(
        discovery
            .runs
            .iter()
            .map(|v| v.reference.run_id.as_str())
            .collect::<Vec<_>>(),
        ["live"]
    );
    assert_eq!(
        discovery
            .deleted
            .iter()
            .map(|v| v.reference.run_id.as_str())
            .collect::<Vec<_>>(),
        ["removed"]
    );
    let missing = fixture.temp.path().join("registry/.run-deletion.lock");
    fs::remove_file(&missing).unwrap();
    assert!(matches!(fixture.discover(), Err(DiscoveryError::Fence(_))));
    assert!(!missing.exists());
    assert!(matches!(
        fixture
            .store
            .discover("foreign", &DiscoveryLimits::default()),
        Err(DiscoveryError::UnknownProject)
    ));
}

#[test]
fn traversal_metadata_and_pending_directory_budgets_are_enforced_without_partial_catalogs() {
    for case in ["depth", "directories", "entries", "runs", "metadata"] {
        let fixture = Fixture::new();
        fixture.add("group/first", "first");
        fixture.add("second", "second");
        write(&fixture.runs().join("note"), b"not a manifest");
        let mut limits = DiscoveryLimits::default();
        match case {
            "depth" => limits.max_depth = 1,
            "directories" => limits.max_directories = 2,
            "entries" => limits.max_entries = 2,
            "runs" => limits.max_runs = 1,
            "metadata" => limits.max_metadata_bytes = 4,
            _ => unreachable!(),
        }
        assert!(
            matches!(
                fixture.store.discover("demo", &limits),
                Err(DiscoveryError::Budget)
            ),
            "{case}"
        );
    }
    assert!(
        DiscoveryLimits {
            max_depth: 0,
            ..DiscoveryLimits::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        DiscoveryLimits {
            max_depth: 33,
            ..DiscoveryLimits::default()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn duplicate_and_overlapping_approved_roots_are_rejected_before_traversal() {
    for nested in [false, true] {
        let fixture = Fixture::new();
        let second = if nested {
            fixture.runs().join("nested")
        } else {
            fixture.runs()
        };
        fs::create_dir_all(&second).unwrap();
        let store = Fixture::store(&fixture.temp, &fixture.fence, vec![fixture.runs(), second]);
        assert!(matches!(
            store.discover("demo", &DiscoveryLimits::default()),
            Err(DiscoveryError::OverlappingRoots)
        ));
    }
}
