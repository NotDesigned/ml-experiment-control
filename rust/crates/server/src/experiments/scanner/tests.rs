use super::*;
use crate::experiments::{ProjectBinding, Store};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};
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
            &temp.path().join("project/project.json"),
            br#"{"schema_version":1,"project":"demo","title":"Synthetic"}"#,
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
                project_file: temp.path().join("project/project.json"),
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
        self.store.discover("demo")
    }
}

#[test]
fn canonical_and_instance_scoped_runs_are_unique_and_payloads_are_not_scanned() {
    let fixture = Fixture::new();
    fixture.add("first", "first");
    // Controller state/instance/campaign layouts are found at any depth.
    let scoped = fixture.runs().join("state/instance-1/campaign-1/second");
    let bytes = native("second");
    write(&scoped.join("manifest.json"), &bytes);
    // Stop at the authoritative parent, including when a science copy has a
    // different ID or a payload entry would be unsafe outside a Run.
    write(
        &scoped.join("collected_run/manifest.json"),
        &native("mirror-copy"),
    );
    symlink("/never-follow-training-data", scoped.join("dataset-link")).unwrap();
    let found = fixture.discover().unwrap();
    assert_eq!(found.runs.len(), 2);
    assert!(found.deleted.is_empty());
    let second = found
        .runs
        .iter()
        .find(|located| located.reference.run_id == "second")
        .unwrap();
    assert_eq!(
        second.reference.relative_components,
        ["state", "instance-1", "campaign-1", "second"]
    );
    assert_eq!(
        second.raw_sha256,
        format!("sha256:{:x}", Sha256::digest(&bytes))
    );
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
        store.discover("demo"),
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
                write(&run.join("manifest.json"), b"broken: [");
            }
            "foreign" => write(
                &run.join("manifest.json"),
                b"{\"project\":\"other\",\"run_id\":\"trial\"}",
            ),
            "science-only" => write(&run.join("collected_run/manifest.json"), &native("copied")),
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
fn unsafe_run_names_fail_closed() {
    let fixture = Fixture::new();
    fixture.add("invalid component", "trial");
    assert!(fixture.discover().is_err());
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
        fixture.store.discover("foreign"),
        Err(DiscoveryError::UnknownProject)
    ));
}

#[test]
fn discovery_reaches_as_deep_as_a_run_reference_can_name_and_no_further() {
    let fixture = Fixture::new();
    fixture.add(&vec!["d"; MAX_DEPTH].join("/"), "deepest");
    let found = fixture.discover().unwrap();
    assert_eq!(found.runs[0].reference.relative_components.len(), MAX_DEPTH);
    fixture.add(&vec!["e"; MAX_DEPTH + 1].join("/"), "too-deep");
    assert!(matches!(fixture.discover(), Err(DiscoveryError::Depth)));
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
            store.discover("demo"),
            Err(DiscoveryError::OverlappingRoots)
        ));
    }
}
