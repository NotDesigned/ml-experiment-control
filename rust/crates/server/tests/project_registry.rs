//! Filesystem acceptance of the native project registry. Only synthetic private
//! temporary roots are used; no scientific files, providers or production state.
use ml_exp_core::{
    canonical::python_json_bytes,
    projects::{LifecycleState, Seed},
};
use ml_exp_server::{
    projects::{Error, Operation, Service, Store, canonical_path},
    read_config,
};
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Barrier},
    thread,
};

const NOW: &str = "2026-10-27T12:00:00Z";
const HEAD: &str = "registry.json";
const EVENTS: &str = "events.jsonl";
const LOCK: &str = ".registry.lock";

fn private_directory(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn write(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn encoded(value: &Value) -> Vec<u8> {
    let mut bytes = python_json_bytes(value).unwrap();
    bytes.push(b'\n');
    bytes
}
fn journal(events: &[Value]) -> Vec<u8> {
    events.iter().flat_map(encoded).collect()
}
fn json_file(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn transition_id(n: u64) -> String {
    format!("{n:032x}")
}
fn action(name: &str) -> Operation {
    Operation::Action {
        project: "demo".into(),
        action: name.into(),
        reason: "synthetic audit".into(),
    }
}
fn register(path: &Path) -> Operation {
    Operation::Register {
        project: "demo".into(),
        path: canonical_path(path.to_str().unwrap()).unwrap(),
    }
}
fn execute(store: &Store, op: &Operation, id: u64, expected: u64) -> Result<Value, Error> {
    store.execute(op, NOW, &transition_id(id), Some(expected), None)
}

struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    manifest: PathBuf,
}
impl Fixture {
    fn fresh() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("projects");
        let manifest = temporary.path().join("project.json");
        Self {
            _temporary: temporary,
            root,
            manifest,
        }
    }
    fn seeded() -> (Self, Store) {
        let fixture = Self::fresh();
        let seed = Seed {
            project: "demo".into(),
            path: canonical_path(fixture.manifest.to_str().unwrap()).unwrap(),
        };
        let store = Store::initialize(&fixture.root, &[seed], NOW, &transition_id(100)).unwrap();
        (fixture, store)
    }
    fn bytes(&self) -> (Vec<u8>, Vec<u8>) {
        (
            fs::read(self.root.join(HEAD)).unwrap(),
            fs::read(self.root.join(EVENTS)).unwrap(),
        )
    }
    fn entries(&self) -> Vec<String> {
        let mut names = fs::read_dir(&self.root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        names.sort();
        names
    }
    fn no_repair_on_error(&self) {
        let before = self.bytes();
        let entries = self.entries();
        assert!(Store::open(&self.root).is_err());
        assert_eq!(self.bytes(), before);
        assert_eq!(self.entries(), entries);
    }
}

#[test]
fn pending_final_transition_is_read_only_until_a_valid_explicit_mutation() {
    let (fixture, store) = Fixture::seeded();
    let original_events = fs::read(fixture.root.join(EVENTS)).unwrap();
    write(&fixture.root.join(EVENTS), b"");
    let before = fixture.bytes();
    for reader in [&store, &Store::open(&fixture.root).unwrap()] {
        let snapshot = reader.snapshot(100).unwrap();
        assert_eq!(snapshot.revision, 1);
        assert!(snapshot.journal_pending);
        assert!(snapshot.events.is_empty());
        assert_eq!(fixture.bytes(), before);
    }
    // Invalid new plans and stale CAS cannot perform an incidental repair.
    assert!(
        store
            .execute(&action("pause"), NOW, "invalid-id", Some(1), None)
            .is_err()
    );
    assert!(execute(&store, &action("pause"), 101, 0).is_err());
    assert_eq!(fixture.bytes(), before);
    execute(&store, &action("pause"), 101, 1).unwrap();
    let snapshot = store.snapshot(100).unwrap();
    assert_eq!(snapshot.revision, 2);
    assert!(!snapshot.journal_pending);
    assert_eq!(snapshot.events.len(), 2);
    assert!(
        fs::read(fixture.root.join(EVENTS))
            .unwrap()
            .starts_with(&original_events)
    );
    assert_eq!(snapshot.projects[0].state, LifecycleState::Paused);
}

#[test]
fn partial_and_unterminated_complete_tails_are_not_repaired_by_reads() {
    for tail_kind in ["partial_ascii", "complete_object", "truncated_utf8"] {
        let complete = tail_kind == "complete_object";
        let (fixture, store) = Fixture::seeded();
        let mut bytes = fs::read(fixture.root.join(EVENTS)).unwrap();
        let original = bytes.clone();
        let note = json!({"event":"NOTE","detail":"科学"});
        if complete {
            let mut tail = encoded(&note);
            tail.pop();
            bytes.extend(tail);
        } else if tail_kind == "truncated_utf8" {
            // The final UTF-8 sequence can be completed by additional bytes;
            // unlike a permanent 0xff byte this is a legitimate crash prefix.
            bytes.extend(b"{\"event\":\"\xe7\xa7");
        } else {
            bytes.extend(b"{\"event\":\"unfinished");
        }
        write(&fixture.root.join(EVENTS), &bytes);
        let before = fixture.bytes();
        let snapshot = store.snapshot(100).unwrap();
        assert!(snapshot.journal_pending);
        assert_eq!(snapshot.events.len(), if complete { 2 } else { 1 });
        assert_eq!(fixture.bytes(), before);
        drop(Store::open(&fixture.root).unwrap());
        assert_eq!(fixture.bytes(), before);
        // Re-registering this already ACTIVE project is an explicit no-op. It
        // may reconcile journal framing, but creates no lifecycle transition.
        execute(&store, &register(&fixture.manifest), 101, 1).unwrap();
        let snapshot = store.snapshot(100).unwrap();
        assert_eq!(snapshot.revision, 1);
        assert!(!snapshot.journal_pending);
        let mut expected = original;
        if complete {
            expected.extend(encoded(&note));
        }
        assert_eq!(fs::read(fixture.root.join(EVENTS)).unwrap(), expected);
    }
}

#[test]
fn missing_prefix_duplicate_transition_and_corrupt_head_are_never_repaired() {
    for corruption in [
        "missing_prefix",
        "all_missing",
        "duplicate",
        "wrong_last",
        "bad_metadata",
        "duplicate_project",
        "duplicate_manifest",
    ] {
        let (fixture, store) = Fixture::seeded();
        execute(&store, &action("pause"), 101, 1).unwrap();
        let events = store.snapshot(100).unwrap().events;
        match corruption {
            "missing_prefix" => write(&fixture.root.join(EVENTS), &journal(&events[1..])),
            "all_missing" => write(&fixture.root.join(EVENTS), b""),
            "duplicate" => write(
                &fixture.root.join(EVENTS),
                &journal(&[events[0].clone(), events[0].clone(), events[1].clone()]),
            ),
            other => {
                let mut head = json_file(&fixture.root.join(HEAD));
                match other {
                    "wrong_last" => {
                        head["_durability"]["last_transition"]["detail"] =
                            json!({"reason":"different evidence"})
                    }
                    "bad_metadata" => head["_durability"]["revision"] = json!(true),
                    "duplicate_project" => {
                        let record = head["projects"][0].clone();
                        head["projects"].as_array_mut().unwrap().push(record);
                    }
                    "duplicate_manifest" => {
                        let mut record = head["projects"][0].clone();
                        record["project"] = json!("another");
                        head["projects"].as_array_mut().unwrap().push(record);
                    }
                    _ => unreachable!(),
                }
                write(&fixture.root.join(HEAD), &encoded(&head));
            }
        }
        let before = fixture.bytes();
        assert!(store.snapshot(100).is_err(), "{corruption}");
        assert!(
            execute(&store, &action("resume"), 102, 2).is_err(),
            "{corruption}"
        );
        fixture.no_repair_on_error();
        assert_eq!(fixture.bytes(), before, "{corruption}");
    }
}

#[test]
fn complete_non_mapping_duplicate_key_and_corrupt_records_are_not_discarded_as_partial_tails() {
    for invalid in [
        b"[]".as_slice(),
        b"{\"event\":1,\"event\":2}",
        b"{\"event\":truX}",
        b"{\"value\":NaN}",
        b"{\"value\":Infinity}",
        b"{\"detail\":\"\xff\"}",
        b"{\"detail\":\"\xff",
        b"garbage\n",
        b"{}\n\n",
    ] {
        let (fixture, store) = Fixture::seeded();
        let mut bytes = fs::read(fixture.root.join(EVENTS)).unwrap();
        bytes.extend(invalid);
        write(&fixture.root.join(EVENTS), &bytes);
        let before = fixture.bytes();
        assert!(store.snapshot(100).is_err(), "invalid tail: {invalid:?}");
        assert!(
            execute(&store, &register(&fixture.manifest), 101, 1).is_err(),
            "invalid tail: {invalid:?}"
        );
        fixture.no_repair_on_error();
        assert_eq!(fixture.bytes(), before);
    }
}

#[test]
fn absent_root_and_missing_authoritative_files_are_not_bootstrapped_by_open_or_existing_store() {
    let fixture = Fixture::fresh();
    assert!(Store::open(&fixture.root).is_err());
    assert!(!fixture.root.exists());
    for missing in [HEAD, EVENTS, LOCK] {
        let (fixture, store) = Fixture::seeded();
        fs::remove_file(fixture.root.join(missing)).unwrap();
        let entries = fixture.entries();
        assert!(Store::open(&fixture.root).is_err(), "{missing}");
        assert!(store.snapshot(100).is_err(), "{missing}");
        assert!(
            execute(&store, &action("pause"), 101, 1).is_err(),
            "{missing}"
        );
        assert_eq!(fixture.entries(), entries);
        assert!(!fixture.root.join(missing).exists());
    }
}

#[test]
fn symlink_and_parent_aliases_are_refused_without_writes() {
    for leaf in [HEAD, EVENTS, LOCK] {
        {
            let (fixture, store) = Fixture::seeded();
            let path = fixture.root.join(leaf);
            let original = fixture._temporary.path().join("original");
            fs::rename(&path, &original).unwrap();
            symlink(&original, &path).unwrap();
            let before = fixture.bytes();
            assert!(Store::open(&fixture.root).is_err(), "{leaf}");
            assert!(store.snapshot(100).is_err(), "{leaf}");
            assert!(execute(&store, &action("pause"), 101, 1).is_err(), "{leaf}");
            assert_eq!(fixture.bytes(), before);
            assert_eq!(fs::read(&original).unwrap(), fs::read(&path).unwrap());
        }
    }
    let (fixture, _store) = Fixture::seeded();
    let moved = fixture._temporary.path().join("moved-projects");
    fs::rename(&fixture.root, &moved).unwrap();
    symlink(&moved, &fixture.root).unwrap();
    let before = fixture.bytes();
    assert!(Store::open(&fixture.root).is_err());
    assert_eq!(fixture.bytes(), before);
}

#[test]
fn concurrent_cas_allows_exactly_one_durable_transition() {
    let (fixture, store) = Fixture::seeded();
    let other = Store::open(&fixture.root).unwrap();
    let start = Arc::new(Barrier::new(3));
    let handles = [store, other]
        .into_iter()
        .enumerate()
        .map(|(i, store)| {
            let start = start.clone();
            thread::spawn(move || {
                start.wait();
                execute(&store, &action("pause"), 101 + i as u64, 1)
            })
        })
        .collect::<Vec<_>>();
    start.wait();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert!(
        results
            .iter()
            .any(|result| matches!(result, Err(Error::Invalid)))
    );
    let state = Store::open(&fixture.root).unwrap().snapshot(100).unwrap();
    assert_eq!(state.revision, 2);
    assert_eq!(state.events.len(), 2);
    assert_eq!(state.projects[0].state, LifecycleState::Paused);
    assert!(!state.journal_pending);
}

#[test]
fn publication_failure_after_head_replace_is_uncertain_and_explicitly_recoverable() {
    const CHILD_ROOT: &str = "ML_EXPD_SYNTHETIC_REGISTRY_LIMIT_CHILD";
    const TEST: &str =
        "publication_failure_after_head_replace_is_uncertain_and_explicitly_recoverable";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        // The parent execs this same native Rust test under a small per-file
        // limit. The already-large journal remains readable; the small atomic
        // head write succeeds, and the following append must fail with EFBIG.
        let store = Store::open(Path::new(&root)).unwrap();
        assert!(matches!(
            execute(&store, &action("pause"), 101, 1),
            Err(Error::EffectUncertain)
        ));
        return;
    }
    let (fixture, store) = Fixture::seeded();
    let mut events = store.snapshot(100).unwrap().events;
    events.extend((0..32).map(|n| json!({"event":"NOTE","n":n,"detail":"x".repeat(2048)})));
    let original_journal = journal(&events);
    assert!(original_journal.len() > 64 * 1024);
    write(&fixture.root.join(EVENTS), &original_journal);
    assert!(fs::metadata(fixture.root.join(HEAD)).unwrap().len() < 4096);
    // Constant shell text only establishes a Unix resource limit and signal
    // disposition. The binary and fixture path are separate argv/environment
    // inputs; no test value is interpolated into shell source.
    let output = Command::new("/bin/sh")
        .env_clear()
        .env(CHILD_ROOT, &fixture.root)
        .args([
            "-c",
            "trap '' XFSZ; ulimit -f 16 || exit 90; exec \"$1\" --exact \"$2\" --nocapture",
            "isolated-registry-test",
        ])
        .arg(std::env::current_exe().unwrap())
        .arg(TEST)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let pending_head = fs::read(fixture.root.join(HEAD)).unwrap();
    let head: Value = serde_json::from_slice(&pending_head).unwrap();
    assert_eq!(head["_durability"]["revision"], 2);
    assert_eq!(
        head["_durability"]["last_transition"]["transition_id"],
        transition_id(101)
    );
    assert_eq!(head["projects"][0]["state"], "PAUSED");
    assert_eq!(
        fs::read(fixture.root.join(EVENTS)).unwrap(),
        original_journal
    );
    let pending = Store::open(&fixture.root).unwrap().snapshot(100).unwrap();
    assert_eq!(pending.revision, 2);
    assert!(pending.journal_pending);
    assert_eq!(pending.events, events);
    assert_eq!(fixture.bytes(), (pending_head, original_journal.clone()));
    // The parent did not inherit the child's file limit. An explicit distinct
    // resume transition repairs only the missing audit event, then advances to
    // revision 3. The original PAUSE business transition is never re-executed.
    execute(&store, &action("resume"), 102, 2).unwrap();
    let recovered = Store::open(&fixture.root).unwrap().snapshot(100).unwrap();
    assert_eq!(recovered.revision, 3);
    assert!(!recovered.journal_pending);
    assert_eq!(recovered.projects[0].state, LifecycleState::Active);
    assert_eq!(recovered.events.len(), events.len() + 2);
    assert_eq!(
        recovered.events[events.len()],
        head["_durability"]["last_transition"]
    );
    assert_eq!(
        recovered.events[events.len() + 1]["transition_id"],
        transition_id(102)
    );
    assert_eq!(
        recovered
            .events
            .iter()
            .filter(|event| event["transition_id"] == transition_id(101))
            .count(),
        1
    );
    assert!(
        fs::read(fixture.root.join(EVENTS))
            .unwrap()
            .starts_with(&original_journal)
    );
}

struct ServiceFixture {
    _temporary: tempfile::TempDir,
    registry: PathBuf,
    a: PathBuf,
    b: PathBuf,
}
impl ServiceFixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let registry = temporary.path().join("registry");
        read_config::initialize(&registry).unwrap();
        let a = temporary.path().join("a");
        let b = temporary.path().join("b");
        for path in [&a, &b] {
            private_directory(&path.join("runs"));
            write(
                &path.join("project.json"),
                &encoded(
                    &json!({"schema_version":1,"project":"demo","title":"Synthetic project","run_roots":["runs"]}),
                ),
            );
        }
        Self {
            _temporary: temporary,
            registry,
            a,
            b,
        }
    }
    fn settings(&self, project: Option<&Path>) -> read_config::Settings {
        let projects = project.map(|path| json!({"project":"demo","project_file":path.join("project.json"),"run_roots":[path.join("runs")],"runs":[],"discover_runs":false})).into_iter().collect::<Vec<_>>();
        read_config::Settings::from_json(
            &serde_json::to_vec(
                &json!({"schema_version":1,"registry_root":self.registry,"projects":projects}),
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn store(&self) -> Store {
        Store::open(&self.registry.join("projects")).unwrap()
    }
    fn bytes(&self) -> (Vec<u8>, Vec<u8>) {
        (
            fs::read(self.registry.join("projects").join(HEAD)).unwrap(),
            fs::read(self.registry.join("projects").join(EVENTS)).unwrap(),
        )
    }
}

#[test]
fn restart_cannot_rebind_an_active_name_to_a_different_configured_manifest() {
    let fixture = ServiceFixture::new();
    let settings = fixture.settings(Some(&fixture.a));
    Service::initialize(&settings, NOW, &transition_id(100)).unwrap();
    let service = Service::open(&settings).unwrap();
    assert_eq!(
        service
            .active_projects()
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        ["demo"]
    );
    drop(service);
    let before = fixture.bytes();
    assert!(matches!(
        Service::open(&fixture.settings(Some(&fixture.b))),
        Err(Error::Binding)
    ));
    assert_eq!(fixture.bytes(), before);
    assert!(Service::open(&settings).is_ok());
    let no_longer_approved = Service::open(&fixture.settings(None)).unwrap();
    assert!(no_longer_approved.active_projects().unwrap().is_empty());
    assert_eq!(no_longer_approved.snapshot().unwrap().projects.len(), 1);
}

#[test]
fn unregister_survives_restart_and_configuration_does_not_resurrect_it() {
    let fixture = ServiceFixture::new();
    let settings = fixture.settings(Some(&fixture.a));
    Service::initialize(&settings, NOW, &transition_id(100)).unwrap();
    let service = Service::open(&settings).unwrap();
    let manifest = fixture.a.join("project.json");
    let authored = fs::read(&manifest).unwrap();
    service
        .execute(
            Operation::Unregister {
                project: "demo".into(),
                reason: "operator request".into(),
            },
            NOW,
            &transition_id(101),
            Some(1),
        )
        .unwrap();
    drop(service);
    let before = fixture.bytes();
    let reopened = Service::open(&settings).unwrap();
    assert!(reopened.snapshot().unwrap().projects.is_empty());
    assert!(reopened.active_projects().unwrap().is_empty());
    assert_eq!(reopened.snapshot().unwrap().revision, 2);
    assert!(Service::initialize(&settings, NOW, &transition_id(102)).is_err());
    assert_eq!(fixture.bytes(), before);
    assert_eq!(fs::read(&manifest).unwrap(), authored);
    // Reactivation is a distinct explicit mutation, never an open-time seed.
    reopened
        .register(
            manifest.to_str().unwrap(),
            NOW,
            &transition_id(102),
            Some(2),
        )
        .unwrap();
    assert_eq!(reopened.snapshot().unwrap().revision, 3);
    assert_eq!(
        reopened.snapshot().unwrap().projects[0].project_file,
        manifest.to_str().unwrap()
    );
}

#[test]
fn service_checks_approved_binding_under_the_mutation_lock_and_rejects_register_bypass() {
    let fixture = ServiceFixture::new();
    let settings = fixture.settings(Some(&fixture.a));
    Service::initialize(&settings, NOW, &transition_id(100)).unwrap();
    let service = Service::open(&settings).unwrap();
    let before = fixture.bytes();
    assert!(
        service
            .execute(
                register(&fixture.b.join("project.json")),
                NOW,
                &transition_id(101),
                Some(1)
            )
            .is_err()
    );
    assert_eq!(fixture.bytes(), before);
    let store = fixture.store();
    execute(
        &store,
        &Operation::Unregister {
            project: "demo".into(),
            reason: "explicit test rebind".into(),
        },
        101,
        1,
    )
    .unwrap();
    execute(&store, &register(&fixture.b.join("project.json")), 102, 2).unwrap();
    let before = fixture.bytes();
    assert!(matches!(service.snapshot(), Err(Error::Binding)));
    assert!(matches!(
        service.execute(action("pause"), NOW, &transition_id(103), Some(3)),
        Err(Error::Binding)
    ));
    assert_eq!(fixture.bytes(), before);
    assert_eq!(
        store.snapshot(100).unwrap().projects[0].state,
        LifecycleState::Active
    );
}
