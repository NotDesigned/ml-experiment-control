//! Independent regressions for activation from a paused lifecycle. Tests use
//! only disposable private files, never production projects or providers.
use ml_exp_core::projects::LifecycleState;
use ml_exp_server::{
    projects::{Operation, Service, Store},
    read_config,
};
use serde_json::json;
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};
const NOW: &str = "2026-10-27T13:00:00Z";
fn id(n: u64) -> String {
    format!("{n:032x}")
}
fn private_dir(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn write(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn source() -> Vec<u8> {
    serde_json::to_vec(&json!({"schema_version":1,"project":"demo","title":"Synthetic project","run_roots":["runs"]})).unwrap()
}
fn action(name: &str) -> Operation {
    Operation::Action {
        project: "demo".into(),
        action: name.into(),
        reason: "explicit synthetic lifecycle test".into(),
    }
}
struct Fixture {
    _temp: tempfile::TempDir,
    registry: PathBuf,
    project: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        let project = temp.path().join("project");
        private_dir(&project.join("runs"));
        write(&project.join("research_project.yaml"), &source());
        read_config::initialize(&registry).unwrap();
        let fixture = Self {
            _temp: temp,
            registry,
            project,
        };
        Service::initialize(&fixture.settings(true), NOW, &id(100)).unwrap();
        fixture
    }
    fn settings(&self, approved: bool) -> read_config::Settings {
        let projects = if approved {
            vec![
                json!({"project":"demo","project_file":self.manifest(),"run_roots":[self.project.join("runs")],"runs":[],"discover_runs":false}),
            ]
        } else {
            vec![]
        };
        read_config::Settings::from_json(
            &serde_json::to_vec(
                &json!({"schema_version":1,"registry_root":self.registry,"projects":projects}),
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn manifest(&self) -> PathBuf {
        self.project.join("research_project.yaml")
    }
    fn service(&self) -> Service {
        Service::open(&self.settings(true)).unwrap()
    }
    fn store(&self) -> Store {
        Store::open(&self.registry.join("projects")).unwrap()
    }
    fn bytes(&self) -> (Vec<u8>, Vec<u8>) {
        (
            fs::read(self.registry.join("projects/registry.json")).unwrap(),
            fs::read(self.registry.join("projects/events.jsonl")).unwrap(),
        )
    }
    fn assert_paused(&self) {
        let s = self.store().snapshot(100).unwrap();
        assert_eq!(s.revision, 2);
        assert_eq!(s.projects.len(), 1);
        assert_eq!(s.projects[0].state, LifecycleState::Paused);
        assert!(!s.journal_pending);
    }
    fn resume(&self, service: &Service) {
        let result = service
            .execute(action("resume"), NOW, &id(102), Some(2))
            .unwrap();
        assert_eq!(result["active"], true);
        assert_eq!(result["project"]["project"], "demo");
        assert_eq!(result["project"]["state"], "ACTIVE");
        let s = self.store().snapshot(100).unwrap();
        assert_eq!(s.revision, 3);
        assert_eq!(s.projects[0].state, LifecycleState::Active);
        assert_eq!(s.events.len(), 3);
        assert_eq!(s.events[2]["event"], "ACTIVE");
    }
}
#[test]
fn resume_revalidates_missing_malformed_foreign_and_unsupported_sources_before_any_write() {
    for corruption in [
        "missing",
        "malformed",
        "foreign",
        "schema",
        "campaign",
        "unsafe_permissions",
    ] {
        let fixture = Fixture::new();
        let service = fixture.service();
        service
            .execute(action("pause"), NOW, &id(101), Some(1))
            .unwrap();
        let before = fixture.bytes();
        match corruption{
   "missing"=>fs::remove_file(fixture.manifest()).unwrap(),
   "malformed"=>write(&fixture.manifest(),b"project: [unfinished"),
   "foreign"=>write(&fixture.manifest(),&serde_json::to_vec(&json!({"project":"foreign","title":"Another project","run_roots":["runs"]})).unwrap()),
   "schema"=>write(&fixture.manifest(),&serde_json::to_vec(&json!({"project":"demo","title":false,"run_roots":["runs"]})).unwrap()),
   "campaign"=>write(&fixture.manifest(),&serde_json::to_vec(&json!({"project":"demo","title":"Unloaded catalog","run_roots":["runs"],"campaigns":[{"name":"unloaded","file":"campaign.yaml"}]})).unwrap()),
   "unsafe_permissions"=>fs::set_permissions(fixture.manifest(),fs::Permissions::from_mode(0o660)).unwrap(),_=>unreachable!()
  }
        let damaged = fs::read(fixture.manifest()).ok();
        assert!(
            service
                .execute(action("resume"), NOW, &id(102), Some(2))
                .is_err(),
            "{corruption}"
        );
        assert_eq!(
            fixture.bytes(),
            before,
            "rejected activation cannot alter registry: {corruption}"
        );
        assert_eq!(
            fs::read(fixture.manifest()).ok(),
            damaged,
            "no source repair: {corruption}"
        );
        fixture.assert_paused();
        let listed = service.list().unwrap();
        assert_eq!(listed["projects"][0]["state"], "PAUSED");
        assert_eq!(listed["durability"]["revision"], 2);
        assert_eq!(
            fixture.bytes(),
            before,
            "readonly list cannot repair: {corruption}"
        );
        write(&fixture.manifest(), &source());
        fixture.resume(&service);
        assert_eq!(fs::read(fixture.manifest()).unwrap(), source());
    }
}
#[test]
fn revoked_approval_cannot_reactivate_a_retained_lifecycle_record() {
    let fixture = Fixture::new();
    let service = fixture.service();
    service
        .execute(action("pause"), NOW, &id(101), Some(1))
        .unwrap();
    drop(service);
    let before = fixture.bytes();
    let revoked = Service::open(&fixture.settings(false)).unwrap();
    assert!(revoked.active_projects().unwrap().is_empty());
    assert_eq!(
        revoked.snapshot().unwrap().projects[0].state,
        LifecycleState::Paused
    );
    assert!(
        revoked
            .execute(action("resume"), NOW, &id(102), Some(2))
            .is_err()
    );
    assert_eq!(fixture.bytes(), before);
    fixture.assert_paused();
    drop(revoked);
    fixture.resume(&fixture.service());
}
#[test]
fn absent_source_does_not_block_pause_archive_restore_or_unregister_tracking() {
    let fixture = Fixture::new();
    let service = fixture.service();
    fs::remove_file(fixture.manifest()).unwrap();
    for (name, revision, expected) in [
        ("pause", 1, "PAUSED"),
        ("archive", 2, "ARCHIVED"),
        ("restore", 3, "PAUSED"),
    ] {
        let result = service
            .execute(action(name), NOW, &id(100 + revision), Some(revision))
            .unwrap();
        assert_eq!(result["active"], false);
        assert_eq!(result["project"]["state"], expected);
        assert!(!fixture.manifest().exists());
    }
    let before = fixture.bytes();
    assert!(
        service
            .execute(action("resume"), NOW, &id(104), Some(4))
            .is_err()
    );
    assert_eq!(fixture.bytes(), before);
    let result = service
        .execute(
            Operation::Unregister {
                project: "demo".into(),
                reason: "remove tracking only".into(),
            },
            NOW,
            &id(104),
            Some(4),
        )
        .unwrap();
    assert_eq!(result["unregistered"]["project"], "demo");
    let s = fixture.store().snapshot(100).unwrap();
    assert_eq!(s.revision, 5);
    assert!(s.projects.is_empty());
    assert!(!fixture.manifest().exists());
    assert!(fixture.project.join("runs").is_dir());
}
