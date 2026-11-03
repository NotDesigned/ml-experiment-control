//! Real existing files and SQLite, without a native workspace marker. No
//! provider credentials, source payloads or old production state are used.
use ml_exp_core::{
    checkpoints::{self, Context},
    records::{Event, normalized_metric},
    state_audit::compare,
};
use ml_exp_server::{deletion::RunFence, read_config, records, state_audit};
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
};
fn private(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn directory(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
struct Fixture {
    _temp: tempfile::TempDir,
    settings: read_config::Settings,
    run: PathBuf,
    config: Value,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        // Explicit fixture seeding, not an operation performed by the audit.
        drop(RunFence::new(&registry).shared().unwrap());
        let project = temp.path().join("demo");
        let run = project.join("runs/train");
        let attempt = run.join("attempts/attempt-001");
        directory(&attempt);
        private(
            &project.join("project.json"),
            &json!({"schema_version":1,"project":"demo","title":"audit fixture"}),
        );
        let manifest = json!({"schema_version":1,"identity_version":2,"project":"demo","run_id":"train","source_id":format!("source.{}","a".repeat(64)),"image_id":format!("sha256:{}","b".repeat(64)),"config_path":"config.json","seed":42,"campaign":"fixture","evaluation":{"protocol_id":"fixture-protocol"},"parameters":{},"resources":{"gpus":0},"container":{"runtime_id":"runtime.fixture"},"backend":{"kind":"slurm"},"inputs":[]});
        private(&run.join("manifest.yaml"), &manifest);
        let mut am = manifest.clone();
        am["attempt_id"] = json!("attempt-001");
        private(&attempt.join("attempt.yaml"), &am);
        let status = json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","state":"SUCCEEDED"});
        private(&run.join("status.json"), &status);
        private(&attempt.join("status.json"), &status);
        let config = json!({"ml_expd":{"project":"demo","run_id":"train","source_id":manifest["source_id"],"image_id":manifest["image_id"],"resources":manifest["resources"],"evaluation":manifest["evaluation"],"parameters":{},"runtime_id":"runtime.fixture","backend":"slurm","inputs":[]}});
        let settings = read_config::Settings::from_json(&serde_json::to_vec(&json!({"schema_version":1,"registry_root":registry,"projects":[{"project":"demo","project_file":project.join("project.json"),"run_roots":[project.join("runs")]}]})).unwrap()).unwrap();
        Self {
            _temp: temp,
            settings,
            run,
            config,
        }
    }
    fn journal(&self) {
        let store = records::Store::open(&self.settings.registry_root, &"c".repeat(32)).unwrap();
        let guard = store.guard("demo", "train").unwrap();
        store
            .bind(&guard, &json!({"enabled":false}), &self.config)
            .unwrap();
        let scope = store.scope(&guard, "attempt-001").unwrap();
        for (i, variant) in ["left", "right"].into_iter().enumerate() {
            let record = json!({"name":"loss","unit":"nats/token","value":2.0+i as f64,"step":10-i,"protocol_id":"fixture-protocol","variant_id":variant});
            store
                .append(
                    &guard,
                    &scope,
                    &[Event {
                        event_id: format!("metric-{i}"),
                        payload: normalized_metric(&record, &self.config).unwrap(),
                    }],
                )
                .unwrap();
        }
    }
    fn checkpoint(&self) -> PathBuf {
        let context = Context {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
            source_id: self.config["ml_expd"]["source_id"].as_str().unwrap().into(),
            image_id: self.config["ml_expd"]["image_id"].as_str().unwrap().into(),
            storage_scope: "storage.fixture".into(),
            state_root: "/shared/demo/train/state".into(),
        };
        let ready = json!({"step":10,"files":[{"path":"checkpoints/step10/model.bin","bytes":4,"sha256":"d".repeat(64)}]});
        let mut checkpoint = checkpoints::document(&context, &ready).unwrap();
        checkpoint["status"] = json!("REGISTERED");
        checkpoint["verification"] = json!("worker-sha256");
        checkpoint["registered_at"] = json!("2026-10-29T00:00:00Z");
        let cdir = self
            .settings
            .registry_root
            .join("persistent-checkpoints/demo/train/attempt-001");
        directory(&cdir);
        let path = cdir.join(format!(
            "{}.json",
            checkpoint["checkpoint_id"].as_str().unwrap()
        ));
        private(&path, &checkpoint);
        let tdir = self
            .settings
            .registry_root
            .join("artifact-transfers/demo/train");
        directory(&tdir);
        private(
            &tdir.join("attempt-001.json"),
            &json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","checkpoint_state":context,"secret":"must-not-appear-in-audit"}),
        );
        path
    }
    fn api(&self) -> ml_exp_server::read_api::Api {
        private(
            &self
                .settings
                .registry_root
                .join(".rust-preview-workspace.json"),
            &json!({"contract":"ml-expd-rust-preview.v1","workspace_id":format!("rust-preview-{}","1".repeat(32))}),
        );
        ml_exp_server::read_api::Api::open(self.settings.clone()).unwrap()
    }
}
fn inventory(path: &Path) -> Vec<(PathBuf, String)> {
    let mut result = Vec::new();
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        let meta = fs::symlink_metadata(&path).unwrap();
        if meta.is_dir() {
            result.extend(inventory(&path));
        } else if meta.is_file() {
            result.push((
                path.clone(),
                ml_exp_core::slurm::helper::sha256_bytes(&fs::read(&path).unwrap()),
            ));
        }
    }
    result.sort();
    result
}
#[test]
fn legacy_namespace_is_observed_without_initialization_or_payload_access() {
    let f = Fixture::new();
    f.journal();
    f.checkpoint();
    // Reading this prohibited log path would fail. Metadata-only audit must not
    // consult training logs or inspect scientific payload directories.
    symlink(
        "/unavailable-scientific-payload",
        f.run.join("attempts/attempt-001/stdout.log"),
    )
    .unwrap();
    let before = inventory(f._temp.path());
    let first = state_audit::capture(&f.settings, None).unwrap();
    assert!(first.issues.is_empty());
    assert_eq!(inventory(f._temp.path()), before);
    assert!(
        !f.settings
            .registry_root
            .join(".rust-preview-workspace.json")
            .exists()
    );
    assert!(
        compare(&first, &state_audit::capture(&f.settings, None).unwrap())
            .unwrap()
            .equivalent
    );
    let text = serde_json::to_string(&first).unwrap();
    assert!(!text.contains("must-not-appear-in-audit"));
    assert!(!text.contains("/shared/"));
    assert!(
        first
            .entries
            .iter()
            .any(|e| e.key.kind == "metrics" && e.records == 2)
    );
    let db = f.settings.registry_root.join("tracking/outbox.sqlite");
    let connection = rusqlite::Connection::open(&db).unwrap();
    connection
        .execute("UPDATE scopes SET confirmed=1", [])
        .unwrap();
    drop(connection);
    let later = state_audit::capture(&f.settings, None).unwrap();
    let compared = compare(&first, &later).unwrap();
    assert!(!compared.equivalent);
    assert!(compared.changed.iter().any(|k| k.kind == "database"));
}
#[test]
fn corrupted_checkpoint_and_foreign_journal_are_issues_without_repair() {
    let f = Fixture::new();
    f.journal();
    let path = f.checkpoint();
    let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    value["files"][0]["sha256"] = json!("e".repeat(64));
    private(&path, &value);
    let first = state_audit::capture(&f.settings, None).unwrap();
    assert!(
        first
            .issues
            .iter()
            .any(|i| i.code == "CHECKPOINT_IDENTITY_CONFLICT")
    );
    assert!(!compare(&first, &first).unwrap().equivalent);
    let db = f.settings.registry_root.join("tracking/outbox.sqlite");
    let connection = rusqlite::Connection::open(db).unwrap();
    let mut config = f.config.clone();
    config["ml_expd"]["image_id"] = json!("foreign");
    connection
        .execute(
            "UPDATE runs SET config=?",
            [serde_json::to_string(&config).unwrap()],
        )
        .unwrap();
    drop(connection);
    let before = inventory(f._temp.path());
    let after = state_audit::capture(&f.settings, None).unwrap();
    assert!(
        after
            .issues
            .iter()
            .any(|i| i.code == "JOURNAL_IDENTITY_CONFLICT")
    );
    assert_eq!(before, inventory(f._temp.path()));
}
#[test]
fn missing_locks_unsafe_entries_and_absent_directories_are_not_created() {
    let f = Fixture::new();
    fs::remove_file(f.settings.registry_root.join(".run-deletion.lock")).unwrap();
    assert!(state_audit::capture(&f.settings, None).is_err());
    assert!(!f.settings.registry_root.join(".run-deletion.lock").exists());
    let f = Fixture::new();
    symlink("/outside", f.settings.registry_root.join("tracking")).unwrap();
    assert!(state_audit::capture(&f.settings, None).is_err());
    let f = Fixture::new();
    let before = inventory(f._temp.path());
    assert!(
        state_audit::capture(&f.settings, None)
            .unwrap()
            .issues
            .is_empty()
    );
    assert_eq!(before, inventory(f._temp.path()));
    assert!(!f.settings.registry_root.join("tracking").exists());
}
#[test]
fn actual_audit_cli_compares_private_snapshots_and_reports_a_changed_identity() {
    let f = Fixture::new();
    let config = f._temp.path().join("reads.json");
    private(&config, &serde_json::to_value(&f.settings).unwrap());
    let execute = |extra: &[&std::ffi::OsStr]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_ml-expd-state-audit"))
            .arg("--config")
            .arg(&config)
            .args(extra)
            .output()
            .unwrap()
    };
    let first = execute(&[]);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let baseline = f._temp.path().join("baseline.json");
    fs::write(&baseline, &first.stdout).unwrap();
    fs::set_permissions(&baseline, fs::Permissions::from_mode(0o600)).unwrap();
    let compared = execute(&[std::ffi::OsStr::new("--compare"), baseline.as_os_str()]);
    assert!(compared.status.success());
    let value: Value = serde_json::from_slice(&compared.stdout).unwrap();
    assert_eq!(value["equivalent"], true);
    assert_eq!(value["production_cutover_authorized"], false);
    let status = f.run.join("status.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&status).unwrap()).unwrap();
    value["state"] = json!("FAILED");
    private(&status, &value);
    let changed = execute(&[std::ffi::OsStr::new("--compare"), baseline.as_os_str()]);
    assert_eq!(changed.status.code(), Some(2));
    let value: Value = serde_json::from_slice(&changed.stdout).unwrap();
    assert_eq!(value["equivalent"], false);
    assert!(!value["changed"].as_array().unwrap().is_empty());
}
#[test]
fn journal_schema_versions_and_duplicate_rows_are_preserved_without_row_order_assumptions() {
    let f = Fixture::new();
    f.journal();
    let db = f.settings.registry_root.join("tracking/outbox.sqlite");
    let change = |sql: &str| {
        rusqlite::Connection::open(&db)
            .unwrap()
            .execute_batch(sql)
            .unwrap();
    };
    change("CREATE TABLE audit_bag(value TEXT); INSERT INTO audit_bag VALUES ('a'),('b'),('a');");
    let first = state_audit::capture(&f.settings, None).unwrap();
    change("DELETE FROM audit_bag; INSERT INTO audit_bag VALUES ('b'),('a'),('a');");
    assert!(
        compare(&first, &state_audit::capture(&f.settings, None).unwrap())
            .unwrap()
            .equivalent
    );
    change("DELETE FROM audit_bag WHERE rowid=(SELECT MIN(rowid) FROM audit_bag WHERE value='a');");
    assert!(
        !compare(&first, &state_audit::capture(&f.settings, None).unwrap())
            .unwrap()
            .equivalent
    );
    let before = state_audit::capture(&f.settings, None).unwrap();
    change("PRAGMA user_version=11;");
    assert!(
        !compare(&before, &state_audit::capture(&f.settings, None).unwrap())
            .unwrap()
            .equivalent
    );
}
#[test]
fn immutable_action_plans_are_hashed_but_never_executed_or_echoed() {
    let f = Fixture::new();
    let actions = f._temp.path().join("actions");
    directory(&actions);
    let action = "action-0123456789abcdef";
    let dir = actions.join(action);
    directory(&dir);
    private(
        &dir.join("plan.json"),
        &json!({"action_id":action,"payload":"do-not-echo-this-command-or-secret"}),
    );
    let first = state_audit::capture(&f.settings, Some(&actions)).unwrap();
    assert!(first.issues.is_empty());
    assert!(
        !serde_json::to_string(&first)
            .unwrap()
            .contains("do-not-echo")
    );
    let before = inventory(f._temp.path());
    assert!(
        compare(
            &first,
            &state_audit::capture(&f.settings, Some(&actions)).unwrap()
        )
        .unwrap()
        .equivalent
    );
    assert_eq!(before, inventory(f._temp.path()));
    assert!(!actions.join("actions.sqlite3").exists());
    private(
        &dir.join("plan.json"),
        &json!({"action_id":action,"payload":"changed"}),
    );
    let compared = compare(
        &first,
        &state_audit::capture(&f.settings, Some(&actions)).unwrap(),
    )
    .unwrap();
    assert!(!compared.equivalent);
    assert!(compared.changed.iter().any(|k| k.kind == "action-plan"));
}

#[test]
fn historical_checkpoints_are_read_without_native_submission_or_registry_repair() {
    let f = Fixture::new();
    let path = f.checkpoint();
    let mut original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let id = original["checkpoint_id"].as_str().unwrap().to_owned();
    original["private_extension"] = json!("do-not-return-this");
    private(&path, &original);
    let api = f.api();
    let before = inventory(f._temp.path());
    let list = api
        .historical_checkpoints("demo", "train", "attempt-001", None)
        .unwrap()
        .unwrap();
    assert_eq!(list["checkpoints"].as_array().unwrap().len(), 1);
    let value = api
        .historical_checkpoints("demo", "train", "attempt-001", Some(&id))
        .unwrap()
        .unwrap();
    assert_eq!(value, list["checkpoints"][0]);
    assert_eq!(value["checkpoint_id"], id);
    assert_eq!(value["files"], original["files"]);
    assert_eq!(value["registered_at"], original["registered_at"]);
    assert!(value.get("private_extension").is_none());
    assert_eq!(inventory(f._temp.path()), before);
    assert!(
        !f.settings
            .registry_root
            .join("persistent-checkpoints/.checkpoints.lock")
            .exists()
    );
}

#[test]
fn historical_checkpoint_corruption_foreign_storage_and_symlinks_fail_without_fallback() {
    let f = Fixture::new();
    let path = f.checkpoint();
    let original = fs::read(&path).unwrap();
    let mut value: Value = serde_json::from_slice(&original).unwrap();
    let id = value["checkpoint_id"].as_str().unwrap().to_owned();
    let api = f.api();
    for (key, replacement) in [
        ("source_id", json!(format!("source.{}", "b".repeat(64)))),
        ("storage_scope", json!("storage.foreign")),
        ("status", json!("PENDING")),
    ] {
        let mut corrupt = value.clone();
        corrupt[key] = replacement;
        private(&path, &corrupt);
        assert_eq!(
            api.historical_checkpoints("demo", "train", "attempt-001", Some(&id))
                .err()
                .unwrap()
                .status,
            409
        );
    }
    value["files"][0]["sha256"] = json!("e".repeat(64));
    private(&path, &value);
    assert_eq!(
        api.historical_checkpoints("demo", "train", "attempt-001", None)
            .err()
            .unwrap()
            .status,
        409
    );
    fs::write(&path, &original).unwrap();
    let transfer = f
        .settings
        .registry_root
        .join("artifact-transfers/demo/train/attempt-001.json");
    let bytes = fs::read(&transfer).unwrap();
    let mut value: Value = serde_json::from_slice(&bytes).unwrap();
    value["checkpoint_state"]["attempt_id"] = json!("attempt-002");
    private(&transfer, &value);
    assert_eq!(
        api.historical_checkpoints("demo", "train", "attempt-001", Some(&id))
            .err()
            .unwrap()
            .status,
        409
    );
    fs::write(&transfer, bytes).unwrap();
    fs::remove_file(&path).unwrap();
    symlink("/unavailable-scientific-payload", &path).unwrap();
    assert_eq!(
        api.historical_checkpoints("demo", "train", "attempt-001", Some(&id))
            .err()
            .unwrap()
            .status,
        409
    );
    assert!(fs::symlink_metadata(&path).unwrap().is_symlink());
}

#[test]
fn withdrawn_deleted_absent_and_native_checkpoint_scopes_remain_distinct() {
    let f = Fixture::new();
    let path = f.checkpoint();
    let id = path.file_stem().unwrap().to_str().unwrap();
    let api = f.api();
    private(
        &path.with_file_name(format!("withdrawn.{id}.json")),
        &json!({"checkpoint_id":id,"status":"REMOVAL_REQUESTED"}),
    );
    assert_eq!(
        api.historical_checkpoints("demo", "train", "attempt-001", Some(id))
            .err()
            .unwrap()
            .status,
        404
    );
    assert_eq!(
        api.historical_checkpoints("demo", "train", "attempt-001", None)
            .unwrap()
            .unwrap()["checkpoints"],
        json!([])
    );
    assert_eq!(
        api.historical_checkpoints("demo", "train", "attempt-002", None)
            .err()
            .unwrap()
            .status,
        404
    );
    let mut run: Value =
        serde_json::from_slice(&fs::read(f.run.join("manifest.yaml")).unwrap()).unwrap();
    run["native_creation"] = json!({"format":"unknown-native-format"});
    private(&f.run.join("manifest.yaml"), &run);
    assert!(
        api.historical_checkpoints("demo", "train", "attempt-001", Some(id))
            .unwrap()
            .is_none()
    );
    let deleted = f.settings.registry_root.join("deleted-runs/demo");
    directory(&deleted);
    private(
        &deleted.join("train.json"),
        &json!({"project":"demo","run_id":"train","state":"DELETED"}),
    );
    assert_eq!(
        api.historical_checkpoints("demo", "train", "attempt-001", Some(id))
            .err()
            .unwrap()
            .status,
        409
    );
}

#[test]
fn source_and_runtime_control_receipts_are_bound_without_reading_payload_trees() {
    let f = Fixture::new();
    let source_id = format!("source.{}", "a".repeat(64));
    let source = f
        .settings
        .registry_root
        .join("source-revisions/sources/demo")
        .join(&source_id);
    directory(&source);
    private(
        &source.join("source.json"),
        &json!({"project":"demo","source_id":source_id,"tree_digest":format!("sha256:{}", "a".repeat(64)),"tree":"tree"}),
    );
    symlink("/unavailable-source-payload", source.join("tree")).unwrap();
    let runtime = f.settings.registry_root.join("runtime-bundles/demo");
    directory(&runtime);
    let id = format!("runtime.{}", "b".repeat(64));
    let path = runtime.join(format!("{id}.json"));
    private(
        &path,
        &json!({"project":"demo","runtime_id":id,"status":"READY","image_id":"sha256:fixture","credential":"never-echo-control-secrets"}),
    );
    // Missing historical seed is preserved, never invented or promoted to
    // native execution authority by a successful metadata comparison.
    for path in [
        f.run.join("manifest.yaml"),
        f.run.join("attempts/attempt-001/attempt.yaml"),
    ] {
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("seed");
        private(&path, &value);
    }
    let before = inventory(f._temp.path());
    let first = state_audit::capture(&f.settings, None).unwrap();
    assert!(first.issues.is_empty());
    assert!(first.entries.iter().any(|v| v.key.kind == "source-receipt"));
    assert!(
        first
            .entries
            .iter()
            .any(|v| v.key.kind == "runtime-receipt")
    );
    assert!(
        !serde_json::to_string(&first)
            .unwrap()
            .contains("never-echo")
    );
    assert_eq!(inventory(f._temp.path()), before);
    private(
        &path,
        &json!({"project":"demo","runtime_id":id,"status":"READY","image_id":"sha256:changed"}),
    );
    let changed = compare(&first, &state_audit::capture(&f.settings, None).unwrap()).unwrap();
    assert!(!changed.equivalent);
    assert!(changed.changed.iter().any(|v| v.kind == "runtime-receipt"));
    assert!(!changed.production_cutover_authorized);
}

#[test]
fn cold_wal_database_does_not_create_auxiliary_files_for_a_read_only_audit() {
    let f = Fixture::new();
    let actions = f._temp.path().join("cold-actions");
    directory(&actions);
    let database = actions.join("actions.sqlite3");
    let connection = rusqlite::Connection::open(&database).unwrap();
    connection.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE preserved(id TEXT); INSERT INTO preserved VALUES ('original');").unwrap();
    drop(connection);
    fs::set_permissions(&database, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(!actions.join("actions.sqlite3-shm").exists());
    let before = inventory(f._temp.path());
    let error = state_audit::capture(&f.settings, Some(&actions))
        .err()
        .unwrap();
    assert_eq!(error.code(), "AUDIT_DATABASE_READ_ONLY_UNAVAILABLE");
    assert_eq!(before, inventory(f._temp.path()));
    assert!(!actions.join("actions.sqlite3-shm").exists());
}
