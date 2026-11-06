use ml_exp_core::projects::{
    self, LifecycleRecord, LifecycleState, PathEvidence, ProjectDefinition, RegistrationSource,
    Registry, Seed,
};
use serde_json::{Value, json};
fn path(raw: &str, canonical: &str) -> PathEvidence {
    PathEvidence {
        provided: raw.into(),
        expanded_absolute: raw.into(),
        canonical_absolute: canonical.into(),
    }
}
fn record() -> Value {
    json!({"project":"demo","project_file":"/approved/repo/experiments/project.json","source":"MANUAL","registered_at":"first","updated_at":"old","private_extra":{"retained":"only on no-op"}})
}
fn registry() -> Registry {
    Registry::from_value(
        &json!({"schema_version":1,"projects":[record()],"extra_top":{"keep":true}}),
        &[path(
            "/approved/repo/experiments/project.json",
            "/approved/repo/experiments/project.json",
        )],
    )
    .unwrap()
}
#[test]
fn no_op_preserves_raw_authority_and_mutation_has_one_exact_event() {
    let original = registry();
    let unchanged = original
        .register(
            "demo",
            &path(
                "/approved/repo/experiments/project.json",
                "/approved/repo/experiments/project.json",
            ),
            RegistrationSource::Onboarding,
            "new",
        )
        .unwrap();
    assert!(unchanged.commit.is_none());
    assert_eq!(
        original.payload()["projects"][0]["private_extra"]["retained"],
        "only on no-op"
    );
    assert_eq!(unchanged.result.registered_at, "first");
    assert_eq!(unchanged.result.updated_at, "old");
    assert_eq!(unchanged.result.source, RegistrationSource::Manual);
    let changed = original.action("demo", "pause", "operator", "new").unwrap();
    let commit = changed.commit.unwrap();
    assert_eq!(
        commit.event,
        json!({"event":"PAUSED","project":"demo","at":"new","detail":{"from":"ACTIVE","reason":"operator"}})
    );
    assert_eq!(commit.payload["extra_top"], json!({"keep":true}));
    assert!(commit.payload["projects"][0].get("private_extra").is_none());
    assert_eq!(
        original.payload()["projects"][0].get("state"),
        None,
        "reducers cannot mutate their input"
    );
}
#[test]
fn first_creation_is_explicit_and_existing_empty_registry_wins_over_seeds() {
    let seed = Seed {
        project: "demo".into(),
        path: path("/approved/project.json", "/approved/project.json"),
    };
    let empty = Registry::empty();
    let result = projects::bootstrap(Some(&empty), &[seed], "now").unwrap();
    assert!(result.commit.is_none());
    assert!(result.result.is_empty());
    let first = projects::bootstrap(None, &[], "now").unwrap();
    assert!(first.commit.is_some());
    assert!(first.result.is_empty());
}
#[test]
fn contradictory_or_missing_path_facts_fail_without_inventing_resolution() {
    assert!(Registry::from_value(&json!({"projects":[record()]}), &[]).is_err());
    let raw = "/approved/repo/experiments/project.json";
    assert!(
        Registry::from_value(
            &json!({"projects":[record()]}),
            &[path(raw, raw), path(raw, "/other/project.json")]
        )
        .is_err()
    );
    for raw in ["relative.yaml", "/a\0b", "/a/../b", "/a//b"] {
        let evidence = path(raw, raw);
        assert!(evidence.validate().is_err(), "{raw:?}");
    }
    assert!(
        path("/a/../b", "/b").validate().is_ok(),
        "explicit canonical fact can establish a normalized target"
    );
}
#[test]
fn registry_schema_version_is_exactly_the_integer_one() {
    assert!(Registry::from_value(&json!({"schema_version":1,"projects":[]}), &[]).is_ok());
    for version in [json!(1.0), json!(true), json!("1"), json!(2), json!(null)] {
        assert!(
            Registry::from_value(&json!({"schema_version":version,"projects":[]}), &[]).is_err(),
            "{version}"
        );
    }
}
#[test]
fn a_project_file_holds_exactly_its_identity_title_and_run_roots() {
    let good = json!({"schema_version":1,"project":"demo","title":"","run_roots":["runs"]});
    let definition = ProjectDefinition::from_value(&good).unwrap();
    assert_eq!(definition.project(), "demo");
    assert_eq!(definition.run_roots, ["runs"]);
    for (key, value) in [
        ("controller", json!({"python":"/untrusted/interpreter"})),
        ("campaigns", json!([])),
        ("metrics_schema", json!({"definitions":{}})),
        ("base_dir", json!(".")),
    ] {
        let mut extra = good.clone();
        extra[key] = value;
        assert!(ProjectDefinition::from_value(&extra).is_err(), "{key}");
    }
    for (key, value) in [
        ("schema_version", json!(2)),
        ("project", json!("../demo")),
        ("title", json!(false)),
        ("run_roots", json!("runs")),
    ] {
        let mut bad = good.clone();
        bad[key] = value;
        assert!(ProjectDefinition::from_value(&bad).is_err(), "{key}");
    }
    let mut missing = good;
    missing.as_object_mut().unwrap().remove("schema_version");
    assert!(ProjectDefinition::from_value(&missing).is_err());
}
#[test]
fn archive_cannot_resume_directly_and_unregister_never_describes_resource_deletion() {
    let active = registry();
    let archive = active.action("demo", "archive", "completed", "t1").unwrap();
    let archived = Registry::from_value(
        &archive.commit.unwrap().payload,
        &[path(
            "/approved/repo/experiments/project.json",
            "/approved/repo/experiments/project.json",
        )],
    )
    .unwrap();
    assert!(archived.action("demo", "resume", "", "t2").is_err());
    assert!(
        archived
            .transition("demo", LifecycleState::Active, "", "t2")
            .is_err()
    );
    let restored = archived.action("demo", "restore", "review", "t2").unwrap();
    assert_eq!(restored.result.state, LifecycleState::Paused);
    let removed = archived
        .unregister("demo", "forget daemon tracking", "t3")
        .unwrap();
    assert!(
        removed.commit.unwrap().payload["projects"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let raw = record();
    let lifecycle = LifecycleRecord::from_value(&raw).unwrap();
    assert_eq!(lifecycle.state, LifecycleState::Active);
    assert_eq!(lifecycle.state_reason, "");
}
