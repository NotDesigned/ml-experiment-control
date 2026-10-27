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
    json!({"project":"demo","project_file":"/approved/repo/experiments/project.yaml","source":"MANUAL","registered_at":"first","updated_at":"old","private_extra":{"retained":"only on no-op"}})
}
fn registry() -> Registry {
    Registry::from_value(
        &json!({"schema_version":1,"projects":[record()],"extra_top":{"keep":true}}),
        &[path(
            "/approved/repo/experiments/project.yaml",
            "/approved/repo/experiments/project.yaml",
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
                "/approved/repo/experiments/project.yaml",
                "/approved/repo/experiments/project.yaml",
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
        path: path("/approved/project.yaml", "/approved/project.yaml"),
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
    let raw = "/approved/repo/experiments/project.yaml";
    assert!(
        Registry::from_value(
            &json!({"projects":[record()]}),
            &[path(raw, raw), path(raw, "/other/project.yaml")]
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
fn paths_use_only_explicitly_approved_daemon_storage() {
    let definition=ProjectDefinition::from_value(&json!({"project":"demo","title":"","run_roots":["runs","runs","/absolute/../retained"],"base_dir":"/untrusted","authored_file":"/untrusted/file","daemon_run_root":"/untrusted/checkpoints"})).unwrap();
    let manifest = path(
        "/approved/repo/experiments/project.yaml",
        "/approved/repo/experiments/project.yaml",
    );
    let plan = projects::project_paths(&definition, &manifest).unwrap();
    assert_eq!(plan.base_dir, "/approved/repo");
    assert_eq!(
        plan.run_roots,
        vec![
            "/approved/repo/runs",
            "/approved/repo/runs",
            "/absolute/../retained"
        ]
    );
    assert!(projects::project_layout(&definition, &manifest, &[], None).is_err());
    let evidence = [
        path("/approved/repo/runs", "/approved/repo/runs"),
        path("/absolute/../retained", "/retained"),
    ];
    let layout = projects::project_layout(&definition, &manifest, &evidence, None).unwrap();
    assert_eq!(
        layout.run_roots,
        vec!["/approved/repo/runs", "/absolute/../retained"]
    );
    assert!(
        !definition
            .to_value()
            .as_object()
            .unwrap()
            .contains_key("daemon_run_root")
    );
    let layout = projects::project_layout(
        &definition,
        &manifest,
        &evidence,
        Some(&path("/daemon", "/daemon")),
    )
    .unwrap();
    assert_eq!(layout.run_roots.last().unwrap(), "/daemon");
}
#[test]
fn schema_values_are_not_scientific_or_execution_authority() {
    let definition=ProjectDefinition::from_value(&json!({"project":"demo","title":"","run_roots":[],"campaigns":[{"name":"c","file":"campaign.yaml","current_revision":{"campaign":"c","project":"demo","revision_id":"arbitrary-authored-string","file":"not-yet-read"}}],"controller":{"python":"/untrusted/interpreter","experimentctl":"/untrusted/tool"},"metrics_schema":{"definitions":{"token":{"unit":"nats/token","required_when":null},"password":{"unit":"bits/raw byte"}}}})).unwrap();
    assert_eq!(
        definition.campaign_files().collect::<Vec<_>>(),
        vec!["campaign.yaml"]
    );
    let value = definition.to_value();
    assert_eq!(
        value["metrics_schema"]["definitions"]["token"]["unit"],
        "nats/token"
    );
    assert!(
        value["metrics_schema"]["definitions"]["password"]
            .get("required_when")
            .is_none()
    );
    assert_eq!(
        value["campaigns"][0]["current_revision"]["revision_id"],
        "arbitrary-authored-string"
    );
    let raw:Value=serde_json::from_str(r#"{"project":"demo","title":"","run_roots":[],"campaigns":[{"name":"c","current_revision":{"campaign":"c","project":"demo","revision_id":"r","file":"f","memberships":[{"run_id":"r","replicate":-0}]}}]}"#).unwrap();
    let normalized = ProjectDefinition::from_value(&raw).unwrap().to_value();
    assert_eq!(
        normalized["campaigns"][0]["current_revision"]["memberships"][0]["replicate"].to_string(),
        "0"
    );
}
#[test]
fn archive_cannot_resume_directly_and_unregister_never_describes_resource_deletion() {
    let active = registry();
    let archive = active.action("demo", "archive", "completed", "t1").unwrap();
    let archived = Registry::from_value(
        &archive.commit.unwrap().payload,
        &[path(
            "/approved/repo/experiments/project.yaml",
            "/approved/repo/experiments/project.yaml",
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
