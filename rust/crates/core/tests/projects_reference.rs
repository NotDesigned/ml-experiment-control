//! Lifecycle registry and project definition outputs over synthetic path evidence.
use ml_exp_core::projects::{
    self, Change, LifecycleRecord, PathEvidence, ProjectDefinition, RegistrationSource, Registry,
    Seed,
};
use serde_json::{Value, json};

const NOW: &str = "2026-10-27T09:41:23.456789+00:00";

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}
fn path(raw: &Value) -> PathEvidence {
    PathEvidence {
        provided: raw["provided"].as_str().unwrap().into(),
        expanded_absolute: raw["expanded_absolute"].as_str().unwrap().into(),
        canonical_absolute: raw["canonical_absolute"].as_str().unwrap().into(),
    }
}
fn evidence(case: &Value) -> Vec<PathEvidence> {
    case["paths"].as_array().unwrap().iter().map(path).collect()
}
fn registry(case: &Value) -> projects::Result<Registry> {
    Registry::from_value(&case["input"]["payload"], &evidence(case))
}
fn result<T>(change: Change<T>, convert: impl FnOnce(T) -> Value) -> Value {
    json!({"result":convert(change.result),"commit":change.commit.map(|c|json!({"payload":c.payload,"event":c.event}))})
}
fn record(r: LifecycleRecord) -> Value {
    r.to_value()
}
fn records(r: Vec<LifecycleRecord>) -> Value {
    Value::Array(r.iter().map(LifecycleRecord::to_value).collect())
}

#[test]
fn registry_payloads_parse_only_from_objects() {
    let case = literal(
        r#"{"input":{"op":"parse","payload":{}},"expected":{"ok":true,"value":{"result":[],"commit":null}},"paths":[]}"#,
    );
    let parsed = registry(&case).unwrap();
    assert_eq!(
        Value::Array(
            parsed
                .records()
                .iter()
                .map(LifecycleRecord::to_value)
                .collect()
        ),
        case["expected"]["value"]["result"]
    );
    let case = literal(
        r#"{"input":{"op":"parse","payload":0},"expected":{"ok":false,"error_type":"AttributeError"},"paths":[]}"#,
    );
    assert!(registry(&case).is_err());
}

#[test]
fn register_is_a_no_op_for_the_same_path_adds_new_projects_and_refuses_duplicates() {
    for case in [
        literal(
            r#"{"input":{"op":"register","payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}]},"project":"alpha","path":"/reference/alpha/experiments/project.json"},"expected":{"ok":true,"value":{"result":{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"},"commit":null}},"paths":[{"provided":"/reference/alpha/experiments/project.json","expanded_absolute":"/reference/alpha/experiments/project.json","canonical_absolute":"/reference/alpha/experiments/project.json"}]}"#,
        ),
        literal(
            r#"{"input":{"op":"register","payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}]},"project":"beta","path":"/reference/beta/project.json"},"expected":{"ok":true,"value":{"result":{"project":"beta","project_file":"/reference/beta/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"2026-10-27T09:41:23.456789+00:00","updated_at":"2026-10-27T09:41:23.456789+00:00","state_reason":"registered"},"commit":{"payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"},{"project":"beta","project_file":"/reference/beta/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"2026-10-27T09:41:23.456789+00:00","updated_at":"2026-10-27T09:41:23.456789+00:00","state_reason":"registered"}]},"event":{"event":"REGISTER","project":"beta","at":"2026-10-27T09:41:23.456789+00:00","detail":{"project":"beta","project_file":"/reference/beta/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"2026-10-27T09:41:23.456789+00:00","updated_at":"2026-10-27T09:41:23.456789+00:00","state_reason":"registered"}}}}},"paths":[{"provided":"/reference/alpha/experiments/project.json","expanded_absolute":"/reference/alpha/experiments/project.json","canonical_absolute":"/reference/alpha/experiments/project.json"},{"provided":"/reference/beta/project.json","expanded_absolute":"/reference/beta/project.json","canonical_absolute":"/reference/beta/project.json"}]}"#,
        ),
    ] {
        let paths = evidence(&case);
        let target = paths
            .iter()
            .find(|p| p.provided == case["input"]["path"].as_str().unwrap())
            .unwrap();
        let change = registry(&case)
            .unwrap()
            .register(
                case["input"]["project"].as_str().unwrap(),
                target,
                RegistrationSource::Manual,
                NOW,
            )
            .unwrap();
        assert_eq!(result(change, record), case["expected"]["value"]);
    }
    let case = literal(
        r#"{"input":{"op":"register","payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}]},"project":"beta","path":"/reference/alpha/experiments/project.json"},"expected":{"ok":false,"error_type":"ProjectRegistryError"},"paths":[{"provided":"/reference/alpha/experiments/project.json","expanded_absolute":"/reference/alpha/experiments/project.json","canonical_absolute":"/reference/alpha/experiments/project.json"}]}"#,
    );
    let paths = evidence(&case);
    assert!(
        registry(&case)
            .unwrap()
            .register("beta", &paths[0], RegistrationSource::Manual, NOW)
            .is_err()
    );
}

#[test]
fn lifecycle_actions_record_one_event_and_refuse_a_repeated_state() {
    for case in [
        literal(
            r#"{"input":{"op":"action","payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}]},"project":"alpha","action":"pause","reason":""},"expected":{"ok":true,"value":{"result":{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"PAUSED","source":"MANUAL","registered_at":"original","updated_at":"2026-10-27T09:41:23.456789+00:00","state_reason":""},"commit":{"payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"PAUSED","source":"MANUAL","registered_at":"original","updated_at":"2026-10-27T09:41:23.456789+00:00","state_reason":""}]},"event":{"event":"PAUSED","project":"alpha","at":"2026-10-27T09:41:23.456789+00:00","detail":{"from":"ACTIVE","reason":""}}}}},"paths":[{"provided":"/reference/alpha/experiments/project.json","expanded_absolute":"/reference/alpha/experiments/project.json","canonical_absolute":"/reference/alpha/experiments/project.json"}]}"#,
        ),
        literal(
            r#"{"input":{"op":"action","payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"PAUSED","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}]},"project":"alpha","action":"resume","reason":""},"expected":{"ok":true,"value":{"result":{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"2026-10-27T09:41:23.456789+00:00","state_reason":""},"commit":{"payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"2026-10-27T09:41:23.456789+00:00","state_reason":""}]},"event":{"event":"ACTIVE","project":"alpha","at":"2026-10-27T09:41:23.456789+00:00","detail":{"from":"PAUSED","reason":""}}}}},"paths":[{"provided":"/reference/alpha/experiments/project.json","expanded_absolute":"/reference/alpha/experiments/project.json","canonical_absolute":"/reference/alpha/experiments/project.json"}]}"#,
        ),
    ] {
        let input = &case["input"];
        let change = registry(&case)
            .unwrap()
            .action(
                input["project"].as_str().unwrap(),
                input["action"].as_str().unwrap(),
                input["reason"].as_str().unwrap_or(""),
                NOW,
            )
            .unwrap();
        assert_eq!(result(change, record), case["expected"]["value"]);
    }
    let case = literal(
        r#"{"input":{"op":"action","payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"PAUSED","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}]},"project":"alpha","action":"pause","reason":""},"expected":{"ok":false,"error_type":"ApplicationError"},"paths":[{"provided":"/reference/alpha/experiments/project.json","expanded_absolute":"/reference/alpha/experiments/project.json","canonical_absolute":"/reference/alpha/experiments/project.json"}]}"#,
    );
    assert!(
        registry(&case)
            .unwrap()
            .action("alpha", "pause", "", NOW)
            .is_err()
    );
}

#[test]
fn unregister_removes_only_the_named_project_and_refuses_an_unknown_one() {
    let case = literal(
        r#"{"input":{"op":"unregister","payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"},{"project":"beta","project_file":"/reference/beta/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}],"custom":7},"project":"alpha","reason":""},"expected":{"ok":true,"value":{"result":{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"},"commit":{"payload":{"schema_version":1,"projects":[{"project":"beta","project_file":"/reference/beta/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}],"custom":7},"event":{"event":"UNREGISTER","project":"alpha","at":"2026-10-27T09:41:23.456789+00:00","detail":{"from":"ACTIVE","project_file":"/reference/alpha/experiments/project.json","reason":""}}}}},"paths":[{"provided":"/reference/alpha/experiments/project.json","expanded_absolute":"/reference/alpha/experiments/project.json","canonical_absolute":"/reference/alpha/experiments/project.json"},{"provided":"/reference/beta/project.json","expanded_absolute":"/reference/beta/project.json","canonical_absolute":"/reference/beta/project.json"}]}"#,
    );
    let change = registry(&case)
        .unwrap()
        .unregister("alpha", case["input"]["reason"].as_str().unwrap_or(""), NOW)
        .unwrap();
    assert_eq!(result(change, record), case["expected"]["value"]);
    let case = literal(
        r#"{"input":{"op":"unregister","payload":{"schema_version":1,"projects":[]},"project":"missing","target":"PAUSED","path":"/reference/new.yaml"},"expected":{"ok":false,"error_type":"ProjectRegistryError"},"paths":[{"provided":"/reference/new.yaml","expanded_absolute":"/reference/new.yaml","canonical_absolute":"/reference/new.yaml"}]}"#,
    );
    assert!(
        registry(&case)
            .unwrap()
            .unregister("missing", "", NOW)
            .is_err()
    );
}

#[test]
fn unregister_all_commits_only_when_a_project_was_registered() {
    for case in [
        literal(
            r#"{"input":{"op":"unregister_all","payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}],"custom":7},"reason":"  "},"expected":{"ok":true,"value":{"result":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}],"commit":{"payload":{"schema_version":1,"projects":[],"custom":7},"event":{"event":"UNREGISTER","project":"","at":"2026-10-27T09:41:23.456789+00:00","detail":{"reason":"  ","unregistered":[{"project":"alpha","from":"ACTIVE","project_file":"/reference/alpha/experiments/project.json"}]}}}}},"paths":[{"provided":"/reference/alpha/experiments/project.json","expanded_absolute":"/reference/alpha/experiments/project.json","canonical_absolute":"/reference/alpha/experiments/project.json"}]}"#,
        ),
        literal(
            r#"{"input":{"op":"unregister_all","payload":{"schema_version":1,"projects":[],"custom":"preserve"},"reason":""},"expected":{"ok":true,"value":{"result":[],"commit":null}},"paths":[]}"#,
        ),
    ] {
        let change = registry(&case)
            .unwrap()
            .unregister_all(case["input"]["reason"].as_str().unwrap_or(""), NOW)
            .unwrap();
        assert_eq!(result(change, records), case["expected"]["value"]);
    }
}

#[test]
fn bootstrap_records_first_creation_and_an_existing_registry_wins_over_seeds() {
    for case in [
        literal(
            r#"{"input":{"op":"bootstrap","payload":{"schema_version":1,"projects":[]},"exists":false,"seeds":[]},"expected":{"ok":true,"value":{"result":[],"commit":{"payload":{"schema_version":1,"projects":[]},"event":{"event":"BOOTSTRAP","project":"","at":"2026-10-27T09:41:23.456789+00:00","detail":{"imported":[]}}}}},"paths":[]}"#,
        ),
        literal(
            r#"{"input":{"op":"bootstrap","payload":{"schema_version":1,"projects":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}]},"exists":true,"seeds":[{"project":"alpha","path":"/reference/alpha.yaml"}]},"expected":{"ok":true,"value":{"result":[{"project":"alpha","project_file":"/reference/alpha/experiments/project.json","state":"ACTIVE","source":"MANUAL","registered_at":"original","updated_at":"old","state_reason":"original reason"}],"commit":null}},"paths":[{"provided":"/reference/alpha/experiments/project.json","expanded_absolute":"/reference/alpha/experiments/project.json","canonical_absolute":"/reference/alpha/experiments/project.json"},{"provided":"/reference/alpha.yaml","expanded_absolute":"/reference/alpha.yaml","canonical_absolute":"/reference/alpha.yaml"}]}"#,
        ),
    ] {
        let paths = evidence(&case);
        let seeds = case["input"]["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|seed| Seed {
                project: seed["project"].as_str().unwrap().into(),
                path: paths
                    .iter()
                    .find(|p| p.provided == seed["path"].as_str().unwrap())
                    .unwrap()
                    .clone(),
            })
            .collect::<Vec<_>>();
        let existing = registry(&case).unwrap();
        let change = projects::bootstrap(
            case["input"]["exists"]
                .as_bool()
                .unwrap()
                .then_some(&existing),
            &seeds,
            NOW,
        )
        .unwrap();
        assert_eq!(result(change, records), case["expected"]["value"]);
    }
}

#[test]
fn definitions_keep_unicode_and_require_a_title() {
    let definition = ProjectDefinition::from_value(
        &json!({"schema_version":1,"project":"alpha","title":"实验项目","run_roots":[]}),
    )
    .unwrap();
    assert_eq!(definition.title, "实验项目");
    assert!(definition.run_roots.is_empty());
    assert!(
        ProjectDefinition::from_value(
            &json!({"schema_version":1,"project":"alpha","run_roots":["runs"]})
        )
        .is_err()
    );
}
