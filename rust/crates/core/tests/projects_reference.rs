//! Actual Python lifecycle/schema observations. Fixtures are generated before
//! testing; this executable only uses native Rust and synthetic path evidence.
use ml_exp_core::{
    canonical::python_json_bytes,
    projects::{
        self, Change, LifecycleRecord, LifecycleState, PathEvidence, ProjectDefinition,
        RegistrationSource, Registry, Seed,
    },
};
use serde_json::{Value, json};
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
fn result<T>(change: Change<T>, convert: impl FnOnce(T) -> Value) -> Value {
    json!({"result":convert(change.result),"commit":change.commit.map(|c|json!({"payload":c.payload,"event":c.event}))})
}
fn record(r: LifecycleRecord) -> Value {
    r.to_value()
}
fn records(r: Vec<LifecycleRecord>) -> Value {
    Value::Array(r.iter().map(LifecycleRecord::to_value).collect())
}
fn evaluate(case: &Value) -> projects::Result<Value> {
    let input = &case["input"];
    let now = "2026-10-27T09:41:23.456789+00:00";
    match case["category"].as_str().unwrap() {
        "registry" => {
            let paths = evidence(case);
            let registry = Registry::from_value(
                input
                    .get("payload")
                    .unwrap_or(&json!({"schema_version":1,"projects":[]})),
                &paths,
            )?;
            let project = input["project"].as_str().unwrap_or("");
            let reason = input["reason"].as_str().unwrap_or("");
            match input["op"].as_str().unwrap() {
                "parse" => Ok(
                    json!({"result":registry.records().iter().map(LifecycleRecord::to_value).collect::<Vec<_>>(),"commit":null}),
                ),
                "active" => Ok(
                    json!({"result":registry.active_records().iter().map(|r|r.to_value()).collect::<Vec<_>>(),"commit":null}),
                ),
                "register" => Ok(result(
                    registry.register(
                        project,
                        paths
                            .iter()
                            .find(|p| p.provided == input["path"].as_str().unwrap())
                            .unwrap(),
                        serde_json::from_value::<RegistrationSource>(
                            input.get("source").cloned().unwrap_or(json!("MANUAL")),
                        )
                        .unwrap(),
                        now,
                    )?,
                    record,
                )),
                "transition" => Ok(result(
                    registry.transition(
                        project,
                        serde_json::from_value::<LifecycleState>(input["target"].clone()).unwrap(),
                        reason,
                        now,
                    )?,
                    record,
                )),
                "action" => Ok(result(
                    registry.action(project, input["action"].as_str().unwrap(), reason, now)?,
                    record,
                )),
                "unregister" => Ok(result(registry.unregister(project, reason, now)?, record)),
                "unregister_all" => Ok(result(registry.unregister_all(reason, now)?, records)),
                "bootstrap" => {
                    let seeds = input["seeds"]
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
                    Ok(result(
                        projects::bootstrap(
                            input["exists"].as_bool().unwrap().then_some(&registry),
                            &seeds,
                            now,
                        )?,
                        records,
                    ))
                }
                _ => panic!("fixture operation"),
            }
        }
        "definition" => Ok(ProjectDefinition::from_value(input)?.to_value()),
        "paths" => {
            let definition = ProjectDefinition::from_value(&input["definition"])?;
            let raw = input["manifest"].as_str().unwrap();
            let manifest = PathEvidence {
                provided: raw.into(),
                expanded_absolute: raw.into(),
                canonical_absolute: raw.into(),
            };
            let candidates = projects::project_paths(&definition, &manifest)?;
            let paths = candidates
                .run_roots
                .iter()
                .map(|p| PathEvidence {
                    provided: p.clone(),
                    expanded_absolute: p.clone(),
                    canonical_absolute: lexically_resolve(p),
                })
                .collect::<Vec<_>>();
            let daemon = input["daemon"].as_str().map(|p| PathEvidence {
                provided: p.into(),
                expanded_absolute: p.into(),
                canonical_absolute: p.into(),
            });
            let layout = projects::project_layout(&definition, &manifest, &paths, daemon.as_ref())?;
            Ok(
                json!({"base_dir":layout.base_dir,"authored_file":layout.authored_file,"run_roots":layout.run_roots}),
            )
        }
        _ => panic!("fixture category"),
    }
}
fn lexically_resolve(path: &str) -> String {
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    format!("/{}", parts.join("/"))
}
#[test]
fn actual_python_project_protocols() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/projects.json")).unwrap();
    assert_eq!(fixture["baseline_commit"].as_str().unwrap().len(), 40);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let actual = evaluate(case);
        if case["expected"]["ok"] == true {
            let actual = actual.unwrap_or_else(|e| panic!("case {index} {}: {e:?}", case["input"]));
            assert_eq!(
                actual, case["expected"]["value"],
                "case {index} {}",
                case["input"]
            );
            // Canonical byte equality also freezes defaults/types/Unicode escaping; JSON
            // equality alone would miss a hash-affecting int-to-float representation.
            assert_eq!(
                python_json_bytes(&actual).unwrap(),
                python_json_bytes(&case["expected"]["value"]).unwrap(),
                "canonical case {index}"
            );
        } else {
            assert!(
                actual.is_err(),
                "case {index} unexpectedly accepted {}",
                case["input"]
            );
        }
    }
}
