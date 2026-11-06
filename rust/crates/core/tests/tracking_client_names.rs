use ml_exp_core::tracking::{self, Projection};
use serde_json::{Value, json};

fn point(name: &str, step: i64) -> Value {
    json!({"name":name,"unit":"nats/token","protocol_id":"p","variant_id":"a","value":4.2,"step":step,"epoch":2.5,"status":"VALID"})
}
fn document(items: &[Value]) -> Value {
    let mut projection = Projection::new();
    for (index, item) in items.iter().enumerate() {
        projection = projection.add(item, index as i64 + 1).unwrap();
    }
    projection
        .at(items.len() as i64)
        .unwrap()
        .client_document()
        .unwrap()
}
fn history(point: Value, display: &Value) -> Value {
    tracking::history_record(
        &json!({"seq":1,"event_id":"metric","payload":{"kind":"metrics","observations":[point]}}),
        display,
    )
    .unwrap()
}

#[test]
fn names_are_exact_in_history_and_live_and_final_summary_without_auto_suffixes() {
    for name in [
        "train/loss".into(),
        "验证 loss".into(),
        "x".repeat(100),
        "curves/custom".into(),
        "loss/by_epoch".into(),
    ] {
        let point = point(&name, 100);
        let display = document(std::slice::from_ref(&point));
        let record = history(point, &display);
        assert_eq!(record[&name], 4.2);
        assert_eq!(record[format!("ml_expd/axes/{name}/step")], 100);
        assert_eq!(record[format!("ml_expd/axes/{name}/epoch")], 2.5);
        assert!(
            record
                .as_object()
                .unwrap()
                .keys()
                .all(|key| key == &name || key.starts_with("ml_expd/"))
        );
        let summary = tracking::summary_patch(
            &display,
            &["curves/unrelated".into(), "results/unrelated".into()],
            "fixture",
        )
        .unwrap();
        assert_eq!(summary.set[&name], 4.2);
        assert_eq!(summary.set["ml_expd/display_version"], 4);
        assert!(summary.delete.is_empty());
        let specs = tracking::definition_specs(&display).unwrap();
        let visible: Vec<_> = specs
            .iter()
            .filter(|spec| spec.options["hidden"] == false)
            .collect();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].name, name);
        assert_eq!(
            visible[0].options["step_metric"],
            format!("ml_expd/axes/{name}/step")
        );
    }
}

#[test]
fn client_controls_shared_names_and_explicit_variant_names() {
    let a = point("loss", 1);
    let mut b = point("loss", 2);
    b["variant_id"] = json!("b");
    b["protocol_id"] = json!("other");
    b["value"] = json!(3.0);
    let display = document(&[a.clone(), b.clone()]);
    assert_eq!(history(a, &display)["loss"], 4.2);
    assert_eq!(history(b, &display)["loss"], 3.0);
    let summary = tracking::summary_patch(&display, &[], "fixture").unwrap();
    assert_eq!(
        summary.set["loss"], 3.0,
        "one client name means one W&B summary"
    );
    let definitions: Value = serde_json::from_str(
        summary.set["ml_expd/display_definitions_json"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(definitions["loss"]["variant_id"], "b");
    let explicit = document(&[point("loss/fast", 1), point("loss/slow", 1)]);
    assert!(
        explicit["series"]
            .as_object()
            .unwrap()
            .values()
            .all(|row| row["curve"] == row["context"]["name"])
    );
    let mut other = point("loss", 2);
    other["variant_id"] = json!("b");
    let collision = json!({"payload":{"kind":"metrics","observations":[point("loss",1),other]}});
    let collision_display = document(collision["payload"]["observations"].as_array().unwrap());
    assert!(
        tracking::curve_record(&collision, &collision_display).is_err(),
        "one history row cannot silently overwrite two values of the same key"
    );
}

#[test]
fn epoch_only_and_scalar_values_keep_the_same_name_and_failures_remove_only_owned_value() {
    let mut epoch = point("validation", 1);
    epoch.as_object_mut().unwrap().remove("step");
    let display = document(&[epoch.clone()]);
    assert_eq!(history(epoch, &display)["validation"], 4.2);
    assert_eq!(
        tracking::definition_specs(&display)
            .unwrap()
            .last()
            .unwrap()
            .options["step_metric"],
        "ml_expd/axes/validation/epoch"
    );
    let mut scalar = point("score", 1);
    scalar.as_object_mut().unwrap().remove("step");
    scalar.as_object_mut().unwrap().remove("epoch");
    assert_eq!(history(scalar.clone(), &document(&[scalar]))["score"], 4.2);
    let mut failed = point("loss", 2);
    failed["summary"] = json!(true);
    failed["status"] = json!("FAILED");
    failed["value"] = Value::Null;
    failed["error"] = json!("NON_FINITE_VALUE");
    let display = document(&[point("loss", 1), failed]);
    let patch = tracking::summary_patch(
        &display,
        &["loss".into(), "curves/client-owned".into()],
        "fixture",
    )
    .unwrap();
    assert!(!patch.set.contains_key("loss"));
    assert_eq!(patch.delete, vec!["loss"]);
}
