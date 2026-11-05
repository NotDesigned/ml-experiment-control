//! New display contract, independent of frozen v2 Python parity fixtures.
use ml_exp_core::tracking::{self, Projection};
use serde_json::{Value, json};

fn point(protocol: &str, name: &str, variant: Value, step: i64) -> Value {
    json!({"name":name,"unit":"nats/token","protocol_id":protocol,"dataset_id":"train",
        "variant_id":variant,"value":4.0,"step":step,"status":"VALID"})
}
fn document(items: &[Value]) -> tracking::Result<Value> {
    let mut projection = Projection::new();
    for (index, item) in items.iter().enumerate() {
        projection = projection.add(item, index as i64 + 1)?;
    }
    projection.at(items.len() as i64)?.stable_document()
}
fn row(display: &Value) -> &Value {
    display["series"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()
}

#[test]
fn cross_run_protocol_and_dataset_differences_do_not_rename_metrics() {
    let first = point("source-seed-lr-a", "train_loss", Value::Null, 1);
    let mut second = point("source-seed-lr-b", "train_loss", Value::Null, 1);
    second["dataset_id"] = json!("another-version");
    let a = document(&[first]).unwrap();
    let b = document(&[second]).unwrap();
    assert_eq!(row(&a)["curve"], "curves/train_loss");
    assert_eq!(row(&a)["curve"], row(&b)["curve"]);
    assert_eq!(row(&a)["result"], "results/train_loss");
    assert_ne!(row(&a)["context"], row(&b)["context"]);
    assert!(row(&a).get("shadow").is_none());
}

#[test]
fn original_names_keep_unicode_slashes_and_more_than_64_characters() {
    for name in ["验证/loss".to_owned(), "x".repeat(100), "a b".to_owned()] {
        let display = document(&[point("p", &name, Value::Null, 1)]).unwrap();
        assert_eq!(row(&display)["curve"], format!("curves/{name}"));
    }
}

#[test]
fn readable_variant_names_are_stable_before_and_after_other_variants_arrive() {
    let a = point("p", "loss", json!("fast/a%"), 1);
    let b = point("p", "loss", json!("slow"), 2);
    let before = document(std::slice::from_ref(&a)).unwrap();
    let after = document(&[a.clone(), b.clone()]).unwrap();
    let key = tracking::identity(&a, true).unwrap();
    assert_eq!(
        before["series"][&key]["curve"],
        "curves/loss/variants/fast%2Fa%25"
    );
    assert_eq!(
        before["series"][&key]["curve"],
        after["series"][&key]["curve"]
    );
    let record = tracking::curve_record(
        &json!({"payload":{"kind":"metrics","observations":[a,b]}}),
        &after,
    )
    .unwrap();
    assert_eq!(record["curves/loss/variants/fast%2Fa%25"], 4.0);
    assert_eq!(record["curves/loss/variants/slow"], 4.0);
    assert!(record.get("curves/loss").is_none());
}

#[test]
fn ambiguous_contexts_and_generated_field_collisions_fail_without_overwriting() {
    let a = point("p", "loss", Value::Null, 1);
    let mut candidates = vec![
        point("another-protocol", "loss", Value::Null, 2),
        point("p", "loss/by_epoch", Value::Null, 2),
    ];
    let mut unit = a.clone();
    unit["unit"] = json!("bits/byte");
    candidates.push(unit);
    for b in candidates {
        assert!(matches!(
            document(&[a.clone(), b]),
            Err(tracking::Error::Display("DISPLAY_NAME_CONFLICT"))
        ));
    }
    assert!(
        document(&[
            point("p", "loss", json!("a"), 1),
            point("p", "loss/variants/a", Value::Null, 2)
        ])
        .is_err()
    );
    // Percent escapes are injective, not a lossy replacement.
    assert!(
        document(&[
            point("p", "loss", json!("a/b"), 1),
            point("p", "loss", json!("a%2Fb"), 2)
        ])
        .is_ok()
    );
}

#[test]
fn stable_definitions_have_one_visible_curve_and_use_scientific_axes() {
    let display = document(&[
        point("p", "loss", Value::Null, 1),
        point("p", "loss", Value::Null, 2),
    ])
    .unwrap();
    let specs = tracking::definition_specs(&display).unwrap();
    let curves: Vec<_> = specs.iter().filter(|s| s.name == "curves/loss").collect();
    assert_eq!(curves.len(), 1);
    assert_eq!(curves[0].options["hidden"], false);
    assert_eq!(curves[0].options["step_metric"], "axes/loss/step");
    let patch = tracking::summary_patch(&display, &[], false, "fixture").unwrap();
    assert_eq!(patch.set["ml_expd/display_version"], 3);
    assert!(!patch.set.contains_key("results/loss"));
}

#[test]
fn stable_summary_preserves_explicit_failure_and_does_not_create_zero() {
    let mut failed = point("p", "loss", Value::Null, 2);
    failed["summary"] = json!(true);
    failed["status"] = json!("FAILED");
    failed["value"] = Value::Null;
    failed["error"] = json!("NON_FINITE_VALUE");
    let display = document(&[point("p", "loss", Value::Null, 1), failed]).unwrap();
    let patch = tracking::summary_patch(&display, &[], true, "fixture").unwrap();
    assert!(!patch.set.contains_key("results/loss"));
    let metadata: Value = serde_json::from_str(
        patch.set["ml_expd/display_definitions_json"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(metadata["results/loss"]["error"], "NON_FINITE_VALUE");
}
#[test]
fn stable_display_history_definitions_and_final_results_are_pinned() {
    let case: Value = serde_json::from_str(r#"{"events":[{"seq":1,"event_id":"fixture-a-1","payload":{"kind":"metrics","observations":[{"name":"train_loss","unit":"nats/token","protocol_id":"protocol-a","dataset_id":"dataset-a","variant_id":null,"value":3.5,"step":1,"status":"VALID"}]}},{"seq":2,"event_id":"fixture-a-2","payload":{"kind":"metrics","observations":[{"name":"train_loss","unit":"nats/token","protocol_id":"protocol-a","dataset_id":"dataset-a","variant_id":null,"value":3.0,"step":2,"status":"VALID"}]}},{"seq":3,"event_id":"fixture-a-3","payload":{"kind":"metrics","observations":[{"name":"val_loss","unit":"nats/token","protocol_id":"protocol-a","dataset_id":"validation-a","variant_id":null,"value":3.75,"epoch":1,"status":"VALID"}]}},{"seq":4,"event_id":"fixture-a-4","payload":{"kind":"metrics","observations":[{"name":"val_loss","unit":"nats/token","protocol_id":"protocol-a","dataset_id":"validation-a","variant_id":null,"value":3.5,"epoch":2,"status":"VALID"}]}},{"seq":5,"event_id":"fixture-a-5","payload":{"kind":"metrics","observations":[{"name":"train_loss","unit":"nats/token","protocol_id":"protocol-a","dataset_id":"dataset-a","variant_id":null,"value":3.0,"step":2,"status":"VALID","summary":true,"checkpoint_id":"checkpoint-a","numerator":30,"denominator":10}]}}],"history":[{"ml_expd/sequence":1,"ml_expd/event_id":"fixture-a-1","ml_expd/kind":"metrics","ml_expd/record":"{\"kind\":\"metrics\",\"observations\":[{\"dataset_id\":\"dataset-a\",\"name\":\"train_loss\",\"protocol_id\":\"protocol-a\",\"status\":\"VALID\",\"step\":1,\"unit\":\"nats/token\",\"value\":3.5,\"variant_id\":null}]}","curves/train_loss":3.5,"axes/train_loss/step":1},{"ml_expd/sequence":2,"ml_expd/event_id":"fixture-a-2","ml_expd/kind":"metrics","ml_expd/record":"{\"kind\":\"metrics\",\"observations\":[{\"dataset_id\":\"dataset-a\",\"name\":\"train_loss\",\"protocol_id\":\"protocol-a\",\"status\":\"VALID\",\"step\":2,\"unit\":\"nats/token\",\"value\":3.0,\"variant_id\":null}]}","curves/train_loss":3.0,"axes/train_loss/step":2},{"ml_expd/sequence":3,"ml_expd/event_id":"fixture-a-3","ml_expd/kind":"metrics","ml_expd/record":"{\"kind\":\"metrics\",\"observations\":[{\"dataset_id\":\"validation-a\",\"epoch\":1,\"name\":\"val_loss\",\"protocol_id\":\"protocol-a\",\"status\":\"VALID\",\"unit\":\"nats/token\",\"value\":3.75,\"variant_id\":null}]}","curves/val_loss/by_epoch":3.75,"axes/val_loss/by_epoch/epoch":1},{"ml_expd/sequence":4,"ml_expd/event_id":"fixture-a-4","ml_expd/kind":"metrics","ml_expd/record":"{\"kind\":\"metrics\",\"observations\":[{\"dataset_id\":\"validation-a\",\"epoch\":2,\"name\":\"val_loss\",\"protocol_id\":\"protocol-a\",\"status\":\"VALID\",\"unit\":\"nats/token\",\"value\":3.5,\"variant_id\":null}]}","curves/val_loss/by_epoch":3.5,"axes/val_loss/by_epoch/epoch":2},{"ml_expd/sequence":5,"ml_expd/event_id":"fixture-a-5","ml_expd/kind":"metrics","ml_expd/record":"{\"kind\":\"metrics\",\"observations\":[{\"checkpoint_id\":\"checkpoint-a\",\"dataset_id\":\"dataset-a\",\"denominator\":10,\"name\":\"train_loss\",\"numerator\":30,\"protocol_id\":\"protocol-a\",\"status\":\"VALID\",\"step\":2,\"summary\":true,\"unit\":\"nats/token\",\"value\":3.0,\"variant_id\":null}]}"}],"definitions":[{"name":"metrics/*","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/*","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"ml_expd/*","options":{"hidden":true}},{"name":"axes/train_loss/step","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/train_loss","options":{"step_metric":"axes/train_loss/step","step_sync":false,"hidden":false,"summary":"none","overwrite":true}},{"name":"axes/train_loss/by_epoch/epoch","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/train_loss/by_epoch","options":{"step_metric":"axes/train_loss/by_epoch/epoch","step_sync":false,"hidden":true,"summary":"none","overwrite":true}},{"name":"results/train_loss","options":{"hidden":false,"overwrite":true}},{"name":"axes/val_loss/step","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/val_loss","options":{"step_metric":"axes/val_loss/step","step_sync":false,"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/val_loss/by_epoch/epoch","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/val_loss/by_epoch","options":{"step_metric":"axes/val_loss/by_epoch/epoch","step_sync":false,"hidden":false,"summary":"none","overwrite":true}},{"name":"results/val_loss","options":{"hidden":false,"overwrite":true}}],"summary":{"results/train_loss":3.0,"results/val_loss":3.5,"ml_expd/display_definitions_json":"{\"results/train_loss\": {\"checkpoint_id\": \"checkpoint-a\", \"dataset_id\": \"dataset-a\", \"denominator\": 10, \"epoch\": null, \"error\": null, \"name\": \"train_loss\", \"numerator\": 30, \"protocol_id\": \"protocol-a\", \"selection\": \"explicit_summary\", \"status\": \"VALID\", \"step\": 2, \"unit\": \"nats/token\", \"variant_id\": null}, \"results/val_loss\": {\"checkpoint_id\": null, \"dataset_id\": \"validation-a\", \"denominator\": null, \"epoch\": 2, \"error\": null, \"name\": \"val_loss\", \"numerator\": null, \"protocol_id\": \"protocol-a\", \"selection\": \"latest_observation\", \"status\": \"VALID\", \"step\": null, \"unit\": \"nats/token\", \"variant_id\": null}}","ml_expd/display_coverage":"fixture","ml_expd/display_through":5,"ml_expd/display_version":3}}"#).unwrap();
    let events = case["events"].as_array().unwrap();
    let items: Vec<_> = events
        .iter()
        .flat_map(|e| e["payload"]["observations"].as_array().unwrap())
        .cloned()
        .collect();
    let display = document(&items).unwrap();
    let records: Vec<_> = events
        .iter()
        .map(|e| tracking::history_record(e, &display).unwrap())
        .collect();
    assert_eq!(json!(records), case["history"]);
    assert_eq!(
        json!(tracking::definition_specs(&display).unwrap()),
        case["definitions"]
    );
    let patch = tracking::summary_patch(&display, &[], true, "fixture").unwrap();
    assert_eq!(patch.set, case["summary"].as_object().unwrap().clone());
}
