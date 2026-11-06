//! The metric-display.v4 contract: W&B keys are the caller's exact metric names.
use ml_exp_core::tracking::{self, Projection};
use serde_json::{Value, json};

fn point(protocol: &str, name: &str, variant: Value, step: i64) -> Value {
    json!({"name":name,"unit":"nats/token","protocol_id":protocol,"dataset_id":"train",
        "variant_id":variant,"value":4.0,"step":step,"status":"VALID"})
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
fn row(display: &Value) -> &Value {
    display["series"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()
}
fn metrics(observations: &[&Value]) -> Value {
    json!({"payload":{"kind":"metrics","observations":observations}})
}

#[test]
fn only_the_v4_display_contract_is_accepted() {
    let event = metrics(&[]);
    let current = json!({"contract":"metric-display.v4","through":0,"series":{}});
    assert_eq!(tracking::curve_record(&event, &current).unwrap(), json!({}));
    assert_eq!(tracking::definition_specs(&current).unwrap().len(), 1);
    assert!(tracking::summary_patch(&current, &[], "fixture").is_ok());
    for display in [
        json!({"contract":"metric-display.v2","through":0,"series":{}}),
        json!({"contract":"metric-display.v3","through":0,"series":{}}),
        json!({"contract":"metric-display.v5","through":0,"series":{}}),
        json!({"contract":4,"through":0,"series":{}}),
        json!({"through":0,"series":{}}),
    ] {
        assert!(tracking::curve_record(&event, &display).is_err());
        assert!(tracking::definition_specs(&display).is_err());
        assert!(tracking::summary_patch(&display, &[], "fixture").is_err());
    }
}

#[test]
fn cross_run_protocol_and_dataset_differences_do_not_rename_metrics() {
    let first = point("source-seed-lr-a", "train_loss", Value::Null, 1);
    let mut second = point("source-seed-lr-b", "train_loss", Value::Null, 1);
    second["dataset_id"] = json!("another-version");
    let a = document(&[first]);
    let b = document(&[second]);
    assert_eq!(row(&a)["curve"], "train_loss");
    assert_eq!(row(&a)["curve"], row(&b)["curve"]);
    assert_eq!(row(&a)["result"], "train_loss");
    assert_ne!(row(&a)["context"], row(&b)["context"]);
    assert!(row(&a).get("shadow").is_none());
}

#[test]
fn original_names_keep_unicode_slashes_and_more_than_64_characters() {
    for name in ["验证/loss".to_owned(), "x".repeat(100), "a b".to_owned()] {
        let display = document(&[point("p", &name, Value::Null, 1)]);
        assert_eq!(row(&display)["curve"], json!(name));
        assert_eq!(row(&display)["result"], json!(name));
    }
}

#[test]
fn a_variant_never_renames_a_metric_before_or_after_other_variants_arrive() {
    let a = point("p", "loss", json!("fast/a%"), 1);
    let b = point("p", "loss", json!("slow"), 2);
    let before = document(std::slice::from_ref(&a));
    let after = document(&[a.clone(), b.clone()]);
    let key = tracking::identity(&a).unwrap();
    assert_eq!(before["series"][&key]["curve"], "loss");
    assert_eq!(
        before["series"][&key]["curve"],
        after["series"][&key]["curve"]
    );
    assert_eq!(after["series"].as_object().unwrap().len(), 2);
    let record = tracking::curve_record(&metrics(&[&a]), &after).unwrap();
    assert_eq!(record["loss"], 4.0);
    assert_eq!(
        record.as_object().unwrap().len(),
        2,
        "the value and its step axis, with no variant keys"
    );
}

#[test]
fn contexts_sharing_a_name_share_one_metric_but_cannot_collide_inside_one_row() {
    let a = point("p", "loss", Value::Null, 1);
    let mut unit = a.clone();
    unit["unit"] = json!("bits/byte");
    for b in [point("another-protocol", "loss", Value::Null, 2), unit] {
        let display = document(&[a.clone(), b.clone()]);
        assert_eq!(display["series"].as_object().unwrap().len(), 2);
        assert_eq!(
            tracking::curve_record(&metrics(&[&b]), &display).unwrap()["loss"],
            4.0
        );
        assert!(matches!(
            tracking::curve_record(&metrics(&[&a, &b]), &display),
            Err(tracking::Error::Display("DISPLAY_NAME_CONFLICT"))
        ));
    }
    // A name the caller chose can look like a generated field without colliding.
    let display = document(&[
        point("p", "loss", Value::Null, 1),
        point("p", "loss/by_epoch", Value::Null, 2),
    ]);
    assert_eq!(display["series"].as_object().unwrap().len(), 2);
}

#[test]
fn definitions_have_one_visible_curve_and_use_scientific_axes() {
    let display = document(&[
        point("p", "loss", Value::Null, 1),
        point("p", "loss", Value::Null, 2),
    ]);
    let specs = tracking::definition_specs(&display).unwrap();
    let curves: Vec<_> = specs.iter().filter(|s| s.name == "loss").collect();
    assert_eq!(curves.len(), 1);
    assert_eq!(curves[0].options["hidden"], false);
    assert_eq!(curves[0].options["step_metric"], "ml_expd/axes/loss/step");
    let patch = tracking::summary_patch(&display, &[], "fixture").unwrap();
    assert_eq!(patch.set["ml_expd/display_version"], 4);
    assert_eq!(patch.set["loss"], 4.0);
}

#[test]
fn summary_preserves_explicit_failure_and_does_not_create_zero() {
    let mut failed = point("p", "loss", Value::Null, 2);
    failed["summary"] = json!(true);
    failed["status"] = json!("FAILED");
    failed["value"] = Value::Null;
    failed["error"] = json!("NON_FINITE_VALUE");
    let display = document(&[point("p", "loss", Value::Null, 1), failed]);
    let patch = tracking::summary_patch(&display, &[], "fixture").unwrap();
    assert!(!patch.set.contains_key("loss"));
    let metadata: Value = serde_json::from_str(
        patch.set["ml_expd/display_definitions_json"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(metadata["loss"]["error"], "NON_FINITE_VALUE");
}

#[test]
fn display_history_definitions_and_final_results_are_pinned() {
    let observation = |name: &str, dataset: &str, axis: &str, position: i64, value: f64| {
        json!({"name":name,"unit":"nats/token","protocol_id":"protocol-a","dataset_id":dataset,
            "variant_id":null,"value":value,axis:position,"status":"VALID"})
    };
    let mut summary = observation("train_loss", "dataset-a", "step", 2, 3.0);
    summary["summary"] = json!(true);
    summary["checkpoint_id"] = json!("checkpoint-a");
    summary["numerator"] = json!(30);
    summary["denominator"] = json!(10);
    let items = [
        observation("train_loss", "dataset-a", "step", 1, 3.5),
        observation("train_loss", "dataset-a", "step", 2, 3.0),
        observation("val_loss", "validation-a", "epoch", 1, 3.75),
        observation("val_loss", "validation-a", "epoch", 2, 3.5),
        summary,
    ];
    let display = document(&items);
    let expected = [
        json!({"train_loss":3.5,"ml_expd/axes/train_loss/step":1}),
        json!({"train_loss":3.0,"ml_expd/axes/train_loss/step":2}),
        json!({"val_loss":3.75,"ml_expd/axes/val_loss/epoch":1}),
        json!({"val_loss":3.5,"ml_expd/axes/val_loss/epoch":2}),
        json!({}),
    ];
    for (index, (item, curves)) in items.iter().zip(expected).enumerate() {
        let sequence = index as i64 + 1;
        let payload = json!({"kind":"metrics","observations":[item]});
        let event_id = format!("fixture-a-{sequence}");
        let mut record = tracking::history_record(
            &json!({"seq":sequence,"event_id":event_id,"payload":payload}),
            &display,
        )
        .unwrap();
        let audit = record
            .as_object_mut()
            .unwrap()
            .remove("ml_expd/record")
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(audit.as_str().unwrap()).unwrap(),
            payload
        );
        let mut whole = curves;
        whole["ml_expd/sequence"] = json!(sequence);
        whole["ml_expd/event_id"] = json!(event_id);
        whole["ml_expd/kind"] = json!("metrics");
        assert_eq!(record, whole);
    }
    let hidden = json!({"hidden":true,"summary":"none","overwrite":true});
    let curve = |axis: &str| json!({"hidden":false,"summary":"none","overwrite":true,"step_metric":axis,"step_sync":false});
    assert_eq!(
        json!(tracking::definition_specs(&display).unwrap()),
        json!([
            {"name":"ml_expd/*","options":hidden},
            {"name":"ml_expd/axes/train_loss/step","options":hidden},
            {"name":"train_loss","options":curve("ml_expd/axes/train_loss/step")},
            {"name":"ml_expd/axes/val_loss/epoch","options":hidden},
            {"name":"val_loss","options":curve("ml_expd/axes/val_loss/epoch")},
        ])
    );
    let patch = tracking::summary_patch(&display, &[], "fixture").unwrap();
    assert!(patch.delete.is_empty());
    let definitions: Value = serde_json::from_str(
        patch.set["ml_expd/display_definitions_json"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        definitions,
        json!({
            "train_loss":{"checkpoint_id":"checkpoint-a","dataset_id":"dataset-a","denominator":10,
                "epoch":null,"error":null,"name":"train_loss","numerator":30,"protocol_id":"protocol-a",
                "selection":"explicit_summary","status":"VALID","step":2,"unit":"nats/token","variant_id":null},
            "val_loss":{"checkpoint_id":null,"dataset_id":"validation-a","denominator":null,
                "epoch":2,"error":null,"name":"val_loss","numerator":null,"protocol_id":"protocol-a",
                "selection":"latest_observation","status":"VALID","step":null,"unit":"nats/token","variant_id":null},
        })
    );
    let mut set = patch.set.clone();
    set.remove("ml_expd/display_definitions_json");
    assert_eq!(
        Value::Object(set),
        json!({"train_loss":3.0,"val_loss":3.5,"ml_expd/display_coverage":"fixture",
            "ml_expd/display_through":5,"ml_expd/display_version":4})
    );
}

fn varied(key: &str, value: Value) -> Value {
    let mut item = point("p", "loss", Value::Null, 1);
    item[key] = value;
    item
}
fn without_axis(value: f64, checkpoint: &str) -> Value {
    let mut item = varied("value", json!(value));
    item["checkpoint_id"] = json!(checkpoint);
    item.as_object_mut().unwrap().remove("step");
    item
}
#[test]
fn a_changed_observation_at_one_position_conflicts_and_publishes_no_value() {
    let same_position = "CONFLICTING_LATEST_OBSERVATIONS";
    for (items, error) in [
        (
            [varied("value", json!(4.0)), varied("value", json!(3.0))],
            same_position,
        ),
        (
            [
                varied("status", json!("VALID")),
                varied("status", json!("PARTIAL")),
            ],
            same_position,
        ),
        (
            [
                varied("checkpoint_id", json!("a")),
                varied("checkpoint_id", json!("b")),
            ],
            same_position,
        ),
        (
            [varied("numerator", json!(4)), varied("numerator", json!(8))],
            same_position,
        ),
        (
            [
                varied("denominator", json!(1)),
                varied("denominator", json!(2)),
            ],
            same_position,
        ),
        (
            [without_axis(4.0, "a"), without_axis(3.0, "a")],
            same_position,
        ),
        (
            [without_axis(4.0, "a"), without_axis(4.0, "b")],
            "UNORDERED_CHECKPOINTS",
        ),
    ] {
        let display = document(&items);
        let latest = &row(&display)["latest"];
        assert_eq!(latest["status"], "CONFLICTING");
        assert_eq!(latest["value"], Value::Null);
        assert_eq!(latest["error"], error);
        let patch = tracking::summary_patch(&display, &["loss".to_owned()], "fixture").unwrap();
        assert!(!patch.set.contains_key("loss"));
        assert_eq!(patch.delete, ["loss"]);
    }
}

#[test]
fn a_changed_explicit_summary_at_one_position_also_conflicts() {
    let summary = |value: f64| {
        let mut item = varied("value", json!(value));
        item["summary"] = json!(true);
        item
    };
    let display = document(&[summary(4.0), summary(3.0)]);
    assert_eq!(row(&display)["selection"], "explicit_summary");
    assert_eq!(row(&display)["latest"]["status"], "CONFLICTING");
    assert_eq!(row(&display)["latest"]["value"], Value::Null);
    let patch = tracking::summary_patch(&display, &[], "fixture").unwrap();
    assert!(!patch.set.contains_key("loss"));
}

#[test]
fn identical_repeats_stay_valid() {
    for items in [
        [
            point("p", "loss", Value::Null, 1),
            point("p", "loss", Value::Null, 1),
        ],
        [without_axis(4.0, "a"), without_axis(4.0, "a")],
    ] {
        let display = document(&items);
        assert_eq!(row(&display)["latest"]["status"], "VALID");
        assert_eq!(row(&display)["latest"]["sequence"], 2);
        let patch = tracking::summary_patch(&display, &[], "fixture").unwrap();
        assert_eq!(patch.set["loss"], 4.0);
    }
}

#[test]
fn a_late_lower_step_never_replaces_the_latest_value() {
    let at = |step: i64, value: f64| {
        let mut item = point("p", "loss", Value::Null, step);
        item["value"] = json!(value);
        item
    };
    let display = document(&[at(3, 2.0), at(1, 4.0), at(2, 3.0)]);
    let latest = &row(&display)["latest"];
    assert_eq!(latest["status"], "VALID");
    assert_eq!(latest["value"], 2.0);
    assert_eq!(latest["step"], 3);
    assert_eq!(latest["sequence"], 1);
    assert_eq!(row(&display)["positions"]["step"], json!([3, 1]));
    let patch = tracking::summary_patch(&display, &[], "fixture").unwrap();
    assert_eq!(patch.set["loss"], 2.0);
}
