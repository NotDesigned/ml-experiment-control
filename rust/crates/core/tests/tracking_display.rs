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

/// These exact records are also used by the explicit zero-GPU official W&B
/// acceptance, so that cloud verification exercises Rust-derived keys/axes.
#[test]
fn official_cloud_fixture_matches_rust_history_definitions_and_final_results() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../compatibility/metric-display-v3.json"
    ))
    .unwrap();
    for case in fixture["runs"].as_array().unwrap() {
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
}
