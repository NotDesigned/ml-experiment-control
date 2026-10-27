//! Frozen Python results; all evaluation at test runtime is native Rust.
use std::cmp::Ordering;

use ml_exp_core::metrics::{
    MetricDefinition, MetricSchema, ObservationInput, metric_result, metric_result_typed,
    numeric_compare, numeric_equal, observations, observations_typed, protocol_identity,
};
use serde_json::{Value, json};

fn number(value: &str) -> f64 {
    match value {
        "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => panic!("unknown reference override"),
    }
}

fn typed_records(input: &Value, records: &[Value]) -> Vec<ObservationInput> {
    let mut records: Vec<_> = records
        .iter()
        .map(|record| ObservationInput::from_json(record).unwrap())
        .collect();
    for entry in input["overrides"].as_array().unwrap() {
        let index = entry["record_index"].as_u64().unwrap_or(0) as usize;
        records[index] = records[index].clone().with_number(
            entry["field"].as_str().unwrap(),
            number(entry["value"].as_str().unwrap()),
        );
    }
    records
}

fn evaluate(case: &Value) -> Result<Value, String> {
    let input = &case["input"];
    let empty = json!({});
    let schema = input.get("schema").unwrap_or(&empty);
    let protocol = input["protocol_id"].as_str();
    let source = input["source"].as_str();
    match case["operation"].as_str().unwrap() {
        "schema" => MetricSchema::from_value(&input["value"])
            .map(|schema| schema.to_value())
            .map_err(|error| error.to_string()),
        "definition" => MetricDefinition::from_value(&input["value"])
            .map(|definition| definition.to_value())
            .map_err(|error| error.to_string()),
        "observations" => {
            let result = if input.get("overrides").is_some() {
                observations_typed(
                    &typed_records(input, &[input["record"].clone()])[0],
                    schema,
                    protocol,
                    source,
                )
            } else {
                observations(&input["record"], schema, protocol, source)
            };
            result.map(Value::Array).map_err(|error| error.to_string())
        }
        "result" => {
            let records = input["records"].as_array().unwrap();
            let result = if input.get("overrides").is_some() {
                metric_result_typed(
                    &typed_records(input, records),
                    input.get("schema"),
                    protocol,
                    source,
                )
            } else {
                metric_result(
                    records,
                    input.get("schema"),
                    protocol,
                    source,
                    input["normalized"].as_bool().unwrap_or(false),
                )
            };
            result.map_err(|error| error.to_string())
        }
        "protocol" => protocol_identity(&input["value"])
            .map(Value::String)
            .map_err(|error| error.to_string()),
        "numeric" => numeric_compare(&input["left"], &input["right"])
            .map(|ordering| json!({"equal":numeric_equal(&input["left"],&input["right"]),"ordering":match ordering {Ordering::Less=>-1,Ordering::Equal=>0,Ordering::Greater=>1}}))
            .map_err(|error| error.to_string()),
        _ => panic!("unknown reference operation"),
    }
}

#[test]
fn complete_metric_domain_matches_frozen_python_results() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/metrics-contract.json")).unwrap();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["reference"]["source_path"],
        "server/src/ml_exp_server/records/contract.py"
    );
    assert_eq!(
        fixture["reference"]["source_sha256"],
        "dae3eff721cee108db92ae8e80f5b9ecd2639c3003e2abef8326f1ab652eb9bb"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(fixture["reference"]["unicode_version"], "15.0.0");
    assert_eq!(cases.len(), 4205);
    let mut failures = Vec::new();
    for case in cases {
        let actual = evaluate(case);
        if case.get("expected_error").is_some() {
            if actual.is_ok() {
                failures.push(
                    format!("{} expected error, got {:?}", case["id"], actual)
                        .chars()
                        .take(1500)
                        .collect::<String>(),
                );
            }
        } else if actual.as_ref().ok() != case.get("expected") {
            failures.push(
                format!(
                    "{} expected={} actual={:?}",
                    case["id"], case["expected"], actual
                )
                .chars()
                .take(3000)
                .collect::<String>(),
            );
        }
    }
    assert!(
        failures.is_empty(),
        "{} reference mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn numeric_identity_and_order_do_not_round_large_integer_contexts() {
    for (left, right) in [
        (json!(1), json!(1.0)),
        (json!(-0.0), json!(0)),
        (json!(true), json!(1)),
    ] {
        assert!(numeric_equal(&left, &right));
        assert_eq!(numeric_compare(&left, &right).unwrap(), Ordering::Equal);
    }
    for (left, right) in [
        (json!(9007199254740993_u64), json!(9007199254740992.0)),
        (json!(-1.5), json!(-2)),
        (json!(0.5), json!(0)),
    ] {
        assert!(!numeric_equal(&left, &right));
        assert_eq!(numeric_compare(&left, &right).unwrap(), Ordering::Greater);
        assert_eq!(numeric_compare(&right, &left).unwrap(), Ordering::Less);
    }
    assert!(!numeric_equal(&json!({}), &json!({})));
    assert!(numeric_compare(&json!("1"), &json!(1)).is_err());
}
