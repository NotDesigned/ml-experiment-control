//! Native metric contract cases; all evaluation at test runtime is Rust.
use std::cmp::Ordering;

use ml_exp_core::{
    canonical::protocol_identity,
    metrics::{
        MetricDefinition, MetricSchema, ObservationInput, metric_result, numeric_compare,
        numeric_equal, observations, observations_typed,
    },
};
use serde_json::{Value, json};

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

#[test]
fn schemas_fill_defaults_keep_unicode_names_and_refuse_unknown_fields() {
    for case in [
        literal(
            r#"{"id":"schema-empty","operation":"schema","input":{"value":{}},"expected":{"schema_version":1,"definitions":{}}}"#,
        ),
        literal(
            r#"{"id":"schema-name-'任意指标/μ'","operation":"schema","input":{"value":{"definitions":{"任意指标/μ":{"unit":"u"}}}},"expected":{"schema_version":1,"definitions":{"任意指标/μ":{"unit":"u","required":false,"description":"","aggregation":"scalar","numerator_unit":null,"denominator_unit":null}}}}"#,
        ),
        literal(
            r#"{"id":"schema-required-when-{'dataset_id': 'd'}","operation":"schema","input":{"value":{"definitions":{"custom":{"unit":"u","required":true,"required_when":{"dataset_id":"d"}}}}},"expected":{"schema_version":1,"definitions":{"custom":{"unit":"u","required":true,"description":"","aggregation":"scalar","numerator_unit":null,"denominator_unit":null,"required_when":{"dataset_id":"d"}}}}}"#,
        ),
    ] {
        let schema = MetricSchema::from_value(&case["input"]["value"]).unwrap();
        assert_eq!(schema.to_value(), case["expected"]);
    }
    for case in [
        literal(
            r#"{"id":"schema-unknown","operation":"schema","input":{"value":{"extra":false}},"expected_error":"ValidationError"}"#,
        ),
        literal(
            r#"{"id":"schema-version-3","operation":"schema","input":{"value":{"schema_version":0}},"expected_error":"ValidationError"}"#,
        ),
    ] {
        assert!(MetricSchema::from_value(&case["input"]["value"]).is_err());
    }
    for value in [
        json!({"schema_version":true}),
        json!({"schema_version":1.0}),
        json!({"schema_version":"1"}),
        json!({"definitions":{"m":{"unit":"u","required":1}}}),
        json!({"definitions":{"m":{"unit":"u","required":0}}}),
        json!({"definitions":{"m":{"unit":"u","required":"true"}}}),
        json!({"definitions":{"m":{"unit":"u","required":"yes"}}}),
        json!({"definitions":{"m":{"unit":"u","required":null}}}),
    ] {
        assert!(MetricSchema::from_value(&value).is_err(), "{value}");
    }
    let strict = json!({"schema_version":1,"definitions":{"m":{"unit":"u","required":false}}});
    assert!(MetricSchema::from_value(&strict).is_ok());
    let case = literal(
        r#"{"id":"definition-unit-8","operation":"definition","input":{"value":{"unit":"\u007f"}},"expected":{"unit":"\u007f","required":false,"description":"","aggregation":"scalar","numerator_unit":null,"denominator_unit":null}}"#,
    );
    let definition = MetricDefinition::from_value(&case["input"]["value"]).unwrap();
    assert_eq!(definition.to_value(), case["expected"]);
    let case = literal(
        r#"{"id":"definition-unit-0","operation":"definition","input":{"value":{"unit":""}},"expected_error":"ValidationError"}"#,
    );
    assert!(MetricDefinition::from_value(&case["input"]["value"]).is_err());
}

#[test]
fn observations_normalize_declared_values_and_report_invalid_names_as_failed() {
    for case in [
        literal(
            r#"{"id":"observation-value-12","operation":"observations","input":{"record":{"name":"validation/loss","value":0,"unit":"nats/token","numerator":25,"denominator":10,"step":4},"schema":{"schema_version":1,"definitions":{"validation/loss":{"unit":"nats/token","required":true,"aggregation":"ratio_of_sums","numerator_unit":"nats","denominator_unit":"token"},"encoded_loss":{"unit":"bits/raw byte"}}},"protocol_id":"p1","source":"/run/metrics.jsonl"},"expected":[{"step":4,"name":"validation/loss","value":0,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":"p1","status":"VALID","error":null,"source":"/run/metrics.jsonl","errors":[],"numerator":25,"denominator":10}]}"#,
        ),
        literal(
            r#"{"id":"observation-name-2","operation":"observations","input":{"record":{"name":1,"value":2.5,"unit":"nats/token","numerator":25,"denominator":10,"step":4},"schema":{"schema_version":1,"definitions":{"validation/loss":{"unit":"nats/token","required":true,"aggregation":"ratio_of_sums","numerator_unit":"nats","denominator_unit":"token"},"encoded_loss":{"unit":"bits/raw byte"}}},"protocol_id":"p1","source":"/run/metrics.jsonl"},"expected":[{"name":null,"value":null,"unit":null,"status":"FAILED","error":"INVALID_METRIC_NAME","errors":["INVALID_METRIC_NAME"],"source":"/run/metrics.jsonl"}]}"#,
        ),
        literal(
            r#"{"id":"wide-observations-501","operation":"observations","input":{"record":{},"schema":{}},"expected":[]}"#,
        ),
    ] {
        let input = &case["input"];
        let actual = observations(
            &input["record"],
            &input["schema"],
            input["protocol_id"].as_str(),
            input["source"].as_str(),
        )
        .unwrap();
        assert_eq!(Value::Array(actual), case["expected"]);
    }
}

#[test]
fn a_non_finite_step_is_not_retained_as_an_axis() {
    let case = literal(
        r#"{"id":"nonfinite-observation-step-nan","operation":"observations","input":{"record":{"name":"validation/loss","value":2.5,"unit":"nats/token","numerator":25,"denominator":10,"step":4},"schema":{"schema_version":1,"definitions":{"validation/loss":{"unit":"nats/token","required":true,"aggregation":"ratio_of_sums","numerator_unit":"nats","denominator_unit":"token"},"encoded_loss":{"unit":"bits/raw byte"}}},"protocol_id":"p1","overrides":[{"field":"step","value":"nan"}]},"expected":[{"name":"validation/loss","value":2.5,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":"p1","status":"VALID","error":null,"source":null,"errors":[],"numerator":25,"denominator":10}]}"#,
    );
    let input = &case["input"];
    let record = ObservationInput::from_json(&input["record"])
        .unwrap()
        .with_number("step", f64::NAN);
    let actual = observations_typed(
        &record,
        &input["schema"],
        input["protocol_id"].as_str(),
        input["source"].as_str(),
    )
    .unwrap();
    assert_eq!(Value::Array(actual), case["expected"]);
}

#[test]
fn metric_results_classify_complete_and_failed_contexts_and_refuse_malformed_normalized_input() {
    for case in [
        literal(
            r#"{"id":"result-value-12","operation":"result","input":{"records":[{"name":"validation/loss","value":0,"unit":"nats/token","numerator":25,"denominator":10,"step":4}],"schema":{"schema_version":1,"definitions":{"validation/loss":{"unit":"nats/token","required":true,"aggregation":"ratio_of_sums","numerator_unit":"nats","denominator_unit":"token"},"encoded_loss":{"unit":"bits/raw byte"}}},"protocol_id":"p1","source":"/run/metrics.jsonl"},"expected":{"state":"COMPLETE","required_metrics":["validation/loss"],"missing_metrics":[],"records":[{"step":4,"name":"validation/loss","value":0,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":"p1","status":"VALID","error":null,"source":"/run/metrics.jsonl","errors":[],"numerator":25,"denominator":10,"sources":["/run/metrics.jsonl"]}],"contexts":[{"identity":{"protocol_id":"p1","checkpoint_id":null,"dataset_id":null,"variant_id":null,"epoch":null,"step":4},"state":"COMPLETE","missing_metrics":[],"records":[{"step":4,"name":"validation/loss","value":0,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":"p1","status":"VALID","error":null,"source":"/run/metrics.jsonl","errors":[],"numerator":25,"denominator":10,"sources":["/run/metrics.jsonl"]}]}],"current":{"identity":{"protocol_id":"p1","checkpoint_id":null,"dataset_id":null,"variant_id":null,"epoch":null,"step":4},"state":"COMPLETE","missing_metrics":[],"records":[{"step":4,"name":"validation/loss","value":0,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":"p1","status":"VALID","error":null,"source":"/run/metrics.jsonl","errors":[],"numerator":25,"denominator":10,"sources":["/run/metrics.jsonl"]}]},"protocol_id":"p1"}}"#,
        ),
        literal(
            r#"{"id":"result-name-2","operation":"result","input":{"records":[{"name":1,"value":2.5,"unit":"nats/token","numerator":25,"denominator":10,"step":4}],"schema":{"schema_version":1,"definitions":{"validation/loss":{"unit":"nats/token","required":true,"aggregation":"ratio_of_sums","numerator_unit":"nats","denominator_unit":"token"},"encoded_loss":{"unit":"bits/raw byte"}}},"protocol_id":"p1","source":"/run/metrics.jsonl"},"expected":{"state":"FAILED","required_metrics":["validation/loss"],"missing_metrics":["validation/loss"],"records":[{"name":null,"value":null,"unit":null,"status":"FAILED","error":"INVALID_METRIC_NAME","errors":["INVALID_METRIC_NAME"],"source":"/run/metrics.jsonl","sources":["/run/metrics.jsonl"]}],"contexts":[{"identity":{"protocol_id":null,"checkpoint_id":null,"dataset_id":null,"variant_id":null,"epoch":null,"step":null},"state":"FAILED","missing_metrics":["validation/loss"],"records":[{"name":null,"value":null,"unit":null,"status":"FAILED","error":"INVALID_METRIC_NAME","errors":["INVALID_METRIC_NAME"],"source":"/run/metrics.jsonl","sources":["/run/metrics.jsonl"]}]}],"current":{"identity":{"protocol_id":null,"checkpoint_id":null,"dataset_id":null,"variant_id":null,"epoch":null,"step":null},"state":"FAILED","missing_metrics":["validation/loss"],"records":[{"name":null,"value":null,"unit":null,"status":"FAILED","error":"INVALID_METRIC_NAME","errors":["INVALID_METRIC_NAME"],"source":"/run/metrics.jsonl","sources":["/run/metrics.jsonl"]}]},"protocol_id":"p1"}}"#,
        ),
        literal(
            r#"{"id":"normalized-input","operation":"result","input":{"records":[{"step":4,"name":"validation/loss","value":2.5,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":null,"status":"VALID","error":null,"source":"/run/m.jsonl","errors":[],"numerator":25,"denominator":10}],"schema":{"schema_version":1,"definitions":{"validation/loss":{"unit":"nats/token","required":true,"aggregation":"ratio_of_sums","numerator_unit":"nats","denominator_unit":"token"},"encoded_loss":{"unit":"bits/raw byte"}}},"normalized":true},"expected":{"state":"COMPLETE","required_metrics":["validation/loss"],"missing_metrics":[],"records":[{"step":4,"name":"validation/loss","value":2.5,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":null,"status":"VALID","error":null,"source":"/run/m.jsonl","errors":[],"numerator":25,"denominator":10,"sources":["/run/m.jsonl"]}],"contexts":[{"identity":{"protocol_id":null,"checkpoint_id":null,"dataset_id":null,"variant_id":null,"epoch":null,"step":4},"state":"COMPLETE","missing_metrics":[],"records":[{"step":4,"name":"validation/loss","value":2.5,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":null,"status":"VALID","error":null,"source":"/run/m.jsonl","errors":[],"numerator":25,"denominator":10,"sources":["/run/m.jsonl"]}]}],"current":{"identity":{"protocol_id":null,"checkpoint_id":null,"dataset_id":null,"variant_id":null,"epoch":null,"step":4},"state":"COMPLETE","missing_metrics":[],"records":[{"step":4,"name":"validation/loss","value":2.5,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":null,"status":"VALID","error":null,"source":"/run/m.jsonl","errors":[],"numerator":25,"denominator":10,"sources":["/run/m.jsonl"]}]},"protocol_id":null}}"#,
        ),
    ] {
        let input = &case["input"];
        let actual = metric_result(
            input["records"].as_array().unwrap(),
            input.get("schema"),
            input["protocol_id"].as_str(),
            input["source"].as_str(),
            input["normalized"].as_bool().unwrap_or(false),
        )
        .unwrap();
        assert_eq!(actual, case["expected"]);
    }
    let case = literal(
        r#"{"id":"normalized-malformed-582","operation":"result","input":{"records":[{}],"normalized":true},"expected_error":"KeyError"}"#,
    );
    let input = &case["input"];
    assert!(metric_result(input["records"].as_array().unwrap(), None, None, None, true).is_err());
}

#[test]
fn a_metric_outside_a_declared_schema_is_undeclared_and_never_complete() {
    let schema = json!({"definitions":{"declared":{"unit":"nats/token"}}});
    let record = json!({"name":"other","value":2.5,"unit":"nats/token","step":4});
    let normalized = observations(&record, &schema, None, None).unwrap();
    assert_eq!(normalized[0]["status"], "UNDECLARED");
    assert_eq!(normalized[0]["error"], "METRIC_NOT_DECLARED");
    assert_eq!(normalized[0]["errors"], json!(["METRIC_NOT_DECLARED"]));
    assert_eq!(normalized[0]["value"], 2.5);
    let result = metric_result(&[record], Some(&schema), None, None, false).unwrap();
    assert_eq!(result["state"], "PARTIAL");
    // An empty schema declares nothing, so it refuses nothing.
    let record = json!({"name":"other","value":2.5,"unit":"nats/token","step":4});
    let open = metric_result(&[record], None, None, None, false).unwrap();
    assert_eq!(open["records"][0]["status"], "VALID");
    assert_eq!(open["state"], "COMPLETE");
}

#[test]
fn units_or_values_that_disagree_in_one_context_conflict_and_lose_their_value() {
    let metric = |value: f64, unit: &str, step: i64| json!({"name":"loss","value":value,"unit":unit,"step":step});
    let units = metric_result(
        &[metric(2.5, "nats", 4), metric(2.5, "bits", 4)],
        None,
        None,
        None,
        false,
    )
    .unwrap();
    assert_eq!(units["state"], "FAILED");
    let records = units["records"].as_array().unwrap();
    assert_eq!(records.len(), 2);
    for record in records {
        assert_eq!(record["status"], "CONFLICTING");
        assert_eq!(record["error"], "CONFLICTING_UNITS");
        assert_eq!(record["value"], Value::Null);
    }
    let values = metric_result(
        &[metric(2.5, "nats", 4), metric(3.0, "nats", 4)],
        None,
        None,
        None,
        false,
    )
    .unwrap();
    assert_eq!(values["state"], "FAILED");
    let records = values["records"].as_array().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["status"], "CONFLICTING");
    assert_eq!(records[0]["error"], "CONFLICTING_OBSERVATIONS");
    assert_eq!(records[0]["value"], Value::Null);
    // Neither a repeat nor another position is a conflict.
    let separate = metric_result(
        &[
            metric(2.5, "nats", 4),
            metric(2.5, "nats", 4),
            metric(2.0, "bits", 5),
        ],
        None,
        None,
        None,
        false,
    )
    .unwrap();
    assert_eq!(separate["state"], "COMPLETE");
    let records = separate["records"].as_array().unwrap();
    assert_eq!(records.len(), 2);
    assert!(records.iter().all(|record| record["status"] == "VALID"));
}

#[test]
fn protocol_identity_digests_stay_pinned() {
    for case in [
        literal(
            r#"{"id":"protocol-585","operation":"protocol","input":{"value":{}},"expected":"sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"}"#,
        ),
        literal(
            r#"{"id":"protocol-589","operation":"protocol","input":{"value":{"value":1e-07}},"expected":"sha256:910750038eb30a40d06da88009b6259aabaf3751b3cb212c616d51abb6b008a4"}"#,
        ),
    ] {
        assert_eq!(
            protocol_identity(&case["input"]["value"]).unwrap(),
            case["expected"].as_str().unwrap()
        );
    }
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
