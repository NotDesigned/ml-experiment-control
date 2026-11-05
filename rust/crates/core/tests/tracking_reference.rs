//! Native tracking contract and display cases; never imports an SDK or contacts W&B.
use ml_exp_core::tracking::{self, Projection, RoutingSettings, WandbOptions, WandbSettings};
use serde_json::{Value, json};

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

#[test]
fn wandb_options_fill_defaults_and_refuse_invalid_documents() {
    for case in [
        literal(
            r#"{"id":"options-basic-0","operation":"options","input":{},"expected":{"enabled":true,"entity":null,"project":null}}"#,
        ),
        literal(
            r#"{"id":"options-entity-1","operation":"options","input":{"entity":"0"},"expected":{"enabled":true,"entity":"0","project":null}}"#,
        ),
    ] {
        let options = WandbOptions::from_value(&case["input"]).unwrap();
        assert_eq!(options.to_value(), case["expected"]);
    }
    for case in [
        literal(
            r#"{"id":"options-basic-6","operation":"options","input":[],"error":"ValidationError"}"#,
        ),
        literal(
            r#"{"id":"options-basic-9","operation":"options","input":{"entity":""},"error":"ValidationError"}"#,
        ),
    ] {
        assert!(WandbOptions::from_value(&case["input"]).is_err());
    }
}

#[test]
fn wandb_settings_refuse_empty_keys_and_unknown_fields() {
    assert!(WandbSettings::from_value(&literal(r#"{}"#)).is_ok());
    assert!(WandbSettings::from_value(&literal(r#"{"api_key":""}"#)).is_err());
    assert!(WandbSettings::from_value(&literal(r#"{"unknown":1}"#)).is_err());
}

fn route(input: &Value) -> Value {
    let raw = &input["settings"];
    let settings = RoutingSettings {
        enabled: raw.get("enabled").and_then(Value::as_bool).unwrap_or(true),
        preparation_enabled: raw
            .get("preparation_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        credentials_configured: raw
            .get("api_key")
            .and_then(Value::as_str)
            .is_some_and(|key| !key.is_empty()),
        resolved_entity: raw
            .get("resolved_entity")
            .and_then(Value::as_str)
            .map(str::to_owned),
        project: raw
            .get("project")
            .and_then(Value::as_str)
            .map(str::to_owned),
    };
    let options = if input["options"].is_null() {
        None
    } else {
        Some(WandbOptions::from_value(&input["options"]).unwrap())
    };
    let project = input["project"].as_str().unwrap();
    json!({
        "run": tracking::resolve(options.as_ref(), &settings, project),
        "preparation": tracking::resolve_preparation(options.as_ref(), &settings, project),
    })
}

#[test]
fn run_routes_follow_credentials_server_settings_and_the_requested_entity() {
    for case in [
        literal(
            r#"{"id":"route-0","operation":"route","input":{"settings":{},"options":null,"project":"fallback-project"},"expected":{"run":{"requested":true,"enabled":false,"reason":"CREDENTIALS_NOT_CONFIGURED","entity":null,"project":"fallback-project"},"preparation":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":null,"project":"fallback-project"}}}"#,
        ),
        literal(
            r#"{"id":"route-1","operation":"route","input":{"settings":{},"options":{},"project":"fallback-project"},"expected":{"run":{"requested":true,"enabled":false,"reason":"CREDENTIALS_NOT_CONFIGURED","entity":null,"project":"fallback-project"},"preparation":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":null,"project":"fallback-project"}}}"#,
        ),
        literal(
            r#"{"id":"route-10","operation":"route","input":{"settings":{"enabled":false},"options":{},"project":"fallback-project"},"expected":{"run":{"requested":true,"enabled":false,"reason":"DISABLED_BY_SERVER","entity":null,"project":"fallback-project"},"preparation":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":null,"project":"fallback-project"}}}"#,
        ),
        literal(
            r#"{"id":"route-24","operation":"route","input":{"settings":{"api_key":"synthetic-key"},"options":{"entity":"requested-team"},"project":"fallback-project"},"expected":{"run":{"requested":true,"enabled":true,"reason":null,"entity":"requested-team","project":"fallback-project"},"preparation":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":"requested-team","project":"fallback-project"}}}"#,
        ),
    ] {
        assert_eq!(route(&case["input"]), case["expected"]);
    }
}

#[test]
fn opt_out_missing_entity_and_server_defaults_decide_the_route() {
    for case in [
        literal(
            r#"{"id":"route-2","operation":"route","input":{"settings":{},"options":{"enabled":false},"project":"fallback-project"},"expected":{"run":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":null,"project":"fallback-project"},"preparation":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":null,"project":"fallback-project"}}}"#,
        ),
        literal(
            r#"{"id":"opt-out-beats-a-usable-route","operation":"route","input":{"settings":{"api_key":"synthetic-key","resolved_entity":"default-team"},"options":{"enabled":false},"project":"fallback-project"},"expected":{"run":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":"default-team","project":"fallback-project"},"preparation":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":"default-team","project":"fallback-project"}}}"#,
        ),
        literal(
            r#"{"id":"route-18","operation":"route","input":{"settings":{"api_key":"synthetic-key"},"options":null,"project":"fallback-project"},"expected":{"run":{"requested":true,"enabled":false,"reason":"DEFAULT_ENTITY_UNAVAILABLE","entity":null,"project":"fallback-project"},"preparation":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":null,"project":"fallback-project"}}}"#,
        ),
        literal(
            r#"{"id":"route-27","operation":"route","input":{"settings":{"api_key":"synthetic-key","resolved_entity":"default-team"},"options":null,"project":"fallback-project"},"expected":{"run":{"requested":true,"enabled":true,"reason":null,"entity":"default-team","project":"fallback-project"},"preparation":{"requested":false,"enabled":false,"reason":"DISABLED_BY_REQUEST","entity":"default-team","project":"fallback-project"}}}"#,
        ),
        literal(
            r#"{"id":"route-26","operation":"route","input":{"settings":{"api_key":"synthetic-key"},"options":{"entity":"requested-team","project":"requested-project","preparation_enabled":true},"project":"fallback-project"},"expected":{"run":{"requested":true,"enabled":true,"reason":null,"entity":"requested-team","project":"requested-project"},"preparation":{"requested":true,"enabled":true,"reason":null,"entity":"requested-team","project":"requested-project"}}}"#,
        ),
        literal(
            r#"{"id":"route-36","operation":"route","input":{"settings":{"api_key":"synthetic-key","resolved_entity":"default-team","project":"default-project","preparation_enabled":true},"options":null,"project":"fallback-project"},"expected":{"run":{"requested":true,"enabled":true,"reason":null,"entity":"default-team","project":"default-project"},"preparation":{"requested":true,"enabled":true,"reason":null,"entity":"default-team","project":"default-project"}}}"#,
        ),
    ] {
        assert_eq!(route(&case["input"]), case["expected"], "{}", case["id"]);
    }
}

#[test]
fn metric_stems_fingerprints_and_identities_stay_pinned() {
    for case in [
        literal(
            r#"{"id":"helpers-3","operation":"helpers","input":{"name":"a b","context":{"name":"a b","unit":"nats/token","protocol_id":"p\u2028x","dataset_id":"d","variant_id":"v"}},"expected":{"stem":"a_b","fingerprint":"c8687a08aa5d6ed2","identity":"{\"dataset_id\":\"d\",\"name\":\"a b\",\"protocol_id\":\"p\\u2028x\",\"unit\":\"nats/token\"}","variant_identity":"{\"dataset_id\":\"d\",\"name\":\"a b\",\"protocol_id\":\"p\\u2028x\",\"unit\":\"nats/token\",\"variant_id\":\"v\"}"}}"#,
        ),
        literal(
            r#"{"id":"helpers-8","operation":"helpers","input":{"name":"","context":{"name":"","unit":"nats/token","protocol_id":"p\u2028x","dataset_id":"d","variant_id":"v"}},"expected":{"stem":"","fingerprint":"e3b0c44298fc1c14","identity":"{\"dataset_id\":\"d\",\"name\":\"\",\"protocol_id\":\"p\\u2028x\",\"unit\":\"nats/token\"}","variant_identity":"{\"dataset_id\":\"d\",\"name\":\"\",\"protocol_id\":\"p\\u2028x\",\"unit\":\"nats/token\",\"variant_id\":\"v\"}"}}"#,
        ),
    ] {
        let input = &case["input"];
        let name = input["name"].as_str().unwrap();
        let actual = json!({
            "stem":tracking::stem(name),
            "fingerprint":tracking::fingerprint(name),
            "identity":tracking::identity(&input["context"],false).unwrap(),
            "variant_identity":tracking::identity(&input["context"],true).unwrap(),
        });
        assert_eq!(actual, case["expected"]);
    }
}

#[test]
fn only_finite_json_numbers_count_as_metric_numbers() {
    for case in [
        literal(r#"{"id":"number-0","operation":"number","input":null,"expected":false}"#),
        literal(r#"{"id":"number-1","operation":"number","input":true,"expected":false}"#),
        literal(r#"{"id":"number-3","operation":"number","input":0,"expected":true}"#),
        literal(r#"{"id":"number-5","operation":"number","input":1.0,"expected":true}"#),
        literal(r#"{"id":"number-11","operation":"number","input":"1","expected":false}"#),
        literal(r#"{"id":"number-12","operation":"number","input":[],"expected":false}"#),
    ] {
        assert_eq!(
            tracking::number(&case["input"]).unwrap(),
            case["expected"].as_bool().unwrap()
        );
    }
}

fn project(case: &Value) -> tracking::Result<Value> {
    let input = &case["input"];
    let items = input["items"].as_array().unwrap();
    let mut projection = Projection::new();
    for (index, item) in items.iter().enumerate() {
        projection = projection.add(item, index as i64 + 1)?;
    }
    projection = projection.at(input
        .get("through")
        .and_then(Value::as_i64)
        .unwrap_or(items.len() as i64))?;
    // These names are explicit inputs, representing reservations in the existing
    // destination-wide display_metrics table.
    let names = case["expected"]["names"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let display = projection.document(&names)?;
    let definitions = tracking::definition_specs(&display)?;
    let events = input
        .get("events")
        .cloned()
        .unwrap_or_else(|| json!([{"payload":{"kind":"metrics","observations":items}}]));
    let curves = events
        .as_array()
        .unwrap()
        .iter()
        .map(|event| tracking::curve_record(event, &display))
        .collect::<tracking::Result<Vec<_>>>()?;
    let keys = input
        .get("remote_keys")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| value.as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let patch = tracking::summary_patch(
        &display,
        &keys,
        input
            .get("terminal")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        input
            .get("coverage")
            .and_then(Value::as_str)
            .unwrap_or("complete"),
    )?;
    Ok(
        json!({"state":projection.state_value(),"names":names,"document":display,"curves":curves,"definitions":definitions,"patch":patch}),
    )
}

#[test]
fn projection_builds_state_document_curves_definitions_and_summary_patch() {
    for case in [
        literal(
            r#"{"id":"empty","operation":"projection","input":{"items":[],"remote_keys":["metrics/old","axes/old","results/old","curves/old","unmanaged","ml_expd/custom"]},"expected":{"state":{"cursor":0,"series":{}},"names":{},"document":{"contract":"metric-display.v2","through":0,"series":{}},"curves":[{}],"definitions":[{"name":"metrics/*","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/*","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"ml_expd/*","options":{"hidden":true}}],"patch":{"delete":["metrics/old","axes/old","results/old","curves/old"],"set":{"ml_expd/display_definitions_json":"{}","ml_expd/display_coverage":"complete","ml_expd/display_through":0,"ml_expd/display_version":2}}}}"#,
        ),
        literal(
            r#"{"id":"one-point","operation":"projection","input":{"items":[{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":4,"status":"VALID","step":1}],"terminal":true},"expected":{"state":{"cursor":1,"series":{"{\"dataset_id\":\"d\",\"name\":\"loss\",\"protocol_id\":\"p\",\"unit\":\"nats/token\",\"variant_id\":null}":{"context":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null},"latest":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":4,"status":"VALID","step":1,"axis":"step","sequence":1},"summary":null,"positions":{"step":[1]}}}},"names":{"{\"dataset_id\":\"d\",\"name\":\"loss\",\"protocol_id\":\"p\",\"unit\":\"nats/token\"}":"loss"},"document":{"contract":"metric-display.v2","through":1,"series":{"{\"dataset_id\":\"d\",\"name\":\"loss\",\"protocol_id\":\"p\",\"unit\":\"nats/token\",\"variant_id\":null}":{"context":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null},"latest":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":4,"status":"VALID","step":1,"axis":"step","sequence":1},"summary":null,"positions":{"step":[1]},"selection":"latest_observation","name":"loss","multiple_variants":false,"curve":"curves/loss","shadow":"curves/loss/variants/af20fbe13de1c144","result":"results/loss"}}},"curves":[{"curves/loss/variants/af20fbe13de1c144":4,"axes/loss/variants/af20fbe13de1c144/step":1,"curves/loss":4,"axes/loss/step":1}],"definitions":[{"name":"metrics/*","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/*","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"ml_expd/*","options":{"hidden":true}},{"name":"axes/loss/step","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/loss","options":{"step_metric":"axes/loss/step","step_sync":false,"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/loss/variants/af20fbe13de1c144/step","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/loss/variants/af20fbe13de1c144","options":{"step_metric":"axes/loss/variants/af20fbe13de1c144/step","step_sync":false,"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/loss/by_epoch/epoch","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/loss/by_epoch","options":{"step_metric":"axes/loss/by_epoch/epoch","step_sync":false,"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/loss/variants/af20fbe13de1c144/by_epoch/epoch","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/loss/variants/af20fbe13de1c144/by_epoch","options":{"step_metric":"axes/loss/variants/af20fbe13de1c144/by_epoch/epoch","step_sync":false,"hidden":true,"summary":"none","overwrite":true}},{"name":"results/loss","options":{"hidden":false,"overwrite":true}}],"patch":{"delete":[],"set":{"results/loss":4,"ml_expd/display_definitions_json":"{\"results/loss\": {\"checkpoint_id\": null, \"dataset_id\": \"d\", \"denominator\": null, \"epoch\": null, \"error\": null, \"name\": \"loss\", \"numerator\": null, \"protocol_id\": \"p\", \"selection\": \"latest_observation\", \"status\": \"VALID\", \"step\": 1, \"unit\": \"nats/token\", \"variant_id\": null}}","ml_expd/display_coverage":"complete","ml_expd/display_through":1,"ml_expd/display_version":2}}}}"#,
        ),
        literal(
            r#"{"id":"summary-preferred","operation":"projection","input":{"items":[{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":4,"status":"VALID","step":1},{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":2,"status":"VALID","step":2,"summary":true},{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":3,"status":"VALID","step":3}],"terminal":true},"expected":{"state":{"cursor":3,"series":{"{\"dataset_id\":\"d\",\"name\":\"loss\",\"protocol_id\":\"p\",\"unit\":\"nats/token\",\"variant_id\":null}":{"context":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null},"latest":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":3,"status":"VALID","step":3,"axis":"step","sequence":3},"summary":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":2,"status":"VALID","step":2,"summary":true,"axis":"step","sequence":2},"positions":{"step":[1,3]}}}},"names":{"{\"dataset_id\":\"d\",\"name\":\"loss\",\"protocol_id\":\"p\",\"unit\":\"nats/token\"}":"loss"},"document":{"contract":"metric-display.v2","through":3,"series":{"{\"dataset_id\":\"d\",\"name\":\"loss\",\"protocol_id\":\"p\",\"unit\":\"nats/token\",\"variant_id\":null}":{"context":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null},"latest":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":2,"status":"VALID","step":2,"summary":true,"axis":"step","sequence":2},"summary":{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":2,"status":"VALID","step":2,"summary":true,"axis":"step","sequence":2},"positions":{"step":[1,3]},"selection":"explicit_summary","name":"loss","multiple_variants":false,"curve":"curves/loss","shadow":"curves/loss/variants/af20fbe13de1c144","result":"results/loss"}}},"curves":[{"curves/loss/variants/af20fbe13de1c144":3,"axes/loss/variants/af20fbe13de1c144/step":3,"curves/loss":3,"axes/loss/step":3}],"definitions":[{"name":"metrics/*","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/*","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"ml_expd/*","options":{"hidden":true}},{"name":"axes/loss/step","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/loss","options":{"step_metric":"axes/loss/step","step_sync":false,"hidden":false,"summary":"none","overwrite":true}},{"name":"axes/loss/variants/af20fbe13de1c144/step","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/loss/variants/af20fbe13de1c144","options":{"step_metric":"axes/loss/variants/af20fbe13de1c144/step","step_sync":false,"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/loss/by_epoch/epoch","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/loss/by_epoch","options":{"step_metric":"axes/loss/by_epoch/epoch","step_sync":false,"hidden":true,"summary":"none","overwrite":true}},{"name":"axes/loss/variants/af20fbe13de1c144/by_epoch/epoch","options":{"hidden":true,"summary":"none","overwrite":true}},{"name":"curves/loss/variants/af20fbe13de1c144/by_epoch","options":{"step_metric":"axes/loss/variants/af20fbe13de1c144/by_epoch/epoch","step_sync":false,"hidden":true,"summary":"none","overwrite":true}},{"name":"results/loss","options":{"hidden":false,"overwrite":true}}],"patch":{"delete":[],"set":{"results/loss":2,"ml_expd/display_definitions_json":"{\"results/loss\": {\"checkpoint_id\": null, \"dataset_id\": \"d\", \"denominator\": null, \"epoch\": null, \"error\": null, \"name\": \"loss\", \"numerator\": null, \"protocol_id\": \"p\", \"selection\": \"explicit_summary\", \"status\": \"VALID\", \"step\": 2, \"unit\": \"nats/token\", \"variant_id\": null}}","ml_expd/display_coverage":"complete","ml_expd/display_through":3,"ml_expd/display_version":2}}}}"#,
        ),
    ] {
        assert_eq!(project(&case).unwrap(), case["expected"]);
    }
}

#[test]
fn projection_refuses_malformed_observations_events_and_a_negative_cursor() {
    for case in [
        literal(
            r#"{"id":"missing-status","operation":"projection","input":{"items":[{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":4,"step":1}]},"error":"KeyError"}"#,
        ),
        literal(
            r#"{"id":"unhashable-variant","operation":"projection","input":{"items":[{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":["a"],"value":4,"status":"VALID","step":1}]},"error":"TypeError"}"#,
        ),
        literal(
            r#"{"id":"malformed-event-0","operation":"projection","input":{"items":[],"events":[{}]},"error":"KeyError"}"#,
        ),
    ] {
        assert!(project(&case).is_err());
    }
    assert!(Projection::new().at(-1).is_err());
}

#[test]
fn finite_parameter_validation_returns_the_same_values() {
    let value = json!({"arbitrary": [1, 1.0, null],"unicode":"中文"});
    assert!(std::ptr::eq(
        tracking::finite_parameters(&value).unwrap(),
        &value
    ));
}
