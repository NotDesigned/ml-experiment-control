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

#[test]
fn projection_refuses_malformed_observations_events_and_a_negative_cursor() {
    let missing_status = literal(
        r#"{"name":"loss","unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":4,"step":1}"#,
    );
    assert!(Projection::new().add(&missing_status, 1).is_err());
    let display = Projection::new().client_document().unwrap();
    assert!(tracking::curve_record(&json!({}), &display).is_err());
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
