//! Frozen observations from the original Python implementation. The test runner
//! uses only Rust, and never imports an SDK or contacts W&B.
use ml_exp_core::tracking::{self, Projection, RoutingSettings, WandbOptions, WandbSettings};
use serde_json::{Value, json};

fn evaluate(case: &Value) -> tracking::Result<Value> {
    let input = &case["input"];
    match case["operation"].as_str().unwrap() {
        "options" => Ok(WandbOptions::from_value(input)?.to_value()),
        "settings" => {
            let settings = WandbSettings::from_value(input)?;
            Ok(
                json!({"public":settings.to_public_value(),"provided":settings.provided_fields().collect::<Vec<_>>(),"has_key":settings.api_key().is_some()}),
            )
        }
        "route" => {
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
                Some(WandbOptions::from_value(&input["options"])?)
            };
            let project = input["project"].as_str().unwrap();
            Ok(
                json!({"run":tracking::resolve(options.as_ref(),&settings,project),"preparation":tracking::resolve_preparation(options.as_ref(),&settings,project)}),
            )
        }
        "parameters" => Ok(json!(
            String::from_utf8(tracking::finite_parameters_json(
                input["raw"].as_str().unwrap().as_bytes()
            )?)
            .unwrap()
        )),
        "helpers" => Ok(json!({
            "stem":tracking::stem(input["name"].as_str().unwrap()),
            "fingerprint":tracking::fingerprint(input["name"].as_str().unwrap()),
            "identity":tracking::identity(&input["context"],false)?,
            "variant_identity":tracking::identity(&input["context"],true)?,
        })),
        "number" => Ok(json!(tracking::number(input)?)),
        "capacity" => {
            let mut projection = Projection::new();
            for index in 0..input.as_u64().unwrap() {
                projection = projection.add(&json!({"name":format!("metric{index}"),"unit":"nats/token","protocol_id":"p","dataset_id":"d","variant_id":null,"value":4,"status":"VALID","step":1}),index as i64+1)?;
            }
            Ok(json!(
                projection.state_value()["series"]
                    .as_object()
                    .unwrap()
                    .len()
            ))
        }
        "projection" => {
            let items = input["items"].as_array().unwrap();
            let mut projection = Projection::new();
            for (index, item) in items.iter().enumerate() {
                projection = projection.add(item, index as i64 + 1)?;
            }
            projection = projection.at(input
                .get("through")
                .and_then(Value::as_i64)
                .unwrap_or(items.len() as i64))?;
            // These names are explicit fixture inputs, representing reservations
            // in the existing destination-wide display_metrics table.
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
        _ => unreachable!(),
    }
}

#[test]
fn tracking_contract_and_display_match_frozen_python_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/tracking.json")).unwrap();
    assert_eq!(
        fixture["reference"]["commit"],
        "479f1d39090759f9a64146dbac55e047f57b629d"
    );
    assert_eq!(
        fixture["reference"]["sources"]["server/src/ml_exp_server/tracking/tracking_contract.py"],
        "d5fa979a914318ee57a8e41fdb0fd144dc846349bdacba5503d8a434f8016f92"
    );
    assert_eq!(
        fixture["reference"]["sources"]["server/src/ml_exp_server/tracking/wandb_display.py"],
        "e7f1ef0ca7d73e14f905ffd7563195896976dd77a18f7dfdc766246da86a094c"
    );
    let cases = fixture["cases"].as_array().unwrap();
    for case in cases {
        let result = evaluate(case);
        if let Some(expected) = case.get("expected") {
            assert_eq!(
                result.unwrap_or_else(|error| panic!("{}: {error}", case["id"])),
                *expected,
                "{}",
                case["id"]
            );
        } else {
            assert!(result.is_err(), "{} unexpectedly accepted", case["id"]);
        }
    }
    assert_eq!(cases.len(), 478);
}

#[test]
fn finite_parameter_validation_returns_the_same_values_and_rejects_raw_nonfinite() {
    let value = json!({"arbitrary": [1, 1.0, null],"unicode":"中文"});
    assert!(std::ptr::eq(
        tracking::finite_parameters(&value).unwrap(),
        &value
    ));
    for raw in [
        b"NaN".as_slice(),
        b"Infinity",
        b"[1e400]",
        b"{\"x\":-Infinity}",
    ] {
        assert!(tracking::finite_parameters_json(raw).is_err());
    }
}

#[test]
fn an_older_cutoff_requires_replaying_the_durable_prefix() {
    let projection = Projection::new().at(10).unwrap();
    assert_eq!(projection.replay_after(11).unwrap(), 10);
    assert_eq!(projection.replay_after(10).unwrap(), 10);
    assert_eq!(projection.replay_after(9).unwrap(), 0);
    assert!(projection.replay_after(-1).is_err());
    assert!(Projection::new().at(-1).is_err());
    let projection = projection
        .add(
            &json!({"name":"loss","unit":"nats/token","value":3,"status":"VALID","step":1}),
            1,
        )
        .unwrap();
    let (retained, after) = projection.clone().for_cutoff(10).unwrap();
    assert_eq!(after, 10);
    assert_eq!(retained.state_value(), projection.state_value());
    let (reset, after) = projection.for_cutoff(9).unwrap();
    assert_eq!(after, 0);
    assert_eq!(reset.state_value(), json!({"cursor":0,"series":{}}));
}
