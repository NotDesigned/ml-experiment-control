//! Executed exclusively in Rust. Fixtures were produced by the frozen Python
//! reference modules, never by this implementation.
use ml_exp_core::records::{self, Event, Scope};
use serde_json::{Value, json};

fn evaluate(case: &Value) -> records::Result<Value> {
    let input = &case["input"];
    match case["operation"].as_str().unwrap() {
        "normalized" => records::normalized_metric(&input["record"], &input["config"]),
        "scope" => {
            let scope = Scope::new(
                input["project"].as_str().unwrap(),
                input["run"].as_str().unwrap(),
                input["attempt"].as_str().unwrap(),
            )
            .encoded()?;
            Ok(
                json!({"scope":scope,"wandb_id":records::wandb_id(input["workspace"].as_str().unwrap(),&scope,&input["route"])?}),
            )
        }
        "fence" => Ok(json!(records::fence_names(
            input["run"].as_str().unwrap(),
            &input["config"]
        )?)),
        "prepared" => {
            let prepared = records::prepare_event(&Event {
                event_id: input["event_id"].as_str().unwrap().into(),
                payload: input["payload"].clone(),
            })?;
            Ok(
                json!({"event_id":prepared.event_id,"body":prepared.body,"checksum":prepared.checksum,"dedup":prepared.dedup}),
            )
        }
        "worker_raw" => {
            let bytes = input["bytes_hex"]
                .as_str()
                .unwrap()
                .as_bytes()
                .chunks_exact(2)
                .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
                .collect::<Vec<_>>();
            Ok(
                serde_json::to_value(records::worker_batch_json(&bytes, &input["config"])?)
                    .unwrap(),
            )
        }
        "worker" => Ok(serde_json::to_value(records::worker_batch(
            &input["body"],
            &input["config"],
        )?)
        .unwrap()),
        _ => unreachable!(),
    }
}

#[test]
fn canonical_record_contract_matches_frozen_python_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/records-store.json")).unwrap();
    assert_eq!(
        fixture["reference"]["sources"]["server/src/ml_exp_server/records/store.py"],
        "8539e42f3e48979120bb4c2b3431e3c486a5c666d4d739a83f002550770a4c79"
    );
    let mut count = 0;
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["operation"] != "import")
    {
        let result = evaluate(case);
        if let Some(expected) = case.get("expected") {
            assert_eq!(
                result.unwrap_or_else(|e| panic!("{}: {e}", case["id"])),
                *expected,
                "{}",
                case["id"]
            );
        } else {
            assert!(result.is_err(), "{} unexpectedly accepted", case["id"]);
        }
        count += 1;
    }
    assert_eq!(count, 278);
}

#[test]
fn raw_worker_nonfinite_values_remain_validation_failures_and_cannot_be_lifecycle_fields() {
    let cases=[
        (br#"{"records":[{"event_id":"m","kind":"metrics","data":{"name":"loss","unit":"nats/token","value":NaN}}]}"#.as_slice(),"nan"),
        (br#"{"records":[{"event_id":"m","kind":"metrics","data":{"name":"loss","unit":"nats/token","value":[Infinity]}}]}"#.as_slice(),"[inf]"),
        (br#"{"records":[{"event_id":"m","kind":"metrics","data":{"name":"loss","unit":"nats/token","value":1e400}}]}"#.as_slice(),"inf"),
    ];
    for (body, invalid) in cases {
        let result = records::worker_batch_json(body, &json!({})).unwrap();
        assert_eq!(result[0].payload["observations"][0]["status"], "FAILED");
        assert_eq!(
            result[0].payload["observations"][0]["invalid_value"],
            invalid
        );
    }
    for body in [
        br#"{"records":[{"event_id":"x","kind":"lifecycle","data":{"step":NaN}}]}"#.as_slice(),
        br#"{"records":[{"event_id":"x","kind":"checkpoints","data":{"pid":1e400}}]}"#.as_slice(),
    ] {
        assert!(records::worker_batch_json(body, &json!({})).is_err());
    }
    let body=br#"{"records":[{"event_id":"x","kind":"metrics","data":{"name":"loss","unit":"u","value":1,"error":{"detail":[Infinity,-Infinity]}}}]}"#;
    let record =
        &records::worker_batch_json(body, &json!({})).unwrap()[0].payload["observations"][0];
    assert_eq!(record["error"], "{'detail': [inf, -inf]}");
}

#[test]
fn nonfinite_token_parser_never_reinterprets_user_strings_or_relaxes_json_syntax() {
    let body=br#"{"records":[{"event_id":"m","kind":"metrics","data":{"name":"loss","unit":"u","value":1,"step":NaN,"error":"\u0000ml-exp-nonfinite:0:0"}}]}"#;
    let events = records::worker_batch_json(body, &json!({})).unwrap();
    let observation = &events[0].payload["observations"][0];
    assert_eq!(observation["error"], "\0ml-exp-nonfinite:0:0");
    assert!(observation.get("step").is_none());
    let quoted=br#"{"records":[{"event_id":"m","kind":"metrics","data":{"name":"loss","unit":"u","value":"NaN"}}]}"#;
    let events = records::worker_batch_json(quoted, &json!({})).unwrap();
    assert_eq!(events[0].payload["observations"][0]["invalid_value"], "NaN");
    for body in [
        b"NaN1".as_slice(),
        b"[Infinityx]".as_slice(),
        b"{} trailing".as_slice(),
        b"{\"records\":[],}".as_slice(),
    ] {
        assert!(records::worker_batch_json(body, &json!({})).is_err());
    }
}
