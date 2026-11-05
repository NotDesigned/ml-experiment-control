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
    assert_eq!(count, 83);
}
