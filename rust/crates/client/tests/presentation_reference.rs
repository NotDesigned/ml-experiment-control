//! Actual baseline projections; the test process uses only Rust and synthetic
//! evidence, with no clock, credentials, network or production state.
use ml_exp_client::presentation;
use serde_json::{Value, json};

#[test]
fn human_summaries_match_frozen_python_and_do_not_mutate_scientific_evidence() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../compatibility/client-presentation.json"
    ))
    .unwrap();
    assert_eq!(
        fixture["reference"]["commit"],
        "6fc4f0b8e5a93baf8e89509d65263b6a84eb6c49"
    );
    assert_eq!(
        fixture["reference"]["source"],
        "client/src/ml_exp_client/presentation.py"
    );
    assert_eq!(
        fixture["reference"]["sha256"],
        "7e72132da981fe9b204a15deeffaa40e326d9f0c9153103eeb4ef15ab71f17df"
    );
    let cases = fixture["cases"].as_array().unwrap();
    for case in cases {
        let input = &case["input"];
        let before = input.clone();
        let result = match case["operation"].as_str().unwrap() {
            "status_summary" => presentation::status_summary(input),
            "progress_summary" => presentation::progress_summary(input),
            "operation_summary" => presentation::operation_summary(input),
            "experiment_summary" => presentation::experiment_summary(
                input,
                case["options"]["state_file"].as_str().unwrap(),
                case["options"]["executed"].as_bool().unwrap(),
                case["options"]["client_config"].as_str(),
            ),
            _ => unreachable!(),
        };
        if let Some(expected) = case.get("expected") {
            assert_eq!(
                result.unwrap_or_else(|error| panic!("{}: {error}", case["name"])),
                expected.as_str().unwrap(),
                "{}",
                case["name"]
            );
        } else {
            assert!(result.is_err(), "{} unexpectedly accepted", case["name"]);
        }
        assert_eq!(*input, before, "{} changed its input", case["name"]);
    }
    assert_eq!(cases.len(), 669);
}

#[test]
fn unavailable_publication_does_not_change_completed_training_or_invent_metric_time() {
    let value = json!({"project":"p","run_id":"r","scheduler":{"state":"SUCCEEDED"},"training":{"status":"COMPLETED","exit_code":0},"publication":{"status":"FAILED","download_available":true},"verification":{"status":"PENDING"},"model":{"last_metric":{"name":"validation_loss","value":0,"unit":"nats/token","status":"VALID"}}});
    let text = presentation::status_summary(&value).unwrap();
    assert!(text.contains("Training: COMPLETED"));
    assert!(text.contains("Results: FAILED; download unavailable"));
    assert!(text.contains("Last metric: validation_loss=0 nats/token; VALID"));
    assert!(text.contains("freshness is unknown"));
    assert!(!text.contains("Training: FAILED"));
}
