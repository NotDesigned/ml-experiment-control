use ml_exp_core::{canonical::python_json_bytes, observation::*};
use serde_json::Value;

#[test]
fn legacy_observational_projections_match() {
    let reference: Value =
        serde_json::from_str(include_str!("../../../compatibility/observation.json")).unwrap();
    for (index, case) in reference["cases"]["merge"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let actual = merge_terminal_observation(
            case["previous"].as_object(),
            case["current"].as_object().unwrap(),
        );
        assert_eq!(
            python_json_bytes(&actual).unwrap(),
            python_json_bytes(&case["expected"]).unwrap(),
            "merge case {index}"
        );
    }
    for case in reference["cases"]["outward"].as_array().unwrap() {
        assert_eq!(sanitized_outward(&case["input"]), case["expected"]);
        assert_eq!(operational_decision(&case["input"]), case["decision"]);
    }
    for case in reference["cases"]["record"].as_array().unwrap() {
        let actual = record_progress(
            &case["previous"],
            case["phase"].as_str().unwrap(),
            case["message"].as_str().unwrap(),
            case["at"].as_str().unwrap(),
            case["updated_unix"].as_f64().unwrap(),
            case.get("timeout"),
        )
        .unwrap();
        assert_eq!(
            python_json_bytes(&actual).unwrap(),
            python_json_bytes(&case["expected"]).unwrap()
        );
    }
    for (index, case) in reference["cases"]["progress"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let actual = progress_view(
            &case["input"],
            case["status"].as_str().unwrap(),
            case["active"].as_bool().unwrap(),
            case["last_activity"].as_f64(),
            case["now"].as_f64().unwrap(),
        )
        .unwrap();
        assert_eq!(
            python_json_bytes(&actual).unwrap(),
            python_json_bytes(&case["expected"]).unwrap(),
            "progress case {index}"
        );
    }
}

#[test]
fn time_inputs_are_explicit_and_finite() {
    assert!(record_progress(&Value::Null, "P", "", "", f64::NAN, None).is_err());
    assert!(progress_view(&Value::Null, "RUNNING", true, None, f64::INFINITY).is_err());
    assert!(
        progress_view(
            &serde_json::json!({"updated_unix":"invalid"}),
            "RUNNING",
            true,
            None,
            1000.0
        )
        .is_err()
    );
}
