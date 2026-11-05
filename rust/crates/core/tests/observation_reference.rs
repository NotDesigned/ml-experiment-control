use ml_exp_core::observation::*;
use serde_json::Value;

#[test]
fn legacy_observational_projections_match() {
    let reference: Value =
        serde_json::from_str(include_str!("../../../compatibility/observation.json")).unwrap();
    for case in reference["cases"]["outward"].as_array().unwrap() {
        assert_eq!(sanitized_outward(&case["input"]), case["expected"]);
        assert_eq!(operational_decision(&case["input"]), case["decision"]);
    }
}
