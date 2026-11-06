use ml_exp_core::observation::*;
use serde_json::Value;

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

#[test]
fn outward_projection_removes_producer_fields_at_every_depth_and_keeps_the_rest() {
    for case in [
        literal(
            r#"{"input":{"child":{"failure_class":"producer","reason":"r"},"failure_class":"bad","metrics":[{"failure_class":"hidden","loss":3.5}]},"expected":{"child":{"reason":"r"},"metrics":[{"loss":3.5}]}}"#,
        ),
        literal(
            r#"{"input":{"step":null,"optimizer_step":5},"expected":{"step":null,"optimizer_step":5}}"#,
        ),
        literal(r#"{"input":null,"expected":null}"#),
    ] {
        assert_eq!(sanitized_outward(&case["input"]), case["expected"]);
    }
}
