use ml_exp_core::observation::*;
use serde_json::Value;

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

#[test]
fn outward_projection_removes_producer_fields_at_every_depth_and_keeps_the_rest() {
    for case in [
        literal(
            r#"{"input":{"decision":{"action":"retry","failure_class":"producer","reason":"r","retries_used":0},"failure_class":"bad","metrics":[{"failure_class":"hidden","loss":3.5}]},"expected":{"decision":{"action":"retry","reason":"r","retries_used":0},"metrics":[{"loss":3.5}]},"decision":{}}"#,
        ),
        literal(
            r#"{"input":{"decision":{"action":null,"reason":false,"extra":true,"resume_checkpoint":"cp","retries_allowed":2,"retries_used":1},"child":{"decision":{"action":"stop"}}},"expected":{"decision":{"reason":false,"resume_checkpoint":"cp","retries_allowed":2,"retries_used":1},"child":{"decision":{"action":"stop"}}},"decision":{}}"#,
        ),
        literal(
            r#"{"input":{"step":null,"optimizer_step":5},"expected":{"step":null,"optimizer_step":5},"decision":{}}"#,
        ),
        literal(r#"{"input":null,"expected":null,"decision":{}}"#),
    ] {
        assert_eq!(sanitized_outward(&case["input"]), case["expected"]);
        assert_eq!(operational_decision(&case["input"]), case["decision"]);
    }
}
