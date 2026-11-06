use ml_exp_core::failures::{self, AssessmentContext};
use serde_json::{Value, json};

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}
fn same_json(actual: &Value, expected: &Value) {
    // arbitrary_precision retains JSON exponent spelling (-1e-6/-1e-06).
    // Canonical bytes compare exact JSON numbers and all other fields, rather
    // than treating those equivalent parser spellings as a mismatch.
    assert_eq!(
        ml_exp_core::canonical::bytes(actual).unwrap(),
        ml_exp_core::canonical::bytes(expected).unwrap(),
    );
}

#[test]
fn terminal_process_evidence_is_applicable_but_contextual_decisions_are_not() {
    for case in [
        literal(
            r#"{"collection":{"attempt_id":"attempt-002","process_state":"FAILED","process_evidence":{"stderr_tail":["TIMEOUT"]}},"context":{"attempt_id":"attempt-002","attempt_state":"FAILED","attempt_started_at":100,"observed_at":200,"naive_utc_offset_seconds":0},"expected":{"failure_summary":{"failure_signature":"TIMEOUT","failure_class":"timeout","phase":"unknown","source":"collection.process_evidence","failure_domain":"process","attempt_id":"attempt-002","observed_at":200.0,"applicability":"APPLICABLE","evidence_source":null},"diagnostic_evidence":[]}}"#,
        ),
        literal(
            r#"{"collection":{"attempt_id":"attempt-002"},"context":{"attempt_id":"attempt-002","attempt_state":"RUNNING","attempt_started_at":100,"observed_at":200,"naive_utc_offset_seconds":0,"decision_observed_at":1,"decision_source_binding":"RUN_ROOT_FALLBACK","decision_evidence_source":"/synthetic/old/decision.json"},"decision":{"failure_class":"unknown"},"expected":{"failure_summary":null,"diagnostic_evidence":[{"kind":"preliminary_failure_classification","failure_class":"unknown","source":"decision.failure_class","attempt_id":"attempt-002","attempt_state":"RUNNING","observed_at":1.0,"applicability":"NON_APPLICABLE","source_binding":"RUN_ROOT_FALLBACK","evidence_source":"/synthetic/old/decision.json","reason":"decision metadata is contextual only and is not exact, fresh, terminal failure evidence"}]}}"#,
        ),
        literal(
            r#"{"collection":{"attempt_id":"attempt-002"},"context":{"attempt_id":"attempt-002","attempt_state":"RUNNING","attempt_started_at":100,"observed_at":200,"naive_utc_offset_seconds":0,"decision_observed_at":1,"decision_source_binding":"RUN_ROOT_FALLBACK","decision_evidence_source":"/synthetic/old/decision.json"},"expected":{"failure_summary":null,"diagnostic_evidence":[]}}"#,
        ),
    ] {
        let context: AssessmentContext = serde_json::from_value(case["context"].clone()).unwrap();
        let result = failures::attempt_failure_evidence_assessment(
            &case["collection"],
            case.get("decision"),
            &context,
        );
        same_json(&result.unwrap(), &case["expected"]);
    }
}

#[test]
fn a_tail_that_is_not_a_list_is_refused() {
    let case = literal(
        r#"{"collection":{"attempt_id":"attempt-002","process_state":"FAILED","process_evidence":{"stderr_tail":12}},"context":{"attempt_id":"attempt-002","attempt_state":"FAILED","attempt_started_at":100,"observed_at":200,"naive_utc_offset_seconds":0}}"#,
    );
    let context: AssessmentContext = serde_json::from_value(case["context"].clone()).unwrap();
    assert!(
        failures::attempt_failure_evidence_assessment(
            &case["collection"],
            case.get("decision"),
            &context
        )
        .is_err()
    );
}

#[test]
fn compaction_drops_raw_output() {
    for (input, expected) in [
        (
            r#"{"raw_stdout":"secret","stderr_tail":["tail"],"nested":{"startup_script":"hidden","value":1}}"#,
            r#"{"nested":{"value":1}}"#,
        ),
        (
            r#"{"nats_per_token":1e-06,"large":1e+20,"zero":-0.0}"#,
            r#"{"nats_per_token":1e-06,"large":1e+20,"zero":-0.0}"#,
        ),
    ] {
        assert_eq!(
            failures::compact_evidence(&literal(input)),
            literal(expected)
        );
    }
}

#[test]
fn unknown_timezone_nonfinite_or_unsupported_time_cannot_promote_a_failure_signal() {
    let collection = json!({"attempt_id":"attempt-001","process_state":"FAILED","process_evidence":{"stderr_tail":["CUDA out of memory"]}});
    for observed in [
        json!("1970-01-01T00:03:20"),
        json!("1970-01-01T00:03:20z"),
        json!("1970-01-01T00:03:60Z"),
        json!("1970-W01-4T00:03:20Z"),
        serde_json::from_str::<Value>("1e999").unwrap(),
    ] {
        let context = AssessmentContext {
            attempt_state: Some("FAILED".into()),
            attempt_started_at: json!(100),
            observed_at: observed,
            ..AssessmentContext::new("attempt-001")
        };
        let assessment =
            failures::attempt_failure_evidence_assessment(&collection, None, &context).unwrap();
        assert!(assessment["failure_summary"].is_null());
        assert_eq!(
            assessment["diagnostic_evidence"][0]["applicability"],
            "UNKNOWN_APPLICABILITY"
        );
    }
    assert_eq!(
        failures::evidence_timestamp(&json!("1970-01-01T00:03:20"), Some(0)),
        Some(200.0)
    );
    assert_eq!(
        failures::evidence_timestamp(&json!("1970-01-01T08:03:20"), Some(8 * 3600)),
        Some(200.0)
    );
    assert_eq!(failures::evidence_timestamp(&json!(true), Some(0)), None);
}
