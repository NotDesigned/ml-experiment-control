use ml_exp_core::failures::{self, AssessmentContext};
use serde_json::{Value, json};

fn same_json(actual: &Value, expected: &Value, name: &Value) {
    // arbitrary_precision retains JSON exponent spelling (-1e-6/-1e-06).
    // Canonical bytes compare exact Python JSON numbers and all other fields,
    // rather than treating those equivalent parser spellings as a mismatch.
    assert_eq!(
        ml_exp_core::canonical::python_json_bytes(actual).unwrap(),
        ml_exp_core::canonical::python_json_bytes(expected).unwrap(),
        "{name}"
    );
}

fn fixtures() -> Value {
    serde_json::from_str(include_str!("../../../compatibility/failures.json")).unwrap()
}
#[test]
fn exact_attempt_failure_assessments_match_actual_python_reference() {
    let fixtures = fixtures();
    assert_eq!(
        fixtures["baseline_commit"],
        "e725faec689fb8395e692e27d00a8896fa7fe5a5"
    );
    assert_eq!(fixtures["cases"].as_array().unwrap().len(), 248);
    for case in fixtures["cases"].as_array().unwrap() {
        let context: AssessmentContext = serde_json::from_value(case["context"].clone()).unwrap();
        let decision = case.get("decision").filter(|value| !value.is_null());
        let result =
            failures::attempt_failure_evidence_assessment(&case["collection"], decision, &context);
        if case.get("error").is_some() {
            assert!(result.is_err(), "{} unexpectedly accepted", case["name"]);
        } else {
            same_json(&result.unwrap(), &case["expected"], &case["name"]);
            same_json(
                &failures::structured_failure_summary(
                    &case["collection"],
                    decision,
                    Some(&context),
                )
                .unwrap()
                .unwrap_or(Value::Null),
                &case["structured"],
                &case["name"],
            );
        }
        assert!(
            failures::structured_failure_summary(&case["collection"], decision, None)
                .unwrap()
                .is_none()
        );
    }
}
#[test]
fn evidence_compaction_digest_and_review_only_predicate_match_python() {
    let fixtures = fixtures();
    for case in fixtures["compact"].as_array().unwrap() {
        assert_eq!(failures::compact_evidence(&case["input"]), case["expected"]);
        assert_eq!(
            failures::evidence_digest(&case["input"]).unwrap(),
            case["digest"]
        );
    }
    for case in fixtures["review_only"].as_array().unwrap() {
        assert_eq!(
            failures::reviewable_unknown_retry(
                &case["decision"],
                case["approval"].as_str().unwrap(),
                case["reason"].as_str()
            ),
            case["expected"].as_bool().unwrap()
        );
        assert!(!failures::reviewable_unknown_retry(
            &case["decision"],
            "automatic",
            Some("approved")
        ));
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
