//! Human status summaries over synthetic evidence; no clock, credentials or network.
use ml_exp_client::presentation;
use serde_json::{Value, json};

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

#[test]
fn status_summary_lines_describe_each_evidence_layer_without_changing_input() {
    for (input, expected) in [
        (
            literal(r#"{}"#),
            "Run: UNKNOWN/UNKNOWN · Attempt not observed\nStage: UNKNOWN\nScheduler: UNKNOWN\nBackend: not observed; job not observed; native not observed\nTraining: UNKNOWN; process not observed\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: NOT_OBSERVED; download unavailable\nVerification: NOT_OBSERVED\nNext: inspect existing evidence; no action specified",
        ),
        (
            literal(
                r#"{"project":"demo","run_id":"trial","attempt_id":"attempt-001","scheduler":{"state":"FAILED"},"training":{"status":"UNKNOWN"},"publication":{"status":"RELEASED","download_available":true},"verification":{"status":"PENDING"}}"#,
            ),
            "Run: demo/trial · attempt-001\nStage: FAILED\nScheduler: FAILED\nBackend: not observed; job not observed; native not observed\nTraining: UNKNOWN; process not observed\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: RELEASED; download unavailable\nVerification: PENDING\nNext: inspect existing evidence; no action specified",
        ),
        (
            literal(
                r#"{"project":"demo","run_id":"trial","attempt_id":"attempt-001","scheduler":{"state":"QUEUED"},"training":{"status":"COMPLETED","exit_code":0},"publication":{"status":"RELEASED","download_available":false},"verification":{"status":"SHA256_VERIFIED"},"queue":{"reason":"Current Limit is CPU: 200 limited by Role PAI.AlgoDeveloper.GPU: 8 limited by Role PAI.AlgoDeveloper.","provider_code":"CheckSelfQuotaFailed","position":1,"blocker":{"CodeType":"ConfigRule","Code":"roleMaximumResource","Reason":"Current Limit is CPU: 200 limited by Role PAI.AlgoDeveloper.GPU: 8 limited by Role PAI.AlgoDeveloper."},"as_of":"2026-10-24T00:00:00Z","stale":true,"estimated_start_at":null},"progress":{"phase":"QUEUED"}}"#,
            ),
            "Run: demo/trial · attempt-001\nStage: QUEUED\nScheduler: QUEUED\nBackend: not observed; job not observed; native not observed\nWaiting: Current Limit is CPU: 200 limited by Role PAI.AlgoDeveloper.GPU: 8 limited by Role PAI.AlgoDeveloper. (roleMaximumResource; CheckSelfQuotaFailed; position 1; observed 2026-10-24T00:00:00Z; stale evidence)\nTraining: COMPLETED; process not observed; exit 0\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: RELEASED; download unavailable\nVerification: SHA256_VERIFIED\nNext: inspect existing evidence; no action specified",
        ),
        (
            literal(
                r#"{"project":"demo","run_id":"trial","attempt_id":"attempt-001","scheduler":{"state":"SUCCEEDED","observed_at":1792051748.25,"stale":false},"training":{"status":"UNKNOWN"},"publication":{"status":"RELEASED","download_available":false},"verification":{"status":"SHA256_VERIFIED"},"worker_phase":{"phase":"TRAINING"},"progress":{"phase":"TRAINING"}}"#,
            ),
            "Run: demo/trial · attempt-001\nStage: TRAINING (last reported worker phase; not a heartbeat)\nScheduler: SUCCEEDED; observed 2026-10-15T08:09:08+00:00\nBackend: not observed; job not observed; native not observed\nTraining: UNKNOWN; process not observed\nWorker phase: TRAINING; last reported; not a heartbeat\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: RELEASED; download unavailable\nVerification: SHA256_VERIFIED\nNext: inspect existing evidence; no action specified",
        ),
        (
            literal(
                r#"{"scheduler":{},"model":{},"training":{"diagnostic":false},"progress":{},"publication":{},"verification":{},"queue":{"position":"中文😀"},"input_delivery":{},"worker_phase":{"diagnostic":[],"step":1,"position":0,"message":"中文😀"},"project":"p","run_id":"r","attempt_id":"attempt-001"}"#,
            ),
            "Run: p/r · attempt-001\nStage: UNKNOWN\nScheduler: UNKNOWN\nBackend: not observed; job not observed; native not observed\nTraining: UNKNOWN; process not observed\nWorker phase: UNKNOWN; last reported; not a heartbeat\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: NOT_OBSERVED; download unavailable\nVerification: NOT_OBSERVED\nNext: inspect existing evidence; no action specified",
        ),
    ] {
        let before = input.clone();
        assert_eq!(presentation::status_summary(&input).unwrap(), expected);
        assert_eq!(input, before);
    }
}

#[test]
fn status_summary_refuses_a_document_that_is_not_an_object() {
    for input in [
        json!(null),
        json!([]),
        json!(0),
        json!(false),
        json!("text"),
    ] {
        assert!(presentation::status_summary(&input).is_err(), "{input}");
    }
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
