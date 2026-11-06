use ml_exp_client::{
    Error,
    observation::Selection,
    status::{self, Step},
};
use ml_exp_core::diagnostic::HttpFailure;
use serde_json::{Value, json};

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}
fn selection() -> Selection<'static> {
    Selection {
        project: Some("p"),
        run: Some("r"),
        attempt: None,
    }
}
fn begin(case: &Value) -> Result<Step, Error> {
    status::begin(
        "https://example.test/ml-expd",
        &case["health"],
        case.get("saved"),
        selection(),
        case["details"].as_bool().unwrap(),
    )
}
/// Answers each planned GET from `case["responses"]`; any other request fails.
fn drive(case: &Value) -> (Result<Value, Error>, Vec<String>) {
    let mut step = begin(case);
    let mut requests = Vec::new();
    loop {
        step = match step {
            Ok(Step::Read(read)) => {
                requests.push(read.path().to_owned());
                assert!(requests.len() <= 12, "nonterminating status plan");
                let value = case["responses"]
                    .get(read.path())
                    .unwrap_or_else(|| panic!("unexpected request {}", read.path()));
                let response = if let Some(error) = value.get("$error") {
                    Err(Error::Http(Box::new(HttpFailure {
                        status: error["status"].as_u64().unwrap() as u16,
                        retryable: false,
                        error_code: error["error_code"].as_str().unwrap().into(),
                        retry_after: None,
                        details: error.get("details").cloned(),
                        response_body: None,
                        body_truncated: false,
                    })))
                } else {
                    Ok(value.clone())
                };
                read.accept(response)
            }
            Ok(Step::Complete(value)) => return (Ok(value), requests),
            Err(error) => return (Err(error), requests),
        };
    }
}

#[test]
fn saved_status_survives_optional_progress_errors_but_preserves_auth_failures() {
    let fixture = literal(
        r#"{"health":{"capabilities":["execution-progress.v1"]},"saved":{"server":"https://example.test/ml-expd","project":"p","run_id":"r","config_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","preparation":{"preparation_id":"preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}},"details":false}"#,
    );
    let responses = literal(
        r#"{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/experiments/p/r/submissions":{"project":"p","run_id":"r","submissions":[{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}]},"/api/submissions/action-aaaaaaaaaaaaaaaa":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"},"/api/submissions/action-aaaaaaaaaaaaaaaa/progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"/api/projects/p/experiment-preparations/preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb":{"project":"p","run_id":"r","preparation_id":"preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","status":"READY","phase":"READY","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}},"/api/runs/p/r/attempts/attempt-001/collection":{"project":"p","run_id":"r","attempt_id":"attempt-001","status":"AVAILABLE","verification":"SHA256_VERIFIED","download_available":true,"training":{"status":"COMPLETED","exit_code":0}},"/api/attempts/p/r%3A%3Aattempt-001/show":{"run_id":"r","attempt":{"attempt_id":"attempt-001"},"scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"}}}"#,
    );
    let training = literal(
        r#"{"status":"COMPLETED","exit_code":0,"process_state":null,"observed_at":null,"stale":true}"#,
    );
    for status in [401, 403, 404, 405, 409, 500, 503] {
        let mut step = begin(&fixture).unwrap();
        let mut count = 0;
        loop {
            match step {
                Step::Read(read) => {
                    count += 1;
                    assert!(count <= 12);
                    let response = if read.path().ends_with("/progress") {
                        Err(Error::Http(Box::new(HttpFailure {
                            status,
                            retryable: status == 503,
                            error_code: "CLOUD_RUN_BUSY".into(),
                            retry_after: None,
                            details: None,
                            response_body: None,
                            body_truncated: false,
                        })))
                    } else {
                        Ok(responses[read.path()].clone())
                    };
                    match read.accept(response) {
                        Ok(next) => step = next,
                        Err(error) => {
                            assert!([401, 403].contains(&status));
                            assert!(matches!(error,Error::Http(ref e) if e.status==status));
                            break;
                        }
                    }
                }
                Step::Complete(value) => {
                    assert!(![401, 403].contains(&status));
                    assert_eq!(value["project"], "p");
                    assert_eq!(value["run_id"], "r");
                    assert_eq!(value["progress"]["availability"], "UNAVAILABLE");
                    assert_eq!(value["progress"]["error"]["status"], status);
                    assert_eq!(value["training"], training);
                    break;
                }
            }
        }
    }
}

#[test]
fn status_layers_stay_independent_across_saved_current_and_capability_paths() {
    for case in [
        literal(
            r#"{"health":{"capabilities":["execution-progress.v1"]},"details":false,"requests":["/api/runs/p/r","/api/runs/p/r/attempts","/api/experiments/p/r/submissions","/api/submissions/action-aaaaaaaaaaaaaaaa/progress","/api/runs/p/r/attempts/attempt-001/collection"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/experiments/p/r/submissions":{"project":"p","run_id":"r","submissions":[{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}]},"/api/submissions/action-aaaaaaaaaaaaaaaa/progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"/api/runs/p/r/attempts/attempt-001/collection":{"project":"p","run_id":"r","attempt_id":"attempt-001","status":"AVAILABLE","verification":"SHA256_VERIFIED","download_available":true,"training":{"status":"COMPLETED","exit_code":0}}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","preparation":{"status":"NOT_OBSERVED","phase":null},"scheduler":{"state":"RUNNING","backend":null,"job_id":null,"native_state":null,"observed_at":null,"stale":true,"reason":null},"training":{"status":"COMPLETED","exit_code":0,"process_state":null,"observed_at":null,"stale":true},"publication":{"status":"AVAILABLE","phase":null,"download_available":true,"diagnostic":null,"error":null},"verification":{"status":"SHA256_VERIFIED"},"progress":{"availability":"AVAILABLE","phase":"TRAINING"},"queue":null,"worker":{"state":null,"observed_at":null,"stale":true},"model":{"state":"step 10","step":10,"last_metric":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"},"metric_record":{"sequence":9,"event_id":"m9","source":"canonical-journal"},"observed_at":null,"stale":true},"input_delivery":null,"worker_phase":null}}"#,
        ),
        literal(
            r#"{"health":{},"details":false,"requests":["/api/runs/p/r","/api/runs/p/r/attempts","/api/runs/p/r/attempts/attempt-001/collection"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/runs/p/r/attempts/attempt-001/collection":{"project":"p","run_id":"r","attempt_id":"attempt-001","status":"NOT_OBSERVED","verification":"NOT_OBSERVED","download_available":false}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","preparation":{"status":"NOT_OBSERVED","phase":null},"scheduler":{"state":"RUNNING","backend":null,"job_id":null,"native_state":null,"observed_at":null,"stale":true,"reason":null},"training":{"status":"UNKNOWN","exit_code":null,"process_state":null,"observed_at":null,"stale":true},"publication":{"status":"NOT_OBSERVED","phase":null,"download_available":false,"diagnostic":null,"error":null},"verification":{"status":"NOT_OBSERVED"},"progress":{"availability":"NOT_OBSERVED","diagnostic":null},"queue":null,"worker":{"state":null,"observed_at":null,"stale":true},"model":{"state":null,"step":null,"last_metric":null,"observed_at":null,"stale":true,"metric_record":null},"input_delivery":null,"worker_phase":null}}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["execution-progress.v1"]},"details":false,"requests":["/api/runs/p/r","/api/runs/p/r/attempts","/api/experiments/p/r/submissions","/api/submissions/action-aaaaaaaaaaaaaaaa/progress","/api/runs/p/r/attempts/attempt-001/collection"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/experiments/p/r/submissions":{"project":"p","run_id":"r","submissions":[{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}]},"/api/submissions/action-aaaaaaaaaaaaaaaa/progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2","timestamp":0}}},"/api/runs/p/r/attempts/attempt-001/collection":{"project":"p","run_id":"r","attempt_id":"attempt-001","status":"AVAILABLE","verification":"SHA256_VERIFIED","download_available":true,"training":{"status":"COMPLETED","exit_code":0}}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","preparation":{"status":"NOT_OBSERVED","phase":null},"scheduler":{"state":"RUNNING","backend":null,"job_id":null,"native_state":null,"observed_at":null,"stale":true,"reason":null},"training":{"status":"COMPLETED","exit_code":0,"process_state":null,"observed_at":null,"stale":true},"publication":{"status":"AVAILABLE","phase":null,"download_available":true,"diagnostic":null,"error":null},"verification":{"status":"SHA256_VERIFIED"},"progress":{"availability":"AVAILABLE","phase":"TRAINING"},"queue":null,"worker":{"state":null,"observed_at":null,"stale":true},"model":{"state":"step 10","step":10,"last_metric":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2","timestamp":0},"metric_record":{"sequence":9,"event_id":"m9","source":"canonical-journal"},"observed_at":0,"stale":null},"input_delivery":null,"worker_phase":null}}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["execution-progress.v1"]},"details":false,"requests":["/api/runs/p/r","/api/runs/p/r/attempts","/api/experiments/p/r/submissions","/api/submissions/action-aaaaaaaaaaaaaaaa/progress","/api/runs/p/r/attempts/attempt-001/collection"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/experiments/p/r/submissions":{"project":"p","run_id":"r","submissions":[{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}]},"/api/submissions/action-aaaaaaaaaaaaaaaa/progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"/api/runs/p/r/attempts/attempt-001/collection":{"project":"p","run_id":"r","attempt_id":"attempt-001","status":"FAILED","verification":"PENDING","download_available":true,"training":{"status":"COMPLETED","exit_code":0}}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","preparation":{"status":"NOT_OBSERVED","phase":null},"scheduler":{"state":"RUNNING","backend":null,"job_id":null,"native_state":null,"observed_at":null,"stale":true,"reason":null},"training":{"status":"COMPLETED","exit_code":0,"process_state":null,"observed_at":null,"stale":true},"publication":{"status":"FAILED","phase":null,"download_available":false,"diagnostic":null,"error":null},"verification":{"status":"PENDING"},"progress":{"availability":"AVAILABLE","phase":"TRAINING"},"queue":null,"worker":{"state":null,"observed_at":null,"stale":true},"model":{"state":"step 10","step":10,"last_metric":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"},"metric_record":{"sequence":9,"event_id":"m9","source":"canonical-journal"},"observed_at":null,"stale":true},"input_delivery":null,"worker_phase":null}}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["execution-progress.v1"]},"saved":{"server":"https://example.test/ml-expd","project":"p","run_id":"r","config_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","submission":{"project":"p","run_id":"r","attempt_id":"attempt-002","submission_id":"action-aaaaaaaaaaaaaaaa","status":"EXECUTING"}},"details":false,"requests":["/api/submissions/action-aaaaaaaaaaaaaaaa","/api/submissions/action-aaaaaaaaaaaaaaaa/progress","/api/runs/p/r","/api/runs/p/r/attempts"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/submissions/action-aaaaaaaaaaaaaaaa/progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-002","phase":"STAGING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"/api/submissions/action-aaaaaaaaaaaaaaaa":{"project":"p","run_id":"r","attempt_id":"attempt-002","submission_id":"action-aaaaaaaaaaaaaaaa","status":"EXECUTING"}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-002","preparation":{"status":"EXECUTING","phase":null},"scheduler":{"state":"NOT_SUBMITTED","backend":null,"job_id":null,"native_state":null,"observed_at":null,"stale":true,"reason":null},"training":{"status":"UNKNOWN","exit_code":null,"process_state":null,"observed_at":null,"stale":true},"publication":{"status":"NOT_OBSERVED","phase":null,"download_available":false,"diagnostic":null,"error":null},"verification":{"status":"NOT_OBSERVED"},"progress":{"availability":"AVAILABLE","phase":"STAGING"},"queue":null,"worker":{"state":null,"observed_at":null,"stale":true},"model":{"state":"step 10","step":10,"last_metric":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"},"metric_record":{"sequence":9,"event_id":"m9","source":"canonical-journal"},"observed_at":null,"stale":true},"input_delivery":null,"worker_phase":null}}"#,
        ),
    ] {
        let (result, requests) = drive(&case);
        assert_eq!(result.unwrap(), case["expected"]);
        assert_eq!(json!(requests), case["requests"]);
    }
}

#[test]
fn foreign_evidence_and_invalid_listings_are_refused() {
    for case in [
        literal(
            r#"{"health":{},"saved":{"server":"https://example.test/ml-expd","project":"p","run_id":"r","config_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}},"details":false,"requests":["/api/submissions/action-aaaaaaaaaaaaaaaa","/api/runs/p/r","/api/runs/p/r/attempts"],"responses":{"/api/submissions/action-aaaaaaaaaaaaaaaa":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"},"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-002","attempts":[{"attempt_id":"attempt-002","identity":"r::attempt-002","state":"RUNNING"}]}},"error":"requested exact Attempt is absent from the Run listing"}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["execution-progress.v1"]},"details":false,"requests":["/api/runs/p/r"],"responses":{"/api/runs/p/r":{"project":"foreign","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}}},"error":"status response differs from the requested exact Run/Attempt"}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["execution-progress.v1"]},"details":false,"requests":["/api/runs/p/r","/api/runs/p/r/attempts"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":null}},"error":"invalid Run Attempt listing"}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["execution-progress.v1"]},"details":false,"requests":["/api/runs/p/r","/api/runs/p/r/attempts","/api/experiments/p/r/submissions"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/experiments/p/r/submissions":{"project":"p","run_id":"r","submissions":[{"project":"foreign","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}]}},"error":"Submission listing contains a different Run identity"}"#,
        ),
    ] {
        let (result, requests) = drive(&case);
        assert_eq!(
            result.unwrap_err().to_string(),
            case["error"].as_str().unwrap()
        );
        assert_eq!(json!(requests), case["requests"]);
    }
    // The saved server is checked before any read is planned.
    let case = literal(
        r#"{"health":{"capabilities":["execution-progress.v1"]},"saved":{"server":"https://foreign.test/ml-expd","project":"p","run_id":"r","config_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","submission":{"project":"p","run_id":"r","attempt_id":"attempt-002","submission_id":"action-aaaaaaaaaaaaaaaa","status":"EXECUTING"}},"details":false,"error":"observation state belongs to a different server or Run"}"#,
    );
    assert_eq!(
        begin(&case).err().unwrap().to_string(),
        case["error"].as_str().unwrap()
    );
}

#[test]
fn collection_auth_failure_is_an_error_but_a_conflict_becomes_explicit_evidence() {
    let case = literal(
        r#"{"health":{"capabilities":["execution-progress.v1"]},"details":false,"requests":["/api/runs/p/r","/api/runs/p/r/attempts","/api/experiments/p/r/submissions","/api/submissions/action-aaaaaaaaaaaaaaaa/progress","/api/runs/p/r/attempts/attempt-001/collection"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/experiments/p/r/submissions":{"project":"p","run_id":"r","submissions":[{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}]},"/api/submissions/action-aaaaaaaaaaaaaaaa/progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"/api/runs/p/r/attempts/attempt-001/collection":{"$error":{"status":401,"error_code":"READ_FAILURE","details":{"phase":"COLLECTION"}}}}}"#,
    );
    let (result, requests) = drive(&case);
    assert!(matches!(result, Err(Error::Http(ref e)) if e.status == 401));
    assert_eq!(json!(requests), case["requests"]);
    let case = literal(
        r#"{"health":{"capabilities":["execution-progress.v1"]},"details":true,"requests":["/api/runs/p/r","/api/runs/p/r/attempts","/api/experiments/p/r/submissions","/api/submissions/action-aaaaaaaaaaaaaaaa/progress","/api/runs/p/r/attempts/attempt-001/collection","/api/attempts/p/r%3A%3Aattempt-001/show"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/experiments/p/r/submissions":{"project":"p","run_id":"r","submissions":[{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}]},"/api/submissions/action-aaaaaaaaaaaaaaaa/progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"/api/runs/p/r/attempts/attempt-001/collection":{"$error":{"status":409,"error_code":"READ_FAILURE","details":{"phase":"COLLECTION"}}},"/api/attempts/p/r%3A%3Aattempt-001/show":{"run_id":"r","attempt":{"attempt_id":"attempt-001"},"scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"}}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","preparation":{"status":"NOT_OBSERVED","phase":null},"scheduler":{"state":"RUNNING","backend":null,"job_id":null,"native_state":null,"observed_at":null,"stale":true,"reason":null},"training":{"status":"UNKNOWN","exit_code":null,"process_state":null,"observed_at":null,"stale":true},"publication":{"status":"NOT_OBSERVED","phase":null,"download_available":false,"diagnostic":null,"error":{"status":409,"error_code":"READ_FAILURE"}},"verification":{"status":"NOT_OBSERVED"},"progress":{"availability":"AVAILABLE","phase":"TRAINING"},"queue":null,"worker":{"state":null,"observed_at":null,"stale":true},"model":{"state":"step 10","step":10,"last_metric":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"},"metric_record":{"sequence":9,"event_id":"m9","source":"canonical-journal"},"observed_at":null,"stale":true},"input_delivery":null,"worker_phase":null,"evidence":{"submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"},"progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"collection_error":{"status":409,"error_code":"READ_FAILURE","details":{"phase":"COLLECTION"}},"run":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"collection":{},"attempt":{"run_id":"r","attempt":{"attempt_id":"attempt-001"},"scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"}}}}}"#,
    );
    let (result, requests) = drive(&case);
    assert_eq!(result.unwrap(), case["expected"]);
    assert_eq!(json!(requests), case["requests"]);
}

#[test]
fn real_http_status_keeps_queue_training_and_publication_independent() {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        time::{Duration, Instant},
    };
    for c in [
        literal(
            r#"{"health":{"capabilities":["execution-progress.v1"]},"details":false,"requests":["/api/runs/p/r","/api/runs/p/r/attempts","/api/experiments/p/r/submissions","/api/submissions/action-aaaaaaaaaaaaaaaa/progress","/api/runs/p/r/attempts/attempt-001/collection"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/experiments/p/r/submissions":{"project":"p","run_id":"r","submissions":[{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}]},"/api/submissions/action-aaaaaaaaaaaaaaaa/progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"/api/runs/p/r/attempts/attempt-001/collection":{"project":"p","run_id":"r","attempt_id":"attempt-001","status":"AVAILABLE","verification":"SHA256_VERIFIED","download_available":true,"training":{"status":"COMPLETED","exit_code":0}}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","preparation":{"status":"NOT_OBSERVED","phase":null},"scheduler":{"state":"RUNNING","backend":null,"job_id":null,"native_state":null,"observed_at":null,"stale":true,"reason":null},"training":{"status":"COMPLETED","exit_code":0,"process_state":null,"observed_at":null,"stale":true},"publication":{"status":"AVAILABLE","phase":null,"download_available":true,"diagnostic":null,"error":null},"verification":{"status":"SHA256_VERIFIED"},"progress":{"availability":"AVAILABLE","phase":"TRAINING"},"queue":null,"worker":{"state":null,"observed_at":null,"stale":true},"model":{"state":"step 10","step":10,"last_metric":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"},"metric_record":{"sequence":9,"event_id":"m9","source":"canonical-journal"},"observed_at":null,"stale":true},"input_delivery":null,"worker_phase":null}}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["execution-progress.v1"]},"details":true,"requests":["/api/runs/p/r","/api/runs/p/r/attempts","/api/experiments/p/r/submissions","/api/submissions/action-aaaaaaaaaaaaaaaa/progress","/api/runs/p/r/attempts/attempt-001/collection","/api/attempts/p/r%3A%3Aattempt-001/show"],"responses":{"/api/runs/p/r":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"/api/experiments/p/r/submissions":{"project":"p","run_id":"r","submissions":[{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}]},"/api/submissions/action-aaaaaaaaaaaaaaaa/progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"/api/runs/p/r/attempts/attempt-001/collection":{"$error":{"status":409,"error_code":"READ_FAILURE","details":{"phase":"COLLECTION"}}},"/api/attempts/p/r%3A%3Aattempt-001/show":{"run_id":"r","attempt":{"attempt_id":"attempt-001"},"scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"}}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","preparation":{"status":"NOT_OBSERVED","phase":null},"scheduler":{"state":"RUNNING","backend":null,"job_id":null,"native_state":null,"observed_at":null,"stale":true,"reason":null},"training":{"status":"UNKNOWN","exit_code":null,"process_state":null,"observed_at":null,"stale":true},"publication":{"status":"NOT_OBSERVED","phase":null,"download_available":false,"diagnostic":null,"error":{"status":409,"error_code":"READ_FAILURE"}},"verification":{"status":"NOT_OBSERVED"},"progress":{"availability":"AVAILABLE","phase":"TRAINING"},"queue":null,"worker":{"state":null,"observed_at":null,"stale":true},"model":{"state":"step 10","step":10,"last_metric":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"},"metric_record":{"sequence":9,"event_id":"m9","source":"canonical-journal"},"observed_at":null,"stale":true},"input_delivery":null,"worker_phase":null,"evidence":{"submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"},"progress":{"submission_id":"action-aaaaaaaaaaaaaaaa","run_id":"r","attempt_id":"attempt-001","phase":"TRAINING","availability":"AVAILABLE","last_metric":{"sequence":9,"event_id":"m9","source":"canonical-journal","observation":{"name":"loss","value":3.2,"unit":"nats/token","step":10,"variant_id":"v2"}}},"collection_error":{"status":409,"error_code":"READ_FAILURE","details":{"phase":"COLLECTION"}},"run":{"project":"p","run_id":"r","evidence":{"scheduler":{"attempt_id":"attempt-001","state":"RUNNING"}}},"attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"}]},"collection":{},"attempt":{"run_id":"r","attempt":{"attempt_id":"attempt-001"},"scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"}}}}}"#,
        ),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let requests = c["requests"].as_array().unwrap().clone();
        let responses = c["responses"].clone();
        let server = std::thread::spawn(move || {
            for path in requests {
                let deadline = Instant::now() + Duration::from_secs(5);
                let (mut socket, _) = loop {
                    match listener.accept() {
                        Ok(v) => break v,
                        Err(e)
                            if e.kind() == std::io::ErrorKind::WouldBlock
                                && Instant::now() < deadline =>
                        {
                            std::thread::sleep(Duration::from_millis(5))
                        }
                        Err(e) => panic!("missing status GET: {e}"),
                    }
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    let mut byte = [0];
                    socket.read_exact(&mut byte).unwrap();
                    request.extend(byte);
                    assert!(request.len() < 8192);
                }
                let request = String::from_utf8(request).unwrap();
                assert!(
                    request
                        .starts_with(&format!("GET /ml-expd{} HTTP/1.1", path.as_str().unwrap()))
                );
                let request = request.to_lowercase();
                assert!(request.contains("authorization: bearer synthetic-status-token\r\n"));
                assert!(request.contains("x-ml-expd-client-protocol: 2\r\n"));
                let value = &responses[path.as_str().unwrap()];
                let (status, code, body) = if let Some(error) = value.get("$error") {
                    (
                        error["status"].as_u64().unwrap(),
                        error["error_code"].as_str().unwrap(),
                        serde_json::to_vec(&serde_json::json!({"detail":error["details"]}))
                            .unwrap(),
                    )
                } else {
                    (200, "API_ERROR", serde_json::to_vec(value).unwrap())
                };
                write!(socket,"HTTP/1.1 {status} Synthetic\r\nX-ML-Expd-Error-Code: {code}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
                socket.write_all(&body).unwrap();
            }
        });
        let client = ml_exp_client::Client::new(&base, "synthetic-status-token".into()).unwrap();
        let value = status::observe(
            &client,
            &c["health"],
            None,
            Selection {
                project: Some("p"),
                run: Some("r"),
                attempt: None,
            },
            c["details"].as_bool().unwrap(),
        )
        .unwrap();
        assert_eq!(value, c["expected"]);
        server.join().unwrap();
    }
}

#[test]
fn duplicate_exact_attempts_are_refused_instead_of_choosing_the_first() {
    let step = status::begin(
        "https://example.test",
        &json!({}),
        None,
        Selection {
            project: Some("p"),
            run: Some("r"),
            attempt: Some("attempt-001"),
        },
        false,
    )
    .unwrap();
    let Step::Read(read) = step else {
        panic!("expected Run read")
    };
    assert_eq!(read.path(), "/api/runs/p/r");
    let step = read
        .accept(Ok(json!({"project":"p","run_id":"r"})))
        .unwrap();
    let Step::Read(read) = step else {
        panic!("expected Attempt listing")
    };
    let row = json!({"attempt_id":"attempt-001","identity":"r::attempt-001","state":"RUNNING"});
    let result = read.accept(Ok(
        json!({"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[row,row]}),
    ));
    assert!(matches!(
        result,
        Err(Error::Observation(
            "Run listing contains duplicate exact Attempts"
        ))
    ));
}
#[test]
fn missing_scheduler_evidence_never_borrows_process_or_listing_state() {
    for (layer_attempt, freshness) in [("attempt-001", "UNKNOWN"), ("attempt-002", "OBSERVED")] {
        let mut step = status::begin(
            "https://example.test",
            &json!({}),
            None,
            Selection {
                project: Some("p"),
                run: Some("r"),
                attempt: Some("attempt-001"),
            },
            false,
        )
        .unwrap();
        let mut reads = Vec::new();
        loop {
            step = match step {
                Step::Read(read) => {
                    reads.push(read.path().to_owned());
                    let body = match read.path() {
                        "/api/runs/p/r" => {
                            json!({"project":"p","run_id":"r","evidence":{"scheduler":{
                            "attempt_id":layer_attempt,"state":null,"freshness":freshness,"evidence_availability":{"availability":"MISSING"}}}})
                        }
                        "/api/runs/p/r/attempts" => {
                            json!({"project":"p","run_id":"r","current_attempt_id":"attempt-001",
                            "attempts":[{"identity":"r::attempt-001","attempt_id":"attempt-001","state":"SUCCEEDED"}]})
                        }
                        "/api/runs/p/r/attempts/attempt-001/collection" => {
                            json!({"project":"p","run_id":"r","attempt_id":"attempt-001",
                            "training":{"status":"COMPLETED","exit_code":0},"status":"NOT_OBSERVED","verification":"PENDING","download_available":false})
                        }
                        "/api/attempts/p/r%3A%3Aattempt-001/show" => {
                            json!({"run_id":"r","attempt":{"attempt_id":"attempt-001"},
                            "scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},
                            "evidence":{"scheduler":{"attempt_id":"attempt-001","state":null,"freshness":"UNKNOWN"}}})
                        }
                        other => panic!("unexpected request {other}"),
                    };
                    read.accept(Ok(body)).unwrap()
                }
                Step::Complete(value) => {
                    assert_eq!(value["scheduler"]["state"], "UNKNOWN");
                    assert_eq!(value["training"]["status"], "COMPLETED");
                    assert_eq!(value["publication"]["download_available"], false);
                    assert_eq!(
                        reads.len(),
                        if layer_attempt == "attempt-001" { 3 } else { 4 }
                    );
                    break;
                }
            };
        }
    }
}
