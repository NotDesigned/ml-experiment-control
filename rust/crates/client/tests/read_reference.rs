use ml_exp_client::{
    Error,
    observation::Selection,
    read::{self, Step, View},
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
fn response(value: &Value) -> Result<Value, Error> {
    if let Some(error) = value.get("$error") {
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
    }
}
/// Answers each planned GET from `case["responses"]`; any other request fails.
fn drive(case: &Value) -> (Result<Value, Error>, Vec<String>) {
    let mut step = read::begin(
        "https://example.test/ml-expd",
        &case["health"],
        case.get("saved"),
        selection(),
        serde_json::from_value::<View>(case["view"].clone()).unwrap(),
    );
    let mut requests = Vec::new();
    loop {
        step = match step {
            Ok(Step::Read(read)) => {
                requests.push(read.path().to_owned());
                assert!(requests.len() <= 4, "nonterminating read plan");
                let value = case["responses"]
                    .get(read.path())
                    .unwrap_or_else(|| panic!("unexpected GET {}", read.path()));
                read.accept(response(value))
            }
            Ok(Step::Complete(value)) => return (Ok(value), requests),
            Err(error) => return (Err(error), requests),
        };
    }
}

#[test]
fn reads_stay_on_the_saved_attempt_and_encode_unicode_metric_filters() {
    for case in [
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"saved":{"server":"https://example.test/ml-expd","project":"p","run_id":"r","config_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}},"view":{"view":"logs","stream":"both","lines":80},"requests":["/api/submissions/action-aaaaaaaaaaaaaaaa","/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80"],"responses":{"/api/submissions/action-aaaaaaaaaaaaaaaa":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"},"/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80":{"project":"p","run_id":"r","attempt_id":"attempt-001","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"saved":{"server":"https://example.test/ml-expd","project":"p","run_id":"r","config_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","preparation":{"preparation_id":"preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}},"view":{"view":"logs","stream":"both","lines":80},"requests":["/api/projects/p/experiment-preparations/preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","/api/submissions/action-aaaaaaaaaaaaaaaa","/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80"],"responses":{"/api/submissions/action-aaaaaaaaaaaaaaaa":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"},"/api/projects/p/experiment-preparations/preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb":{"project":"p","run_id":"r","preparation_id":"preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"SUCCEEDED"}},"/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80":{"project":"p","run_id":"r","attempt_id":"attempt-001","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"saved":{"contract":"training-operation.v1","server":"https://example.test/ml-expd","project":"p","run_id":"r","operation":"retry","source_attempt_id":"attempt-001","plan":{"action_id":"action-aaaaaaaaaaaaaaaa","operation":"RETRY_ATTEMPT","run_id":"r","attempt_id":"attempt-002","scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},"execution":{"status":"PREPARED"},"intent_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","gate_bundle_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},"request":{"new_attempt_id":"attempt-002"}},"view":{"view":"metrics","keys":null,"names":null,"max_points":2000},"requests":["/api/actions/action-aaaaaaaaaaaaaaaa","/api/attempts/p/r%3A%3Aattempt-002/metrics?max_points=2000&format=records"],"responses":{"/api/actions/action-aaaaaaaaaaaaaaaa":{"action_id":"action-aaaaaaaaaaaaaaaa","operation":"RETRY_ATTEMPT","run_id":"r","attempt_id":"attempt-002","scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},"execution":{"status":"PREPARED"},"intent_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","gate_bundle_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},"/api/attempts/p/r%3A%3Aattempt-002/metrics?max_points=2000&format=records":{"project":"p","run_id":"r","attempt_id":"attempt-002","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-002","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"view":{"view":"metrics","keys":"loss,, val_loss , loss","names":["validation loss","训练/损失","*~"],"max_points":2000},"requests":["/api/runs/p/r/attempts","/api/attempts/p/r%3A%3Aattempt-001/metrics?max_points=2000&format=records&name=validation+loss&name=%E8%AE%AD%E7%BB%83%2F%E6%8D%9F%E5%A4%B1&name=%2A~&name=loss&name=val_loss&name=loss"],"responses":{"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001"}]},"/api/attempts/p/r%3A%3Aattempt-001/metrics?max_points=2000&format=records&name=validation+loss&name=%E8%AE%AD%E7%BB%83%2F%E6%8D%9F%E5%A4%B1&name=%2A~&name=loss&name=val_loss&name=loss":{"project":"p","run_id":"r","attempt_id":"attempt-001","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}}"#,
        ),
    ] {
        let (result, requests) = drive(&case);
        assert_eq!(result.unwrap(), case["expected"]);
        assert_eq!(json!(requests), case["requests"]);
    }
}

#[test]
fn invalid_views_and_unbound_saved_state_are_refused_before_any_request() {
    for case in [
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"view":{"view":"logs","stream":"both","lines":0},"requests":[],"responses":{},"error":"logs require stdout/stderr/both and 1..10000 lines"}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"view":{"view":"metrics","keys":null,"names":["损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损损"],"max_points":2000},"requests":[],"responses":{},"error":"name filters must be valid metric names"}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["Metric-records.v1"]},"view":{"view":"metrics","keys":null,"names":["loss"],"max_points":2000},"requests":[],"responses":{},"error":"server lacks metric-records.v1; --keys retains the legacy field-filter behavior"}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"saved":{"server":"https://example.test/ml-expd","project":"p","run_id":"r","config_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},"view":{"view":"logs","stream":"both","lines":80},"requests":[],"responses":{},"error":"saved state has no bound Attempt yet; use run status --state to inspect preparation"}"#,
        ),
    ] {
        let (result, requests) = drive(&case);
        assert_eq!(
            result.unwrap_err().to_string(),
            case["error"].as_str().unwrap()
        );
        assert!(requests.is_empty());
    }
}

#[test]
fn a_failed_read_keeps_its_http_status() {
    let case = literal(
        r#"{"health":{"capabilities":["metric-records.v1"]},"view":{"view":"logs","stream":"both","lines":80},"requests":["/api/runs/p/r/attempts","/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80"],"responses":{"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001"}]},"/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80":{"$error":{"status":409,"error_code":"READ_FAILURE","details":{"stage":"read","safe_to_retry":false}}}}}"#,
    );
    let (result, requests) = drive(&case);
    assert!(matches!(result, Err(Error::Http(ref failure)) if failure.status == 409));
    assert_eq!(json!(requests), case["requests"]);
}

#[test]
fn a_reply_for_another_run_or_attempt_is_refused() {
    let listing = json!({"project":"p","run_id":"r","current_attempt_id":"attempt-001",
        "attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001"}]});
    let mut foreign = listing.clone();
    foreign["run_id"] = json!("other");
    let mut absent = listing.clone();
    absent["attempts"] = json!([{"attempt_id":"attempt-002","identity":"r::attempt-002"}]);
    for (attempts, logs, count, error) in [
        (
            foreign,
            Value::Null,
            1,
            "Attempt listing differs from the requested Run",
        ),
        (
            absent,
            Value::Null,
            1,
            "current Attempt is absent from the Run listing",
        ),
        (
            listing.clone(),
            json!({"project":"p","run_id":"r","attempt_id":"attempt-002","stdout":["another Attempt"]}),
            2,
            "read response differs from the requested exact Run/Attempt",
        ),
        (
            listing.clone(),
            json!({"project":"p","run_id":"other","attempt_id":"attempt-001","stdout":["another Run"]}),
            2,
            "read response differs from the requested exact Run/Attempt",
        ),
    ] {
        let case = json!({"health":{"capabilities":["metric-records.v1"]},
            "view":{"view":"logs","stream":"both","lines":80},
            "responses":{"/api/runs/p/r/attempts":attempts,
                "/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80":logs}});
        let (result, requests) = drive(&case);
        assert_eq!(result.unwrap_err().to_string(), error);
        assert_eq!(requests.len(), count);
    }
}

#[test]
fn duplicate_current_attempts_are_rejected_and_a_resolved_attempt_stays_pinned() {
    let current = || {
        read::begin(
            "https://example.test/ml-expd",
            &json!({}),
            None,
            Selection {
                project: Some("p"),
                run: Some("r"),
                attempt: None,
            },
            View::Logs {
                stream: "both".into(),
                lines: 80,
            },
        )
        .unwrap()
    };
    let Step::Read(read) = current() else {
        panic!("expected one current listing")
    };
    let row = json!({"attempt_id":"attempt-001","identity":"r::attempt-001"});
    let result = read.accept(Ok(
        json!({"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[row,row]}),
    ));
    assert!(matches!(
        result,
        Err(Error::Observation(
            "Run listing contains duplicate exact Attempts"
        ))
    ));

    let Step::Read(read) = current() else {
        panic!("expected one current listing")
    };
    let Step::Read(read) = read
        .accept(Ok(
            json!({"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[row]}),
        ))
        .unwrap()
    else {
        panic!("expected pinned log read")
    };
    assert_eq!(
        read.path(),
        "/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80"
    );
    let result = json!({"project":"p","run_id":"r","attempt_id":"attempt-001","current_attempt_id":"attempt-002","stdout":["still the original Attempt"]});
    let Step::Complete(actual) = read.accept(Ok(result.clone())).unwrap() else {
        panic!("a pinned read must not discover another Attempt")
    };
    assert_eq!(actual, result);
}
#[test]
fn real_http_reads_keep_prefix_credentials_identity_and_unmodified_metric_dimensions() {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        time::{Duration, Instant},
    };
    for case in [
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"view":{"view":"metrics","keys":"loss,, val_loss , loss","names":["validation loss","训练/损失","*~"],"max_points":2000},"requests":["/api/runs/p/r/attempts","/api/attempts/p/r%3A%3Aattempt-001/metrics?max_points=2000&format=records&name=validation+loss&name=%E8%AE%AD%E7%BB%83%2F%E6%8D%9F%E5%A4%B1&name=%2A~&name=loss&name=val_loss&name=loss"],"responses":{"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001"}]},"/api/attempts/p/r%3A%3Aattempt-001/metrics?max_points=2000&format=records&name=validation+loss&name=%E8%AE%AD%E7%BB%83%2F%E6%8D%9F%E5%A4%B1&name=%2A~&name=loss&name=val_loss&name=loss":{"project":"p","run_id":"r","attempt_id":"attempt-001","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-001","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"saved":{"contract":"training-operation.v1","server":"https://example.test/ml-expd","project":"p","run_id":"r","operation":"retry","source_attempt_id":"attempt-001","plan":{"action_id":"action-aaaaaaaaaaaaaaaa","operation":"RETRY_ATTEMPT","run_id":"r","attempt_id":"attempt-002","scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},"execution":{"status":"PREPARED"},"intent_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","gate_bundle_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},"request":{"new_attempt_id":"attempt-002"}},"view":{"view":"metrics","keys":null,"names":null,"max_points":2000},"requests":["/api/actions/action-aaaaaaaaaaaaaaaa","/api/attempts/p/r%3A%3Aattempt-002/metrics?max_points=2000&format=records"],"responses":{"/api/actions/action-aaaaaaaaaaaaaaaa":{"action_id":"action-aaaaaaaaaaaaaaaa","operation":"RETRY_ATTEMPT","run_id":"r","attempt_id":"attempt-002","scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},"execution":{"status":"PREPARED"},"intent_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","gate_bundle_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},"/api/attempts/p/r%3A%3Aattempt-002/metrics?max_points=2000&format=records":{"project":"p","run_id":"r","attempt_id":"attempt-002","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}},"expected":{"project":"p","run_id":"r","attempt_id":"attempt-002","stdout":["loss 3.1"],"stderr":[],"metrics":[{"name":"validation_loss","value":3.1,"unit":"nats/token","step":10,"variant_id":"masked","protocol_sha256":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","checkpoint_sha256":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","numerator":31.0,"denominator":10,"dataset":"fineweb-v1"}],"evaluation_status":"PARTIAL","next_cursor":null}}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"view":{"view":"logs","stream":"both","lines":80},"requests":["/api/runs/p/r/attempts","/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80"],"responses":{"/api/runs/p/r/attempts":{"project":"p","run_id":"r","current_attempt_id":"attempt-001","attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001"}]},"/api/attempts/p/r%3A%3Aattempt-001/logs?stream=both&lines=80":{"$error":{"status":409,"error_code":"READ_FAILURE","details":{"stage":"read","safe_to_retry":false}}}},"error_status":409}"#,
        ),
        literal(
            r#"{"health":{"capabilities":["metric-records.v1"]},"saved":{"server":"https://example.test/ml-expd","project":"p","run_id":"r","config_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},"view":{"view":"logs","stream":"both","lines":80},"requests":[],"responses":{},"error":"saved state has no bound Attempt yet; use run status --state to inspect preparation"}"#,
        ),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let requests = case["requests"].as_array().unwrap().clone();
        let responses = case["responses"].clone();
        let server = std::thread::spawn(move || {
            for path in requests {
                let deadline = Instant::now() + Duration::from_secs(5);
                let (mut socket, _) = loop {
                    match listener.accept() {
                        Ok(value) => break value,
                        Err(error)
                            if error.kind() == std::io::ErrorKind::WouldBlock
                                && Instant::now() < deadline =>
                        {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("missing exact read GET: {error}"),
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
                    assert!(request.len() < 16384);
                }
                let request = String::from_utf8(request).unwrap();
                assert!(request.starts_with(&format!(
                    "GET /ml-expd{} HTTP/1.1\r\n",
                    path.as_str().unwrap()
                )));
                let headers = request.to_ascii_lowercase();
                assert_eq!(
                    headers
                        .matches("authorization: bearer synthetic-read-token\r\n")
                        .count(),
                    1
                );
                assert_eq!(
                    headers.matches("x-ml-expd-client-protocol: 2\r\n").count(),
                    1
                );
                let response = &responses[path.as_str().unwrap()];
                let (status, body) = if let Some(error) = response.get("$error") {
                    (
                        error["status"].as_u64().unwrap(),
                        json!({"detail":error["details"]}),
                    )
                } else {
                    (200, response.clone())
                };
                let body = serde_json::to_vec(&body).unwrap();
                write!(socket, "HTTP/1.1 {status} Synthetic\r\nX-ML-Expd-Error-Code: READ_FAILURE\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
                socket.write_all(&body).unwrap();
            }
            // Success, HTTP errors and an unbound saved state all finish
            // without POSTs, retry requests or a fallback to current/latest.
            let deadline = Instant::now() + Duration::from_millis(100);
            while Instant::now() < deadline {
                assert!(
                    matches!(listener.accept(), Err(error) if error.kind()==std::io::ErrorKind::WouldBlock)
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        let client = ml_exp_client::Client::new(&base, "synthetic-read-token".into()).unwrap();
        let saved = case.get("saved").map(|value| {
            let mut saved = value.clone();
            saved["server"] = json!(base);
            saved
        });
        let result = read::observe(
            &client,
            &case["health"],
            saved.as_ref(),
            selection(),
            serde_json::from_value(case["view"].clone()).unwrap(),
        );
        if let Some(status) = case["error_status"].as_u64() {
            assert!(
                matches!(result, Err(Error::Http(ref error)) if u64::from(error.status)==status && error.error_code=="READ_FAILURE")
            );
        } else if let Some(expected) = case["error"].as_str() {
            assert_eq!(result.unwrap_err().to_string(), expected);
        } else {
            assert_eq!(result.unwrap(), case["expected"]);
        }
        server.join().unwrap();
    }
}
