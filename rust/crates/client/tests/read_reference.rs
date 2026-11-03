use ml_exp_client::{
    Error,
    observation::Selection,
    read::{self, Step, View},
};
use ml_exp_core::diagnostic::HttpFailure;
use serde_json::{Value, json};

fn fixtures() -> Value {
    serde_json::from_str(include_str!("../../../compatibility/client-read.json")).unwrap()
}
fn selection(value: &Value) -> Selection<'_> {
    Selection {
        project: value["project"].as_str(),
        run: value["run"].as_str(),
        attempt: value["attempt"].as_str(),
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

#[test]
fn exact_attempt_read_plans_match_frozen_python_reference() {
    let fixtures = fixtures();
    assert_eq!(
        fixtures["baseline_commit"],
        "e725faec689fb8395e692e27d00a8896fa7fe5a5"
    );
    for case in fixtures["cases"].as_array().unwrap() {
        let mut step = read::begin(
            case["base"].as_str().unwrap(),
            &case["health"],
            case.get("saved").filter(|value| !value.is_null()),
            selection(&case["selection"]),
            serde_json::from_value::<View>(case["view"].clone()).unwrap(),
        );
        let mut requests = Vec::new();
        loop {
            step = match step {
                Ok(Step::Read(read)) => {
                    requests.push(read.path().to_owned());
                    assert!(requests.len() <= 4, "nonterminating {}", case["name"]);
                    let value = case["responses"].get(read.path()).unwrap_or_else(|| {
                        panic!("unexpected GET {}: {}", case["name"], read.path())
                    });
                    read.accept(response(value))
                }
                Ok(Step::Complete(value)) => {
                    assert_eq!(value, case["expected"], "{}", case["name"]);
                    break;
                }
                Err(error) => {
                    if let Some(status) = case["error_status"].as_u64() {
                        assert!(
                            matches!(error, Error::Http(ref failure) if u64::from(failure.status)==status),
                            "{}: {error}",
                            case["name"]
                        );
                    } else {
                        assert_eq!(
                            error.to_string(),
                            case["error"].as_str().unwrap(),
                            "{}",
                            case["name"]
                        );
                    }
                    break;
                }
            };
        }
        assert_eq!(
            requests,
            case["requests"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().to_owned())
                .collect::<Vec<_>>(),
            "{}",
            case["name"]
        );
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
    let fixtures = fixtures();
    for name in [
        "preparation-discovers-original-submission",
        "record-mixed-filters",
        "saved-action-retry",
        "http-read-409",
        "saved-no-attempt",
    ] {
        let case = fixtures["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap()
            .clone();
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
        let saved = case
            .get("saved")
            .filter(|value| !value.is_null())
            .map(|value| {
                let mut saved = value.clone();
                saved["server"] = json!(base);
                saved
            });
        let result = read::observe(
            &client,
            &case["health"],
            saved.as_ref(),
            selection(&case["selection"]),
            serde_json::from_value(case["view"].clone()).unwrap(),
        );
        if let Some(status) = case["error_status"].as_u64() {
            assert!(
                matches!(result, Err(Error::Http(ref error)) if u64::from(error.status)==status && error.error_code=="READ_FAILURE")
            );
        } else if let Some(expected) = case["error"].as_str() {
            assert_eq!(result.unwrap_err().to_string(), expected);
        } else {
            assert_eq!(result.unwrap(), case["expected"], "{name}");
        }
        server.join().unwrap();
    }
}
