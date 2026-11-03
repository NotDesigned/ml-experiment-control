use ml_exp_client::{
    Error,
    observation::Selection,
    status::{self, Step},
};
use ml_exp_core::diagnostic::HttpFailure;
use serde_json::Value;

#[test]
fn status_steps_match_python_and_preserve_exact_evidence_layers() {
    let x: Value =
        serde_json::from_str(include_str!("../../../compatibility/client-status.json")).unwrap();
    for c in x["cases"].as_array().unwrap() {
        let mut step = status::begin(
            c["base"].as_str().unwrap(),
            &c["health"],
            c.get("saved").filter(|v| !v.is_null()),
            Selection {
                project: c["selection"]["project"].as_str(),
                run: c["selection"]["run"].as_str(),
                attempt: c["selection"]["attempt"].as_str(),
            },
            c["details"].as_bool().unwrap(),
        );
        let mut requests = Vec::new();
        loop {
            step = match step {
                Ok(Step::Read(read)) => {
                    requests.push(read.path().to_owned());
                    assert!(requests.len() <= 12, "nonterminating {}", c["name"]);
                    let value = c["responses"].get(read.path()).unwrap_or_else(|| {
                        panic!("unexpected request {}: {}", c["name"], read.path())
                    });
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
                Ok(Step::Complete(value)) => {
                    assert_eq!(value, c["expected"], "{}", c["name"]);
                    break;
                }
                Err(error) => {
                    if let Some(status) = c["error_status"].as_u64() {
                        assert!(
                            matches!(error,Error::Http(ref e) if u64::from(e.status)==status),
                            "{}: {error}",
                            c["name"]
                        );
                    } else {
                        assert_eq!(
                            error.to_string(),
                            c["error"].as_str().unwrap(),
                            "{}",
                            c["name"]
                        );
                    }
                    break;
                }
            };
        }
        assert_eq!(
            requests,
            c["requests"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<Vec<_>>(),
            "{}",
            c["name"]
        );
    }
}

#[test]
fn real_http_status_keeps_queue_training_and_publication_independent() {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        time::{Duration, Instant},
    };
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/client-status.json")).unwrap();
    for name in ["current-exact-journal-fallback", "collection-http-409"] {
        let c = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap()
            .clone();
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
        assert_eq!(value, c["expected"], "{name}");
        server.join().unwrap();
    }
}

#[test]
fn duplicate_exact_attempts_are_refused_instead_of_choosing_the_first() {
    use serde_json::json;
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
fn native_missing_scheduler_evidence_never_borrows_process_completion() {
    use serde_json::json;
    for (layer_attempt, freshness) in [("attempt-001", "UNKNOWN"), ("attempt-002", "OBSERVED")] {
        let mut step = status::begin(
            "https://example.test",
            &json!({"capabilities":["result-collection-read.v1"]}),
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
