use ml_exp_client::observation::{self, Selection, Step};
use serde_json::Value;

#[test]
fn saved_identity_observation_matches_legacy_reads_without_guessing_an_attempt() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../compatibility/client-observation.json"
    ))
    .unwrap();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let saved = case.get("saved").filter(|v| !v.is_null());
        if let Some(saved) = saved {
            match observation::reference(saved) {
                Ok(hints) => assert_eq!(
                    serde_json::to_value(hints).unwrap(),
                    case["hints"],
                    "hints {index}"
                ),
                Err(error) => assert_eq!(
                    error.to_string(),
                    case["hints_error"].as_str().unwrap(),
                    "hints error {index}"
                ),
            }
        }
        let mut requests = Vec::new();
        let selection = Selection {
            project: case["selection"]["project"].as_str(),
            run: case["selection"]["run"].as_str(),
            attempt: case["selection"]["attempt"].as_str(),
        };
        let mut result = observation::begin(case["base"].as_str().unwrap(), saved, selection);
        loop {
            match result {
                Ok(Step::Read(read)) => {
                    requests.push(Value::String(read.path().to_owned()));
                    result = match case["responses"].get(read.path()) {
                        Some(value) => read.accept(value.clone()),
                        None => Err(ml_exp_client::Error::Observation(
                            "unexpected fixture request",
                        )),
                    };
                }
                Ok(Step::Complete(reference)) => {
                    assert_eq!(
                        serde_json::to_value(reference).unwrap(),
                        case["expected"],
                        "complete case {index}"
                    );
                    break;
                }
                Err(error) => {
                    assert_eq!(
                        error.to_string(),
                        case["error"].as_str().unwrap(),
                        "error case {index}"
                    );
                    break;
                }
            }
        }
        assert_eq!(
            requests,
            *case["requests"].as_array().unwrap(),
            "requests case {index}"
        );
    }
}

#[test]
fn effect_shell_follows_only_the_original_preparation_and_submission_over_get() {
    use serde_json::json;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        time::Duration,
    };
    for foreign in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let preparation = format!("preparation-{}", "b".repeat(32));
        let submission = format!("action-{}", "a".repeat(16));
        let current = json!({"project":if foreign {"other"}else{"p"},"run_id":"r","preparation_id":preparation,
            "submission":{"project":"p","run_id":"r","attempt_id":"attempt-007","submission_id":submission}});
        let final_submission = current["submission"].clone();
        let expected_path =
            format!("GET /ml-expd/api/projects/p/experiment-preparations/{preparation} HTTP/1.1");
        let final_path = format!("GET /ml-expd/api/submissions/{submission} HTTP/1.1");
        listener.set_nonblocking(true).unwrap();
        let thread = std::thread::spawn(move || {
            let replies = if foreign {
                vec![(expected_path, current)]
            } else {
                vec![(expected_path, current), (final_path, final_submission)]
            };
            for (expected, reply) in replies {
                let until = std::time::Instant::now() + Duration::from_secs(5);
                let (mut stream, _) = loop {
                    match listener.accept() {
                        Ok(value) => break value,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(
                                std::time::Instant::now() < until,
                                "expected GET was not sent"
                            );
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(error) => panic!("listener failed: {error}"),
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut headers = Vec::new();
                while !headers.ends_with(b"\r\n\r\n") {
                    let mut byte = [0];
                    stream.read_exact(&mut byte).unwrap();
                    headers.extend_from_slice(&byte);
                    assert!(headers.len() < 8192);
                }
                let headers = String::from_utf8(headers).unwrap();
                assert!(headers.starts_with(&expected));
                assert!(
                    headers
                        .to_lowercase()
                        .contains("x-ml-expd-client-protocol: 2\r\n")
                );
                let body = serde_json::to_vec(&reply).unwrap();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(&body).unwrap();
            }
        });
        let saved = json!({"server":base,"project":"p","run_id":"r","config_sha256":"synthetic",
            "preparation":{"preparation_id":preparation}});
        let client = ml_exp_client::Client::new(&base, "native-test".into()).unwrap();
        let result = observation::resolve(&client, Some(&saved), Selection::default());
        if foreign {
            assert!(result.is_err());
        } else {
            let result = result.unwrap();
            assert_eq!(result.attempt_id.as_deref(), Some("attempt-007"));
            assert_eq!(result.submission.unwrap()["submission_id"], submission);
        }
        thread.join().unwrap();
    }
}
