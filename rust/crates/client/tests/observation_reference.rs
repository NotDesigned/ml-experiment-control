use ml_exp_client::observation::{self, Selection, Step};
use serde_json::{Value, json};

const BASE: &str = "https://example.invalid/ml-expd";

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}
fn run_selection() -> Selection<'static> {
    Selection {
        project: Some("p"),
        run: Some("r"),
        attempt: None,
    }
}
/// Answers each planned GET from `responses`; any other request is a failure.
fn follow(
    saved: Option<&Value>,
    selected: Selection<'_>,
    responses: &Value,
) -> (Result<Value, String>, Vec<String>) {
    let mut requests = Vec::new();
    let mut result = observation::begin(BASE, saved, selected);
    loop {
        match result {
            Ok(Step::Read(read)) => {
                requests.push(read.path().to_owned());
                let reply = responses
                    .get(read.path())
                    .unwrap_or_else(|| panic!("unexpected GET {}", read.path()))
                    .clone();
                result = read.accept(reply);
            }
            Ok(Step::Complete(reference)) => {
                return (Ok(serde_json::to_value(reference).unwrap()), requests);
            }
            Err(error) => return (Err(error.to_string()), requests),
        }
    }
}

#[test]
fn a_selection_without_saved_state_needs_no_read() {
    let expected = literal(
        r#"{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","attempt_id":null,"state_kind":null,"saved":{},"submission":null,"action":null,"preparation":null}"#,
    );
    let (result, requests) = follow(None, run_selection(), &json!({}));
    assert_eq!(result.unwrap(), expected);
    assert!(requests.is_empty());
}

#[test]
fn saved_state_reads_only_its_own_submission_preparation_or_action() {
    for case in [
        literal(
            r#"{"saved":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","config_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}},"responses":{"/api/submissions/action-aaaaaaaaaaaaaaaa":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}},"requests":["/api/submissions/action-aaaaaaaaaaaaaaaa"],"hints":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","attempt_id":"attempt-001","state_kind":"experiment","saved":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","config_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}}},"expected":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","attempt_id":"attempt-001","state_kind":"experiment","saved":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","config_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}},"submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"},"action":null,"preparation":null}}"#,
        ),
        literal(
            r#"{"saved":{"config_sha256":"x","project":"p","run_id":"r","preparation":{"preparation_id":"preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","project":"p","run_id":"r"}},"responses":{"/api/projects/p/experiment-preparations/preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb":{"project":"p","run_id":"r","preparation_id":"preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}},"requests":["/api/projects/p/experiment-preparations/preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],"hints":{"server":null,"project":"p","run_id":"r","attempt_id":null,"state_kind":"experiment","saved":{"config_sha256":"x","project":"p","run_id":"r","preparation":{"preparation_id":"preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","project":"p","run_id":"r"}}},"expected":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","attempt_id":null,"state_kind":"experiment","saved":{"config_sha256":"x","project":"p","run_id":"r","preparation":{"preparation_id":"preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","project":"p","run_id":"r"}},"submission":null,"action":null,"preparation":{"project":"p","run_id":"r","preparation_id":"preparation-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}}"#,
        ),
        literal(
            r#"{"saved":{"contract":"training-operation.v1","server":"https://example.invalid/ml-expd","project":"p","run_id":"r","operation":"retry","source_attempt_id":"attempt-001","plan":{"action_id":"action-aaaaaaaaaaaaaaaa","operation":"RETRY_ATTEMPT","run_id":"r","attempt_id":"attempt-002","scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},"execution":{"status":"PREPARED"},"intent_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","gate_bundle_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},"request":{"new_attempt_id":"attempt-002"}},"responses":{"/api/actions/action-aaaaaaaaaaaaaaaa":{"action_id":"action-aaaaaaaaaaaaaaaa","operation":"RETRY_ATTEMPT","run_id":"r","attempt_id":"attempt-002","scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},"execution":{"status":"PREPARED"},"intent_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","gate_bundle_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}},"requests":["/api/actions/action-aaaaaaaaaaaaaaaa"],"hints":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","attempt_id":"attempt-002","state_kind":"training-operation","saved":{"contract":"training-operation.v1","server":"https://example.invalid/ml-expd","project":"p","run_id":"r","operation":"retry","source_attempt_id":"attempt-001","plan":{"action_id":"action-aaaaaaaaaaaaaaaa","operation":"RETRY_ATTEMPT","run_id":"r","attempt_id":"attempt-002","scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},"execution":{"status":"PREPARED"},"intent_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","gate_bundle_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},"request":{"new_attempt_id":"attempt-002"}}},"expected":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","attempt_id":"attempt-002","state_kind":"training-operation","saved":{"contract":"training-operation.v1","server":"https://example.invalid/ml-expd","project":"p","run_id":"r","operation":"retry","source_attempt_id":"attempt-001","plan":{"action_id":"action-aaaaaaaaaaaaaaaa","operation":"RETRY_ATTEMPT","run_id":"r","attempt_id":"attempt-002","scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},"execution":{"status":"PREPARED"},"intent_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","gate_bundle_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},"request":{"new_attempt_id":"attempt-002"}},"submission":null,"action":{"action_id":"action-aaaaaaaaaaaaaaaa","operation":"RETRY_ATTEMPT","run_id":"r","attempt_id":"attempt-002","scope":{"project":"p","scope_type":"attempt","object_id":"r::attempt-001"},"execution":{"status":"PREPARED"},"intent_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","gate_bundle_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},"preparation":null}}"#,
        ),
    ] {
        let hints = observation::reference(&case["saved"]).unwrap();
        assert_eq!(serde_json::to_value(hints).unwrap(), case["hints"]);
        let (result, requests) = follow(Some(&case["saved"]), run_selection(), &case["responses"]);
        assert_eq!(result.unwrap(), case["expected"]);
        assert_eq!(json!(requests), case["requests"]);
    }
}

#[test]
fn foreign_or_incomplete_saved_state_is_refused_without_guessing_an_attempt() {
    for case in [
        literal(
            r#"{"saved":{"server":"https://other.invalid","project":"p","run_id":"r","config_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}},"responses":{},"requests":[],"error":"observation state belongs to a different server or Run"}"#,
        ),
        literal(
            r#"{"saved":{"server":"https://example.invalid/ml-expd","project":null,"run_id":"r","config_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}},"responses":{},"requests":[],"error":"invalid saved server or Run identity"}"#,
        ),
        literal(
            r#"{"saved":{},"responses":{},"requests":[],"error":"observation requires an experiment state or training-operation.v1 plan"}"#,
        ),
        literal(
            r#"{"saved":{"contract":"training-operation.v1","operation":"retry"},"responses":{},"requests":[],"error":"training preparation reply is unknown; inspect its recovery URL and attach the original Action; no Attempt was guessed"}"#,
        ),
        literal(
            r#"{"saved":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","config_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}},"responses":{"/api/submissions/action-aaaaaaaaaaaaaaaa":{"project":"p","run_id":"r","attempt_id":"attempt-003","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}},"requests":["/api/submissions/action-aaaaaaaaaaaaaaaa"],"error":"Submission identity differs from the saved exact Attempt"}"#,
        ),
    ] {
        let (result, requests) = follow(Some(&case["saved"]), run_selection(), &case["responses"]);
        assert_eq!(result.unwrap_err(), case["error"].as_str().unwrap());
        assert_eq!(json!(requests), case["requests"]);
    }
    let case = literal(
        r#"{"saved":{"server":"https://example.invalid/ml-expd","project":"p","run_id":"r","config_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}},"responses":{},"requests":[],"error":"requested identity differs from the saved exact Run/Attempt"}"#,
    );
    let selected = Selection {
        attempt: Some("attempt-003"),
        ..run_selection()
    };
    let (result, requests) = follow(Some(&case["saved"]), selected, &case["responses"]);
    assert_eq!(result.unwrap_err(), case["error"].as_str().unwrap());
    assert!(requests.is_empty());
    let invalid = literal(
        r#"{"server":"https://example.invalid/ml-expd","project":null,"run_id":"r","config_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","submission":{"project":"p","run_id":"r","attempt_id":"attempt-001","submission_id":"action-aaaaaaaaaaaaaaaa","status":"PREPARED"}}"#,
    );
    assert_eq!(
        observation::reference(&invalid).err().unwrap().to_string(),
        "invalid saved server or Run identity"
    );
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
