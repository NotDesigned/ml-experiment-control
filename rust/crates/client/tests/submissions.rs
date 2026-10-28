//! Typed native submissions on a real loopback HTTP socket; no scheduler used.
use ml_exp_client::{
    Client, Error,
    submissions::{self, Command},
};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::TcpListener,
    time::{Duration, Instant},
};
const ID: &str = "action-0123456789abcdef";
fn health() -> Value {
    json!({"api_protocol_version":2,"min_client_protocol_version":2,"capabilities":["native-slurm-submissions.v1"],"scheduler_mutations":true})
}
fn response() -> Value {
    json!({"submission_id":ID,"project":"demo","run_id":"train","attempt_id":"attempt-001","operation":"SUBMIT_RUN","status":"PREPARED","confirmation":format!("EXECUTE {ID}"),"execution":{"status":"PREPARED","revision":0}})
}
#[test]
fn plans_reject_invalid_identity_budget_reason_and_confirmation_without_io() {
    for cap in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(
            submissions::plan(Command::Prepare {
                project: "demo",
                run: "train",
                max_gpu_hours: cap,
                reason: ""
            })
            .is_err()
        );
    }
    assert!(
        submissions::plan(Command::Prepare {
            project: "../foreign",
            run: "train",
            max_gpu_hours: 1.,
            reason: ""
        })
        .is_err()
    );
    assert!(
        submissions::plan(Command::Authorize {
            submission: ID,
            reason: "bad\nreason"
        })
        .is_err()
    );
    assert!(
        submissions::plan(Command::Execute {
            submission: ID,
            confirmation: "wrong"
        })
        .is_err()
    );
    assert!(
        submissions::plan(Command::Show {
            submission: "action-foreign"
        })
        .is_err()
    );
    let plan = submissions::plan(Command::Existing {
        project: "demo",
        run: "train",
    })
    .unwrap();
    assert!(!plan.is_mutation());
    assert!(submissions::response(&plan, response()).is_ok());
    for field in [
        "project",
        "run_id",
        "attempt_id",
        "submission_id",
        "operation",
        "confirmation",
        "status",
    ] {
        let mut bad = response();
        bad[field] = json!("foreign");
        assert!(submissions::response(&plan, bad).is_err(), "{field}");
    }
    let plan = submissions::plan(Command::Show { submission: ID }).unwrap();
    let mut bad = response();
    bad["submission_id"] = json!("action-1111111111111111");
    bad["confirmation"] = json!("EXECUTE action-1111111111111111");
    assert!(submissions::response(&plan, bad).is_err());
}
#[test]
fn one_post_is_sent_and_lost_or_foreign_response_never_replays_it() {
    for mode in ["success", "drop", "foreign"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            listener.set_nonblocking(true).unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut socket = loop {
                match listener.accept() {
                    Ok((s, _)) => break s,
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(e) => panic!("missing request {e}"),
                }
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut head = Vec::new();
            while !head.ends_with(b"\r\n\r\n") {
                let mut b = [0];
                socket.read_exact(&mut b).unwrap();
                head.push(b[0]);
                assert!(head.len() < 16384);
            }
            let head = String::from_utf8(head).unwrap().to_lowercase();
            assert!(head.starts_with("post /api/runs/demo/train/submissions http/1.1\r\n"));
            assert!(head.contains("authorization: bearer synthetic-token"));
            assert!(head.contains("x-ml-expd-client-protocol: 2"));
            let size: usize = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .parse()
                .unwrap();
            assert!(size < 65536);
            let mut body = vec![0; size];
            socket.read_exact(&mut body).unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["max_gpu_hours"], 1.);
            assert_eq!(body["reason"], "reviewed");
            if mode != "drop" {
                let mut result = response();
                if mode == "foreign" {
                    result["project"] = json!("foreign");
                }
                let body = serde_json::to_vec(&result).unwrap();
                write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
                socket.write_all(&body).unwrap();
            }
            drop(socket);
            let deadline = Instant::now() + Duration::from_millis(120);
            while Instant::now() < deadline {
                match listener.accept() {
                    Ok(_) => panic!("automatic replay"),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(e) => panic!("listener {e}"),
                }
            }
        });
        let client = Client::new(&format!("http://{address}"), "synthetic-token".into()).unwrap();
        let plan = submissions::plan(Command::Prepare {
            project: "demo",
            run: "train",
            max_gpu_hours: 1.,
            reason: "reviewed",
        })
        .unwrap();
        let result = submissions::execute(&client, &plan, &health());
        if mode == "success" {
            assert_eq!(result.unwrap()["submission_id"], ID);
        } else {
            assert!(matches!(result, Err(Error::SubmissionOutcomeUnknown(_))));
        }
        worker.join().unwrap();
    }
}
#[test]
fn disabled_or_missing_capability_prevents_mutation_before_connecting() {
    let client = Client::new("http://127.0.0.1:1", "synthetic-token".into()).unwrap();
    let plan = submissions::plan(Command::Execute {
        submission: ID,
        confirmation: "EXECUTE action-0123456789abcdef",
    })
    .unwrap();
    for field in ["capabilities", "scheduler_mutations"] {
        let mut h = health();
        h[field] = if field == "capabilities" {
            json!([])
        } else {
            json!(false)
        };
        assert!(matches!(
            submissions::execute(&client, &plan, &h),
            Err(Error::Observation(_))
        ));
    }
    let mut malformed = health();
    malformed["capabilities"] = Value::Null;
    assert!(matches!(
        submissions::execute(&client, &plan, &malformed),
        Err(Error::Protocol)
    ));
}
