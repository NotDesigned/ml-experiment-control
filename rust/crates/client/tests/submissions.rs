//! Typed native submissions on a real loopback HTTP socket; no scheduler used.
use ml_exp_client::{
    Client, Error,
    submissions::{self, Command},
};
use ml_exp_core::first_submission::native::creation::Request;
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
    for (mode, creates) in ["success", "drop", "foreign"]
        .into_iter()
        .flat_map(|mode| [(mode, false), (mode, true)])
    {
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
            assert!(head.starts_with(if creates {
                "post /api/projects/demo/runs http/1.1\r\n"
            } else {
                "post /api/runs/demo/train/submissions http/1.1\r\n"
            }));
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
            if creates {
                assert_eq!(body["run_id"], "train");
                assert_eq!(body["resources"]["gpus"], 1);
            } else {
                assert_eq!(body["max_gpu_hours"], 1.);
                assert_eq!(body["reason"], "reviewed");
            }
            if mode != "drop" {
                let mut result = if creates {
                    creation_receipt()
                } else {
                    response()
                };
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
        let request = creation_request();
        let plan = submissions::plan(if creates {
            Command::Create {
                project: "demo",
                request: &request,
            }
        } else {
            Command::Prepare {
                project: "demo",
                run: "train",
                max_gpu_hours: 1.,
                reason: "reviewed",
            }
        })
        .unwrap();
        let mut h = health();
        h["capabilities"] = json!(["native-run-creation.v1", "native-slurm-submissions.v1"]);
        let result = submissions::execute(&client, &plan, &h);
        if mode == "success" {
            let value = result.unwrap();
            if creates {
                assert_eq!(value["creation_state"], "CREATED");
            } else {
                assert_eq!(value["submission_id"], ID);
            }
        } else if creates {
            assert!(matches!(result, Err(Error::RunCreationOutcomeUnknown(_))));
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

#[test]
fn run_creation_is_a_separate_capability_and_refuses_foreign_receipts() {
    let request = creation_request();
    let plan = submissions::plan(Command::Create {
        project: "demo",
        request: &request,
    })
    .unwrap();
    let client = Client::new("http://127.0.0.1:1", "synthetic-token".into()).unwrap();
    assert!(matches!(
        submissions::execute(&client, &plan, &health()),
        Err(Error::Observation(_))
    ));
    let receipt = creation_receipt();
    assert!(submissions::response(&plan, receipt.clone()).is_ok());
    for field in [
        "project",
        "run_id",
        "runtime_id",
        "source_id",
        "attempt_id",
        "creation_state",
        "request_sha256",
        "run_sha256",
        "protocol_id",
    ] {
        let mut bad = receipt.clone();
        bad[field] = json!("foreign");
        assert!(submissions::response(&plan, bad).is_err(), "{field}");
    }
}

fn creation_request() -> ml_exp_core::first_submission::native::creation::Request {
    let request:Request=serde_json::from_value(json!({"run_id":"train","executor":"slurm-test","runtime_id":format!("runtime.{}","1".repeat(64)),
        "command":["python","train.py"],"resources":{"gpus":1,"memory_gb":4},"walltime_seconds":300,"seed":42})).unwrap();

    request
}

fn creation_receipt() -> Value {
    let request = creation_request();
    let hash = format!("sha256:{}", "1".repeat(64));
    let receipt = json!({"project":"demo","run_id":"train","runtime_id":request.runtime_id,"source_id":format!("source.{}","2".repeat(64)),
        "attempt_id":"attempt-001","creation_state":"CREATED","request_sha256":request.identity("demo").unwrap(),
        "scheduler_mutation_performed":false,"run_sha256":hash,"attempt_sha256":hash,"protocol_id":hash,"reused":false});

    receipt
}

#[test]
fn worker_refresh_is_explicit_bounded_and_never_requires_scheduler_mutation_permission() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let plan = submissions::plan(Command::Refresh { submission: ID }).unwrap();
    let client = Client::new(
        &format!("http://{address}"),
        "synthetic-bearer-not-production".into(),
    )
    .unwrap();
    assert!(matches!(
        submissions::execute(&client, &plan, &health()),
        Err(Error::Observation(_))
    ));
    let worker = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut head = Vec::new();
        while !head.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            head.push(byte[0]);
            assert!(head.len() < 16384);
        }
        assert!(
            String::from_utf8(head)
                .unwrap()
                .starts_with(&format!("POST /api/submissions/{ID}/refresh HTTP/1.1"))
        );
        // A dropped observation reply does not make scheduler submission uncertain.
    });
    let mut allowed = health();
    allowed["capabilities"] = json!(["native-worker-observation.v1"]);
    allowed["scheduler_mutations"] = json!(false);
    assert!(matches!(
        submissions::execute(&client, &plan, &allowed),
        Err(Error::Transport)
    ));
    worker.join().unwrap();
}

#[test]
fn checkpoint_queries_and_registration_replies_preserve_exact_source_identity() {
    use ml_exp_core::checkpoints::{self, Context};
    let context = Context {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        source_id: format!("source.{}", "1".repeat(64)),
        image_id: format!("sha256:{}", "2".repeat(64)),
        storage_scope: "shared".into(),
        state_root: "/shared/demo/train/outputs".into(),
    };
    let doc=checkpoints::register(&checkpoints::document(&context,&json!({"step":4,"files":[{"path":"checkpoints/step-4/state.bin","bytes":7,"sha256":"3".repeat(64)}]})).unwrap(),"2026-10-29T00:00:00Z").unwrap();
    let id = doc["checkpoint_id"].as_str().unwrap();
    let list = submissions::plan(Command::Checkpoints {
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        checkpoint: None,
    })
    .unwrap();
    let value =
        json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","checkpoints":[doc]});
    assert!(!list.is_mutation());
    assert_eq!(submissions::response(&list, value.clone()).unwrap(), value);
    let exact = submissions::plan(Command::Checkpoints {
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        checkpoint: Some(id),
    })
    .unwrap();
    assert_eq!(submissions::response(&exact, doc.clone()).unwrap(), doc);
    let mut wrong = value.clone();
    wrong["checkpoints"][0]["attempt_id"] = json!("attempt-002");
    assert!(submissions::response(&list, wrong).is_err());
    let refresh = submissions::plan(Command::Checkpoint { submission: ID }).unwrap();
    let mut reply =
        json!({"submission_id":ID,"checkpoint":doc,"scheduler_mutation_performed":false});
    assert!(submissions::response(&refresh, reply.clone()).is_ok());
    reply["submission_id"] = json!("action-1111111111111111");
    assert!(submissions::response(&refresh, reply).is_err());
    assert!(
        submissions::plan(Command::Checkpoints {
            project: "demo",
            run: "../foreign",
            attempt: "attempt-001",
            checkpoint: None
        })
        .is_err()
    );
}

#[test]
fn gate_replacement_requires_exact_original_and_request_identity() {
    let request = "a".repeat(32);
    let confirmation = format!("RENEW {ID}");
    let plan = submissions::plan(Command::Renew {
        submission: ID,
        request_id: &request,
        confirmation: &confirmation,
    })
    .unwrap();
    let mut value = response();
    let replacement = "action-1123456789abcdef";
    value["submission_id"] = json!(replacement);
    value["confirmation"] = json!(format!("EXECUTE {replacement}"));
    value["renewed_from"] = json!(ID);
    value["renewal_request_id"] = json!(request);
    value["scheduler_mutation_performed"] = json!(false);
    assert!(submissions::response(&plan, value.clone()).is_ok());
    for key in [
        "renewed_from",
        "renewal_request_id",
        "scheduler_mutation_performed",
    ] {
        let mut bad = value.clone();
        bad[key] = json!("foreign");
        assert!(submissions::response(&plan, bad).is_err());
    }
    assert!(
        submissions::plan(Command::Renew {
            submission: ID,
            request_id: &request,
            confirmation: "wrong"
        })
        .is_err()
    );
}
