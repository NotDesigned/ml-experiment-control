use ml_exp_client::{
    Error,
    observation::Selection,
    watch::{self, Decision, End, Watch},
};
use serde_json::{Value, json};
use std::{cell::Cell, time::Duration};

fn value(attempt: Option<&str>, scheduler: &str, publication: &str) -> Value {
    json!({"project":"p","run_id":"r","attempt_id":attempt,
        "scheduler":{"state":scheduler},"publication":{"status":publication}})
}
fn selection() -> Selection<'static> {
    Selection {
        project: Some("p"),
        run: Some("r"),
        attempt: None,
    }
}
fn waiting(decision: Decision) -> Watch {
    match decision {
        Decision::Wait { watch, delay, .. } => {
            assert!(delay > Duration::ZERO && delay <= Duration::from_secs(5));
            watch
        }
        Decision::Complete { .. } => panic!("unexpected completed observation"),
    }
}

#[test]
fn pure_polling_decisions_match_actual_python_watch_conditions() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/client-watch.json")).unwrap();
    assert_eq!(
        fixture["baseline_commit"],
        "e725faec689fb8395e692e27d00a8896fa7fe5a5"
    );
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 113);
    for case in fixture["cases"].as_array().unwrap() {
        let watch = Watch::new(
            Duration::from_secs_f64(case["duration"].as_f64().unwrap()),
            selection(),
        )
        .unwrap();
        let original = case["input"].clone();
        let decision = watch
            .accept(
                original.clone(),
                Duration::from_secs_f64(case["elapsed"].as_f64().unwrap()),
            )
            .unwrap();
        match decision {
            Decision::Wait { delay, value, .. } => {
                assert_eq!(case["decision"], "wait");
                assert_eq!(
                    delay,
                    Duration::from_secs_f64(case["delay"].as_f64().unwrap())
                );
                assert_eq!(value, original);
            }
            Decision::Complete { value, .. } => {
                assert_eq!(case["decision"], "complete");
                assert_eq!(value, original);
            }
        }
    }
}

#[test]
fn a_preparation_can_bind_once_but_never_switch_or_lose_its_attempt() {
    let watch = Watch::new(Duration::from_secs(60), Selection::default()).unwrap();
    let watch = waiting(
        watch
            .accept(value(None, "NOT_SUBMITTED", "NOT_OBSERVED"), Duration::ZERO)
            .unwrap(),
    );
    assert_eq!(watch.selection().project, Some("p"));
    assert_eq!(watch.selection().run, Some("r"));
    assert_eq!(watch.selection().attempt, None);
    let watch = waiting(
        watch
            .accept(
                value(Some("attempt-001"), "RUNNING", "EXECUTING"),
                Duration::from_secs(5),
            )
            .unwrap(),
    );
    assert_eq!(watch.selection().attempt, Some("attempt-001"));
    assert!(matches!(
        watch.accept(
            value(Some("attempt-002"), "RUNNING", "EXECUTING"),
            Duration::from_secs(10)
        ),
        Err(Error::Observation(
            "watch response differs from the pinned exact Run/Attempt"
        ))
    ));
    for next in [None, Some("attempt-002")] {
        let watch = Watch::new(
            Duration::from_secs(60),
            Selection {
                attempt: Some("attempt-001"),
                ..selection()
            },
        )
        .unwrap();
        assert!(
            watch
                .accept(value(next, "RUNNING", "EXECUTING"), Duration::ZERO)
                .is_err()
        );
    }
    for key in ["project", "run_id"] {
        let watch = Watch::new(Duration::from_secs(60), selection()).unwrap();
        let mut foreign = value(Some("attempt-001"), "RUNNING", "EXECUTING");
        foreign[key] = json!("foreign");
        assert!(watch.accept(foreign, Duration::ZERO).is_err());
    }
}

#[test]
fn elapsed_is_monotonic_and_timeout_never_becomes_failure_or_publication_success() {
    let watch = Watch::new(Duration::from_secs(60), selection()).unwrap();
    let watch = waiting(
        watch
            .accept(
                value(Some("attempt-001"), "SUCCEEDED", "EXECUTING"),
                Duration::from_secs(10),
            )
            .unwrap(),
    );
    assert!(matches!(
        watch.accept(
            value(Some("attempt-001"), "SUCCEEDED", "EXECUTING"),
            Duration::from_secs(9)
        ),
        Err(Error::Observation(
            "watch elapsed time must not move backwards"
        ))
    ));
    let watch = Watch::new(Duration::ZERO, selection()).unwrap();
    let initial = value(None, "NOT_SUBMITTED", "NOT_OBSERVED");
    let Decision::Complete { value, reason } =
        watch.accept(initial.clone(), Duration::ZERO).unwrap()
    else {
        panic!("zero observation window must finish after its initial read")
    };
    assert_eq!(reason, End::ObservationWindowEnded);
    assert_eq!(value, initial);
}

#[test]
fn injected_clock_stops_at_budget_and_callback_errors_do_not_start_more_reads() {
    let clock = Cell::new(Duration::ZERO);
    let reads = Cell::new(0);
    let updates = Cell::new(0);
    let result = watch::observe_with(
        Duration::from_secs(10),
        selection(),
        |selected| {
            if reads.get() > 0 {
                assert_eq!(selected.attempt, Some("attempt-001"));
            }
            reads.set(reads.get() + 1);
            Ok(value(Some("attempt-001"), "RUNNING", "EXECUTING"))
        },
        || clock.get(),
        |delay| {
            assert!(delay <= Duration::from_secs(5));
            clock.set(clock.get() + delay);
        },
        |_| {
            updates.set(updates.get() + 1);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(reads.get(), 2); // No extra GET at t=10, unlike the old loop.
    assert_eq!(updates.get(), 2);
    assert_eq!(clock.get(), Duration::from_secs(10));
    assert_eq!(result["scheduler"]["state"], "RUNNING");
    assert_eq!(result["publication"]["status"], "EXECUTING");

    let reads = Cell::new(0);
    let result = watch::observe_with(
        Duration::from_secs(10),
        selection(),
        |_| {
            reads.set(reads.get() + 1);
            Ok(value(Some("attempt-001"), "RUNNING", "EXECUTING"))
        },
        || Duration::ZERO,
        |_| panic!("callback error must prevent sleep"),
        |_| Err(Error::Observation("synthetic display failure")),
    );
    assert!(matches!(
        result,
        Err(Error::Observation("synthetic display failure"))
    ));
    assert_eq!(reads.get(), 1);

    let clock = Cell::new(Duration::ZERO);
    let reads = Cell::new(0);
    watch::observe_with(
        Duration::from_secs(10),
        selection(),
        |_| {
            reads.set(reads.get() + 1);
            Ok(value(Some("attempt-001"), "RUNNING", "EXECUTING"))
        },
        || clock.get(),
        |_| panic!("an expired callback cannot sleep"),
        |_| {
            clock.set(Duration::from_secs(10));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(reads.get(), 1);

    let clock = Cell::new(Duration::ZERO);
    let reads = Cell::new(0);
    let result = watch::observe_with(
        Duration::from_secs(20),
        selection(),
        |selected| {
            reads.set(reads.get() + 1);
            if reads.get() == 1 {
                Ok(value(Some("attempt-001"), "RUNNING", "EXECUTING"))
            } else {
                assert_eq!(selected.attempt, Some("attempt-001"));
                Err(Error::Transport)
            }
        },
        || clock.get(),
        |delay| clock.set(clock.get() + delay),
        |_| Ok(()),
    );
    assert!(matches!(result, Err(Error::Transport)));
    assert_eq!(reads.get(), 2); // A failed observation is never polled again.
}

#[test]
fn real_http_two_rounds_do_not_follow_a_changed_current_attempt() {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        time::Instant,
    };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        for round in 0..2 {
            for path in ["/api/runs/p/r", "/api/runs/p/r/attempts"] {
                let deadline = Instant::now() + Duration::from_secs(5);
                let (mut socket, _) = loop {
                    match listener.accept() {
                        Ok(value) => break value,
                        Err(error)
                            if error.kind() == std::io::ErrorKind::WouldBlock
                                && Instant::now() < deadline =>
                        {
                            std::thread::sleep(Duration::from_millis(5))
                        }
                        Err(error) => panic!("missing exact watch GET: {error}"),
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
                assert!(request.starts_with(&format!("GET /ml-expd{path} HTTP/1.1\r\n")));
                assert!(
                    request
                        .to_ascii_lowercase()
                        .contains("authorization: bearer synthetic-watch-token\r\n")
                );
                let body = if path.ends_with("/attempts") {
                    json!({"project":"p","run_id":"r","current_attempt_id":if round==0 {"attempt-001"} else {"attempt-002"},"attempts":[{"attempt_id":"attempt-001","identity":"r::attempt-001","state":if round==0 {"RUNNING"} else {"SUCCEEDED"}},{"attempt_id":"attempt-002","identity":"r::attempt-002","state":"RUNNING"}]})
                } else {
                    json!({"project":"p","run_id":"r","evidence":{}})
                };
                let body = serde_json::to_vec(&body).unwrap();
                write!(
                    socket,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                socket.write_all(&body).unwrap();
            }
        }
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            assert!(
                matches!(listener.accept(),Err(error) if error.kind()==std::io::ErrorKind::WouldBlock)
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    });
    let client = ml_exp_client::Client::new(&base, "synthetic-watch-token".into()).unwrap();
    let clock = Cell::new(Duration::ZERO);
    let result = watch::observe_with(
        Duration::from_secs(10),
        selection(),
        |selected| ml_exp_client::status::observe(&client, &json!({}), None, selected, false),
        || clock.get(),
        |delay| clock.set(clock.get() + delay),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(result["attempt_id"], "attempt-001");
    assert_eq!(result["scheduler"]["state"], "SUCCEEDED");
    assert_eq!(result["publication"]["status"], "NOT_OBSERVED");
    server.join().unwrap();
}
