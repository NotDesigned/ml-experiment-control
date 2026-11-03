//! Synthetic loopback and real CLI processes only: no daemon/provider state.
use ml_exp_client::{
    Client, Error,
    projects::{self, Action, Command as ProjectCommand},
};
use ml_exp_io::write_atomic;
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Command, Output},
    time::{Duration, Instant},
};
fn health() -> Value {
    json!({"min_client_protocol_version":2,"api_protocol_version":2,"capabilities":["project-lifecycle.v1"]})
}
fn record(project: &str, state: &str) -> Value {
    json!({"project":project,"project_file":"/approved/project.yaml","state":state,"source":"MANUAL","registered_at":"first","updated_at":"now","state_reason":"test"})
}
struct Request {
    head: String,
    body: Value,
}
fn accept(listener: &TcpListener) -> TcpStream {
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        match listener.accept() {
            Ok((stream, _)) => return stream,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(2))
            }
            Err(e) => panic!("missing expected client request: {e}"),
        }
    }
}
fn read(stream: &mut TcpStream) -> Request {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    while !bytes.ends_with(b"\r\n\r\n") {
        let mut b = [0];
        stream.read_exact(&mut b).unwrap();
        bytes.push(b[0]);
        assert!(bytes.len() <= 16 * 1024);
    }
    let head = String::from_utf8(bytes).unwrap().to_lowercase();
    let len = head
        .lines()
        .find_map(|l| {
            l.strip_prefix("content-length: ")
                .map(|s| s.parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    assert!(len <= projects::MAX_REQUEST_BYTES);
    let mut body = vec![0; len];
    stream.read_exact(&mut body).unwrap();
    Request {
        head,
        body: if body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&body).unwrap()
        },
    }
}
fn reply(stream: &mut TcpStream, status: &str, body: &Value) {
    let body = serde_json::to_vec(body).unwrap();
    write!(stream,"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
    stream.write_all(&body).unwrap();
}
fn no_repeat(listener: &TcpListener) {
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_millis(120);
    loop {
        match listener.accept() {
            Ok(_) => panic!("unexpected automatic replay"),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(2))
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return,
            Err(e) => panic!("listener: {e}"),
        }
    }
}
fn command(home: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ml-exp-rs"));
    cmd.env_clear().env("HOME", home);
    cmd
}
fn output_json(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    serde_json::from_slice(&output.stdout).unwrap()
}
fn authenticated(request: &Request, method: &str, path: &str) {
    assert!(
        request
            .head
            .starts_with(&format!("{method} /ml-expd{path} http/1.1\r\n")),
        "{}",
        request.head
    );
    assert_eq!(
        request
            .head
            .matches("authorization: bearer project-test-secret\r\n")
            .count(),
        1
    );
    assert_eq!(
        request
            .head
            .matches("x-ml-expd-client-protocol: 2\r\n")
            .count(),
        1
    );
}
#[test]
fn cli_all_project_commands_use_the_baseline_routes_and_exactly_one_mutation() {
    for (args, path, body, response) in [
        (
            vec!["project", "list"],
            "/api/project-lifecycle",
            Value::Null,
            json!({"projects":[record("demo","ACTIVE")],"events":[],"durability":{"revision":7,"journal_pending":false}}),
        ),
        (
            vec![
                "project",
                "register",
                "/approved/project.yaml",
                "--expected-revision",
                "7",
            ],
            "/api/project-lifecycle/register",
            json!({"source":{"kind":"daemon_path","manifest_path":"/approved/project.yaml"},"expected_revision":7}),
            json!({"project":record("demo","ACTIVE")}),
        ),
        (
            vec!["project", "pause", "demo", "--reason", "维护"],
            "/api/project-lifecycle/demo/pause",
            json!({"reason":"维护"}),
            json!({"project":record("demo","PAUSED"),"active":false}),
        ),
        (
            vec!["project", "resume", "demo"],
            "/api/project-lifecycle/demo/resume",
            json!({"reason":""}),
            json!({"project":record("demo","ACTIVE"),"active":true}),
        ),
        (
            vec!["project", "archive", "demo", "--reason", "完成"],
            "/api/project-lifecycle/demo/archive",
            json!({"reason":"完成"}),
            json!({"project":record("demo","ARCHIVED"),"active":false}),
        ),
        (
            vec!["project", "restore", "demo"],
            "/api/project-lifecycle/demo/restore",
            json!({"reason":""}),
            json!({"project":record("demo","PAUSED"),"active":false}),
        ),
        (
            vec!["project", "unregister", "demo", "--expected-revision", "9"],
            "/api/project-lifecycle/demo/unregister",
            json!({"reason":"","expected_revision":9}),
            json!({"unregistered":record("demo","ARCHIVED")}),
        ),
        (
            vec![
                "project",
                "unregister-all",
                "--reason",
                "explicit tracking removal",
                "--expected-revision",
                "10",
            ],
            "/api/project-lifecycle/unregister-all",
            json!({"reason":"explicit tracking removal","expected_revision":10}),
            json!({"unregistered":[record("demo","PAUSED")]}),
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let settings = temp.path().join("settings.json");
        write_atomic(
            &settings,
            &serde_json::to_vec(&json!({"url":base,"token":"project-test-secret"})).unwrap(),
            true,
        )
        .unwrap();
        let expected = response.clone();
        let thread = std::thread::spawn(move || {
            let mut socket = accept(&listener);
            authenticated(&read(&mut socket), "get", "/api/health");
            reply(&mut socket, "200 OK", &health());
            drop(socket);
            let mut socket = accept(&listener);
            let request = read(&mut socket);
            authenticated(&request, if body.is_null() { "get" } else { "post" }, path);
            assert_eq!(request.body, body);
            if !body.is_null() {
                assert_eq!(
                    request
                        .head
                        .matches("content-type: application/json\r\n")
                        .count(),
                    1
                );
            }
            reply(&mut socket, "200 OK", &response);
            drop(socket);
            no_repeat(&listener);
        });
        let actual = output_json(
            command(temp.path())
                .arg("--config")
                .arg(&settings)
                .args(args)
                .output()
                .unwrap(),
        );
        assert_eq!(actual, expected);
        assert!(!actual.to_string().contains("project-test-secret"));
        thread.join().unwrap();
    }
}
#[test]
fn invalid_parameters_are_rejected_before_config_or_token_loading_and_network() {
    let temp = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let token = temp.path().join("missing-token");
    let malformed_config = temp.path().join("malformed-config");
    write_atomic(&malformed_config, b"not-json", true).unwrap();
    ml_exp_io::create_private_dir(temp.path()).unwrap();
    // A missing credential is enough to distinguish pure preflight from token I/O;
    // malformed explicit configuration also proves validation precedes config I/O.
    let long = "长".repeat(4001);
    let badpath = "/".to_owned() + &"x".repeat(4096);
    for args in [
        vec!["project", "register", "relative/client-local.yaml"],
        vec!["project", "register", &badpath],
        vec!["project", "pause", "../outside"],
        vec!["project", "archive", "demo"],
        vec!["project", "archive", "demo", "--reason", "\u{1c}"],
        vec!["project", "pause", "demo", "--reason", &long],
    ] {
        let output = command(temp.path())
            .env("ML_EXPD_API_TOKEN_FILE", &token)
            .args(["--url", &base, "--config"])
            .arg(&malformed_config)
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(
            error.contains("invalid project operation")
                || error.contains("archiving a project requires"),
            "{error}"
        );
        assert!(!error.contains("cannot read"));
    }
    no_repeat(&listener);
}
#[test]
fn dropped_cli_mutation_response_is_uncertain_and_is_never_replayed() {
    let temp = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
    let thread = std::thread::spawn(move || {
        let mut socket = accept(&listener);
        read(&mut socket);
        reply(&mut socket, "200 OK", &health());
        drop(socket);
        let mut socket = accept(&listener);
        let request = read(&mut socket);
        authenticated(&request, "post", "/api/project-lifecycle/demo/pause");
        assert_eq!(request.body, json!({"reason":""}));
        drop(socket);
        no_repeat(&listener);
    });
    let output = command(temp.path())
        .env("ML_EXPD_API_URL", base)
        .env("ML_EXPD_API_TOKEN", "project-test-secret")
        .args(["project", "pause", "demo"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("outcome is not confirmed"));
    assert!(error.contains("query project list"));
    assert!(!error.contains("project-test-secret"));
    thread.join().unwrap();
}
#[test]
fn typed_library_posts_preserve_redirect_409_and_invalid_identity_boundaries() {
    for kind in ["redirect", "conflict", "foreign", "malformed", "oversized"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            let mut socket = accept(&listener);
            let request = read(&mut socket);
            authenticated(&request, "post", "/api/project-lifecycle/demo/pause");
            match kind{
    "redirect"=>socket.write_all(b"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://127.0.0.1:1/leak\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap(),
    "conflict"=>{socket.write_all(b"HTTP/1.1 409 Conflict\r\nX-ML-Expd-Error-Code: PROJECT_OPERATION_UNCERTAIN\r\nContent-Length: 39\r\nConnection: close\r\n\r\n{\"detail\":\"Bearer project-test-secret\"}").unwrap();},
    "foreign"=>reply(&mut socket,"200 OK",&json!({"project":record("foreign","PAUSED")})),
    "malformed"=>socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nnope").unwrap(),
    "oversized"=>{let size=8*1024*1024+1;write!(socket,"HTTP/1.1 200 OK\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n").unwrap();let _=socket.write_all(&vec![b' ';size]);},_=>unreachable!()
   };
            drop(socket);
            no_repeat(&listener);
        });
        let client = Client::new(&base, "project-test-secret".into()).unwrap();
        let plan = projects::plan(ProjectCommand::Change {
            project: "demo",
            action: Action::Pause,
            reason: "",
            expected_revision: None,
        })
        .unwrap();
        let error = projects::execute(&client, &plan, &health()).unwrap_err();
        match kind {
            "redirect" => assert!(matches!(&error,Error::Http(e)if e.status==307)),
            "conflict" => {
                assert!(matches!(&error,Error::Http(e)if e.status==409&&!e.retryable));
                assert!(!format!("{error:?}").contains("project-test-secret"));
            }
            _ => assert!(matches!(&error, Error::ProjectOutcomeUnknown(_))),
        };
        assert!(!error.to_string().contains("project-test-secret"));
        thread.join().unwrap();
    }
}
#[test]
fn protocol_or_capability_failure_never_sends_the_planned_post() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let client = Client::new(&base, "project-test-secret".into()).unwrap();
    let plan = projects::plan(ProjectCommand::UnregisterAll {
        reason: "tracking only",
        expected_revision: Some(0),
    })
    .unwrap();
    for health in [
        json!({}),
        json!({"min_client_protocol_version":2,"api_protocol_version":2}),
        json!({"min_client_protocol_version":2,"api_protocol_version":2,"capabilities":["training-submit.v1"]}),
    ] {
        assert!(projects::execute(&client, &plan, &health).is_err());
    }
    no_repeat(&listener);
}
#[test]
fn request_plans_are_bounded_and_explicit_without_io() {
    for reason in ["x".repeat(4000), "\0".repeat(4000), "🦀".repeat(4000)] {
        assert!(
            projects::plan(ProjectCommand::Change {
                project: "demo",
                action: Action::Pause,
                reason: &reason,
                expected_revision: Some(u64::MAX)
            })
            .is_ok()
        );
    }
    assert!(
        projects::plan(ProjectCommand::Change {
            project: "demo",
            action: Action::Pause,
            reason: &"x".repeat(4001),
            expected_revision: None
        })
        .is_err()
    );
    assert!(
        projects::plan(ProjectCommand::Register {
            manifest_path: "/nul\0path",
            expected_revision: None
        })
        .is_err()
    );
    let too_long = "/".to_owned() + &"x".repeat(4096);
    assert!(
        projects::plan(ProjectCommand::Register {
            manifest_path: &too_long,
            expected_revision: None
        })
        .is_err()
    );
}

#[test]
fn registration_response_must_bind_the_requested_canonical_daemon_path() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
    let thread = std::thread::spawn(move || {
        let mut socket = accept(&listener);
        let request = read(&mut socket);
        authenticated(&request, "post", "/api/project-lifecycle/register");
        assert_eq!(
            request.body,
            json!({"source":{"kind":"daemon_path","manifest_path":"/approved/expected.yaml"}})
        );
        reply(
            &mut socket,
            "200 OK",
            &json!({"project":record("demo","ACTIVE")}),
        );
        drop(socket);
        no_repeat(&listener);
    });
    let client = Client::new(&base, "project-test-secret".into()).unwrap();
    let plan = projects::plan(ProjectCommand::Register {
        manifest_path: "/approved/expected.yaml",
        expected_revision: None,
    })
    .unwrap();
    assert!(matches!(
        projects::execute(&client, &plan, &health()),
        Err(Error::ProjectOutcomeUnknown(_))
    ));
    thread.join().unwrap();
}
