//! Real CLI process tests: no global environment mutation or external service.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use ml_exp_io::{create_private_dir, write_atomic};

fn command(home: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ml-exp-rs"));
    command.env_clear().env("HOME", home);
    command
}

fn json_output(output: Output) -> serde_json::Value {
    assert!(
        output.status.success(),
        "client failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn native_contract_validation_does_not_load_config_or_contact_http() {
    let temp = tempfile::tempdir().unwrap();
    let mut child = command(temp.path())
        .args(["contract", "intent-validate"])
        .env("ML_EXPD_CLIENT_CONFIG", temp.path().join("missing.json"))
        .env("ML_EXPD_API_TOKEN", "must-not-be-read")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"kind":"SUBMIT_RUN","title":"native","draft":"{}"}"#)
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["kind"], "SUBMIT_RUN");
    assert_eq!(value["resource_estimate"], "unknown");
    assert!(output.stderr.is_empty());
}

#[test]
fn default_preview_config_is_private_and_does_not_replace_existing_client_config() {
    let temp = tempfile::tempdir().unwrap();
    let legacy = temp.path().join(".config/ml-expd/client.json");
    create_private_dir(legacy.parent().unwrap()).unwrap();
    let original = br#"{"url":"https://legacy.invalid","token":"legacy-test-secret"}"#;
    write_atomic(&legacy, original, true).unwrap();

    let displayed = json_output(
        command(temp.path())
            .args(["config", "show"])
            .output()
            .unwrap(),
    );
    assert_eq!(displayed["url"], serde_json::Value::Null);
    assert_eq!(displayed["token_configured"], false);
    let token = temp.path().join("token");
    write_atomic(&token, b"preview-test-secret\n", true).unwrap();
    let displayed = json_output(
        command(temp.path())
            .args([
                "--url",
                "https://preview.invalid/ml-expd/",
                "config",
                "set",
                "--token-file",
            ])
            .arg(&token)
            .output()
            .unwrap(),
    );
    assert_eq!(displayed["url"], "https://preview.invalid/ml-expd");
    assert_eq!(displayed["token_configured"], true);
    assert!(!displayed.to_string().contains("preview-test-secret"));
    assert!(
        temp.path()
            .join(".config/ml-expd-rust/client.json")
            .is_file()
    );
    assert_eq!(std::fs::read(&legacy).unwrap(), original);

    let explicit = json_output(
        command(temp.path())
            .arg("--config")
            .arg(&legacy)
            .args(["config", "show"])
            .output()
            .unwrap(),
    );
    assert_eq!(explicit["url"], "https://legacy.invalid");
    assert!(!explicit.to_string().contains("legacy-test-secret"));
    assert_eq!(std::fs::read(&legacy).unwrap(), original);
}

#[test]
fn environment_url_and_token_candidates_drive_the_real_read_only_request() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("settings.json");
    write_atomic(
        &config,
        br#"{"url":"https://saved.invalid","token":"saved-test-secret"}"#,
        true,
    )
    .unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
            assert!(request.len() < 8192);
        }
        let request = String::from_utf8(request).unwrap().to_lowercase();
        assert!(request.starts_with("get /ml-expd/api/health http/1.1"));
        assert_eq!(
            request
                .matches("authorization: bearer environment-test-secret\r\n")
                .count(),
            1
        );
        assert!(!request.contains("saved-test-secret"));
        assert_eq!(
            request.matches("x-ml-expd-client-protocol: 2\r\n").count(),
            1
        );
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 68\r\nConnection: close\r\n\r\n{\"ok\":true,\"min_client_protocol_version\":2,\"api_protocol_version\":2}",
            )
            .unwrap();
    });
    let displayed = json_output(
        command(temp.path())
            .env("ML_EXPD_API_URL", &base)
            .env("ML_EXPD_API_TOKEN", "environment-test-secret")
            .env("ML_EXPD_CLIENT_CONFIG", &config)
            .arg("check")
            .output()
            .unwrap(),
    );
    assert_eq!(
        displayed,
        serde_json::json!({"ok":true,"min_client_protocol_version":2,"api_protocol_version":2})
    );
    server.join().unwrap();

    let conflict = command(temp.path())
        .env("ML_EXPD_API_URL", &base)
        .env("ML_EXPD_CLIENT_CONFIG", &config)
        .arg("check")
        .output()
        .unwrap();
    assert!(!conflict.status.success());
    assert!(conflict.stdout.is_empty());
    assert!(String::from_utf8_lossy(&conflict.stderr).contains("different API URL"));
    assert!(!String::from_utf8_lossy(&conflict.stderr).contains("saved-test-secret"));
}

#[test]
fn environment_token_file_does_not_use_an_ambient_system_proxy() {
    let temp = tempfile::tempdir().unwrap();
    let token_file = temp.path().join("environment-token");
    write_atomic(&token_file, b"environment-file-test-secret\n", true).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
            assert!(request.len() < 8192);
        }
        let request = String::from_utf8(request).unwrap().to_lowercase();
        assert!(request.starts_with("get /api/health http/1.1"));
        assert_eq!(
            request
                .matches("authorization: bearer environment-file-test-secret\r\n")
                .count(),
            1
        );
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 68\r\nConnection: close\r\n\r\n{\"ok\":true,\"min_client_protocol_version\":2,\"api_protocol_version\":2}",
            )
            .unwrap();
    });
    let displayed = json_output(
        command(temp.path())
            .env("ML_EXPD_API_URL", &base)
            .env("ML_EXPD_API_TOKEN_FILE", &token_file)
            .env("HTTP_PROXY", "http://127.0.0.1:1")
            .env("ALL_PROXY", "http://127.0.0.1:1")
            .env("NO_PROXY", "")
            .arg("check")
            .output()
            .unwrap(),
    );
    assert_eq!(
        displayed,
        serde_json::json!({"ok":true,"min_client_protocol_version":2,"api_protocol_version":2})
    );
    server.join().unwrap();
}

#[test]
fn environment_token_file_and_xdg_config_paths_remain_supported() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let token = temp.path().join("private-token");
    write_atomic(&token, b"file-test-secret", true).unwrap();
    let output = command(temp.path())
        .env("XDG_CONFIG_HOME", &xdg)
        .args([
            "--url",
            "https://preview.invalid",
            "config",
            "set",
            "--token-file",
        ])
        .arg(&token)
        .output()
        .unwrap();
    assert_eq!(json_output(output)["token_configured"], true);
    assert!(xdg.join("ml-expd-rust/client.json").is_file());
    assert!(
        !temp
            .path()
            .join(".config/ml-expd-rust/client.json")
            .exists()
    );

    let ambiguous = command(temp.path())
        .env("ML_EXPD_API_URL", "https://preview.invalid")
        .env("ML_EXPD_API_TOKEN", "inline-test-secret")
        .env("ML_EXPD_API_TOKEN_FILE", &token)
        .arg("check")
        .output()
        .unwrap();
    assert!(!ambiguous.status.success());
    assert!(ambiguous.stdout.is_empty());
    assert_eq!(
        String::from_utf8(ambiguous.stderr).unwrap().trim(),
        "invalid API token"
    );
}

#[test]
fn state_preflight_rejects_misdirected_credentials_before_the_handshake() {
    use serde_json::json;
    let temp = tempfile::tempdir().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let state = temp.path().join("saved.json");
    let saved = json!({"server":base,"project":"p","run_id":"r","config_sha256":"frozen"});
    write_atomic(&state, &serde_json::to_vec(&saved).unwrap(), true).unwrap();
    for (url, project) in [
        (None, None),
        (Some("https://foreign.invalid"), None),
        (Some(base.as_str()), Some("foreign")),
    ] {
        let mut cmd = command(temp.path());
        cmd.env("ML_EXPD_API_TOKEN", "private-preflight-test-secret");
        if let Some(url) = url {
            cmd.args(["--url", url]);
        }
        cmd.args(["run", "status", "--state"]).arg(&state);
        if let Some(project) = project {
            cmd.args(["--project", project]);
        }
        let output = cmd.output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(!stderr.contains("private-preflight-test-secret"));
        assert!(!stderr.contains("frozen"));
        assert!(matches!(listener.accept(), Err(e) if e.kind()==std::io::ErrorKind::WouldBlock));
    }
    let invalid = temp.path().join("invalid.json");
    write_atomic(
        &invalid,
        b"{\"server\":\"https://foreign.invalid\",\"contract\":\"training-operation.v1\"}",
        true,
    )
    .unwrap();
    let output = command(temp.path())
        .args(["--url", &base, "run", "status", "--state"])
        .arg(invalid)
        .env("ML_EXPD_API_TOKEN", "private-preflight-test-secret")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(matches!(listener.accept(), Err(e) if e.kind()==std::io::ErrorKind::WouldBlock));
}

#[test]
fn real_status_cli_preserves_json_evidence_and_has_a_human_summary() {
    use serde_json::{Value, json};
    use std::net::TcpListener;
    let fixtures: Value =
        serde_json::from_str(include_str!("../../../compatibility/client-status.json")).unwrap();
    let fixture = fixtures["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "current-exact-journal-fallback")
        .unwrap();
    for (human, watch) in [(false, false), (true, false), (false, true)] {
        let temp = tempfile::tempdir().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let mut responses = vec![("/api/health".to_string(), {
            let mut health = fixture["health"].clone();
            health["api_protocol_version"] = json!(2);
            health["min_client_protocol_version"] = json!(2);
            health
        })];
        responses.extend(fixture["requests"].as_array().unwrap().iter().map(|p| {
            let path = p.as_str().unwrap();
            (path.to_owned(), fixture["responses"][path].clone())
        }));
        let server = serve_responses(listener, responses, "cli-status-test-secret");
        let mut cmd = command(temp.path());
        cmd.env("ML_EXPD_API_TOKEN", "cli-status-test-secret")
            .args([
                "--url",
                &base,
                "run",
                "status",
                "--project",
                "p",
                "--run",
                "r",
            ]);
        if !human {
            cmd.arg("--json");
        }
        if watch {
            cmd.args(["--watch", "--seconds", "0"]);
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("cli-status-test-secret"));
        if human {
            let summary = String::from_utf8(output.stdout).unwrap();
            assert!(summary.contains("RUNNING"));
            assert!(!summary.starts_with('{'));
        } else {
            let value: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(value, fixture["expected"]);
        }
        server.join().unwrap();
    }
}

#[test]
fn real_logs_and_metrics_commands_follow_the_original_saved_attempt() {
    use serde_json::{Value, json};
    use std::net::TcpListener;
    let fixtures: Value =
        serde_json::from_str(include_str!("../../../compatibility/client-read.json")).unwrap();
    for (name, view) in [
        ("saved-exact-attempt", "logs"),
        ("saved-action-retry", "metrics"),
    ] {
        let fixture = fixtures["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let state = temp.path().join("saved.json");
        let mut saved = fixture["saved"].clone();
        saved["server"] = json!(&base);
        write_atomic(&state, &serde_json::to_vec(&saved).unwrap(), true).unwrap();
        let mut health = fixture["health"].clone();
        health["api_protocol_version"] = json!(2);
        health["min_client_protocol_version"] = json!(2);
        let mut responses = vec![("/api/health".to_string(), health)];
        responses.extend(fixture["requests"].as_array().unwrap().iter().map(|p| {
            let path = p.as_str().unwrap();
            (path.to_owned(), fixture["responses"][path].clone())
        }));
        let server = serve_responses(listener, responses, "cli-read-test-secret");
        let result = json_output(
            command(temp.path())
                .env("ML_EXPD_API_TOKEN", "cli-read-test-secret")
                .args(["--url", &base, "run", view, "--state"])
                .arg(&state)
                .output()
                .unwrap(),
        );
        assert_eq!(result, fixture["expected"]);
        server.join().unwrap();
    }
}

fn serve_responses(
    listener: std::net::TcpListener,
    responses: Vec<(String, serde_json::Value)>,
    token: &'static str,
) -> std::thread::JoinHandle<()> {
    use std::time::Instant;
    std::thread::spawn(move || {
        for (path, body) in responses {
            let deadline = Instant::now() + Duration::from_secs(10);
            let (mut socket, _) = loop {
                match listener.accept() {
                    Ok(value) => break value,
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(5))
                    }
                    Err(e) => panic!("missing CLI request {path}: {e}"),
                }
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            while !bytes.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                socket.read_exact(&mut byte).unwrap();
                bytes.push(byte[0]);
                assert!(bytes.len() < 8192);
            }
            let request = String::from_utf8(bytes).unwrap();
            assert!(
                request.starts_with(&format!("GET /ml-expd{path} HTTP/1.1")),
                "{request}"
            );
            assert!(
                request
                    .to_ascii_lowercase()
                    .contains(&format!("authorization: bearer {token}\r\n"))
            );
            let body = serde_json::to_vec(&body).unwrap();
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            socket.write_all(&body).unwrap();
        }
    })
}

#[test]
fn invalid_watch_duration_is_rejected_before_connection_or_configuration_io() {
    let temp = tempfile::tempdir().unwrap();
    for seconds in ["NaN", "inf", "-1", "1e300"] {
        let output = command(temp.path())
            .args(["run", "status", "--watch"])
            .arg(format!("--seconds={seconds}"))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap().trim(),
            "status --seconds must be finite and nonnegative"
        );
    }
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[test]
fn allocation_contract_preserves_counts_and_rejects_incomplete_evidence() {
    let temp = tempfile::tempdir().unwrap();
    let node = "NodeName=n1 Partitions=gpu CfgTRES=cpu=16,gres/gpu=8 AllocTRES=cpu=4,gres/gpu=3 CPUTot=16 CPUAlloc=4 RealMemory=1024 AllocMem=256 State=MIXED";
    for valid in [true, false] {
        let evidence = if valid {
            format!("{node}\nML_EXPD_QUEUE\n123\n")
        } else {
            node.to_owned()
        };
        let input = serde_json::json!({"output": evidence, "partition": "gpu"});
        let mut child = command(temp.path())
            .args(["contract", "slurm-allocation-validate"])
            .env("ML_EXPD_CLIENT_CONFIG", temp.path().join("missing.json"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.to_string().as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert_eq!(output.status.success(), valid);
        if valid {
            let snapshot = json_output(output);
            assert_eq!(snapshot["resources"]["gpus"]["unallocated"], 5);
            assert_eq!(snapshot["resources"]["memory_bytes"]["allocated"], 268435456);
            assert_eq!(snapshot["pending_queue_entries"], 1);
        } else {
            assert!(output.stdout.is_empty());
        }
    }
}
