//! Real CLI process tests: no global environment mutation or external service.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Output};
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
                b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
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
    assert_eq!(displayed, serde_json::json!({"ok":true}));
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
                b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
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
    assert_eq!(displayed, serde_json::json!({"ok":true}));
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
