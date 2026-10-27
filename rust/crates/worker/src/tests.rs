use super::*;
use std::{fs, os::unix::fs::PermissionsExt, sync::atomic::Ordering, time::Instant};

fn shell(script: &str) -> Vec<String> {
    vec!["sh".into(), "-c".into(), script.into()]
}

#[cfg(target_os = "linux")]
fn not_running(pid: &str) -> bool {
    let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return true;
    };
    // Orphan zombies can await the container's PID 1; they cannot write logs or
    // consume execution resources. Do not mistake them for live descendants.
    stat.rsplit_once(')')
        .is_some_and(|(_, fields)| fields.trim_start().starts_with('Z'))
}

#[test]
fn captures_failure_and_does_not_overwrite_prior_outputs() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("outputs");
    let argv = shell("printf hello; printf failure >&2; exit 7");
    let result = execute(root.path(), &output, &argv, Duration::from_secs(5)).unwrap();
    assert_eq!(result.exit_code, Some(7));
    assert!(!result.timed_out);
    assert!(!result.succeeded());
    assert!(result.error.is_none());
    assert_eq!(
        fs::read_to_string(output.join("stdout.log")).unwrap(),
        "hello"
    );
    let original = fs::read(output.join("execution.json")).unwrap();
    assert!(execute(root.path(), &output, &argv, Duration::from_secs(5)).is_err());
    assert_eq!(fs::read(output.join("execution.json")).unwrap(), original);
}

#[test]
fn spawn_failures_publish_safe_stage_and_kind_without_command_secrets() {
    let root = tempfile::tempdir().unwrap();
    let blocked = root.path().join("not-executable");
    fs::write(&blocked, "#!/bin/sh\nexit 0\n").unwrap();
    fs::set_permissions(&blocked, fs::Permissions::from_mode(0o600)).unwrap();
    for (name, argv, expected) in [
        (
            "missing",
            vec![
                root.path().join("missing-command").display().to_string(),
                "secret-value".into(),
            ],
            "NOT_FOUND",
        ),
        (
            "blocked",
            vec![blocked.display().to_string()],
            "PERMISSION_DENIED",
        ),
    ] {
        let output = root.path().join(name);
        let result = execute(root.path(), &output, &argv, Duration::from_secs(2)).unwrap();
        assert!(!result.succeeded());
        let error = result.error.unwrap();
        assert_eq!(error.stage, "SPAWN");
        assert_eq!(error.code, expected);
        assert!(result.exit_code.is_none());
        let receipt = fs::read_to_string(output.join("execution.json")).unwrap();
        assert!(!receipt.contains("secret-value"));
        assert!(!receipt.contains(root.path().to_str().unwrap()));
    }
}

#[test]
fn substituted_fifo_and_symlink_logs_fail_without_opening_them() {
    let root = tempfile::tempdir().unwrap();
    for (name, command) in [
        (
            "fifo",
            "rm \"$OUTPUT_DIR/stdout.log\"; mkfifo \"$OUTPUT_DIR/stdout.log\"",
        ),
        (
            "link",
            "rm \"$OUTPUT_DIR/stdout.log\"; ln -s /dev/zero \"$OUTPUT_DIR/stdout.log\"",
        ),
    ] {
        let output = root.path().join(name);
        let start = Instant::now();
        let result = execute(
            root.path(),
            &output,
            &shell(command),
            Duration::from_secs(1),
        )
        .unwrap();
        assert!(start.elapsed() < Duration::from_secs(3));
        assert!(!result.succeeded());
        let error = result.error.unwrap();
        assert_eq!(error.stage, "LOG_CAPTURE");
        assert_eq!(error.code, "LOG_IDENTITY_OR_CONTENT_CHANGED");
        assert!(result.stdout_sha256.is_none());
        assert!(output.join("execution.json").is_file());
    }
}

#[cfg(target_os = "linux")]
#[test]
fn normal_leader_exit_stops_background_writers_before_hashing() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("outputs");
    let result = execute(
        root.path(),
        &output,
        &shell("(while :; do printf background; sleep 0.02; done) & echo $! > \"$OUTPUT_DIR/child.pid\"; printf parent; exit 0"),
        Duration::from_secs(5),
    ).unwrap();
    assert!(result.succeeded());
    let pid = fs::read_to_string(output.join("child.pid")).unwrap();
    for _ in 0..50 {
        if not_running(pid.trim()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        not_running(pid.trim()),
        "background child survived successful execution"
    );
    let bytes = fs::read(output.join("stdout.log")).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(fs::read(output.join("stdout.log")).unwrap(), bytes);
    assert_eq!(
        result.stdout_sha256.unwrap(),
        ml_exp_io::sha256_file(&output.join("stdout.log")).unwrap()
    );
}

#[cfg(target_os = "linux")]
#[test]
fn timeout_kills_term_ignoring_leader_and_descendants() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("outputs");
    let start = Instant::now();
    let result = execute(
        root.path(),
        &output,
        &shell("trap '' TERM; (trap '' TERM; while :; do sleep 1; done) & echo $! > \"$OUTPUT_DIR/child.pid\"; wait"),
        Duration::from_millis(100),
    ).unwrap();
    assert!(result.timed_out);
    assert_eq!(result.termination_signal, Some(libc::SIGKILL));
    assert!(!result.cancelled);
    assert!(start.elapsed() < Duration::from_secs(5));
    let pid = fs::read_to_string(output.join("child.pid")).unwrap();
    for _ in 0..50 {
        if not_running(pid.trim()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(not_running(pid.trim()), "background child survived timeout");
}

#[test]
fn pre_cancelled_execution_creates_receipt_without_starting_command() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("outputs");
    let cancelled = AtomicBool::new(false);
    cancelled.store(true, Ordering::Relaxed);
    let result = execute_with_cancel(
        root.path(),
        &output,
        &shell("touch unexpected"),
        Duration::from_secs(5),
        &cancelled,
    )
    .unwrap();
    assert!(result.cancelled);
    assert!(!result.timed_out);
    assert!(!result.succeeded());
    assert!(result.exit_code.is_none());
    assert!(!root.path().join("unexpected").exists());
    assert!(output.join("execution.json").is_file());
}
