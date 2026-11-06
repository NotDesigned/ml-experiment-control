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
    assert_eq!(
        result.stdout_sha256.as_deref(),
        Some("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
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

#[test]
fn a_log_flood_is_a_marker_and_each_log_keeps_both_ends() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("outputs");
    let result = execute(
        root.path(),
        &output,
        &shell(
            "printf first; head -c 9437184 /dev/zero; printf last; \
             printf first >&2; head -c 9437184 /dev/zero >&2; printf last >&2",
        ),
        Duration::from_secs(5),
    )
    .unwrap();
    assert_eq!(result.exit_code, Some(0));
    assert!(result.succeeded());
    assert!(result.error.is_none());
    let sent = 9437184 + "firstlast".len() as u64;
    for (name, hash, dropped) in [
        (
            "stdout.log",
            &result.stdout_sha256,
            result.stdout_dropped_bytes,
        ),
        (
            "stderr.log",
            &result.stderr_sha256,
            result.stderr_dropped_bytes,
        ),
    ] {
        let log = fs::read(output.join(name)).unwrap();
        assert!(log.len() <= 8 * 1024 * 1024);
        assert!(log.starts_with(b"first") && log.ends_with(b"last"));
        assert_eq!(
            log[4 * 1024 * 1024..][..ml_exp_io::process::SEAM.len()],
            *ml_exp_io::process::SEAM
        );
        assert_eq!(
            dropped,
            sent - (log.len() - ml_exp_io::process::SEAM.len()) as u64
        );
        assert_eq!(
            hash.as_deref(),
            Some(ml_exp_io::sha256_file(&output.join(name)).unwrap().as_str())
        );
    }
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("execution.json")).unwrap()).unwrap();
    assert_eq!(receipt["contract"], "native-local-execution.v1");
    assert!(receipt["error"].is_null());
    assert_eq!(receipt["stdout_dropped_bytes"], result.stdout_dropped_bytes);
    assert_eq!(receipt["stderr_dropped_bytes"], result.stderr_dropped_bytes);
    assert_eq!(receipt.as_object().unwrap().len(), 10);
}

#[test]
fn the_end_of_a_flooded_log_is_in_the_file_while_the_job_runs() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("outputs");
    let result = execute(
        root.path(),
        &output,
        &shell(
            "head -c 9437184 /dev/zero; printf LAST-LINE; i=0; \
             while [ \"$(tail -c 9 \"$OUTPUT_DIR/stdout.log\")\" != LAST-LINE ]; do \
               i=$((i+1)); [ $i -gt 200 ] && exit 9; sleep 0.01; \
             done",
        ),
        Duration::from_secs(5),
    )
    .unwrap();
    assert_eq!(result.exit_code, Some(0));
    assert!(result.succeeded());
    assert!(result.stdout_dropped_bytes > 0);
}

#[test]
fn direct_writes_and_truncation_cannot_supply_unbounded_or_incomplete_log_hashes() {
    let root = tempfile::tempdir().unwrap();
    for (name, script, size) in [
        (
            "oversized-stdout",
            "head -c 9437184 /dev/zero > \"$OUTPUT_DIR/stdout.log\"",
            9437184,
        ),
        (
            "changed-stdout",
            "printf changed > \"$OUTPUT_DIR/stdout.log\"",
            7,
        ),
        (
            "oversized-stderr",
            "head -c 9437184 /dev/zero > \"$OUTPUT_DIR/stderr.log\"",
            9437184,
        ),
        (
            "truncated",
            "printf captured; while ! test -s \"$OUTPUT_DIR/stdout.log\"; do sleep 0.002; done; : > \"$OUTPUT_DIR/stdout.log\"",
            0,
        ),
        (
            "same-length-overwrite",
            "printf hello; while ! test -s \"$OUTPUT_DIR/stdout.log\"; do sleep 0.002; done; printf xxxxx > \"$OUTPUT_DIR/stdout.log\"",
            5,
        ),
    ] {
        let output = root.path().join(name);
        let started = Instant::now();
        let result = execute(root.path(), &output, &shell(script), Duration::from_secs(2)).unwrap();
        assert_eq!(result.exit_code, Some(0));
        assert!(!result.succeeded());
        let error = result.error.as_ref().unwrap();
        assert_eq!(
            (error.stage, error.code),
            ("LOG_CAPTURE", "LOG_IDENTITY_OR_CONTENT_CHANGED")
        );
        let log = if name == "oversized-stderr" {
            "stderr.log"
        } else {
            "stdout.log"
        };
        assert_eq!(fs::metadata(output.join(log)).unwrap().len(), size);
        if log == "stdout.log" {
            assert!(result.stdout_sha256.is_none());
        } else {
            assert!(result.stderr_sha256.is_none());
        }
        assert!(started.elapsed() < Duration::from_secs(2));
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("execution.json")).unwrap()).unwrap();
        assert_eq!(receipt["error"]["code"], "LOG_IDENTITY_OR_CONTENT_CHANGED");
    }
}

#[test]
fn a_storage_failure_names_where_and_why() {
    let cause = std::io::Error::from_raw_os_error(13);
    let text = crate::managed::Error::storage("managed.rs:1", &cause).to_string();
    assert!(text.contains("managed.rs:1"), "{text}");
    assert!(text.contains("PermissionDenied"), "{text}");
    let long = "x".repeat(2000);
    let text = crate::managed::Error::storage("managed.rs:1", &long).to_string();
    assert!(text.len() < 400, "bounded: {}", text.len());
}
