#![cfg(target_os = "linux")]
use std::{
    fs,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn alive(pid: &str) -> bool {
    fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        stat.rsplit_once(')')
            .is_some_and(|(_, fields)| !fields.trim_start().starts_with('Z'))
    })
}

#[test]
fn cli_term_and_int_cancel_process_groups_and_publish_receipts() {
    for signal in [libc::SIGTERM, libc::SIGINT] {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("outputs");
        let mut worker = Worker(
            Command::new(env!("CARGO_BIN_EXE_ml-exp-worker-rs"))
                .arg("--workdir")
                .arg(root.path())
                .arg("--output")
                .arg(&output)
                .args([
                    "--timeout-seconds",
                    "60",
                    "--",
                    "sh",
                    "-c",
                    "sleep 60 & echo $! > \"$OUTPUT_DIR/child.pid\"; wait",
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let start = Instant::now();
        while !output.join("child.pid").is_file() && start.elapsed() < Duration::from_secs(5) {
            assert!(
                worker.0.try_wait().unwrap().is_none(),
                "worker exited before training started"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let pid = fs::read_to_string(output.join("child.pid")).unwrap();
        // SAFETY: this is the exact worker Child PID, which remains owned and
        // unreaped here; the test signals only its own process.
        assert_eq!(unsafe { libc::kill(worker.0.id() as i32, signal) }, 0);
        let start = Instant::now();
        let status = loop {
            if let Some(status) = worker.0.try_wait().unwrap() {
                break status;
            }
            assert!(
                start.elapsed() < Duration::from_secs(6),
                "worker ignored cancellation"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(!status.success());
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("execution.json")).unwrap()).unwrap();
        assert_eq!(receipt["cancelled"], true);
        assert_eq!(receipt["timed_out"], false);
        assert_eq!(receipt["termination_signal"], libc::SIGTERM);
        assert!(receipt["error"].is_null());
        for _ in 0..50 {
            if !alive(pid.trim()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !alive(pid.trim()),
            "training child survived worker cancellation"
        );
    }
}

fn caught(pid: u32, signal: i32) -> bool {
    fs::read_to_string(format!("/proc/{pid}/status")).is_ok_and(|status| {
        status
            .lines()
            .find_map(|line| line.strip_prefix("SigCgt:"))
            .and_then(|mask| u64::from_str_radix(mask.trim(), 16).ok())
            .is_some_and(|mask| mask & (1 << (signal - 1)) != 0)
    })
}

/// A Complete past its SSH limit must keep going, or a dataset that takes longer than
/// the limit would restart from nothing on every retry. TERM still stops it.
#[test]
fn the_data_helper_outlives_a_lost_ssh_session_and_still_stops_on_term() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let config = root.path().join("data.json");
    fs::write(
        &config,
        format!(
            r#"{{"root":"{}","target":{{"backend_id":"slurm","profile_sha256":"sha256:{}","storage_scope":"sha256:{}"}}}}"#,
            root.path().display(),
            "1".repeat(64),
            "2".repeat(64)
        ),
    )
    .unwrap();
    fs::set_permissions(&config, fs::Permissions::from_mode(0o600)).unwrap();
    // The open stdin holds the helper waiting for a request that does not come.
    let mut helper = Worker(
        Command::new(env!("CARGO_BIN_EXE_ml-exp-worker-rs"))
            .args(["data-helper", "--stdio", "--config"])
            .arg(&config)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let pid = helper.0.id();
    let start = Instant::now();
    while !caught(pid, libc::SIGHUP) {
        assert!(helper.0.try_wait().unwrap().is_none());
        assert!(start.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(10));
    }
    // SAFETY: this is the exact helper Child PID, which remains owned and unreaped
    // here; the test signals only its own process.
    assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGHUP) }, 0);
    std::thread::sleep(Duration::from_millis(300));
    assert!(helper.0.try_wait().unwrap().is_none());
    // SAFETY: as above.
    assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGTERM) }, 0);
    let start = Instant::now();
    let status = loop {
        if let Some(status) = helper.0.try_wait().unwrap() {
            break status;
        }
        assert!(start.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(status.code(), Some(1));
}
