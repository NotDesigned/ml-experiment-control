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
