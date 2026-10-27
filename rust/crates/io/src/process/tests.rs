use super::*;
use std::{
    fs,
    io::{Seek, SeekFrom},
    thread,
};

fn shell(script: &str) -> Vec<String> {
    vec!["/bin/sh".into(), "-c".into(), script.into()]
}

fn run(script: &str, root: &Path, input: Input<'_>, limit: usize, timeout: Duration) -> Outcome {
    execute(Request {
        argv: &shell(script),
        cwd: root,
        environment: Environment::Replace(&BTreeMap::from([(
            "PATH".into(),
            "/usr/bin:/bin".into(),
        )])),
        stdin: input,
        stdout: Capture::Memory { limit },
        stderr: Capture::Memory { limit },
        deadline: Instant::now() + timeout,
        termination_grace: Duration::from_millis(50),
        drain_grace: Duration::from_millis(50),
        cancelled: &AtomicBool::new(false),
    })
    .unwrap()
}

#[test]
fn preserves_exact_output_exit_and_explicit_environment() {
    let root = tempfile::tempdir().unwrap();
    let result = run(
        "test -z \"${HOME+x}\" || exit 99; printf out; printf err >&2; exit 7",
        root.path(),
        Input::Null,
        100,
        Duration::from_secs(2),
    );
    assert_eq!(result.exit_code, Some(7));
    assert_eq!(result.stdout.bytes, b"out");
    assert_eq!(result.stderr.bytes, b"err");
    assert_eq!(result.stdout.received_bytes, 3);
    assert_eq!(result.stdout.retained_bytes, 3);
    assert_eq!(
        result.stdout.prefix_sha256,
        format!("{:x}", Sha256::digest(b"out"))
    );
    assert!(result.stdout.complete && result.stderr.complete);
    assert!(!result.stdout.truncated && !result.stderr.truncated);
    assert!(result.stdin_complete && result.failure.is_none());

    let result = execute(Request {
        argv: &shell("printf '%s' \"$ONLY\""),
        cwd: root.path(),
        environment: Environment::Inherit(&BTreeMap::from([("ONLY".into(), "override".into())])),
        stdin: Input::Null,
        stdout: Capture::Memory { limit: 20 },
        stderr: Capture::Memory { limit: 0 },
        deadline: Instant::now() + Duration::from_secs(2),
        termination_grace: Duration::ZERO,
        drain_grace: Duration::from_millis(50),
        cancelled: &AtomicBool::new(false),
    })
    .unwrap();
    assert_eq!(result.stdout.bytes, b"override");
}

#[test]
fn drains_both_output_floods_after_their_distinct_limits() {
    let root = tempfile::tempdir().unwrap();
    let result = execute(Request {
        argv: &shell(
            "head -c 1048576 /dev/zero; head -c 1048576 /dev/zero >&2; printf done > marker",
        ),
        cwd: root.path(),
        environment: Environment::Replace(&BTreeMap::from([(
            "PATH".into(),
            "/usr/bin:/bin".into(),
        )])),
        stdin: Input::Null,
        stdout: Capture::Memory { limit: 113 },
        stderr: Capture::Memory { limit: 257 },
        deadline: Instant::now() + Duration::from_secs(3),
        termination_grace: Duration::from_millis(50),
        drain_grace: Duration::from_millis(50),
        cancelled: &AtomicBool::new(false),
    })
    .unwrap();
    assert_eq!(result.exit_code, Some(0));
    assert!(result.failure.is_none());
    assert_eq!(fs::read(root.path().join("marker")).unwrap(), b"done");
    for (capture, limit) in [(&result.stdout, 113), (&result.stderr, 257)] {
        assert_eq!(capture.received_bytes, 1048576);
        assert_eq!(capture.bytes, vec![0; limit]);
        assert_eq!(capture.retained_bytes, limit as u64);
        assert_eq!(
            capture.prefix_sha256,
            format!("{:x}", Sha256::digest(vec![0; limit]))
        );
        assert!(capture.truncated && capture.complete);
    }
}

#[test]
fn pumps_large_stdin_while_draining_stdout_without_deadlock() {
    let root = tempfile::tempdir().unwrap();
    let input = vec![42; 2 * 1024 * 1024];
    let result = run(
        "head -c 262144 /dev/zero; cat > received; printf diagnostic >&2",
        root.path(),
        Input::Bytes(&input),
        100,
        Duration::from_secs(3),
    );
    assert_eq!(result.exit_code, Some(0));
    assert!(result.failure.is_none());
    assert_eq!(result.stdin_bytes_sent, input.len() as u64);
    assert!(result.stdin_complete);
    assert_eq!(fs::read(root.path().join("received")).unwrap(), input);
    assert_eq!(result.stdout.received_bytes, 262144);
    assert!(result.stdout.truncated);
    assert_eq!(result.stderr.bytes, b"diagnostic");
}

#[test]
fn regular_file_stdin_streams_from_zero_without_moving_its_offset() {
    let root = tempfile::tempdir().unwrap();
    let bytes = (0u8..=255).cycle().take(300000).collect::<Vec<_>>();
    let path = root.path().join("input");
    fs::write(&path, &bytes).unwrap();
    let mut file = File::open(&path).unwrap();
    file.seek(SeekFrom::Start(17)).unwrap();
    let result = run(
        "cat > received",
        root.path(),
        Input::File {
            file: &file,
            length: bytes.len() as u64,
        },
        10,
        Duration::from_secs(3),
    );
    assert_eq!(result.exit_code, Some(0));
    assert!(result.stdin_complete && result.failure.is_none());
    assert_eq!(fs::read(root.path().join("received")).unwrap(), bytes);
    assert_eq!(file.stream_position().unwrap(), 17);

    let result = execute(Request {
        argv: &shell("touch unexpected"),
        cwd: root.path(),
        environment: Environment::Replace(&BTreeMap::new()),
        stdin: Input::File {
            file: &file,
            length: 1,
        },
        stdout: Capture::Memory { limit: 10 },
        stderr: Capture::Memory { limit: 10 },
        deadline: Instant::now() + Duration::from_secs(1),
        termination_grace: Duration::ZERO,
        drain_grace: Duration::ZERO,
        cancelled: &AtomicBool::new(false),
    });
    assert!(result.is_err());
    assert!(!root.path().join("unexpected").exists());
}

#[test]
fn blocked_stdin_is_covered_by_the_same_deadline() {
    let root = tempfile::tempdir().unwrap();
    let input = vec![1; 2 * 1024 * 1024];
    let started = Instant::now();
    let result = run(
        "sleep 10",
        root.path(),
        Input::Bytes(&input),
        10,
        Duration::from_millis(80),
    );
    assert!(result.spawned && result.timed_out);
    assert!(!result.stdin_complete);
    assert!(result.stdin_bytes_sent < input.len() as u64);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn early_closed_stdin_reports_incomplete_delivery_without_sigpipe() {
    let root = tempfile::tempdir().unwrap();
    let input = vec![1; 2 * 1024 * 1024];
    let result = run(
        "exec 0<&-; sleep 0.03; exit 0",
        root.path(),
        Input::Bytes(&input),
        10,
        Duration::from_secs(2),
    );
    assert!(!result.stdin_complete);
    assert_eq!(
        result.failure,
        Some(Failure {
            stage: "STDIN",
            code: "BROKEN_PIPE"
        })
    );
}

#[test]
fn file_capture_is_bounded_and_keeps_no_second_memory_copy() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("stdout");
    let mut file = File::create(&path).unwrap();
    let result = execute(Request {
        argv: &shell("head -c 1048576 /dev/zero"),
        cwd: root.path(),
        environment: Environment::Inherit(&BTreeMap::new()),
        stdin: Input::Null,
        stdout: Capture::File {
            file: &mut file,
            limit: 4096,
        },
        stderr: Capture::Memory { limit: 10 },
        deadline: Instant::now() + Duration::from_secs(3),
        termination_grace: Duration::from_millis(50),
        drain_grace: Duration::from_millis(50),
        cancelled: &AtomicBool::new(false),
    })
    .unwrap();
    assert_eq!(result.exit_code, Some(0));
    assert!(result.stdout.bytes.is_empty() && result.stdout.truncated);
    assert_eq!(result.stdout.received_bytes, 1048576);
    assert_eq!(result.stdout.retained_bytes, 4096);
    assert_eq!(fs::metadata(&path).unwrap().len(), 4096);
    assert_eq!(
        result.stdout.prefix_sha256,
        format!("{:x}", Sha256::digest(vec![0; 4096]))
    );
}

#[test]
fn cancellation_remains_responsive_during_a_continuous_output_flood() {
    let root = tempfile::tempdir().unwrap();
    let cancelled = AtomicBool::new(false);
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(50));
            cancelled.store(true, Ordering::Relaxed);
        });
        let started = Instant::now();
        let result = execute(Request {
            argv: &shell("while :; do printf flood; printf diagnostic >&2; done"),
            cwd: root.path(),
            environment: Environment::Inherit(&BTreeMap::new()),
            stdin: Input::Null,
            stdout: Capture::Memory { limit: 20 },
            stderr: Capture::Memory { limit: 30 },
            deadline: Instant::now() + Duration::from_secs(3),
            termination_grace: Duration::from_millis(50),
            drain_grace: Duration::from_millis(50),
            cancelled: &cancelled,
        })
        .unwrap();
        assert!(result.cancelled && !result.timed_out);
        assert!(result.stdout.truncated && result.stderr.truncated);
        assert!(started.elapsed() < Duration::from_secs(1));
    });
}

#[test]
fn expired_and_cancelled_requests_do_not_spawn() {
    let root = tempfile::tempdir().unwrap();
    for cancelled in [false, true] {
        let result = execute(Request {
            argv: &shell("touch unexpected"),
            cwd: root.path(),
            environment: Environment::Inherit(&BTreeMap::new()),
            stdin: Input::Null,
            stdout: Capture::Memory { limit: 10 },
            stderr: Capture::Memory { limit: 10 },
            deadline: Instant::now(),
            termination_grace: Duration::ZERO,
            drain_grace: Duration::ZERO,
            cancelled: &AtomicBool::new(cancelled),
        })
        .unwrap();
        assert!(!result.spawned);
        assert_eq!(
            result.stdout.prefix_sha256,
            format!("{:x}", Sha256::digest([]))
        );
        assert_eq!(result.stderr.prefix_sha256, result.stdout.prefix_sha256);
        assert_eq!(result.cancelled, cancelled);
        assert_eq!(result.timed_out, !cancelled);
    }
    assert!(!root.path().join("unexpected").exists());
}

#[test]
fn spawn_failure_exposes_only_safe_stage_and_kind() {
    let root = tempfile::tempdir().unwrap();
    let result = execute(Request {
        argv: &["/does-not-exist/test-only-secret".into()],
        cwd: root.path(),
        environment: Environment::Replace(&BTreeMap::new()),
        stdin: Input::Null,
        stdout: Capture::Memory { limit: 10 },
        stderr: Capture::Memory { limit: 10 },
        deadline: Instant::now() + Duration::from_secs(1),
        termination_grace: Duration::ZERO,
        drain_grace: Duration::ZERO,
        cancelled: &AtomicBool::new(false),
    })
    .unwrap();
    assert!(!result.spawned);
    assert_eq!(
        result.stdout.prefix_sha256,
        format!("{:x}", Sha256::digest([]))
    );
    assert_eq!(
        result.failure,
        Some(Failure {
            stage: "SPAWN",
            code: "NOT_FOUND"
        })
    );
    assert!(!format!("{result:?}").contains("test-only-secret"));
}

#[test]
fn externally_reaped_leader_disables_further_group_signals() {
    let child = Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .process_group(0)
        .spawn()
        .unwrap();
    let mut group = Group {
        child,
        reaped: false,
    };
    group.child.wait().unwrap();
    let error = group.exited().unwrap_err();
    assert_eq!(error_code(&error), "CHILD_OWNERSHIP_LOST");
    assert!(group.reaped);
    assert_eq!(
        group.signal(libc::SIGKILL).unwrap_err().raw_os_error(),
        Some(libc::ECHILD)
    );
}

#[test]
fn output_sink_failure_still_stops_the_child_and_returns_bounded_evidence() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("readonly");
    fs::write(&path, b"").unwrap();
    let mut file = File::open(&path).unwrap();
    let started = Instant::now();
    let result = execute(Request {
        argv: &shell("printf output; sleep 10"),
        cwd: root.path(),
        environment: Environment::Inherit(&BTreeMap::new()),
        stdin: Input::Null,
        stdout: Capture::File {
            file: &mut file,
            limit: 10,
        },
        stderr: Capture::Memory { limit: 10 },
        deadline: Instant::now() + Duration::from_secs(2),
        termination_grace: Duration::from_millis(50),
        drain_grace: Duration::from_millis(50),
        cancelled: &AtomicBool::new(false),
    })
    .unwrap();
    assert_eq!(result.failure.as_ref().unwrap().stage, "STDOUT");
    assert_eq!(result.stdout.received_bytes, 6);
    assert_eq!(result.stdout.retained_bytes, 0);
    assert!(result.stdout.truncated);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn termination_handlers_can_write_more_than_pipe_capacity_during_cleanup() {
    let root = tempfile::tempdir().unwrap();
    let result = execute(Request {
        argv: &shell("trap 'head -c 262144 /dev/zero; exit 0' TERM; while :; do sleep 1; done"),
        cwd: root.path(),
        environment: Environment::Inherit(&BTreeMap::new()),
        stdin: Input::Null,
        stdout: Capture::Memory { limit: 100 },
        stderr: Capture::Memory { limit: 100 },
        deadline: Instant::now() + Duration::from_millis(100),
        termination_grace: Duration::from_millis(500),
        drain_grace: Duration::from_millis(50),
        cancelled: &AtomicBool::new(false),
    })
    .unwrap();
    assert!(result.timed_out);
    assert_eq!(result.exit_code, Some(0));
    assert_eq!(result.stdout.received_bytes, 262144);
    assert!(result.stdout.complete && result.stdout.truncated);
    assert!(result.failure.is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn escaped_descendant_cannot_extend_pipe_wait_past_the_final_drain_budget() {
    let root = tempfile::tempdir().unwrap();
    let started = Instant::now();
    let result = run(
        "setsid /bin/sh -c 'echo $$ > escaped.pid; sleep 0.3' & while ! test -s escaped.pid; do sleep 0.002; done; exit 0",
        root.path(),
        Input::Null,
        100,
        Duration::from_secs(2),
    );
    assert_eq!(result.exit_code, Some(0));
    assert!(started.elapsed() < Duration::from_millis(250));
    assert_eq!(
        result.failure,
        Some(Failure {
            stage: "OUTPUT_DRAIN",
            code: "OUTPUT_DRAIN_TIMEOUT"
        })
    );
    assert!(!result.stdout.complete && !result.stderr.complete);
    // The deliberately escaped fixture self-terminates. Never signal a PID that
    // this supervisor does not own; merely await the bounded fixture's exit.
    let pid = fs::read_to_string(root.path().join("escaped.pid")).unwrap();
    for _ in 0..200 {
        let stat = fs::read_to_string(format!("/proc/{}/stat", pid.trim()));
        if stat.as_ref().map_or(true, |value| {
            value
                .rsplit_once(')')
                .is_some_and(|(_, fields)| fields.trim_start().starts_with('Z'))
        }) {
            return;
        }
        thread::sleep(Duration::from_millis(5));
    }
    panic!("self-terminating escaped fixture did not exit");
}

#[test]
fn blocked_stdin_handles_default_sigpipe_in_an_independent_process_and_restores_its_mask() {
    const CHILD: &str = "ML_EXP_PROCESS_SIGPIPE_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let result = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "process::tests::blocked_stdin_handles_default_sigpipe_in_an_independent_process_and_restores_its_mask", "--nocapture"])
            .env(CHILD, "1").output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    // SAFETY: only this dedicated test subprocess changes the disposition; no
    // parallel tests are run in it. This proves safety without Rust's default
    // ignored SIGPIPE disposition concealing a broken-pipe write failure.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    fn blocked() -> bool {
        // SAFETY: initialized writable signal set; a null set only queries the
        // calling thread's current mask and leaves it unchanged.
        let mut mask: libc::sigset_t = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), &mut mask) },
            0
        );
        (unsafe { libc::sigismember(&mask, libc::SIGPIPE) }) == 1
    }
    let before = blocked();
    let root = tempfile::tempdir().unwrap();
    let input = vec![1; 2 * 1024 * 1024];
    let result = run(
        "exec 0<&-; sleep 0.03",
        root.path(),
        Input::Bytes(&input),
        10,
        Duration::from_secs(2),
    );
    assert_eq!(
        result.failure,
        Some(Failure {
            stage: "STDIN",
            code: "BROKEN_PIPE"
        })
    );
    assert_eq!(blocked(), before);

    let (writer, reader) = std::os::unix::net::UnixStream::pair().unwrap();
    drop(reader);
    let frame = [0, 0, 0, 1, b'x'];
    assert_eq!(
        crate::framing::write_frame(
            &writer,
            &frame,
            16,
            Instant::now() + Duration::from_secs(1),
            &AtomicBool::new(false)
        )
        .unwrap_err()
        .kind(),
        io::ErrorKind::BrokenPipe
    );
    assert_eq!(blocked(), before);

    // An already pending signal belongs to the caller. Preserve both it and
    // the caller's blocked mask across a second broken-pipe execution.
    // SAFETY: only this isolated test subprocess manipulates its own thread's
    // signal mask and queues a thread-directed signal to itself.
    let mut set: libc::sigset_t = unsafe { std::mem::zeroed() };
    let mut previous: libc::sigset_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGPIPE);
        assert_eq!(
            libc::pthread_sigmask(libc::SIG_BLOCK, &set, &mut previous),
            0
        );
        assert_eq!(libc::pthread_kill(libc::pthread_self(), libc::SIGPIPE), 0);
    }
    let result = run(
        "exec 0<&-; sleep 0.03",
        root.path(),
        Input::Bytes(&input),
        10,
        Duration::from_secs(2),
    );
    assert_eq!(
        result.failure,
        Some(Failure {
            stage: "STDIN",
            code: "BROKEN_PIPE"
        })
    );
    assert!(blocked());
    assert_eq!(
        crate::framing::write_frame(
            &writer,
            &frame,
            16,
            Instant::now() + Duration::from_secs(1),
            &AtomicBool::new(false)
        )
        .unwrap_err()
        .kind(),
        io::ErrorKind::BrokenPipe
    );
    assert!(blocked());
    let mut pending: libc::sigset_t = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::sigpending(&mut pending) }, 0);
    assert_eq!(unsafe { libc::sigismember(&pending, libc::SIGPIPE) }, 1);
    let mut received = 0;
    assert_eq!(unsafe { libc::sigwait(&set, &mut received) }, 0);
    assert_eq!(received, libc::SIGPIPE);
    assert_eq!(
        unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &previous, std::ptr::null_mut()) },
        0
    );
}

#[test]
fn file_short_writes_hash_only_the_successfully_retained_prefix() {
    const CHILD: &str = "ML_EXP_PROCESS_SHORT_WRITE_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "process::tests::file_short_writes_hash_only_the_successfully_retained_prefix",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    // SAFETY: this isolated test process changes only its own signal disposition
    // and file-size limit. It deterministically forces a real regular-file
    // short write followed by EFBIG; no other tests execute in this subprocess.
    unsafe {
        libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
        assert_eq!(
            libc::setrlimit(
                libc::RLIMIT_FSIZE,
                &libc::rlimit {
                    rlim_cur: 7,
                    rlim_max: 7
                }
            ),
            0
        );
    }
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("capture");
    let mut file = File::create(&path).unwrap();
    let result = execute(Request {
        argv: &shell("printf 12345678901234567890"),
        cwd: root.path(),
        environment: Environment::Inherit(&BTreeMap::new()),
        stdin: Input::Null,
        stdout: Capture::File {
            file: &mut file,
            limit: 100,
        },
        stderr: Capture::Memory { limit: 100 },
        deadline: Instant::now() + Duration::from_secs(2),
        termination_grace: Duration::from_millis(50),
        drain_grace: Duration::from_millis(50),
        cancelled: &AtomicBool::new(false),
    })
    .unwrap();
    assert_eq!(result.failure.as_ref().unwrap().stage, "STDOUT");
    assert_eq!(result.stdout.received_bytes, 20);
    assert_eq!(result.stdout.retained_bytes, 7);
    assert_eq!(
        result.stdout.prefix_sha256,
        format!("{:x}", Sha256::digest(b"1234567"))
    );
    assert_eq!(fs::read(&path).unwrap(), b"1234567");
    assert!(result.stdout.truncated);
}
