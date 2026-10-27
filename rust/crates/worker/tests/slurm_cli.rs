//! Real binary, temporary metadata and fake sbatch only. No SSH or scheduler.
#![cfg(unix)]
#![forbid(unsafe_code)]
use ml_exp_core::slurm::helper::{self, FrozenIntent, ReplyOutcome, Request, Scope, Target};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::{Child, Command, Output, Stdio},
    time::{Duration, Instant},
};
const ERROR: &str = "native Slurm helper failed; state may be uncertain; no retry attempted\n";
fn target() -> Target {
    Target {
        backend_id: "slurm-test".into(),
        profile_sha256: helper::sha256_bytes(b"profile"),
        storage_scope: helper::sha256_bytes(b"storage"),
    }
}
fn request() -> Request {
    let manifest = "run_id: train\n".to_owned();
    let script = format!(
        "#!/bin/sh\n#SBATCH --job-name=train--attempt-001\n#SBATCH --comment=ml-exp-{}\n\nexit 0\n",
        "a".repeat(32)
    );
    Request::prepare(
        FrozenIntent {
            target: target(),
            scope: Scope {
                project: "demo".into(),
                run_id: "train".into(),
                attempt_id: "attempt-001".into(),
            },
            submission_token: "a".repeat(32),
            scheduler_name: "train--attempt-001".into(),
            manifest_sha256: helper::sha256_bytes(manifest.as_bytes()),
            script_sha256: helper::sha256_bytes(script.as_bytes()),
        },
        manifest,
        script,
    )
    .unwrap()
}
struct Fixture {
    _root: tempfile::TempDir,
    config: PathBuf,
    helper_root: PathBuf,
    calls: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let helper_root = root.path().join("ledger");
        let config = root.path().join("helper.json");
        let program = root.path().join("sbatch");
        let calls = root.path().join("calls");
        fs::write(&program,"#!/bin/sh\n[ \"$#\" -eq 3 ] && [ \"$1\" = --parsable ] && [ \"$2\" = --job-name=train--attempt-001 ] && [ \"$3\" = --comment=ml-exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ] || exit 99\ncat >/dev/null\nprintf 'call\\n' >> \"$COUNTER\"\nprintf '12345;synthetic\\n'\n").unwrap();
        fs::set_permissions(&program, fs::Permissions::from_mode(0o700)).unwrap();
        let value = json!({"root":helper_root,"target":target(),"sbatch_program":program,"environment":{"PATH":"/usr/bin:/bin","COUNTER":calls,"SYNTHETIC_SECRET":"never-print-this-synthetic-value"},"lock_timeout_seconds":1,"execution_timeout_seconds":1,"stdio_timeout_seconds":1});
        fs::write(&config, serde_json::to_vec(&value).unwrap()).unwrap();
        fs::set_permissions(&config, fs::Permissions::from_mode(0o600)).unwrap();
        Self {
            _root: root,
            config,
            helper_root,
            calls,
        }
    }
    fn command(&self, mode: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ml-exp-worker-rs"));
        command
            .args(["slurm-helper", "--config"])
            .arg(&self.config)
            .arg(mode)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }
    fn initialize(&self) {
        let output = self.command("--initialize").output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        assert!(output.stderr.is_empty());
    }
    fn rpc(&self, frame: &[u8]) -> Output {
        let mut child = self.command("--stdio").spawn().unwrap();
        child.stdin.take().unwrap().write_all(frame).unwrap();
        collect(child)
    }
    fn update(&self, change: impl FnOnce(&mut Value)) {
        let mut value: Value = serde_json::from_slice(&fs::read(&self.config).unwrap()).unwrap();
        change(&mut value);
        fs::write(&self.config, serde_json::to_vec(&value).unwrap()).unwrap();
    }
}
fn collect(mut child: Child) -> Output {
    let started = Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if started.elapsed() > Duration::from_secs(5) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("helper exceeded bounded test lifetime");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn rejected(output: Output) {
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(String::from_utf8(output.stderr).unwrap(), ERROR);
}
fn reply(output: Output, request: &Request) -> helper::Reply {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    helper::decode_reply(&output.stdout, request).unwrap()
}
#[test]
fn binary_prepares_submits_and_inspects_one_frozen_claim_without_replaying_sbatch() {
    let fixture = Fixture::new();
    fixture.initialize();
    let prepare = request();
    let frame = helper::encode_request(&prepare).unwrap();
    assert_eq!(
        reply(fixture.rpc(&frame), &prepare).outcome,
        ReplyOutcome::Prepared
    );
    let submit = Request::submit(prepare.intent().clone()).unwrap();
    let frame = helper::encode_request(&submit).unwrap();
    for _ in 0..2 {
        assert_eq!(
            reply(fixture.rpc(&frame), &submit).outcome,
            ReplyOutcome::Submitted {
                job_id: "12345".into()
            }
        );
    }
    let inspect = Request::inspect(prepare.intent().clone()).unwrap();
    assert_eq!(
        reply(
            fixture.rpc(&helper::encode_request(&inspect).unwrap()),
            &inspect
        )
        .outcome,
        ReplyOutcome::Submitted {
            job_id: "12345".into()
        }
    );
    assert_eq!(fs::read_to_string(&fixture.calls).unwrap(), "call\n");
    let mut foreign = prepare.intent().clone();
    foreign.target.backend_id = "foreign".into();
    let foreign = Request::inspect(foreign).unwrap();
    assert!(matches!(
        reply(
            fixture.rpc(&helper::encode_request(&foreign).unwrap()),
            &foreign
        )
        .outcome,
        ReplyOutcome::Rejected { .. }
    ));
    assert_eq!(fs::read_to_string(&fixture.calls).unwrap(), "call\n");
}
#[test]
fn ordinary_rpc_never_bootstraps_or_repairs_missing_helper_locks() {
    let fixture = Fixture::new();
    let frame = helper::encode_request(&request()).unwrap();
    rejected(fixture.rpc(&frame));
    assert!(!fixture.helper_root.exists());
    fixture.initialize();
    let marker = fs::read(fixture.helper_root.join("root.json")).unwrap();
    rejected(fixture.command("--initialize").output().unwrap());
    assert_eq!(
        fs::read(fixture.helper_root.join("root.json")).unwrap(),
        marker
    );
    fs::remove_file(fixture.helper_root.join(".slurm-helper.lock")).unwrap();
    rejected(fixture.rpc(&frame));
    assert!(!fixture.helper_root.join(".slurm-helper.lock").exists());
}
#[test]
fn malformed_oversized_and_multiple_frames_fail_before_writing_any_claim() {
    let fixture = Fixture::new();
    fixture.initialize();
    let registry = fs::read(fixture.helper_root.join("registry.json")).unwrap();
    let frame = helper::encode_request(&request()).unwrap();
    for frame in [
        vec![],
        vec![0, 0],
        (9u32 * 1024 * 1024).to_be_bytes().to_vec(),
        vec![0, 0, 0, 1, b'{'],
        [frame.clone(), frame].concat(),
    ] {
        rejected(fixture.rpc(&frame));
    }
    assert_eq!(
        fs::read(fixture.helper_root.join("registry.json")).unwrap(),
        registry
    );
    assert!(!fixture.helper_root.join("demo").exists());
    assert!(!fixture.calls.exists());
}
#[test]
fn incomplete_input_or_missing_eof_times_out_without_publishing_a_claim() {
    let fixture = Fixture::new();
    fixture.initialize();
    let frame = helper::encode_request(&request()).unwrap();
    for sent in [vec![], frame] {
        let mut child = fixture.command("--stdio").spawn().unwrap();
        child.stdin.as_mut().unwrap().write_all(&sent).unwrap();
        let started = Instant::now();
        rejected(collect(child));
        assert!(started.elapsed() < Duration::from_secs(3));
    }
    assert!(!fixture.helper_root.join("demo").exists());
}
#[test]
fn private_local_configuration_and_mode_flags_cannot_be_overridden_by_a_frame() {
    for (name, value) in [
        ("lock_timeout_seconds", json!(0)),
        ("lock_timeout_seconds", json!(31)),
        ("execution_timeout_seconds", json!(301)),
        ("stdio_timeout_seconds", json!(0)),
        ("stdio_timeout_seconds", json!(31)),
        ("root", json!("relative")),
        ("sbatch_program", json!("sbatch")),
        ("unexpected", json!("never-print-this-synthetic-value")),
        ("environment", json!({"BAD-NAME":"x"})),
        ("environment", json!({"VALID":"bad\u{0}value"})),
    ] {
        let fixture = Fixture::new();
        fixture.update(|config| config[name] = value);
        rejected(fixture.command("--initialize").output().unwrap());
        assert!(!fixture.helper_root.exists());
    }
    let fixture = Fixture::new();
    fs::set_permissions(&fixture.config, fs::Permissions::from_mode(0o644)).unwrap();
    rejected(fixture.command("--initialize").output().unwrap());
    assert!(!fixture.helper_root.exists());
    let fixture = Fixture::new();
    let output = fixture
        .command("--stdio")
        .arg("--initialize")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(output.stderr, b"invalid native Slurm helper arguments\n");
    assert!(!fixture.helper_root.exists());
}
#[test]
fn redirected_regular_file_is_rejected_instead_of_promising_a_pipe_deadline() {
    let fixture = Fixture::new();
    fixture.initialize();
    let file = fixture._root.path().join("frame");
    fs::write(&file, helper::encode_request(&request()).unwrap()).unwrap();
    let mut command = fixture.command("--stdio");
    command.stdin(fs::File::open(file).unwrap());
    rejected(command.output().unwrap());
    assert!(!fixture.helper_root.join("demo").exists());
}

#[test]
fn explicit_environment_is_bounded_and_cannot_override_scheduler_identity() {
    let fixture = Fixture::new();
    fixture.update(|config| {
        config["environment"] =
            Value::Object((0..128).map(|i| (format!("E{i}"), json!("v"))).collect())
    });
    fixture.initialize();
    for variable in [
        "SBATCH_JOB_NAME",
        "SBATCH_COMMENT",
        "SBATCH_X",
        "SLURM_CLUSTERS",
        "SLURM_CONF",
        "SLURM_CONF_SERVER",
    ] {
        let fixture = Fixture::new();
        fixture.update(|config| {
            config["environment"] = json!({variable:"never-print-this-synthetic-value"})
        });
        rejected(fixture.command("--initialize").output().unwrap());
        assert!(!fixture.helper_root.exists());
    }
    let fixture = Fixture::new();
    fixture.update(|config| {
        config["environment"] =
            Value::Object((0..129).map(|i| (format!("E{i}"), json!("v"))).collect())
    });
    rejected(fixture.command("--initialize").output().unwrap());
    assert!(!fixture.helper_root.exists());
    let fixture = Fixture::new();
    fixture.update(|config| config["environment"] = json!({"X":"x".repeat(64*1024)}));
    rejected(fixture.command("--initialize").output().unwrap());
    assert!(!fixture.helper_root.exists());
}

#[test]
fn blocked_stdout_exits_with_uncertainty_within_its_write_budget() {
    use std::os::{fd::OwnedFd, unix::net::UnixStream};
    let fixture = Fixture::new();
    fixture.initialize();
    let (mut writer, _reader) = UnixStream::pair().unwrap();
    writer.set_nonblocking(true).unwrap();
    let mut filled = 0;
    loop {
        match writer.write(&[0; 8192]) {
            Ok(count) => filled += count,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
            other => panic!("cannot fill synthetic output transport: {other:?}"),
        }
        assert!(filled < 16 * 1024 * 1024);
    }
    writer.set_nonblocking(false).unwrap();
    let mut command = fixture.command("--stdio");
    command.stdout(Stdio::from(OwnedFd::from(writer)));
    let mut child = command.spawn().unwrap();
    let inspect = Request::inspect(request().intent().clone()).unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&helper::encode_request(&inspect).unwrap())
        .unwrap();
    let started = Instant::now();
    rejected(collect(child));
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(!fixture.helper_root.join("demo").exists());
}
#[cfg(target_os = "linux")]
#[test]
fn term_interrupts_a_stalled_request_and_exits_without_stdout_or_state_changes() {
    let fixture = Fixture::new();
    fixture.initialize();
    fixture.update(|config| config["stdio_timeout_seconds"] = json!(30));
    let mut child = fixture.command("--stdio").spawn().unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert!(child.try_wait().unwrap().is_none());
    assert!(
        Command::new("/bin/kill")
            .args(["-TERM", &child.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let started = Instant::now();
    rejected(collect(child));
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(!fixture.helper_root.join("demo").exists());
}
