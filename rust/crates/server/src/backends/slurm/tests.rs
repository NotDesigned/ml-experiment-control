use super::*;
mod action_tests;
mod submission_tests;
use ml_exp_core::slurm::{
    helper::{self, FrozenIntent, OperationTag, ReplyOutcome, Request, Scope, Target},
    scheduler_job_name,
};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

const CHILD_ROOT: &str = "ML_EXPD_SYNTHETIC_SSH_ROOT";
const SECRET: &str = "SYNTHETIC_CAPABILITY_DO_NOT_LOG";

fn target() -> Target {
    Target {
        backend_id: "slurm-test".into(),
        profile_sha256: helper::sha256_bytes(b"profile"),
        storage_scope: helper::sha256_bytes(b"storage"),
    }
}

fn request() -> Request {
    let token = "a".repeat(32);
    let manifest = format!("run_id: train\ncapability: {SECRET}\n");
    let name = scheduler_job_name("train", "attempt-001").unwrap();
    let script = format!(
        "#!/usr/bin/env bash\n#SBATCH --job-name={name}\n#SBATCH --comment=ml-exp-{token}\n\ntrue\n"
    );
    let intent = FrozenIntent {
        staging: None,
        target: target(),
        scope: Scope {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
        },
        submission_token: token,
        scheduler_name: name,
        manifest_sha256: helper::sha256_bytes(manifest.as_bytes()),
        script_sha256: helper::sha256_bytes(script.as_bytes()),
    };
    Request::prepare(intent, manifest, script).unwrap()
}

fn quote(path: &Path) -> String {
    ml_exp_core::slurm::shell_quote(path.to_str().unwrap()).unwrap()
}

fn config(root: &Path, mode: &str) -> SshConfig {
    fs::write(root.join("mode"), mode).unwrap();
    fs::write(
        root.join("ssh-config"),
        "Host fake-gateway\n  HostName invalid.example\n",
    )
    .unwrap();
    fs::write(root.join("known-hosts"), "synthetic test fixture\n").unwrap();
    let script = root.join("fake-ssh");
    // The test harness's own stdout is redirected: only the framed response
    // file becomes fake SSH stdout. The child is native Rust, not a Python
    // bridge, and never reaches a network or an actual scheduler.
    let program = format!(
        "#!/bin/sh\nset -eu\nprintf 'call\\n' >> {calls}\nprintf '%s\\0' \"$@\" > {argv}\nexport {CHILD_ROOT}={root}\n{executable} --exact backends::slurm::tests::fake_ssh_child --nocapture > {harness} 2>&1\ncat {response}\ncat {diagnostic} >&2\nexit \"$(cat {exit})\"\n",
        calls = quote(&root.join("calls")),
        argv = quote(&root.join("argv")),
        root = quote(root),
        executable = quote(&std::env::current_exe().unwrap()),
        harness = quote(&root.join("harness.log")),
        response = quote(&root.join("response.bin")),
        diagnostic = quote(&root.join("diagnostic.bin")),
        exit = quote(&root.join("exit"))
    );
    fs::write(&script, program).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
    SshConfig {
        target: target(),
        ssh_binary: script,
        config_file: root.join("ssh-config"),
        known_hosts_file: root.join("known-hosts"),
        target_alias: "fake-gateway".into(),
        helper_path: "/opt/ml-expd/bin/ml-exp-worker-rs".into(),
        helper_config_path: "/etc/ml-expd/slurm-helper.json".into(),
        local_cwd: root.into(),
        agent_socket: Some(root.join("agent.sock")),
        budget: Budget {
            timeout: Duration::from_secs(5),
            connect_timeout_seconds: 1,
            termination_grace: Duration::from_millis(20),
            drain_grace: Duration::from_millis(100),
        },
    }
}

fn ledger_config(root: &Path) -> ml_exp_worker::slurm::HelperConfig {
    // Trusted test-operator configuration, independent of the request frame.
    ml_exp_worker::slurm::HelperConfig {
        root: root.join("remote-helper"),
        shared_run_root: fs::read_to_string(root.join("native-shared-root"))
            .ok()
            .map(PathBuf::from),
        target: target(),
        sbatch_program: root.join("sbatch"),
        environment: BTreeMap::from([
            ("PATH".into(), "/usr/bin:/bin".into()),
            (
                "COUNTER".into(),
                root.join("sbatch-calls").display().to_string(),
            ),
            (
                "EVIDENCE".into(),
                root.join("evidence").display().to_string(),
            ),
            (
                "CANCEL_CALLS".into(),
                root.join("cancel-calls").display().to_string(),
            ),
        ]),
        lock_timeout: Duration::from_secs(2),
        execution_timeout: Duration::from_secs(1),
        scheduler: Some(ml_exp_worker::slurm::SchedulerTools {
            sacct_program: root.join("sacct"),
            squeue_program: root.join("squeue"),
            scancel_program: root.join("scancel"),
            partition: "h100".into(),
        }),
    }
}

#[test]
fn fake_ssh_child() {
    let Some(root) = std::env::var_os(CHILD_ROOT) else {
        return;
    };
    let root = PathBuf::from(root);
    let mode = fs::read_to_string(root.join("mode")).unwrap();
    fs::write(root.join("started"), b"started").unwrap();
    assert!(std::env::var_os("HOME").is_none());
    assert!(std::env::var_os("SSH_ASKPASS").is_none());
    assert_eq!(std::env::var("PATH").unwrap(), "/usr/bin:/bin");
    assert_eq!(
        std::env::var_os("SSH_AUTH_SOCK"),
        Some(root.join("agent.sock").into_os_string())
    );
    if mode == "hang" {
        std::thread::sleep(Duration::from_secs(20));
        return;
    }
    let mut frame = Vec::new();
    std::io::stdin().read_to_end(&mut frame).unwrap();
    let request = helper::decode_request(&frame).unwrap();
    fs::write(root.join("stdin.bin"), &frame).unwrap();
    let reply = if mode.starts_with("ledger") {
        let ledger = ml_exp_worker::slurm::Helper::open(ledger_config(&root)).unwrap();
        ledger.handle(&request, &AtomicBool::new(false)).unwrap()
    } else {
        let outcome = match request.operation() {
            OperationTag::PrepareSubmission => ReplyOutcome::Prepared,
            OperationTag::Submit | OperationTag::InspectSubmission => ReplyOutcome::Submitted {
                job_id: "24827".into(),
            },
            OperationTag::ObserveJob
            | OperationTag::ObserveWorker
            | OperationTag::ObserveCheckpoint
            | OperationTag::CancelJob => ReplyOutcome::Rejected {
                code: helper::Rejection::UnsupportedOperation,
            },
        };
        helper::Reply::for_request(&request, outcome).unwrap()
    };
    let mut response = helper::encode_reply(&reply).unwrap();
    let mut diagnostic = Vec::new();
    let mut exit = 0;
    match mode.as_str() {
        "normal" | "ledger" => {}
        "ledger-lost-submit" => {
            if request.operation() == OperationTag::Submit {
                response.clear();
                exit = 17;
            }
        }
        "ledger-lost-response" => {
            // The remote effect and durable receipt precede transport failure.
            // A later explicit Inspect can read it; the transport never retries.
            response.clear();
            exit = 255;
        }
        "malformed" => response = b"not a reply".to_vec(),
        "truncated" => {
            response.pop();
        }
        "trailing" => response.extend_from_slice(b"extra"),
        "foreign" => {
            let other = Request::inspect(request.intent().clone()).unwrap();
            let other = helper::Reply::for_request(&other, ReplyOutcome::Prepared).unwrap();
            response = helper::encode_reply(&other).unwrap();
        }
        "flood" => response = vec![b'x'; helper::MAX_FRAME_BYTES + 64],
        "secret-echo" => {
            response = frame.clone();
            diagnostic = frame;
        }
        "ssh255" => {
            diagnostic = b"Host key verification failed".to_vec();
            exit = 255;
        }
        "remote-unknown" => {
            response = helper::encode_reply(
                &helper::Reply::for_request(
                    &request,
                    ReplyOutcome::Unknown {
                        reason: helper::UnknownReason::SubmitStarted,
                    },
                )
                .unwrap(),
            )
            .unwrap()
        }
        "remote-rejected" => {
            response = helper::encode_reply(
                &helper::Reply::for_request(
                    &request,
                    ReplyOutcome::Rejected {
                        code: helper::Rejection::Unprepared,
                    },
                )
                .unwrap(),
            )
            .unwrap()
        }
        "stderr-flood" => diagnostic = vec![b'x'; 9000],
        other => panic!("unknown synthetic case: {other}"),
    }
    fs::write(root.join("response.bin"), response).unwrap();
    fs::write(root.join("diagnostic.bin"), diagnostic).unwrap();
    fs::write(root.join("exit"), exit.to_string()).unwrap();
}

fn calls(root: &Path) -> usize {
    fs::read_to_string(root.join("calls"))
        .unwrap()
        .lines()
        .count()
}

#[test]
fn prepare_submit_and_inspect_use_one_fixed_ssh_invocation_each_and_private_stdin() {
    let root = tempfile::tempdir().unwrap();
    let config = config(root.path(), "normal");
    let prepare = request();
    for request in [
        prepare.clone(),
        Request::submit(prepare.intent().clone()).unwrap(),
        Request::inspect(prepare.intent().clone()).unwrap(),
    ] {
        let result = call(&config, &request, &AtomicBool::new(false)).unwrap();
        assert!(!result.safe_to_retry());
        let CallOutcome::Receipt { reply, local } = result else {
            panic!("expected a bound reply");
        };
        assert_eq!(reply.identity, request.identity().unwrap());
        assert!(
            local.spawned && local.stdin_complete && local.stdout_complete && local.stderr_complete
        );
        assert_eq!(local.exit_code, Some(0));
        assert_eq!(
            fs::read(root.path().join("stdin.bin")).unwrap(),
            helper::encode_request(&request).unwrap()
        );
    }
    assert_eq!(calls(root.path()), 3);
    let argv = fs::read(root.path().join("argv")).unwrap();
    let argv = argv
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf8(part.to_vec()).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        &argv[argv.len() - 7..],
        [
            "--",
            "fake-gateway",
            "/opt/ml-expd/bin/ml-exp-worker-rs",
            "slurm-helper",
            "--stdio",
            "--config",
            "/etc/ml-expd/slurm-helper.json"
        ]
    );
    for required in [
        "StrictHostKeyChecking=yes",
        "ConnectionAttempts=1",
        "ForwardAgent=no",
        "ControlMaster=no",
        "PasswordAuthentication=no",
        "ForkAfterAuthentication=no",
        "StdinNull=no",
    ] {
        assert!(argv.iter().any(|value| value == required));
    }
    assert!(
        argv.iter()
            .any(|value| value
                == &format!("UserKnownHostsFile={}", config.known_hosts_file.display()))
    );
    assert!(
        !argv
            .iter()
            .any(|value| value.contains(SECRET)
                || value.contains(&prepare.intent().submission_token))
    );
}

#[test]
fn lost_ssh_response_is_recovered_from_real_native_helper_ledger_without_second_sbatch() {
    let root = tempfile::tempdir().unwrap();
    let config = config(root.path(), "ledger");
    let sbatch = root.path().join("sbatch");
    fs::write(
        &sbatch,
        format!(
            "#!/bin/sh\n[ \"$#\" -eq 3 ] && [ \"$1\" = --parsable ] && [ \"$2\" = --job-name=train--attempt-001 ] && [ \"$3\" = --comment=ml-exp-{} ] || exit 99\ncat >/dev/null\nprintf 'call\\n' >> \"$COUNTER\"\nprintf '24827;test-cluster\\n'\n",
            "a".repeat(32)
        ),
    )
    .unwrap();
    fs::set_permissions(&sbatch, fs::Permissions::from_mode(0o700)).unwrap();
    drop(ml_exp_worker::slurm::Helper::initialize(ledger_config(root.path())).unwrap());
    let prepare = request();
    let cancelled = AtomicBool::new(false);
    let prepared = call(&config, &prepare, &cancelled).unwrap();
    assert!(matches!(
        prepared,
        CallOutcome::Receipt {
            reply: helper::Reply {
                outcome: ReplyOutcome::Prepared,
                ..
            },
            ..
        }
    ));
    let submit = Request::submit(prepare.intent().clone()).unwrap();
    fs::write(root.path().join("mode"), "ledger-lost-response").unwrap();
    let lost = call(&config, &submit, &cancelled).unwrap();
    assert!(!lost.safe_to_retry());
    assert!(matches!(
        lost,
        CallOutcome::Uncertain {
            reason: Uncertainty::SshExit { .. },
            ..
        }
    ));
    assert_eq!(
        fs::read_to_string(root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
    fs::write(root.path().join("mode"), "ledger").unwrap();
    for request in [Request::inspect(prepare.intent().clone()).unwrap(), submit] {
        let result = call(&config, &request, &cancelled).unwrap();
        assert!(!result.safe_to_retry());
        let CallOutcome::Receipt { reply, .. } = result else {
            panic!("explicit reconciliation did not recover the remote receipt");
        };
        assert_eq!(reply.identity, request.identity().unwrap());
        assert_eq!(
            reply.outcome,
            ReplyOutcome::Submitted {
                job_id: "24827".into()
            }
        );
    }
    assert_eq!(calls(root.path()), 4);
    assert_eq!(
        fs::read_to_string(root.path().join("sbatch-calls")).unwrap(),
        "call\n"
    );
    let claim = root
        .path()
        .join("remote-helper/demo/runs/train/.submission-attempt-001");
    let (manifest, script) = prepare.prepared_content().unwrap();
    assert_eq!(
        fs::read(claim.join("manifest.yaml")).unwrap(),
        manifest.as_bytes()
    );
    assert_eq!(
        fs::read(claim.join("job.sbatch")).unwrap(),
        script.as_bytes()
    );
}

#[test]
fn failed_or_foreign_replies_are_uncertain_without_retries_or_raw_output() {
    for mode in [
        "malformed",
        "truncated",
        "trailing",
        "foreign",
        "flood",
        "secret-echo",
        "ssh255",
        "remote-unknown",
    ] {
        let root = tempfile::tempdir().unwrap();
        let config = config(root.path(), mode);
        let request = request();
        let result = call(&config, &request, &AtomicBool::new(false)).unwrap();
        assert!(!result.safe_to_retry());
        assert_eq!(calls(root.path()), 1);
        let text = format!("{result:?}");
        assert!(!text.contains(SECRET));
        assert!(!text.contains(&request.intent().submission_token));
        let CallOutcome::Uncertain { reason, local } = result else {
            panic!("{mode} was accepted");
        };
        assert!(local.spawned);
        match mode {
            "foreign" => assert_eq!(reason, Uncertainty::ForeignReply),
            "flood" => assert_eq!(reason, Uncertainty::ReplyTooLarge),
            "ssh255" => assert_eq!(
                reason,
                Uncertainty::SshExit {
                    diagnostic: SshDiagnostic::HostKeyRejected
                }
            ),
            "remote-unknown" => assert_eq!(
                reason,
                Uncertainty::Remote {
                    reason: helper::UnknownReason::SubmitStarted
                }
            ),
            _ => assert!(matches!(reason, Uncertainty::InvalidReply { .. })),
        }
    }
}

#[test]
fn bounded_stderr_and_remote_rejection_are_not_positive_absence() {
    for mode in ["stderr-flood", "remote-rejected"] {
        let root = tempfile::tempdir().unwrap();
        let result = call(
            &config(root.path(), mode),
            &request(),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!result.safe_to_retry());
        let CallOutcome::Receipt { reply, local } = result else {
            panic!("bound receipt was lost");
        };
        if mode == "stderr-flood" {
            assert!(local.stderr_truncated);
            assert_eq!(local.stderr_bytes, 9000);
        } else {
            assert_eq!(
                reply.outcome,
                ReplyOutcome::Rejected {
                    code: helper::Rejection::Unprepared
                }
            );
        }
    }
}

#[test]
fn timeout_and_cancellation_stop_local_processes_but_never_infer_remote_absence() {
    for cancellation in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut config = config(root.path(), "hang");
        config.budget.timeout = if cancellation {
            Duration::from_secs(5)
        } else {
            Duration::from_millis(250)
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        let signal = cancelled.clone();
        let marker = root.path().join("started");
        let controller = std::thread::spawn(move || {
            if cancellation {
                let start = Instant::now();
                while !marker.exists() && start.elapsed() < Duration::from_secs(2) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                signal.store(true, Ordering::Relaxed);
            }
        });
        let start = Instant::now();
        let result = call(&config, &request(), &cancelled).unwrap();
        controller.join().unwrap();
        assert!(start.elapsed() < Duration::from_secs(3));
        assert_eq!(calls(root.path()), 1);
        let CallOutcome::Uncertain { reason, .. } = result else {
            panic!("lost remote effect was accepted");
        };
        assert_eq!(
            reason,
            if cancellation {
                Uncertainty::Cancelled
            } else {
                Uncertainty::DeadlineExceeded
            }
        );
    }
}

#[test]
fn invalid_targets_commands_or_budgets_never_spawn_and_local_spawn_failures_stay_uncertain() {
    let root = tempfile::tempdir().unwrap();
    let base = config(root.path(), "normal");
    let request = request();
    for field in [
        "alias",
        "helper",
        "helper_config",
        "config",
        "target",
        "budget",
    ] {
        let mut config = base.clone();
        match field {
            "alias" => config.target_alias = "-oProxyCommand=touch".into(),
            "helper" => config.helper_path = "/bin/helper; touch /tmp/forbidden".into(),
            "helper_config" => config.helper_config_path = "/etc/config$(touch)".into(),
            "config" => config.config_file = "/tmp/config\nProxyCommand=bad".into(),
            "target" => config.target.profile_sha256 = helper::sha256_bytes(b"foreign"),
            "budget" => config.budget.timeout = Duration::ZERO,
            _ => unreachable!(),
        }
        assert!(call(&config, &request, &AtomicBool::new(false)).is_err());
        assert!(!root.path().join("calls").exists());
    }
    let mut missing = base;
    missing.ssh_binary = root.path().join("missing-ssh");
    let result = call(&missing, &request, &AtomicBool::new(false)).unwrap();
    assert!(matches!(
        result,
        CallOutcome::Uncertain {
            reason: Uncertainty::LocalProcess {
                stage: "SPAWN",
                code: "NOT_FOUND"
            },
            ..
        }
    ));
    assert!(!result.safe_to_retry());
}

#[test]
fn lost_cancel_ssh_reply_is_observed_without_second_scheduler_effect() {
    let root = tempfile::tempdir().unwrap();
    let config = config(root.path(), "ledger");
    for (name, body) in [
        ("sbatch", "cat >/dev/null; printf '24827\\n'"),
        ("sacct", "cat \"$EVIDENCE\""),
        ("squeue", "exit 0"),
        (
            "scancel",
            "[ \"$#\" = 1 ] && [ \"$1\" = 24827 ] || exit 99; printf 'call\\n' >> \"$CANCEL_CALLS\"",
        ),
    ] {
        let path = root.path().join(name);
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    drop(ml_exp_worker::slurm::Helper::initialize(ledger_config(root.path())).unwrap());
    let prepare = request();
    let cancelled = AtomicBool::new(false);
    for request in [
        &prepare,
        &Request::submit(prepare.intent().clone()).unwrap(),
    ] {
        assert!(matches!(
            call(&config, request, &cancelled).unwrap(),
            CallOutcome::Receipt { .. }
        ));
    }
    let evidence = root.path().join("evidence");
    fs::write(
        &evidence,
        "24827|train--attempt-001|ml-exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa|h100|RUNNING|0:01|0:0",
    )
    .unwrap();
    let observe = Request::observe_job(prepare.intent().clone(), "24827".into()).unwrap();
    assert!(matches!(
        call(&config, &observe, &cancelled).unwrap(),
        CallOutcome::Receipt {
            reply: helper::Reply {
                outcome: ReplyOutcome::JobObserved { .. },
                ..
            },
            ..
        }
    ));
    let cancel = Request::cancel_job(prepare.intent().clone(), "24827".into()).unwrap();
    fs::write(root.path().join("mode"), "ledger-lost-response").unwrap();
    let lost = call(&config, &cancel, &cancelled).unwrap();
    assert!(matches!(lost, CallOutcome::Uncertain { .. }));
    assert!(!lost.safe_to_retry());
    fs::write(root.path().join("mode"), "ledger").unwrap();
    assert!(matches!(
        call(&config, &cancel, &cancelled).unwrap(),
        CallOutcome::Receipt {
            reply: helper::Reply {
                outcome: ReplyOutcome::CancelPending { .. },
                ..
            },
            ..
        }
    ));
    fs::write(
        &evidence,
        "24827|train--attempt-001|ml-exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa|h100|CANCELLED|0:02|0:15",
    )
    .unwrap();
    assert!(
        matches!(call(&config,&cancel,&cancelled).unwrap(),CallOutcome::Receipt {reply:helper::Reply {outcome:ReplyOutcome::JobObserved {observation},..},..} if observation.status.state==ml_exp_core::state::RunState::Cancelled)
    );
    assert_eq!(
        fs::read_to_string(root.path().join("cancel-calls")).unwrap(),
        "call\n"
    );
}
