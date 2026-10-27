use ml_exp_core::slurm::helper::{self, Reply, ReplyOutcome, Request, Target};
use ml_exp_io::process::{self, Capture, Environment, Input};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};

/// Supplied by the operator's trusted backend configuration, never an RPC
/// command. OpenSSH authenticates through that explicit configuration and an
/// optional local agent. Private key/password material is not an argv field.
#[derive(Clone, Debug)]
pub struct SshConfig {
    pub target: Target,
    pub ssh_binary: PathBuf,
    pub config_file: PathBuf,
    pub known_hosts_file: PathBuf,
    pub target_alias: String,
    pub helper_path: String,
    pub helper_config_path: String,
    pub local_cwd: PathBuf,
    pub agent_socket: Option<PathBuf>,
    pub budget: Budget,
}

#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub timeout: Duration,
    pub connect_timeout_seconds: u32,
    pub termination_grace: Duration,
    pub drain_grace: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ValidationError {
    #[error("invalid trusted Slurm SSH configuration field: {0}")]
    Configuration(&'static str),
    #[error("helper request differs from the trusted backend/profile/storage target")]
    Target,
    #[error("invalid frozen Slurm helper request")]
    Request,
}

/// Local evidence only. A local PID/SSH exit or successful pipe write cannot
/// establish whether a remote scheduler effect occurred.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LocalEvidence {
    pub spawned: bool,
    pub exit_code: Option<i32>,
    pub termination_signal: Option<i32>,
    pub stdin_bytes_sent: u64,
    pub stdin_complete: bool,
    pub stdout_bytes: u64,
    pub stdout_complete: bool,
    pub stdout_truncated: bool,
    pub stderr_bytes: u64,
    pub stderr_complete: bool,
    pub stderr_truncated: bool,
    pub process_failure: Option<LocalFailure>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalFailure {
    pub stage: &'static str,
    pub code: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SshDiagnostic {
    HostKeyRejected,
    AuthenticationFailed,
    ConnectionFailed,
    Unclassified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uncertainty {
    Cancelled,
    DeadlineExceeded,
    LocalBoundary {
        code: &'static str,
    },
    LocalProcess {
        stage: &'static str,
        code: &'static str,
    },
    SshExit {
        diagnostic: SshDiagnostic,
    },
    IncompleteInput,
    IncompleteOutput,
    ReplyTooLarge,
    ForeignReply,
    InvalidReply {
        code: helper::Error,
    },
    Remote {
        reason: helper::UnknownReason,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallOutcome {
    /// Authenticated, exactly bound helper receipt. SUBMITTED is a durable
    /// remote ledger receipt, not verified scheduler ownership or job success.
    Receipt { reply: Reply, local: LocalEvidence },
    Uncertain {
        reason: Uncertainty,
        local: LocalEvidence,
    },
}
impl CallOutcome {
    /// Even PREPARED cannot close/fence a delayed Submit request. This boundary
    /// offers no positive absence or automatic retry decision.
    pub fn safe_to_retry(&self) -> bool {
        false
    }
}

fn component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
}
fn safe_path(value: &str) -> bool {
    value.len() <= 4096
        && value.starts_with('/')
        && value.len() > 1
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"/._-".contains(&c))
        && value[1..]
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}
fn local_path(value: &Path) -> Option<&str> {
    value.to_str().filter(|value| safe_path(value))
}

impl SshConfig {
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.target
            .validate()
            .map_err(|_| ValidationError::Configuration("target"))?;
        for (field, value) in [
            ("ssh_binary", &self.ssh_binary),
            ("config_file", &self.config_file),
            ("known_hosts_file", &self.known_hosts_file),
            ("local_cwd", &self.local_cwd),
        ] {
            if local_path(value).is_none() {
                return Err(ValidationError::Configuration(field));
            }
        }
        if !component(&self.target_alias) {
            return Err(ValidationError::Configuration("target_alias"));
        }
        // OpenSSH joins remote argv into shell text. Only this fixed executable
        // grammar and literal subcommand are allowed; Vec<String> is not quoting.
        if !safe_path(&self.helper_path) {
            return Err(ValidationError::Configuration("helper_path"));
        }
        if !safe_path(&self.helper_config_path) {
            return Err(ValidationError::Configuration("helper_config_path"));
        }
        if self
            .agent_socket
            .as_deref()
            .is_some_and(|socket| local_path(socket).is_none())
        {
            return Err(ValidationError::Configuration("agent_socket"));
        }
        if self.budget.timeout.is_zero()
            || self.budget.timeout > Duration::from_secs(3600)
            || !(1..=300).contains(&self.budget.connect_timeout_seconds)
            || self.budget.termination_grace > Duration::from_secs(30)
            || self.budget.drain_grace > Duration::from_secs(30)
        {
            return Err(ValidationError::Configuration("budget"));
        }
        Ok(())
    }
    fn argv(&self) -> Vec<String> {
        let mut argv = vec![
            self.ssh_binary.to_string_lossy().into_owned(),
            "-F".into(),
            self.config_file.to_string_lossy().into_owned(),
        ];
        for option in [
            "BatchMode=yes".to_owned(),
            "StrictHostKeyChecking=yes".into(),
            format!("UserKnownHostsFile={}", self.known_hosts_file.display()),
            "GlobalKnownHostsFile=/dev/null".into(),
            "PasswordAuthentication=no".into(),
            "KbdInteractiveAuthentication=no".into(),
            "NumberOfPasswordPrompts=0".into(),
            "ConnectionAttempts=1".into(),
            format!("ConnectTimeout={}", self.budget.connect_timeout_seconds),
            "ControlMaster=no".into(),
            "ControlPath=none".into(),
            "ControlPersist=no".into(),
            "ClearAllForwardings=yes".into(),
            "ForwardAgent=no".into(),
            "PermitLocalCommand=no".into(),
            "ForkAfterAuthentication=no".into(),
            "StdinNull=no".into(),
            "RequestTTY=no".into(),
            "LogLevel=ERROR".into(),
        ] {
            argv.extend(["-o".into(), option]);
        }
        argv.extend([
            "-T".into(),
            "--".into(),
            self.target_alias.clone(),
            self.helper_path.clone(),
            "slurm-helper".into(),
            "--stdio".into(),
            "--config".into(),
            self.helper_config_path.clone(),
        ]);
        argv
    }
    fn environment(&self) -> BTreeMap<String, String> {
        let mut environment = BTreeMap::from([
            ("PATH".into(), "/usr/bin:/bin".into()),
            ("LANG".into(), "C".into()),
        ]);
        if let Some(socket) = &self.agent_socket {
            environment.insert(
                "SSH_AUTH_SOCK".into(),
                socket.to_string_lossy().into_owned(),
            );
        }
        environment
    }
}

fn evidence(outcome: &process::Outcome) -> LocalEvidence {
    LocalEvidence {
        spawned: outcome.spawned,
        exit_code: outcome.exit_code,
        termination_signal: outcome.termination_signal,
        stdin_bytes_sent: outcome.stdin_bytes_sent,
        stdin_complete: outcome.stdin_complete,
        stdout_bytes: outcome.stdout.received_bytes,
        stdout_complete: outcome.stdout.complete,
        stdout_truncated: outcome.stdout.truncated,
        stderr_bytes: outcome.stderr.received_bytes,
        stderr_complete: outcome.stderr.complete,
        stderr_truncated: outcome.stderr.truncated,
        process_failure: outcome.failure.as_ref().map(|failure| LocalFailure {
            stage: failure.stage,
            code: failure.code,
        }),
    }
}
fn ssh_diagnostic(stderr: &[u8]) -> SshDiagnostic {
    // Return fixed categories, never untrusted raw output or a secret-bearing
    // remote error string. Stderr itself is bounded and discarded after use.
    let text = String::from_utf8_lossy(stderr);
    if text.contains("Host key verification failed")
        || text.contains("REMOTE HOST IDENTIFICATION HAS CHANGED")
    {
        SshDiagnostic::HostKeyRejected
    } else if text.contains("Permission denied")
        || text.contains("No supported authentication methods")
    {
        SshDiagnostic::AuthenticationFailed
    } else if text.contains("Connection refused")
        || text.contains("Connection timed out")
        || text.contains("Could not resolve hostname")
    {
        SshDiagnostic::ConnectionFailed
    } else {
        SshDiagnostic::Unclassified
    }
}

/// Invoke exactly once. Validation errors occur before this invocation, but do
/// not prove the absence of an earlier remote effect with the same token.
/// Request/token/manifest bytes travel only through stdin; raw pipe content is
/// never included in the returned evidence, Debug or an error message.
pub fn call(
    config: &SshConfig,
    request: &Request,
    cancelled: &AtomicBool,
) -> Result<CallOutcome, ValidationError> {
    config.validate()?;
    if request.intent().target != config.target {
        return Err(ValidationError::Target);
    }
    let deadline = Instant::now() + config.budget.timeout;
    let frame = helper::encode_request(request).map_err(|_| ValidationError::Request)?;
    let argv = config.argv();
    let environment = config.environment();
    let outcome = match process::execute(process::Request {
        argv: &argv,
        cwd: &config.local_cwd,
        environment: Environment::Replace(&environment),
        stdin: Input::Bytes(&frame),
        stdout: Capture::Memory {
            limit: helper::MAX_FRAME_BYTES,
        },
        stderr: Capture::Memory { limit: 8192 },
        deadline,
        termination_grace: config.budget.termination_grace,
        drain_grace: config.budget.drain_grace,
        cancelled,
    }) {
        Ok(outcome) => outcome,
        Err(error) => {
            return Ok(CallOutcome::Uncertain {
                reason: Uncertainty::LocalBoundary {
                    code: process::error_code(&error),
                },
                local: LocalEvidence::default(),
            });
        }
    };
    let local = evidence(&outcome);
    let reason = if outcome.cancelled {
        Some(Uncertainty::Cancelled)
    } else if outcome.timed_out {
        Some(Uncertainty::DeadlineExceeded)
    } else if !outcome.spawned {
        Some(Uncertainty::LocalProcess {
            stage: outcome
                .failure
                .as_ref()
                .map_or("SPAWN", |failure| failure.stage),
            code: outcome
                .failure
                .as_ref()
                .map_or("NOT_STARTED", |failure| failure.code),
        })
    } else if outcome.exit_code != Some(0) || outcome.termination_signal.is_some() {
        Some(Uncertainty::SshExit {
            diagnostic: ssh_diagnostic(&outcome.stderr.bytes),
        })
    } else if let Some(failure) = &outcome.failure {
        Some(Uncertainty::LocalProcess {
            stage: failure.stage,
            code: failure.code,
        })
    } else if !outcome.stdin_complete {
        Some(Uncertainty::IncompleteInput)
    } else if !outcome.stdout.complete || !outcome.stderr.complete {
        Some(Uncertainty::IncompleteOutput)
    } else if outcome.stdout.truncated {
        Some(Uncertainty::ReplyTooLarge)
    } else {
        None
    };
    if let Some(reason) = reason {
        return Ok(CallOutcome::Uncertain { reason, local });
    }
    Ok(match helper::decode_reply(&outcome.stdout.bytes, request) {
        Ok(Reply {
            outcome: ReplyOutcome::Unknown { reason },
            ..
        }) => CallOutcome::Uncertain {
            reason: Uncertainty::Remote { reason },
            local,
        },
        Ok(reply) => CallOutcome::Receipt { reply, local },
        Err(helper::Error::Identity) => CallOutcome::Uncertain {
            reason: Uncertainty::ForeignReply,
            local,
        },
        Err(code) => CallOutcome::Uncertain {
            reason: Uncertainty::InvalidReply { code },
            local,
        },
    })
}
