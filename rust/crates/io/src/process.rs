//! Bounded, execute-once Unix child I/O. No shell, retry, signal handler or
//! environment policy is chosen implicitly. Killing this local process group
//! does not prove that a remote command has stopped or never started.
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Read, Write},
    os::{
        fd::AsRawFd,
        unix::{
            fs::FileExt,
            process::{CommandExt, ExitStatusExt},
        },
    },
    path::Path,
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

/// A caller must explicitly preserve or replace the inherited environment.
pub enum Environment<'a> {
    Inherit(&'a BTreeMap<String, String>),
    Replace(&'a BTreeMap<String, String>),
}

/// Files are read from offset zero without changing the caller's file offset.
/// A regular, fixed-size file is required; arbitrary blocking readers are not
/// accepted because their read operation cannot be bounded by this deadline.
pub enum Input<'a> {
    Null,
    Bytes(&'a [u8]),
    File { file: &'a File, length: u64 },
}

/// Retain a prefix in memory or in an already-open empty regular file. File
/// capture writes from offset zero without changing its caller's cursor. Excess
/// bytes are counted and drained, never used to grow memory or the capture file.
pub enum Capture<'a> {
    Memory { limit: usize },
    File { file: &'a mut File, limit: usize },
}

pub struct Request<'a> {
    pub argv: &'a [String],
    pub cwd: &'a Path,
    pub environment: Environment<'a>,
    pub stdin: Input<'a>,
    pub stdout: Capture<'a>,
    pub stderr: Capture<'a>,
    /// Absolute execution deadline, including stdin pipe writes and output
    /// draining. Cleanup may use the two separately declared finite budgets.
    pub deadline: Instant,
    /// Separate, finite budgets for termination and final pipe draining.
    pub termination_grace: Duration,
    pub drain_grace: Duration,
    pub cancelled: &'a AtomicBool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Captured {
    /// Empty for file captures, whose caller retains the file descriptor.
    pub bytes: Vec<u8>,
    pub received_bytes: u64,
    pub retained_bytes: u64,
    /// SHA256 of the exact retained pipe prefix, including file short writes.
    /// This remains the empty SHA256 when no child was spawned or no bytes
    /// were retained; it never describes an independently modified log file.
    pub prefix_sha256: String,
    pub truncated: bool,
    /// False when EOF could not be observed within the final drain budget.
    pub complete: bool,
}
impl Default for Captured {
    fn default() -> Self {
        Self {
            bytes: Vec::new(),
            received_bytes: 0,
            retained_bytes: 0,
            prefix_sha256: format!("{:x}", Sha256::digest([])),
            truncated: false,
            complete: false,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Failure {
    pub stage: &'static str,
    pub code: &'static str,
}

#[derive(Debug, Default)]
pub struct Outcome {
    pub spawned: bool,
    pub exit_code: Option<i32>,
    pub termination_signal: Option<i32>,
    pub timed_out: bool,
    pub cancelled: bool,
    pub stdin_bytes_sent: u64,
    pub stdin_complete: bool,
    pub stdout: Captured,
    pub stderr: Captured,
    pub failure: Option<Failure>,
}
impl Outcome {
    /// Bounded, sanitized evidence only; it does not establish a remote outcome
    /// or authorize a retry. The original exit and capture flags stay separate.
    pub fn diagnostic(&self) -> serde_json::Value {
        let captured = |value: &Captured| {
            serde_json::json!({
                "text": ml_exp_core::diagnostic::text(&String::from_utf8_lossy(&value.bytes), ""),
                "received_bytes": value.received_bytes,
                "retained_bytes": value.retained_bytes,
                "truncated": value.truncated,
                "complete": value.complete,
            })
        };
        serde_json::json!({
            "spawned": self.spawned,
            "exit_code": self.exit_code,
            "termination_signal": self.termination_signal,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "stdin_complete": self.stdin_complete,
            "failure": self.failure.as_ref().map(|f| serde_json::json!({"stage": f.stage, "code": f.code})),
            "stdout": captured(&self.stdout),
            "stderr": captured(&self.stderr),
        })
    }

    fn fail(&mut self, stage: &'static str, error: &io::Error) {
        if self.failure.is_none() {
            self.failure = Some(Failure {
                stage,
                code: error_code(error),
            });
        }
    }
}

pub fn error_code(error: &io::Error) -> &'static str {
    if error.raw_os_error() == Some(libc::ECHILD) {
        return "CHILD_OWNERSHIP_LOST";
    }
    match error.kind() {
        io::ErrorKind::NotFound => "NOT_FOUND",
        io::ErrorKind::PermissionDenied => "PERMISSION_DENIED",
        io::ErrorKind::InvalidInput => "INVALID_INPUT",
        io::ErrorKind::BrokenPipe => "BROKEN_PIPE",
        io::ErrorKind::UnexpectedEof => "UNEXPECTED_EOF",
        _ => "IO_ERROR",
    }
}

/// Retain the unreaped leader until every group signal is finished. Reaping it
/// with try_wait before cleanup would permit PID/PGID reuse by an unrelated job.
/// Embedders must not reap children owned here from another thread or handler.
struct Group {
    child: Child,
    reaped: bool,
}
impl Group {
    fn signal(&self, signal: i32) -> io::Result<()> {
        if self.reaped {
            return Err(io::Error::from_raw_os_error(libc::ECHILD));
        }
        let pid = i32::try_from(self.child.id())
            .map_err(|_| io::Error::other("child PID exceeds native range"))?;
        // SAFETY: process_group(0) makes this unreaped child's PID its group ID.
        // No wait/reap occurs until the final group signal has been sent.
        if unsafe { libc::kill(-pid, signal) } == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error)
        }
    }
    fn exited(&mut self) -> io::Result<bool> {
        // SAFETY: zeroed siginfo_t is valid writable C storage. waitid writes the
        // exact child's status; WNOWAIT intentionally leaves its identity owned.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let result = unsafe {
            libc::waitid(
                libc::P_PID,
                self.child.id() as libc::id_t,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if result == -1 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ECHILD) {
                self.reaped = true;
            }
            if error.kind() == io::ErrorKind::Interrupted {
                return Ok(false);
            }
            return Err(error);
        }
        // SAFETY: successful waitid defines si_pid, including zero for no exit.
        Ok(unsafe { info.si_pid() } != 0)
    }
    fn reap(&mut self) -> io::Result<ExitStatus> {
        let result = self.child.wait();
        if result.is_ok()
            || result.as_ref().err().and_then(io::Error::raw_os_error) == Some(libc::ECHILD)
        {
            self.reaped = true;
        }
        result
    }
}
impl Drop for Group {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.signal(libc::SIGKILL);
            let _ = self.child.wait();
        }
    }
}

fn nonblocking(fd: &impl AsRawFd) -> io::Result<()> {
    // SAFETY: this borrowed descriptor remains owned by its pipe object. These
    // fcntl operations change only that pipe's file status flags.
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
    if flags == -1
        || unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

// A pipe write may generate SIGPIPE. Block it only in this calling thread,
// preserving the previous mask/pending state, rather than installing a global
// handler. The child is spawned before this guard, so it inherits no new mask.
pub(crate) struct PipeSignals {
    previous: libc::sigset_t,
    set: libc::sigset_t,
    pending_before: bool,
    broken_pipe: bool,
}
impl PipeSignals {
    pub(crate) fn block() -> io::Result<Self> {
        // SAFETY: these are valid writable C signal sets, initialized before use.
        let mut set: libc::sigset_t = unsafe { std::mem::zeroed() };
        let mut previous: libc::sigset_t = unsafe { std::mem::zeroed() };
        let mut pending: libc::sigset_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, libc::SIGPIPE);
        }
        let code = unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &set, &mut previous) };
        if code != 0 {
            return Err(io::Error::from_raw_os_error(code));
        }
        if unsafe { libc::sigpending(&mut pending) } == -1 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::pthread_sigmask(libc::SIG_SETMASK, &previous, std::ptr::null_mut());
            }
            return Err(error);
        }
        Ok(Self {
            previous,
            set,
            pending_before: unsafe { libc::sigismember(&pending, libc::SIGPIPE) } == 1,
            broken_pipe: false,
        })
    }
    pub(crate) fn mark_broken_pipe(&mut self) {
        self.broken_pipe = true;
    }
}
impl Drop for PipeSignals {
    fn drop(&mut self) {
        if self.broken_pipe && !self.pending_before {
            // SAFETY: initialized writable signal set. EPIPE from this thread's
            // pipe write generated a thread-directed SIGPIPE unless ignored.
            // If it is pending, no other thread can consume it. sigwait is then
            // immediately ready; unlike sigtimedwait it is also available on
            // Darwin. A preexisting pending SIGPIPE is deliberately preserved.
            let mut pending: libc::sigset_t = unsafe { std::mem::zeroed() };
            if unsafe { libc::sigpending(&mut pending) } == 0
                && unsafe { libc::sigismember(&pending, libc::SIGPIPE) } == 1
            {
                let mut signal = 0;
                while unsafe { libc::sigwait(&self.set, &mut signal) } == libc::EINTR {}
            }
        }
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &self.previous, std::ptr::null_mut());
        }
    }
}

struct CaptureState<'a> {
    target: Capture<'a>,
    value: Captured,
    digest: Sha256,
    failed: bool,
}
impl<'a> CaptureState<'a> {
    fn new(target: Capture<'a>) -> io::Result<Self> {
        if let Capture::File { file, .. } = &target {
            let metadata = file.metadata()?;
            if !metadata.is_file() || metadata.len() != 0 {
                return Err(io::Error::other("capture requires a regular file"));
            }
        }
        Ok(Self {
            target,
            value: Captured::default(),
            digest: Sha256::new(),
            failed: false,
        })
    }
    fn push(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.value.received_bytes = self.value.received_bytes.saturating_add(bytes.len() as u64);
        let limit = match &self.target {
            Capture::Memory { limit } | Capture::File { limit, .. } => *limit as u64,
        };
        let retain = bytes
            .len()
            .min(limit.saturating_sub(self.value.retained_bytes) as usize);
        self.value.truncated |= retain < bytes.len() || self.failed;
        if self.failed {
            return Ok(());
        }
        match &mut self.target {
            Capture::Memory { .. } => {
                self.value.bytes.extend_from_slice(&bytes[..retain]);
                self.value.retained_bytes += retain as u64;
                self.digest.update(&bytes[..retain]);
            }
            Capture::File { file, .. } => {
                let mut remaining = &bytes[..retain];
                while !remaining.is_empty() {
                    match file.write_at(remaining, self.value.retained_bytes) {
                        Ok(0) => {
                            self.failed = true;
                            self.value.truncated = true;
                            return Err(io::Error::new(
                                io::ErrorKind::WriteZero,
                                "capture write failed",
                            ));
                        }
                        Ok(n) => {
                            self.value.retained_bytes += n as u64;
                            self.digest.update(&remaining[..n]);
                            remaining = &remaining[n..];
                        }
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                        Err(error) => {
                            self.failed = true;
                            self.value.truncated = true;
                            return Err(error);
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn finish(mut self) -> Captured {
        self.value.prefix_sha256 = format!("{:x}", self.digest.finalize());
        self.value
    }
    fn drain(&mut self, pipe: &mut Option<impl Read>) -> io::Result<bool> {
        let Some(reader) = pipe else {
            return Ok(false);
        };
        let mut progressed = false;
        let mut buffer = [0; 64 * 1024];
        // Fairness bound: a continuous flood cannot starve stdin, stderr,
        // deadline checks, cancellation or checking the exact child's status.
        for _ in 0..16 {
            match reader.read(&mut buffer) {
                Ok(0) => {
                    self.value.complete = true;
                    *pipe = None;
                    break;
                }
                Ok(n) => {
                    progressed = true;
                    self.push(&buffer[..n])?;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => break,
                Err(error) => return Err(error),
            }
        }
        Ok(progressed)
    }
}

struct InputState<'a> {
    input: Input<'a>,
    buffer: Vec<u8>,
    offset: usize,
    sent: u64,
    length: u64,
}
impl<'a> InputState<'a> {
    fn new(input: Input<'a>) -> io::Result<Self> {
        let length = match &input {
            Input::Null => 0,
            Input::Bytes(bytes) => bytes.len() as u64,
            Input::File { file, length } => {
                let metadata = file.metadata()?;
                if !metadata.is_file() || metadata.len() != *length {
                    return Err(io::Error::other(
                        "stdin requires a regular file with the declared length",
                    ));
                }
                *length
            }
        };
        Ok(Self {
            input,
            buffer: Vec::new(),
            offset: 0,
            sent: 0,
            length,
        })
    }
    fn pump(&mut self, pipe: &mut Option<ChildStdin>) -> io::Result<bool> {
        if self.sent == self.length {
            *pipe = None;
            return Ok(false);
        }
        let Some(writer) = pipe else {
            return Ok(false);
        };
        let bytes = match &self.input {
            Input::Null => unreachable!(),
            Input::Bytes(bytes) => &bytes[self.sent as usize..],
            Input::File { file, .. } => {
                if self.offset == self.buffer.len() {
                    self.buffer
                        .resize((self.length - self.sent).min(64 * 1024) as usize, 0);
                    let n = file.read_at(&mut self.buffer, self.sent)?;
                    if n == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "stdin file was truncated",
                        ));
                    }
                    self.buffer.truncate(n);
                    self.offset = 0;
                }
                &self.buffer[self.offset..]
            }
        };
        match writer.write(&bytes[..bytes.len().min(64 * 1024)]) {
            Ok(0) => Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "stdin write failed",
            )),
            Ok(n) => {
                self.sent += n as u64;
                self.offset += n;
                if self.sent == self.length {
                    *pipe = None;
                }
                Ok(true)
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                Ok(false)
            }
            Err(error) => Err(error),
        }
    }
}

fn drain_outputs(
    stdout: &mut CaptureState<'_>,
    out: &mut Option<ChildStdout>,
    stderr: &mut CaptureState<'_>,
    err: &mut Option<ChildStderr>,
    result: &mut Outcome,
) -> bool {
    let mut progressed = false;
    match stdout.drain(out) {
        Ok(value) => progressed |= value,
        Err(error) => result.fail("STDOUT", &error),
    }
    match stderr.drain(err) {
        Ok(value) => progressed |= value,
        Err(error) => result.fail("STDERR", &error),
    }
    progressed
}

/// Execute at most once. Spawn failure is distinguishable from a started child;
/// neither timeout nor a local exit code is proof about remote side effects.
/// Regular-file I/O and process spawn/wait still rely on the OS returning from
/// its syscalls; the deadline bounds pipe waiting, not an uninterruptible kernel.
pub fn execute(request: Request<'_>) -> io::Result<Outcome> {
    if request.argv.is_empty()
        || request.argv[0].is_empty()
        || !request.cwd.is_dir()
        || request.termination_grace > Duration::from_secs(30)
        || request.drain_grace > Duration::from_secs(30)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid bounded process request",
        ));
    }
    let mut input = InputState::new(request.stdin)?;
    let mut stdout = CaptureState::new(request.stdout)?;
    let mut stderr = CaptureState::new(request.stderr)?;
    let mut result = Outcome {
        cancelled: request.cancelled.load(Ordering::Relaxed),
        stdin_complete: input.length == 0,
        ..Outcome::default()
    };
    if result.cancelled || Instant::now() >= request.deadline {
        result.timed_out = !result.cancelled;
        result.stdout.complete = true;
        result.stderr.complete = true;
        return Ok(result);
    }
    let mut command = Command::new(&request.argv[0]);
    command
        .args(&request.argv[1..])
        .current_dir(request.cwd)
        .process_group(0)
        .stdin(if matches!(input.input, Input::Null) {
            Stdio::null()
        } else {
            Stdio::piped()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let env = match request.environment {
        Environment::Inherit(env) => env,
        Environment::Replace(env) => {
            command.env_clear();
            env
        }
    };
    command.envs(env);
    let child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            result.fail("SPAWN", &error);
            result.stdout.complete = true;
            result.stderr.complete = true;
            return Ok(result);
        }
    };
    result.spawned = true;
    let mut group = Group {
        child,
        reaped: false,
    };
    let mut in_pipe = group.child.stdin.take();
    let mut out_pipe = group.child.stdout.take();
    let mut err_pipe = group.child.stderr.take();
    let setup = (|| {
        if let Some(pipe) = &in_pipe {
            nonblocking(pipe)?;
        }
        nonblocking(
            out_pipe
                .as_ref()
                .ok_or_else(|| io::Error::other("missing stdout pipe"))?,
        )?;
        nonblocking(
            err_pipe
                .as_ref()
                .ok_or_else(|| io::Error::other("missing stderr pipe"))?,
        )?;
        PipeSignals::block()
    })();
    let mut signals = match setup {
        Ok(guard) => Some(guard),
        Err(error) => {
            result.fail("PIPE_SETUP", &error);
            None
        }
    };
    if result.failure.is_some() {
        // A pipe whose nonblocking setup failed must never be read in cleanup.
        out_pipe = None;
        err_pipe = None;
    }
    if result.failure.is_none() {
        loop {
            let mut progressed = drain_outputs(
                &mut stdout,
                &mut out_pipe,
                &mut stderr,
                &mut err_pipe,
                &mut result,
            );
            match input.pump(&mut in_pipe) {
                Ok(value) => progressed |= value,
                Err(error) => {
                    if error.kind() == io::ErrorKind::BrokenPipe {
                        if let Some(guard) = &mut signals {
                            guard.broken_pipe = true;
                        }
                    }
                    result.fail("STDIN", &error);
                }
            }
            match group.exited() {
                Ok(true) => break,
                Ok(false) => {}
                Err(error) => {
                    result.fail("WAIT", &error);
                    break;
                }
            }
            if result.failure.is_some() {
                break;
            }
            if request.cancelled.load(Ordering::Relaxed) {
                result.cancelled = true;
                break;
            }
            if Instant::now() >= request.deadline {
                result.timed_out = true;
                break;
            }
            if !progressed {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
    drop(in_pipe);
    if let Err(error) = group.signal(libc::SIGTERM) {
        result.fail("CLEANUP", &error);
    }
    let grace = Instant::now();
    while !group.reaped && grace.elapsed() < request.termination_grace {
        drain_outputs(
            &mut stdout,
            &mut out_pipe,
            &mut stderr,
            &mut err_pipe,
            &mut result,
        );
        match group.exited() {
            Ok(true) => break,
            Ok(false) => {}
            Err(error) => {
                result.fail("CLEANUP", &error);
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    if let Err(error) = group.signal(libc::SIGKILL) {
        result.fail("CLEANUP", &error);
    }
    match group.reap() {
        Ok(status) => {
            result.exit_code = status.code();
            result.termination_signal = status.signal();
        }
        Err(error) => result.fail("CLEANUP", &error),
    }
    // If a descendant escaped the group, it may retain a pipe indefinitely.
    // Never wait for its EOF without a second, explicit finite budget.
    let drain = Instant::now();
    while (out_pipe.is_some() || err_pipe.is_some()) && drain.elapsed() < request.drain_grace {
        let progressed = drain_outputs(
            &mut stdout,
            &mut out_pipe,
            &mut stderr,
            &mut err_pipe,
            &mut result,
        );
        if !progressed {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    if (out_pipe.is_some() || err_pipe.is_some()) && result.failure.is_none() {
        result.failure = Some(Failure {
            stage: "OUTPUT_DRAIN",
            code: "OUTPUT_DRAIN_TIMEOUT",
        });
    }
    drop(group);
    result.stdin_bytes_sent = input.sent;
    result.stdin_complete = input.sent == input.length;
    if !result.stdin_complete && result.failure.is_none() && !result.timed_out && !result.cancelled
    {
        result.failure = Some(Failure {
            stage: "STDIN",
            code: "STDIN_INCOMPLETE",
        });
    }
    result.stdout = stdout.finish();
    result.stderr = stderr.finish();
    Ok(result)
}

#[cfg(test)]
mod tests;
