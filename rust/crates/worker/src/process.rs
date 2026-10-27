use crate::Execution;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, Metadata, OpenOptions},
    io::{self, Read, Seek, SeekFrom},
    os::unix::{
        fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
        process::{CommandExt, ExitStatusExt},
    },
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

struct Log {
    path: PathBuf,
    file: File,
}

impl Log {
    fn create(path: PathBuf) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        Ok(Self { path, file })
    }

    fn verify_path(&self, expected: &Metadata) -> io::Result<()> {
        // lstat never opens a substituted FIFO or follows a substituted symlink.
        let current = fs::symlink_metadata(&self.path)?;
        if !current.is_file() || current.dev() != expected.dev() || current.ino() != expected.ino()
        {
            return Err(io::Error::other(
                "log path no longer names its original file",
            ));
        }
        Ok(())
    }

    fn hash(&mut self) -> io::Result<String> {
        let expected = self.file.metadata()?;
        self.verify_path(&expected)?;
        self.file.seek(SeekFrom::Start(0))?;
        let mut remaining = expected.len();
        let mut digest = Sha256::new();
        let mut buffer = [0; 64 * 1024];
        // Bound the read to the retained FD's original size, even if a process
        // outside this process group is still writing. A changed file is failed.
        while remaining > 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            let n = self.file.read(&mut buffer[..count])?;
            if n == 0 {
                return Err(io::Error::other("log changed while being hashed"));
            }
            digest.update(&buffer[..n]);
            remaining -= n as u64;
        }
        let current = self.file.metadata()?;
        self.verify_path(&expected)?;
        if current.len() != expected.len()
            || current.mtime() != expected.mtime()
            || current.mtime_nsec() != expected.mtime_nsec()
            || current.ctime() != expected.ctime()
            || current.ctime_nsec() != expected.ctime_nsec()
        {
            return Err(io::Error::other("log changed while being hashed"));
        }
        Ok(format!("{:x}", digest.finalize()))
    }
}

/// Own the unreaped group leader until all process-group signals are finished.
/// Child::try_wait would reap it and permit PID/PGID reuse before final cleanup.
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
        // SAFETY: process_group(0) made the child's PID its process-group ID.
        // We do not reap this leader until after the last signal, so this ID
        // cannot be reused by an unrelated process group during supervision.
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
        // SAFETY: an all-zero siginfo_t is valid writable C storage. waitid
        // fills it when this exact child exits; WNOWAIT deliberately keeps the
        // child unreaped. No Rust references alias this storage during the call.
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
                // An embedding process must not reap children owned here. If it
                // does, PID ownership is lost: never signal a possibly reused ID.
                self.reaped = true;
            }
            if error.kind() == io::ErrorKind::Interrupted {
                return Ok(false);
            }
            return Err(error);
        }
        // SAFETY: waitid succeeded and the si_pid field is defined for its
        // child-status result, including zero for no available child exit.
        Ok(unsafe { info.si_pid() } != 0)
    }

    fn finish(&mut self) -> io::Result<ExitStatus> {
        self.signal(libc::SIGTERM)?;
        let grace = Instant::now();
        while !self.exited()? && grace.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(20));
        }
        // A completed leader may have left descendants behind. Kill the entire
        // group before reaping the leader, also when its own exit was successful.
        self.signal(libc::SIGKILL)?;
        match self.child.wait() {
            Ok(status) => {
                self.reaped = true;
                Ok(status)
            }
            Err(error) => {
                if error.raw_os_error() == Some(libc::ECHILD) {
                    self.reaped = true;
                }
                Err(error)
            }
        }
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

fn error_code(error: &io::Error) -> &'static str {
    if error.raw_os_error() == Some(libc::ECHILD) {
        return "CHILD_OWNERSHIP_LOST";
    }
    match error.kind() {
        io::ErrorKind::NotFound => "NOT_FOUND",
        io::ErrorKind::PermissionDenied => "PERMISSION_DENIED",
        io::ErrorKind::InvalidInput => "INVALID_INPUT",
        _ => "IO_ERROR",
    }
}

pub(super) fn execute(
    workdir: &Path,
    output: &Path,
    argv: &[String],
    timeout: Duration,
    cancelled: &AtomicBool,
) -> io::Result<Execution> {
    // Atomic private creation, not create-then-chmod. Existing output is refused.
    fs::DirBuilder::new().mode(0o700).create(output)?;
    let mut stdout = Log::create(output.join("stdout.log"))?;
    let mut stderr = Log::create(output.join("stderr.log"))?;
    let mut result = Execution {
        contract: "rust-local-execution-preview.v1",
        exit_code: None,
        termination_signal: None,
        timed_out: false,
        cancelled: cancelled.load(Ordering::Relaxed),
        error: None,
        stdout_sha256: None,
        stderr_sha256: None,
    };
    if !result.cancelled {
        let mut command = Command::new(&argv[0]);
        command
            .args(&argv[1..])
            .current_dir(workdir)
            .env("OUTPUT_DIR", output.canonicalize()?)
            .stdin(Stdio::null())
            .stdout(stdout.file.try_clone()?)
            .stderr(stderr.file.try_clone()?)
            .process_group(0);
        match command.spawn() {
            Err(error) => result.fail("SPAWN", error_code(&error)),
            Ok(child) => {
                let mut group = Group {
                    child,
                    reaped: false,
                };
                let start = Instant::now();
                loop {
                    match group.exited() {
                        Ok(true) => break,
                        Err(error) => {
                            result.fail("WAIT", error_code(&error));
                            break;
                        }
                        Ok(false) => {}
                    }
                    if cancelled.load(Ordering::Relaxed) {
                        result.cancelled = true;
                        break;
                    }
                    if start.elapsed() >= timeout {
                        result.timed_out = true;
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                match group.finish() {
                    Ok(status) => {
                        result.exit_code = status.code();
                        result.termination_signal = status.signal();
                    }
                    Err(error) => result.fail("CLEANUP", error_code(&error)),
                }
                // Emergency cleanup precedes inspection/publication even if an
                // earlier wait or cleanup operation returned an error.
                drop(group);
            }
        }
    }
    for (log, field) in [
        (&mut stdout, &mut result.stdout_sha256),
        (&mut stderr, &mut result.stderr_sha256),
    ] {
        match log.hash() {
            Ok(hash) => *field = Some(hash),
            Err(_) => {
                if result.error.is_none() {
                    result.error = Some(crate::Failure {
                        stage: "LOG_CAPTURE",
                        code: "LOG_IDENTITY_OR_CONTENT_CHANGED",
                    });
                }
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn externally_reaped_leader_disables_further_group_signals() {
        let child = Command::new("sh")
            .args(["-c", "exit 0"])
            .process_group(0)
            .spawn()
            .unwrap();
        let mut group = Group {
            child,
            reaped: false,
        };
        // Deliberately simulate an embedder stealing this library's wait status.
        group.child.wait().unwrap();
        let error = group.exited().unwrap_err();
        assert_eq!(error_code(&error), "CHILD_OWNERSHIP_LOST");
        assert!(group.reaped);
        assert_eq!(
            group.signal(libc::SIGKILL).unwrap_err().raw_os_error(),
            Some(libc::ECHILD)
        );
    }
}
