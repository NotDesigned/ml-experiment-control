use crate::Execution;
use ml_exp_io::process;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, Metadata, OpenOptions},
    io::{self, Read, Seek, SeekFrom},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
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

    fn hash(&mut self, captured_bytes: u64, captured_sha256: &str) -> io::Result<String> {
        let expected = self.file.metadata()?;
        self.verify_path(&expected)?;
        // User code can open this path independently of the stdout/stderr pipe.
        // Its filesystem writes are not a process-capture quota. Fail closed if
        // the log no longer has the exact bounded size recorded by our pump.
        if captured_bytes > LOG_LIMIT as u64 || expected.len() != captured_bytes {
            return Err(io::Error::other("log differs from bounded capture length"));
        }
        self.file.seek(SeekFrom::Start(0))?;
        let mut remaining = captured_bytes;
        let mut digest = Sha256::new();
        let mut buffer = [0; 64 * 1024];
        // Bound the read to the supervisor's capture count, never a length that
        // user code supplied by changing the file. Later changes are also failed.
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
        let actual = format!("{:x}", digest.finalize());
        if actual != captured_sha256 {
            return Err(io::Error::other("log differs from captured pipe bytes"));
        }
        Ok(actual)
    }
}

// This bounds pipe capture and hashing, not user code's filesystem allocation.
// A failed output limit is reported through the existing receipt error field;
// a truncated or independently changed log cannot be a complete success.
const LOG_LIMIT: usize = 8 * 1024 * 1024;

pub(super) fn execute(
    workdir: &Path,
    output: &Path,
    argv: &[String],
    timeout: Duration,
    cancelled: &AtomicBool,
) -> io::Result<Execution> {
    fs::DirBuilder::new().mode(0o700).create(output)?;
    let mut stdout = Log::create(output.join("stdout.log"))?;
    let mut stderr = Log::create(output.join("stderr.log"))?;
    let environment = BTreeMap::from([(
        "OUTPUT_DIR".into(),
        output.canonicalize()?.display().to_string(),
    )]);
    let outcome = process::execute(process::Request {
        argv,
        cwd: workdir,
        environment: process::Environment::Inherit(&environment),
        stdin: process::Input::Null,
        stdout: process::Capture::File {
            file: &mut stdout.file,
            limit: LOG_LIMIT,
        },
        stderr: process::Capture::File {
            file: &mut stderr.file,
            limit: LOG_LIMIT,
        },
        deadline: Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| io::Error::other("invalid native execution deadline"))?,
        termination_grace: Duration::from_secs(2),
        drain_grace: Duration::from_millis(250),
        cancelled,
    })?;
    let mut result = Execution {
        contract: "rust-local-execution-preview.v1",
        exit_code: outcome.exit_code,
        termination_signal: outcome.termination_signal,
        timed_out: outcome.timed_out,
        cancelled: outcome.cancelled,
        error: outcome.failure.map(|failure| crate::Failure {
            stage: failure.stage,
            code: failure.code,
        }),
        stdout_sha256: None,
        stderr_sha256: None,
    };
    if outcome.stdout.truncated || outcome.stderr.truncated {
        result.fail("LOG_CAPTURE", "OUTPUT_LIMIT_EXCEEDED");
    }
    for (log, field, captured_bytes, captured_sha256) in [
        (
            &mut stdout,
            &mut result.stdout_sha256,
            outcome.stdout.retained_bytes,
            outcome.stdout.prefix_sha256.as_str(),
        ),
        (
            &mut stderr,
            &mut result.stderr_sha256,
            outcome.stderr.retained_bytes,
            outcome.stderr.prefix_sha256.as_str(),
        ),
    ] {
        match log.hash(captured_bytes, captured_sha256) {
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
