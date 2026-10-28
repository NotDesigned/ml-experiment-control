//! One authenticated SSH stream, bounded channel and zero API-host disk spool.
//! Download failures never mutate a scheduler or authorize release.
use super::SshConfig;
use axum::body::Bytes;
use ml_exp_core::{
    artifacts::publication::State,
    slurm::helper::{self, ReplyOutcome, Request},
};
use ml_exp_io::framing;
use sha2::{Digest, Sha256};
use std::{
    io,
    process::{Child, ChildStdout, Command, Stdio},
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
fn invalid() -> io::Error {
    io::Error::other("backend artifact stream unavailable or changed")
}
pub struct Download {
    child: Child,
    stdout: ChildStdout,
    expected: State,
    deadline: Instant,
}
impl Drop for Download {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}
impl Download {
    pub fn open(config: &SshConfig, request: &Request, expected: &State) -> io::Result<Self> {
        config.validate().map_err(|_| invalid())?;
        request.validate().map_err(|_| invalid())?;
        if config.target != request.intent().target
            || !matches!(
                request.artifact_command(),
                Some(ml_exp_core::artifacts::publication::Command::Download { .. })
            )
        {
            return Err(invalid());
        }
        let mut argv = config.argv();
        let flag = argv
            .iter_mut()
            .find(|v| v.as_str() == "--stdio")
            .ok_or_else(invalid)?;
        *flag = "--artifact-stream".into();
        let mut child = Command::new(&argv[0])
            .args(&argv[1..])
            .current_dir(&config.local_cwd)
            .env_clear()
            .envs(config.environment())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdout = child.stdout.take().ok_or_else(invalid)?;
        let mut download = Self {
            child,
            stdout,
            expected: expected.clone(),
            deadline: Instant::now() + Duration::from_secs(3600),
        };
        let deadline = Instant::now() + config.budget.timeout.min(Duration::from_secs(30));
        let flag = AtomicBool::new(false);
        let frame = helper::encode_request(request).map_err(|_| invalid())?;
        let stdin = download.child.stdin.take().ok_or_else(invalid)?;
        framing::write_frame(&stdin, &frame, helper::MAX_METADATA_BYTES, deadline, &flag)?;
        drop(stdin);
        let frame = framing::read_prefix(
            &download.stdout,
            helper::MAX_METADATA_BYTES,
            deadline,
            &flag,
        )?;
        let reply = helper::decode_reply(&frame, request).map_err(|_| invalid())?;
        let ReplyOutcome::Artifacts { publication: state } = reply.outcome else {
            return Err(invalid());
        };
        if state.bundle != expected.bundle || !state.available().map_err(|_| invalid())? {
            return Err(invalid());
        }
        Ok(download)
    }
    fn send(
        &self,
        sender: &mpsc::Sender<io::Result<Bytes>>,
        mut item: io::Result<Bytes>,
    ) -> io::Result<()> {
        loop {
            if Instant::now() >= self.deadline {
                return Err(invalid());
            }
            match sender.try_send(item) {
                Ok(()) => return Ok(()),
                Err(mpsc::error::TrySendError::Closed(_)) => return Err(invalid()),
                Err(mpsc::error::TrySendError::Full(back)) => {
                    item = back;
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
    pub fn pump(mut self, sender: mpsc::Sender<io::Result<Bytes>>) -> io::Result<()> {
        let flag = AtomicBool::new(false);
        let mut left = self.expected.bundle.archive.bytes;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        let result = (|| {
            while left > 0 {
                if sender.is_closed() {
                    return Err(invalid());
                }
                let count = framing::read_payload(
                    &self.stdout,
                    &mut buffer[..left.min(65536) as usize],
                    self.deadline,
                    &flag,
                )?;
                if count == 0 {
                    return Err(invalid());
                }
                left -= count as u64;
                hash.update(&buffer[..count]);
                // Keep the final chunk until EOF, digest and exit are all verified.
                if left == 0 {
                    if format!("{:x}", hash.clone().finalize())
                        != self.expected.bundle.archive.sha256
                        || framing::read_payload(&self.stdout, &mut [0], self.deadline, &flag)? != 0
                    {
                        return Err(invalid());
                    }
                    loop {
                        if let Some(status) = self.child.try_wait()? {
                            if !status.success() {
                                return Err(invalid());
                            }
                            break;
                        }
                        if Instant::now() >= self.deadline {
                            return Err(invalid());
                        }
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
                self.send(&sender, Ok(Bytes::copy_from_slice(&buffer[..count])))?;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = self.send(&sender, Err(invalid()));
        }
        result
    }
}
