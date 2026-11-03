//! Persistent bounded SDK pipe. State and replay decisions live in Rust core.
use super::{Error, Result};
use ml_exp_io::framing;
use serde_json::Value;
use std::{
    os::unix::process::CommandExt,
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};
const BOUND: usize = 8 * 1024 * 1024;
pub(super) struct Sdk {
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
    _directory: tempfile::TempDir,
}
impl Sdk {
    pub(super) fn validate_request(request: &Value) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec(request).map_err(|_| Error::Transport)?;
        if bytes.is_empty() || bytes.len() > BOUND {
            return Err(Error::Transport);
        }
        Ok(bytes)
    }
    pub(super) fn open(python: &Path) -> Result<Self> {
        if !python.is_absolute() {
            return Err(Error::Configuration);
        }
        let directory = tempfile::tempdir().map_err(|_| Error::Transport)?;
        let mut command = Command::new(python);
        command
            .args(["-I", "-u", "-c", include_str!("sdk.py")])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .process_group(0);
        command
            .env_clear()
            .env("PATH", "/usr/local/bin:/usr/bin:/bin")
            .env("HOME", directory.path())
            .current_dir(directory.path())
            .env("WANDB_CONFIG_DIR", directory.path())
            .env("WANDB_CACHE_DIR", directory.path())
            .env("LANG", "C.UTF-8")
            .env("WANDB_BASE_URL", "https://api.wandb.ai")
            .env("WANDB_ERROR_REPORTING", "false");
        let mut child = command.spawn().map_err(|_| Error::Transport)?;
        let input = child.stdin.take().ok_or(Error::Transport)?;
        let output = child.stdout.take().ok_or(Error::Transport)?;
        Ok(Self {
            child,
            input,
            output,
            _directory: directory,
        })
    }
    pub(super) fn call(&mut self, request: &Value, cancelled: &AtomicBool) -> Result<Value> {
        let bytes = Self::validate_request(request)?;
        let mut frame = Vec::with_capacity(bytes.len() + 4);
        frame.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        frame.extend_from_slice(&bytes);
        let deadline = Instant::now() + Duration::from_secs(90);
        framing::write_frame(&self.input, &frame, BOUND, deadline, cancelled)
            .map_err(|_| Error::Transport)?;
        let frame = framing::read_prefix(&self.output, BOUND, deadline, cancelled)
            .map_err(|_| Error::Transport)?;
        let value =
            Value::Object(ml_exp_core::json::mapping(&frame[4..]).map_err(|_| Error::Transport)?);
        match value["error"].as_str() {
            Some("WANDB_SDK_NOT_INSTALLED") => Err(Error::SdkMissing),
            Some(code) => match code {
                "WANDB_AUTHENTICATION_FAILED" => {
                    Err(Error::SdkRequest("WANDB_AUTHENTICATION_FAILED"))
                }
                "WANDB_PERMISSION_DENIED" => Err(Error::SdkRequest("WANDB_PERMISSION_DENIED")),
                "WANDB_REMOTE_RUN_MISSING" => Err(Error::SdkRequest("WANDB_REMOTE_RUN_MISSING")),
                "WANDB_INIT_FAILED" => Err(Error::SdkRequest("WANDB_INIT_FAILED")),
                "WANDB_HISTORY_READ_FAILED" => Err(Error::SdkRequest("WANDB_HISTORY_READ_FAILED")),
                "WANDB_REMOTE_READ_FAILED" => Err(Error::SdkRequest("WANDB_REMOTE_READ_FAILED")),
                "WANDB_WRITE_FAILED" => Err(Error::SdkRequest("WANDB_WRITE_FAILED")),
                "WANDB_DEFAULT_ENTITY_FAILED" => {
                    Err(Error::SdkRequest("WANDB_DEFAULT_ENTITY_FAILED"))
                }
                _ => Err(Error::Transport),
            },
            None => Ok(value),
        }
    }
}
impl Drop for Sdk {
    fn drop(&mut self) {
        // The leader remains unreaped and process_group(0) pins its group ID.
        // This also terminates SDK descendants; a timeout cannot leave one
        // unbounded SDK service process for each failed call.
        if let Ok(pid) = i32::try_from(self.child.id()) {
            // SAFETY: the owned unreaped child was placed in its own group.
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
        let _ = self.child.wait();
    }
}
