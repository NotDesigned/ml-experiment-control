//! Native process supervision, managed execution and backend worker protocols.
use ml_exp_io::write_atomic;
use serde::Serialize;
use std::{io, path::Path, sync::atomic::AtomicBool, time::Duration};

#[cfg(unix)]
pub mod cloud;
pub mod cloud_artifacts;
#[cfg(unix)]
pub mod cloud_data;
#[cfg(unix)]
pub mod cloud_exchange;
#[cfg(unix)]
pub mod data_import;
#[cfg(unix)]
pub mod datasets;
#[cfg(unix)]
pub mod managed;
#[cfg(unix)]
mod process;
#[cfg(unix)]
pub mod slurm;

#[derive(Debug, Serialize)]
pub struct Failure {
    pub stage: &'static str,
    pub code: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Execution {
    pub contract: &'static str,
    pub exit_code: Option<i32>,
    pub termination_signal: Option<i32>,
    pub timed_out: bool,
    pub cancelled: bool,
    pub error: Option<Failure>,
    pub stdout_sha256: Option<String>,
    pub stderr_sha256: Option<String>,
}

impl Execution {
    pub fn succeeded(&self) -> bool {
        self.exit_code == Some(0) && !self.timed_out && !self.cancelled && self.error.is_none()
    }

    #[cfg(unix)]
    fn fail(&mut self, stage: &'static str, code: &'static str) {
        if self.error.is_none() {
            self.error = Some(Failure { stage, code });
        }
    }
}

/// Execute once; no automatic retry or reuse of an existing output directory.
pub fn execute(
    workdir: &Path,
    output: &Path,
    argv: &[String],
    timeout: Duration,
) -> io::Result<Execution> {
    execute_with_cancel(workdir, output, argv, timeout, &AtomicBool::new(false))
}

/// The caller owns signal handling. The CLI sets this flag for SIGTERM/SIGINT;
/// embedding the library never installs process-wide handlers implicitly.
pub fn execute_with_cancel(
    workdir: &Path,
    output: &Path,
    argv: &[String],
    timeout: Duration,
    cancelled: &AtomicBool,
) -> io::Result<Execution> {
    if argv.is_empty() || argv[0].is_empty() || timeout.is_zero() || !workdir.is_dir() {
        return Err(io::Error::other("invalid native execution request"));
    }
    #[cfg(unix)]
    {
        let result = process::execute(workdir, output, argv, timeout, cancelled)?;
        write_atomic(
            &output.join("execution.json"),
            &serde_json::to_vec_pretty(&result)?,
            true,
        )?;
        Ok(result)
    }
    #[cfg(not(unix))]
    {
        let _ = (output, cancelled);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native process supervision currently requires Unix",
        ))
    }
}

#[cfg(all(test, unix))]
mod tests;

#[cfg(unix)]
pub mod materialize;

#[cfg(unix)]
pub mod maintenance;
