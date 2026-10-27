//! Trusted local shell for the native Slurm helper, separate from RPC data.
#![forbid(unsafe_code)]
use clap::{ArgGroup, Parser, error::ErrorKind};
use ml_exp_core::slurm::helper::{self, Target};
use ml_exp_io::framing;
use ml_exp_worker::slurm::{Helper, HelperConfig};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    io,
    path::PathBuf,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(name="ml-exp-worker-rs slurm-helper",about="Scoped native Slurm metadata helper; explicit initialization is separate from RPC",group(ArgGroup::new("mode").required(true).args(["stdio","initialize"]).multiple(false)))]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    stdio: bool,
    #[arg(long)]
    initialize: bool,
}
// Deliberately no Debug: environment can contain operator credentials. This
// document is never deserialized from a request frame and never printed.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    root: PathBuf,
    target: Target,
    sbatch_program: PathBuf,
    environment: BTreeMap<String, String>,
    lock_timeout_seconds: u64,
    execution_timeout_seconds: u64,
    stdio_timeout_seconds: u64,
}
impl Config {
    fn load(path: &std::path::Path) -> io::Result<Self> {
        if !path.is_absolute() {
            return Err(io::Error::other("invalid helper configuration"));
        }
        let bytes = ml_exp_io::read_private(path, 128 * 1024, true)?;
        let value: Self = serde_json::from_slice(&bytes)
            .map_err(|_| io::Error::other("invalid helper configuration"))?;
        if !(1..=30).contains(&value.stdio_timeout_seconds) {
            return Err(io::Error::other("invalid helper stdio deadline"));
        }
        Ok(value)
    }
    fn helper(self) -> (HelperConfig, Duration) {
        (
            HelperConfig {
                root: self.root,
                target: self.target,
                sbatch_program: self.sbatch_program,
                environment: self.environment,
                lock_timeout: Duration::from_secs(self.lock_timeout_seconds),
                execution_timeout: Duration::from_secs(self.execution_timeout_seconds),
            },
            Duration::from_secs(self.stdio_timeout_seconds),
        )
    }
}
fn cancelled() -> io::Result<Arc<AtomicBool>> {
    let flag = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(signal, Arc::clone(&flag))?;
    }
    Ok(flag)
}
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let (config, timeout) = Config::load(&args.config)?.helper();
    config.validate()?;
    if args.initialize {
        Helper::initialize(config)?;
        return Ok(());
    }
    let cancelled = cancelled()?;
    let frame = framing::read_frame(
        std::io::stdin(),
        helper::MAX_METADATA_BYTES,
        Instant::now() + timeout,
        &cancelled,
    )?;
    let request = helper::decode_request(&frame)?;
    let helper = Helper::open(config)?;
    let reply = helper.handle(&request, &cancelled)?;
    let frame = helper::encode_reply(&reply)?;
    framing::write_frame(
        std::io::stdout(),
        &frame,
        helper::MAX_METADATA_BYTES,
        Instant::now() + timeout,
        &cancelled,
    )?;
    Ok(())
}
pub(crate) fn main_from(args: impl IntoIterator<Item = OsString>) -> ExitCode {
    let args = match Args::try_parse_from(args) {
        Ok(args) => args,
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) =>
        {
            let _ = error.print();
            return ExitCode::SUCCESS;
        }
        Err(_) => {
            eprintln!("invalid native Slurm helper arguments");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => {
            eprintln!("native Slurm helper failed; state may be uncertain; no retry attempted");
            ExitCode::FAILURE
        }
    }
}
