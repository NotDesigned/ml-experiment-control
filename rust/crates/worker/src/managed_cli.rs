//! Operator-only CLI for a private, locally staged sealed launch configuration.
use clap::Parser;
use ml_exp_worker::managed;
use std::{
    ffi::OsString,
    path::PathBuf,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
};

#[derive(Parser)]
#[command(
    name = "managed-launch",
    about = "Execute one sealed local Attempt; remote fetch and managed data/publication are unavailable"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long, conflicts_with = "inspect")]
    initialize: bool,
    /// Bootstrap a fresh scoped worker and execute it once, never adopt state.
    #[arg(long, conflicts_with_all = ["initialize", "inspect"])]
    initialize_and_run: bool,
    #[arg(long)]
    inspect: bool,
    /// Original job-script binding of the locally staged configuration bytes.
    #[arg(long)]
    sha256: Option<String>,
}
pub fn main_from(arguments: impl IntoIterator<Item = OsString>) -> ExitCode {
    let args = Args::parse_from(arguments);
    let result = (|| {
        let config = managed::load_config(&args.config, args.sha256.as_deref())?;
        if args.initialize || args.initialize_and_run {
            let mut runtime = std::collections::BTreeMap::new();
            for key in &config.runtime_environment {
                if let Some(value) = std::env::var_os(key) {
                    runtime.insert(
                        key.clone(),
                        value
                            .into_string()
                            .map_err(|_| managed::Error::Configuration)?,
                    );
                }
            }
            let prepared = managed::initialize_with_environment(&config, &runtime)?;
            if args.initialize {
                return Ok(prepared);
            }
        }
        if args.inspect {
            return managed::inspect(&config);
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
            signal_hook::flag::register(signal, Arc::clone(&cancelled))
                .map_err(|_| managed::Error::Storage)?;
        }
        managed::run(&config, &cancelled)
    })();
    match result {
        Ok(value) => {
            println!(
                "{}",
                serde_json::to_string(&value).expect("managed observation JSON")
            );
            if args.initialize
                || args.inspect
                || value
                    .receipt
                    .as_ref()
                    .is_some_and(managed::Receipt::succeeded)
            {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("{error}; no automatic retry attempted");
            ExitCode::FAILURE
        }
    }
}
