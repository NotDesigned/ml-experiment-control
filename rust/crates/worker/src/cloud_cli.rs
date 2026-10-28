use clap::Parser;
use std::{
    ffi::OsString,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
};
#[derive(Parser)]
#[command(
    name = "ml-exp-worker-rs cloud-launch",
    about = "Start or inspect one digest-bound cloud Attempt without network bootstrap"
)]
struct Args {
    #[arg(long)]
    package_json: String,
    #[arg(long)]
    inspect: bool,
}
pub(crate) fn main_from(args: impl IntoIterator<Item = OsString>) -> ExitCode {
    let args = match Args::try_parse_from(args) {
        Ok(a) => a,
        Err(e)
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            let _ = e.print();
            return ExitCode::SUCCESS;
        }
        Err(_) => {
            eprintln!("invalid cloud worker arguments");
            return ExitCode::from(2);
        }
    };
    let work = || -> ml_exp_worker::managed::Result<_> {
        let package = ml_exp_worker::cloud::decode(args.package_json.as_bytes())?;
        let path =
            std::env::current_exe().map_err(|_| ml_exp_worker::managed::Error::Configuration)?;
        ml_exp_worker::cloud::verify_binary(&path, &package.worker_sha256)?;
        if args.inspect {
            return ml_exp_worker::managed::inspect(&package.config);
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
            signal_hook::flag::register(signal, Arc::clone(&cancelled))
                .map_err(|_| ml_exp_worker::managed::Error::Configuration)?;
        }
        let runtime = package
            .config
            .runtime_environment
            .iter()
            .filter_map(|name| std::env::var(name).ok().map(|v| (name.clone(), v)))
            .collect();
        ml_exp_worker::cloud::run(&package, &runtime, &cancelled)
    };
    match work() {
        Ok(receipt) => {
            println!(
                "ML_EXPD_CLOUD_WORKER={}",
                serde_json::to_string(&receipt).expect("worker receipt")
            );
            if receipt
                .receipt
                .as_ref()
                .is_some_and(|r| r.training == ml_exp_core::managed::evidence::Training::Completed)
            {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("cloud launch failed: {error}");
            ExitCode::FAILURE
        }
    }
}
