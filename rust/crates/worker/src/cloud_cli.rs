use clap::Parser;
use std::{
    ffi::OsString,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
};
#[derive(Parser)]
#[command(
    name = "ml-exp-worker-rs cloud-launch",
    about = "Start or inspect one cloud Attempt from its saved launch file"
)]
struct Args {
    /// Read launch metadata through the existing Attempt exchange.
    #[arg(long)]
    package_url: String,
    #[arg(long, hide = true)]
    package_token: String,
    #[arg(long)]
    package_sha256: String,
    #[arg(long)]
    inspect: bool,
    /// Retry only retained metadata delivery; never start or initialize training.
    #[arg(long, conflicts_with = "inspect")]
    sync: bool,
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
        let bytes = ml_exp_worker::cloud::download_raw(
            &args.package_url,
            &args.package_token,
            &args.package_sha256,
        )?;
        // Before the package is decoded: a later server's may not decode in this worker.
        ml_exp_worker::cloud_bootstrap::enter(&bytes, &args.package_url, &args.package_token)?;
        let package = ml_exp_worker::cloud::scoped(&args.package_url, &args.package_token, &bytes)?;
        if ml_exp_worker::cloud_bootstrap::running()? != package.worker_sha256 {
            return Err(ml_exp_worker::managed::Error::Binding);
        }
        if args.inspect {
            return ml_exp_worker::managed::inspect(&package.config);
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
            signal_hook::flag::register(signal, Arc::clone(&cancelled))
                .map_err(|_| ml_exp_worker::managed::Error::Configuration)?;
        }
        if args.sync {
            let drained = ml_exp_worker::cloud_exchange::drain(&package, &cancelled)?;
            // A preempted worker's registered checkpoints are copied here too: the drain
            // above has registered its last ready generation, if it had one.
            if package.checkpoint_archive {
                match ml_exp_worker::cloud_checkpoints::archive(&package, &cancelled) {
                    Ok(true) => {}
                    Ok(false) => eprintln!("ML_EXPD_CHECKPOINT_ARCHIVE=INCOMPLETE"),
                    Err(error) => eprintln!("ML_EXPD_CHECKPOINT_ARCHIVE=DEFERRED {error}"),
                }
            }
            if !drained {
                // A worker that never reported its end, as after a preemption, sends no
                // more. This sync succeeds only when a packet built from storage as it
                // is now was acknowledged and every registered checkpoint has a verified
                // copy; a purge of an unfinished Attempt requires such a sync.
                let observed = ml_exp_worker::managed::sample(&package.config)?;
                if package.checkpoint_archive
                    && observed.phase != ml_exp_worker::managed::Phase::Finished
                    && ml_exp_worker::cloud_exchange::deliver_current(&package, &cancelled)?
                    && ml_exp_worker::cloud_checkpoints::archive(&package, &cancelled)?
                    && ml_exp_worker::cloud_checkpoints::report_synced(&package)?
                {
                    eprintln!("ML_EXPD_SYNC=UNFINISHED_WORKER_DELIVERED");
                    return Ok(observed);
                }
                return Err(ml_exp_worker::managed::Error::Uncertain);
            }
            ml_exp_worker::cloud_artifacts::publish(&package)?;
            return ml_exp_worker::managed::sample(&package.config);
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
            if receipt.succeeded() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        // An inspection or a collector is not the launch, so neither reports one.
        Err(error) if args.inspect || args.sync => failed(&error, |_| None),
        Err(error) => failed(&error, |text| {
            ml_exp_worker::cloud_startup::report(&args.package_url, &args.package_token, text)
        }),
    }
}
/// The exit status is the launch's own, whatever became of its report.
fn failed(
    error: &ml_exp_worker::managed::Error,
    report: impl FnOnce(&str) -> Option<ml_exp_worker::cloud_startup::Delivery>,
) -> ExitCode {
    eprintln!("cloud launch failed: {error}");
    if let Some(delivery) = report(&error.to_string()) {
        eprintln!("ML_EXPD_STARTUP_FAILURE_REPORT={delivery}");
    }
    ExitCode::FAILURE
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_worker::{cloud_startup::Delivery, managed::Error};

    #[test]
    fn a_failed_launch_reports_its_error_and_still_fails() {
        for delivery in [Some(Delivery::Recorded), Some(Delivery::Failed(None)), None] {
            let mut sent = Vec::new();
            let status = failed(
                &Error::Storage {
                    at: "crates/worker/src/cloud.rs:123",
                    cause: "PermissionDenied".into(),
                },
                |text| {
                    sent.push(text.to_owned());
                    delivery
                },
            );
            assert_eq!(status, ExitCode::FAILURE);
            assert_eq!(
                sent,
                [
                    "managed launch storage failed at crates/worker/src/cloud.rs:123: PermissionDenied"
                ]
            );
        }
    }
}
