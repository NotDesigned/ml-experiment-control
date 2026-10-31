use clap::Parser;
use std::{path::PathBuf, process::ExitCode, sync::Arc, time::Duration};
#[cfg(unix)]
mod cli;
#[cfg(unix)]
mod cloud_cli;
#[cfg(unix)]
mod data_cli;
#[cfg(unix)]
mod data_copy_cli;
#[cfg(unix)]
mod import_cli;
#[cfg(unix)]
mod maintenance_cli;
#[cfg(unix)]
mod managed_cli;
#[cfg(unix)]
mod materialize_cli;
#[cfg(unix)]
mod preparation_cli;

#[derive(Parser)]
#[command(
    name = "ml-exp-worker-rs",
    version,
    about = "Native local process preview; not a replacement for frozen managed workers",
    after_help = "Slurm metadata: ml-exp-worker-rs slurm-helper --config /absolute/private.json --stdio\nExplicit bootstrap: replace --stdio with --initialize. See slurm-helper --help.\nSealed local worker: ml-exp-worker-rs managed-launch --config /absolute/private.json [--initialize|--inspect]"
)]
struct Args {
    #[arg(long)]
    workdir: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long,value_parser=clap::value_parser!(u64).range(1..))]
    timeout_seconds: u64,
    #[arg(last = true, required = true)]
    command: Vec<String>,
}

fn run(args: Args) -> std::io::Result<ml_exp_worker::Execution> {
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    #[cfg(unix)]
    {
        for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
            signal_hook::flag::register(signal, Arc::clone(&cancelled))?;
        }
    }
    ml_exp_worker::execute_with_cancel(
        &args.workdir,
        &args.output,
        &args.command,
        Duration::from_secs(args.timeout_seconds),
        &cancelled,
    )
}

fn main() -> ExitCode {
    #[cfg(unix)]
    {
        let mut arguments = std::env::args_os();
        let binary = arguments.next().expect("binary argv");
        match arguments
            .next()
            .as_deref()
            .and_then(std::ffi::OsStr::to_str)
        {
            Some("--capabilities") if arguments.next().is_none() => {
                println!(
                    "{}",
                    serde_json::json!({"contract":"native-worker-capabilities.v1",
                    "features":["storage-layout.v1","script-preparation.v1","shared-cache-identity.v2"]})
                );
                return ExitCode::SUCCESS;
            }
            Some("data-copy") => {
                return data_copy_cli::main_from(std::iter::once(binary).chain(arguments));
            }
            Some("script-prepare") => {
                return preparation_cli::main_from(std::iter::once(binary).chain(arguments));
            }
            Some("storage-maintain") => {
                return maintenance_cli::main_from(std::iter::once(binary).chain(arguments));
            }
            Some("image-materialize") => {
                return materialize_cli::main_from(std::iter::once(binary).chain(arguments));
            }
            Some("cloud-launch") => {
                return cloud_cli::main_from(std::iter::once(binary).chain(arguments));
            }
            Some("data-import") => {
                return import_cli::main_from(std::iter::once(binary).chain(arguments));
            }
            Some("data-helper") => {
                return data_cli::main_from(std::iter::once(binary).chain(arguments));
            }
            Some("slurm-helper") => {
                return cli::main_from(std::iter::once(binary).chain(arguments));
            }
            Some("managed-launch") => {
                return managed_cli::main_from(std::iter::once(binary).chain(arguments));
            }
            _ => {}
        }
    }
    match run(Args::parse()) {
        Ok(result) => {
            println!(
                "{}",
                serde_json::to_string(&result).expect("execution is JSON")
            );
            if result.succeeded() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(_) => {
            eprintln!("native execution setup or receipt publication failed; no retry attempted");
            ExitCode::FAILURE
        }
    }
}
