use clap::Parser;
use std::{path::PathBuf, process::ExitCode, sync::Arc, time::Duration};
#[cfg(unix)]
mod cli;

#[derive(Parser)]
#[command(
    name = "ml-exp-worker-rs",
    version,
    about = "Native local process preview; not a replacement for frozen managed workers",
    after_help = "Slurm metadata: ml-exp-worker-rs slurm-helper --config /absolute/private.json --stdio\nExplicit bootstrap: replace --stdio with --initialize. See slurm-helper --help."
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
        if arguments
            .next()
            .is_some_and(|value| value == "slurm-helper")
        {
            return cli::main_from(std::iter::once(binary).chain(arguments));
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
