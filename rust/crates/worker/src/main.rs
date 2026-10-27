use clap::Parser;
use std::{path::PathBuf, process::ExitCode, sync::Arc, time::Duration};

#[derive(Parser)]
#[command(
    name = "ml-exp-worker-rs",
    version,
    about = "Native local process preview; not a replacement for frozen managed workers"
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
