use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
#[command(
    name = "ml-expd-state-audit",
    version,
    about = "Read existing frozen metadata and journals without initializing or changing them"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// Optional existing Action root; no Action command is executed.
    #[arg(long)]
    action_root: Option<PathBuf>,
    /// A private snapshot from a previous audit, never execution authority.
    #[arg(long)]
    compare: Option<PathBuf>,
}
fn run(args: Args) -> Result<bool, &'static str> {
    let settings = ml_exp_server::read_config::load(&args.config)
        .map_err(|_| "invalid private audit configuration")?;
    let snapshot = ml_exp_server::state_audit::capture(&settings, args.action_root.as_deref())
        .map_err(|error| error.code())?;
    let (value, success) = if let Some(path) = args.compare {
        let bytes = ml_exp_io::read_private(&path, 32 * 1024 * 1024, true)
            .map_err(|_| "invalid private comparison snapshot")?;
        let baseline: ml_exp_core::state_audit::Snapshot =
            serde_json::from_value(serde_json::Value::Object(
                ml_exp_core::json::mapping(&bytes).map_err(|_| "invalid comparison snapshot")?,
            ))
            .map_err(|_| "invalid comparison snapshot")?;
        let result = ml_exp_core::state_audit::compare(&baseline, &snapshot)?;
        let success = result.equivalent;
        (
            serde_json::to_value(result).map_err(|_| "audit output")?,
            success,
        )
    } else {
        let success = snapshot.issues.is_empty();
        (
            serde_json::to_value(snapshot).map_err(|_| "audit output")?,
            success,
        )
    };
    let text = serde_json::to_string_pretty(&value).map_err(|_| "audit output")?;
    if text.len() > 32 * 1024 * 1024 {
        return Err("audit output exceeds the 32 MiB metadata limit");
    }
    println!("{text}");
    Ok(success)
}
fn main() -> std::process::ExitCode {
    match run(Args::parse()) {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::from(2),
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
