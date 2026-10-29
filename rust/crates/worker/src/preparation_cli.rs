use clap::Parser;
use std::{
    ffi::OsString,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
};
#[derive(Parser)]
#[command(
    name = "script-prepare",
    about = "Prepare one frozen persistent input cache; never run training"
)]
struct Args {
    #[arg(long)]
    plan_json: String,
}
pub fn main_from(arguments: impl IntoIterator<Item = OsString>) -> ExitCode {
    let result = (|| -> ml_exp_worker::managed::Result<_> {
        let args = Args::try_parse_from(arguments)
            .map_err(|_| ml_exp_worker::managed::Error::Configuration)?;
        if args.plan_json.len() > 64 * 1024 {
            return Err(ml_exp_worker::managed::Error::Configuration);
        }
        let value = ml_exp_core::json::mapping(args.plan_json.as_bytes())
            .map_err(|_| ml_exp_worker::managed::Error::Configuration)?;
        let plan: ml_exp_core::data_preparation::Plan =
            serde_json::from_value(serde_json::Value::Object(value))
                .map_err(|_| ml_exp_worker::managed::Error::Configuration)?;
        let cancelled = Arc::new(AtomicBool::new(false));
        for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
            signal_hook::flag::register(signal, Arc::clone(&cancelled))
                .map_err(|_| ml_exp_worker::managed::Error::Configuration)?;
        }
        ml_exp_worker::data_preparation::prepare(&plan, true, &cancelled)
    })();
    match result {
        Ok(_) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
