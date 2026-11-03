use clap::Parser;
use std::{ffi::OsString, io::Read, path::PathBuf, process::ExitCode};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
}
pub fn main_from(args: impl IntoIterator<Item = OsString>) -> ExitCode {
    let args = match Args::try_parse_from(args) {
        Ok(a) => a,
        Err(e) => {
            let _ = e.print();
            return ExitCode::from(2);
        }
    };
    let result = (|| -> std::io::Result<_> {
        let config = ml_exp_io::read_private(&args.config, 65536, true)?;
        let config = serde_json::from_slice(&config)?;
        let mut bytes = Vec::new();
        std::io::stdin().take(65537).read_to_end(&mut bytes)?;
        if bytes.len() > 65536 {
            return Err(std::io::Error::other("materializer request too large"));
        }
        let request = serde_json::from_value(serde_json::Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(std::io::Error::other)?,
        ))?;
        ml_exp_worker::materialize::run(config, request)
    })();
    match result {
        Ok(value) => {
            println!("{value}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("native OCI conversion failed: {error}");
            ExitCode::FAILURE
        }
    }
}
