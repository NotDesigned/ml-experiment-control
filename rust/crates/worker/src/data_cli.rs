//! Explicit dataset helper bootstrap and bounded part ingress.
use clap::{ArgGroup, Parser};
use ml_exp_core::datasets;
use ml_exp_io::framing;
use ml_exp_worker::datasets::{Config, Store};
use std::{
    ffi::OsString,
    io,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};
#[derive(Parser)]
#[command(name="ml-exp-worker-rs data-helper",group(ArgGroup::new("mode").required(true).args(["stdio","initialize"]).multiple(false)))]
struct Args {
    #[arg(long)]
    config: std::path::PathBuf,
    #[arg(long)]
    stdio: bool,
    #[arg(long)]
    initialize: bool,
}
fn run(args: Args) -> io::Result<()> {
    let bytes = ml_exp_io::read_private(&args.config, 128 * 1024, true)?;
    let mapping = ml_exp_core::json::mapping(&bytes)
        .map_err(|_| io::Error::other("invalid private dataset configuration"))?;
    let config: Config = serde_json::from_value(serde_json::Value::Object(mapping))
        .map_err(|_| io::Error::other("invalid private dataset configuration"))?;
    if args.initialize {
        return Store::initialize(config);
    }
    let flag = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(signal, Arc::clone(&flag))?;
    }
    let deadline = Instant::now() + Duration::from_secs(90);
    let input = io::stdin();
    let frame = framing::read_prefix(&input, datasets::MAX_METADATA, deadline, &flag)?;
    let request = datasets::decode_request(&frame)
        .map_err(|_| io::Error::other("invalid dataset request"))?;
    let count = match request.command {
        datasets::Command::Part { bytes, .. } => bytes as usize,
        _ => 0,
    };
    let mut payload = vec![0u8; count];
    let mut read = 0;
    while read < count {
        let maximum = (count - read).min(65536);
        let n = framing::read_payload(&input, &mut payload[read..read + maximum], deadline, &flag)?;
        if n == 0 {
            return Err(io::Error::other("truncated dataset part"));
        }
        read += n;
    }
    if framing::read_payload(&input, &mut [0], deadline, &flag)? != 0 {
        return Err(io::Error::other("unexpected dataset payload bytes"));
    }
    let reply = Store::open(config)?.handle(&request, &payload, &flag)?;
    let frame =
        datasets::encode_reply(&reply).map_err(|_| io::Error::other("invalid dataset receipt"))?;
    framing::write_frame(
        io::stdout(),
        &frame,
        datasets::MAX_METADATA,
        Instant::now() + Duration::from_secs(90),
        &flag,
    )
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
            eprintln!("invalid dataset helper arguments");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => {
            eprintln!(
                "dataset delivery is unconfirmed; inspect the same asset before retry; no scheduler operation was issued"
            );
            ExitCode::FAILURE
        }
    }
}
