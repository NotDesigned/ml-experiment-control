use clap::Parser;
use ml_exp_core::slurm::delivery::{CODE_ABSENT, CODE_READY, CODE_REFUSED, CODE_REUSED};
use ml_exp_worker::managed::Error;
use std::{ffi::OsString, path::PathBuf, process::ExitCode};
#[derive(Parser)]
#[command(
    name = "code-receive",
    about = "Check, or receive on standard input, one code source on a cluster's shared storage; never run training"
)]
struct Args {
    /// The directory that holds every delivered code source, each named by its identity.
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    source: String,
    /// Only say whether the source is there and proven, hashing every file.
    #[arg(long)]
    check: bool,
}
pub fn main_from(arguments: impl IntoIterator<Item = OsString>) -> ExitCode {
    let result = (|| -> ml_exp_worker::managed::Result<&'static str> {
        let args = Args::try_parse_from(arguments).map_err(|_| Error::Configuration)?;
        if !ml_exp_core::managed::absolute(&args.root) {
            return Err(Error::Configuration);
        }
        if args.check {
            return Ok(
                if ml_exp_worker::cloud_code::check_shared(&args.root, &args.source)? {
                    CODE_READY
                } else {
                    CODE_ABSENT
                },
            );
        }
        let reused = ml_exp_worker::cloud_code::receive_shared(
            &args.root,
            &args.source,
            &mut std::io::stdin().lock(),
        )?;
        Ok(if reused { CODE_REUSED } else { CODE_READY })
    })();
    match result {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            println!("{CODE_REFUSED}");
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
