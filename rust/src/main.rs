use experiment_redact::redact_lines;
use std::env;
use std::io::{self, Read};
use std::process::ExitCode;

const HELP: &str = "Redact credentials in experiment log text from standard input.\n\nUsage: experiment-redact\n\nOptions:\n  -h, --help      Print help";

fn run() -> Result<(), &'static str> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments == ["-h"] || arguments == ["--help"] {
        println!("{HELP}");
        return Ok(());
    }
    if !arguments.is_empty() {
        return Err("experiment-redact: unexpected arguments; input suppressed");
    }
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|_| "experiment-redact: could not read standard input")?;
    print!("{}", redact_lines(&input));
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}
