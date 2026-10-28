use clap::Parser;
use ml_exp_worker::data_import;
use std::{
    ffi::OsString,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};
#[derive(Parser)]
#[command(
    name = "ml-exp-worker-rs data-import",
    about = "Verify and cache a private image-file or approved HTTPS dataset on backend storage"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    initialize: bool,
}
struct Timer {
    stop: std::sync::mpsc::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Timer {
    fn start(flag: Arc<AtomicBool>, seconds: u64) -> io::Result<Self> {
        let (stop, receiver) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("dataset-deadline".into())
            .spawn(move || {
                if matches!(
                    receiver.recv_timeout(std::time::Duration::from_secs(seconds)),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                ) {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            })?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
fn run(args: Args) -> Result<(), data_import::Error> {
    let config = data_import::load(&args.config)?;
    if args.initialize {
        return data_import::initialize(&config);
    }
    let flag = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(signal, Arc::clone(&flag)).map_err(|_| data_import::Error {
            code: "SIGNAL_HANDLER_UNAVAILABLE",
            http_status: None,
            errno: None,
        })?;
    }
    let _timer = Timer::start(Arc::clone(&flag), config.timeout_seconds).map_err(|_| {
        data_import::Error {
            code: "DEADLINE_HANDLER_UNAVAILABLE",
            http_status: None,
            errno: None,
        }
    })?;
    let start = Instant::now();
    let mut baseline = None;
    let mut output = io::stdout().lock();
    let receipt = data_import::materialize(&config, &flag, &mut |progress| {
        let baseline = *baseline.get_or_insert(progress.received_bytes);
        let elapsed = start.elapsed().as_secs_f64();
        let mut value = serde_json::to_value(progress)?;
        value["phase"] = serde_json::to_value(progress.stage)?;
        value
            .as_object_mut()
            .expect("progress object")
            .remove("stage");
        value["at"] = serde_json::json!(
            time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .map_err(io::Error::other)?
        );
        value["expected_bytes"] = serde_json::json!(progress.total_bytes);
        value["elapsed_seconds"] = serde_json::json!(elapsed);
        value["bytes_per_second"] = serde_json::json!(if elapsed > 0. {
            progress.received_bytes.saturating_sub(baseline) as f64 / elapsed
        } else {
            0.
        });
        value["code"] = serde_json::json!(progress.error_code);
        writeln!(output, "{}{}", ml_exp_core::input_progress::MARKER, value)?;
        output.flush()
    })?;
    writeln!(
        output,
        "ML_EXPD_DATA_READY={}",
        serde_json::to_string(&receipt).map_err(|_| data_import::Error {
            code: "RECEIPT_ENCODING_FAILED",
            http_status: None,
            errno: None
        })?
    )
    .map_err(|_| data_import::Error {
        code: "RECEIPT_OUTPUT_UNAVAILABLE",
        http_status: None,
        errno: None,
    })?;
    Ok(())
}
pub(crate) fn main_from(args: impl IntoIterator<Item = OsString>) -> ExitCode {
    let args = match Args::try_parse_from(args) {
        Ok(args) => args,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            let _ = error.print();
            return ExitCode::SUCCESS;
        }
        Err(_) => {
            eprintln!("invalid private dataset import arguments");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
