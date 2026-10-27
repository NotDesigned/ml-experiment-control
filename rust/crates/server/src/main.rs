use clap::Parser;
use std::{net::SocketAddr, path::PathBuf};

#[derive(Parser)]
#[command(
    name = "ml-expd-rs",
    version,
    about = "Isolated Rust HTTP preview; provider mutations and production state are unavailable"
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:18765")]
    bind: SocketAddr,
    #[arg(long)]
    #[arg(required_unless_present = "initialize_registry")]
    token_file: Option<PathBuf>,
    /// Private read-only configuration for an isolated Rust preview registry.
    #[arg(long, conflicts_with = "initialize_registry")]
    read_config: Option<PathBuf>,
    /// Explicitly initialize a fresh registry; existing targets are refused.
    #[arg(long, conflicts_with_all = ["token_file", "read_config"])]
    initialize_registry: Option<PathBuf>,
}
#[tokio::main]
async fn main() -> std::process::ExitCode {
    let args = Args::parse();
    if let Some(path) = &args.initialize_registry {
        return match ml_exp_server::read_config::initialize(path) {
            Ok(workspace) => {
                println!("{workspace}");
                std::process::ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    if !args.bind.ip().is_loopback() {
        eprintln!("Rust preview requires loopback binding; native TLS has not been ported");
        return std::process::ExitCode::FAILURE;
    }
    let auth = match ml_exp_server::Auth::load(
        args.token_file
            .as_deref()
            .expect("clap requires token unless initializing"),
    ) {
        Ok(auth) => auth,
        Err(error) => {
            eprintln!("{error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let reads = match args.read_config.as_deref() {
        Some(path) => match ml_exp_server::read_config::load(path)
            .and_then(ml_exp_server::read_api::Api::open)
        {
            Ok(reads) => Some(std::sync::Arc::new(reads)),
            Err(error) => {
                eprintln!("{error}");
                return std::process::ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let listener = match tokio::net::TcpListener::bind(args.bind).await {
        Ok(listener) => listener,
        Err(_) => {
            eprintln!("cannot bind Rust preview address");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Ok(address) = listener.local_addr() {
        println!("{address}");
    }
    let result = axum::serve(listener, ml_exp_server::app(auth, reads))
        .with_graceful_shutdown(async {
            #[cfg(unix)]
            {
                if let Ok(mut term) =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                {
                    tokio::select! { _=tokio::signal::ctrl_c()=>{}, _=term.recv()=>{} }
                }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
        })
        .await;
    if result.is_err() {
        eprintln!("Rust preview server stopped with an I/O failure");
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}
