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
    token_file: PathBuf,
}
#[tokio::main]
async fn main() -> std::process::ExitCode {
    let args = Args::parse();
    if !args.bind.ip().is_loopback() {
        eprintln!("Rust preview requires loopback binding; native TLS has not been ported");
        return std::process::ExitCode::FAILURE;
    }
    let auth = match ml_exp_server::Auth::load(&args.token_file) {
        Ok(auth) => auth,
        Err(error) => {
            eprintln!("{error}");
            return std::process::ExitCode::FAILURE;
        }
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
    let result = axum::serve(listener, ml_exp_server::app(auth))
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
