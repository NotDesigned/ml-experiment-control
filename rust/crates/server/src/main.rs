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
    #[arg(required_unless_present_any = ["initialize_registry", "initialize_project_lifecycle", "initialize_source_archives"])]
    token_file: Option<PathBuf>,
    /// Private approved source configuration for an isolated Rust preview.
    #[arg(long, conflicts_with = "initialize_registry")]
    read_config: Option<PathBuf>,
    /// Explicitly initialize a fresh registry; existing targets are refused.
    #[arg(long, conflicts_with_all = ["token_file", "read_config"])]
    initialize_registry: Option<PathBuf>,
    /// Explicitly create the lifecycle ledger from approved Projects once.
    #[arg(long, requires = "read_config", conflicts_with_all = ["initialize_registry", "token_file", "project_lifecycle"])]
    initialize_project_lifecycle: bool,
    /// Enable tracking-only metadata changes using an existing lifecycle ledger.
    #[arg(long, requires = "read_config", conflicts_with = "initialize_registry")]
    project_lifecycle: bool,
    /// Explicitly create immutable source storage from the approved policy.
    #[arg(long, requires = "read_config", conflicts_with_all = ["initialize_registry", "token_file", "project_lifecycle", "source_archives"])]
    initialize_source_archives: bool,
    /// Enable immutable source archive storage for this isolated preview.
    #[arg(long, requires = "read_config", conflicts_with = "initialize_registry")]
    source_archives: bool,
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
    if args.initialize_project_lifecycle {
        return match initialize_lifecycle(args.read_config.as_deref().expect("required config")) {
            Ok(()) => {
                println!("project lifecycle initialized; research files and jobs unchanged");
                std::process::ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    if args.initialize_source_archives {
        return match initialize_sources(args.read_config.as_deref().expect("required config")) {
            Ok(()) => {
                println!("source archive storage initialized; no repository code executed");
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
        Some(path) => match open_api(path, args.project_lifecycle, args.source_archives) {
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

fn initialize_lifecycle(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let settings = ml_exp_server::read_config::load(path)?;
    let (now, identity) = ml_exp_server::projects::transition_context()?;
    ml_exp_server::projects::Service::initialize(&settings, &now, &identity)?;
    Ok(())
}

fn open_api(
    path: &std::path::Path,
    lifecycle: bool,
    source_archives: bool,
) -> Result<ml_exp_server::read_api::Api, Box<dyn std::error::Error>> {
    let settings = ml_exp_server::read_config::load(path)?;
    let service = if lifecycle {
        Some(std::sync::Arc::new(ml_exp_server::projects::Service::open(
            &settings,
        )?))
    } else {
        None
    };
    let api = ml_exp_server::read_api::Api::open(settings.clone())?;
    let api = match service {
        Some(service) => api.with_lifecycle(service)?,
        None => api,
    };
    if source_archives {
        let source = std::sync::Arc::new(ml_exp_server::sources::Service::open(
            &settings,
            api.lifecycle(),
        )?);
        return Ok(api.with_sources(source)?);
    }
    Ok(api)
}

fn initialize_sources(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let settings = ml_exp_server::read_config::load(path)?;
    ml_exp_server::sources::Service::initialize(&settings)?;
    Ok(())
}
