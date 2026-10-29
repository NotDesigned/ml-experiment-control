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
    /// Enable explicit administrative deletion plans in an initialized namespace.
    #[arg(long,requires_all=["read_config","source_archives"])]
    administration: bool,
    #[arg(long,requires="read_config",conflicts_with_all=["token_file","administration"])]
    initialize_administration: bool,
    #[arg(long)]
    #[arg(required_unless_present_any = ["initialize_registry", "initialize_project_lifecycle", "initialize_source_archives", "initialize_submissions", "initialize_cloud_submissions", "initialize_administration"])]
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
    /// Explicit offline initialization of a fresh native submission namespace.
    #[arg(long, requires = "read_config", conflicts_with_all = ["initialize_registry", "token_file", "initialize_project_lifecycle", "initialize_source_archives", "submission_config"])]
    initialize_submissions: bool,
    /// Explicit offline creation of a fresh cloud Run and effect namespace.
    #[arg(long,requires="read_config",conflicts_with_all=["initialize_registry","token_file","cloud_config"])]
    initialize_cloud_submissions: bool,
    /// Trusted native Slurm profiles and published-image receipts, never client input.
    #[arg(long, requires_all = ["read_config", "source_archives"], conflicts_with = "initialize_registry")]
    submission_config: Option<PathBuf>,
    /// Independent approved PAI/SenseCore configuration; native HTTPS effects.
    #[arg(long,requires_all=["read_config","source_archives"],conflicts_with="initialize_registry")]
    cloud_config: Option<PathBuf>,
    /// Optional private official-cloud W&B credentials and SDK interpreter.
    #[arg(long, requires = "read_config")]
    tracking_config: Option<PathBuf>,
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
    if args.initialize_administration {
        let result =
            ml_exp_server::read_config::load(args.read_config.as_deref().expect("required config"))
                .map_err(|_| "invalid read configuration".to_string())
                .and_then(|settings| {
                    ml_exp_server::administration::Service::initialize(&settings)
                        .map_err(|e| e.to_string())
                });
        return match result {
            Ok(()) => {
                println!("administration namespace initialized; no payload removed");
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
    if args.initialize_submissions {
        return match ml_exp_server::read_config::load(
            args.read_config.as_deref().expect("required config"),
        )
        .map_err(|_| "invalid read configuration")
        .and_then(|settings| {
            ml_exp_server::submissions::Service::initialize(&settings)
                .map_err(|_| "cannot initialize fresh submission namespace")
        }) {
            Ok(()) => {
                println!("native submission storage initialized; no scheduler job submitted");
                std::process::ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    if args.initialize_cloud_submissions {
        let result =
            ml_exp_server::read_config::load(args.read_config.as_deref().expect("required config"))
                .map_err(|_| "invalid read configuration")
                .and_then(|settings| {
                    ml_exp_server::cloud_runs::Service::initialize(&settings)
                        .map_err(|_| "cannot initialize fresh cloud namespace")
                });
        return match result {
            Ok(()) => {
                println!("cloud submission storage initialized; no provider job submitted");
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
    let tracking = match args
        .tracking_config
        .as_deref()
        .map(|path| {
            ml_exp_server::tracking::Config::load(path)
                .and_then(ml_exp_server::tracking::Runtime::open)
        })
        .transpose()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("{error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Some(error) = tracking.as_ref().and_then(|runtime| runtime.diagnostic()) {
        eprintln!("optional W&B publisher: {error}");
    }
    let (reads, polling) = match args.read_config.as_deref() {
        Some(path) => {
            let path = path.to_owned();
            let lifecycle = args.project_lifecycle;
            let archives = args.source_archives;
            let submission = args.submission_config.clone();
            let cloud = args.cloud_config.clone();
            let administration = args.administration;
            let routing = tracking.as_ref().map(|runtime| runtime.routing().clone());
            let opened = tokio::task::spawn_blocking(move || {
                open_api(
                    &path,
                    lifecycle,
                    archives,
                    submission.as_deref(),
                    cloud.as_deref(),
                    routing,
                )
                .and_then(|(api, polling)| {
                    if !administration {
                        return Ok((api, polling));
                    }
                    let settings = ml_exp_server::read_config::load(&path)?;
                    let service =
                        std::sync::Arc::new(ml_exp_server::administration::Service::open(
                            settings,
                            api.submissions(),
                            api.cloud(),
                        )?);
                    Ok((api.with_administration(service)?, polling))
                })
                .map_err(|e| e.to_string())
            })
            .await
            .unwrap_or_else(|_| Err("native service initialization failed".into()));
            match opened {
                Ok((reads, polling)) => (Some(std::sync::Arc::new(reads)), polling),
                Err(error) => {
                    eprintln!("{error}");
                    return std::process::ExitCode::FAILURE;
                }
            }
        }
        None => (None, None),
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
    let _observer = match polling.filter(|policy| policy.enabled) {
        Some(policy) => match ml_exp_server::background::Observer::start(
            reads
                .as_ref()
                .and_then(|api| api.submissions())
                .expect("submission polling configuration"),
            policy,
        ) {
            Ok(observer) => Some(observer),
            Err(error) => {
                eprintln!("{error}");
                return std::process::ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let _cloud_observer = match reads
        .as_ref()
        .and_then(|api| api.cloud())
        .filter(|s| s.polling_enabled())
    {
        Some(service) => match ml_exp_server::cloud_runs::Observer::start(service) {
            Ok(observer) => Some(observer),
            Err(_) => {
                eprintln!("cloud scheduler observer unavailable");
                return std::process::ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let _publisher = match tracking.filter(|runtime| runtime.routing().enabled) {
        Some(runtime) => match ml_exp_server::tracking::Publisher::start(
            reads
                .as_ref()
                .cloned()
                .expect("tracking requires configured native services"),
            runtime,
        ) {
            Ok(publisher) => Some(publisher),
            Err(error) => {
                eprintln!("optional W&B publisher: {error}");
                None
            }
        },
        None => None,
    };
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
    submission_config: Option<&std::path::Path>,
    cloud_config: Option<&std::path::Path>,
    tracking: Option<ml_exp_core::tracking::RoutingSettings>,
) -> Result<
    (
        ml_exp_server::read_api::Api,
        Option<ml_exp_core::polling::Policy>,
    ),
    Box<dyn std::error::Error>,
> {
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
        let mut api = api.with_sources(source.clone())?;
        if let Some(path) = cloud_config {
            let config = ml_exp_server::cloud_runs::Settings::load(path)?;
            let cloud = ml_exp_server::cloud_runs::Service::open(
                settings.clone(),
                config,
                source.clone(),
                api.lifecycle(),
                tracking.clone().unwrap_or_default(),
            )?;
            api = api.with_cloud(std::sync::Arc::new(cloud))?;
        }
        if let Some(path) = submission_config {
            let config = ml_exp_server::submissions::config::load(path)?;
            let submissions = std::sync::Arc::new(
                ml_exp_server::submissions::Service::open(
                    settings,
                    source,
                    api.lifecycle(),
                    config.backends,
                    config.enabled,
                )?
                .with_runtime_builder(api.cloud())
                .with_run_creation(config.creation)?
                .with_tracking(tracking.unwrap_or_default()),
            );
            return Ok((api.with_submissions(submissions)?, Some(config.polling)));
        }
        return Ok((api, None));
    }
    Ok((api, None))
}

fn initialize_sources(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let settings = ml_exp_server::read_config::load(path)?;
    ml_exp_server::sources::Service::initialize(&settings)?;
    Ok(())
}
