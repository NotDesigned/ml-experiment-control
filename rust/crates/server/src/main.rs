use clap::Parser;
use std::{net::SocketAddr, path::PathBuf};

#[derive(Parser)]
#[command(
    name = "ml-expd-rs",
    version,
    about = "Native experiment API in an isolated registry with explicitly approved backend execution"
)]
struct Args {
    /// One private deployment file; initialization remains a separate command.
    #[arg(long,conflicts_with_all=["bind","token_file","read_config","historical_read_config","history_config","inspect_historical_config","initialize_registry","initialize_project_lifecycle","initialize_source_archives","initialize_submissions","initialize_cloud_submissions","initialize_administration","project_lifecycle","source_archives","submission_config","cloud_config","tracking_config","administration"])]
    config: Option<PathBuf>,
    /// Validate local configuration and private dependencies without a listener,
    /// SDK process, store initialization or provider request.
    #[arg(long, requires = "config", conflicts_with = "initialize")]
    check: bool,
    /// Initialize all configured stores in a completely new native registry.
    /// Existing/partial registries are refused; no provider job is created.
    #[arg(long, requires = "config", conflicts_with = "check")]
    initialize: bool,
    #[arg(long, default_value = "127.0.0.1:18765")]
    bind: SocketAddr,
    /// Enable explicit administrative deletion plans in an initialized namespace.
    #[arg(long,requires_all=["read_config","source_archives"])]
    administration: bool,
    #[arg(long,requires="read_config",conflicts_with_all=["token_file","administration"])]
    initialize_administration: bool,
    #[arg(long)]
    #[arg(required_unless_present_any = ["config", "initialize_registry", "initialize_project_lifecycle", "initialize_source_archives", "initialize_submissions", "initialize_cloud_submissions", "initialize_administration", "inspect_historical_config"])]
    token_file: Option<PathBuf>,
    /// Private approved source configuration for an isolated Rust preview.
    #[arg(long, conflicts_with = "initialize_registry")]
    read_config: Option<PathBuf>,
    /// Observe an original Python namespace with mandatory kernel-enforced
    /// read-only access. No initialization, provider effects or SDK publication.
    #[arg(long,conflicts_with_all=["read_config","initialize_registry","initialize_project_lifecycle","initialize_source_archives","initialize_submissions","initialize_cloud_submissions","initialize_administration","project_lifecycle","source_archives","submission_config","cloud_config","tracking_config","administration"])]
    historical_read_config: Option<PathBuf>,
    /// Combine a native writable workspace with original experiments served
    /// through the same API under enforced read-only access.
    #[arg(long,requires="read_config",conflicts_with_all=["historical_read_config","inspect_historical_config","initialize_registry","initialize_project_lifecycle","initialize_source_archives","initialize_submissions","initialize_cloud_submissions","initialize_administration"])]
    history_config: Option<PathBuf>,
    /// Observe existing state and print a pinned private historical reader
    /// configuration; starts no server and writes no original metadata.
    #[arg(long,conflicts_with_all=["token_file","read_config","historical_read_config","initialize_registry","initialize_project_lifecycle","initialize_source_archives","initialize_submissions","initialize_cloud_submissions","initialize_administration","project_lifecycle","source_archives","submission_config","cloud_config","tracking_config","administration"])]
    inspect_historical_config: Option<PathBuf>,
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
    let mut args = Args::parse();
    if let Some(path) = args.config.take() {
        let settings = match ml_exp_server::deployment::Settings::load(&path) {
            Ok(value) => value,
            Err(_) => {
                eprintln!("private deployment configuration is invalid or unavailable");
                return std::process::ExitCode::FAILURE;
            }
        };
        if args.initialize || args.check {
            let initialize = args.initialize;
            let result = tokio::task::spawn_blocking(move || {
                if initialize {
                    settings.initialize()
                } else {
                    settings.check().and_then(|value| {
                        serde_json::to_value(value)
                            .map_err(|_| ml_exp_server::deployment::Error { stage: "report" })
                    })
                }
            })
            .await
            .unwrap_or(Err(ml_exp_server::deployment::Error {
                stage: "configuration_worker",
            }));
            return match result {
                Ok(value) => {
                    println!("{value}");
                    std::process::ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    std::process::ExitCode::FAILURE
                }
            };
        }
        let settings =
            match tokio::task::spawn_blocking(move || settings.check().map(|_| settings)).await {
                Ok(Ok(settings)) => settings,
                Ok(Err(error)) => {
                    eprintln!("{error}");
                    return std::process::ExitCode::FAILURE;
                }
                Err(_) => {
                    eprintln!("native configuration worker unavailable");
                    return std::process::ExitCode::FAILURE;
                }
            };
        args.bind = settings.bind;
        args.token_file = Some(settings.token_file);
        args.read_config = Some(settings.read_config);
        args.history_config = settings.history_config;
        args.submission_config = settings.submission_config;
        args.cloud_config = settings.cloud_config;
        args.tracking_config = settings.tracking_config;
        args.project_lifecycle = settings.project_lifecycle;
        args.source_archives = settings.source_archives;
        args.administration = settings.administration;
    }
    if let Some(path) = &args.inspect_historical_config {
        let result =
            ml_exp_server::read_config::load(path).and_then(ml_exp_server::history::inspect);
        return match result {
            Ok(settings) => match serde_json::to_string_pretty(&settings) {
                Ok(value) => {
                    println!("{value}");
                    std::process::ExitCode::SUCCESS
                }
                Err(_) => {
                    eprintln!("historical configuration could not be encoded");
                    std::process::ExitCode::FAILURE
                }
            },
            Err(_) => {
                eprintln!("historical namespace or mandatory read-only isolation is unavailable");
                std::process::ExitCode::FAILURE
            }
        };
    }
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
            let history = args.history_config.clone();
            let administration = args.administration;
            let routing = tracking.as_ref().map(|runtime| runtime.routing().clone());
            let opened = tokio::task::spawn_blocking(move || {
                open_api(
                    &path,
                    lifecycle,
                    archives,
                    submission.as_deref(),
                    cloud.as_deref(),
                    history.as_deref(),
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
        None => match args.historical_read_config.as_deref() {
            Some(path) => {
                let path = path.to_owned();
                let result = tokio::task::spawn_blocking(move || {
                    ml_exp_server::history::Settings::load(&path)
                        .and_then(ml_exp_server::history::open)
                })
                .await;
                match result {
                    Ok(Ok(api)) => (Some(std::sync::Arc::new(api)), None),
                    _ => {
                        eprintln!(
                            "historical namespace or mandatory read-only isolation is unavailable"
                        );
                        return std::process::ExitCode::FAILURE;
                    }
                }
            }
            None => (None, None),
        },
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
    let _historical_publisher = match tracking
        .as_ref()
        .filter(|runtime| runtime.routing().enabled)
        .and_then(|runtime| runtime.historical_handoff().map(|path| (runtime, path)))
    {
        Some((runtime, path)) => {
            let open = || -> Result<_, ml_exp_server::tracking::Error> {
                let historical = ml_exp_server::history::Settings::load(
                    args.history_config
                        .as_deref()
                        .ok_or(ml_exp_server::tracking::Error::Configuration)?,
                )
                .map_err(|_| ml_exp_server::tracking::Error::Configuration)?;
                let native = ml_exp_server::read_config::load(
                    args.read_config
                        .as_deref()
                        .ok_or(ml_exp_server::tracking::Error::Configuration)?,
                )
                .map_err(|_| ml_exp_server::tracking::Error::Configuration)?;
                let source = ml_exp_server::tracking::handoff::Handoff::open(
                    ml_exp_server::tracking::handoff::Config::load(path)?,
                    historical,
                    &native.registry_root,
                )?;
                ml_exp_server::tracking::Publisher::start(source, runtime.clone())
            };
            match open() {
                Ok(publisher) => Some(publisher),
                Err(error) => {
                    eprintln!("historical W&B handoff: {error}");
                    return std::process::ExitCode::FAILURE;
                }
            }
        }
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
    history_config: Option<&std::path::Path>,
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
    let mut api = ml_exp_server::read_api::Api::open(settings.clone())?;
    if let Some(path) = history_config {
        api = api.with_history(ml_exp_server::history::Bridge::open(
            ml_exp_server::history::Settings::load(path)?,
        )?)?;
    }
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
            )?
            .with_history(api.history())?;
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
                .with_history(api.history())?
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
