use clap::{Parser, Subcommand};
use ml_exp_client::{
    Client, config, credentials, observation, presentation, read, session, status, watch,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "ml-exp-rs",
    version,
    about = "Rust rewrite preview: private settings and exact read-only observations"
)]
struct Args {
    #[arg(long, global = true, env = "ML_EXPD_API_URL")]
    url: Option<String>,
    #[arg(long, global = true, env = "ML_EXPD_CLIENT_CONFIG")]
    config: Option<PathBuf>,
    /// Return complete JSON rather than the human status summary.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Validate the authenticated protocol handshake (no job submission).
    Check,
    /// Read a protocol-2 JSON endpoint under /api/.
    Get { path: String },
    /// Observe one exact experiment; these commands never start or retry work.
    Run {
        #[command(subcommand)]
        command: Run,
    },
    /// Save URL-bound private credentials, or show redacted settings.
    Config {
        #[command(subcommand)]
        command: Config,
    },
}
#[derive(Subcommand)]
enum Config {
    Set {
        #[arg(long)]
        token_file: PathBuf,
    },
    Show,
}
#[derive(clap::Args)]
struct Target {
    /// Original private experiment or training-operation JSON state.
    #[arg(long)]
    state: Option<PathBuf>,
    #[arg(long)]
    project: Option<String>,
    #[arg(long)]
    run: Option<String>,
    #[arg(long)]
    attempt: Option<String>,
}
impl Target {
    fn selection(&self) -> observation::Selection<'_> {
        observation::Selection {
            project: self.project.as_deref(),
            run: self.run.as_deref(),
            attempt: self.attempt.as_deref(),
        }
    }
}
#[derive(Subcommand)]
enum Run {
    /// Show independent scheduler, training and result-publication evidence.
    Status {
        #[command(flatten)]
        target: Target,
        #[arg(long)]
        details: bool,
        /// Observe the same Attempt without submitting or restarting it.
        #[arg(long)]
        watch: bool,
        /// Observation budget; an in-flight HTTP read keeps its own timeout.
        #[arg(long, default_value_t = 1800.0)]
        seconds: f64,
    },
    /// Read bounded logs from the bound Attempt.
    Logs {
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value = "both", value_parser = ["both", "stdout", "stderr"])]
        stream: String,
        #[arg(long, default_value_t = 80)]
        lines: i64,
    },
    /// Query arbitrary declared metrics with their original context and units.
    Metrics {
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value_t = 2000)]
        max_points: i64,
        #[arg(long)]
        keys: Option<String>,
        #[arg(long = "name")]
        names: Vec<String>,
    },
}
impl Run {
    fn target(&self) -> &Target {
        match self {
            Self::Status { target, .. }
            | Self::Logs { target, .. }
            | Self::Metrics { target, .. } => target,
        }
    }
}

enum Output {
    Json(serde_json::Value),
    Text(String),
}
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let watch_duration = match &args.command {
        Command::Run {
            command:
                Run::Status {
                    watch: true,
                    seconds,
                    ..
                },
        } => Some(
            std::time::Duration::try_from_secs_f64(*seconds).map_err(|_| {
                ml_exp_client::Error::Observation("status --seconds must be finite and nonnegative")
            })?,
        ),
        _ => None,
    };
    let config = args.config.unwrap_or_else(|| {
        let xdg = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);
        let home = std::env::var_os("HOME").map(PathBuf::from);
        config::default_config_path(xdg.as_deref(), home.as_deref())
    });
    let settings = credentials::load_settings(&config)?;
    let result = match args.command {
        Command::Config {
            command: Config::Show,
        } => Output::Json(settings.public()),
        Command::Config {
            command: Config::Set { token_file },
        } => {
            let url = args
                .url
                .as_deref()
                .or(settings.url.as_deref())
                .ok_or("set --url")?;
            let settings = credentials::settings_from_token_file(url, token_file)?;
            credentials::save_settings(&settings, &config)?;
            Output::Json(settings.public())
        }
        command => {
            let target = match &command {
                Command::Run { command } => Some(command.target()),
                _ => None,
            };
            let saved = target
                .and_then(|t| t.state.as_deref())
                .map(session::load_saved)
                .transpose()?;
            let env_token = std::env::var("ML_EXPD_API_TOKEN").ok();
            let env_file = std::env::var_os("ML_EXPD_API_TOKEN_FILE").map(PathBuf::from);
            let plan = session::connection(
                &settings,
                args.url.as_deref(),
                env_token.as_deref(),
                env_file.as_deref(),
                saved.as_ref(),
                target.map(Target::selection).unwrap_or_default(),
            )?;
            if let Some(target) = target {
                // Reject incomplete or conflicting local identities before even
                // reading the credential or sending the handshake.
                observation::begin(&plan.base_url, saved.as_ref(), target.selection())?;
            }
            let token = credentials::load_token(plan.token_source)?;
            let client = Client::new(&plan.base_url, token)?;
            match command {
                Command::Check => Output::Json(client.negotiate()?),
                Command::Get { path } => Output::Json(client.get(&path)?),
                Command::Run { command } => {
                    let health = client.negotiate()?;
                    match command {
                        Run::Status {
                            target, details, ..
                        } => {
                            let value = if let Some(duration) = watch_duration {
                                watch::observe(
                                    &client,
                                    &health,
                                    saved.as_ref(),
                                    target.selection(),
                                    details,
                                    duration,
                                    |value| {
                                        let update = if args.json {
                                            serde_json::to_string(value)
                                                .map_err(|_| ml_exp_client::Error::Response)?
                                        } else {
                                            presentation::status_summary(value).map_err(|_| {
                                                ml_exp_client::Error::Observation(
                                                    "cannot render malformed experiment evidence",
                                                )
                                            })?
                                        };
                                        eprintln!("{update}");
                                        Ok(())
                                    },
                                )?
                            } else {
                                status::observe(
                                    &client,
                                    &health,
                                    saved.as_ref(),
                                    target.selection(),
                                    details,
                                )?
                            };
                            if args.json {
                                Output::Json(value)
                            } else {
                                Output::Text(presentation::status_summary(&value)?)
                            }
                        }
                        Run::Logs {
                            target,
                            stream,
                            lines,
                        } => Output::Json(read::observe(
                            &client,
                            &health,
                            saved.as_ref(),
                            target.selection(),
                            read::View::Logs { stream, lines },
                        )?),
                        Run::Metrics {
                            target,
                            max_points,
                            keys,
                            names,
                        } => Output::Json(read::observe(
                            &client,
                            &health,
                            saved.as_ref(),
                            target.selection(),
                            read::View::Metrics {
                                max_points,
                                keys,
                                names: (!names.is_empty()).then_some(names),
                            },
                        )?),
                    }
                }
                _ => unreachable!(),
            }
        }
    };
    match result {
        Output::Json(value) => println!("{}", serde_json::to_string_pretty(&value)?),
        Output::Text(text) => println!("{text}"),
    }
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run(Args::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
