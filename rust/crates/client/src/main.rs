use clap::{Parser, Subcommand};
use ml_exp_client::{Client, config, credentials};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "ml-exp-rs",
    version,
    about = "Rust rewrite preview: private configuration and read-only API calls"
)]
struct Args {
    #[arg(long, global = true, env = "ML_EXPD_API_URL")]
    url: Option<String>,
    #[arg(long, global = true, env = "ML_EXPD_CLIENT_CONFIG")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Inspect the server handshake (no job submission).
    Check,
    /// Read any protocol-2 JSON endpoint under /api/.
    Get { path: String },
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

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let config = args.config.unwrap_or_else(|| {
        let xdg = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);
        let home = std::env::var_os("HOME").map(PathBuf::from);
        config::default_config_path(xdg.as_deref(), home.as_deref())
    });
    let settings = credentials::load_settings(&config)?;
    let result = match args.command {
        Command::Config {
            command: Config::Show,
        } => settings.public(),
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
            settings.public()
        }
        command => {
            let env_token = std::env::var("ML_EXPD_API_TOKEN").ok();
            let env_file = std::env::var_os("ML_EXPD_API_TOKEN_FILE").map(PathBuf::from);
            let plan = settings.resolve(
                args.url.as_deref(),
                env_token.as_deref(),
                env_file.as_deref(),
            )?;
            let token = credentials::load_token(plan.token_source)?;
            let client = Client::new(&plan.base_url, token)?;
            match command {
                Command::Check => client.get("/api/health")?,
                Command::Get { path } => client.get(&path)?,
                _ => unreachable!(),
            }
        }
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
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
