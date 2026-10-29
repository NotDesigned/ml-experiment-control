use clap::{Parser, Subcommand};
use ml_exp_client::{
    Client, artifacts, config, credentials, datasets, observation, presentation, projects, read,
    session, sources, status, submissions, watch,
};
use std::{
    io::{self, Read},
    path::PathBuf,
};

#[derive(Parser)]
#[command(
    name = "ml-exp-rs",
    version,
    about = "Rust rewrite preview: private settings, exact observations and project lifecycle"
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
    /// Read scheduler allocation evidence, or explicitly refresh the cache.
    Backend {
        executor: Option<String>,
        #[arg(long, requires = "executor")]
        refresh: bool,
    },
    /// Plan and execute administrative payload removal or Run retirement.
    Admin {
        #[command(subcommand)]
        command: ml_exp_client::administration::Command,
    },
    /// Create, inspect and stop an explicitly leased SSH debug machine.
    Debug {
        #[command(subcommand)]
        command: ml_exp_client::debug::Command,
    },
    /// Build, inspect and reconcile a reusable Dockerfile environment.
    #[command(name = "env", alias = "runtime")]
    Environment {
        #[command(subcommand)]
        command: ml_exp_client::runtime::Command,
    },
    /// Install a checksum-verified standalone native release.
    Update {
        #[arg(long, default_value = "0.1.0")]
        version: String,
        #[arg(long)]
        check: bool,
    },
    /// Upload or inspect immutable datasets on a configured backend.
    Data {
        #[command(subcommand)]
        command: Data,
    },
    /// Validate the authenticated protocol handshake (no job submission).
    Check,
    /// Read a protocol-2 JSON endpoint under /api/.
    Get { path: String },
    /// Create, submit or observe an experiment through explicit native operations.
    Run {
        #[command(subcommand)]
        command: Run,
    },
    /// Register and manage daemon tracking; research files and jobs are unchanged.
    Project {
        #[command(subcommand)]
        command: Project,
    },
    /// Pack and publish immutable source, or inspect an exact source identity.
    Source {
        #[command(subcommand)]
        command: Source,
    },
    /// Save URL-bound private credentials, or show redacted settings.
    Config {
        #[command(subcommand)]
        command: Config,
    },
    /// Run a pure native contract validator without credentials or HTTP.
    Contract {
        #[command(subcommand)]
        command: Contract,
    },
}
#[derive(Subcommand)]
enum Data {
    /// Import a complete plain tar already in an approved OSS dataset prefix.
    ImportOss {
        #[arg(long)]
        project: String,
        #[arg(long)]
        executor: String,
        #[arg(long)]
        object_key: String,
        #[arg(long)]
        sha256: String,
        #[arg(long)]
        archive_bytes: u64,
    },
    /// List retained dataset identities and delivery readiness; no backend query.
    List {
        #[arg(long)]
        project: String,
        #[arg(long)]
        executor: String,
    },
    Download {
        #[arg(long)]
        project: String,
        #[arg(long)]
        executor: String,
        #[arg(long)]
        asset: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Build the same immutable data image and advance its zero-GPU NAS copy.
    Publish {
        #[arg(long)]
        project: String,
        #[arg(long)]
        executor: String,
        #[arg(long)]
        asset: String,
        #[arg(long)]
        request_id: String,
        #[arg(long)]
        confirmation: String,
    },
    Upload {
        #[arg(long)]
        project: String,
        #[arg(long)]
        executor: String,
        #[arg(long)]
        directory: PathBuf,
    },
    Show {
        #[arg(long)]
        project: String,
        #[arg(long)]
        executor: String,
        #[arg(long)]
        asset: String,
    },
}

#[derive(Subcommand)]
enum Contract {
    /// Validate one JSON OperationIntent read from standard input.
    IntentValidate,
    /// Parse one Slurm allocation snapshot from standard input.
    SlurmAllocationValidate,
}
#[derive(Subcommand)]
enum Source {
    /// Pack local code without running it, then upload once to an approved Project.
    Import {
        #[arg(long)]
        project: String,
        #[arg(long)]
        directory: PathBuf,
    },
    /// Read and verify one immutable source metadata record.
    Show {
        #[arg(long)]
        project: String,
        #[arg(long)]
        source: String,
    },
    /// Reconcile a possibly lost publication response using its archive SHA.
    Lookup {
        #[arg(long)]
        project: String,
        #[arg(long)]
        sha256: String,
    },
}
impl Source {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        let project = match self {
            Self::Import { project, .. }
            | Self::Show { project, .. }
            | Self::Lookup { project, .. } => project,
        };
        ml_exp_core::identity::validate_id(project)?;
        match self {
            Self::Show { source, .. } => {
                ml_exp_core::sources::validate_source_id(source)?;
            }
            Self::Lookup { sha256, .. } => {
                ml_exp_core::sources::validate_archive_digest(sha256)?;
            }
            _ => {}
        }
        Ok(())
    }
}
#[derive(clap::Args)]
struct ProjectTarget {
    project: String,
    #[arg(long, default_value = "")]
    reason: String,
    /// Compare-and-swap against a revision returned by project list.
    #[arg(long)]
    expected_revision: Option<u64>,
}
#[derive(Subcommand)]
enum Project {
    /// List registered projects and lifecycle audit events.
    List,
    /// Register an approved manifest already on the daemon host; uploads no file.
    Register {
        /// Absolute daemon-host path, not a path on this client.
        manifest_path: String,
        #[arg(long)]
        expected_revision: Option<u64>,
    },
    /// Pause daemon tracking without stopping training jobs.
    Pause(ProjectTarget),
    /// Resume a paused project.
    Resume(ProjectTarget),
    /// Archive daemon tracking; requires a nonempty --reason.
    Archive(ProjectTarget),
    /// Restore an archived project to paused.
    Restore(ProjectTarget),
    /// Remove one registration; research files and jobs are unchanged.
    Unregister(ProjectTarget),
    /// Remove all registrations; research files and jobs are unchanged.
    UnregisterAll {
        #[arg(long, default_value = "")]
        reason: String,
        #[arg(long)]
        expected_revision: Option<u64>,
    },
}
impl Project {
    fn plan(&self) -> Result<projects::Plan, ml_exp_client::Error> {
        let (target, action) = match self {
            Self::List => return projects::plan(projects::Command::List),
            Self::Register {
                manifest_path,
                expected_revision,
            } => {
                return projects::plan(projects::Command::Register {
                    manifest_path,
                    expected_revision: *expected_revision,
                });
            }
            Self::UnregisterAll {
                reason,
                expected_revision,
            } => {
                return projects::plan(projects::Command::UnregisterAll {
                    reason,
                    expected_revision: *expected_revision,
                });
            }
            Self::Pause(t) => (t, projects::Action::Pause),
            Self::Resume(t) => (t, projects::Action::Resume),
            Self::Archive(t) => (t, projects::Action::Archive),
            Self::Restore(t) => (t, projects::Action::Restore),
            Self::Unregister(t) => (t, projects::Action::Unregister),
        };
        projects::plan(projects::Command::Change {
            project: &target.project,
            action,
            reason: &target.reason,
            expected_revision: target.expected_revision,
        })
    }
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
    /// Prepare a bounded parameter/seed matrix; saves every recovery identity.
    Batch {
        #[arg(long)]
        project: String,
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        state_dir: PathBuf,
    },
    /// Read one status row per saved experiment without submitting any work.
    BatchStatus {
        #[arg(long)]
        state_dir: PathBuf,
    },
    /// Advance environment and data preparation for every saved experiment.
    BatchContinue {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        confirmation: String,
    },
    /// Submit prepared children once, within this batch's concurrency cap.
    BatchSubmit {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        max_active: usize,
        #[arg(long)]
        confirmation: String,
    },
    /// Freeze one experiment configuration; save its recovery identity before sending.
    Start {
        #[arg(long)]
        project: String,
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        state: PathBuf,
    },
    /// Advance the same environment/Run preparation; never authorize or submit training.
    Continue {
        #[arg(long)]
        project: String,
        preparation: String,
        #[arg(long)]
        confirmation: String,
    },
    /// Deliver retained cloud metadata with an explicit zero-GPU CPU job.
    Sync {
        submission: String,
        #[arg(long)]
        request_id: String,
        #[arg(long)]
        confirmation: String,
    },
    /// Show the frozen W&B route and local publication cursors/errors; no cloud request.
    Tracking {
        #[arg(long)]
        project: String,
        #[arg(long)]
        run: String,
    },
    /// Ingest the next metrics history page; repeat until COMPLETE. Does not change the job.
    History { submission: String },
    /// Append a new Attempt from an explicit checkpoint and cumulative budget; does not submit.
    Retry {
        submission: String,
        #[arg(long)]
        confirmation: String,
        #[arg(long)]
        run: String,
        #[arg(long)]
        attempt: String,
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        max_gpu_hours: f64,
        #[arg(long, default_value = "")]
        reason: String,
    },
    /// Cancel an exact owned job; repeated calls reconcile the same cancel intent.
    Cancel {
        submission: String,
        #[arg(long)]
        confirmation: String,
    },
    /// Create immutable Run/Attempt metadata from one local JSON configuration; does not submit.
    Create {
        #[arg(long)]
        project: String,
        #[arg(long)]
        file: PathBuf,
    },
    /// Prepare the already-authored first Run/Attempt; never dispatch a job.
    Prepare {
        #[arg(long)]
        project: String,
        #[arg(long)]
        run: String,
        #[arg(long)]
        max_gpu_hours: f64,
        #[arg(long, default_value = "")]
        reason: String,
    },
    /// Recover the original preparation after a lost response, without creating one.
    Preparation {
        #[arg(long)]
        project: String,
        #[arg(long)]
        run: String,
    },
    /// Read the exact original Submission.
    Submission { submission: String },
    /// Replace an expired, never-started gate; the replacement needs fresh authorization.
    Renew {
        submission: String,
        #[arg(long)]
        request_id: String,
        #[arg(long)]
        confirmation: String,
    },
    /// Authorize a reviewed preparation; does not submit it.
    Authorize {
        submission: String,
        #[arg(long, default_value = "")]
        reason: String,
    },
    /// Dispatch once using the exact confirmation shown by preparation.
    Execute {
        submission: String,
        #[arg(long)]
        confirmation: String,
    },
    /// Inspect the original scheduler identity; never submit a second job.
    Reconcile { submission: String },
    /// Sample bounded logs, metrics and process evidence from the exact remote worker.
    Refresh { submission: String },
    /// Collect completed process outputs on backend storage.
    Collect { submission: String },
    /// Read the original backend result bundle and retention.
    Artifacts { submission: String },
    /// List files in the verified result archive; reads metadata only.
    Files { submission: String },
    /// Download one file while verifying the archive; never ACK or release it.
    File {
        submission: String,
        #[arg(long)]
        path: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Download, verify every file, then acknowledge (use --keep to retain the generated archive).
    Download {
        submission: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        keep: bool,
    },
    /// Verify a complete backend checkpoint and register its metadata.
    Checkpoint { submission: String },
    /// List registered checkpoints, or read one exact --checkpoint identity.
    Checkpoints {
        #[arg(long)]
        project: String,
        #[arg(long)]
        run: String,
        #[arg(long, default_value = "attempt-001")]
        attempt: String,
        #[arg(long)]
        checkpoint: Option<String>,
    },
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
    fn target(&self) -> Option<&Target> {
        match self {
            Self::Status { target, .. }
            | Self::Logs { target, .. }
            | Self::Metrics { target, .. } => Some(target),
            _ => None,
        }
    }
    fn submission_plan(&self) -> Result<Option<submissions::Plan>, ml_exp_client::Error> {
        let command = match self {
            Self::Sync {
                submission,
                request_id,
                confirmation,
            } => submissions::Command::Sync {
                submission,
                request_id,
                confirmation,
            },
            Self::Tracking { project, run } => submissions::Command::Tracking { project, run },
            Self::History { submission } => submissions::Command::History { submission },
            Self::Cancel {
                submission,
                confirmation,
            } => submissions::Command::Cancel {
                submission,
                confirmation,
            },
            Self::Prepare {
                project,
                run,
                max_gpu_hours,
                reason,
            } => submissions::Command::Prepare {
                project,
                run,
                max_gpu_hours: *max_gpu_hours,
                reason,
            },
            Self::Preparation { project, run } => submissions::Command::Existing { project, run },
            Self::Submission { submission } => submissions::Command::Show { submission },
            Self::Renew {
                submission,
                request_id,
                confirmation,
            } => submissions::Command::Renew {
                submission,
                request_id,
                confirmation,
            },
            Self::Authorize { submission, reason } => {
                submissions::Command::Authorize { submission, reason }
            }
            Self::Execute {
                submission,
                confirmation,
            } => submissions::Command::Execute {
                submission,
                confirmation,
            },
            Self::Reconcile { submission } => submissions::Command::Reconcile { submission },
            Self::Refresh { submission } => submissions::Command::Refresh { submission },
            Self::Checkpoint { submission } => submissions::Command::Checkpoint { submission },
            Self::Checkpoints {
                project,
                run,
                attempt,
                checkpoint,
            } => submissions::Command::Checkpoints {
                project,
                run,
                attempt,
                checkpoint: checkpoint.as_deref(),
            },
            _ => return Ok(None),
        };
        submissions::plan(command).map(Some)
    }
}

enum Output {
    Json(serde_json::Value),
    Text(String),
}
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if let Command::Backend {
        executor: Some(executor),
        ..
    } = &args.command
    {
        ml_exp_core::identity::validate_id(executor)?;
    }
    let batch = match &args.command {
        Command::Run {
            command: Run::Batch { project, file, .. },
        } => Some(ml_exp_client::batch::load(project, file)?),
        _ => None,
    };
    let experiment = match &args.command {
        Command::Run {
            command: Run::Start { project, file, .. },
        } => Some(ml_exp_client::preparations::load(project, file)?),
        _ => None,
    };
    if let Command::Update { version, check } = &args.command {
        println!(
            "{}",
            serde_json::to_string_pretty(&ml_exp_client::update::run(version, *check)?)?
        );
        return Ok(());
    }
    let admin_plan = if let Command::Admin { command } = &args.command {
        Some(command.plan()?)
    } else {
        None
    };
    let debug_plan = if let Command::Debug { command } = &args.command {
        Some(command.plan()?)
    } else {
        None
    };
    let runtime_plan = if let Command::Environment { command } = &args.command {
        Some(command.plan()?)
    } else {
        None
    };
    if let Command::Data { command } = &args.command {
        match command {
            Data::ImportOss {
                project,
                executor,
                object_key,
                sha256,
                archive_bytes,
            } => {
                datasets::validate(project, executor, None)?;
                ml_exp_core::datasets::oss::Import {
                    object_key: object_key.clone(),
                    sha256: sha256.clone(),
                    archive_bytes: *archive_bytes,
                }
                .validate()?;
            }
            Data::Download {
                project,
                executor,
                asset,
                ..
            } => datasets::validate(project, executor, Some(asset))?,
            Data::Publish {
                project,
                executor,
                asset,
                request_id,
                confirmation,
            } => {
                datasets::validate(project, executor, Some(asset))?;
                ml_exp_core::submission::validate_submission_token(request_id)?;
                if confirmation != &format!("PUBLISH {asset}") {
                    return Err("invalid exact data publication confirmation".into());
                }
            }
            Data::List { project, executor }
            | Data::Upload {
                project, executor, ..
            } => datasets::validate(project, executor, None)?,
            Data::Show {
                project,
                executor,
                asset,
            } => datasets::validate(project, executor, Some(asset))?,
        }
    }
    // Business parameter validation precedes config or credential file loading.
    if let Command::Run {
        command:
            Run::Collect { submission }
            | Run::Artifacts { submission }
            | Run::Files { submission }
            | Run::File { submission, .. }
            | Run::Download { submission, .. },
    } = &args.command
    {
        ml_exp_core::operations::validate_submission_id(submission).map_err(|_| {
            ml_exp_client::Error::Observation("invalid submission identity; no request sent")
        })?;
    }

    if let Command::Contract {
        command: Contract::IntentValidate,
    } = &args.command
    {
        let mut input = String::new();
        io::stdin()
            .take(8 * 1024 * 1024 + 1)
            .read_to_string(&mut input)?;
        if input.len() > 8 * 1024 * 1024 {
            return Err("contract input exceeds 8 MiB".into());
        }
        let value: serde_json::Value = serde_json::from_str(&input)?;
        let intent = ml_exp_core::intent::OperationIntent::from_value(&value)
            .map_err(|error| format!("invalid operation intent: {error}"))?;
        println!("{}", serde_json::to_string_pretty(&intent)?);
        return Ok(());
    }
    if let Command::Contract {
        command: Contract::SlurmAllocationValidate,
    } = &args.command
    {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Input {
            output: String,
            partition: String,
        }
        let mut input = String::new();
        io::stdin()
            .take(8 * 1024 * 1024 + 1)
            .read_to_string(&mut input)?;
        if input.len() > 8 * 1024 * 1024 {
            return Err("contract input exceeds 8 MiB".into());
        }
        let request: Input = serde_json::from_str(&input)?;
        let snapshot = ml_exp_core::slurm::parse_allocation(&request.output, &request.partition)
            .map_err(|error| format!("invalid Slurm allocation: {error}"))?;
        println!("{}", serde_json::to_string_pretty(&snapshot)?);
        return Ok(());
    }
    let project_plan = match &args.command {
        Command::Project { command } => Some(command.plan()?),
        _ => None,
    };
    let submission_plan = match &args.command {
        Command::Run {
            command: Run::Create { project, file },
        } => {
            ml_exp_core::identity::validate_id(project)?;
            let mut bytes = Vec::new();
            std::fs::File::open(file)?
                .take(ml_exp_core::first_submission::native::creation::MAX_REQUEST_BYTES as u64 + 1)
                .read_to_end(&mut bytes)?;
            let request =
                ml_exp_core::first_submission::native::creation::Request::from_slice(&bytes)?;
            Some(submissions::plan(submissions::Command::Create {
                project,
                request: &request,
            })?)
        }
        Command::Run {
            command:
                Run::Retry {
                    submission,
                    confirmation,
                    run,
                    attempt,
                    checkpoint,
                    max_gpu_hours,
                    reason,
                },
        } => {
            let request = ml_exp_core::first_submission::native::retry::Request {
                confirmation: confirmation.clone(),
                resume_from: ml_exp_core::checkpoints::Reference {
                    run_id: run.clone(),
                    attempt_id: attempt.clone(),
                    checkpoint_id: checkpoint.clone(),
                },
                max_gpu_hours: *max_gpu_hours,
                reason: reason.clone(),
            };
            Some(submissions::plan(submissions::Command::Retry {
                submission,
                request: &request,
            })?)
        }
        Command::Run { command } => command.submission_plan()?,
        _ => None,
    };
    if let Command::Source { command } = &args.command {
        command.validate()?;
    }
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
                Command::Run { command } => command.target(),
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
                Command::Backend { executor, refresh } => {
                    client.negotiate()?;
                    Output::Json(if let Some(executor) = executor {
                        let path = format!("/api/backends/{executor}/capacity");
                        if refresh {
                            client.post(&format!("{path}/refresh"), &serde_json::json!({}))?
                        } else {
                            client.get(&path)?
                        }
                    } else {
                        client.get("/api/executors")?
                    })
                }
                Command::Admin { .. } => {
                    let health = client.negotiate()?;
                    Output::Json(ml_exp_client::administration::execute(
                        &client,
                        admin_plan.ok_or("admin plan unavailable")?,
                        &health,
                    )?)
                }
                Command::Debug { .. } => {
                    let health = client.negotiate()?;
                    Output::Json(ml_exp_client::debug::execute(
                        &client,
                        debug_plan.ok_or("debug plan unavailable")?,
                        &health,
                    )?)
                }
                Command::Environment { .. } => {
                    let health = client.negotiate()?;
                    Output::Json(ml_exp_client::runtime::execute(
                        &client,
                        runtime_plan.ok_or("runtime plan unavailable")?,
                        &health,
                    )?)
                }
                Command::Data { command } => {
                    let health = client.negotiate()?;
                    Output::Json(match command {
                        Data::ImportOss { project, executor, object_key, sha256, archive_bytes } => datasets::import_oss(&client, &project, &executor,
                            ml_exp_core::datasets::oss::Import { object_key, sha256, archive_bytes }, &health)?,
                        Data::List{project,executor}=>datasets::list(&client,&project,&executor,&health)?,
                        Data::Download{project,executor,asset,output}=>datasets::download(&client,&project,&executor,&asset,&output,&health)?,
                        Data::Publish{project,executor,asset,request_id,confirmation}=>client.post(&format!("/api/projects/{project}/datasets/{executor}/{asset}/publish"),&serde_json::json!({"request_id":request_id,"confirmation":confirmation}))?,
                        Data::Upload {
                            project,
                            executor,
                            directory,
                        } => datasets::upload(&client, &project, &executor, &directory, &health)?,
                        Data::Show {
                            project,
                            executor,
                            asset,
                        } => datasets::show(&client, &project, &executor, &asset, &health)?,
                    })
                }
                Command::Check => Output::Json(client.negotiate()?),
                Command::Get { path } => Output::Json(client.get(&path)?),
                Command::Project { .. } => {
                    let health = client.negotiate()?;
                    Output::Json(projects::execute(
                        &client,
                        project_plan.as_ref().expect("validated project plan"),
                        &health,
                    )?)
                }
                Command::Source { command } => {
                    let health = client.negotiate()?;
                    Output::Json(match command {
                        Source::Import { project, directory } => {
                            sources::import_directory(&client, &project, &directory, &health)?
                        }
                        Source::Show { project, source } => {
                            sources::read(&client, &project, &source, &health)?
                        }
                        Source::Lookup { project, sha256 } => {
                            sources::lookup_archive(&client, &project, &sha256, &health)?
                        }
                    })
                }
                Command::Run { command } => {
                    let health = client.negotiate()?;
                    match command {
                        Run::Batch {
                            project, state_dir, ..
                        } => Output::Json(ml_exp_client::batch::prepare(
                            &client,
                            &project,
                            batch.as_ref().ok_or("batch unavailable")?,
                            &state_dir,
                            &health,
                        )?),
                        Run::BatchStatus { state_dir } => {
                            Output::Json(ml_exp_client::batch::status(&client, &state_dir)?)
                        }
                        Run::BatchContinue {
                            state_dir,
                            confirmation,
                        } => Output::Json(ml_exp_client::batch::advance(
                            &client,
                            &state_dir,
                            &confirmation,
                        )?),
                        Run::BatchSubmit {
                            state_dir,
                            max_active,
                            confirmation,
                        } => Output::Json(ml_exp_client::batch::dispatch(
                            &client,
                            &state_dir,
                            max_active,
                            &confirmation,
                            &health,
                        )?),
                        Run::Start { project, state, .. } => {
                            Output::Json(ml_exp_client::preparations::prepare(
                                &client,
                                &project,
                                experiment.as_ref().ok_or("experiment unavailable")?,
                                &state,
                                &health,
                            )?)
                        }
                        Run::Continue {
                            project,
                            preparation,
                            confirmation,
                        } => Output::Json(ml_exp_client::preparations::advance(
                            &client,
                            &project,
                            &preparation,
                            &confirmation,
                        )?),
                        Run::Collect { submission } => {
                            Output::Json(artifacts::observe(&client, &submission, true, &health)?)
                        }
                        Run::Artifacts { submission } => {
                            Output::Json(artifacts::observe(&client, &submission, false, &health)?)
                        }
                        Run::Files { submission } => {
                            Output::Json(artifacts::files(&client, &submission, &health)?)
                        }
                        Run::File {
                            submission,
                            path,
                            output,
                        } => Output::Json(artifacts::download_file(
                            &client,
                            &submission,
                            &path,
                            &output,
                            &health,
                        )?),
                        Run::Download {
                            submission,
                            output,
                            keep,
                        } => Output::Json(artifacts::download(
                            &client,
                            &submission,
                            &output,
                            !keep,
                            &health,
                        )?),
                        Run::Tracking { .. }
                        | Run::Sync { .. }
                        | Run::History { .. }
                        | Run::Retry { .. }
                        | Run::Create { .. }
                        | Run::Prepare { .. }
                        | Run::Preparation { .. }
                        | Run::Submission { .. }
                        | Run::Authorize { .. }
                        | Run::Renew { .. }
                        | Run::Execute { .. }
                        | Run::Cancel { .. }
                        | Run::Reconcile { .. }
                        | Run::Refresh { .. }
                        | Run::Checkpoint { .. }
                        | Run::Checkpoints { .. } => Output::Json(submissions::execute(
                            &client,
                            submission_plan.as_ref().expect("validated native plan"),
                            &health,
                        )?),
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
