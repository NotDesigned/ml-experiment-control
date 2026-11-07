use clap::{CommandFactory, Parser, Subcommand};
use ml_exp_client::{
    Client, artifacts, config, credentials, datasets, observation, presentation, projects, read,
    session, sources, status, submissions, watch,
};
use std::{io::Read, path::PathBuf};

#[derive(Parser)]
#[command(
    name = "ml-exp",
    version,
    about = "Submit experiments, inspect progress and retrieve verified results",
    after_help = "Every command, including recovery and administration: ml-exp --help-all"
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
    /// Submit a source directory with its Dockerfile: confirm once, then build and send.
    #[command(
        after_help = "Repeating the same command finds the same job and sends nothing new. \
Change the code, the settings or --seed to run again.\n\n\
For every field and each separate step, see `ml-exp run start` and `ml-exp run execute`."
    )]
    Submit {
        /// Directory holding the code and its Dockerfile.
        #[arg(default_value = ".")]
        directory: PathBuf,
        #[arg(long)]
        gpus: Option<u32>,
        /// Longest the job may run, such as 90s, 10m, 2h or 1d.
        #[arg(long, value_parser = ml_exp_client::submit::parse_time)]
        time: Option<u64>,
        /// Uses the saved project unless given.
        #[arg(long)]
        project: Option<String>,
        /// Uses the saved executor unless given; `ml-exp backend` lists them. A prefix of
        /// fixed shapes, such as `sensecore-n6ls`, takes the smallest shape that fits.
        #[arg(long)]
        executor: Option<String>,
        /// Sized from --gpus unless given. A fixed shape always runs whole.
        #[arg(long)]
        cpus: Option<u32>,
        /// In GB. Sized from --gpus unless given. A fixed shape always runs whole.
        #[arg(long)]
        memory: Option<u32>,
        /// An uploaded dataset, as NAME=asset.HASH. Training reads it through DATASETS_JSON.
        #[arg(long, value_parser = ml_exp_client::submit::parse_data)]
        data: Vec<(String, String)>,
        /// An environment variable for the job, as KEY=VALUE.
        #[arg(long, value_parser = ml_exp_client::submit::parse_env)]
        env: Vec<(String, String)>,
        /// PAI only: `allow_idle` (the default for a job with GPUs) may also use idle
        /// resources and can be preempted; `idle_only` uses only those; `guaranteed` uses
        /// only the quota. A different choice is a different job.
        #[arg(long, value_parser = ["guaranteed", "allow_idle", "idle_only"])]
        allocation: Option<String>,
        /// 0 unless given.
        #[arg(long)]
        seed: Option<i64>,
        /// Names the Run. Derived from the directory and the inputs unless given.
        #[arg(long)]
        run: Option<String>,
        /// Dockerfile inside the directory, `Dockerfile` unless given.
        #[arg(long)]
        dockerfile: Option<String>,
        /// An experiment file for what the options do not cover, such as metrics,
        /// W&B or scheduling. Options given here win.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Skip the question. Required without a terminal.
        #[arg(long)]
        yes: bool,
        /// Return after submitting rather than showing the job's status until it ends.
        #[arg(long)]
        detach: bool,
        /// The command to run in the image, after `--`.
        #[arg(last = true)]
        command: Vec<String>,
    },
    /// Show the latest submission, or the one named with --project and --run.
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
    /// Read the logs of the latest submission, or the one named with --project and --run.
    Logs {
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value = "both", value_parser = ["both", "stdout", "stderr"])]
        stream: String,
        #[arg(long, default_value_t = 80)]
        lines: i64,
    },
    /// Download and verify the results of the latest submission, or of a named Run.
    Results {
        /// Act on this Run rather than the latest submission from this machine.
        #[arg(long)]
        run: Option<String>,
        /// Only needed when two Projects used the same Run name.
        #[arg(long, requires = "run")]
        project: Option<String>,
        /// Named after the Run unless given.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Keep the server's copy after downloading.
        #[arg(long)]
        keep: bool,
        /// Publish results an ended job could not upload, with a zero-GPU job.
        #[arg(long, conflicts_with_all = ["output", "keep"])]
        collect: bool,
    },
    /// Cancel the latest submission or a named Run, after saying which one.
    Cancel {
        /// Act on this Run rather than the latest submission from this machine.
        #[arg(long)]
        run: Option<String>,
        /// Only needed when two Projects used the same Run name.
        #[arg(long, requires = "run")]
        project: Option<String>,
        /// Skip the question. Required without a terminal.
        #[arg(long)]
        yes: bool,
    },
    /// Run the latest submission's Run or a named Run again, from the start or a checkpoint.
    Retry {
        /// Act on this Run rather than the latest submission from this machine.
        #[arg(long)]
        run: Option<String>,
        /// Only needed when two Projects used the same Run name.
        #[arg(long, requires = "run")]
        project: Option<String>,
        /// A checkpoint from `ml-exp run checkpoints`. From the Run's frozen start unless given.
        #[arg(long)]
        checkpoint: Option<String>,
        /// Skip the question. Required without a terminal.
        #[arg(long)]
        yes: bool,
        /// Return after submitting rather than showing the job's status until it ends.
        #[arg(long)]
        detach: bool,
    },
    /// Read scheduler allocation evidence, or explicitly refresh the cache.
    Backend {
        executor: Option<String>,
        #[arg(long, requires = "executor")]
        refresh: bool,
    },
    /// Plan and execute administrative payload removal or Run retirement.
    #[command(hide = true)]
    Admin {
        #[command(subcommand)]
        command: ml_exp_client::administration::Command,
    },
    /// Create, inspect and stop an explicitly leased SSH debug machine.
    #[command(hide = true)]
    Debug {
        #[command(subcommand)]
        command: ml_exp_client::debug::Command,
    },
    /// Build, inspect and reconcile a reusable Dockerfile environment.
    #[command(name = "env")]
    Environment {
        #[command(subcommand)]
        command: ml_exp_client::runtime::Command,
    },
    /// Install a checksum-verified standalone native release.
    #[command(hide = true)]
    Update {
        /// Exact native version; omitted selects the latest stable native release.
        #[arg(long)]
        version: Option<String>,
        #[arg(long)]
        check: bool,
    },
    /// Upload or inspect immutable datasets on a configured backend.
    Data {
        #[command(subcommand)]
        command: Data,
    },
    /// Validate the authenticated connection (no job submission).
    #[command(hide = true)]
    Check,
    /// Read a JSON endpoint under /api/.
    #[command(hide = true)]
    Get { path: String },
    /// Create, submit or observe an experiment through explicit native operations.
    Run {
        #[command(subcommand)]
        command: Run,
    },
    /// Register and manage daemon tracking; research files and jobs are unchanged.
    #[command(hide = true)]
    Project {
        #[command(subcommand)]
        command: Project,
    },
    /// Pack and publish immutable source, or inspect an exact source identity.
    #[command(hide = true)]
    Source {
        #[command(subcommand)]
        command: Source,
    },
    /// Save URL-bound private credentials, or show redacted settings.
    Config {
        #[command(subcommand)]
        command: Config,
    },
}
#[derive(Subcommand)]
enum Data {
    /// Inspect an existing PAI data copy with one read-only, zero-GPU diagnostic.
    Diagnose {
        #[arg(long)]
        project: String,
        #[arg(long)]
        executor: String,
        #[arg(long)]
        asset: String,
        #[arg(long)]
        request_id: String,
        /// Read retained diagnostic evidence only; no provider query or creation.
        #[arg(long, conflicts_with = "provider")]
        status: bool,
        /// Ask the provider for this copy's reason, events and log tail; creates nothing.
        #[arg(long)]
        provider: bool,
        #[arg(long, default_value_t = 200, requires = "provider")]
        lines: u32,
    },
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
    /// Prepare data on persistent storage with a zero-GPU copy (PAI: existing Runtime).
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
        /// PAI: reuse this READY Runtime; no image build is performed.
        #[arg(long)]
        runtime: Option<String>,
    },
    /// Upload a directory. On PAI and SenseCore the parts go straight to object storage;
    /// Slurm has none and an older server may not offer it, so those go through the server.
    Upload {
        #[arg(long)]
        project: String,
        #[arg(long)]
        executor: String,
        #[arg(long)]
        directory: PathBuf,
        /// Send the parts through the server, when object storage cannot be reached.
        #[arg(long)]
        via_server: bool,
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
        token_file: Option<PathBuf>,
        /// The Project `submit` uses when none is given.
        #[arg(long)]
        project: Option<String>,
        /// The executor `submit` uses when none is given.
        #[arg(long)]
        executor: Option<String>,
    },
    Show,
}
#[derive(clap::Args)]
struct Target {
    /// Original private experiment JSON state.
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
    /// Upload an existing local JSONL stream; never start, stop or signal training.
    Attach {
        #[arg(long)]
        project: String,
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        metrics: PathBuf,
        #[arg(long)]
        state: PathBuf,
        #[arg(long, conflicts_with = "finish")]
        watch: bool,
        #[arg(long, value_parser = ["succeeded", "failed", "cancelled"])]
        finish: Option<String>,
    },
    /// Read local ingress cursor and daemon W&B progress; no process/provider probe.
    LocalStatus {
        #[arg(long)]
        project: String,
        #[arg(long)]
        run: String,
    },
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
    /// Advance the same environment/Run preparation; never submit training.
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
        #[arg(long, default_value = "")]
        confirmation: String,
        /// Read retained evidence only; never submit or query the provider.
        #[arg(long, conflicts_with = "retry_failed_job")]
        status: bool,
        /// One CPU retry under the same request, bound to this exact FAILED job.
        #[arg(long)]
        retry_failed_job: Option<String>,
        /// Create or observe one read-only CPU diagnostic; no worker replay.
        #[arg(long, conflicts_with = "retry_failed_job")]
        diagnose: bool,
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
    /// Append a new Attempt from an explicit checkpoint or full restart; does not submit.
    Retry {
        submission: String,
        #[arg(long, default_value = "")]
        confirmation: String,
        #[arg(long)]
        run: String,
        #[arg(long)]
        attempt: String,
        #[arg(
            long,
            required_unless_present = "from_start",
            conflicts_with = "from_start"
        )]
        checkpoint: Option<String>,
        /// Restart from the beginning after the parent job is terminal.
        #[arg(long)]
        from_start: bool,
        #[arg(long, default_value_t = 0.)]
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
        #[arg(long, default_value_t = 0.)]
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
    /// Replace an expired, never-started gate with a fresh one.
    Renew {
        submission: String,
        #[arg(long)]
        request_id: String,
        #[arg(long)]
        confirmation: String,
    },
    /// Dispatch once using the exact confirmation shown by preparation.
    Execute {
        submission: String,
        #[arg(long)]
        confirmation: String,
        /// Why this is being sent; recorded with the execution.
        #[arg(long, default_value = "")]
        reason: String,
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
    /// Ask the provider why this Attempt is in its state: reason, pods, events and log tail.
    Diagnose {
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value_t = 200)]
        lines: u32,
    },
    /// Query arbitrary declared metrics with their original context and units.
    Metrics {
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value_t = 2000)]
        max_points: i64,
        #[arg(long = "name")]
        names: Vec<String>,
    },
}
impl Run {
    fn target(&self) -> Option<&Target> {
        match self {
            Self::Status { target, .. }
            | Self::Logs { target, .. }
            | Self::Diagnose { target, .. }
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
                status,
                retry_failed_job,
                diagnose,
            } => {
                if *diagnose {
                    submissions::Command::SyncDiagnostic {
                        submission,
                        request_id,
                        status: *status,
                    }
                } else if *status {
                    submissions::Command::SyncStatus {
                        submission,
                        request_id,
                    }
                } else if let Some(id) = retry_failed_job {
                    submissions::Command::SyncRetry {
                        submission,
                        request_id,
                        request: ml_exp_core::cloud::exchange::SyncRetry {
                            expected_failed_job_id: id.clone(),
                            confirmation: confirmation.clone(),
                        },
                    }
                } else {
                    submissions::Command::Sync {
                        submission,
                        request_id,
                        confirmation,
                    }
                }
            }
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
            Self::Preparation { project, run } => submissions::Command::Discovery { project, run },
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
            Self::Execute {
                submission,
                confirmation,
                reason,
            } => submissions::Command::Execute {
                submission,
                confirmation,
                reason,
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
fn config_path(selected: Option<PathBuf>) -> PathBuf {
    selected.unwrap_or_else(|| {
        let xdg = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from);
        config::default_config_path(xdg.as_deref(), home.as_deref())
    })
}
fn submissions_directory(config: &std::path::Path) -> PathBuf {
    config
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("submissions")
}
/// The records of the named Run, or the latest submission's when none is named.
fn chosen(
    directory: &std::path::Path,
    project: Option<&str>,
    run: Option<&str>,
) -> Result<PathBuf, ml_exp_client::Error> {
    match run {
        Some(run) => ml_exp_client::submit::run_directory(directory, project, run),
        None => Ok(directory.to_owned()),
    }
}
/// The short commands are the `run` commands, with the latest submission filled in.
fn expand(command: Command, config: &std::path::Path) -> Result<Command, ml_exp_client::Error> {
    let directory = submissions_directory(config);
    let unnamed = |target: &Target| {
        target.state.is_none() && target.project.is_none() && target.run.is_none()
    };
    let fill = |target: &mut Target| -> Result<(), ml_exp_client::Error> {
        if unnamed(target) {
            let latest = ml_exp_client::submit::latest(&directory)?;
            target.project = Some(latest.project);
            target.run = Some(latest.run_id);
        }
        Ok(())
    };
    Ok(match command {
        Command::Status {
            mut target,
            details,
            watch,
            seconds,
        } => {
            // A submit that has not sent a job yet has only its preparation to show.
            if unnamed(&target) {
                let latest = ml_exp_client::submit::latest(&directory)?;
                if latest.submission_id.is_none() {
                    return Ok(Command::Run {
                        command: Run::Preparation {
                            project: latest.project,
                            run: latest.run_id,
                        },
                    });
                }
            }
            fill(&mut target)?;
            Command::Run {
                command: Run::Status {
                    target,
                    details,
                    watch,
                    seconds,
                },
            }
        }
        Command::Logs {
            mut target,
            stream,
            lines,
        } => {
            if unnamed(&target)
                && ml_exp_client::submit::latest(&directory)?
                    .submission_id
                    .is_none()
            {
                return Err(ml_exp_client::Error::Observation(
                    "the latest submission has not sent a job, so it has no logs; \
                     `ml-exp status` shows its preparation",
                ));
            }
            fill(&mut target)?;
            Command::Run {
                command: Run::Logs {
                    target,
                    stream,
                    lines,
                },
            }
        }
        Command::Results {
            output,
            keep,
            collect,
            run,
            project,
        } => {
            let latest = ml_exp_client::submit::latest(&chosen(
                &directory,
                project.as_deref(),
                run.as_deref(),
            )?)?;
            let submission = latest
                .submission_id
                .ok_or(ml_exp_client::Error::Observation(
                    "the latest submission is still being prepared, so it has no results yet",
                ))?;
            let name = format!("{}{}", latest.run_id, sources::RESULTS_SUFFIX);
            Command::Run {
                command: if collect {
                    Run::Collect { submission }
                } else {
                    Run::Download {
                        output: output.unwrap_or_else(|| name.into()),
                        submission,
                        keep,
                    }
                },
            }
        }
        other => other,
    })
}
fn run(mut args: Args) -> Result<(), Box<dyn std::error::Error>> {
    args.command = expand(args.command, &config_path(args.config.clone()))?;
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
            serde_json::to_string_pretty(&ml_exp_client::update::run(version.as_deref(), *check)?)?
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
            Data::Diagnose {
                project,
                executor,
                asset,
                request_id,
                ..
            } => {
                datasets::validate(project, executor, Some(asset))?;
                ml_exp_core::submission::validate_submission_token(request_id)?;
            }
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
                runtime,
            } => {
                datasets::validate(project, executor, Some(asset))?;
                ml_exp_core::submission::validate_submission_token(request_id)?;
                if confirmation != &format!("PUBLISH {asset}") {
                    return Err("invalid exact data publication confirmation".into());
                }
                if runtime.as_deref().is_some_and(|id| {
                    !id.strip_prefix("runtime.").is_some_and(|s| {
                        ml_exp_core::slurm::helper::valid_digest(&format!("sha256:{s}"))
                    })
                }) {
                    return Err("invalid data materialization Runtime identity".into());
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
        ml_exp_core::operations::validate_action_id(submission).map_err(|_| {
            ml_exp_client::Error::Observation("invalid submission identity; no request sent")
        })?;
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
                    from_start,
                    max_gpu_hours,
                    reason,
                },
        } => {
            let request = ml_exp_core::first_submission::native::retry::Request {
                confirmation: confirmation.clone(),
                resume_from: checkpoint.as_ref().map(|checkpoint| {
                    ml_exp_core::checkpoints::Reference {
                        run_id: run.clone(),
                        attempt_id: attempt.clone(),
                        checkpoint_id: checkpoint.clone(),
                    }
                }),
                restart_from: if *from_start {
                    Some(ml_exp_core::first_submission::native::retry::StartFrom {
                        run_id: run.clone(),
                        attempt_id: attempt.clone(),
                    })
                } else {
                    None
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
    let config = config_path(args.config);
    let settings = credentials::load_settings(&config)?;
    let result = match args.command {
        Command::Config {
            command: Config::Show,
        } => Output::Json(settings.public()),
        Command::Config {
            command:
                Config::Set {
                    token_file,
                    project,
                    executor,
                },
        } => {
            if token_file.is_none() && project.is_none() && executor.is_none() {
                return Err("set --token-file, --project or --executor".into());
            }
            for id in [&project, &executor].into_iter().flatten() {
                ml_exp_core::identity::validate_id(id)?;
            }
            let project = project.or_else(|| settings.project.clone());
            let executor = executor.or_else(|| settings.executor.clone());
            let mut settings = match token_file {
                Some(token_file) => {
                    let url = args
                        .url
                        .as_deref()
                        .or(settings.url.as_deref())
                        .ok_or("set --url")?;
                    credentials::settings_from_token_file(url, token_file)?
                }
                None => settings,
            };
            settings.project = project;
            settings.executor = executor;
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
                // reading the credential or sending the health request.
                observation::begin(&plan.base_url, saved.as_ref(), target.selection())?;
            }
            let token = credentials::load_token(plan.token_source)?;
            let client = Client::new(&plan.base_url, token)?;
            match command {
                Command::Submit {
                    directory,
                    gpus,
                    time,
                    project,
                    executor,
                    cpus,
                    memory,
                    data,
                    env,
                    allocation,
                    seed,
                    run,
                    dockerfile,
                    file,
                    yes,
                    detach,
                    command,
                } => {
                    let file = file
                        .map(|path| {
                            std::fs::read(&path).map_err(|_| {
                                format!("cannot read the experiment file {}", path.display())
                            })
                        })
                        .transpose()?;
                    let spec = ml_exp_client::submit::spec(
                        ml_exp_client::submit::Options {
                            project,
                            executor,
                            directory,
                            dockerfile,
                            command,
                            gpus,
                            cpus,
                            memory_gb: memory,
                            walltime_seconds: time,
                            seed,
                            run_id: run,
                            data,
                            env,
                            allocation,
                        },
                        file.as_deref(),
                        settings.project.as_deref(),
                        settings.executor.as_deref(),
                    )?;
                    ml_exp_core::identity::validate_id(&spec.project)?;
                    ml_exp_core::identity::validate_id(&spec.executor)?;
                    report(
                        ml_exp_client::submit::run(
                            &client,
                            &spec,
                            &submissions_directory(&config),
                            hooks(&mut |summary| confirm(summary, yes), &mut printed, !detach),
                        )?,
                        args.json,
                    )?
                }
                Command::Cancel { yes, run, project } => {
                    let value = ml_exp_client::submit::cancel(
                        &client,
                        &chosen(
                            &submissions_directory(&config),
                            project.as_deref(),
                            run.as_deref(),
                        )?,
                        hooks(&mut |summary| confirm(summary, yes), &mut printed, false),
                    )?;
                    if args.json {
                        Output::Json(value)
                    } else {
                        Output::Text(
                            "Cancel requested. `ml-exp status` shows when it has stopped.".into(),
                        )
                    }
                }
                Command::Retry {
                    checkpoint,
                    yes,
                    detach,
                    run,
                    project,
                } => report(
                    ml_exp_client::submit::retry(
                        &client,
                        &chosen(
                            &submissions_directory(&config),
                            project.as_deref(),
                            run.as_deref(),
                        )?,
                        checkpoint,
                        hooks(&mut |summary| confirm(summary, yes), &mut printed, !detach),
                    )?,
                    args.json,
                )?,
                Command::Backend { executor, refresh } => {
                    client.health()?;
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
                    let health = client.health()?;
                    Output::Json(ml_exp_client::administration::execute(
                        &client,
                        admin_plan.ok_or("admin plan unavailable")?,
                        &health,
                    )?)
                }
                Command::Debug { .. } => {
                    let health = client.health()?;
                    Output::Json(ml_exp_client::debug::execute(
                        &client,
                        debug_plan.ok_or("debug plan unavailable")?,
                        &health,
                    )?)
                }
                Command::Environment { .. } => {
                    let health = client.health()?;
                    Output::Json(ml_exp_client::runtime::execute(
                        &client,
                        runtime_plan.ok_or("runtime plan unavailable")?,
                        &health,
                    )?)
                }
                Command::Data { command } => {
                    let health = client.health()?;
                    Output::Json(match command {
                        Data::Diagnose { project,executor,asset,request_id,provider:true,lines,.. } => {
                            if !(1..=2000).contains(&lines) {
                                return Err(ml_exp_client::Error::Observation("diagnose requires 1..2000 lines").into());
                            }
                            client.post(&format!("/api/projects/{project}/datasets/{executor}/{asset}/publications/{request_id}/provider-diagnostics/refresh?lines={lines}"),&serde_json::json!({}))?
                        },
                        Data::Diagnose { project,executor,asset,request_id,status,.. } => {
                            let path=format!("/api/projects/{project}/datasets/{executor}/{asset}/publications/{request_id}/diagnostic");
                            let value=if status { client.get(&path)? } else { client.post(&path,&serde_json::json!({}))? };
                            ml_exp_client::submissions::diagnostic_response(value,&serde_json::json!({"project":project,"executor":executor,"asset_id":asset,"request_id":request_id}))?
                        },
                        Data::ImportOss { project, executor, object_key, sha256, archive_bytes } => datasets::import_oss(&client, &project, &executor,
                            ml_exp_core::datasets::oss::Import { object_key, sha256, archive_bytes }, &health)?,
                        Data::List{project,executor}=>datasets::list(&client,&project,&executor,&health)?,
                        Data::Download{project,executor,asset,output}=>datasets::download(&client,&project,&executor,&asset,&output,&health)?,
                        Data::Publish{project,executor,asset,request_id,confirmation,runtime}=>client.post(&format!("/api/projects/{project}/datasets/{executor}/{asset}/publish"),&serde_json::json!({"request_id":request_id,"confirmation":confirmation,"runtime_id":runtime}))?,
                        Data::Upload {
                            project,
                            executor,
                            directory,
                            via_server,
                        } => datasets::upload(
                            &client,
                            &project,
                            &executor,
                            &directory,
                            &health,
                            via_server,
                        )?,
                        Data::Show {
                            project,
                            executor,
                            asset,
                        } => datasets::show(&client, &project, &executor, &asset, &health)?,
                    })
                }
                Command::Check => Output::Json(client.health()?),
                Command::Get { path } => Output::Json(client.get(&path)?),
                Command::Project { .. } => {
                    let health = client.health()?;
                    Output::Json(projects::execute(
                        &client,
                        project_plan.as_ref().expect("validated project plan"),
                        &health,
                    )?)
                }
                Command::Source { command } => {
                    let health = client.health()?;
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
                Command::Run {
                    command:
                        Run::Attach {
                            project,
                            file,
                            metrics,
                            state,
                            watch,
                            finish,
                        },
                } => {
                    let finish = finish
                        .map(|f| serde_json::from_value(serde_json::json!(f.to_uppercase())))
                        .transpose()?;
                    let mode = if watch {
                        ml_exp_client::local::Mode::Watch
                    } else {
                        finish
                            .map(ml_exp_client::local::Mode::Finish)
                            .unwrap_or(ml_exp_client::local::Mode::Once)
                    };
                    Output::Json(ml_exp_client::local::attach(
                        &client,
                        &project,
                        &file,
                        &metrics,
                        &state,
                        mode,
                        |value| {
                            if watch {
                                println!("{}", value);
                            }
                        },
                    )?)
                }
                Command::Run {
                    command: Run::LocalStatus { project, run },
                } => {
                    client.health()?;
                    Output::Json(ml_exp_client::local::status(&client, &project, &run)?)
                }
                Command::Run { command } => {
                    let health = client.health()?;
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
                        Run::Attach { .. } | Run::LocalStatus { .. } => {
                            unreachable!("local commands handled above")
                        }
                        Run::Tracking { .. }
                        | Run::Sync { .. }
                        | Run::History { .. }
                        | Run::Retry { .. }
                        | Run::Create { .. }
                        | Run::Prepare { .. }
                        | Run::Preparation { .. }
                        | Run::Submission { .. }
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
                            saved.as_ref(),
                            target.selection(),
                            read::View::Logs { stream, lines },
                        )?),
                        Run::Diagnose { target, lines } => {
                            let value = read::provider_diagnostics(
                                &client,
                                saved.as_ref(),
                                target.selection(),
                                lines,
                            )?;
                            if args.json {
                                Output::Json(value)
                            } else {
                                Output::Text(presentation::provider_diagnostics(&value))
                            }
                        }
                        Run::Metrics {
                            target,
                            max_points,
                            names,
                        } => Output::Json(read::observe(
                            &client,
                            saved.as_ref(),
                            target.selection(),
                            read::View::Metrics {
                                max_points,
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
type Confirm<'a> = &'a mut dyn FnMut(&str) -> Result<bool, ml_exp_client::Error>;
fn hooks<'a>(
    confirm: Confirm<'a>,
    progress: &'a mut dyn FnMut(&str),
    follow: bool,
) -> ml_exp_client::submit::Hooks<'a> {
    ml_exp_client::submit::Hooks {
        confirm,
        progress,
        poll: std::time::Duration::from_secs(5),
        deadline: std::time::Duration::from_secs(6 * 3600),
        silence: std::time::Duration::from_secs(600),
        follow,
        follow_poll: std::time::Duration::from_secs(30),
    }
}
fn printed(line: &str) {
    eprintln!("{line}");
}
/// The output is complete, and the command still fails if the job did not succeed.
fn report(value: serde_json::Value, json: bool) -> Result<Output, Box<dyn std::error::Error>> {
    let failed = ml_exp_client::submit::failure(&value);
    let output = if json {
        Output::Json(value)
    } else {
        Output::Text(ml_exp_client::submit::report(&value))
    };
    let Some(reason) = failed else {
        return Ok(output);
    };
    match output {
        Output::Json(value) => println!("{}", serde_json::to_string_pretty(&value)?),
        Output::Text(text) => println!("{text}"),
    }
    Err(reason.into())
}
/// Show what will be spent and ask, once. No terminal means no one can answer.
fn confirm(summary: &str, yes: bool) -> Result<bool, ml_exp_client::Error> {
    use std::io::{IsTerminal, Write};
    eprintln!("{summary}");
    if yes {
        return Ok(true);
    }
    if !std::io::stdin().is_terminal() {
        return Err(ml_exp_client::Error::Observation(
            "nothing was done: pass --yes to go ahead without a terminal",
        ));
    }
    eprint!("Proceed? [y/N] ");
    std::io::stderr().flush().ok();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|_| ml_exp_client::Error::Observation("nothing was done: no answer"))?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes"))
}
/// The default help lists what most people need. This one hides nothing.
fn help_all() -> std::process::ExitCode {
    let mut command = Args::command();
    let names: Vec<String> = command
        .get_subcommands()
        .map(|sub| sub.get_name().to_owned())
        .collect();
    for name in names {
        command = command.mut_subcommand(name, |sub| sub.hide(false));
    }
    if command.print_help().is_err() {
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}
fn client() -> std::process::ExitCode {
    if std::env::args().nth(1).as_deref() == Some("--help-all") {
        return help_all();
    }
    match run(Args::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
/// The command tree is large, and it outgrew the 1 MB main thread of a Windows debug build.
const STACK_BYTES: usize = 8 * 1024 * 1024;
fn main() -> std::process::ExitCode {
    let thread = std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(client);
    match thread.map(std::thread::JoinHandle::join) {
        Ok(Ok(code)) => code,
        Ok(Err(_)) => std::process::ExitCode::FAILURE,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
