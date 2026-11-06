//! The common case in one command: pack the source, confirm once, prepare and submit.
//! Every step is an existing operation. The identity comes from the inputs, so the
//! same command finds the same Submission again and never sends a second job.
use crate::{
    Client, Error, observation, preparations, presentation, read, runtime, sources, submissions,
};
use ml_exp_core::{canonical, identity, preparation};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const DOCKERFILE_LIMIT: u64 = 256 * 1024;
const REASON: &str = "ml-exp submit";
/// Build progress is read on every poll, and the build is driven on every Nth.
const ADVANCE_EVERY: u32 = 6;
/// Failed reads in a row before watching stops. The job is not affected.
const READ_FAILURES: u32 = 10;

#[derive(Default)]
pub struct Spec {
    pub project: String,
    pub executor: String,
    pub directory: PathBuf,
    pub dockerfile: String,
    pub command: Vec<String>,
    pub gpus: u32,
    pub cpus: Option<u32>,
    pub memory_gb: Option<u32>,
    pub walltime_seconds: u64,
    pub seed: i64,
    pub run_id: Option<String>,
    pub data: Vec<(String, String)>,
    pub env: BTreeMap<String, String>,
    /// Further Run fields from a configuration file, such as `metrics_schema`.
    pub extra: Map<String, Value>,
    pub request_id: Option<String>,
    pub max_gpu_hours: Option<f64>,
    pub reason: Option<String>,
}

/// What the command line gave. Anything it leaves out can come from a file or the
/// saved settings, and the command line wins.
#[derive(Default)]
pub struct Options {
    pub project: Option<String>,
    pub executor: Option<String>,
    pub directory: PathBuf,
    pub dockerfile: Option<String>,
    pub command: Vec<String>,
    pub gpus: Option<u32>,
    pub cpus: Option<u32>,
    pub memory_gb: Option<u32>,
    pub walltime_seconds: Option<u64>,
    pub seed: Option<i64>,
    pub run_id: Option<String>,
    pub data: Vec<(String, String)>,
    pub env: Vec<(String, String)>,
}

/// The experiment file `run start` reads, with everything `submit` derives optional.
/// A fixed source or Runtime is refused: `submit` always packs the directory.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    request_id: Option<String>,
    #[serde(default)]
    runtime: FileRuntime,
    #[serde(default)]
    run: FileRun,
    #[serde(default)]
    approval: FileApproval,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileRuntime {
    dockerfile: Option<String>,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileApproval {
    max_gpu_hours: Option<f64>,
    reason: Option<String>,
}
#[derive(Default, Deserialize)]
struct FileRun {
    executor: Option<String>,
    command: Option<Vec<String>>,
    #[serde(default)]
    resources: FileResources,
    walltime_seconds: Option<u64>,
    seed: Option<i64>,
    run_id: Option<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    data: Vec<FileData>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileResources {
    gpus: Option<u32>,
    cpus: Option<u32>,
    memory_gb: Option<u32>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileData {
    name: String,
    asset_id: String,
}

/// Merge the command line, an optional experiment file and the saved settings.
pub fn spec(
    options: Options,
    file: Option<&[u8]>,
    saved_project: Option<&str>,
    saved_executor: Option<&str>,
) -> Result<Spec, Error> {
    let file: File = match file {
        Some(bytes) => {
            let value = ml_exp_core::json::mapping(bytes)
                .map_err(|_| Error::Submit("the experiment file is not a JSON object".into()))?;
            serde_json::from_value(Value::Object(value)).map_err(|error| {
                Error::Submit(format!(
                    "the experiment file is not usable: {error}. \
                     `submit` derives the source and base image itself; \
                     use `ml-exp run start` to fix them"
                ))
            })?
        }
        None => File::default(),
    };
    if file.run.extra.contains_key("runtime_id") {
        return Err(Error::Submit(
            "run.runtime_id is not used by `submit`, which builds from the directory; \
             use `ml-exp run start` to reuse a Runtime"
                .into(),
        ));
    }
    let needed = |what: &str| Error::Submit(format!("nothing says {what}"));
    let mut data: Vec<(String, String)> = file
        .run
        .data
        .into_iter()
        .map(|input| (input.name, input.asset_id))
        .collect();
    data.extend(options.data);
    let mut env = file.run.env;
    env.extend(options.env);
    let command = if options.command.is_empty() {
        file.run.command.unwrap_or_default()
    } else {
        options.command
    };
    if command.is_empty() {
        return Err(needed("what to run: put the command after `--`"));
    }
    Ok(Spec {
        project: options
            .project
            .or_else(|| saved_project.map(str::to_owned))
            .ok_or_else(|| {
                needed("which Project: pass --project, or save one with `ml-exp config set --project P`")
            })?,
        executor: options
            .executor
            .or(file.run.executor)
            .or_else(|| saved_executor.map(str::to_owned))
            .ok_or_else(|| {
                needed(
                    "which executor: pass --executor, or save one with `ml-exp config set --executor E`; \
                     `ml-exp backend` lists them",
                )
            })?,
        directory: options.directory,
        dockerfile: options
            .dockerfile
            .or(file.runtime.dockerfile)
            .unwrap_or_else(|| "Dockerfile".into()),
        command,
        gpus: options
            .gpus
            .or(file.run.resources.gpus)
            .ok_or_else(|| needed("how many GPUs: pass --gpus"))?,
        cpus: options.cpus.or(file.run.resources.cpus),
        memory_gb: options.memory_gb.or(file.run.resources.memory_gb),
        walltime_seconds: options
            .walltime_seconds
            .or(file.run.walltime_seconds)
            .ok_or_else(|| needed("how long it may run: pass --time"))?,
        seed: options.seed.or(file.run.seed).unwrap_or(0),
        run_id: options.run_id.or(file.run.run_id),
        data,
        env,
        extra: file.run.extra,
        request_id: file.request_id,
        max_gpu_hours: file.approval.max_gpu_hours,
        reason: file.approval.reason,
    })
}

/// Everything derived before anything is sent.
#[derive(Debug, PartialEq)]
pub struct Plan {
    pub request_id: String,
    pub preparation_id: String,
    pub run_id: String,
    pub base_image: String,
    pub executor: String,
    pub gpus: u32,
    pub cpus: u32,
    pub memory_gb: u32,
    pub max_gpu_hours: f64,
}

pub struct Hooks<'a> {
    /// Shown what will be spent. Returning false stops before anything is sent.
    pub confirm: &'a mut dyn FnMut(&str) -> Result<bool, Error>,
    pub progress: &'a mut dyn FnMut(&str),
    pub poll: Duration,
    pub deadline: Duration,
    /// Keep showing the backend's status until the job ends.
    pub follow: bool,
    pub follow_poll: Duration,
}

/// The last submission from this machine, so short commands need no identifiers.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Latest {
    pub project: String,
    pub run_id: String,
    pub submission_id: String,
}
const LATEST: &str = "latest.json";
/// Only a convenience: failing to save this must not disturb a submission.
fn remember(directory: &Path, latest: &Latest) {
    if let Ok(bytes) = serde_json::to_vec(latest) {
        let _ = ml_exp_io::write_atomic(&directory.join(LATEST), &bytes, false);
    }
}
pub fn latest(directory: &Path) -> Result<Latest, Error> {
    let none = || {
        Error::Observation(
            "no submission from this machine yet; use `ml-exp submit`, or name one with --project and --run",
        )
    };
    let bytes = ml_exp_io::read_private(&directory.join(LATEST), 4096, true).map_err(|_| none())?;
    serde_json::from_slice(&bytes).map_err(|_| none())
}

/// `90`, `90s`, `10m`, `2h` or `1d`.
pub fn parse_time(value: &str) -> Result<u64, String> {
    let (number, unit) = match value.char_indices().last() {
        Some((index, unit)) if unit.is_ascii_alphabetic() => (&value[..index], unit),
        _ => (value, 's'),
    };
    let factor = match unit {
        's' => 1,
        'm' => 60,
        'h' => 3600,
        'd' => 86400,
        _ => return Err("use a time such as 90s, 10m, 2h or 1d".into()),
    };
    number
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(factor))
        .filter(|seconds| *seconds > 0)
        .ok_or_else(|| "use a time such as 90s, 10m, 2h or 1d".into())
}

/// `KEY=VALUE`.
pub fn parse_env(value: &str) -> Result<(String, String), String> {
    value
        .split_once('=')
        .filter(|(key, _)| !key.is_empty())
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .ok_or_else(|| "use KEY=VALUE".into())
}

/// `NAME=asset.HASH`.
pub fn parse_data(value: &str) -> Result<(String, String), String> {
    value
        .split_once('=')
        .filter(|(name, asset)| !name.is_empty() && asset.starts_with("asset."))
        .map(|(name, asset)| (name.to_owned(), asset.to_owned()))
        .ok_or_else(|| "use NAME=asset.HASH; ml-exp data upload returns the asset".into())
}

/// The last external image, which the builder requires `base_image` to equal. The
/// builder's parser stays the authority: this only saves writing the digest twice.
pub fn base_image(dockerfile: &str) -> Result<String, Error> {
    let mut stages = Vec::new();
    let mut last = None;
    let joined = dockerfile.replace("\\\r\n", " ").replace("\\\n", " ");
    for line in joined.lines() {
        let mut words = line.split_whitespace();
        if !words
            .next()
            .is_some_and(|word| word.eq_ignore_ascii_case("FROM"))
        {
            continue;
        }
        let parts: Vec<&str> = words.filter(|word| !word.starts_with("--")).collect();
        let Some(image) = parts.first() else {
            return Err(Error::Submit("the Dockerfile has an empty FROM".into()));
        };
        let image = image.to_string();
        if image != "scratch" && !stages.contains(&image.to_ascii_lowercase()) {
            if !pinned(&image) {
                return Err(Error::Submit(format!(
                    "FROM {image} must be pinned as IMAGE@sha256:DIGEST; \
                     `ml-exp get /api/environments` lists approved bases"
                )));
            }
            last = Some(image);
        }
        if parts.len() == 3 && parts[1].eq_ignore_ascii_case("AS") {
            stages.push(parts[2].to_ascii_lowercase());
        }
    }
    last.ok_or_else(|| {
        Error::Submit(
            "the Dockerfile needs a FROM pinned as IMAGE@sha256:DIGEST; \
             `ml-exp get /api/environments` lists approved bases"
                .into(),
        )
    })
}
fn pinned(image: &str) -> bool {
    image.split_once("@sha256:").is_some_and(|(name, digest)| {
        !name.is_empty()
            && digest.len() == 64
            && digest
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    })
}

/// `--executor` names one executor, or a family of fixed shapes by its prefix. A family
/// takes the smallest shape that holds the request, and the question shows which.
pub fn choose<'a>(spec: &Spec, rows: &'a [Value]) -> Result<&'a Value, Error> {
    if let Some(row) = rows.iter().find(|row| row["executor_id"] == spec.executor) {
        return Ok(row);
    }
    let prefix = format!("{}-", spec.executor);
    let family: Vec<&Value> = rows
        .iter()
        .filter(|row| {
            row["limits"]["fixed_resources"] == true
                && row["enabled"] == true
                && row["run_creation"] == true
                && row["executor_id"]
                    .as_str()
                    .is_some_and(|id| id.starts_with(&prefix))
        })
        .collect();
    if family.is_empty() {
        return Err(Error::Submit(format!(
            "no executor named {}; `ml-exp backend` lists them",
            spec.executor
        )));
    }
    let size = |row: &Value, key: &str| row["limits"][key].as_u64().unwrap_or(0);
    family
        .into_iter()
        .filter(|row| {
            u64::from(spec.gpus) <= size(row, "gpus")
                && u64::from(spec.cpus.unwrap_or(0)) <= size(row, "cpus")
                && u64::from(spec.memory_gb.unwrap_or(0)) <= size(row, "memory_gb")
        })
        .min_by_key(|row| {
            (
                size(row, "gpus"),
                size(row, "cpus"),
                size(row, "memory_gb"),
                row["executor_id"].to_string(),
            )
        })
        .ok_or_else(|| {
            Error::Submit(format!(
                "no {} shape holds this request; `ml-exp backend` lists them",
                spec.executor
            ))
        })
}

/// GPUs, CPUs and memory for this executor. A fixed shape always runs whole, so the
/// request only has to fit. Otherwise they are sized from the GPU count unless given.
pub fn resources(spec: &Spec, executor: &str, limits: &Value) -> Result<(u32, u32, u32), Error> {
    let limit = |key: &str| limits[key].as_u64();
    let cap = |wanted: u32, key: &str| match limit(key) {
        Some(max) => wanted.min(u32::try_from(max).unwrap_or(u32::MAX)),
        None => wanted,
    };
    let exceeded = |what: &str, wanted: u64, max: u64| {
        Error::Submit(format!(
            "{what} {wanted} exceeds {max} on {executor}; `ml-exp backend` lists each executor's limits"
        ))
    };
    if let Some(max) = limit("gpus").filter(|max| u64::from(spec.gpus) > *max) {
        return Err(exceeded("--gpus", spec.gpus.into(), max));
    }
    if let Some(max) = limit("walltime_seconds").filter(|max| spec.walltime_seconds > *max) {
        return Err(exceeded("--time in seconds", spec.walltime_seconds, max));
    }
    let scaled = |each: u32, minimum: u32| spec.gpus.saturating_mul(each).max(minimum);
    let fixed = limits["fixed_resources"] == true;
    let (gpus, default_cpus, default_memory) = if fixed {
        (
            cap(u32::MAX, "gpus"),
            cap(u32::MAX, "cpus"),
            cap(u32::MAX, "memory_gb"),
        )
    } else {
        (
            spec.gpus,
            cap(scaled(8, 2), "cpus"),
            cap(scaled(32, 4), "memory_gb"),
        )
    };
    if spec.gpus == 0 && gpus > 0 {
        return Err(Error::Submit(format!(
            "{executor} always has {gpus} GPUs; use a CPU executor for --gpus 0"
        )));
    }
    let cpus = spec.cpus.unwrap_or(default_cpus);
    let memory = spec.memory_gb.unwrap_or(default_memory);
    if let Some(max) = limit("cpus").filter(|max| u64::from(cpus) > *max) {
        return Err(exceeded("--cpus", cpus.into(), max));
    }
    if let Some(max) = limit("memory_gb").filter(|max| u64::from(memory) > *max) {
        return Err(exceeded("--memory", memory.into(), max));
    }
    Ok(if fixed {
        (gpus, default_cpus, default_memory)
    } else {
        (gpus, cpus, memory)
    })
}

pub fn plan(
    spec: &Spec,
    server: &str,
    archive_sha256: &str,
    dockerfile: &str,
    executor: &Value,
    directory_name: &str,
) -> Result<Plan, Error> {
    let base_image = base_image(dockerfile)?;
    let chosen = executor["executor_id"]
        .as_str()
        .ok_or(Error::Response)?
        .to_owned();
    let (gpus, cpus, memory_gb) = resources(spec, &chosen, &executor["limits"])?;
    ml_exp_core::environment::validate(&spec.env).map_err(|_| {
        Error::Submit(
            "an environment variable has an invalid or reserved name; \
             the platform sets directories, Run identity and ML_EXPD_ names itself"
                .into(),
        )
    })?;
    // The reason and display text stay out: only what changes the job is identity.
    let identity = canonical::digest(&json!([
        "ml-exp-submit.v1",
        server,
        spec.project,
        chosen,
        archive_sha256,
        spec.dockerfile,
        base_image,
        spec.command,
        gpus,
        cpus,
        memory_gb,
        spec.walltime_seconds,
        spec.seed,
        spec.run_id,
        spec.data,
        spec.env,
        spec.extra,
    ]))
    .map_err(|_| Error::Response)?;
    let request_id = spec
        .request_id
        .clone()
        .unwrap_or_else(|| identity[..32].to_owned());
    let run_id = match &spec.run_id {
        Some(run) => run.clone(),
        None => format!("{}-{}", slug(directory_name), &identity[..8]),
    };
    identity::validate_id(&run_id)
        .map_err(|_| Error::Submit(format!("{run_id} is not a valid Run name")))?;
    let preparation_id = preparation::id(&spec.project, &request_id).map_err(Error::Observation)?;
    Ok(Plan {
        request_id,
        preparation_id,
        run_id,
        base_image,
        executor: chosen,
        gpus,
        cpus,
        memory_gb,
        max_gpu_hours: spec
            .max_gpu_hours
            .unwrap_or((f64::from(gpus) * spec.walltime_seconds as f64 / 36.0).ceil() / 100.0),
    })
}
impl Plan {
    fn latest(&self, spec: &Spec, submission: &str) -> Latest {
        Latest {
            project: spec.project.clone(),
            run_id: self.run_id.clone(),
            submission_id: submission.to_owned(),
        }
    }
}
fn slug(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .take(40)
        .collect();
    let cleaned = cleaned.trim_start_matches(['-', '_']);
    if cleaned.is_empty() {
        "run".into()
    } else {
        cleaned.into()
    }
}

/// What the user agrees to. It must show everything that decides the spend.
pub fn summary(spec: &Spec, plan: &Plan) -> String {
    format!(
        "Run {}\n  executor  {}\n  command   {}\n  image     {}\n  resources {} GPU, {} CPU, {} GB\n  time      up to {} seconds\n  spend     up to {} GPU hours",
        plan.run_id,
        plan.executor,
        spec.command.join(" "),
        plan.base_image,
        plan.gpus,
        plan.cpus,
        plan.memory_gb,
        spec.walltime_seconds,
        plan.max_gpu_hours,
    )
}

fn request(spec: &Spec, plan: &Plan, source_id: &str) -> Result<preparation::Request, Error> {
    let mut run = json!({
        "run_id": plan.run_id,
        "executor": plan.executor,
        "command": spec.command,
        "resources": {"gpus": plan.gpus, "cpus": plan.cpus, "memory_gb": plan.memory_gb},
        "walltime_seconds": spec.walltime_seconds,
        "seed": spec.seed,
    });
    if !spec.env.is_empty() {
        run["env"] = json!(spec.env);
    }
    if !spec.data.is_empty() {
        run["data"] = spec
            .data
            .iter()
            .map(|(name, asset)| json!({"name": name, "asset_id": asset}))
            .collect();
    }
    if let Value::Object(run) = &mut run {
        run.extend(spec.extra.clone());
    }
    let request: preparation::Request = serde_json::from_value(json!({
        "request_id": plan.request_id,
        "runtime": {
            "source_id": source_id,
            "executor": plan.executor,
            "base_image": plan.base_image,
            "dockerfile": spec.dockerfile,
        },
        "run": run,
        "approval": {
            "max_gpu_hours": plan.max_gpu_hours,
            "reason": spec.reason.as_deref().unwrap_or(REASON),
        },
    }))
    .map_err(|_| Error::Response)?;
    request
        .validate(&spec.project)
        .map_err(Error::Observation)?;
    Ok(request)
}

/// Why waiting longer cannot help. The next run of the command is the retry.
fn stopped(view: &Value) -> Option<String> {
    if let Some(code) = view["error_code"].as_str() {
        return Some(format!("data preparation failed with {code}"));
    }
    let runtime = &view["runtime"];
    if runtime["status"] == "PREPARED" && !runtime["build_error"].is_null() {
        let code = runtime["build_error"]["code"].as_str().unwrap_or("UNKNOWN");
        let id = runtime["runtime_id"].as_str().unwrap_or("RUNTIME");
        return Some(format!(
            "the image build failed with {code}; see `ml-exp env logs {id}`"
        ));
    }
    if view["data"]["new_request_allowed"] == true {
        return Some(
            "the dataset copy failed; see `ml-exp data diagnose`, then run this command again"
                .into(),
        );
    }
    None
}
/// The Runtime still being built, if any. Progress is read for this one.
fn building(view: &Value) -> Option<&str> {
    let runtime = &view["runtime"];
    matches!(
        runtime["status"].as_str(),
        Some("PREPARED" | "EXECUTING" | "RECONCILE_REQUIRED")
    )
    .then(|| runtime["runtime_id"].as_str())
    .flatten()
}
fn stage(view: &Value) -> &'static str {
    if building(view).is_some() {
        "building the image"
    } else if view["phase"] == "DATA_PREPARATION" {
        "preparing data"
    } else {
        "preparing"
    }
}
/// Only the lines not shown yet. Each read returns the end of the log, so the two
/// reads overlap wherever the end of the old one is the start of the new one.
pub fn new_lines<'a>(shown: &[String], current: &'a [String]) -> &'a [String] {
    let overlap = (0..=shown.len().min(current.len()))
        .rev()
        .find(|size| shown[shown.len() - size..] == current[..*size])
        .unwrap_or(0);
    &current[overlap..]
}
#[derive(Default)]
struct BuildView {
    phase: String,
    lines: Vec<String>,
    warned: bool,
}
/// Best effort: a failed read only means less to show, and never affects the build.
fn show_build(
    client: &Client,
    spec: &Spec,
    runtime: &str,
    health: &Value,
    view: &mut BuildView,
    progress: &mut dyn FnMut(&str),
) {
    let command = runtime::Command::Progress {
        project: spec.project.clone(),
        runtime: runtime.to_owned(),
    };
    let Some(value) = command
        .plan()
        .and_then(|plan| runtime::execute(client, plan, health))
        .ok()
    else {
        return;
    };
    let state = &value["progress"];
    if let Some(phase) = state["phase"].as_str().filter(|phase| *phase != view.phase) {
        progress(&match state["message"].as_str() {
            Some(message) => format!("[build] {phase}: {message}"),
            None => format!("[build] {phase}"),
        });
        view.phase = phase.to_owned();
    }
    let lines: Vec<String> = value["lines"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|line| line.as_str().map(str::to_owned))
        .collect();
    for line in new_lines(&view.lines, &lines) {
        progress(&format!("  {line}"));
    }
    view.lines = lines;
    if state["no_progress_warning"] == true && !view.warned {
        progress("[build] no new output for two minutes; it may still be working");
    }
    view.warned = state["no_progress_warning"] == true;
}

fn show(client: &Client, id: &str, health: &Value) -> Result<Value, Error> {
    submissions::execute(
        client,
        &submissions::plan(submissions::Command::Show { submission: id })?,
        health,
    )
}

/// Past PREPARED and AUTHORIZED, the job may exist. Only observing is allowed then.
fn dispatched(status: &Value) -> bool {
    !status
        .as_str()
        .is_some_and(|status| matches!(status, "PREPARED" | "AUTHORIZED"))
}

fn dispatch(client: &Client, id: &str, health: &Value) -> Result<Value, Error> {
    let mut current = show(client, id, health)?;
    if current["next_action"] == "REPREPARE" {
        return Err(Error::Submit(format!(
            "the prepared Submission expired before it was sent, and nothing was submitted; \
             run `ml-exp run renew {id}`"
        )));
    }
    if current["status"] == "PREPARED" {
        current = submissions::execute(
            client,
            &submissions::plan(submissions::Command::Authorize {
                submission: id,
                reason: REASON,
            })?,
            health,
        )?;
    }
    if current["status"] != "AUTHORIZED" {
        return Ok(current);
    }
    let confirmation = current["confirmation"]
        .as_str()
        .ok_or(Error::Response)?
        .to_owned();
    submissions::execute(
        client,
        &submissions::plan(submissions::Command::Execute {
            submission: id,
            confirmation: &confirmation,
        })?,
        health,
    )
}

/// A reply was lost after the claim: find out what happened to the original job.
fn observe(client: &Client, id: &str, current: Value, health: &Value) -> Result<Value, Error> {
    if matches!(
        current["status"].as_str(),
        Some("EXECUTING" | "RECONCILE_REQUIRED")
    ) {
        submissions::execute(
            client,
            &submissions::plan(submissions::Command::Reconcile { submission: id })?,
            health,
        )
    } else {
        Ok(current)
    }
}

/// The scheduler's state, normalized, from either kind of backend.
fn job_state(submission: &Value) -> Option<&str> {
    submission
        .pointer("/cloud_observation/observation/state")
        .or_else(|| submission.pointer("/execution/result/observation/status/state"))
        .and_then(Value::as_str)
}
fn finished(state: &str) -> bool {
    matches!(
        state,
        "SUCCEEDED" | "FAILED" | "CANCELLED" | "TIMEOUT" | "PREEMPTED"
    )
}
/// The backend's own words when it gives them, such as PAI's `EnvPreparing`.
fn describe(submission: &Value) -> Option<String> {
    let state = job_state(submission)?;
    let observation = &submission["cloud_observation"]["observation"];
    let own = observation["provider_state"]
        .as_str()
        .filter(|own| !own.eq_ignore_ascii_case(state));
    let mut line = match own {
        Some(own) => format!("[job] {state} ({own})"),
        None => format!("[job] {state}"),
    };
    if let Some(reason) = observation["queue"]["reason"].as_str() {
        line.push_str(&format!(": {reason}"));
    }
    Some(line)
}
/// Read-only. Stopping here, for any reason, leaves the job running.
fn follow(
    client: &Client,
    id: &str,
    current: Value,
    health: &Value,
    hooks: &mut Hooks<'_>,
) -> Value {
    (hooks.progress)("Ctrl-C stops watching; the job keeps running");
    let refresh = |client: &Client| {
        submissions::plan(submissions::Command::Refresh { submission: id })
            .and_then(|plan| submissions::execute(client, &plan, health))
    };
    let mut latest = current;
    let mut last = String::new();
    let mut failures = 0;
    loop {
        match refresh(client) {
            Ok(value) => {
                failures = 0;
                latest = value;
                if let Some(line) = describe(&latest).filter(|line| *line != last) {
                    (hooks.progress)(&line);
                    last = line;
                }
                if job_state(&latest).is_some_and(finished) {
                    return latest;
                }
            }
            Err(_) if failures < READ_FAILURES => failures += 1,
            Err(_) => {
                (hooks.progress)(
                    "stopped watching: the status cannot be read. The job keeps running",
                );
                return latest;
            }
        }
        std::thread::sleep(hooks.follow_poll);
    }
}
/// What the backend says went wrong, when a job ended without success.
fn explain_failure(client: &Client, project: &str, run: &str, hooks: &mut Hooks<'_>) {
    let selection = observation::Selection {
        project: Some(project),
        run: Some(run),
        attempt: None,
    };
    if let Ok(value) = read::provider_diagnostics(client, None, selection, 40) {
        (hooks.progress)(&presentation::provider_diagnostics(&value));
    }
}
fn watch(
    client: &Client,
    project: &str,
    run: &str,
    id: &str,
    current: Value,
    health: &Value,
    hooks: &mut Hooks<'_>,
) -> Value {
    if !hooks.follow || current["status"] != "VERIFIED" {
        return current;
    }
    let latest = follow(client, id, current, health, hooks);
    if job_state(&latest).is_some_and(|state| finished(state) && state != "SUCCEEDED") {
        explain_failure(client, project, run, hooks);
    }
    latest
}

fn outcome(spec: &Spec, plan: &Plan, state: &Path, submission: &Value, existing: bool) -> Value {
    let id = submission["submission_id"].as_str().unwrap_or_default();
    let state = state.display();
    json!({
        "project": spec.project,
        "run_id": plan.run_id,
        "preparation_id": plan.preparation_id,
        "submission_id": id,
        "status": submission["status"],
        "job_state": job_state(submission),
        "already_submitted": existing,
        "state_file": state.to_string(),
        "next": ["ml-exp status", "ml-exp logs", "ml-exp results"],
    })
}

/// The default output. `--json` gives the same fields.
pub fn report(value: &Value) -> String {
    let text = |key: &str| value[key].as_str().unwrap_or_default();
    let mut lines = vec![format!(
        "{} {}: {}",
        if value["already_submitted"] == true {
            "Already submitted"
        } else {
            "Submitted"
        },
        text("run_id"),
        value["job_state"].as_str().unwrap_or(text("status"))
    )];
    lines.extend(
        value["next"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|next| format!("  {next}")),
    );
    lines.join("\n")
}

pub fn run(
    client: &Client,
    spec: &Spec,
    state_directory: &Path,
    mut hooks: Hooks<'_>,
) -> Result<Value, Error> {
    let health = client.health()?;
    let executors = client.get("/api/executors")?;
    let rows = executors["executors"]
        .as_array()
        .ok_or(Error::Response)?
        .as_slice();
    let executor = choose(spec, rows)?;
    let dockerfile = read_dockerfile(&spec.directory, &spec.dockerfile)?;
    let archive = sources::pack_directory(&spec.directory)?;
    let name = spec
        .directory
        .canonicalize()
        .ok()
        .and_then(|path| path.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
    let plan = plan(
        spec,
        client.base_url(),
        &archive.sha256,
        &dockerfile,
        executor,
        &name,
    )?;
    ml_exp_io::create_private_dir(state_directory).map_err(|_| Error::Config)?;
    let state = state_directory.join(format!("{}.json", plan.preparation_id));
    let path = format!(
        "/api/projects/{}/experiment-preparations/{}",
        spec.project, plan.preparation_id
    );
    let saved = match client.get(&path) {
        Ok(view) => Some(view),
        Err(Error::Http(failure)) if failure.status == 404 => None,
        Err(error) => return Err(error),
    };
    if let Some(id) = saved
        .as_ref()
        .and_then(|view| view["submission"]["submission_id"].as_str())
    {
        let current = show(client, id, &health)?;
        if dispatched(&current["status"]) {
            remember(state_directory, &plan.latest(spec, id));
            let current = observe(client, id, current, &health)?;
            let current = watch(
                client,
                &spec.project,
                &plan.run_id,
                id,
                current,
                &health,
                &mut hooks,
            );
            return Ok(outcome(spec, &plan, &state, &current, true));
        }
    }
    if executor["enabled"] != true || executor["run_creation"] != true {
        return Err(Error::Submit(format!(
            "{} is not accepting new Runs",
            plan.executor
        )));
    }
    if health["scheduler_mutations"] != true {
        return Err(Error::Observation(
            "the server does not allow submissions right now; nothing was sent",
        ));
    }
    if !(hooks.confirm)(&summary(spec, &plan))? {
        return Err(Error::Observation("not submitted; nothing was sent"));
    }
    let source_id = publish(client, spec, archive, &health, hooks.progress)?;
    let request = request(spec, &plan, &source_id)?;
    preparations::prepare(client, &spec.project, &request, &state, &health)?;
    let id = prepared(client, spec, &plan, &health, &mut hooks)?;
    remember(state_directory, &plan.latest(spec, &id));
    let current = match dispatch(client, &id, &health) {
        Ok(current) => current,
        Err(Error::SubmissionOutcomeUnknown(_)) => {
            let current = show(client, &id, &health)?;
            observe(client, &id, current, &health)?
        }
        Err(error) => return Err(error),
    };
    let current = watch(
        client,
        &spec.project,
        &plan.run_id,
        &id,
        current,
        &health,
        &mut hooks,
    );
    Ok(outcome(spec, &plan, &state, &current, false))
}

/// Stop the latest job, after saying which one.
pub fn cancel(client: &Client, directory: &Path, hooks: Hooks<'_>) -> Result<Value, Error> {
    let health = client.health()?;
    let latest = latest(directory)?;
    let id = &latest.submission_id;
    let current = show(client, id, &health)?;
    let question = format!(
        "Cancel {} ({}), now {}",
        latest.run_id,
        current["attempt_id"].as_str().unwrap_or("unknown attempt"),
        describe(&current).unwrap_or_else(|| current["status"].to_string())
    );
    if !(hooks.confirm)(&question)? {
        return Err(Error::Observation("not cancelled; nothing was sent"));
    }
    submissions::execute(
        client,
        &submissions::plan(submissions::Command::Cancel {
            submission: id,
            confirmation: &format!("CANCEL {id}"),
        })?,
        &health,
    )
}

/// Run the latest job's Run again as a new Attempt, from a checkpoint or the start.
/// Repeating this after a lost reply finds the same Attempt. Once its job may exist,
/// only observing is left, as in `submit`.
pub fn retry(
    client: &Client,
    directory: &Path,
    checkpoint: Option<String>,
    mut hooks: Hooks<'_>,
) -> Result<Value, Error> {
    use ml_exp_core::{
        checkpoints::Reference,
        first_submission::native::retry::{Request, StartFrom},
    };
    let health = client.health()?;
    let latest = latest(directory)?;
    let parent = &latest.submission_id;
    let current = show(client, parent, &health)?;
    let attempt = current["attempt_id"]
        .as_str()
        .ok_or(Error::Response)?
        .to_owned();
    let summary = &current["preflight_summary"];
    let spend = summary["max_gpu_hours"].as_f64().unwrap_or(0.0);
    let question = format!(
        "Run {} again after {attempt}, from {}\n  resources {}\n  spend     up to {spend} more GPU hours",
        latest.run_id,
        checkpoint.as_deref().unwrap_or("the start"),
        summary["resources"],
    );
    if !(hooks.confirm)(&question)? {
        return Err(Error::Observation("not submitted; nothing was sent"));
    }
    let start = StartFrom {
        run_id: latest.run_id.clone(),
        attempt_id: attempt.clone(),
    };
    let request = Request {
        confirmation: String::new(),
        restart_from: checkpoint.is_none().then_some(start),
        resume_from: checkpoint.map(|checkpoint_id| Reference {
            run_id: latest.run_id.clone(),
            attempt_id: attempt.clone(),
            checkpoint_id,
        }),
        max_gpu_hours: spend,
        reason: REASON.into(),
    };
    submissions::execute(
        client,
        &submissions::plan(submissions::Command::Retry {
            submission: parent,
            request: &request,
        })?,
        &health,
    )?;
    let prepared = submissions::execute(
        client,
        &submissions::plan(submissions::Command::Prepare {
            project: &latest.project,
            run: &latest.run_id,
            max_gpu_hours: spend,
            reason: REASON,
        })?,
        &health,
    )?;
    let id = prepared["submission_id"]
        .as_str()
        .ok_or(Error::Response)?
        .to_owned();
    let next = Latest {
        submission_id: id.clone(),
        ..latest
    };
    remember(directory, &next);
    let sent = match dispatch(client, &id, &health) {
        Ok(sent) => sent,
        Err(Error::SubmissionOutcomeUnknown(_)) => {
            let current = show(client, &id, &health)?;
            observe(client, &id, current, &health)?
        }
        Err(error) => return Err(error),
    };
    let sent = watch(
        client,
        &next.project,
        &next.run_id,
        &id,
        sent,
        &health,
        &mut hooks,
    );
    Ok(json!({
        "project": next.project,
        "run_id": next.run_id,
        "attempt_id": sent["attempt_id"],
        "submission_id": id,
        "status": sent["status"],
        "job_state": job_state(&sent),
        "already_submitted": false,
        "next": ["ml-exp status", "ml-exp logs", "ml-exp results"],
    }))
}

fn read_dockerfile(directory: &Path, name: &str) -> Result<String, Error> {
    let path = directory.join(name);
    let missing = || {
        Error::Submit(format!(
            "{} needs a Dockerfile starting with FROM IMAGE@sha256:DIGEST; \
             `ml-exp get /api/environments` lists approved bases",
            directory.display()
        ))
    };
    if std::fs::metadata(&path).map_err(|_| missing())?.len() > DOCKERFILE_LIMIT {
        return Err(Error::Submit("the Dockerfile exceeds 256 KiB".into()));
    }
    std::fs::read_to_string(&path).map_err(|_| missing())
}

/// Identical bytes have one receipt, so a previous upload is simply reused.
fn publish(
    client: &Client,
    spec: &Spec,
    archive: sources::PreparedArchive,
    health: &Value,
    progress: &mut dyn FnMut(&str),
) -> Result<String, Error> {
    let receipt = match sources::lookup_archive(client, &spec.project, &archive.sha256, health) {
        Ok(receipt) => receipt,
        Err(Error::Http(failure)) if failure.status == 404 => {
            progress("uploading source");
            let plan = sources::plan_archive(&spec.project, &archive.sha256, archive.bytes)?;
            sources::upload(client, &plan, archive.file, health)?
        }
        Err(error) => return Err(error),
    };
    receipt["source_id"]
        .as_str()
        .map(str::to_owned)
        .ok_or(Error::Response)
}

/// Repeat the same safe continue until a Submission exists. A failure stops here
/// rather than being retried: continue would start a failed build again.
fn prepared(
    client: &Client,
    spec: &Spec,
    plan: &Plan,
    health: &Value,
    hooks: &mut Hooks<'_>,
) -> Result<String, Error> {
    let id = &plan.preparation_id;
    let confirmation = format!("PREPARE {id}");
    let started = Instant::now();
    let mut lost_replies = 0;
    let mut last = "";
    let mut runtime: Option<String> = None;
    let mut build = BuildView::default();
    let mut tick = 0;
    loop {
        if tick % ADVANCE_EVERY == 0 {
            match preparations::advance(client, &spec.project, id, &confirmation) {
                Ok(view) => {
                    lost_replies = 0;
                    if let Some(submission) = view["submission"]["submission_id"].as_str() {
                        return Ok(submission.to_owned());
                    }
                    if let Some(reason) = stopped(&view) {
                        return Err(Error::Submit(format!(
                            "{reason}. No training job was submitted"
                        )));
                    }
                    let now = stage(&view);
                    if now != last {
                        (hooks.progress)(now);
                        last = now;
                    }
                    runtime = building(&view).map(str::to_owned);
                }
                Err(Error::Transport) if lost_replies < 3 => lost_replies += 1,
                Err(error) => return Err(error),
            }
        }
        tick += 1;
        if let Some(runtime) = &runtime {
            show_build(client, spec, runtime, health, &mut build, hooks.progress);
        }
        if started.elapsed() >= hooks.deadline {
            return Err(Error::Submit(
                "still preparing, and no training job was submitted. \
                 Run this command again to keep waiting"
                    .into(),
            ));
        }
        std::thread::sleep(hooks.poll);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const PINNED: &str = "docker.io/library/python@sha256:0000000000000000000000000000000000000000000000000000000000000000";
    fn spec() -> Spec {
        Spec {
            project: "demo".into(),
            executor: "pai-h200".into(),
            directory: "source".into(),
            dockerfile: "Dockerfile".into(),
            command: vec!["python3".into(), "train.py".into()],
            gpus: 1,
            cpus: None,
            memory_gb: None,
            walltime_seconds: 600,
            seed: 0,
            ..Spec::default()
        }
    }
    fn planned(spec: &Spec) -> Plan {
        plan(
            spec,
            "https://api.example",
            &"a".repeat(64),
            &format!("FROM {PINNED}\n"),
            &json!({"executor_id":spec.executor,"limits":{}}),
            "source",
        )
        .unwrap()
    }
    #[test]
    fn times_read_as_a_person_writes_them() {
        for (text, seconds) in [
            ("90", 90),
            ("90s", 90),
            ("10m", 600),
            ("2h", 7200),
            ("1d", 86400),
        ] {
            assert_eq!(parse_time(text), Ok(seconds), "{text}");
        }
        for text in ["", "0", "m", "10x", "-5m", "1.5h", "99999999999999999999d"] {
            assert!(parse_time(text).is_err(), "{text}");
        }
    }
    #[test]
    fn the_command_line_wins_over_the_file_and_the_file_over_the_saved_settings() {
        let file = br#"{"request_id":"11111111111111111111111111111111",
            "runtime":{"dockerfile":"docker/Dockerfile"},
            "run":{"executor":"from-file","command":["from-file"],
                "resources":{"gpus":2,"cpus":3,"memory_gb":4},"walltime_seconds":60,"seed":7,
                "run_id":"file-run","env":{"A":"file","B":"file"},
                "data":[{"name":"x","asset_id":"asset.x"}],
                "metrics_schema":{"definitions":{}}},
            "approval":{"max_gpu_hours":9.0,"reason":"because"}}"#;
        let merged = spec_of(
            Options {
                project: Some("p".into()),
                gpus: Some(1),
                env: vec![("A".into(), "line".into())],
                ..Options::default()
            },
            Some(file),
        )
        .unwrap();
        assert_eq!(merged.gpus, 1);
        assert_eq!(merged.cpus, Some(3));
        assert_eq!(merged.executor, "from-file");
        assert_eq!(merged.command, ["from-file"]);
        assert_eq!(merged.dockerfile, "docker/Dockerfile");
        assert_eq!(merged.walltime_seconds, 60);
        assert_eq!(merged.seed, 7);
        assert_eq!(merged.run_id.as_deref(), Some("file-run"));
        assert_eq!(merged.env["A"], "line");
        assert_eq!(merged.env["B"], "file");
        assert_eq!(merged.data, [("x".to_owned(), "asset.x".to_owned())]);
        assert!(merged.extra.contains_key("metrics_schema"));
        assert_eq!(
            merged.request_id.as_deref(),
            Some("11111111111111111111111111111111")
        );
        assert_eq!(merged.max_gpu_hours, Some(9.0));
        assert_eq!(merged.reason.as_deref(), Some("because"));
    }
    fn spec_of(options: Options, file: Option<&[u8]>) -> Result<Spec, Error> {
        super::spec(options, file, Some("saved-project"), Some("saved-executor"))
    }
    fn complete() -> Options {
        Options {
            gpus: Some(1),
            walltime_seconds: Some(60),
            command: vec!["run".into()],
            ..Options::default()
        }
    }
    #[test]
    fn saved_settings_fill_what_is_left_and_defaults_are_stated() {
        let merged = spec_of(complete(), None).unwrap();
        assert_eq!(merged.project, "saved-project");
        assert_eq!(merged.executor, "saved-executor");
        assert_eq!(merged.dockerfile, "Dockerfile");
        assert_eq!(merged.seed, 0);
    }
    #[test]
    fn whatever_is_missing_is_named() {
        let cases: [(Options, &str); 3] = [
            (
                Options {
                    gpus: None,
                    ..complete()
                },
                "--gpus",
            ),
            (
                Options {
                    walltime_seconds: None,
                    ..complete()
                },
                "--time",
            ),
            (
                Options {
                    command: vec![],
                    ..complete()
                },
                "--",
            ),
        ];
        for (options, word) in cases {
            let text = spec_of(options, None).err().unwrap().to_string();
            assert!(text.contains(word), "{text}");
        }
        let text = super::spec(complete(), None, None, Some("e"))
            .err()
            .unwrap()
            .to_string();
        assert!(text.contains("--project"), "{text}");
        let text = super::spec(complete(), None, Some("p"), None)
            .err()
            .unwrap()
            .to_string();
        assert!(text.contains("--executor"), "{text}");
    }
    #[test]
    fn a_file_cannot_fix_what_submit_derives() {
        for file in [
            &br#"{"runtime":{"source_id":"source.x"}}"#[..],
            br#"{"runtime":{"base_image":"x"}}"#,
            br#"{"run":{"runtime_id":"runtime.x"}}"#,
            br#"{"run":{"resources":{"nodes":2}}}"#,
            br#"{"unknown":true}"#,
            b"[]",
        ] {
            assert!(spec_of(complete(), Some(file)).is_err());
        }
    }
    #[test]
    fn extra_fields_and_environment_change_the_job_and_reach_the_request() {
        let base = planned(&spec());
        let mut with_env = spec();
        with_env.env.insert("KEY".into(), "value".into());
        assert_ne!(planned(&with_env).request_id, base.request_id);
        let mut with_extra = spec();
        with_extra
            .extra
            .insert("metrics_schema".into(), json!({"definitions":{}}));
        assert_ne!(planned(&with_extra).request_id, base.request_id);
        let request = request(
            &with_env,
            &planned(&with_env),
            &format!("source.{}", "2".repeat(64)),
        )
        .unwrap();
        assert_eq!(request.run["env"]["KEY"], "value");
        let mut reserved = spec();
        reserved.env.insert("OUTPUT_DIR".into(), "x".into());
        assert!(
            plan(
                &reserved,
                "https://api.example",
                &"a".repeat(64),
                &format!("FROM {PINNED}\n"),
                &json!({"executor_id":"pai-h200","limits":{}}),
                "source"
            )
            .is_err()
        );
    }
    #[test]
    fn environment_reads_key_equals_value() {
        assert_eq!(parse_env("A=b=c"), Ok(("A".to_owned(), "b=c".to_owned())));
        assert!(parse_env("A").is_err());
        assert!(parse_env("=b").is_err());
    }
    #[test]
    fn data_names_an_uploaded_asset() {
        assert_eq!(
            parse_data("train=asset.abc"),
            Ok(("train".to_owned(), "asset.abc".to_owned()))
        );
        for text in ["train", "=asset.abc", "train=./data"] {
            assert!(parse_data(text).is_err(), "{text}");
        }
    }
    #[test]
    fn the_base_image_is_the_last_external_from() {
        let other = PINNED.replace("python", "ubuntu");
        for (text, expected) in [
            (format!("FROM {PINNED}\nRUN true\n"), PINNED),
            (format!("from {PINNED} as build\nFROM build\n"), PINNED),
            (format!("FROM --platform=linux/amd64 {PINNED}\n"), PINNED),
            (
                format!("FROM {PINNED} AS a\nFROM {other}\n"),
                other.as_str(),
            ),
            (
                format!("FROM {other} AS a\nFROM {PINNED}\nFROM a\n"),
                PINNED,
            ),
            (format!("FROM \\\n  {PINNED}\n"), PINNED),
            (format!("# FROM {other}\nFROM {PINNED}\n"), PINNED),
            (format!("FROM {PINNED} AS a\nFROM scratch\n"), PINNED),
        ] {
            assert_eq!(base_image(&text).unwrap(), expected, "{text}");
        }
        for text in [
            "",
            "FROM\n",
            "FROM python:3.12\n",
            "FROM scratch\n",
            "RUN true\n",
        ] {
            assert!(base_image(text).is_err(), "{text}");
        }
    }
    #[test]
    fn the_same_inputs_are_the_same_job_and_anything_that_changes_the_job_is_not() {
        let base = planned(&spec());
        assert_eq!(base, planned(&spec()));
        assert!(base.run_id.starts_with("source-"));
        let changes: [fn(&mut Spec); 8] = [
            |s| s.seed = 1,
            |s| s.gpus = 2,
            |s| s.walltime_seconds = 601,
            |s| s.command.push("--flag".into()),
            |s| s.executor = "other".into(),
            |s| s.project = "other".into(),
            |s| s.run_id = Some("named".into()),
            |s| s.data.push(("train".into(), "asset.abc".into())),
        ];
        for change in changes {
            let mut changed = spec();
            change(&mut changed);
            assert_ne!(planned(&changed).request_id, base.request_id);
        }
        let other_source = plan(
            &spec(),
            "https://api.example",
            &"b".repeat(64),
            &format!("FROM {PINNED}\n"),
            &json!({"executor_id":"pai-h200","limits":{}}),
            "source",
        )
        .unwrap();
        assert_ne!(other_source.request_id, base.request_id);
        assert_ne!(other_source.run_id, base.run_id);
    }
    #[test]
    fn resources_follow_the_gpus_and_stay_within_the_executor() {
        let mut one = spec();
        assert_eq!(resources(&one, "pai", &json!({})).unwrap(), (1, 8, 32));
        assert_eq!(
            resources(&one, "pai", &json!({"cpus":4,"memory_gb":8})).unwrap(),
            (1, 4, 8)
        );
        one.gpus = 0;
        assert_eq!(resources(&one, "pai", &json!({})).unwrap(), (0, 2, 4));
        one.gpus = 2;
        assert!(resources(&one, "pai", &json!({"gpus":1})).is_err());
        one.gpus = 1;
        one.cpus = Some(9);
        assert!(resources(&one, "pai", &json!({"cpus":8})).is_err());
        one.cpus = None;
        one.memory_gb = Some(9);
        assert!(resources(&one, "pai", &json!({"memory_gb":8})).is_err());
        one.memory_gb = None;
        assert!(resources(&one, "pai", &json!({"walltime_seconds":599})).is_err());
    }
    fn shape(id: &str, gpus: u32, cpus: u32, memory_gb: u32) -> Value {
        json!({"executor_id":id,"enabled":true,"run_creation":true,
            "limits":{"gpus":gpus,"cpus":cpus,"memory_gb":memory_gb,"fixed_resources":true}})
    }
    #[test]
    fn a_fixed_shape_runs_whole_and_the_request_only_has_to_fit() {
        let limits = shape("x", 2, 28, 396)["limits"].clone();
        let mut one = spec();
        assert_eq!(resources(&one, "x", &limits).unwrap(), (2, 28, 396));
        one.cpus = Some(20);
        one.memory_gb = Some(300);
        assert_eq!(resources(&one, "x", &limits).unwrap(), (2, 28, 396));
        one.cpus = Some(29);
        assert!(resources(&one, "x", &limits).is_err());
        one.cpus = None;
        one.memory_gb = Some(397);
        assert!(resources(&one, "x", &limits).is_err());
        one.memory_gb = None;
        one.gpus = 3;
        assert!(resources(&one, "x", &limits).is_err());
        one.gpus = 0;
        assert!(resources(&one, "x", &limits).is_err());
        let mut planned_shape = spec();
        planned_shape.executor = "x".into();
        let plan = plan(
            &planned_shape,
            "https://api.example",
            &"a".repeat(64),
            &format!("FROM {PINNED}\n"),
            &shape("x", 2, 28, 396),
            "source",
        )
        .unwrap();
        assert_eq!((plan.gpus, plan.cpus, plan.memory_gb), (2, 28, 396));
        assert_eq!(plan.max_gpu_hours, 0.34, "the spend covers the whole shape");
    }
    #[test]
    fn a_family_takes_the_smallest_shape_that_holds_the_request() {
        let rows = [
            shape("sc-8gpu-a", 8, 64, 1024),
            shape("sc-2gpu-a", 2, 44, 460),
            shape("sc-2gpu-b", 2, 28, 396),
            shape("sc-1gpu-a", 1, 22, 230),
            shape("sc-1gpu-b", 1, 14, 198),
        ];
        let chosen = |gpus, cpus, memory_gb| {
            let mut request = spec();
            request.executor = "sc".into();
            request.gpus = gpus;
            request.cpus = cpus;
            request.memory_gb = memory_gb;
            choose(&request, &rows)
                .ok()
                .map(|row| row["executor_id"].as_str().unwrap().to_owned())
        };
        assert_eq!(chosen(1, None, None).as_deref(), Some("sc-1gpu-b"));
        assert_eq!(chosen(1, Some(20), None).as_deref(), Some("sc-1gpu-a"));
        assert_eq!(chosen(1, None, Some(300)).as_deref(), Some("sc-2gpu-b"));
        assert_eq!(chosen(3, None, None).as_deref(), Some("sc-8gpu-a"));
        assert_eq!(chosen(9, None, None), None);
        assert_eq!(chosen(1, Some(65), None), None);
    }
    #[test]
    fn a_family_is_only_fixed_shapes_that_accept_runs_and_an_exact_name_wins() {
        let mut closed = shape("sc-a", 1, 8, 32);
        closed["run_creation"] = json!(false);
        let plain =
            json!({"executor_id":"sc-b","enabled":true,"run_creation":true,"limits":{"gpus":1}});
        let mut request = spec();
        request.executor = "sc".into();
        assert!(choose(&request, &[closed, plain]).is_err());
        let rows = [shape("sc-a", 1, 8, 32), shape("sc", 8, 8, 8)];
        assert_eq!(choose(&request, &rows).unwrap()["executor_id"], "sc");
    }
    #[test]
    fn spend_is_rounded_up_so_the_budget_covers_the_time() {
        assert_eq!(planned(&spec()).max_gpu_hours, 0.17);
        let mut zero = spec();
        zero.gpus = 0;
        assert_eq!(planned(&zero).max_gpu_hours, 0.0);
        let mut slug_only = spec();
        slug_only.run_id = Some("bad name".into());
        assert!(
            plan(
                &slug_only,
                "https://api.example",
                &"a".repeat(64),
                &format!("FROM {PINNED}\n"),
                &json!({"executor_id":"pai-h200","limits":{}}),
                "source"
            )
            .is_err()
        );
        assert_eq!(slug("--My project!"), "Myproject");
        assert_eq!(slug("..."), "run");
    }
    #[test]
    fn only_failures_that_waiting_cannot_fix_stop_the_wait() {
        assert!(stopped(&json!({"runtime":{"status":"EXECUTING"}})).is_none());
        assert!(stopped(&json!({"runtime":{"status":"PREPARED"}})).is_none());
        assert!(
            stopped(
                &json!({"runtime":{"status":"EXECUTING","build_error":{"code":"BUILD_EXECUTING"}}})
            )
            .is_none()
        );
        for view in [
            json!({"runtime":{"status":"PREPARED","build_error":{"code":"BAD"}}}),
            json!({"error_code":"SCRIPT_FAILED"}),
            json!({"data":{"new_request_allowed":true}}),
        ] {
            assert!(stopped(&view).is_some(), "{view}");
        }
    }
    #[test]
    fn only_new_log_lines_are_shown() {
        let lines = |text: &[&str]| -> Vec<String> { text.iter().map(|s| s.to_string()).collect() };
        let shown = lines(&["a", "b", "c"]);
        assert!(new_lines(&shown, &lines(&["a", "b", "c"])).is_empty());
        assert_eq!(new_lines(&shown, &lines(&["b", "c", "d"])), lines(&["d"]));
        assert_eq!(new_lines(&shown, &lines(&["x", "y"])), lines(&["x", "y"]));
        assert_eq!(new_lines(&[], &lines(&["a"])), lines(&["a"]));
        assert!(new_lines(&shown, &[]).is_empty());
        let repeated = lines(&["a", "a"]);
        assert_eq!(
            new_lines(&repeated, &lines(&["a", "a", "a"])),
            lines(&["a"])
        );
    }
    #[test]
    fn the_job_line_uses_the_backends_own_words() {
        let job = |state: &str, own: &str, queue: Value| {
            json!({"cloud_observation":{"observation":
                {"state":state,"provider_state":own,"queue":queue}}})
        };
        assert_eq!(
            describe(&job("QUEUED", "EnvPreparing", Value::Null)).unwrap(),
            "[job] QUEUED (EnvPreparing)"
        );
        assert_eq!(
            describe(&job("RUNNING", "Running", Value::Null)).unwrap(),
            "[job] RUNNING"
        );
        assert_eq!(
            describe(&job("QUEUED", "Queuing", json!({"reason":"no quota"}))).unwrap(),
            "[job] QUEUED (Queuing): no quota"
        );
        let slurm = json!({"execution":{"result":{"observation":{"status":{"state":"RUNNING"}}}}});
        assert_eq!(job_state(&slurm), Some("RUNNING"));
        assert_eq!(describe(&slurm).unwrap(), "[job] RUNNING");
        assert!(describe(&json!({})).is_none());
    }
    #[test]
    fn a_job_that_may_exist_is_never_treated_as_unsent() {
        for status in ["PREPARED", "AUTHORIZED"] {
            assert!(!dispatched(&json!(status)));
        }
        for status in ["EXECUTING", "RECONCILE_REQUIRED", "VERIFIED", "BLOCKED"] {
            assert!(dispatched(&json!(status)));
        }
        assert!(dispatched(&Value::Null));
    }
}
