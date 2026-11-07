//! The common case in one command: pack the source, confirm once, prepare and submit.
//! Every step is an existing operation. The identity comes from the inputs, so the
//! same command finds the same Submission again and never sends a second job.
use crate::{
    Client, Error, dockerfile, observation, preparations, presentation, read, runtime, sources,
    submissions,
};
use ml_exp_core::{canonical, cloud::build, identity, preparation};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
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
    /// `--allocation`, as a name clap has already checked.
    pub allocation: Option<String>,
}

/// The experiment file `run start` reads, with everything `submit` derives optional.
/// A fixed source or Runtime is refused: `submit` always packs the directory.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
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
                     `submit` derives the request ID, source and base image itself; \
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
    if file.run.extra.contains_key("code_source_id") {
        return Err(Error::Submit(
            "run.code_source_id is not used by `submit`, which packs the directory; \
             use `ml-exp run start` to fix one"
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
    data.retain(|(name, _)| !options.data.iter().any(|(given, _)| given == name));
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
    // Into the Run fields the identity already covers, the same as the file's spelling.
    // Left out, the identity is what it always was, so a repeat still finds its old job.
    let mut extra = file.run.extra;
    if let Some(allocation) = options.allocation {
        extra
            .entry("scheduling")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| Error::Submit("run.scheduling in the file is not an object".into()))?
            .insert("allocation".into(), json!(allocation));
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
            .map(|path| path.trim_start_matches("./").to_owned())
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
        extra,
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
    /// `None` where the executor has no such choice.
    pub allocation: Option<String>,
}

pub struct Hooks<'a> {
    /// Shown what will be spent. Returning false stops before anything is sent.
    pub confirm: &'a mut dyn FnMut(&str) -> Result<bool, Error>,
    pub progress: &'a mut dyn FnMut(&str),
    pub poll: Duration,
    pub deadline: Duration,
    /// How long to wait for a server that reports nothing about a job before saying so.
    pub silence: Duration,
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
    /// None while it is still being prepared: nothing can be running yet.
    pub submission_id: Option<String>,
    /// A send began and its outcome was never learned. Repeating `retry` must find
    /// out what happened to this one rather than start another.
    #[serde(default)]
    pub unsettled: bool,
}
const LATEST: &str = "latest.json";
const RUNS: &str = "runs";
/// Only a convenience: failing to save this must not disturb a submission.
fn remember(directory: &Path, latest: &Latest) {
    let Ok(bytes) = serde_json::to_vec(latest) else {
        return;
    };
    // Whatever this acted on is now the latest, even if it was named. Each Run also keeps
    // its own record, so it can still be named after a newer one.
    let root = records_root(directory);
    let _ = ml_exp_io::write_atomic(&root.join(LATEST), &bytes, false);
    let own = run_records(root, &latest.project, &latest.run_id);
    if std::fs::create_dir_all(&own).is_ok() {
        let _ = ml_exp_io::write_atomic(&own.join(LATEST), &bytes, false);
    }
}
fn run_records(root: &Path, project: &str, run: &str) -> PathBuf {
    root.join(RUNS).join(project).join(run)
}
/// The directory holding the latest record, given it or a named Run's records beneath it.
fn records_root(directory: &Path) -> &Path {
    let mut above = directory.ancestors();
    match (above.nth(2), above.next()) {
        (Some(runs), Some(root)) if runs.file_name() == Some(OsStr::new(RUNS)) => root,
        _ => directory,
    }
}
/// Where the record of a named Run is kept, for the commands that act on the latest
/// one by default. `--project` is only needed when two Projects used the same Run name.
pub fn run_directory(directory: &Path, project: Option<&str>, run: &str) -> Result<PathBuf, Error> {
    let runs = directory.join(RUNS);
    let valid = |id: &str| {
        ml_exp_core::identity::validate_id(id)
            .map(|_| ())
            .map_err(|_| Error::Submit(format!("{id} is not a valid name")))
    };
    valid(run)?;
    let unknown = || {
        Error::Submit(format!(
            "this machine has no record of submitting {run}; `ml-exp run --help` acts on a \
             Submission by its ID"
        ))
    };
    if let Some(project) = project {
        valid(project)?;
        let found = runs.join(project).join(run);
        return if found.join(LATEST).is_file() {
            Ok(found)
        } else {
            Err(unknown())
        };
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(&runs)
        .map_err(|_| unknown())?
        .flatten()
        .map(|entry| entry.path().join(run))
        .filter(|path| path.join(LATEST).is_file())
        .collect();
    match found.len() {
        0 => Err(unknown()),
        1 => Ok(found.remove(0)),
        _ => Err(Error::Submit(format!(
            "more than one Project has a Run named {run}; say which with --project"
        ))),
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
/// The latest Submission, for commands that need one to exist.
fn sent(latest: &Latest) -> Result<&str, Error> {
    latest.submission_id.as_deref().ok_or_else(|| {
        Error::Submit(format!(
            "the latest `ml-exp submit`, {}, has not sent a job. Any earlier job is unaffected; \
             `ml-exp run --help` acts on one by its ID",
            latest.run_id
        ))
    })
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

/// One instruction per entry: continuations joined, comments gone, as the builder reads them.
pub(crate) fn instructions(dockerfile: &str) -> Vec<String> {
    let mut all = Vec::new();
    let mut current = String::new();
    for line in dockerfile.lines().map(str::trim) {
        if line.starts_with('#') {
            continue;
        }
        if let Some(head) = line.strip_suffix('\\') {
            current.push_str(head);
            current.push(' ');
        } else {
            current.push_str(line);
            all.push(std::mem::take(&mut current));
        }
    }
    all.push(current);
    all
}
/// The last external image, which the builder requires `base_image` to equal. The
/// builder's parser stays the authority: this saves writing the digest twice, and
/// refuses what the builder is known to refuse before anything is confirmed or sent.
pub fn base_image(dockerfile: &str) -> Result<String, Error> {
    let refused = Error::Submit;
    for line in dockerfile.lines() {
        if let Some(comment) = line.trim().strip_prefix('#') {
            let word = comment.trim_start().to_ascii_lowercase();
            if ["syntax", "escape"].iter().any(|directive| {
                word.strip_prefix(directive)
                    .is_some_and(|rest| rest.trim_start().starts_with('='))
            }) {
                return Err(refused(
                    "the builder does not support `# syntax=` or `# escape=`; remove that line"
                        .into(),
                ));
            }
        }
    }
    let mut stages = Vec::new();
    let mut last = None;
    for line in instructions(dockerfile) {
        let mut words = line.split_whitespace();
        if !words
            .next()
            .is_some_and(|word| word.eq_ignore_ascii_case("FROM"))
        {
            continue;
        }
        let mut parts: Vec<&str> = words.map(|word| word.trim_matches(['"', '\''])).collect();
        if let Some(flag) = parts.first().filter(|word| word.starts_with("--")) {
            if *flag != "--platform=linux/amd64" {
                return Err(refused(format!(
                    "the builder supports only linux/amd64 and no other FROM option, not {flag}"
                )));
            }
            parts.remove(0);
        }
        let named = parts.len() == 3 && parts[1].eq_ignore_ascii_case("AS");
        if parts.len() != 1 && !named {
            return Err(refused(format!(
                "this FROM is not one the builder accepts: {line}"
            )));
        }
        let image = parts[0];
        if image != "scratch" && !stages.contains(&image.to_ascii_lowercase()) {
            if !pinned(image) {
                return Err(refused(format!(
                    "FROM {image} must be pinned as IMAGE@sha256:DIGEST; \
                     `docker buildx imagetools inspect IMAGE` prints the digest"
                )));
            }
            last = Some(image.to_owned());
        }
        if named {
            stages.push(parts[2].to_ascii_lowercase());
        }
    }
    last.ok_or_else(|| {
        refused(
            "the Dockerfile needs a FROM pinned as IMAGE@sha256:DIGEST; \
             `docker buildx imagetools inspect IMAGE` prints the digest"
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

/// Who this job is, known from the inputs alone and before asking the server anything.
pub struct Identity {
    pub request_id: String,
    pub preparation_id: String,
    pub run_id: String,
    pub base_image: String,
}
pub fn identify(
    spec: &Spec,
    content_sha256: &str,
    dockerfile: &str,
    directory_name: &str,
) -> Result<Identity, Error> {
    let base_image = base_image(dockerfile)?;
    // What was asked for, not what today's limits and shapes resolve it to, so that a
    // repeat finds the same job even if the operator has changed them since. The server
    // is left out: preparations are already per server, and it can have many spellings.
    // The content, not the archive, so that a change of compression changes nothing.
    let identity = canonical::digest(&json!([
        "ml-exp-submit.v2",
        spec.project,
        spec.executor,
        directory_name,
        content_sha256,
        spec.dockerfile,
        base_image,
        spec.command,
        spec.gpus,
        spec.cpus,
        spec.memory_gb,
        spec.walltime_seconds,
        spec.seed,
        spec.run_id,
        spec.data,
        spec.env,
        spec.extra,
        spec.max_gpu_hours.map(|hours| hours.to_string()),
        spec.reason,
    ]))
    .map_err(|_| Error::Response)?;
    let request_id = identity[..32].to_owned();
    let run_id = match &spec.run_id {
        Some(run) => run.clone(),
        None => format!("{}-{}", slug(directory_name), &identity[..8]),
    };
    identity::validate_id(&run_id)
        .map_err(|_| Error::Submit(format!("{run_id} is not a valid Run name")))?;
    let preparation_id = preparation::id(&spec.project, &request_id).map_err(Error::Observation)?;
    Ok(Identity {
        request_id,
        preparation_id,
        run_id,
        base_image,
    })
}
/// What it will run on and spend, which depends on the executor as it is today.
pub fn resolve(spec: &Spec, identity: Identity, executor: &Value) -> Result<Plan, Error> {
    let chosen = executor["executor_id"]
        .as_str()
        .ok_or(Error::Response)?
        .to_owned();
    let (gpus, cpus, memory_gb) = resources(spec, &chosen, &executor["limits"])?;
    if gpus == 0 && executor["kind"] == "slurm" {
        return Err(Error::Submit(format!(
            "{chosen} is a Slurm executor, which needs at least one GPU; use --gpus 1 or a \
             CPU executor from `ml-exp backend`"
        )));
    }
    let allocation = allocation(spec, &chosen, executor, gpus)?;
    ml_exp_core::environment::validate(&spec.env).map_err(|_| {
        Error::Submit(
            "an environment variable has an invalid or reserved name; \
             the platform sets directories, Run identity and ML_EXPD_ names itself"
                .into(),
        )
    })?;
    Ok(Plan {
        request_id: identity.request_id,
        preparation_id: identity.preparation_id,
        run_id: identity.run_id,
        base_image: identity.base_image,
        executor: chosen,
        gpus,
        cpus,
        memory_gb,
        max_gpu_hours: spec
            .max_gpu_hours
            .unwrap_or((f64::from(gpus) * spec.walltime_seconds as f64 / 36.0).ceil() / 100.0),
        allocation,
    })
}
/// What this new job will ask for. The default is decided here, and `request` freezes
/// it in the Run. Slurm and SenseCore have nothing to choose, so both ask what they did.
fn allocation(
    spec: &Spec,
    chosen: &str,
    executor: &Value,
    gpus: u32,
) -> Result<Option<String>, Error> {
    use ml_exp_core::cloud::pai::Allocation;
    let scheduling = spec.extra.get("scheduling");
    if scheduling.is_some_and(|value| !value.is_object()) {
        return Err(Error::Submit(
            "run.scheduling in the file is not an object".into(),
        ));
    }
    let asked = scheduling
        .and_then(|value| value.get("allocation"))
        .and_then(Value::as_str);
    match executor["kind"].as_str() {
        Some("pai-dlc") => Ok(Some(
            asked
                .unwrap_or(Allocation::for_new_run(gpus).name())
                .to_owned(),
        )),
        Some("sensecore") if asked.is_some_and(|name| name != Allocation::Guaranteed.name()) => {
            Err(Error::Submit(format!(
                "{chosen} is a SenseCore executor, which has no idle resources to allow; \
                 remove the allocation choice"
            )))
        }
        Some("slurm") if scheduling.is_some() => Err(Error::Submit(format!(
            "{chosen} is a Slurm executor, which has no allocation or priority choice"
        ))),
        _ => Ok(None),
    }
}
/// What a saved preparation will run, whatever today's executors would resolve to. It
/// is frozen on the server, and it is what the user must be shown.
fn frozen(identity: Identity, view: &Value) -> Result<Plan, Error> {
    let frozen = &view["frozen"];
    let number = |key: &str| {
        frozen[key]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(Error::Response)
    };
    Ok(Plan {
        request_id: identity.request_id,
        preparation_id: identity.preparation_id,
        run_id: identity.run_id,
        base_image: identity.base_image,
        executor: frozen["executor"]
            .as_str()
            .ok_or(Error::Response)?
            .to_owned(),
        gpus: number("gpus")?,
        cpus: number("cpus")?,
        memory_gb: number("memory_gb")?,
        max_gpu_hours: frozen["max_gpu_hours"].as_f64().ok_or(Error::Response)?,
        allocation: frozen["allocation"].as_str().map(str::to_owned),
    })
}
pub fn plan(
    spec: &Spec,
    content_sha256: &str,
    dockerfile: &str,
    executor: &Value,
    directory_name: &str,
) -> Result<Plan, Error> {
    resolve(
        spec,
        identify(spec, content_sha256, dockerfile, directory_name)?,
        executor,
    )
}
fn latest_for(spec: &Spec, run_id: &str, submission: Option<&str>, unsettled: bool) -> Latest {
    Latest {
        project: spec.project.clone(),
        run_id: run_id.to_owned(),
        submission_id: submission.map(str::to_owned),
        unsettled,
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
pub fn summary(spec: &Spec, plan: &Plan, files: usize, environment: &str) -> String {
    let mut lines = vec![
        format!("Run {}", plan.run_id),
        format!("  executor  {}", plan.executor),
        format!("  command   {}", spec.command.join(" ")),
        format!("  image     {}", plan.base_image),
        format!(
            "  source    {files} files from {}",
            spec.directory.display()
        ),
        format!("  environment {environment}"),
        format!(
            "  resources {} GPU, {} CPU, {} GB",
            plan.gpus, plan.cpus, plan.memory_gb
        ),
        format!("  time      up to {} seconds", spec.walltime_seconds),
        format!("  spend     up to {} GPU hours", plan.max_gpu_hours),
    ];
    match plan.allocation.as_deref() {
        Some("guaranteed") => lines.push("  allocation guaranteed: only the quota".into()),
        Some(name) => {
            let may = if name == "idle_only" { "only" } else { "also" };
            lines.push(format!(
                "  allocation {name}: uses {may} idle resources, so THIS JOB MAY BE PREEMPTED and \
                 end early. Write checkpoints; --allocation guaranteed avoids this"
            ));
        }
        None => {}
    }
    if !spec.data.is_empty() {
        let names: Vec<&str> = spec.data.iter().map(|(name, _)| name.as_str()).collect();
        lines.push(format!("  data      {}", names.join(", ")));
    }
    if !spec.env.is_empty() {
        let names: Vec<&str> = spec.env.keys().map(String::as_str).collect();
        lines.push(format!("  env       {}", names.join(", ")));
    }
    if !spec.extra.is_empty() {
        let names: Vec<&str> = spec.extra.keys().map(String::as_str).collect();
        lines.push(format!("  also      {}", names.join(", ")));
    }
    lines.join("\n")
}

fn request(
    spec: &Spec,
    plan: &Plan,
    source_id: &str,
    code_source_id: Option<&str>,
) -> Result<preparation::Request, Error> {
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
    if let Some(code) = code_source_id {
        run["code_source_id"] = json!(code);
    }
    // A chosen allocation is already here. This freezes a defaulted one in the new Run,
    // since a Run with no `scheduling` means guaranteed. A guaranteed job stays absent,
    // and its Run is what it was before there was any choice.
    if let Some(name) = plan
        .allocation
        .as_deref()
        .filter(|name| *name != "guaranteed")
    {
        run["scheduling"]["allocation"] = json!(name);
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

/// Why waiting longer cannot help. Stopping the wait never stops anything on the server.
///
/// A missing receipt alone proves nothing: after an API restart the build may still be
/// running, and it reports just that. Only the builder saying it is idle ends that wait.
fn stopped(view: &Value, builder_idle: bool) -> Option<String> {
    if let Some(code) = view["error_code"].as_str() {
        return Some(format!("data preparation failed with {code}"));
    }
    let runtime = &view["runtime"];
    let error = &runtime["build_error"];
    let idle = matches!(
        runtime["status"].as_str(),
        Some("PREPARED" | "RECONCILE_REQUIRED")
    );
    if idle && !error.is_null() && error["code"] != "BUILD_EXECUTING" {
        let id = runtime["runtime_id"].as_str().unwrap_or("RUNTIME");
        let project = view["project"].as_str().unwrap_or("PROJECT");
        let logs = format!("`ml-exp env logs {id} --project {project}`");
        // The original cause is kept beside a later "receipt missing".
        let cause = if error["build_failure"].is_null() {
            error
        } else {
            &error["build_failure"]
        };
        let code = cause["code"].as_str().unwrap_or("UNKNOWN");
        let step = runtime["operator_step"]
            .as_str()
            .map(|step| format!(". {step}"))
            .unwrap_or_default();
        if matches!(code, "BUILD_RECEIPT_MISSING" | "NATIVE_BUILD_RPC_UNCERTAIN") {
            // Proof that nothing was left means it is not running, whatever was last seen.
            return (builder_idle || runtime["retry_safe"] == true).then(|| {
                format!(
                    "the image build is not running and left no result ({code}), so its outcome \
                     is unresolved. See {logs}. Running this command again checks once more \
                     and builds again if nothing was left behind; changing the code or the \
                     Dockerfile starts a new build{step}"
                )
            });
        }
        let reason = cause["details"]["reason"]
            .as_str()
            .or(cause["details"]["error_class"].as_str())
            .map(|reason| format!(" ({reason})"))
            .unwrap_or_default();
        return Some(format!(
            "the image build failed with {code}{reason}. See {logs}. Running this command \
             again builds again if nothing was left behind; changing the code or the \
             Dockerfile starts a new build{step}"
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
/// What an earlier command's build left behind is proven safe to start over. One
/// command does this once: a build that fails while it watches is reported, not repeated.
fn may_build_again(view: &Value) -> bool {
    let runtime = &view["runtime"];
    runtime["retry_safe"] == true && !runtime["build_error"].is_null()
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
) -> Option<bool> {
    let command = runtime::Command::Progress {
        project: spec.project.clone(),
        runtime: runtime.to_owned(),
    };
    let value = command
        .plan()
        .and_then(|plan| runtime::execute(client, plan, health))
        .ok()?;
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
    state["status"].as_str().map(|status| status != "EXECUTING")
}

fn show(client: &Client, id: &str, health: &Value) -> Result<Value, Error> {
    submissions::execute(
        client,
        &submissions::plan(submissions::Command::Show { submission: id })?,
        health,
    )
}

/// Only this can be sent. Anything else may already have a job, or can never have
/// one until it is prepared again, so sending is out of the question.
fn unsent(status: &Value) -> bool {
    status.as_str() == Some("PREPARED")
}
/// Prepared again or never: BLOCKED and an expired gate have no job.
fn may_exist(status: &Value) -> bool {
    !unsent(status) && status != "BLOCKED"
}

/// The one request that can start a job. Repeating it is safe: the server claims an
/// Action once, and refuses or answers from the record after that.
fn dispatch(client: &Client, id: &str, health: &Value) -> Result<Value, Error> {
    let current = show(client, id, health)?;
    if current["next_action"] == "REPREPARE" {
        return Err(Error::Submit(format!(
            "the prepared Submission {id} expired or was blocked, and nothing was submitted; \
             see `ml-exp run renew --help`"
        )));
    }
    if !unsent(&current["status"]) {
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
            reason: REASON,
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

/// Send once and say truthfully what is known afterwards. A lost reply says nothing about
/// whether the claim happened, so the server's record decides: EXECUTING or later goes on
/// to reconcile, and still PREPARED is reported as not sent, never as success.
fn send(client: &Client, id: &str, health: &Value) -> Result<Value, Error> {
    let current = match dispatch(client, id, health) {
        Ok(current) => current,
        Err(Error::SubmissionOutcomeUnknown(_)) => show(client, id, health)?,
        Err(error) => return Err(error),
    };
    if unsent(&current["status"]) {
        return Err(Error::Submit(format!(
            "as far as the server shows, no job has been sent: {id} is still {}. \
             Run this command again",
            current["status"].as_str().unwrap_or("unsent")
        )));
    }
    observe(client, id, current, health)
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
fn provider_state(submission: &Value) -> &str {
    submission["cloud_observation"]["observation"]["provider_state"]
        .as_str()
        .unwrap_or_default()
}
/// SenseCore also reports CANCELLED for a job that may still be running.
fn over(submission: &Value) -> bool {
    job_state(submission).is_some_and(finished)
        && !ml_exp_core::cloud::sensecore::may_still_run(provider_state(submission))
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
/// A provider read for a cloud job, which is what brings its view up to date. Slurm's
/// view comes from background polling, and only `GET` overlays it.
fn refreshed(client: &Client, id: &str, current: Value, health: &Value) -> Result<Value, Error> {
    if current.get("cloud_observation").is_none() {
        return show(client, id, health);
    }
    submissions::plan(submissions::Command::Refresh { submission: id })
        .and_then(|plan| submissions::execute(client, &plan, health))?;
    show(client, id, health)
}
/// Whether the view carries state newer than the moment of submission.
fn reporting(submission: &Value) -> bool {
    submission.get("cloud_observation").is_some()
        || submission.get("native_scheduler_observation").is_some()
}
/// Read-only. Stopping here, for any reason, leaves the job running. A long job is only
/// ever watched longer: time alone is not a reason to stop, only having nothing to read.
fn follow(
    client: &Client,
    id: &str,
    current: Value,
    health: &Value,
    hooks: &mut Hooks<'_>,
) -> Value {
    (hooks.progress)("Ctrl-C stops watching; the job keeps running");
    let mut latest = current;
    let mut last = String::new();
    let mut failures = 0;
    let mut changed = Instant::now();
    loop {
        // A cloud read that failed keeps the old state beside a diagnostic.
        let read = refreshed(client, id, latest.clone(), health).and_then(|value| {
            if value["cloud_observation"]["diagnostic"].is_null() {
                Ok(value)
            } else {
                Err(Error::Observation("the provider read failed"))
            }
        });
        match read {
            Ok(value) => {
                failures = 0;
                latest = value;
                if let Some(line) = describe(&latest).filter(|line| *line != last) {
                    (hooks.progress)(&line);
                    last = line;
                    changed = Instant::now();
                }
                if over(&latest) {
                    return latest;
                }
                if changed.elapsed() >= hooks.silence {
                    let why = if !reporting(&latest) {
                        Some(
                            "this server is not reporting the job's state, so it may have \
                             ended. Its background polling may be off",
                        )
                    } else if ml_exp_core::cloud::sensecore::may_still_run(provider_state(&latest))
                    {
                        Some("the provider has paused or is deleting the job")
                    } else {
                        None
                    };
                    if let Some(why) = why {
                        (hooks.progress)(&format!("stopped watching: {why}"));
                        latest["watching_stopped"] = json!(true);
                        return latest;
                    }
                }
            }
            Err(_) if failures < READ_FAILURES => failures += 1,
            Err(_) => {
                (hooks.progress)(
                    "stopped watching: the status cannot be read. The job keeps running",
                );
                latest["watching_stopped"] = json!(true);
                return latest;
            }
        }
        std::thread::sleep(hooks.follow_poll);
    }
}
/// What the backend says went wrong, when a job ended without success.
fn explain_failure(
    client: &Client,
    project: &str,
    run: &str,
    allocation: Option<&str>,
    hooks: &mut Hooks<'_>,
) {
    // PAI names no preempted state: a reclaimed job reads as FAILED or CANCELLED.
    if allocation.is_some_and(|name| name != "guaranteed") {
        (hooks.progress)(
            "This job could use idle resources, and PAI does not report a preemption as such: \
             it may have been reclaimed rather than have failed",
        );
    }
    let selection = observation::Selection {
        project: Some(project),
        run: Some(run),
        attempt: None,
    };
    if let Ok(value) = read::provider_diagnostics(client, None, selection, 40) {
        (hooks.progress)(&presentation::provider_diagnostics(&value));
    }
}
#[allow(clippy::too_many_arguments)]
fn watch(
    client: &Client,
    project: &str,
    run: &str,
    id: &str,
    allocation: Option<&str>,
    current: Value,
    health: &Value,
    hooks: &mut Hooks<'_>,
) -> Value {
    if !hooks.follow || current["status"] != "VERIFIED" {
        return current;
    }
    let latest = follow(client, id, current, health, hooks);
    if over(&latest) && job_state(&latest) != Some("SUCCEEDED") {
        explain_failure(client, project, run, allocation, hooks);
    }
    latest
}

fn outcome(
    spec: &Spec,
    identity: (&str, &str),
    state: &Path,
    submission: &Value,
    existing: bool,
) -> Value {
    let (run_id, preparation_id) = identity;
    let id = submission["submission_id"].as_str().unwrap_or_default();
    let state = state.display();
    json!({
        "project": spec.project,
        "run_id": run_id,
        "preparation_id": preparation_id,
        "submission_id": id,
        "status": submission["status"],
        "job_state": job_state(submission),
        "job_over": over(submission),
        "watching_stopped": submission["watching_stopped"] == true,
        "already_submitted": existing,
        "state_file": state.to_string(),
        "next": ["ml-exp status", "ml-exp logs", "ml-exp results"],
    })
}

/// Why this command should not read as a success, even though its output is complete.
pub fn failure(value: &Value) -> Option<String> {
    if value["job_over"] == true {
        return value["job_state"]
            .as_str()
            .filter(|state| *state != "SUCCEEDED")
            .map(|state| format!("the job ended {state}"));
    }
    if value["status"] != "VERIFIED" {
        return Some("the job is not confirmed yet; `ml-exp status` shows it".into());
    }
    (value["watching_stopped"] == true)
        .then(|| "stopped watching before the job ended; `ml-exp status` shows it".into())
}

/// The default output. `--json` gives the same fields.
pub fn report(value: &Value) -> String {
    let text = |key: &str| value[key].as_str().unwrap_or_default();
    let confirmed = value["status"] == "VERIFIED";
    let mut lines = vec![format!(
        "{} {}: {}",
        if !confirmed {
            "Sent, but not confirmed yet;"
        } else if value["already_submitted"] == true {
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
    let dockerfile = read_dockerfile(&spec.directory, &spec.dockerfile)?;
    let archive = sources::pack_directory(&spec.directory)?;
    if !archive.paths.iter().any(|path| *path == spec.dockerfile) {
        return Err(Error::Submit(format!(
            "{} is not among the files to upload; it must be inside the directory, \
             and not ignored by .gitignore",
            spec.dockerfile
        )));
    }
    let name = spec
        .directory
        .canonicalize()
        .ok()
        .and_then(|path| path.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
    let identity = identify(spec, &archive.content_sha256, &dockerfile, &name)?;
    ml_exp_io::create_private_dir(state_directory).map_err(|_| Error::Config)?;
    // The identity leaves the server out, and this state file belongs to one server.
    let server = canonical::digest(&json!(client.base_url())).map_err(|_| Error::Response)?;
    let state = state_directory.join(format!("{}-{}.json", identity.preparation_id, &server[..8]));
    let path = format!(
        "/api/projects/{}/experiment-preparations/{}",
        spec.project, identity.preparation_id
    );
    let saved = match client.get(&path) {
        Ok(view) => Some(view),
        Err(Error::Http(failure)) if failure.status == 404 => None,
        Err(error) => return Err(error),
    };
    // Before anything about today's executors can refuse: a job that exists stays reachable.
    if let Some(id) = saved
        .as_ref()
        .and_then(|view| view["submission"]["submission_id"].as_str())
    {
        let current = show(client, id, &health)?;
        if may_exist(&current["status"]) {
            // A later `retry` of this Run stays its record; this must not move it back.
            let known = latest(&run_records(
                state_directory,
                &spec.project,
                &identity.run_id,
            ))
            .ok()
            .or_else(|| {
                latest(state_directory)
                    .ok()
                    .filter(|newest| newest.run_id == identity.run_id)
            })
            .filter(|record| record.submission_id.is_some());
            remember(
                state_directory,
                &known.unwrap_or_else(|| latest_for(spec, &identity.run_id, Some(id), false)),
            );
            let current = observe(client, id, current, &health)?;
            let current = watch(
                client,
                &spec.project,
                &identity.run_id,
                id,
                saved
                    .as_ref()
                    .and_then(|view| view["frozen"]["allocation"].as_str()),
                current,
                &health,
                &mut hooks,
            );
            return Ok(outcome(
                spec,
                (&identity.run_id, &identity.preparation_id),
                &state,
                &current,
                true,
            ));
        }
    }
    let (plan, delivery) = match &saved {
        Some(view) => (
            frozen(identity, view)?,
            Delivery {
                environment: None,
                line: saved_environment(view, archive.paths.len()),
            },
        ),
        None => {
            let executors = client.get("/api/executors")?;
            let rows = executors["executors"]
                .as_array()
                .ok_or(Error::Response)?
                .as_slice();
            let executor = choose(spec, rows)?;
            let plan = resolve(spec, identity, executor)?;
            if executor["enabled"] != true || executor["run_creation"] != true {
                return Err(Error::Submit(format!(
                    "{} is not accepting new Runs",
                    plan.executor
                )));
            }
            let delivery = deliver(client, spec, &plan, &dockerfile, &archive, executor)?;
            (plan, delivery)
        }
    };
    if health["scheduler_mutations"] != true {
        return Err(Error::Observation(
            "the server does not allow submissions right now; nothing was sent",
        ));
    }
    let question = summary(spec, &plan, archive.paths.len(), &delivery.line);
    if !(hooks.confirm)(&question)? {
        return Err(Error::Observation("not submitted; nothing was sent"));
    }
    remember(
        state_directory,
        &latest_for(spec, &plan.run_id, None, false),
    );
    // Already prepared means already frozen. Otherwise the server has none, so a local
    // record from a request that never arrived must not stop this one.
    if saved.is_none() {
        let _ = std::fs::remove_file(&state);
        let (source_id, code_source_id) = match delivery.environment {
            Some(environment) => {
                let what = "uploading code";
                let code = publish(client, spec, archive, what, &health, hooks.progress)?;
                let what = "uploading what the Dockerfile uses";
                let source = publish(client, spec, environment, what, &health, hooks.progress)?;
                (source, Some(code))
            }
            None => {
                let what = "uploading source";
                let source = publish(client, spec, archive, what, &health, hooks.progress)?;
                (source, None)
            }
        };
        let request = request(spec, &plan, &source_id, code_source_id.as_deref())?;
        preparations::prepare(client, &spec.project, &request, &state, &health)?;
    }
    let id = prepared(client, spec, &plan, &health, &mut hooks)?;
    remember(
        state_directory,
        &latest_for(spec, &plan.run_id, Some(&id), true),
    );
    let current = send(client, &id, &health)?;
    remember(
        state_directory,
        &latest_for(
            spec,
            &plan.run_id,
            Some(&id),
            current["status"] != "VERIFIED",
        ),
    );
    let current = watch(
        client,
        &spec.project,
        &plan.run_id,
        &id,
        plan.allocation.as_deref(),
        current,
        &health,
        &mut hooks,
    );
    Ok(outcome(
        spec,
        (&plan.run_id, &plan.preparation_id),
        &state,
        &current,
        false,
    ))
}

/// Stop the latest job, after saying which one.
pub fn cancel(client: &Client, directory: &Path, hooks: Hooks<'_>) -> Result<Value, Error> {
    let health = client.health()?;
    let latest = latest(directory)?;
    let id = sent(&latest)?;
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

/// Run the latest Run again as a new Attempt, from a checkpoint or the start.
///
/// Repeating this must never start a further job by accident. A prepared Attempt that was
/// not sent is finished rather than replaced. If a send's outcome was never learned, the
/// repeat only finds out what happened. The server decides whether the last job has ended;
/// the client only refuses early where its own view is fresh.
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
    let parent = sent(&latest)?.to_owned();
    let current = show(client, &parent, &health)?;
    let attempt = current["attempt_id"]
        .as_str()
        .ok_or(Error::Response)?
        .to_owned();
    if current["status"] == "BLOCKED" {
        dispatch(client, &parent, &health)?;
        return Err(Error::Submit(format!(
            "{parent} is blocked; nothing was sent"
        )));
    }
    let existing_attempt = unsent(&current["status"]) || latest.unsettled;
    if checkpoint.is_some() && existing_attempt {
        return Err(Error::Submit(format!(
            "{attempt} of {} already exists and --checkpoint would not apply to it; \
             run `ml-exp retry` to finish it first",
            latest.run_id
        )));
    }
    let settled = |id: &str, unsettled: bool| Latest {
        project: latest.project.clone(),
        run_id: latest.run_id.clone(),
        submission_id: Some(id.to_owned()),
        unsettled,
    };
    let spend_of = |view: &Value| {
        view["preflight_summary"]["max_gpu_hours"]
            .as_f64()
            .unwrap_or(0.0)
    };
    let spend = spend_of(&current);
    let resources = current["preflight_summary"]["resources"].clone();
    let id = if unsent(&current["status"]) {
        let question = format!(
            "Send {} {attempt}, prepared but not sent yet\n  resources {resources}\n  spend     up to {spend} more GPU hours",
            latest.run_id,
        );
        if !(hooks.confirm)(&question)? {
            return Err(Error::Observation("not submitted; nothing was sent"));
        }
        parent.clone()
    } else {
        // Anything unresolved is resolved first, and only what is confirmed can be followed.
        let current = observe(client, &parent, current, &health)?;
        if current["status"] != "VERIFIED" {
            remember(directory, &settled(&parent, true));
            return Err(Error::Submit(format!(
                "{attempt} of {} is {}: a job may exist and its outcome is unresolved. \
                 Nothing new was sent; try again later, or see `ml-exp run reconcile --help`",
                latest.run_id,
                current["status"].as_str().unwrap_or("unknown"),
            )));
        }
        if latest.unsettled {
            remember(directory, &settled(&parent, false));
            let current = watch(
                client,
                &latest.project,
                &latest.run_id,
                &parent,
                None,
                current,
                &health,
                &mut hooks,
            );
            return Ok(retried(&latest, &parent, &current, true));
        }
        let current = refreshed(client, &parent, current.clone(), &health).unwrap_or(current);
        if reporting(&current) && current.get("cloud_observation").is_some() && !over(&current) {
            return Err(Error::Submit(format!(
                "{attempt} of {} has not ended ({}); `ml-exp cancel` stops it",
                latest.run_id,
                job_state(&current).unwrap_or("unknown"),
            )));
        }
        let question = format!(
            "Run {} again after {attempt}, from {}\n  resources {resources}\n  spend     up to {spend} more GPU hours",
            latest.run_id,
            checkpoint
                .as_deref()
                .unwrap_or("the start it was frozen with"),
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
                submission: &parent,
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
        prepared["submission_id"]
            .as_str()
            .ok_or(Error::Response)?
            .to_owned()
    };
    remember(directory, &settled(&id, true));
    let current = send(client, &id, &health)?;
    remember(directory, &settled(&id, current["status"] != "VERIFIED"));
    let current = watch(
        client,
        &latest.project,
        &latest.run_id,
        &id,
        None,
        current,
        &health,
        &mut hooks,
    );
    Ok(retried(&latest, &id, &current, false))
}
fn retried(latest: &Latest, id: &str, current: &Value, existing: bool) -> Value {
    json!({
        "project": latest.project,
        "run_id": latest.run_id,
        "attempt_id": current["attempt_id"],
        "submission_id": id,
        "status": current["status"],
        "job_state": job_state(current),
        "job_over": over(current),
        "watching_stopped": current["watching_stopped"] == true,
        "already_submitted": existing,
        "next": ["ml-exp status", "ml-exp logs", "ml-exp results"],
    })
}

fn read_dockerfile(directory: &Path, name: &str) -> Result<String, Error> {
    let path = directory.join(name);
    let missing = || {
        Error::Submit(format!(
            "{} needs a Dockerfile starting with FROM IMAGE@sha256:DIGEST; \
             `docker buildx imagetools inspect IMAGE` prints the digest",
            directory.display()
        ))
    };
    if std::fs::metadata(&path).map_err(|_| missing())?.len() > DOCKERFILE_LIMIT {
        return Err(Error::Submit("the Dockerfile exceeds 256 KiB".into()));
    }
    std::fs::read_to_string(&path).map_err(|_| missing())
}

/// Whether the code leaves the image, and how the question puts it. `environment` is the
/// archive to build from when it does, and the whole directory is the code either way.
struct Delivery {
    environment: Option<sources::PreparedArchive>,
    line: String,
}
/// Decided only by the directory and the executor, so a repeat decides the same.
fn deliver(
    client: &Client,
    spec: &Spec,
    plan: &Plan,
    dockerfile: &str,
    archive: &sources::PreparedArchive,
    executor: &Value,
) -> Result<Delivery, Error> {
    use dockerfile::Split;
    let files = archive.paths.len();
    let delivery = &executor["capabilities"]["code_delivery"];
    // An earlier server names only the worker, and its Runtimes are named by that.
    let worker = delivery["runtime_identity"]
        .as_str()
        .or(delivery["worker_sha256"].as_str());
    let mib = |bytes: u64| bytes / (1024 * 1024);
    let split = match worker {
        None => Split::Whole("this executor keeps code in the image".into()),
        // Too large to go with a job is no reason to stop one that works from the image.
        Some(_) if archive.expanded_bytes > ml_exp_core::sources::MAX_CODE_BYTES => {
            Split::Whole(format!(
                "the code is {} MiB and only {} MiB can go with a job; move large files to data \
                 with `ml-exp data upload` to stop rebuilding on every change",
                mib(archive.expanded_bytes),
                mib(ml_exp_core::sources::MAX_CODE_BYTES),
            ))
        }
        Some(_) if !sources::listing_fits(&archive.paths) => {
            Split::Whole("the list of files is too long to go with a job".into())
        }
        Some(_) if spec.extra.contains_key("data_preparation") => {
            Split::Whole("data_preparation needs its code in the image".into())
        }
        Some(_) => dockerfile::split(dockerfile, &spec.dockerfile, &archive.paths),
    };
    let built_from = match &split {
        Split::Files(paths) => paths.as_slice(),
        Split::Whole(_) => archive.paths.as_slice(),
    };
    let reused =
        worker.is_some_and(|worker| environment_ready(client, spec, plan, built_from, worker));
    Ok(match split {
        Split::Files(paths) => {
            let line = if reused {
                format!("reused; {files} code files go with the job")
            } else {
                format!(
                    "will be built from {} files; {files} code files go with the job",
                    paths.len()
                )
            };
            Delivery {
                environment: Some(sources::pack_subset(
                    archive,
                    &paths,
                    (spec.dockerfile.as_str(), dockerfile),
                )?),
                line,
            }
        }
        Split::Whole(why) => Delivery {
            environment: None,
            line: if reused {
                format!("reused; its {files} files hold the code, because {why}")
            } else if worker.is_none() {
                format!("built with the code, {files} files, because {why}")
            } else {
                format!("will be built with the code, {files} files, because {why}")
            },
        },
    })
}
/// Only for the question, so anything it cannot learn reads as a build to come.
fn environment_ready(
    client: &Client,
    spec: &Spec,
    plan: &Plan,
    files: &[String],
    worker: &str,
) -> bool {
    let id = sources::local_source_id(&spec.directory, files)
        .ok()
        .and_then(|source_id| {
            build::Definition {
                source_id,
                executor: plan.executor.clone(),
                base_image: plan.base_image.clone(),
                dockerfile: spec.dockerfile.clone(),
            }
            .offered_id(&spec.project, worker)
            .ok()
        });
    id.is_some_and(|id| {
        client
            .get(&format!("/api/projects/{}/runtimes/{id}", spec.project))
            .is_ok_and(|view| view["status"] == "READY")
    })
}
/// A saved preparation already decided, and the server has what it decided.
fn saved_environment(view: &Value, files: usize) -> String {
    let state = if view["runtime"]["status"] == "READY" {
        "reused"
    } else {
        "is being built"
    };
    if view["frozen"]["code_source_id"].is_string() {
        format!("{state}; {files} code files go with the job")
    } else {
        format!("{state}; its {files} files hold the code")
    }
}

/// Identical bytes have one receipt, so a previous upload is simply reused.
fn publish(
    client: &Client,
    spec: &Spec,
    archive: sources::PreparedArchive,
    what: &str,
    health: &Value,
    progress: &mut dyn FnMut(&str),
) -> Result<String, Error> {
    let receipt = match sources::lookup_archive(client, &spec.project, &archive.sha256, health) {
        Ok(receipt) => receipt,
        Err(Error::Http(failure)) if failure.status == 404 => {
            progress(what);
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

/// Repeat the same safe continue until a Submission exists. A failure that happens
/// here stops here rather than being retried: continue would start it again. Only what
/// an earlier command left is started over, once, by the next continue.
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
    let mut builder_idle = false;
    let mut tick = 0;
    let mut first_view = true;
    loop {
        if tick % ADVANCE_EVERY == 0 {
            match preparations::advance(client, &spec.project, id, &confirmation) {
                Ok(view) => {
                    lost_replies = 0;
                    if let Some(submission) = view["submission"]["submission_id"].as_str() {
                        return Ok(submission.to_owned());
                    }
                    if std::mem::take(&mut first_view) && may_build_again(&view) {
                        (hooks.progress)("the last build left nothing behind; building again");
                    } else if let Some(reason) = stopped(&view, builder_idle) {
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
            if let Some(idle) =
                show_build(client, spec, runtime, health, &mut build, hooks.progress)
            {
                builder_idle = idle;
            }
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
            &"a".repeat(64),
            &format!("FROM {PINNED}\n"),
            &json!({"executor_id":spec.executor,"limits":{}}),
            "source",
        )
        .unwrap()
    }
    fn row(executor: &str, kind: &str) -> Value {
        json!({"executor_id":executor,"kind":kind,"limits":{}})
    }
    fn on(spec: &Spec, executor: &Value) -> Result<Plan, Error> {
        plan(
            spec,
            &"a".repeat(64),
            &format!("FROM {PINNED}\n"),
            executor,
            "source",
        )
    }
    fn frozen_run(spec: &Spec, plan: &Plan) -> Map<String, Value> {
        let source = format!("source.{}", "b".repeat(64));
        request(spec, plan, &source, None).unwrap().run
    }
    fn choosing(allocation: &str) -> Spec {
        let mut chosen = spec();
        chosen
            .extra
            .insert("scheduling".into(), json!({"allocation": allocation}));
        chosen
    }
    /// Computed separately from the array `identify` hashes, as it stood before this
    /// option existed. A repeat of an old command must still find its old job.
    #[test]
    fn with_no_choice_the_identity_is_the_one_saved_jobs_have() {
        let plan = on(&spec(), &row("pai-h200", "pai-dlc")).unwrap();
        assert_eq!(plan.request_id, "d15e3a7b64b5196d01b80a6cb8d74587");
        assert_eq!(plan.run_id, "source-d15e3a7b");
    }
    #[test]
    fn a_new_pai_job_with_gpus_freezes_the_default_and_the_question_says_it_may_be_preempted() {
        let spec = spec();
        let plan = on(&spec, &row("pai-h200", "pai-dlc")).unwrap();
        assert_eq!(plan.allocation.as_deref(), Some("allow_idle"));
        assert_eq!(
            frozen_run(&spec, &plan)["scheduling"],
            json!({"allocation":"allow_idle"})
        );
        let question = summary(&spec, &plan, 1, "built");
        assert!(question.contains("allocation allow_idle"), "{question}");
        assert!(question.contains("MAY BE PREEMPTED"), "{question}");
    }
    #[test]
    fn each_allocation_is_its_own_identity_and_the_flag_is_the_file_spelling() {
        let from_flag = |allocation: &str| {
            spec_of(
                Options {
                    project: Some("demo".into()),
                    executor: Some("pai-h200".into()),
                    directory: "source".into(),
                    command: vec!["python3".into(), "train.py".into()],
                    gpus: Some(1),
                    walltime_seconds: Some(600),
                    allocation: Some(allocation.into()),
                    ..Options::default()
                },
                None,
            )
            .unwrap()
        };
        let pai = row("pai-h200", "pai-dlc");
        let none = on(&spec(), &pai).unwrap();
        let mut ids = vec![none.request_id];
        for name in ["guaranteed", "allow_idle", "idle_only"] {
            let plan = on(&from_flag(name), &pai).unwrap();
            assert_eq!(plan.allocation.as_deref(), Some(name));
            assert_ne!(plan.run_id, none.run_id, "{name}");
            let file = on(&choosing(name), &pai).unwrap();
            assert_eq!(file.request_id, plan.request_id, "{name}");
            ids.push(plan.request_id);
        }
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 4);
        // The flag wins over the file, and keeps what else the file said.
        let file = br#"{"run":{"scheduling":{"allocation":"idle_only","priority":7}}}"#;
        let merged = spec_of(
            Options {
                project: Some("demo".into()),
                executor: Some("pai-h200".into()),
                command: vec!["x".into()],
                gpus: Some(1),
                walltime_seconds: Some(600),
                allocation: Some("guaranteed".into()),
                ..Options::default()
            },
            Some(file),
        )
        .unwrap();
        assert_eq!(
            merged.extra["scheduling"],
            json!({"allocation":"guaranteed","priority":7})
        );
    }
    #[test]
    fn a_guaranteed_choice_says_so_and_a_priority_alone_gets_the_default() {
        let chosen = choosing("guaranteed");
        let plan = on(&chosen, &row("pai-h200", "pai-dlc")).unwrap();
        let question = summary(&chosen, &plan, 1, "built");
        assert!(question.contains("allocation guaranteed"), "{question}");
        assert!(!question.contains("PREEMPTED"), "{question}");
        assert_eq!(
            frozen_run(&chosen, &plan)["scheduling"]["allocation"],
            "guaranteed"
        );

        let mut priority = spec();
        priority
            .extra
            .insert("scheduling".into(), json!({"priority":7}));
        let plan = on(&priority, &row("pai-h200", "pai-dlc")).unwrap();
        assert_eq!(
            frozen_run(&priority, &plan)["scheduling"],
            json!({"allocation":"allow_idle","priority":7})
        );
    }
    #[test]
    fn a_pai_job_with_no_gpus_stays_as_it_was() {
        let mut cpu = spec();
        cpu.gpus = 0;
        let plan = on(&cpu, &row("pai-native-cpu", "pai-dlc")).unwrap();
        assert_eq!(plan.allocation.as_deref(), Some("guaranteed"));
        assert!(!frozen_run(&cpu, &plan).contains_key("scheduling"));
    }
    #[test]
    fn slurm_and_sensecore_are_asked_nothing_new() {
        for executor in [row("slurm-h100", "slurm"), row("sense", "sensecore")] {
            let spec = spec();
            let plan = on(&spec, &executor).unwrap();
            assert_eq!(plan.allocation, None);
            assert!(!frozen_run(&spec, &plan).contains_key("scheduling"));
            assert!(!summary(&spec, &plan, 1, "built").contains("allocation"));
        }
        for name in ["allow_idle", "idle_only"] {
            let refused = on(&choosing(name), &row("sense", "sensecore"));
            assert!(matches!(refused, Err(Error::Submit(_))), "{name}");
        }
        assert!(on(&choosing("guaranteed"), &row("sense", "sensecore")).is_ok());
        assert!(matches!(
            on(&choosing("guaranteed"), &row("slurm-h100", "slurm")),
            Err(Error::Submit(_))
        ));
    }
    #[test]
    fn a_saved_preparation_is_shown_as_it_was_frozen() {
        let identity = identify(
            &spec(),
            &"a".repeat(64),
            &format!("FROM {PINNED}\n"),
            "source",
        )
        .unwrap();
        let view = json!({"frozen":{"executor":"pai-h200","gpus":1,"cpus":8,"memory_gb":32,
            "max_gpu_hours":1.0,"allocation":"guaranteed"}});
        let plan = frozen(identity, &view).unwrap();
        assert_eq!(plan.allocation.as_deref(), Some("guaranteed"));
        // An older server does not say.
        let identity = identify(
            &spec(),
            &"a".repeat(64),
            &format!("FROM {PINNED}\n"),
            "source",
        )
        .unwrap();
        let view = json!({"frozen":{"executor":"pai-h200","gpus":1,"cpus":8,"memory_gb":32,
            "max_gpu_hours":1.0}});
        assert_eq!(frozen(identity, &view).unwrap().allocation, None);
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
        let file = br#"{"runtime":{"dockerfile":"docker/Dockerfile"},
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
            br#"{"request_id":"11111111111111111111111111111111"}"#,
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
            None,
        )
        .unwrap();
        assert!(!request.run.contains_key("code_source_id"));
        assert_eq!(request.run["env"]["KEY"], "value");
        let mut reserved = spec();
        reserved.env.insert("OUTPUT_DIR".into(), "x".into());
        assert!(
            plan(
                &reserved,
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
    fn an_experiment_file_cannot_fix_the_code_source() {
        let file = br#"{"run":{"executor":"pai","command":["train"],"code_source_id":"source.x"}}"#;
        let refused = spec_of(
            Options {
                project: Some("p".into()),
                gpus: Some(1),
                walltime_seconds: Some(60),
                ..Options::default()
            },
            Some(file),
        )
        .err()
        .unwrap();
        assert!(refused.to_string().contains("code_source_id"), "{refused}");
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
        assert!(stopped_now(&json!({"runtime":{"status":"EXECUTING"}})).is_none());
        assert!(stopped_now(&json!({"runtime":{"status":"PREPARED"}})).is_none());
        assert!(
            stopped_now(
                &json!({"runtime":{"status":"EXECUTING","build_error":{"code":"BUILD_EXECUTING"}}})
            )
            .is_none()
        );
        for view in [
            json!({"runtime":{"status":"PREPARED","build_error":{"code":"BAD"}}}),
            json!({"runtime":{"status":"RECONCILE_REQUIRED","build_error":{"code":"BAD"}}}),
            json!({"error_code":"SCRIPT_FAILED"}),
            json!({"data":{"new_request_allowed":true}}),
        ] {
            assert!(stopped_now(&view).is_some(), "{view}");
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
    fn only_a_prepared_submission_can_be_sent_and_only_a_sent_one_may_have_a_job() {
        assert!(unsent(&json!("PREPARED")));
        assert!(!may_exist(&json!("PREPARED")));
        assert!(!unsent(&json!("BLOCKED")));
        assert!(!may_exist(&json!("BLOCKED")));
        for status in ["EXECUTING", "RECONCILE_REQUIRED", "VERIFIED"] {
            assert!(!unsent(&json!(status)));
            assert!(may_exist(&json!(status)));
        }
        assert!(may_exist(&Value::Null), "unknown is treated as maybe sent");
    }
    #[test]
    fn a_job_that_may_still_run_is_not_over() {
        let job = |state: &str, own: &str| json!({"cloud_observation":{"observation":{"state":state,"provider_state":own}}});
        assert!(over(&job("FAILED", "Failed")));
        assert!(over(&job("CANCELLED", "STOPPED")));
        assert!(!over(&job("CANCELLED", "DELETING")));
        assert!(!over(&job("CANCELLED", "SUSPENDED")));
        assert!(!over(&job("RUNNING", "Running")));
    }
    #[test]
    fn what_the_builder_refuses_is_refused_before_anything_is_confirmed() {
        for text in [
            format!("# syntax=docker/dockerfile:1\nFROM {PINNED}\n"),
            format!("# escape = `\nFROM {PINNED}\n"),
            format!("FROM --platform=linux/arm64 {PINNED}\n"),
            format!("FROM {PINNED} foo\n"),
            format!("FROM {PINNED} AS\n"),
        ] {
            assert!(base_image(&text).is_err(), "{text}");
        }
        for text in [
            format!("# a comment\nFROM \"{PINNED}\"\n"),
            format!("FROM \\\n  # note\n  {PINNED}\n"),
        ] {
            assert_eq!(base_image(&text).unwrap(), PINNED, "{text}");
        }
    }
    #[test]
    fn an_unconfirmed_send_says_so_and_a_failed_job_is_named() {
        let reported = report(&json!({"status":"RECONCILE_REQUIRED","run_id":"r","next":[]}));
        assert!(reported.contains("not confirmed"), "{reported}");
        let reported = report(&json!({"status":"VERIFIED","run_id":"r","next":[]}));
        assert!(reported.starts_with("Submitted"), "{reported}");
        assert_eq!(
            failure(&json!({"status":"VERIFIED","job_state":"FAILED","job_over":true})).as_deref(),
            Some("the job ended FAILED")
        );
        assert!(
            failure(&json!({"status":"VERIFIED","job_state":"SUCCEEDED","job_over":true}))
                .is_none()
        );
        assert!(failure(&json!({"status":"VERIFIED","job_state":"RUNNING"})).is_none());
        assert!(
            failure(&json!({"status":"VERIFIED","job_state":"CANCELLED"})).is_none(),
            "may still run"
        );
        assert!(failure(&json!({"status":"RECONCILE_REQUIRED"})).is_some());
        assert!(failure(&json!({"status":"VERIFIED","watching_stopped":true})).is_some());
    }
    fn stopped_now(view: &Value) -> Option<String> {
        stopped(view, false)
    }
    #[test]
    fn an_earlier_run_can_still_be_named() {
        let root = tempfile::tempdir().unwrap();
        let record = |project: &str, run: &str| Latest {
            project: project.into(),
            run_id: run.into(),
            submission_id: Some(format!("action-{run}")),
            unsettled: false,
        };
        remember(root.path(), &record("demo", "first"));
        remember(root.path(), &record("demo", "second"));
        remember(root.path(), &record("other", "second"));
        assert_eq!(latest(root.path()).unwrap().project, "other");
        let first = run_directory(root.path(), None, "first").unwrap();
        assert_eq!(latest(&first).unwrap().run_id, "first");
        assert!(
            run_directory(root.path(), None, "second").is_err(),
            "ambiguous"
        );
        let second = run_directory(root.path(), Some("demo"), "second").unwrap();
        assert_eq!(latest(&second).unwrap().project, "demo");
        assert!(run_directory(root.path(), Some("demo"), "absent").is_err());
        assert!(run_directory(root.path(), Some(".."), "first").is_err());
        // Acting on a named Run makes it the latest, without another copy beneath it.
        let mut newer = record("demo", "first");
        newer.submission_id = Some("action-retry".into());
        remember(&first, &newer);
        assert!(!first.join(RUNS).exists());
        assert_eq!(latest(root.path()).unwrap(), newer);
        assert_eq!(latest(&first).unwrap(), newer);
    }
    #[test]
    fn proof_that_nothing_was_left_ends_the_wait_within_one_command() {
        let view = json!({"project":"demo","runtime":{"status":"PREPARED","retry_safe":true,
            "runtime_id":"runtime.x","build_error":{"code":"BUILD_RECEIPT_MISSING"}}});
        assert!(
            stopped(&view, false).is_some(),
            "the next continue would build again"
        );
    }
    #[test]
    fn only_a_proven_failure_may_build_again() {
        let view = |safe| json!({"runtime":{"retry_safe":safe,"build_error":{"code":"X"}}});
        assert!(may_build_again(&view(true)));
        assert!(!may_build_again(&view(false)));
        assert!(
            !may_build_again(&json!({"runtime":{"retry_safe":true,"build_error":null}})),
            "a fresh Runtime has nothing to recover from"
        );
    }
    #[test]
    fn a_failure_with_no_reason_shows_its_error_class() {
        let view = json!({"runtime":{"status":"RECONCILE_REQUIRED","runtime_id":"runtime.x",
            "build_error":{"code":"BUILD_REQUEST_FAILED","details":{"error_class":"KeyError"}}}});
        let text = stopped_now(&view).unwrap();
        assert!(text.contains("BUILD_REQUEST_FAILED (KeyError)"), "{text}");
    }
    #[test]
    fn a_missing_receipt_waits_until_the_builder_says_it_is_idle() {
        for code in ["BUILD_RECEIPT_MISSING", "NATIVE_BUILD_RPC_UNCERTAIN"] {
            let view = json!({"project":"demo","runtime":{"status":"RECONCILE_REQUIRED",
                "runtime_id":"runtime.x","build_error":{"code":code}}});
            assert!(
                stopped(&view, false).is_none(),
                "the build may still be running"
            );
            let text = stopped(&view, true).unwrap();
            assert!(
                text.contains("unresolved") && !text.contains("failed"),
                "{text}"
            );
            assert!(text.contains("--project demo"), "{text}");
        }
    }
    #[test]
    fn an_uncertain_cause_waits_too_even_beside_a_later_missing_receipt() {
        let view = json!({"runtime":{"status":"RECONCILE_REQUIRED","runtime_id":"runtime.x",
            "build_error":{"code":"BUILD_RECEIPT_MISSING",
                "build_failure":{"code":"NATIVE_BUILD_RPC_UNCERTAIN"}}}});
        assert!(
            stopped(&view, false).is_none(),
            "the build may still be running"
        );
        assert!(stopped(&view, true).unwrap().contains("unresolved"));
    }
    #[test]
    fn a_saved_preparation_shows_what_was_frozen_not_what_today_would_choose() {
        let identity = identify(
            &spec(),
            &"a".repeat(64),
            &format!("FROM {PINNED}\n"),
            "source",
        )
        .unwrap();
        let view = json!({"frozen":{"executor":"sc-2gpu","gpus":2,"cpus":28,"memory_gb":396,"max_gpu_hours":0.34}});
        let plan = frozen(identity, &view).unwrap();
        assert_eq!(plan.executor, "sc-2gpu");
        assert_eq!((plan.gpus, plan.cpus, plan.memory_gb), (2, 28, 396));
        assert_eq!(plan.max_gpu_hours, 0.34);
        let identity = identify(
            &spec(),
            &"a".repeat(64),
            &format!("FROM {PINNED}\n"),
            "source",
        )
        .unwrap();
        assert!(
            frozen(identity, &json!({})).is_err(),
            "unknown is refused, not guessed"
        );
    }
    #[test]
    fn a_build_that_failed_stops_the_wait_with_its_original_cause() {
        let view = json!({"runtime":{"status":"RECONCILE_REQUIRED","runtime_id":"runtime.x",
            "build_error":{"code":"BUILD_RECEIPT_MISSING",
                "build_failure":{"code":"BAD_DOCKERFILE","details":{"reason":"no_digest"}}}}});
        let text = stopped_now(&view).unwrap();
        assert!(
            text.contains("BAD_DOCKERFILE") && text.contains("no_digest"),
            "{text}"
        );
    }
    #[test]
    fn a_family_choice_or_a_changed_limit_is_still_the_same_job() {
        let mut family = spec();
        family.executor = "sc".into();
        let at = |row: Value| {
            plan(
                &family,
                &"a".repeat(64),
                &format!("FROM {PINNED}\n"),
                &row,
                "source",
            )
            .unwrap()
        };
        let small = at(shape("sc-1gpu-a", 1, 14, 198));
        let large = at(shape("sc-2gpu-a", 2, 28, 396));
        assert_eq!(small.request_id, large.request_id);
        assert_eq!(small.run_id, large.run_id);
        assert_ne!(small.executor, large.executor);
    }
    #[test]
    fn the_directory_name_is_part_of_the_job_and_of_its_name() {
        let named = |name: &str| {
            plan(
                &spec(),
                &"a".repeat(64),
                &format!("FROM {PINNED}\n"),
                &json!({"executor_id":"pai-h200","limits":{}}),
                name,
            )
            .unwrap()
        };
        assert_ne!(named("exp").request_id, named("exp2").request_id);
        assert!(named("exp").run_id.starts_with("exp-"));
    }
}
