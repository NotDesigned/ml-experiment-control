//! Optional MLflow publication over REST, beside and independent of W&B. Each
//! Attempt becomes one MLflow run in its project's experiment. A point's
//! timestamp is the run's start time plus its journal sequence, so every batch
//! is deterministic: a 200 from `runs/log-batch` advances the cursor, and after
//! an uncertain outcome the same batch is simply sent again. MLflow's SQL store
//! keeps one row per identical point, so a replay adds nothing.
#[cfg(test)]
pub(crate) mod fake;
mod mapping;
mod rest;
#[cfg(test)]
mod tests;

use super::Source;
use crate::background::Worker;
use crate::records::{RecordGuard, Store};
use ml_exp_core::{records::Scope, tracking::MlflowRouting};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

/// Batches one scope sends before the publisher turns to the next Run.
const CHUNKS_PER_VISIT: usize = 8;
/// Journal events read for one batch; MLflow takes at most 1000 points.
const EVENTS_PER_BATCH: i64 = 1000;
const WINDOW_BYTES: usize = 4 * 1024 * 1024;
/// An idle scope costs a few local reads and no MLflow request.
const IDLE_SECONDS: u64 = 2;
const DISCOVER_SECONDS: u64 = 30;

const ACK_PENDING: &str = "MLFLOW_ACK_PENDING";
const SYNCED: &str = "MLFLOW_SYNCED";
const FINISHED: &str = "MLFLOW_FINISHED";
const RETRY_PENDING: &str = "MLFLOW_RETRY_PENDING";
const STOPPED: &str = "MLFLOW_STOPPED";

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid MLflow publisher configuration")]
    Configuration,
    #[error("MLflow journal or private publication store unavailable")]
    Storage,
    #[error("MLflow publication identity is invalid")]
    Identity,
    #[error("MLflow publication cannot prove the Run's journal authority")]
    Journal,
    #[error("MLflow server unreachable; journal and cursor retained")]
    Unreachable,
    #[error("MLflow refused the configured credentials")]
    AuthFailed,
    #[error("MLflow has no such resource")]
    NotFound,
    #[error("MLflow resource already exists")]
    AlreadyExists,
    #[error("MLflow rejected a publication request")]
    Rejected,
    #[error("MLflow server error")]
    ServerError,
    #[error("MLflow response is invalid")]
    ResponseInvalid,
    #[error("the MLflow run of this Attempt was deleted")]
    RunMissing,
    #[error("the MLflow experiment of this project is deleted")]
    ExperimentDeleted,
}
impl Error {
    /// The fixed diagnostic shown in tracking status.
    pub fn code(self) -> &'static str {
        match self {
            Self::Configuration => "MLFLOW_CONFIGURATION_INVALID",
            Self::Storage => "PUBLISHER_UNAVAILABLE",
            Self::Identity => "MLFLOW_RECONCILE_REQUIRED",
            Self::Journal => "JOURNAL_AUTHORITY_UNAVAILABLE",
            Self::Unreachable => "MLFLOW_UNREACHABLE",
            Self::AuthFailed => "MLFLOW_AUTH_FAILED",
            Self::NotFound | Self::AlreadyExists | Self::Rejected => "MLFLOW_REQUEST_REJECTED",
            Self::ServerError => "MLFLOW_SERVER_ERROR",
            Self::ResponseInvalid => "MLFLOW_RESPONSE_INVALID",
            Self::RunMissing => "MLFLOW_RUN_MISSING",
            Self::ExperimentDeleted => "MLFLOW_EXPERIMENT_DELETED",
        }
    }
}
type Result<T> = std::result::Result<T, Error>;

/// The optional `mlflow` block of `tracking_config`.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default = "enabled")]
    pub enabled: bool,
    pub url: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password_file: Option<PathBuf>,
    #[serde(default)]
    pub ca_file: Option<PathBuf>,
    #[serde(default)]
    pub experiment_prefix: Option<String>,
    #[serde(default)]
    pub projects: Option<BTreeSet<String>>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
}
fn enabled() -> bool {
    true
}
impl Config {
    pub fn from_value(value: Value) -> Result<Self> {
        let config: Self = serde_json::from_value(value).map_err(why!(Error::Configuration))?;
        config.base()?;
        let printable = |text: &str, limit: usize| {
            !text.is_empty() && text.chars().count() <= limit && !text.chars().any(char::is_control)
        };
        match (&config.username, &config.password_file) {
            (None, None) => {}
            (Some(username), Some(path))
                if printable(username, 256) && !username.contains(':') && path.is_absolute() => {}
            _ => return Err(Error::Configuration),
        }
        if config
            .ca_file
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
            || config
                .experiment_prefix
                .as_deref()
                .is_some_and(|prefix| !printable(prefix, 128))
            || config.projects.as_ref().is_some_and(|projects| {
                projects
                    .iter()
                    .any(|p| ml_exp_core::identity::validate_id(p).is_err())
            })
            || config
                .timeout_seconds
                .is_some_and(|seconds| !(1..=300).contains(&seconds))
        {
            return Err(Error::Configuration);
        }
        Ok(config)
    }
    /// The server address without a trailing slash. Credentials never travel
    /// in clear text: plain HTTP is accepted only for a loopback server.
    fn base(&self) -> Result<String> {
        let url = url::Url::parse(&self.url).map_err(why!(Error::Configuration))?;
        let loopback = match url.host() {
            Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
            Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
            Some(url::Host::Domain(name)) => name == "localhost",
            None => return Err(Error::Configuration),
        };
        if !matches!(url.scheme(), "https" | "http")
            || url.scheme() == "http" && self.username.is_some() && !loopback
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(Error::Configuration);
        }
        Ok(url.as_str().trim_end_matches('/').to_owned())
    }
    fn password(&self) -> Result<Option<String>> {
        let Some(path) = &self.password_file else {
            return Ok(None);
        };
        let bytes = ml_exp_io::read_private(path, 4096, true).map_err(why!(
            Error::Configuration,
            "MLflow password file unreadable"
        ))?;
        let password = std::str::from_utf8(&bytes)
            .map_err(why!(
                Error::Configuration,
                "MLflow password encoding invalid"
            ))?
            .trim_end_matches(['\r', '\n']);
        if password.is_empty() || password.chars().any(char::is_control) {
            return Err(Error::Configuration);
        }
        Ok(Some(password.to_owned()))
    }
    /// A CA certificate is public, so it may be readable by anyone.
    fn ca(&self) -> Result<Option<Vec<u8>>> {
        use std::io::Read;
        const LIMIT: u64 = 1024 * 1024;
        let Some(path) = &self.ca_file else {
            return Ok(None);
        };
        let mut pem = Vec::new();
        std::fs::File::open(path)
            .and_then(|file| file.take(LIMIT + 1).read_to_end(&mut pem))
            .map_err(why!(Error::Configuration, "MLflow CA file unreadable"))?;
        if pem.len() as u64 > LIMIT {
            return Err(Error::Configuration);
        }
        Ok(Some(pem))
    }
    /// Local validation only: no MLflow request.
    pub fn check_references(&self) -> Result<()> {
        self.password()?;
        if let Some(pem) = self.ca()? {
            reqwest::Certificate::from_pem(&pem).map_err(why!(Error::Configuration))?;
        }
        Ok(())
    }
    pub fn routing(&self) -> MlflowRouting {
        MlflowRouting {
            enabled: self.enabled,
            experiment_prefix: self.experiment_prefix.clone().unwrap_or_default(),
            projects: self.projects.clone(),
        }
    }
}

pub struct Runtime {
    config: Config,
    base: String,
    client: rest::Client,
}
impl Runtime {
    pub fn open(config: Config) -> Result<Self> {
        let base = config.base()?;
        let auth = match (&config.username, config.password()?) {
            (Some(username), Some(password)) => Some((username.clone(), password)),
            _ => None,
        };
        let client = rest::Client::new(
            &base,
            auth,
            config.ca()?.as_deref(),
            Duration::from_secs(config.timeout_seconds.unwrap_or(60)),
        )?;
        Ok(Self {
            config,
            base,
            client,
        })
    }
    pub fn enabled(&self) -> bool {
        self.config.enabled
    }
    pub fn routing(&self) -> MlflowRouting {
        self.config.routing()
    }
}

/// Private publication state of one scope, saved in `tracking/outbox.sqlite`.
/// Every field may be absent, so a later field never breaks an earlier state.
#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    #[serde(default)]
    status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    /// Every journal sequence up to this one is committed on MLflow.
    #[serde(default)]
    confirmed: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    experiment_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
    /// The MLflow run's start time (ms): a point's timestamp is this plus its sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    timestamp_base: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    params_logged: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    skipped_params: Vec<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    skipped_points: u64,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    skipped_names: BTreeSet<String>,
    /// The MLflow run carries its terminal status for every record up to `confirmed`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    finished: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    terminal_status: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    failures: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_attempt_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_confirmed_at: Option<String>,
}
fn is_zero(value: &u64) -> bool {
    *value == 0
}

/// What the publisher thread remembers between visits: nothing it cannot read again.
#[derive(Default)]
pub(crate) struct Session {
    experiments: BTreeMap<String, String>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Next {
    /// More records are waiting: come back as soon as the other Runs had a turn.
    Backlog,
    Idle,
}

pub struct Publisher {
    _worker: Worker,
}
impl Publisher {
    /// One thread of its own, so a slow or failing MLflow server never delays
    /// W&B publication, and a W&B failure never delays MLflow.
    pub fn start(service: Arc<dyn Source>, runtime: Runtime) -> Result<Self> {
        if !runtime.enabled() {
            return Err(Error::Configuration);
        }
        let store = service.publisher_store().map_err(why!(Error::Storage))?;
        let leader = leader(&store)?;
        let worker = Worker::spawn("native-mlflow", move |stop| {
            let _leader = leader;
            let flag = stop.flag();
            let start = Instant::now();
            let policy = ml_exp_core::polling::Policy {
                enabled: true,
                ..Default::default()
            };
            let mut schedule = ml_exp_core::polling::Schedule::default();
            let mut session = Session::default();
            let mut candidates = BTreeSet::new();
            let mut discover_at = 0;
            while !stop.requested() {
                let elapsed = start.elapsed().as_secs();
                if elapsed >= discover_at {
                    if let Ok(scopes) = candidate_scopes(&store) {
                        candidates = scopes;
                    }
                    discover_at = elapsed.saturating_add(DISCOVER_SECONDS);
                }
                let (next, selected) = schedule.select(&candidates, elapsed, &policy);
                schedule = next;
                let Some((scope, _)) = selected else {
                    if !stop.pause() {
                        break;
                    }
                    continue;
                };
                let outcome =
                    publish_scope(&*service, &store, &runtime, &mut session, &scope, flag);
                let now = start.elapsed().as_secs();
                schedule = match &outcome {
                    Ok(Next::Backlog) => schedule.due_at(&scope, now),
                    Ok(Next::Idle) => schedule.due_at(&scope, now.saturating_add(IDLE_SECONDS)),
                    Err(_) => schedule.outcome(&scope, now, &policy, false, false),
                };
                if let Err(error) = outcome {
                    if !stop.requested() {
                        record_failure(&store, &scope, error);
                    }
                }
            }
        })
        .map_err(why!(Error::Storage))?;
        Ok(Self { _worker: worker })
    }
}

/// Its own leadership lock beside the journal: one MLflow publisher per registry,
/// whatever the W&B publisher holds.
fn leader(store: &Store) -> Result<ml_exp_io::directory::FileLock> {
    let parent = store.path().parent().ok_or(Error::Storage)?;
    ml_exp_io::directory::Directory::open(parent, false)
        .and_then(|directory| {
            directory.lock(
                OsStr::new(".mlflow-publisher.lock"),
                ml_exp_io::directory::LockOptions {
                    create: true,
                    ..Default::default()
                },
            )
        })
        .map_err(why!(Error::Storage))
}

/// Scopes of MLflow Runs with records to send or a status to set.
pub(crate) fn candidate_scopes(store: &Store) -> Result<BTreeSet<String>> {
    let mut scopes = BTreeSet::new();
    for (project, run) in store.mlflow_runs().map_err(why!(Error::Storage))? {
        let Ok(guard) = store.guard(&project, &run) else {
            continue;
        };
        for scope in store.run_scopes(&guard).map_err(why!(Error::Storage))? {
            let view = &scope["mlflow"];
            if view["status"] == STOPPED {
                continue;
            }
            if view["finished"] != true || view["pending"].as_i64().unwrap_or(1) > 0 {
                scopes.insert(scope["id"].as_str().ok_or(Error::Identity)?.to_owned());
            }
        }
    }
    Ok(scopes)
}

fn load(store: &Store, guard: &RecordGuard<'_>, scope: &str) -> Result<State> {
    Ok(store
        .mlflow_state(guard, scope)
        .map_err(why!(Error::Storage))?
        .map(serde_json::from_value)
        .transpose()
        .map_err(why!(Error::Identity))?
        .unwrap_or_default())
}
fn save(store: &Store, guard: &RecordGuard<'_>, scope: &str, state: &State) -> Result<()> {
    store
        .save_mlflow_state(
            guard,
            scope,
            &serde_json::to_value(state).map_err(why!(Error::Identity))?,
        )
        .map_err(why!(Error::Storage))
}
fn now() -> Result<String> {
    Ok(crate::projects::transition_context()
        .map_err(why!(Error::Storage))?
        .0)
}
fn now_ms() -> i64 {
    i64::try_from(time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000)
        .unwrap_or(i64::MAX)
}

/// A failed visit leaves the cursor where it was and says why.
pub(crate) fn record_failure(store: &Store, scope: &str, error: Error) {
    let _ = failure(store, scope, error);
}
fn failure(store: &Store, scope: &str, error: Error) -> Result<()> {
    let parsed = Scope::parse(scope).map_err(why!(Error::Identity))?;
    let guard = store
        .guard(&parsed.project, &parsed.run)
        .map_err(why!(Error::Storage))?;
    let mut state = load(store, &guard, scope)?;
    state.status = RETRY_PENDING.into();
    state.error = Some(error.code().into());
    state.failures = state.failures.saturating_add(1);
    state.last_attempt_at = Some(now()?);
    save(store, &guard, scope, &state)
}

fn publish_scope(
    service: &dyn Source,
    store: &Store,
    runtime: &Runtime,
    session: &mut Session,
    scope: &str,
    cancelled: &AtomicBool,
) -> Result<Next> {
    let parsed = Scope::parse(scope).map_err(why!(Error::Identity))?;
    let _authority = service
        .tracking_journal(&parsed.project, &parsed.run)
        .map_err(why!(Error::Journal))?;
    let ended = parsed.attempt.as_str().and_then(|attempt| {
        service
            .provider_terminal(&parsed.project, &parsed.run, attempt)
            .ok()
            .flatten()
    });
    publish_ended(store, runtime, session, scope, ended.as_deref(), cancelled)
}

/// `ended` is the provider's proven terminal state of the Attempt, as for W&B: a
/// job the provider stopped never writes METRICS_DRAINED, so its MLflow run is
/// set FAILED once every journal record is committed.
pub(crate) fn publish_ended(
    store: &Store,
    runtime: &Runtime,
    session: &mut Session,
    scope: &str,
    ended: Option<&str>,
    cancelled: &AtomicBool,
) -> Result<Next> {
    let parsed = Scope::parse(scope).map_err(why!(Error::Identity))?;
    let attempt = match &parsed.attempt {
        Value::String(attempt) => attempt.clone(),
        other => other.to_string(),
    };
    let guard = store
        .guard(&parsed.project, &parsed.run)
        .map_err(why!(Error::Storage))?;
    let binding = store
        .binding(&guard)
        .map_err(why!(Error::Storage))?
        .ok_or(Error::Identity)?;
    let route = &binding["route"]["mlflow"];
    if route["enabled"] != true {
        return Err(Error::Identity);
    }
    let experiment = route["experiment"].as_str().ok_or(Error::Identity)?;
    let total = store.total(&guard, scope).map_err(why!(Error::Storage))?;
    let mut state = load(store, &guard, scope)?;
    if state.status == STOPPED || state.finished && state.confirmed >= total {
        return Ok(Next::Idle);
    }
    if state.confirmed > total {
        return Err(Error::Identity);
    }
    let terminal = |confirmed: i64| -> Result<Option<&'static str>> {
        if confirmed != total {
            return Ok(None);
        }
        let lifecycle = store
            .latest_lifecycle(&guard, scope)
            .map_err(why!(Error::Storage))?;
        let drained = ml_exp_core::records::ended(&lifecycle);
        Ok((drained || ended.is_some()).then_some(
            if ml_exp_core::records::succeeded(&lifecycle, ended) {
                "FINISHED"
            } else {
                "FAILED"
            },
        ))
    };
    // Nothing to send and nothing to finish: no MLflow request at all.
    if state.run_id.is_some() && state.confirmed == total && terminal(total)?.is_none() {
        if state.status != SYNCED || state.error.is_some() {
            state.status = SYNCED.into();
            state.error = None;
            save(store, &guard, scope, &state)?;
        }
        return Ok(Next::Idle);
    }
    let at = now()?;
    state.last_attempt_at = Some(at.clone());
    let experiment_id = ensure_experiment(runtime, session, experiment, &mut state)?;
    let run_id = match state.run_id.clone() {
        Some(run_id) => run_id,
        None => {
            let tag = format!("{}/{}/{attempt}", parsed.project, parsed.run);
            let info = match runtime
                .client
                .search_run(&experiment_id, "ml_expd.scope", &tag)?
            {
                Some(info) => info,
                None => {
                    let mut tags = vec![
                        ("ml_expd.project".to_owned(), parsed.project.clone()),
                        ("ml_expd.run".to_owned(), parsed.run.clone()),
                        ("ml_expd.attempt".to_owned(), attempt.clone()),
                        ("ml_expd.scope".to_owned(), tag),
                    ];
                    // With code mounting, `source_id` names the environment build;
                    // `code_source_id` names the code the job ran.
                    for key in [
                        "source_id",
                        "code_source_id",
                        "runtime_id",
                        "image_id",
                        "backend",
                    ] {
                        if let Some(value) = binding["config"]["ml_expd"][key].as_str() {
                            tags.push((format!("ml_expd.{key}"), value.to_owned()));
                        }
                    }
                    runtime.client.create_run(
                        &experiment_id,
                        &format!("{}/{attempt}", parsed.run),
                        now_ms(),
                        &tags,
                    )?
                }
            };
            state.run_id = Some(info.run_id.clone());
            state.timestamp_base = Some(info.start_time);
            state.url = Some(format!(
                "{}/#/experiments/{experiment_id}/runs/{}",
                runtime.base, info.run_id
            ));
            state.status = ACK_PENDING.into();
            save(store, &guard, scope, &state)?;
            info.run_id
        }
    };
    let base = state.timestamp_base.ok_or(Error::Identity)?;
    if !state.params_logged {
        let (params, skipped) = mapping::params(&binding["config"]);
        for chunk in params.chunks(mapping::MAX_PARAMS) {
            runtime
                .client
                .log_batch(&run_id, &[], chunk)
                .map_err(|error| delivery_error(runtime, &run_id, error))
                .or_else(|error| stop_if_missing(store, &guard, scope, &mut state, error))?;
            if state.status == STOPPED {
                return Ok(Next::Idle);
            }
        }
        state.params_logged = true;
        state.skipped_params = skipped;
        save(store, &guard, scope, &state)?;
    }
    for _ in 0..CHUNKS_PER_VISIT {
        if state.confirmed >= total {
            break;
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Unreachable);
        }
        let events = store
            .events_bounded(
                &guard,
                scope,
                state.confirmed,
                EVENTS_PER_BATCH,
                WINDOW_BYTES,
            )
            .map_err(why!(Error::Storage))?;
        if events.is_empty() {
            return Err(Error::Identity);
        }
        let batch = mapping::next_batch(&events, state.confirmed, base);
        for chunk in &batch.chunks {
            if let Err(error) = runtime
                .client
                .log_batch(&run_id, chunk, &[])
                .map_err(|error| delivery_error(runtime, &run_id, error))
            {
                stop_if_missing(store, &guard, scope, &mut state, error)?;
                return Ok(Next::Idle);
            }
        }
        state.confirmed = batch.through;
        state.skipped_points = state.skipped_points.saturating_add(batch.skipped.count);
        for name in batch.skipped.names {
            if state.skipped_names.len() < mapping::SHOWN_SKIPPED {
                state.skipped_names.insert(name);
            }
        }
        state.last_confirmed_at = Some(at.clone());
        state.status = if state.confirmed < total {
            ACK_PENDING
        } else {
            SYNCED
        }
        .into();
        state.error = None;
        state.failures = 0;
        save(store, &guard, scope, &state)?;
    }
    if state.confirmed < total {
        return Ok(Next::Backlog);
    }
    if let Some(status) = terminal(state.confirmed)? {
        runtime
            .client
            .update_run(&run_id, status, now_ms())
            .map_err(|error| delivery_error(runtime, &run_id, error))
            .or_else(|error| stop_if_missing(store, &guard, scope, &mut state, error))?;
        if state.status == STOPPED {
            return Ok(Next::Idle);
        }
        state.finished = true;
        state.terminal_status = Some(status.into());
        state.status = FINISHED.into();
    } else {
        state.status = SYNCED.into();
    }
    state.error = None;
    state.failures = 0;
    save(store, &guard, scope, &state)?;
    Ok(Next::Idle)
}

fn ensure_experiment(
    runtime: &Runtime,
    session: &mut Session,
    name: &str,
    state: &mut State,
) -> Result<String> {
    if let Some(id) = &state.experiment_id {
        return Ok(id.clone());
    }
    let id = match session.experiments.get(name) {
        Some(id) => id.clone(),
        None => {
            let found = match runtime.client.experiment_by_name(name)? {
                Some(found) => found,
                None => match runtime.client.create_experiment(name) {
                    Ok(id) => (id, "active".into()),
                    // Created meanwhile by someone else: use theirs.
                    Err(Error::AlreadyExists) => runtime
                        .client
                        .experiment_by_name(name)?
                        .ok_or(Error::ResponseInvalid)?,
                    Err(error) => return Err(error),
                },
            };
            if found.1 == "deleted" {
                return Err(Error::ExperimentDeleted);
            }
            session.experiments.insert(name.into(), found.0.clone());
            found.0
        }
    };
    state.experiment_id = Some(id.clone());
    Ok(id)
}

/// A refused write to a run MLflow no longer has, or has deleted, means a person
/// removed it. It is never recreated: a partial copy under the same name would
/// mislead, so the scope stops and says why.
fn delivery_error(runtime: &Runtime, run: &str, error: Error) -> Error {
    if !matches!(error, Error::NotFound | Error::Rejected) {
        return error;
    }
    match runtime.client.get_run(run) {
        Ok(None) => Error::RunMissing,
        Ok(Some(found)) if found["info"]["lifecycle_stage"] == "deleted" => Error::RunMissing,
        Ok(Some(_)) => error,
        Err(other) => other,
    }
}
fn stop_if_missing(
    store: &Store,
    guard: &RecordGuard<'_>,
    scope: &str,
    state: &mut State,
    error: Error,
) -> Result<()> {
    if error != Error::RunMissing {
        return Err(error);
    }
    state.status = STOPPED.into();
    state.error = Some(error.code().into());
    save(store, guard, scope, state)
}
