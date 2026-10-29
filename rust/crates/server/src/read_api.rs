//! The read-only application shell. Trusted configuration chooses locations;
//! requests choose identities and views. No index, bootstrap or provider call.
use crate::{
    deletion::RunFence,
    experiments::{self, RunRef, Snapshot, Store},
    read_config,
    read_model::{self, LiveMetrics, LogQuery, MetricFormat, MetricQuery},
};
use ml_exp_core::identity::{validate_attempt, validate_id, validate_observation_id};
use ml_exp_io::directory::Directory;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    path::PathBuf,
    sync::Arc,
};

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("{code}")]
pub struct Error {
    pub status: u16,
    pub code: &'static str,
    pub detail: &'static str,
}
fn error(status: u16, code: &'static str, detail: &'static str) -> Error {
    Error {
        status,
        code,
        detail,
    }
}
impl From<crate::projects::Error> for Error {
    fn from(value: crate::projects::Error) -> Self {
        use crate::projects::Error as E;
        match value {
            E::Busy => error(
                503,
                "PROJECT_REGISTRY_BUSY",
                "project registry is busy; no operation was dispatched",
            ),
            E::EffectUncertain => error(
                500,
                "PROJECT_OPERATION_UNCERTAIN",
                "project state may have changed; query the lifecycle ledger before retry",
            ),
            E::Binding => error(
                409,
                "PROJECT_SOURCE_BINDING_MISMATCH",
                "registered Project identity differs from approved source configuration",
            ),
            E::UnsupportedDefinition => error(
                422,
                "PROJECT_DEFINITION_UNSUPPORTED",
                "native registration requires empty campaigns; Campaign loading is unavailable",
            ),
            E::Transition(ml_exp_core::projects::Error::UnknownProject) => {
                error(404, "UNKNOWN_PROJECT", "registered Project is unavailable")
            }
            _ => error(
                409,
                "PROJECT_LIFECYCLE_BLOCKED",
                "project lifecycle or its durable evidence cannot be validated",
            ),
        }
    }
}
impl From<experiments::DiscoveryError> for Error {
    fn from(value: experiments::DiscoveryError) -> Self {
        use experiments::DiscoveryError as E;
        match value {
            E::Budget => error(
                413,
                "EXPERIMENT_DISCOVERY_LIMIT",
                "canonical Run discovery exceeds its traversal or metadata budget",
            ),
            E::UnknownProject => error(
                404,
                "EXPERIMENT_NOT_FOUND",
                "the Project is not configured for discovery",
            ),
            E::DuplicateRun | E::OverlappingRoots => error(
                409,
                "EXPERIMENT_DISCOVERY_AMBIGUOUS",
                "approved roots do not establish unique canonical Runs",
            ),
            _ => error(
                409,
                "EXPERIMENT_DISCOVERY_UNAVAILABLE",
                "canonical Run discovery encountered changed, unsafe or unavailable evidence",
            ),
        }
    }
}
impl From<experiments::Error> for Error {
    fn from(value: experiments::Error) -> Self {
        match value {
            experiments::Error::UnknownProject | experiments::Error::Missing => error(
                404,
                "EXPERIMENT_NOT_FOUND",
                "the configured experiment or manifest is unavailable",
            ),
            experiments::Error::SnapshotLimit => error(
                413,
                "EXPERIMENT_READ_LIMIT",
                "experiment metadata exceeds the read budget",
            ),
            experiments::Error::Binding => error(
                422,
                "INVALID_EXPERIMENT_IDENTITY",
                "invalid configured experiment identity",
            ),
            _ => error(
                409,
                "EXPERIMENT_EVIDENCE_UNAVAILABLE",
                "experiment evidence changed, was deleted or cannot be validated",
            ),
        }
    }
}
impl From<read_model::Error> for Error {
    fn from(value: read_model::Error) -> Self {
        match value {
            read_model::Error::Query(_) => {
                error(422, "INVALID_READ_QUERY", "invalid experiment read query")
            }
            read_model::Error::UnknownAttempt => {
                error(404, "ATTEMPT_NOT_FOUND", "the exact Attempt is unavailable")
            }
            read_model::Error::OutputLimit => error(
                413,
                "READ_RESPONSE_LIMIT",
                "experiment read exceeds the response limit",
            ),
            _ => error(
                409,
                "EXPERIMENT_EVIDENCE_UNAVAILABLE",
                "experiment evidence cannot be projected safely",
            ),
        }
    }
}

pub enum View {
    Run,
    Attempts,
    Show { attempt: String },
    Logs { attempt: String, query: LogQuery },
    Metrics { attempt: String, query: MetricQuery },
    Collection { attempt: String },
}
pub struct Query {
    pub project: String,
    pub run: String,
    pub view: View,
}

/// Strict query decoding preserves valid Unicode without lossy replacement.
/// All invalid values return static errors rather than echoing request content.
pub(crate) fn parameters(raw: Option<&str>) -> Result<Vec<(String, String)>, Error> {
    let raw = raw.unwrap_or("");
    if raw.len() > 64 * 1024 {
        return Err(error(
            414,
            "READ_QUERY_LIMIT",
            "experiment query is too large",
        ));
    }
    for component in raw.split(['&', '=']) {
        let bytes = component.as_bytes();
        let mut position = 0;
        while position < bytes.len() {
            if bytes[position] == b'%' {
                if position + 2 >= bytes.len()
                    || !bytes[position + 1].is_ascii_hexdigit()
                    || !bytes[position + 2].is_ascii_hexdigit()
                {
                    return Err(invalid_query());
                }
                position += 3;
            } else {
                position += 1;
            }
        }
        percent_encoding::percent_decode_str(component)
            .decode_utf8()
            .map_err(|_| invalid_query())?;
    }
    Ok(url::form_urlencoded::parse(raw.as_bytes())
        .into_owned()
        .collect())
}
fn invalid_query() -> Error {
    error(422, "INVALID_READ_QUERY", "invalid experiment read query")
}
pub fn no_parameters(raw: Option<&str>) -> Result<(), Error> {
    if parameters(raw)?.is_empty() {
        Ok(())
    } else {
        Err(invalid_query())
    }
}
pub fn metric_query(raw: Option<&str>) -> Result<MetricQuery, Error> {
    let mut query = MetricQuery::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut names = Vec::new();
    for (key, value) in parameters(raw)? {
        if key != "name" && !seen.insert(key.clone()) {
            return Err(invalid_query());
        }
        match key.as_str() {
            "format" => {
                query.format = match value.as_str() {
                    "records" => MetricFormat::Records,
                    "legacy" => MetricFormat::Legacy,
                    _ => return Err(invalid_query()),
                }
            }
            "max_points" => query.max_points = value.parse().map_err(|_| invalid_query())?,
            "keys" => query.keys = Some(value),
            "name" => names.push(value),
            _ => return Err(invalid_query()),
        }
    }
    query.names = (!names.is_empty()).then_some(names);
    query.validate()?;
    Ok(query)
}
pub fn log_query(raw: Option<&str>) -> Result<LogQuery, Error> {
    let mut query = LogQuery::default();
    let mut seen = std::collections::BTreeSet::new();
    for (key, value) in parameters(raw)? {
        if !seen.insert(key.clone()) {
            return Err(invalid_query());
        }
        match key.as_str() {
            "stream" => {
                query.stream = match value.as_str() {
                    "both" => read_model::LogStream::Both,
                    "stdout" => read_model::LogStream::Stdout,
                    "stderr" => read_model::LogStream::Stderr,
                    _ => return Err(invalid_query()),
                }
            }
            "lines" => query.lines = value.parse().map_err(|_| invalid_query())?,
            _ => return Err(invalid_query()),
        }
    }
    query.validate()?;
    Ok(query)
}
pub fn exact_identity(identity: &str) -> Result<(String, String), Error> {
    let (run, attempt) = identity.rsplit_once("::").ok_or_else(invalid_query)?;
    validate_observation_id(run).map_err(|_| invalid_query())?;
    validate_attempt(attempt).map_err(|_| invalid_query())?;
    Ok((run.into(), attempt.into()))
}

pub struct Api {
    store: Store,
    registry_root: PathBuf,
    registry: Directory,
    fence: RunFence,
    workspace_id: String,
    read_scope_sha256: String,
    runs: BTreeMap<(String, String), RunRef>,
    projects: Vec<String>,
    discovery_projects: BTreeSet<String>,
    project_files: BTreeMap<String, PathBuf>,
    lifecycle: Option<Arc<crate::projects::Service>>,
    sources: Option<Arc<crate::sources::Service>>,
    actions: Option<crate::operations::Store>,
    submissions: Option<Arc<crate::submissions::Service>>,
    cloud: Option<Arc<crate::cloud_runs::Service>>,
    administration: Option<Arc<crate::administration::Service>>,
}
impl Api {
    pub fn open(settings: read_config::Settings) -> Result<Self, read_config::Error> {
        settings.validate()?;
        let read_scope_sha256 = settings.identity()?;
        let (registry, workspace_id) = read_config::workspace(&settings.registry_root)?;
        let fence = RunFence::new(&settings.registry_root);
        let store =
            Store::open(settings.bindings(), fence.clone()).map_err(|_| read_config::Error)?;
        let runs = settings
            .runs()
            .into_iter()
            .map(|r| ((r.project.clone(), r.run_id.clone()), r))
            .collect();
        let projects = settings
            .projects
            .iter()
            .map(|p| p.project.clone())
            .collect();
        let discovery_projects = settings
            .projects
            .iter()
            .filter(|project| project.discovery_enabled())
            .map(|project| project.project.clone())
            .collect();
        let project_files = settings
            .projects
            .iter()
            .map(|project| (project.project.clone(), project.project_file.clone()))
            .collect();
        // This is the fixed isolated namespace, never a request-selected path.
        // Missing Action storage stays absent; startup and GET do not create it.
        let actions = if registry
            .entry_exists(OsStr::new("actions"))
            .map_err(|_| read_config::Error)?
        {
            Some(
                crate::operations::Store::open_existing(settings.registry_root.join("actions"))
                    .map_err(|_| read_config::Error)?,
            )
        } else {
            None
        };
        Ok(Self {
            store,
            registry_root: settings.registry_root,
            registry,
            fence,
            workspace_id,
            read_scope_sha256,
            runs,
            projects,
            discovery_projects,
            project_files,
            lifecycle: None,
            sources: None,
            actions,
            submissions: None,
            cloud: None,
            administration: None,
        })
    }
    pub fn with_lifecycle(
        mut self,
        service: Arc<crate::projects::Service>,
    ) -> Result<Self, read_config::Error> {
        if !service.matches_configuration(
            &self.registry_root,
            &self.project_files,
            &self.workspace_id,
        ) {
            return Err(read_config::Error);
        }
        self.lifecycle = Some(service);
        Ok(self)
    }
    pub fn lifecycle(&self) -> Option<Arc<crate::projects::Service>> {
        self.lifecycle.clone()
    }
    pub fn with_sources(
        mut self,
        service: Arc<crate::sources::Service>,
    ) -> Result<Self, read_config::Error> {
        if !service.matches_configuration(
            &self.registry_root,
            &self.project_files,
            &self.workspace_id,
            self.lifecycle.as_ref(),
        ) {
            return Err(read_config::Error);
        }
        self.sources = Some(service);
        Ok(self)
    }
    pub fn sources(&self) -> Option<Arc<crate::sources::Service>> {
        self.sources.clone()
    }
    pub fn with_submissions(
        mut self,
        service: Arc<crate::submissions::Service>,
    ) -> Result<Self, read_config::Error> {
        if !service.matches_configuration(
            &self.registry_root,
            &self.project_files,
            &self.workspace_id,
            self.sources.as_ref(),
            self.lifecycle.as_ref(),
            &self.read_scope_sha256,
        ) {
            return Err(read_config::Error);
        }
        self.submissions = Some(service);
        Ok(self)
    }
    pub fn submissions(&self) -> Option<Arc<crate::submissions::Service>> {
        self.submissions.clone()
    }
    pub fn with_cloud(
        mut self,
        service: Arc<crate::cloud_runs::Service>,
    ) -> Result<Self, read_config::Error> {
        if !service.matches(
            &self.registry_root,
            &self.read_scope_sha256,
            self.sources.as_ref(),
        ) {
            return Err(read_config::Error);
        }
        self.cloud = Some(service);
        Ok(self)
    }
    pub fn cloud(&self) -> Option<Arc<crate::cloud_runs::Service>> {
        self.cloud.clone()
    }
    pub fn active_project_count(&self) -> Result<usize, Error> {
        self.validate_registry()?;
        match &self.lifecycle {
            Some(service) => Ok(service.active_projects()?.len()),
            None => Ok(self.projects.len()),
        }
    }
    fn ensure_active(&self, project: &str) -> Result<(), Error> {
        if let Some(service) = &self.lifecycle {
            let active = service.active_projects()?;
            if !active.contains(project) {
                return Err(error(
                    404,
                    "PROJECT_INACTIVE",
                    "Project is not actively tracked in this workspace",
                ));
            }
        }
        Ok(())
    }
    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }
    pub fn project_count(&self) -> usize {
        self.projects.len()
    }
    pub fn submission(&self, id: &str, now_unix: f64) -> Result<Value, Error> {
        if ml_exp_core::operations::validate_submission_id(id).is_err() {
            return Err(invalid_query());
        }
        self.validate_registry()?;
        let actions = self.actions.as_ref().ok_or_else(|| {
            error(
                501,
                "SUBMISSION_STORAGE_NOT_CONFIGURED",
                "existing native Action storage is not configured",
            )
        })?;
        let snapshot = actions.observe(id).map_err(|cause| match cause {
            crate::operations::Error::NotFound => {
                error(404, "UNKNOWN_SUBMISSION", "exact Submission is unavailable")
            }
            _ => error(
                409,
                "SUBMISSION_EVIDENCE_UNAVAILABLE",
                "committed Submission evidence cannot be validated; no recovery was attempted",
            ),
        })?;
        let invalid_clock = || error(503, "CLOCK_UNAVAILABLE", "observation clock is unavailable");
        if !now_unix.is_finite() {
            return Err(invalid_clock());
        }
        let now = time::OffsetDateTime::from_unix_timestamp_nanos((now_unix * 1e9) as i128)
            .map_err(|_| invalid_clock())?
            .format(&time::format_description::well_known::Rfc3339)
            .map_err(|_| invalid_clock())?;
        if snapshot["native_cloud"]["format"] == "native-cloud-action.v1" {
            return self
                .cloud
                .as_ref()
                .ok_or_else(|| {
                    error(
                        501,
                        "CLOUD_NOT_CONFIGURED",
                        "cloud submission service is not configured",
                    )
                })?
                .get(id, &now)
                .map_err(crate::cloud_runs::failure);
        }
        if let Some(service) = &self.submissions {
            return service
                .get(id, &now)
                .map_err(crate::submissions::http::failure);
        }
        let value =
            ml_exp_core::operations::submission_view(&snapshot, &now).map_err(
                |cause| match cause {
                    ml_exp_core::operations::Error::Invalid("Action is not a Submission") => error(
                        404,
                        "UNKNOWN_SUBMISSION",
                        "Action is not an experiment Submission",
                    ),
                    _ => error(
                        409,
                        "SUBMISSION_EVIDENCE_UNAVAILABLE",
                        "Action is not a valid bound Submission",
                    ),
                },
            )?;
        let project = value["project"].as_str().ok_or_else(invalid_query)?;
        let run = value["run_id"].as_str().ok_or_else(invalid_query)?;
        if !self.project_files.contains_key(project) {
            return Err(error(
                404,
                "UNKNOWN_SUBMISSION",
                "Submission Project is not approved",
            ));
        }
        let guard = self
            .fence
            .existing_observation_guard(project, run)
            .map_err(|_| {
                error(
                    409,
                    "SUBMISSION_EVIDENCE_UNAVAILABLE",
                    "Submission Run is deleted or unavailable",
                )
            })?;
        // Validate the configured Run without resolving or borrowing its current
        // Attempt. A first planned Attempt may not yet have a manifest on disk.
        self.ensure_active(project)?;
        let (reference, discovery) = self.locate_run(project, run)?;
        let metadata = self.store.snapshot_metadata(&reference)?;
        if discovery.as_ref().is_some_and(|found| {
            found
                .runs
                .iter()
                .find(|entry| entry.reference.run_id == run)
                .is_none_or(|entry| entry.raw_sha256 != metadata.run.raw_sha256)
        }) {
            return Err(error(
                409,
                "EXPERIMENT_DISCOVERY_CHANGED",
                "Run binding changed during Submission observation",
            ));
        }
        if let Some(before) = discovery {
            let after = self
                .store
                .discover(project, &experiments::DiscoveryLimits::default())?;
            if !before.same_bindings(&after) {
                return Err(error(
                    409,
                    "EXPERIMENT_DISCOVERY_CHANGED",
                    "Run bindings changed during Submission observation",
                ));
            }
        }
        guard.require_observed_live(project, run).map_err(|_| {
            error(
                409,
                "SUBMISSION_EVIDENCE_UNAVAILABLE",
                "Submission Run changed during observation",
            )
        })?;
        self.ensure_active(project)?;
        self.validate_registry()?;
        Ok(experiment_redact::redact_json(
            ml_exp_core::observation::sanitized_outward(&value),
        ))
    }
    fn locate_run(
        &self,
        project: &str,
        run: &str,
    ) -> Result<(RunRef, Option<experiments::Discovery>), Error> {
        let discovery = if self.discovery_projects.contains(project) {
            Some(
                self.store
                    .discover(project, &experiments::DiscoveryLimits::default())?,
            )
        } else {
            None
        };
        let reference = match &discovery {
            Some(found) => found
                .runs
                .iter()
                .find(|entry| entry.reference.run_id == run)
                .map(|entry| &entry.reference),
            None => self.runs.get(&(project.into(), run.into())),
        }
        .cloned()
        .ok_or_else(|| {
            error(
                404,
                "EXPERIMENT_NOT_FOUND",
                "the exact Run is unavailable in approved storage",
            )
        })?;
        Ok((reference, discovery))
    }
    fn validate_registry(&self) -> Result<(), Error> {
        self.registry.verify_private_parent().map_err(|_| {
            error(
                409,
                "READ_WORKSPACE_UNAVAILABLE",
                "isolated read workspace is unavailable",
            )
        })?;
        let (_, id) = read_config::workspace(&self.registry_root).map_err(|_| {
            error(
                409,
                "READ_WORKSPACE_UNAVAILABLE",
                "isolated read workspace is unavailable",
            )
        })?;
        if id != self.workspace_id {
            return Err(error(
                409,
                "READ_WORKSPACE_CHANGED",
                "isolated workspace identity changed",
            ));
        }
        Ok(())
    }
    pub fn projects(&self, now_unix: f64) -> Result<Value, Error> {
        self.validate_registry()?;
        let visible = match &self.lifecycle {
            Some(service) => Some(service.active_projects()?),
            None => None,
        };
        let mut values = Vec::new();
        let mut value_bytes = 2;
        for project in &self.projects {
            if visible
                .as_ref()
                .is_some_and(|visible| !visible.contains(project))
            {
                continue;
            }
            let document = self.store.project(project, None)?;
            let title = document
                .metadata
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or(project);
            let discovery = if self.discovery_projects.contains(project) {
                Some(
                    self.store
                        .discover(project, &experiments::DiscoveryLimits::default())?,
                )
            } else {
                None
            };
            let references: Vec<_> = match &discovery {
                Some(found) => found.runs.iter().map(|run| &run.reference).collect(),
                None => self
                    .runs
                    .values()
                    .filter(|run| &run.project == project)
                    .collect(),
            };
            let mut rows = Vec::new();
            let mut row_bytes = 2;
            for reference in references {
                let snapshot = self.store.snapshot_metadata(reference)?;
                if discovery.as_ref().is_some_and(|found| {
                    found
                        .runs
                        .iter()
                        .find(|run| run.reference.run_id == reference.run_id)
                        .is_none_or(|run| run.raw_sha256 != snapshot.run.raw_sha256)
                }) {
                    return Err(error(
                        409,
                        "EXPERIMENT_DISCOVERY_CHANGED",
                        "Run bindings changed during project aggregation",
                    ));
                }
                let row = read_model::run_summary(&snapshot, now_unix)?;
                row_bytes = read_model::array_size(row_bytes, &row)?;
                rows.push(row);
            }
            let value = read_model::project_summary(project, title, &rows)?;
            value_bytes = read_model::array_size(value_bytes, &value)?;
            values.push(value);
            self.store.project(project, Some(&document.raw_sha256))?;
            if let Some(before) = discovery {
                let after = self
                    .store
                    .discover(project, &experiments::DiscoveryLimits::default())?;
                if !before.same_bindings(&after) {
                    return Err(error(
                        409,
                        "EXPERIMENT_DISCOVERY_CHANGED",
                        "canonical Run bindings changed during project aggregation",
                    ));
                }
            }
        }
        self.validate_registry()?;
        if let Some(service) = &self.lifecycle {
            if Some(service.active_projects()?) != visible {
                return Err(error(
                    409,
                    "PROJECT_LIFECYCLE_CHANGED",
                    "active Project tracking changed during observation",
                ));
            }
        }
        Ok(Value::Array(values))
    }
    fn live_metrics(
        &self,
        snapshot: &Snapshot,
        attempt: Option<&str>,
    ) -> Result<Vec<Value>, Error> {
        let project = &snapshot.project;
        let run = &snapshot.run_id;
        let Some(attempt) = attempt else {
            return Ok(Vec::new());
        };
        let tracking = match self.registry.child(OsStr::new("tracking"), false) {
            Ok(directory) => directory,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(_) => {
                return Err(error(
                    409,
                    "JOURNAL_UNAVAILABLE",
                    "canonical experiment journal is unavailable",
                ));
            }
        };
        if !tracking
            .entry_exists(OsStr::new("outbox.sqlite"))
            .map_err(|_| {
                error(
                    409,
                    "JOURNAL_UNAVAILABLE",
                    "canonical experiment journal is unavailable",
                )
            })?
        {
            return Ok(Vec::new());
        }
        let records = crate::records::Store::read_only(&self.registry_root);
        let guard = records.guard(project, run).map_err(|_| {
            error(
                409,
                "JOURNAL_UNAVAILABLE",
                "canonical experiment journal is unavailable",
            )
        })?;
        if records.workspace_id(&guard).map_err(|_| {
            error(
                409,
                "JOURNAL_IDENTITY_UNAVAILABLE",
                "canonical journal identity is unavailable",
            )
        })? != self
            .workspace_id
            .strip_prefix("rust-preview-")
            .ok_or_else(|| {
                error(
                    409,
                    "READ_WORKSPACE_CHANGED",
                    "isolated workspace identity changed",
                )
            })?
        {
            return Err(error(
                409,
                "JOURNAL_IDENTITY_MISMATCH",
                "canonical journal belongs to a different workspace",
            ));
        }
        let binding = records.binding(&guard).map_err(|_| {
            error(
                409,
                "JOURNAL_IDENTITY_UNAVAILABLE",
                "canonical journal binding is unavailable",
            )
        })?;
        if binding.is_none() {
            return records
                .metric_records_bounded(&guard, attempt, 0, 0)
                .map_err(|_| {
                    error(
                        409,
                        "JOURNAL_IDENTITY_UNAVAILABLE",
                        "unbound canonical journal contains unresolved records",
                    )
                });
        }
        if !journal_binding_matches(snapshot, binding.as_ref()) {
            return Err(error(
                409,
                "JOURNAL_IDENTITY_MISMATCH",
                "canonical journal does not match the frozen Run",
            ));
        }
        records
            .metric_records_bounded(&guard, attempt, 8 * 1024 * 1024, 100_000)
            .map_err(|e| match e {
                crate::records::Error::Invalid("metric history exceeds read budget") => error(
                    413,
                    "METRIC_HISTORY_LIMIT",
                    "complete metric history exceeds the read budget",
                ),
                _ => error(
                    409,
                    "JOURNAL_UNAVAILABLE",
                    "canonical experiment journal is unavailable",
                ),
            })
    }
    fn native_view(
        &self,
        value: Value,
        project: &str,
        run: &str,
        attempt: Option<&str>,
        now_unix: f64,
    ) -> Result<Value, Error> {
        if let Some(cloud) = &self.cloud {
            if cloud
                .handles_run(project, run)
                .map_err(crate::cloud_runs::failure)?
            {
                let now = time::OffsetDateTime::from_unix_timestamp_nanos((now_unix * 1e9) as i128)
                    .map_err(|_| invalid_query())?
                    .format(&time::format_description::well_known::Rfc3339)
                    .map_err(|_| invalid_query())?;
                return cloud
                    .project_view(value, project, run, attempt, &now)
                    .map_err(crate::cloud_runs::failure);
            }
        }
        let Some(service) = &self.submissions else {
            return Ok(value);
        };
        let now = time::OffsetDateTime::from_unix_timestamp_nanos((now_unix * 1e9) as i128)
            .map_err(|_| invalid_query())?
            .format(&time::format_description::well_known::Rfc3339)
            .map_err(|_| invalid_query())?;
        let submission = match attempt
            .map(|attempt| service.for_attempt(project, run, attempt, &now))
            .unwrap_or_else(|| service.for_run(project, run, &now))
        {
            Ok(value) => value,
            Err(crate::submissions::Error::Missing) => return Ok(value),
            Err(cause) => return Err(crate::submissions::http::failure(cause)),
        };
        let projected = if let Some(attempt) = attempt {
            if submission["attempt_id"] != attempt {
                return Ok(value);
            }
            ml_exp_core::first_submission::native::project_attempt(
                value,
                &submission,
                &ml_exp_core::slurm::helper::Scope {
                    project: project.into(),
                    run_id: run.into(),
                    attempt_id: attempt.into(),
                },
            )
        } else {
            ml_exp_core::first_submission::native::project_run(value, &submission)
        };
        projected.map_err(|_| {
            error(
                409,
                "SUBMISSION_EVIDENCE_UNAVAILABLE",
                "native scheduler receipt cannot be projected onto this exact Attempt",
            )
        })
    }
    pub fn query(&self, query: Query, now_unix: f64) -> Result<Value, Error> {
        validate_id(&query.project).map_err(|_| invalid_query())?;
        validate_observation_id(&query.run).map_err(|_| invalid_query())?;
        if !now_unix.is_finite() {
            return Err(error(
                503,
                "CLOCK_UNAVAILABLE",
                "observation clock is unavailable",
            ));
        }
        self.validate_registry()?;
        self.ensure_active(&query.project)?;
        // Acquire the existing deletion guard before looking for physical
        // manifests, so a removed-but-tombstoned Run cannot appear unsubmitted.
        let guard = self
            .fence
            .existing_observation_guard(&query.project, &query.run)
            .map_err(|_| {
                error(
                    409,
                    "EXPERIMENT_EVIDENCE_UNAVAILABLE",
                    "Run is unavailable or deleted",
                )
            })?;
        let (reference, discovery) = self.locate_run(&query.project, &query.run)?;
        let located = discovery
            .as_ref()
            .and_then(|d| d.runs.iter().find(|r| r.reference.run_id == query.run));
        let mut snapshot: Snapshot = match &query.view {
            View::Run | View::Attempts => self.store.snapshot(&reference)?,
            View::Show { attempt }
            | View::Logs { attempt, .. }
            | View::Metrics { attempt, .. }
            | View::Collection { attempt } => self.store.snapshot_attempt(&reference, attempt)?,
        };
        if located.is_some_and(|run| run.raw_sha256 != snapshot.run.raw_sha256) {
            return Err(error(
                409,
                "EXPERIMENT_DISCOVERY_CHANGED",
                "Run manifest changed between discovery and observation",
            ));
        }
        if let Some(service) = &self.submissions {
            service
                .apply_observation(&mut snapshot)
                .map_err(crate::submissions::http::failure)?;
            service
                .apply_artifacts(&mut snapshot)
                .map_err(crate::submissions::http::failure)?;
        }
        if let Some(cloud) = &self.cloud {
            cloud
                .apply_observation(&mut snapshot)
                .map_err(crate::cloud_runs::failure)?;
        }
        let value = match query.view {
            View::Run => {
                let attempt = snapshot.current_attempt_id.as_deref();
                let records = self.live_metrics(&snapshot, attempt)?;
                let value = read_model::run_detail(
                    &snapshot,
                    now_unix,
                    attempt.map(|attempt_id| LiveMetrics {
                        attempt_id,
                        records: &records,
                    }),
                )?;
                self.native_view(value, &query.project, &query.run, None, now_unix)?
            }
            View::Attempts => read_model::run_attempts(&snapshot)?,
            View::Show { attempt } => self.native_view(
                read_model::attempt_show(&snapshot, &attempt, now_unix)?,
                &query.project,
                &query.run,
                Some(&attempt),
                now_unix,
            )?,
            View::Logs { attempt, query } => read_model::attempt_logs(&snapshot, &attempt, &query)?,
            View::Metrics {
                attempt,
                query: metric_query,
            } => {
                let records = self.live_metrics(&snapshot, Some(&attempt))?;
                read_model::attempt_metrics(
                    &snapshot,
                    &attempt,
                    &metric_query,
                    Some(LiveMetrics {
                        attempt_id: &attempt,
                        records: &records,
                    }),
                )?
            }
            View::Collection { attempt } => self.native_view(
                read_model::attempt_collection(&snapshot, &attempt, now_unix)?,
                &query.project,
                &query.run,
                Some(&attempt),
                now_unix,
            )?,
        };
        guard
            .require_observed_live(&query.project, &query.run)
            .map_err(|_| {
                error(
                    409,
                    "EXPERIMENT_EVIDENCE_UNAVAILABLE",
                    "Run was deleted or changed during observation",
                )
            })?;
        if let Some(before) = discovery {
            let after = self
                .store
                .discover(&query.project, &experiments::DiscoveryLimits::default())?;
            if !before.same_bindings(&after) {
                return Err(error(
                    409,
                    "EXPERIMENT_DISCOVERY_CHANGED",
                    "canonical Run bindings changed during observation",
                ));
            }
        }
        self.ensure_active(&query.project)?;
        self.validate_registry()?;
        Ok(value)
    }
}

/// Refuse a known authentication-secret echo instead of changing scientific
/// numbers, metric names or units. Called after ordinary outward redaction.
pub fn contains_secret(value: &Value, secret: &str) -> bool {
    if secret.is_empty() {
        return false;
    }
    match value {
        Value::String(v) => v.contains(secret),
        Value::Number(v) => v.to_string().contains(secret),
        Value::Array(v) => v.iter().any(|v| contains_secret(v, secret)),
        Value::Object(v) => v
            .iter()
            .any(|(k, v)| k.contains(secret) || contains_secret(v, secret)),
        _ => false,
    }
}

/// Match the baseline publication-config subset against immutable Run inputs.
/// A matching project/run string alone cannot authorize a copied journal.
pub fn journal_binding_matches(snapshot: &Snapshot, binding: Option<&Value>) -> bool {
    let Some(ml) = binding
        .and_then(|v| v.get("config"))
        .and_then(|v| v.get("ml_expd"))
        .and_then(Value::as_object)
    else {
        return false;
    };
    if ml.get("project").and_then(Value::as_str) != Some(&snapshot.project)
        || ml.get("run_id").and_then(Value::as_str) != Some(&snapshot.run_id)
    {
        return false;
    }
    if ml
        .get("native_run_sha256")
        .is_some_and(|v| v.as_str() != Some(&snapshot.run.raw_sha256))
    {
        return false;
    }
    let mut definition = snapshot.run.metadata.clone();
    if let Some(resolved) = definition
        .get("resolved_config")
        .and_then(Value::as_object)
        .cloned()
    {
        definition.extend(resolved);
    }
    let same = |left: &Value, right: Option<&Value>| {
        right.is_some_and(|right| {
            match (
                ml_exp_core::canonical::digest(left),
                ml_exp_core::canonical::digest(right),
            ) {
                (Ok(a), Ok(b)) => a == b,
                _ => false,
            }
        })
    };
    for key in [
        "run_id",
        "source_id",
        "image_id",
        "resources",
        "evaluation",
        "parameters",
        "resume_from",
    ] {
        if let Some(value) = definition.get(key) {
            if !same(value, ml.get(key)) {
                return false;
            }
        }
    }
    let runtime = definition
        .get("container")
        .and_then(|v| v.get("runtime_id"))
        .unwrap_or(&Value::Null);
    let backend = definition
        .get("backend")
        .and_then(|v| v.get("kind"))
        .unwrap_or(&Value::Null);
    let inputs = definition
        .get("inputs")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|v| {
                    let fields = ["asset_id", "sha256", "mount_path"]
                        .into_iter()
                        .filter_map(|key| v.get(key).map(|value| (key.to_string(), value.clone())))
                        .collect::<serde_json::Map<String, Value>>();
                    Value::Object(fields)
                })
                .collect()
        })
        .unwrap_or_default();
    same(runtime, ml.get("runtime_id"))
        && same(backend, ml.get("backend"))
        && same(&Value::Array(inputs), ml.get("inputs"))
}

impl Api {
    pub fn with_administration(
        mut self,
        service: Arc<crate::administration::Service>,
    ) -> Result<Self, read_config::Error> {
        if !service.matches(&self.registry_root) {
            return Err(read_config::Error);
        }
        self.administration = Some(service);
        Ok(self)
    }
    pub fn administration(&self) -> Option<Arc<crate::administration::Service>> {
        self.administration.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn query_decoding_preserves_unicode_and_rejects_ambiguous_or_lossy_inputs() {
        let q = metric_query(Some(
            "format=records&name=%E8%AE%AD%E7%BB%83%2Floss&name=token&max_points=2",
        ))
        .unwrap();
        assert_eq!(q.names.unwrap(), vec!["训练/loss", "token"]);
        for q in [
            "name=%FF",
            "name=%GG",
            "name=%",
            "max_points=1&max_points=2",
            "format=records&keys=loss",
            "format=legacy&name=loss",
            "unknown=x",
            "max_points=0",
        ] {
            assert!(metric_query(Some(q)).is_err(), "{q}");
        }
        for q in [
            "lines=0",
            "lines=-1",
            "lines=10001",
            "stream=foreign",
            "stream=both&stream=stderr",
            "secret=value",
        ] {
            assert!(log_query(Some(q)).is_err(), "{q}");
        }
        assert!(no_parameters(None).is_ok());
        assert!(no_parameters(Some("x=1")).is_err());
        assert_eq!(
            exact_identity("r::attempt-001").unwrap(),
            ("r".into(), "attempt-001".into())
        );
        for s in ["r", "r::latest", "r::attempt-001::x", "../r::attempt-001"] {
            assert!(exact_identity(s).is_err());
        }
    }
    #[test]
    fn known_secret_echo_detection_never_changes_scientific_content() {
        let secret = "1234567890123456789012345678901234567890";
        let numeric: Value = serde_json::from_str(secret).unwrap();
        for value in [
            json!({"note":format!("prefix-{secret}-suffix")}),
            json!({secret:2.5}),
            json!({"value":numeric}),
        ] {
            let original = value.clone();
            assert!(contains_secret(&value, secret));
            assert_eq!(value, original);
        }
        assert!(!contains_secret(
            &json!({"token":2.5,"password":3.5,"unit":"nats/token"}),
            secret
        ));
    }
}
