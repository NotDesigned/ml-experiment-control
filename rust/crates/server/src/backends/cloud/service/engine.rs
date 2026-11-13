use super::{Definition, Settings};
use crate::{
    backends::cloud::{
        self, Client, Credentials,
        action::Runner,
        adapter::{Adapter, Shared},
        lock_diagnostics,
        outbox::Outbox,
    },
    deletion::RunFence,
    experiments,
    operations::{self, ClaimContext, Execute},
    projects, read_config, sources,
};
use ml_exp_core::{
    canonical,
    cloud::{
        action::{self, Execution},
        run,
    },
    first_submission::native::{self, Approval, Binding, creation},
    operations::FrozenFiles,
    slurm::helper,
};
use ml_exp_io::directory::{Directory, FileLock, LockOptions, Wait};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    path::Path,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};
const NAME: &str = "cloud-submissions";
const LOCK: &str = ".cloud-runs.lock";
struct Backend {
    definition: Definition,
    adapter: Adapter<Shared>,
    collector: Option<Adapter<Shared>>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Prepared {
    pub(super) execution: Execution,
    pub(super) plan: Value,
    pub(super) approval: Approval,
}
pub struct Service {
    settings: read_config::Settings,
    workspace: String,
    pub(super) directory: Directory,
    pub(super) slots: Directory,
    pub(super) actions: operations::Store,
    pub(super) outbox: Outbox,
    sync_outbox: Outbox,
    fence: RunFence,
    sources: Arc<sources::Service>,
    lifecycle: Option<Arc<projects::Service>>,
    backends: BTreeMap<String, Backend>,
    enabled: bool,
    creation: bool,
    tracking: ml_exp_core::tracking::RoutingSettings,
    exchange: Option<exchange::Authority>,
    objects: Option<crate::objects::Store>,
    builds: Option<builds::Settings>,
    active_builds: Arc<std::sync::Mutex<std::collections::BTreeSet<String>>>,
    debug: Option<debug::Runtime>,
    lock_holder: lock_diagnostics::Holder,
    pub(super) polling: ml_exp_core::polling::Policy,
    pub(super) archive: super::config::CheckpointArchive,
}
pub(super) fn op(error: operations::Error) -> cloud::Error {
    if matches!(error, operations::Error::NotFound) {
        return missing();
    }
    cloud::Error {
        code: "CLOUD_ACTION_UNAVAILABLE".into(),
        http_status: None,
        uncertain: false,
    }
}
pub(super) fn io(_: std::io::Error) -> cloud::Error {
    cloud::Error::evidence()
}
fn encode(value: &impl Serialize) -> cloud::Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(why!(cloud::Error::evidence()))
}
pub(super) fn read(dir: &Directory, name: &str) -> cloud::Result<Value> {
    let bytes = dir.read_whole(OsStr::new(name)).map_err(io)?;
    let value = Value::Object(
        ml_exp_core::json::own_mapping(&bytes).map_err(why!(cloud::Error::evidence()))?,
    );
    if encode(&value)? != bytes {
        return Err(cloud::Error::evidence());
    }
    Ok(value)
}
fn write(dir: &Directory, name: &str, value: &impl Serialize) -> cloud::Result<()> {
    dir.publish_new(OsStr::new(name), &encode(value)?)
        .map_err(io)
}
fn missing() -> cloud::Error {
    cloud::Error {
        code: "CLOUD_SUBMISSION_NOT_FOUND".into(),
        http_status: Some(404),
        uncertain: false,
    }
}
/// One signed REST client for one adapter of this backend.
fn client_transport(definition: &Definition) -> cloud::Result<Shared> {
    let credentials = Credentials::load(&definition.credentials_file)?;
    if !credentials.matches_profile(&definition.profile) {
        return Err(cloud::Error::configuration());
    }
    let mut client = Client::new(
        credentials,
        Duration::from_secs(definition.timeout_seconds.into()),
    )?;
    if let Some(path) = &definition.registry_auth_file {
        client = client.with_registry_auth(path)?;
    }
    Ok(Arc::new(client))
}
/// A short hold answers its neighbours with a short delay rather than a failure.
const LOCK_WAIT: Duration = Duration::from_secs(5);

impl Service {
    pub fn initialize(settings: &read_config::Settings) -> cloud::Result<()> {
        settings
            .validate()
            .map_err(why!(cloud::Error::configuration()))?;
        let (root, _) = read_config::workspace(&settings.registry_root)
            .map_err(why!(cloud::Error::configuration()))?;
        let directory = root.create_new_child(OsStr::new(NAME)).map_err(io)?;
        directory
            .create_new_child(OsStr::new("slots"))
            .map_err(io)?;
        directory
            .create_new_regular(OsStr::new(LOCK))
            .map_err(io)?
            .sync_all()
            .map_err(io)?;
        directory
            .create_new_child(OsStr::new("effects"))
            .map_err(io)?;
        Outbox::initialize(&settings.registry_root.join(NAME).join("effects"))?;
        directory
            .create_new_child(OsStr::new("sync-effects"))
            .map_err(io)?;
        Outbox::initialize(&settings.registry_root.join(NAME).join("sync-effects"))?;
        operations::Store::open(settings.registry_root.join("actions")).map_err(op)?;
        crate::checkpoints::Store::open(&settings.registry_root)
            .map_err(why!(cloud::Error::evidence()))?;
        Ok(())
    }
    pub fn open(
        settings: read_config::Settings,
        config: Settings,
        sources: Arc<sources::Service>,
        lifecycle: Option<Arc<projects::Service>>,
        tracking: ml_exp_core::tracking::RoutingSettings,
    ) -> cloud::Result<Self> {
        Self::open_using(
            settings,
            config,
            sources,
            lifecycle,
            tracking,
            &client_transport,
        )
    }
    /// Test seam: every adapter of every backend sends through `transport`, and
    /// no credential file is read. Debug sessions and image builds keep their
    /// own `Client` and do not pass through this seam.
    #[doc(hidden)]
    pub fn open_with_transport(
        settings: read_config::Settings,
        config: Settings,
        sources: Arc<sources::Service>,
        lifecycle: Option<Arc<projects::Service>>,
        tracking: ml_exp_core::tracking::RoutingSettings,
        transport: Shared,
    ) -> cloud::Result<Self> {
        Self::open_using(settings, config, sources, lifecycle, tracking, &|_| {
            Ok(transport.clone())
        })
    }
    fn open_using(
        settings: read_config::Settings,
        config: Settings,
        sources: Arc<sources::Service>,
        lifecycle: Option<Arc<projects::Service>>,
        tracking: ml_exp_core::tracking::RoutingSettings,
        transport: &dyn Fn(&Definition) -> cloud::Result<Shared>,
    ) -> cloud::Result<Self> {
        settings
            .validate()
            .map_err(why!(cloud::Error::configuration()))?;
        let (root, workspace) = read_config::workspace(&settings.registry_root)
            .map_err(why!(cloud::Error::configuration()))?;
        let files = settings
            .projects
            .iter()
            .map(|p| (p.project.clone(), p.project_file.clone()))
            .collect();
        if !sources.matches_configuration(
            &settings.registry_root,
            &files,
            &workspace,
            lifecycle.as_ref(),
        ) {
            return Err(cloud::Error::configuration());
        }
        let directory = root.child(OsStr::new(NAME), false).map_err(io)?;
        let slots = directory.child(OsStr::new("slots"), false).map_err(io)?;
        directory.open_regular_read(OsStr::new(LOCK)).map_err(io)?;
        let mut backends = BTreeMap::new();
        for definition in config.backends {
            definition.validate()?;
            let adapter = Adapter::new(definition.profile.clone(), transport(&definition)?)?;
            let collector = definition
                .collector
                .as_ref()
                .map(|profile| Adapter::new(profile.clone(), transport(&definition)?))
                .transpose()?;
            if backends
                .insert(
                    definition.executor.clone(),
                    Backend {
                        definition,
                        adapter,
                        collector,
                    },
                )
                .is_some()
            {
                return Err(cloud::Error::configuration());
            }
        }
        if backends.is_empty() && config.builds.is_none() && config.debug.is_none() {
            return Err(cloud::Error::configuration());
        }
        let actions =
            operations::Store::open_existing(settings.registry_root.join("actions")).map_err(op)?;
        let outbox = Outbox::open(&settings.registry_root.join(NAME).join("effects"))?;
        let sync_outbox = Outbox::open(&settings.registry_root.join(NAME).join("sync-effects"))?;
        let fence = RunFence::new(&settings.registry_root);
        let exchange = config.exchange.map(exchange::Authority::load).transpose()?;
        let objects = config
            .artifacts
            .map(crate::objects::Store::open)
            .transpose()
            .map_err(io)?;
        if let Some(builds) = &config.builds {
            builds.validate()?;
        }
        Ok(Self {
            settings,
            workspace,
            directory,
            slots,
            actions,
            outbox,
            sync_outbox,
            fence,
            sources,
            lifecycle,
            backends,
            enabled: config.enabled,
            creation: config.allow_run_creation,
            tracking,
            exchange,
            objects,
            builds: config.builds,
            active_builds: Default::default(),
            debug: config.debug.map(debug::Runtime::open).transpose()?,
            lock_holder: lock_diagnostics::Holder::new("cloud-lock"),
            polling: config.polling,
            archive: config.checkpoint_archive,
        })
    }
    pub fn matches(
        &self,
        registry: &Path,
        scope: &str,
        sources: Option<&Arc<sources::Service>>,
    ) -> bool {
        self.settings.registry_root == registry
            && self.settings.identity().is_ok_and(|s| s == scope)
            && sources.is_some_and(|s| Arc::ptr_eq(s, &self.sources))
    }
    pub fn accepts(&self, executor: &str) -> bool {
        self.backends.contains_key(executor)
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn polling_enabled(&self) -> bool {
        self.polling.enabled || self.has_debug()
    }
    pub fn can_create_runs(&self) -> bool {
        self.creation
    }
    pub fn has_collectors(&self) -> bool {
        self.exchange.is_some() && self.backends.values().any(|b| b.collector.is_some())
    }
    /// `label` names the caller in the lock diagnostics; it is a static string.
    fn lock(&self, label: &'static str) -> cloud::Result<lock_diagnostics::Guard> {
        let locked = self.directory.lock(
            OsStr::new(LOCK),
            LockOptions {
                wait: Wait::Until {
                    deadline: std::time::Instant::now() + LOCK_WAIT,
                    cancelled: None,
                },
                ..Default::default()
            },
        );
        match locked {
            Ok(lock) => Ok(self.lock_holder.acquired(lock, label)),
            Err(cause) => {
                if cause.kind() == std::io::ErrorKind::WouldBlock {
                    self.lock_holder.busy(label);
                }
                Err((why_lock!(
                    cloud::Error {
                        code: "CLOUD_RUN_BUSY".into(),
                        http_status: Some(503),
                        uncertain: false,
                    },
                    cloud::Error::evidence()
                ))(cause))
            }
        }
    }
    fn attempt_slot(project: &str, run: &str, attempt: &str) -> cloud::Result<String> {
        Ok(format!(
            "{}.json",
            native::attempt_slot(project, run, attempt).map_err(why!(cloud::Error::evidence()))?
        ))
    }
    fn current_attempt(&self, project: &str, run: &str) -> cloud::Result<String> {
        let root = self
            .authority(project, run)?
            .child(OsStr::new(run), false)
            .map_err(io)?;
        let status = read(&root, "status.json")?;
        if status["project"] != project || status["run_id"] != run {
            return Err(cloud::Error::evidence());
        }
        let attempt = status["attempt_id"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        ml_exp_core::first_submission::native::retry::number(attempt)
            .map_err(why!(cloud::Error::evidence()))?;
        Ok(attempt.into())
    }
    fn saved(&self, project: &str, run: &str) -> cloud::Result<Option<Prepared>> {
        self.saved_attempt(project, run, &self.current_attempt(project, run)?)
    }
    fn saved_attempt(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
    ) -> cloud::Result<Option<Prepared>> {
        let name = Self::attempt_slot(project, run, attempt)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(None);
        }
        let prepared: Prepared = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(why!(cloud::Error::evidence()))?;
        action::validate(&prepared.plan, &prepared.execution)
            .map_err(why!(cloud::Error::evidence()))?;
        if prepared.execution.binding.project != project
            || prepared.execution.binding.run_id != run
            || prepared.execution.binding.attempt_id != attempt
            || prepared.plan["native_cloud"]["approval"] != json!(prepared.approval)
        {
            return Err(cloud::Error::evidence());
        }
        Ok(Some(prepared))
    }
    fn load(&self, action: &str) -> cloud::Result<Prepared> {
        let snapshot = self.actions.observe(action).map_err(op)?;
        if snapshot["native_cloud"]["format"] != "native-cloud-action.v1" {
            return Err(missing());
        }
        let project = snapshot["scope"]["project"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        let run = snapshot["run_id"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        let attempt = snapshot["attempt_id"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        let archived =
            ml_exp_core::renewal::archive(action).map_err(why!(cloud::Error::evidence()))?;
        let prepared = match crate::renewal::read(&self.slots, &archived).map_err(io)? {
            Some(value) => value,
            None => self
                .saved_attempt(project, run, attempt)?
                .ok_or_else(missing)?,
        };
        if prepared.plan["action_id"] != action {
            return Err(cloud::Error::evidence());
        }
        self.require_live(
            &prepared.execution.job.project,
            &prepared.execution.job.run_id,
        )?;
        action::validate(&prepared.plan, &prepared.execution)
            .map_err(why!(cloud::Error::evidence()))?;
        Ok(prepared)
    }
}

/// Read configuration dependencies only. Constructing an HTTP client sends no
/// requests; no namespace, lock, provider effect or SDK process is opened here.
pub(super) fn check_configuration(config: Settings) -> cloud::Result<()> {
    let invalid = |code: &str| cloud::Error {
        code: code.into(),
        http_status: None,
        uncertain: false,
    };
    for definition in config.backends {
        let credentials = Credentials::load(&definition.credentials_file).map_err(why!(
            invalid("CLOUD_PROVIDER_CREDENTIALS_INVALID"),
            "provider credentials invalid"
        ))?;
        if !credentials.matches_profile(&definition.profile) {
            return Err(invalid("CLOUD_PROVIDER_CREDENTIALS_INVALID"));
        }
        let mut client = Client::new(
            credentials,
            Duration::from_secs(definition.timeout_seconds.into()),
        )?;
        if let Some(path) = &definition.registry_auth_file {
            client = client.with_registry_auth(path).map_err(why!(
                invalid("CLOUD_REGISTRY_AUTH_INVALID"),
                "registry credentials invalid"
            ))?;
        }
        drop(client);
    }
    if let Some(exchange) = config.exchange {
        exchange::Authority::load(exchange)
            .map_err(why!(invalid("CLOUD_EXCHANGE_CONFIGURATION_INVALID")))?;
    }
    if let Some(objects) = config.artifacts {
        crate::objects::Store::open(objects)
            .map_err(why!(invalid("CLOUD_OBJECTS_CONFIGURATION_INVALID")))?;
    }
    if let Some(builds) = config.builds {
        builds
            .validate()
            .map_err(why!(invalid("CLOUD_BUILDER_CONFIGURATION_INVALID")))?;
    }
    if let Some(debug) = config.debug {
        debug::Runtime::open(debug).map_err(why!(invalid("CLOUD_DEBUG_CONFIGURATION_INVALID")))?;
    }
    Ok(())
}
mod actions;
mod artifacts;
mod authority;
mod authorship;
pub(super) mod builds;
mod checkpoint_archive;
mod derived;
mod diagnostics;
mod exchange;
mod observation;
mod rebuild;
mod retry;
pub mod startup_failure;
mod state_purge;
mod synchronization;

mod datasets;
mod oss_import;

pub(super) mod debug;

mod capacity;
mod data_copies;
mod data_delivery;
mod maintenance;
mod materialization;
mod metric_gap;
mod preparations;
pub use preparations::{Continue, ExternalExecutors};
mod renewal;
mod sync_jobs;
