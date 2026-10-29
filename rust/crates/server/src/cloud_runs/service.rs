use super::{Definition, Settings};
use crate::{
    backends::cloud::{
        self, Client, Credentials, action::Runner, adapter::Adapter, outbox::Outbox,
    },
    deletion::RunFence,
    experiments,
    operations::{self, ClaimContext, Execute},
    projects, read_api, read_config, sources,
};
use fs2::FileExt;
use ml_exp_core::{
    canonical,
    cloud::{
        action::{self, Execution},
        run,
    },
    first_submission::native::{self, Approval, Binding, creation},
    operations::{FrozenFiles, Policy},
    slurm::helper,
};
use ml_exp_io::directory::Directory;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fs::File,
    path::Path,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};
const NAME: &str = "cloud-submissions";
const LOCK: &str = ".cloud-runs.lock";
struct Backend {
    definition: Definition,
    adapter: Adapter,
    collector: Option<Adapter>,
    data_copier: Option<Adapter>,
}
#[derive(Serialize, Deserialize)]
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
    anchor: File,
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
    debug: Option<debug::Runtime>,
    pub(super) polling: ml_exp_core::polling::Policy,
}
struct Lock<'a> {
    service: &'a Service,
    file: File,
}
impl Lock<'_> {
    fn validate(&self) -> cloud::Result<()> {
        self.service
            .directory
            .verify_private_parent()
            .map_err(|_| cloud::Error::evidence())?;
        for file in [&self.file, &self.service.anchor] {
            self.service
                .directory
                .validate_regular_entry(OsStr::new(LOCK), file)
                .map_err(|_| cloud::Error::evidence())?;
        }
        Ok(())
    }
}
impl Drop for Lock<'_> {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}
pub(super) fn op(_: operations::Error) -> cloud::Error {
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
    serde_json::to_vec(value).map_err(|_| cloud::Error::evidence())
}
pub(super) fn read(dir: &Directory, name: &str) -> cloud::Result<Value> {
    let bytes = dir
        .read_regular(OsStr::new(name), 8 * 1024 * 1024)
        .map_err(io)?;
    let value =
        Value::Object(ml_exp_core::json::mapping(&bytes).map_err(|_| cloud::Error::evidence())?);
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
impl Service {
    pub fn initialize(settings: &read_config::Settings) -> cloud::Result<()> {
        settings
            .validate()
            .map_err(|_| cloud::Error::configuration())?;
        let (root, _) = read_config::workspace(&settings.registry_root)
            .map_err(|_| cloud::Error::configuration())?;
        let directory = root.create_new_child(OsStr::new(NAME)).map_err(io)?;
        directory
            .create_new_child(OsStr::new("slots"))
            .map_err(io)?;
        directory
            .create_new_regular(OsStr::new(LOCK))
            .map_err(io)?
            .sync_all()
            .map_err(io)?;
        let outbox = directory
            .create_new_child(OsStr::new("effects"))
            .map_err(io)?;
        outbox.verify_private_parent().map_err(io)?;
        Outbox::initialize(&settings.registry_root.join(NAME).join("effects"))?;
        directory
            .create_new_child(OsStr::new("sync-effects"))
            .map_err(io)?;
        Outbox::initialize(&settings.registry_root.join(NAME).join("sync-effects"))?;
        operations::Store::open(settings.registry_root.join("actions")).map_err(op)?;
        crate::checkpoints::Store::open(&settings.registry_root)
            .map_err(|_| cloud::Error::evidence())?;
        Ok(())
    }
    pub fn open(
        settings: read_config::Settings,
        config: Settings,
        sources: Arc<sources::Service>,
        lifecycle: Option<Arc<projects::Service>>,
        tracking: ml_exp_core::tracking::RoutingSettings,
    ) -> cloud::Result<Self> {
        settings
            .validate()
            .map_err(|_| cloud::Error::configuration())?;
        let (root, workspace) = read_config::workspace(&settings.registry_root)
            .map_err(|_| cloud::Error::configuration())?;
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
        directory.verify_private_parent().map_err(io)?;
        let slots = directory.child(OsStr::new("slots"), false).map_err(io)?;
        let anchor = directory.open_existing_lock(OsStr::new(LOCK)).map_err(io)?;
        let mut backends = BTreeMap::new();
        for definition in config.backends {
            definition.validate()?;
            let mut client = Client::new(
                Credentials::load(&definition.credentials_file)?,
                Duration::from_secs(definition.timeout_seconds.into()),
            )?;
            if let Some(path) = &definition.registry_auth_file {
                client = client.with_registry_auth(path)?;
            }
            let adapter = Adapter::new(definition.profile.clone(), client)?;
            let collector = definition
                .collector
                .as_ref()
                .map(|profile| {
                    let mut client = Client::new(
                        Credentials::load(&definition.credentials_file)?,
                        Duration::from_secs(definition.timeout_seconds.into()),
                    )?;
                    if let Some(path) = &definition.registry_auth_file {
                        client = client.with_registry_auth(path)?;
                    }
                    Adapter::new(profile.clone(), client)
                })
                .transpose()?;
            let data_copier = definition
                .data_copier
                .as_ref()
                .map(|profile| {
                    let mut client = Client::new(
                        Credentials::load(&definition.credentials_file)?,
                        Duration::from_secs(definition.timeout_seconds.into()),
                    )?;
                    if let Some(path) = &definition.registry_auth_file {
                        client = client.with_registry_auth(path)?;
                    }
                    Adapter::new(profile.clone(), client)
                })
                .transpose()?;
            if backends
                .insert(
                    definition.executor.clone(),
                    Backend {
                        definition,
                        adapter,
                        collector,
                        data_copier,
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
            anchor,
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
            debug: config.debug.map(debug::Runtime::open).transpose()?,
            polling: config.polling,
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
    fn lock(&self) -> cloud::Result<Lock<'_>> {
        self.directory
            .validate_regular_entry(OsStr::new(LOCK), &self.anchor)
            .map_err(io)?;
        let file = self
            .directory
            .open_existing_lock(OsStr::new(LOCK))
            .map_err(io)?;
        file.try_lock_exclusive().map_err(|_| cloud::Error {
            code: "CLOUD_RUN_BUSY".into(),
            http_status: Some(503),
            uncertain: false,
        })?;
        let lock = Lock {
            service: self,
            file,
        };
        lock.validate()?;
        Ok(lock)
    }
    fn attempt_slot(project: &str, run: &str, attempt: &str) -> cloud::Result<String> {
        Ok(format!(
            "{}.json",
            native::attempt_slot(project, run, attempt).map_err(|_| cloud::Error::evidence())?
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
            .map_err(|_| cloud::Error::evidence())?;
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
            .map_err(|_| cloud::Error::evidence())?;
        action::validate(&prepared.plan, &prepared.execution)
            .map_err(|_| cloud::Error::evidence())?;
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
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(missing)?;
        if prepared.plan["action_id"] != action {
            return Err(cloud::Error::evidence());
        }
        self.check(&prepared)?;
        Ok(prepared)
    }
}
pub(crate) fn failure(error: cloud::Error) -> read_api::Error {
    let (status, code, detail) = if error.uncertain {
        (
            500,
            "CLOUD_OPERATION_UNCERTAIN",
            "cloud operation may have started; query and reconcile the original identity; do not replay",
        )
    } else {
        match (error.code.as_str(), error.http_status) {
            ("INPUT_NOT_READY", _) => (
                409,
                "INPUT_NOT_READY",
                "the original input asset has not reached verified READY; inspect its delivery record",
            ),
            ("LAUNCH_METADATA_TOO_LARGE", _) => (
                422,
                "LAUNCH_METADATA_TOO_LARGE",
                "launch metadata exceeds the control package budget; dataset payload size is independent",
            ),
            ("DEBUG_MACHINE_LIMIT", _) => (
                409,
                "DEBUG_MACHINE_LIMIT",
                "the configured active debug machine limit has been reached",
            ),
            ("ARTIFACTS_ADMINISTRATIVELY_REMOVED", _) => (
                410,
                "ARTIFACTS_ADMINISTRATIVELY_REMOVED",
                "result objects were administratively removed; collection cannot recreate them",
            ),
            (_, Some(422)) => (
                422,
                "INVALID_CLOUD_REQUEST",
                "cloud request violates the configured resource or protocol limits",
            ),
            (_, Some(410)) => (
                410,
                "CLOUD_OBJECT_GONE",
                "the original cloud object has been released or retired",
            ),
            (_, Some(401)) => (
                401,
                "WORKER_AUTH_REQUIRED",
                "Attempt worker authentication required",
            ),
            (_, Some(403)) => (
                403,
                "CLOUD_OPERATION_DISABLED",
                "the operator has disabled this operation",
            ),
            (_, Some(404)) => (
                404,
                "CLOUD_SUBMISSION_NOT_FOUND",
                "the original cloud submission is unavailable",
            ),
            (_, Some(503)) => (
                503,
                "CLOUD_RUN_BUSY",
                "the original cloud operation is busy",
            ),
            _ => (
                409,
                "CLOUD_EVIDENCE_UNAVAILABLE",
                "cloud configuration, frozen authority or provider evidence cannot be validated",
            ),
        }
    };
    read_api::Error {
        status,
        code,
        detail,
    }
}

mod actions;
mod artifacts;
mod authority;
mod authorship;
pub(super) mod builds;
mod exchange;
mod observation;
mod retry;

mod datasets;

pub(super) mod debug;

mod capacity;
mod data_delivery;
mod maintenance;
mod preparations;

#[cfg(test)]
mod failure_tests {
    use super::*;
    #[test]
    fn public_failures_keep_actionable_types_and_never_echo_untrusted_codes() {
        for (code, status) in [
            ("INPUT_NOT_READY", 409),
            ("LAUNCH_METADATA_TOO_LARGE", 422),
            ("DEBUG_MACHINE_LIMIT", 409),
            ("ARTIFACTS_ADMINISTRATIVELY_REMOVED", 410),
        ] {
            let public = failure(cloud::Error {
                code: code.into(),
                http_status: None,
                uncertain: false,
            });
            assert_eq!(public.code, code);
            assert_eq!(public.status, status);
        }
        let public = failure(cloud::Error {
            code: "untrusted-secret-provider-error".into(),
            http_status: Some(422),
            uncertain: false,
        });
        assert_eq!(public.status, 422);
        assert!(!public.detail.contains("secret"));
        // A possible effect always takes precedence over a convenient status.
        let public = failure(cloud::Error {
            code: "LAUNCH_METADATA_TOO_LARGE".into(),
            http_status: Some(422),
            uncertain: true,
        });
        assert_eq!(public.status, 500);
        assert_eq!(public.code, "CLOUD_OPERATION_UNCERTAIN");
    }
}
