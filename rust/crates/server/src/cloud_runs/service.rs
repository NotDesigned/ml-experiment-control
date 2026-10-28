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
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Prepared {
    execution: Execution,
    plan: Value,
    approval: Approval,
}
pub struct Service {
    settings: read_config::Settings,
    workspace: String,
    directory: Directory,
    slots: Directory,
    anchor: File,
    actions: operations::Store,
    outbox: Outbox,
    fence: RunFence,
    sources: Arc<sources::Service>,
    lifecycle: Option<Arc<projects::Service>>,
    backends: BTreeMap<String, Backend>,
    enabled: bool,
    creation: bool,
    tracking: ml_exp_core::tracking::RoutingSettings,
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
fn op(_: operations::Error) -> cloud::Error {
    cloud::Error {
        code: "CLOUD_ACTION_UNAVAILABLE".into(),
        http_status: None,
        uncertain: false,
    }
}
fn io(_: std::io::Error) -> cloud::Error {
    cloud::Error::evidence()
}
fn encode(value: &impl Serialize) -> cloud::Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|_| cloud::Error::evidence())
}
fn read(dir: &Directory, name: &str) -> cloud::Result<Value> {
    let bytes = dir
        .read_regular(OsStr::new(name), 1024 * 1024)
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
        operations::Store::open(settings.registry_root.join("actions")).map_err(op)?;
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
            if backends
                .insert(
                    definition.executor.clone(),
                    Backend {
                        definition,
                        adapter,
                    },
                )
                .is_some()
            {
                return Err(cloud::Error::configuration());
            }
        }
        if backends.is_empty() {
            return Err(cloud::Error::configuration());
        }
        let actions =
            operations::Store::open_existing(settings.registry_root.join("actions")).map_err(op)?;
        let outbox = Outbox::open(&settings.registry_root.join(NAME).join("effects"))?;
        let fence = RunFence::new(&settings.registry_root);
        Ok(Self {
            settings,
            workspace,
            directory,
            slots,
            anchor,
            actions,
            outbox,
            fence,
            sources,
            lifecycle,
            backends,
            enabled: config.enabled,
            creation: config.allow_run_creation,
            tracking,
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
    pub fn can_create_runs(&self) -> bool {
        self.creation
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
    fn slot(project: &str, run: &str) -> cloud::Result<String> {
        Ok(format!(
            "{}.json",
            native::attempt_slot(project, run, "attempt-001")
                .map_err(|_| cloud::Error::evidence())?
        ))
    }
    fn saved(&self, project: &str, run: &str) -> cloud::Result<Option<Prepared>> {
        let name = Self::slot(project, run)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(None);
        }
        let value = read(&self.slots, &name)?;
        let prepared: Prepared =
            serde_json::from_value(value).map_err(|_| cloud::Error::evidence())?;
        action::validate(&prepared.plan, &prepared.execution)
            .map_err(|_| cloud::Error::evidence())?;
        if prepared.execution.binding.project != project
            || prepared.execution.binding.run_id != run
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
        let prepared = self.saved(project, run)?.ok_or_else(missing)?;
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
        match error.http_status {
            Some(403) => (
                403,
                "CLOUD_OPERATION_DISABLED",
                "the operator has disabled this operation",
            ),
            Some(404) => (
                404,
                "CLOUD_SUBMISSION_NOT_FOUND",
                "the original cloud submission is unavailable",
            ),
            Some(503) => (
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
mod authority;
mod authorship;
mod observation;
