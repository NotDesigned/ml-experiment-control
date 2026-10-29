//! Opt-in administrative plans, distinct from download acknowledgements. Keep
//! small immutable manifests/audit records; payload removal needs a reviewed
//! plan and exact terminal scheduler evidence from the original backend.
use crate::{cloud_runs, deletion::RunFence, experiments, read_config, submissions};
use ml_exp_core::{
    checkpoints::{self},
    maintenance::{Plan, Selection},
};
use ml_exp_io::directory::Directory;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{ffi::OsStr, path::Path, sync::Arc};
fn invalid() -> crate::read_api::Error {
    crate::read_api::Error {
        status: 409,
        code: "ADMINISTRATION_BLOCKED",
        detail: "administrative identity, terminal evidence, checkpoint dependencies or original plan cannot be verified",
    }
}
type Result<T> = std::result::Result<T, crate::read_api::Error>;
/// Run authorship and retirement share this admission lock. This is separate
/// from the read-only Run fence so dependency scans can retain normal guards.
pub(crate) struct Admission(std::fs::File);
impl Drop for Admission {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.0);
    }
}
fn admission(root: &Path, exclusive: bool) -> Result<Option<Admission>> {
    let root = Directory::open(root, false).map_err(|_| invalid())?;
    let directory = match root.child(OsStr::new("administration"), false) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(invalid()),
    };
    let file = directory
        .open_existing_lock(OsStr::new(".admission.lock"))
        .map_err(|_| invalid())?;
    if exclusive {
        fs2::FileExt::try_lock_exclusive(&file)
    } else {
        fs2::FileExt::try_lock_shared(&file)
    }
    .map_err(|_| invalid())?;
    directory
        .validate_regular_entry(OsStr::new(".admission.lock"), &file)
        .map_err(|_| invalid())?;
    Ok(Some(Admission(file)))
}
pub(crate) fn authorship_guard(root: &Path) -> Result<Option<Admission>> {
    admission(root, false)
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Choice {
    Outputs,
    State,
    Checkpoint { checkpoint_id: String },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub request_id: String,
    pub selection: Choice,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    action: String,
    cloud: bool,
    plan: Plan,
    checkpoints: Vec<Value>,
    started: bool,
    receipt: Option<Value>,
    object_cursor: u64,
    objects_complete: bool,
}
pub struct Service {
    settings: read_config::Settings,
    directory: Directory,
    slurm: Option<Arc<submissions::Service>>,
    cloud: Option<Arc<cloud_runs::Service>>,
}
impl Service {
    pub fn initialize(settings: &read_config::Settings) -> Result<()> {
        let (root, _) = read_config::workspace(&settings.registry_root).map_err(|_| invalid())?;
        let directory = root
            .create_new_child(OsStr::new("administration"))
            .map_err(|_| invalid())?;
        directory
            .open_lock(OsStr::new(".lock"))
            .map_err(|_| invalid())?;
        directory
            .open_lock(OsStr::new(".admission.lock"))
            .map_err(|_| invalid())?;
        Ok(())
    }
    pub fn open(
        settings: read_config::Settings,
        slurm: Option<Arc<submissions::Service>>,
        cloud: Option<Arc<cloud_runs::Service>>,
    ) -> Result<Self> {
        let (root, _) = read_config::workspace(&settings.registry_root).map_err(|_| invalid())?;
        let directory = root
            .child(OsStr::new("administration"), false)
            .map_err(|_| invalid())?;
        directory.verify_private_parent().map_err(|_| invalid())?;
        directory
            .open_existing_lock(OsStr::new(".lock"))
            .map_err(|_| invalid())?;
        Ok(Self {
            settings,
            directory,
            slurm,
            cloud,
        })
    }
    pub fn matches(&self, root: &Path) -> bool {
        self.settings.registry_root == root
    }
    fn lock(&self) -> Result<std::fs::File> {
        let file = self
            .directory
            .open_existing_lock(OsStr::new(".lock"))
            .map_err(|_| invalid())?;
        fs2::FileExt::try_lock_exclusive(&file).map_err(|_| invalid())?;
        Ok(file)
    }
    fn name(id: &str) -> Result<String> {
        if ml_exp_core::sources::validate_archive_digest(id).is_err() {
            return Err(invalid());
        }
        Ok(format!("{id}.json"))
    }
    fn read(&self, id: &str) -> Result<Saved> {
        let bytes = self
            .directory
            .read_regular(OsStr::new(&Self::name(id)?), 1024 * 1024)
            .map_err(|_| invalid())?;
        let saved: Saved = serde_json::from_value(Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?,
        ))
        .map_err(|_| invalid())?;
        if saved.plan.id().map_err(|_| invalid())? != id
            || saved
                .receipt
                .as_ref()
                .is_some_and(|r| saved.plan.receipt().ok().as_ref() != Some(r))
        {
            return Err(invalid());
        }
        Ok(saved)
    }
    fn save(&self, saved: &Saved) -> Result<()> {
        self.directory
            .replace_regular(
                OsStr::new(&Self::name(&saved.plan.id().map_err(|_| invalid())?)?),
                &serde_json::to_vec(saved).map_err(|_| invalid())?,
            )
            .map_err(|_| invalid())
    }
    fn view(saved: &Saved) -> Result<Value> {
        let id = saved.plan.id().map_err(|_| invalid())?;
        Ok(
            json!({"plan_id":id,"submission_id":saved.action,"identity":saved.plan.config.binding().map_err(|_|invalid())?.identity,"selection":saved.plan.selection,"paths":saved.plan.paths().map_err(|_|invalid())?,"status":if saved.receipt.is_some()&&saved.objects_complete{"COMPLETED"}else if saved.started{"RECONCILE_REQUIRED"}else{"PREPARED"},"confirmation":format!("DELETE {id}"),"scientific_records_preserved":true,"object_parts_released":saved.object_cursor,"receipt":saved.receipt}),
        )
    }
    pub fn status(&self, id: &str) -> Result<Value> {
        Self::view(&self.read(id)?)
    }
    fn checkpoint_list(&self, cloud: bool, action: &str) -> Result<Vec<Value>> {
        // Scope comes from the committed Action read, never caller input.
        let now = crate::projects::transition_context()
            .map_err(|_| invalid())?
            .0;
        let value = if cloud {
            self.cloud
                .as_ref()
                .ok_or_else(invalid)?
                .get(action, &now)
                .map_err(|_| invalid())?
        } else {
            self.slurm
                .as_ref()
                .ok_or_else(invalid)?
                .get(action, &now)
                .map_err(|_| invalid())?
        };
        let (p, r, a) = (
            value["project"].as_str().ok_or_else(invalid)?,
            value["run_id"].as_str().ok_or_else(invalid)?,
            value["attempt_id"].as_str().ok_or_else(invalid)?,
        );
        let result = if cloud {
            self.cloud
                .as_ref()
                .ok_or_else(invalid)?
                .checkpoints(p, r, a, None)
                .map_err(|_| invalid())?
        } else {
            self.slurm
                .as_ref()
                .ok_or_else(invalid)?
                .checkpoints(p, r, a, None)
                .map_err(|_| invalid())?
        };
        result["checkpoints"]
            .as_array()
            .cloned()
            .ok_or_else(invalid)
    }
    pub fn prepare(&self, action: &str, request: Request, now: &str) -> Result<Value> {
        ml_exp_core::operations::validate_submission_id(action).map_err(|_| invalid())?;
        ml_exp_core::submission::validate_submission_token(&request.request_id)
            .map_err(|_| invalid())?;
        let _lock = self.lock()?;
        for name in self
            .directory
            .entry_names_limited(32768)
            .map_err(|_| invalid())?
        {
            let Some(id) = name.to_str().and_then(|n| n.strip_suffix(".json")) else {
                continue;
            };
            let saved = self.read(id)?;
            if saved.plan.request_id == request.request_id {
                let same = match (&request.selection, &saved.plan.selection) {
                    (Choice::Outputs, Selection::Outputs) | (Choice::State, Selection::State) => {
                        true
                    }
                    (
                        Choice::Checkpoint { checkpoint_id },
                        Selection::Checkpoint { checkpoint },
                    ) => checkpoint["checkpoint_id"] == *checkpoint_id,
                    _ => false,
                };
                if saved.action != action || !same {
                    return Err(invalid());
                }
                return Self::view(&saved);
            }
        }
        let cloud = self
            .cloud
            .as_ref()
            .is_some_and(|c| c.handles_action(action).unwrap_or(false));
        let checkpoints = if matches!(request.selection, Choice::Outputs) {
            Vec::new()
        } else {
            self.checkpoint_list(cloud, action)?
        };
        let selection = match request.selection {
            Choice::Outputs => Selection::Outputs,
            Choice::State => Selection::State,
            Choice::Checkpoint { checkpoint_id } => Selection::Checkpoint {
                checkpoint: Box::new(
                    checkpoints
                        .iter()
                        .find(|c| c["checkpoint_id"] == checkpoint_id)
                        .cloned()
                        .ok_or_else(invalid)?,
                ),
            },
        };
        let plan = if cloud {
            self.cloud
                .as_ref()
                .ok_or_else(invalid)?
                .maintenance_plan(action, selection, request.request_id, now)
                .map_err(|_| invalid())?
        } else {
            self.slurm
                .as_ref()
                .ok_or_else(invalid)?
                .maintenance_plan(action, selection, request.request_id, now)
                .map_err(|_| invalid())?
        };
        let checkpoints = match &plan.selection {
            Selection::Checkpoint { checkpoint } => vec![(**checkpoint).clone()],
            _ => checkpoints,
        };
        self.dependencies(&checkpoints)?;
        let id = plan.id().map_err(|_| invalid())?;
        if self
            .directory
            .entry_exists(OsStr::new(&Self::name(&id)?))
            .map_err(|_| invalid())?
        {
            return self.status(&id);
        }
        let saved = Saved {
            action: action.into(),
            cloud,
            plan,
            checkpoints,
            started: false,
            receipt: None,
            object_cursor: 0,
            objects_complete: !cloud,
        };
        self.save(&saved)?;
        Self::view(&saved)
    }
    fn dependencies(&self, checkpoints: &[Value]) -> Result<()> {
        if checkpoints.is_empty() {
            return Ok(());
        }
        let store = experiments::Store::open(
            self.settings.bindings(),
            RunFence::new(&self.settings.registry_root),
        )
        .map_err(|_| invalid())?;
        for project in &self.settings.projects {
            let discovered = store
                .discover(&project.project, &experiments::DiscoveryLimits::default())
                .map_err(|_| invalid())?;
            for entry in discovered.runs {
                let snapshot = store
                    .snapshot_metadata(&entry.reference)
                    .map_err(|_| invalid())?;
                for manifest in std::iter::once(&snapshot.run.metadata)
                    .chain(snapshot.attempts.iter().map(|a| &a.manifest.metadata))
                {
                    let reference = manifest.get("resume_from").unwrap_or(&Value::Null);
                    if checkpoints.iter().any(|c| {
                        c["project"] == project.project
                            && reference["checkpoint_id"] == c["checkpoint_id"]
                            && reference["run_id"] == c["run_id"]
                            && reference["attempt_id"] == c["attempt_id"]
                    }) {
                        return Err(invalid());
                    }
                }
            }
        }
        Ok(())
    }
    fn withdraw(&self, saved: &Saved) -> Result<()> {
        let store = crate::checkpoints::Store::open_existing(&self.settings.registry_root)
            .map_err(|_| invalid())?;
        for document in &saved.checkpoints {
            let (context, _) =
                checkpoints::validate_document(document, None, None).map_err(|_| invalid())?;
            store
                .withdraw(
                    &context,
                    document["checkpoint_id"].as_str().ok_or_else(invalid)?,
                    &saved.plan.id().map_err(|_| invalid())?,
                )
                .map_err(|_| invalid())?;
        }
        Ok(())
    }
    pub fn execute(&self, id: &str, confirmation: &str) -> Result<Value> {
        if confirmation != format!("DELETE {id}") {
            return Err(invalid());
        }
        let admission = admission(&self.settings.registry_root, true)?.ok_or_else(invalid)?;
        let lock = self.lock()?;
        let mut saved = self.read(id)?;
        if !saved.started {
            self.dependencies(&saved.checkpoints)?;
            saved.started = true;
            self.save(&saved)?;
        }
        // Re-entering after any interrupted withdrawal completes every marker
        // before the backend can remove bytes.
        self.withdraw(&saved)?;
        drop(lock);
        drop(admission);
        if saved.receipt.is_none() {
            if saved.cloud {
                let _ = self
                    .cloud
                    .as_ref()
                    .ok_or_else(invalid)?
                    .maintenance_execute(&saved.action, &saved.plan)
                    .map_err(|_| invalid())?;
            } else {
                let receipt = self
                    .slurm
                    .as_ref()
                    .ok_or_else(invalid)?
                    .maintenance_execute(&saved.action, &saved.plan)
                    .map_err(|_| invalid())?;
                let _lock = self.lock()?;
                saved = self.read(id)?;
                saved.receipt = Some(receipt);
                self.save(&saved)?;
            }
        }
        self.finish(id)
    }
    fn finish(&self, id: &str) -> Result<Value> {
        let _lock = self.lock()?;
        let mut saved = self.read(id)?;
        if saved.receipt.is_some() && !saved.objects_complete {
            if matches!(saved.plan.selection, Selection::Outputs) {
                for _ in 0..64 {
                    let (complete, cursor) = self
                        .cloud
                        .as_ref()
                        .ok_or_else(invalid)?
                        .maintenance_release_object(&saved.action, &saved.plan, saved.object_cursor)
                        .map_err(|_| invalid())?;
                    saved.object_cursor = cursor;
                    saved.objects_complete = complete;
                    self.save(&saved)?;
                    if complete {
                        break;
                    }
                }
            } else {
                saved.objects_complete = true;
                self.save(&saved)?;
            }
        }
        Self::view(&saved)
    }
    pub fn receive(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
        receipt: Value,
    ) -> Result<Value> {
        let id = receipt["plan_id"].as_str().ok_or_else(invalid)?.to_owned();
        let _lock = self.lock()?;
        let mut saved = self.read(&id)?;
        let binding = saved.plan.config.binding().map_err(|_| invalid())?;
        if !saved.started
            || !saved.cloud
            || binding.identity.project != project
            || binding.identity.run_id != run
            || binding.identity.attempt_id != attempt
            || saved.plan.receipt().map_err(|_| invalid())? != receipt
        {
            return Err(invalid());
        }
        self.cloud
            .as_ref()
            .ok_or_else(invalid)?
            .maintenance_auth(&saved.action, &saved.plan, token)
            .map_err(|_| invalid())?;
        saved.receipt = Some(receipt);
        self.save(&saved)?;
        Ok(json!({"plan_id":id,"status":"RECEIVED"}))
    }
    pub fn retire(&self, project: &str, run: &str, confirmation: &str, now: &str) -> Result<Value> {
        if confirmation != format!("DELETE {project}/{run}") {
            return Err(invalid());
        }
        let _admission = admission(&self.settings.registry_root, true)?.ok_or_else(invalid)?;
        let fence = RunFence::new(&self.settings.registry_root);
        let store = experiments::Store::open(self.settings.bindings(), fence.clone())
            .map_err(|_| invalid())?;
        let discovered = store
            .discover(project, &experiments::DiscoveryLimits::default())
            .map_err(|_| invalid())?;
        let reference = &discovered
            .runs
            .iter()
            .find(|r| r.reference.run_id == run)
            .ok_or_else(invalid)?
            .reference;
        let snapshot = store.snapshot_metadata(reference).map_err(|_| invalid())?;
        // The caller retires tracking separately from physical payloads. Every
        // Attempt must have exact terminal native provider evidence.
        for attempt in &snapshot.attempts {
            let value = if let Some(cloud) = &self.cloud {
                if cloud.handles_run(project, run).map_err(|_| invalid())? {
                    cloud
                        .for_attempt_admin(project, run, &attempt.attempt_id, now)
                        .map_err(|_| invalid())?
                } else {
                    self.slurm
                        .as_ref()
                        .ok_or_else(invalid)?
                        .for_attempt(project, run, &attempt.attempt_id, now)
                        .map_err(|_| invalid())?
                }
            } else {
                self.slurm
                    .as_ref()
                    .ok_or_else(invalid)?
                    .for_attempt(project, run, &attempt.attempt_id, now)
                    .map_err(|_| invalid())?
            };
            let action = value["submission_id"].as_str().ok_or_else(invalid)?;
            let request_id = ml_exp_core::canonical::digest(&json!([
                "retire",
                project,
                run,
                &attempt.attempt_id
            ]))
            .map_err(|_| invalid())?[..32]
                .to_owned();
            if self
                .cloud
                .as_ref()
                .is_some_and(|c| c.handles_action(action).unwrap_or(false))
            {
                self.cloud
                    .as_ref()
                    .ok_or_else(invalid)?
                    .maintenance_plan(action, Selection::Outputs, request_id, now)
                    .map_err(|_| invalid())?;
            } else {
                self.slurm
                    .as_ref()
                    .ok_or_else(invalid)?
                    .maintenance_plan(action, Selection::Outputs, request_id, now)
                    .map_err(|_| invalid())?;
            }
        }
        let guard = fence.exclusive_existing(false).map_err(|_| invalid())?;
        let facts = serde_json::Map::from_iter([
            ("run_sha256".into(), json!(snapshot.run.raw_sha256)),
            ("payloads_removed".into(), json!(false)),
            ("reason".into(), json!("explicit administrative retirement")),
        ]);
        guard
            .mark(project, run, &facts, now)
            .map_err(|_| invalid())?;
        Ok(
            json!({"project":project,"run_id":run,"status":"DELETED","payloads_removed":false,"audit_records_preserved":true}),
        )
    }
}
