//! Durable experiment setup over the same build and submission contracts.
//! Continuing may build the environment and freeze an Action; it never
//! authorizes or executes training. GET only reads retained evidence.
use super::*;
use ml_exp_core::preparation::Request;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    project: String,
    request: Request,
    worker_sha256: String,
    started: bool,
    submission: Option<Value>,
}
fn name(id: &str) -> cloud::Result<String> {
    if !id.strip_prefix("preparation-").is_some_and(|s| {
        s.len() == 32
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    }) {
        return Err(cloud::Error::configuration());
    }
    Ok(format!("{id}.setup.json"))
}
impl Service {
    /// Check known provider limits and control metadata before spending time on
    /// image construction. Dataset bytes are unbounded by this metadata budget.
    fn setup_preflight(
        &self,
        project: &str,
        request: &Request,
        worker: &str,
        slurm: Option<&crate::submissions::Service>,
    ) -> cloud::Result<()> {
        let run = request
            .resolve(project, worker)
            .map_err(|_| cloud::Error::configuration())?;
        if let Some(history) = &self.history {
            history
                .require_available(project, &run.run_id)
                .map_err(|error| cloud::Error {
                    code: error.code.into(),
                    http_status: Some(error.status),
                    uncertain: false,
                })?;
        }
        if let Some(backend) = self.backends.get(&run.executor) {
            if matches!(
                backend.adapter.profile,
                cloud::adapter::Profile::Sensecore(_)
            ) && run
                .scheduling
                .as_ref()
                .is_some_and(|s| s.allocation != ml_exp_core::cloud::pai::Allocation::Guaranteed)
            {
                return Err(cloud::Error {
                    code: "INVALID_CLOUD_REQUEST".into(),
                    http_status: Some(422),
                    uncertain: false,
                });
            }
            let runtime = if let Some(recipe) = &request.runtime {
                ml_exp_core::cloud::run::Runtime {
                    runtime_id: run.runtime_id.clone(),
                    source_id: recipe.source_id.clone(),
                    image: recipe.base_image.clone(),
                    source_dir: "/ml-expd/source".into(),
                    worker_path: "/usr/local/bin/ml-exp-worker-rs".into(),
                    worker_sha256: worker.into(),
                    worker_features: Default::default(),
                }
            } else {
                self.runtime(project, &run.executor, &run.runtime_id)?
            };
            let resources = &run.resources;
            let job = ml_exp_core::cloud::Job {
                project: project.into(),
                run_id: run.run_id.clone(),
                attempt_id: "attempt-001".into(),
                submission_token: "0".repeat(32),
                source_id: runtime.source_id,
                image: runtime.image,
                command: run.command.clone(),
                gpus: resources.gpus,
                cpus: resources.cpus,
                memory_gb: resources
                    .memory_gb
                    .ok_or_else(cloud::Error::configuration)?,
                max_time_seconds: run.walltime_seconds,
                priority: run.scheduling.as_ref().map(|s| s.priority).unwrap_or(5),
            };
            match &backend.adapter.profile {
                cloud::adapter::Profile::Pai(p) => p.create(
                    &job,
                    run.scheduling
                        .as_ref()
                        .map(|s| s.allocation)
                        .unwrap_or(backend.definition.allocation),
                ),
                cloud::adapter::Profile::Sensecore(p) => p.create(&job),
            }
            .map_err(|_| cloud::Error {
                code: "INVALID_CLOUD_REQUEST".into(),
                http_status: Some(422),
                uncertain: false,
            })?;
            let mut bytes = encode(request)?.len();
            for input in &run.data {
                let definition = if let Some(r) = backend
                    .definition
                    .datasets
                    .iter()
                    .find(|r| r.project == project && r.asset_id == input.asset_id)
                {
                    r.definition.clone()
                } else if let Some(r) = backend
                    .definition
                    .imports
                    .iter()
                    .find(|r| r.project == project && r.definition.asset_id() == input.asset_id)
                {
                    r.definition.clone()
                } else {
                    self.uploaded_data(project, &run.executor, &input.asset_id)?
                        .ok_or_else(cloud::Error::evidence)?
                };
                bytes = bytes
                    .checked_add(encode(&definition)?.len().saturating_mul(2))
                    .ok_or_else(cloud::Error::configuration)?;
            }
            // Leave room for manifests, bindings, channel and environment.
            if bytes > 32 * 1024 {
                return Err(cloud::Error {
                    code: "LAUNCH_METADATA_TOO_LARGE".into(),
                    http_status: Some(422),
                    uncertain: false,
                });
            }
        } else if !slurm.is_some_and(|s| s.accepts(&run.executor)) || run.scheduling.is_some() {
            return Err(cloud::Error::configuration());
        }
        if !self.backends.contains_key(&run.executor)
            && run.data_preparation.is_some()
            && !slurm.is_some_and(|s| s.supports_script_preparation(&run.executor))
        {
            return Err(cloud::Error {
                code: "SCRIPT_PREPARATION_NOT_CONFIGURED".into(),
                http_status: Some(422),
                uncertain: false,
            });
        }
        Ok(())
    }
    fn setup_load(&self, project: &str, id: &str) -> cloud::Result<Saved> {
        let saved: Saved = serde_json::from_value(read(&self.slots, &name(id)?)?)
            .map_err(|_| cloud::Error::evidence())?;
        if saved.project != project
            || saved
                .request
                .id(project)
                .map_err(|_| cloud::Error::evidence())?
                != id
        {
            return Err(cloud::Error::evidence());
        }
        Ok(saved)
    }
    fn setup_view(&self, saved: &Saved) -> cloud::Result<Value> {
        let id = saved
            .request
            .id(&saved.project)
            .map_err(|_| cloud::Error::evidence())?;
        let run = saved
            .request
            .resolve(&saved.project, &saved.worker_sha256)
            .map_err(|_| cloud::Error::evidence())?;
        let runtime = if saved.request.runtime.is_some() {
            self.runtime_status(&saved.project, &run.runtime_id)?
        } else {
            json!({"runtime_id":run.runtime_id,"status":"EXISTING"})
        };
        let mut value = json!({"project":saved.project,"preparation_id":id,"run_id":run.run_id,"attempt_id":if saved.submission.is_some(){Some("attempt-001")}else{None},"runtime":runtime,
            "status":if saved.submission.is_some(){"READY"}else if saved.started{"PREPARING"}else{"PREPARED"},"phase":if saved.submission.is_some(){"SUBMISSION_PREPARED"}else{"ENVIRONMENT_PREPARATION"},
            "confirmation":format!("PREPARE {id}"),"submission":saved.submission,"training_authorized":false,"scheduler_submitted":false});
        if saved.submission.is_none() && saved.started && self.accepts(&run.executor) {
            if let Some(progress) = self.script_status(&saved.project, &run.run_id)? {
                value["phase"] = json!(if progress["status"] == "READY" {
                    "DATA_PREPARED"
                } else {
                    "DATA_PREPARATION"
                });
                value["status"] = if progress["status"] == "READY" {
                    json!("PREPARING")
                } else {
                    progress["status"].clone()
                };
                value["data_preparation"] = progress["data_preparation"].clone();
                value["error_code"] = progress["error_code"].clone();
            }
        }
        Ok(value)
    }
    pub fn experiment_prepare(
        &self,
        project: &str,
        request: Request,
        slurm: Option<&crate::submissions::Service>,
    ) -> cloud::Result<Value> {
        request
            .validate(project)
            .map_err(|_| cloud::Error::configuration())?;
        let id = request
            .id(project)
            .map_err(|_| cloud::Error::configuration())?;
        self.project_hash(project)?;
        // Reserve the deterministic request slot before any build effect. A
        // repeated request ID with another payload is always a conflict.
        {
            let _lock = self.lock()?;
            if self
                .slots
                .entry_exists(OsStr::new(&name(&id)?))
                .map_err(io)?
            {
                let saved = self.setup_load(project, &id)?;
                if json!(saved.request) != json!(request) {
                    return Err(cloud::Error::evidence());
                }
                return self.setup_view(&saved);
            }
        }
        let worker = self
            .builds
            .as_ref()
            .map(|s| s.worker_sha256.clone())
            .unwrap_or_else(|| format!("sha256:{}", "0".repeat(64)));
        self.setup_preflight(project, &request, &worker, slurm)?;
        if let Some(definition) = &request.runtime {
            self.prepare_runtime(project, definition.clone())?;
        }
        let lock = self.lock()?;
        if self
            .slots
            .entry_exists(OsStr::new(&name(&id)?))
            .map_err(io)?
        {
            let saved = self.setup_load(project, &id)?;
            if json!(saved.request) != json!(request) {
                return Err(cloud::Error::evidence());
            }
            return self.setup_view(&saved);
        }
        let saved = Saved {
            project: project.into(),
            request,
            worker_sha256: worker,
            started: false,
            submission: None,
        };
        lock.validate()?;
        write(&self.slots, &name(&id)?, &saved)?;
        self.setup_view(&saved)
    }
    pub fn experiment_status(&self, project: &str, id: &str) -> cloud::Result<Value> {
        self.setup_view(&self.setup_load(project, id)?)
    }
    pub fn experiment_continue(
        self: &Arc<Self>,
        project: &str,
        id: &str,
        confirmation: &str,
        slurm: Option<Arc<crate::submissions::Service>>,
        context: crate::submissions::Context<'_>,
    ) -> cloud::Result<Value> {
        let crate::submissions::Context {
            now,
            expires,
            transition_id: transition,
        } = context;
        if !self.enabled || confirmation != format!("PREPARE {id}") {
            return Err(cloud::Error::configuration());
        }
        let mut saved = {
            let lock = self.lock()?;
            let mut saved = self.setup_load(project, id)?;
            if saved.submission.is_some() {
                return self.setup_view(&saved);
            }
            saved.started = true;
            lock.validate()?;
            self.slots
                .replace_regular(OsStr::new(&name(id)?), &encode(&saved)?)
                .map_err(io)?;
            saved
        };
        let run = saved
            .request
            .resolve(project, &saved.worker_sha256)
            .map_err(|_| cloud::Error::evidence())?;
        self.setup_preflight(
            project,
            &saved.request,
            &saved.worker_sha256,
            slurm.as_deref(),
        )?;
        if saved.request.runtime.is_some() {
            let runtime = self.runtime_status(project, &run.runtime_id)?;
            if runtime["status"] != "READY" {
                if runtime["retry_safe"] == true {
                    self.execute_runtime(
                        project,
                        &run.runtime_id,
                        &format!("BUILD {}", run.runtime_id),
                    )?;
                } else {
                    self.reconcile_runtime(project, &run.runtime_id)?;
                }
                return self.setup_view(&saved);
            }
        }
        let submission = if self.accepts(&run.executor) {
            if self.backends[&run.executor].data_copier.is_some() {
                for input in &run.data {
                    if self
                        .data_saved(project, &run.executor, &input.asset_id)?
                        .is_some_and(|s| s.state.ready.is_none())
                    {
                        let request_id = canonical::digest(&json!([id, &input.asset_id]))
                            .map_err(|_| cloud::Error::evidence())?[..32]
                            .to_owned();
                        let data = self.publish_dataset(
                            project,
                            &run.executor,
                            &input.asset_id,
                            &request_id,
                        )?;
                        if data["ready"].is_null() {
                            let mut view = self.setup_view(&saved)?;
                            view["phase"] = json!("DATA_PREPARATION");
                            view["data"] = data;
                            return Ok(view);
                        }
                    }
                }
            }
            self.create_run(project, run.clone(), now)?;
            if self.prepare_script(project, &run.run_id)?.is_some() {
                return self.setup_view(&saved);
            }
            self.prepare(
                project,
                &run.run_id,
                saved.request.approval.clone(),
                now,
                expires,
                transition,
            )?
        } else {
            let slurm = slurm.ok_or_else(cloud::Error::configuration)?;
            slurm
                .create_run(project, run.clone(), now)
                .map_err(|_| cloud::Error::evidence())?;
            slurm
                .prepare(
                    project,
                    &run.run_id,
                    saved.request.approval.clone(),
                    crate::submissions::Context {
                        now,
                        expires,
                        transition_id: transition,
                    },
                )
                .map_err(|_| cloud::Error::evidence())?
        };
        let lock = self.lock()?;
        saved = self.setup_load(project, id)?;
        if saved
            .submission
            .as_ref()
            .is_some_and(|old| old["submission_id"] != submission["submission_id"])
        {
            return Err(cloud::Error::evidence());
        }
        saved.submission = Some(submission);
        lock.validate()?;
        self.slots
            .replace_regular(OsStr::new(&name(id)?), &encode(&saved)?)
            .map_err(io)?;
        self.setup_view(&saved)
    }
}
