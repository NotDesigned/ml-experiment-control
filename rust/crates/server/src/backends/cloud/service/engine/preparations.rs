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
/// The executors this service does not run, each with whether it can run a
/// data-preparation script. The caller supplies them; refusing here, before a build or
/// a copy, is the only reason this service looks.
pub type ExternalExecutors = std::collections::BTreeMap<String, bool>;
/// How far a continue got. Nothing here submits a Run on another backend: the
/// caller does that with no lock held, then records it with `experiment_finish`.
pub enum Continue {
    View(Value),
    SubmitElsewhere {
        run: Box<ml_exp_core::first_submission::native::creation::Request>,
        approval: ml_exp_core::first_submission::native::Approval,
    },
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
    /// Check provider resources, input references and the dataset manifest budget
    /// before image construction or a CPU copy can spend anything. The compiled
    /// worker package checks its actual serialized size later; request bytes and
    /// dataset inventories are not a package-size estimate.
    fn setup_preflight(
        &self,
        project: &str,
        request: &Request,
        worker: &str,
        external: Option<&ExternalExecutors>,
    ) -> cloud::Result<()> {
        let run = request
            .resolve(project, worker)
            .map_err(why!(cloud::Error::configuration()))?;
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
                read_only_storage: false,
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
            .map_err(why!(cloud::Error {
                code: "INVALID_CLOUD_REQUEST".into(),
                http_status: Some(422),
                uncertain: false,
            }))?;
            let mut mounts = Vec::with_capacity(run.data.len());
            for input in &run.data {
                let receipt = match backend
                    .definition
                    .datasets
                    .iter()
                    .find(|r| r.project == project && r.asset_id == input.asset_id)
                {
                    Some(receipt) => receipt.clone(),
                    None => {
                        self.undelivered_receipt(backend, project, &run.executor, &input.asset_id)?
                    }
                };
                mounts.push(ml_exp_core::datasets::Mount {
                    name: input.name.clone(),
                    receipt,
                });
            }
            ml_exp_core::datasets::manifest_budget(&mounts).map_err(|_| cloud::Error {
                code: "DATASET_MANIFESTS_TOO_LARGE".into(),
                http_status: Some(422),
                uncertain: false,
            })?;
        } else if !external.is_some_and(|e| e.contains_key(&run.executor))
            || run.scheduling.is_some()
        {
            return Err(cloud::Error::configuration());
        }
        if !self.backends.contains_key(&run.executor)
            && run.data_preparation.is_some()
            && external.and_then(|e| e.get(&run.executor)) != Some(&true)
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
        let slot = name(id)?;
        if !self.slots.entry_exists(OsStr::new(&slot)).map_err(io)? {
            return Err(missing());
        }
        let saved: Saved = serde_json::from_value(read(&self.slots, &slot)?)
            .map_err(why!(cloud::Error::evidence()))?;
        if saved.project != project
            || saved
                .request
                .id(project)
                .map_err(why!(cloud::Error::evidence()))?
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
            .map_err(why!(cloud::Error::evidence()))?;
        let run = saved
            .request
            .resolve(&saved.project, &saved.worker_sha256)
            .map_err(why!(cloud::Error::evidence()))?;
        let runtime = if saved.request.runtime.is_some() {
            self.runtime_status(&saved.project, &run.runtime_id)?
        } else {
            json!({"runtime_id":run.runtime_id,"status":"EXISTING"})
        };
        let mut value = json!({"project":saved.project,"preparation_id":id,"run_id":run.run_id,"attempt_id":if saved.submission.is_some(){Some("attempt-001")}else{None},"runtime":runtime,
            "status":if saved.submission.is_some(){"READY"}else if saved.started{"PREPARING"}else{"PREPARED"},"phase":if saved.submission.is_some(){"SUBMISSION_PREPARED"}else{"ENVIRONMENT_PREPARATION"},
            "confirmation":format!("PREPARE {id}"),"submission":saved.submission,"training_authorized":false,"scheduler_submitted":false,
            "frozen":{"executor":run.executor,"gpus":run.resources.gpus,"cpus":run.resources.cpus,"memory_gb":run.resources.memory_gb,"max_gpu_hours":saved.request.approval.max_gpu_hours}});
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
        external: Option<&ExternalExecutors>,
    ) -> cloud::Result<Value> {
        request
            .validate(project)
            .map_err(why!(cloud::Error::configuration()))?;
        let id = request
            .id(project)
            .map_err(why!(cloud::Error::configuration()))?;
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
        self.setup_preflight(project, &request, &worker, external)?;
        if let Some(definition) = &request.runtime {
            self.prepare_runtime(project, definition.clone())?;
        }
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
        let saved = Saved {
            project: project.into(),
            request,
            worker_sha256: worker,
            started: false,
            submission: None,
        };
        write(&self.slots, &name(&id)?, &saved)?;
        self.setup_view(&saved)
    }
    pub fn experiment_status(&self, project: &str, id: &str) -> cloud::Result<Value> {
        self.setup_view(&self.setup_load(project, id)?)
    }
    /// Discover retained preparation slots before a Run or Action exists.
    pub fn experiment_for_run(&self, project: &str, run: &str) -> cloud::Result<Option<Value>> {
        ml_exp_core::identity::validate_id(project)
            .and_then(|_| ml_exp_core::identity::validate_id(run))
            .map_err(why!(cloud::Error::configuration()))?;
        self.project_hash(project)?;
        let mut found = None;
        for slot in self.slots.entry_names().map_err(io)? {
            let Some(id) = slot.to_str().and_then(|s| s.strip_suffix(".setup.json")) else {
                continue;
            };
            let value = read(&self.slots, &name(id)?)?;
            if value["project"] != project || value["request"]["run"]["run_id"] != run {
                continue;
            }
            if found.is_some() {
                return Err(cloud::Error {
                    code: "PREPARATION_ID_REQUIRED".into(),
                    http_status: Some(409),
                    uncertain: false,
                });
            }
            found = Some(self.experiment_status(project, id)?);
        }
        Ok(found)
    }
    pub fn experiment_continue(
        self: &Arc<Self>,
        project: &str,
        id: &str,
        confirmation: &str,
        external: Option<&ExternalExecutors>,
        context: crate::runs::Context<'_>,
    ) -> cloud::Result<Continue> {
        let crate::runs::Context {
            now,
            expires,
            transition_id: transition,
        } = context;
        if !self.enabled || confirmation != format!("PREPARE {id}") {
            return Err(cloud::Error::configuration());
        }
        let saved = {
            let _lock = self.lock()?;
            let mut saved = self.setup_load(project, id)?;
            if saved.submission.is_some() {
                return self.setup_view(&saved).map(Continue::View);
            }
            saved.started = true;
            self.slots
                .replace_regular(OsStr::new(&name(id)?), &encode(&saved)?)
                .map_err(io)?;
            saved
        };
        let run = saved
            .request
            .resolve(project, &saved.worker_sha256)
            .map_err(why!(cloud::Error::evidence()))?;
        self.setup_preflight(project, &saved.request, &saved.worker_sha256, external)?;
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
                return self.setup_view(&saved).map(Continue::View);
            }
        }
        if !self.accepts(&run.executor) {
            return Ok(Continue::SubmitElsewhere {
                run: Box::new(run),
                approval: saved.request.approval,
            });
        }
        let backend = &self.backends[&run.executor];
        let pai = matches!(backend.adapter.profile, cloud::adapter::Profile::Pai(_));
        if backend.data_copier.is_some() || pai && backend.collector.is_some() {
            for input in &run.data {
                if let Some(data_saved) = self
                    .data_saved(project, &run.executor, &input.asset_id)?
                    .filter(|s| s.state.ready.is_none())
                {
                    let request_id = data_saved.copy_request_id.unwrap_or(
                        canonical::digest(&json!([id, &input.asset_id]))
                            .map_err(why!(cloud::Error::evidence()))?[..32]
                            .to_owned(),
                    );
                    let data = self.publish_dataset(
                        project,
                        &run.executor,
                        &input.asset_id,
                        &request_id,
                        pai.then_some(run.runtime_id.as_str()),
                    )?;
                    if data["ready"].is_null() {
                        let mut view = self.setup_view(&saved)?;
                        view["phase"] = json!("DATA_PREPARATION");
                        view["data"] = data;
                        return Ok(Continue::View(view));
                    }
                }
            }
        }
        self.create_run(project, run.clone(), now)?;
        if self.prepare_script(project, &run.run_id)?.is_some() {
            return self.setup_view(&saved).map(Continue::View);
        }
        let submission = self.prepare(
            project,
            &run.run_id,
            saved.request.approval.clone(),
            now,
            expires,
            transition,
        )?;
        self.experiment_finish(project, id, submission)
            .map(Continue::View)
    }
    /// Record the prepared submission. A repeat must name the same one, because its
    /// Action already exists and nothing may point this preparation at another.
    pub fn experiment_finish(
        &self,
        project: &str,
        id: &str,
        submission: Value,
    ) -> cloud::Result<Value> {
        let _lock = self.lock()?;
        let mut saved = self.setup_load(project, id)?;
        if saved
            .submission
            .as_ref()
            .is_some_and(|old| old["submission_id"] != submission["submission_id"])
        {
            return Err(cloud::Error::evidence());
        }
        saved.submission = Some(submission);
        self.slots
            .replace_regular(OsStr::new(&name(id)?), &encode(&saved)?)
            .map_err(io)?;
        self.setup_view(&saved)
    }
}
