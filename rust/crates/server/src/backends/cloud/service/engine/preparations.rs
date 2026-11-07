//! Durable experiment setup over the same build and submission contracts.
//! Continuing may build the environment and freeze an Action; it never
//! executes training. GET only reads retained evidence.
use super::*;
use ml_exp_core::preparation::Request;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    project: String,
    request: Request,
    worker_sha256: String,
    /// What the Runtime's identity is built from, when that is not `worker_sha256`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_identity: Option<String>,
    started: bool,
    submission: Option<Value>,
}
impl Saved {
    fn frozen(&self) -> &str {
        self.runtime_identity
            .as_deref()
            .unwrap_or(&self.worker_sha256)
    }
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
        frozen: &str,
        external: Option<&ExternalExecutors>,
    ) -> cloud::Result<()> {
        let run = request
            .resolve(project, frozen)
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
            if let Some(code) = &run.code_source_id {
                self.check_code(project, code)?;
            }
            let (source_id, image) = if let Some(recipe) = &request.runtime {
                (recipe.source_id.clone(), recipe.base_image.clone())
            } else {
                let runtime = self.runtime(project, &run.executor, &run.runtime_id)?;
                if run.code_source_id.is_some() {
                    self.check_new_code_run(project, &run.run_id, &runtime)?;
                }
                (runtime.source_id, runtime.image)
            };
            let resources = &run.resources;
            let job = ml_exp_core::cloud::Job {
                project: project.into(),
                run_id: run.run_id.clone(),
                attempt_id: "attempt-001".into(),
                submission_token: "0".repeat(32),
                source_id,
                image,
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
        } else if run.code_source_id.is_some() {
            // Before any image is built for a Run that could never be created.
            return Err(cloud::Error {
                code: "CODE_DELIVERY_UNSUPPORTED".into(),
                http_status: Some(422),
                uncertain: false,
            });
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
    /// An identity that names a worker can only be built with that worker, so an unbuilt
    /// preparation frozen with an old one could never finish. Nothing ran under the old
    /// identity, so it moves on, but only once its build is proven to have left nothing
    /// behind. A built Runtime keeps its identity, and an identity that names no worker
    /// has nothing to follow. A frozen identity with no record at all was left by a stop
    /// between saving and creating it, and nothing can have run under that either.
    ///
    /// True means a record of the current identity must exist before this is relied on.
    fn follow_current_worker(&self, saved: &mut Saved) -> cloud::Result<bool> {
        let Some(settings) = &self.builds else {
            return Ok(false);
        };
        let Some(definition) = &saved.request.runtime else {
            return Ok(false);
        };
        let identity = self.bootstrap_identity(&definition.executor);
        let old = saved
            .request
            .resolve(&saved.project, saved.frozen())
            .map_err(why!(cloud::Error::evidence()))?;
        let recorded = self.runtime_recorded(&old.runtime_id)?;
        if saved.frozen() == identity.unwrap_or(&settings.worker_sha256) {
            return Ok(!recorded);
        }
        let unbuilt = !recorded || {
            let status = self.runtime_status(&saved.project, &old.runtime_id)?;
            status["runtime"].is_null() && status["retry_safe"] == true
        };
        if unbuilt {
            saved.worker_sha256 = settings.worker_sha256.clone();
            saved.runtime_identity = identity.map(str::to_owned);
            return Ok(true);
        }
        Ok(false)
    }
    fn setup_view(&self, saved: &Saved) -> cloud::Result<Value> {
        let id = saved
            .request
            .id(&saved.project)
            .map_err(why!(cloud::Error::evidence()))?;
        let run = saved
            .request
            .resolve(&saved.project, saved.frozen())
            .map_err(why!(cloud::Error::evidence()))?;
        let runtime = if saved.request.runtime.is_some() {
            if self.runtime_recorded(&run.runtime_id)? {
                self.runtime_status(&saved.project, &run.runtime_id)?
            } else {
                // A read creates nothing. The next continue does.
                Self::unrecorded_runtime(&saved.project, &run.runtime_id)
            }
        } else {
            json!({"runtime_id":run.runtime_id,"status":"EXISTING"})
        };
        let mut value = json!({"project":saved.project,"preparation_id":id,"run_id":run.run_id,"attempt_id":if saved.submission.is_some(){Some("attempt-001")}else{None},"runtime":runtime,
            "status":if saved.submission.is_some(){"READY"}else if saved.started{"PREPARING"}else{"PREPARED"},"phase":if saved.submission.is_some(){"SUBMISSION_PREPARED"}else{"ENVIRONMENT_PREPARATION"},
            "confirmation":format!("PREPARE {id}"),"submission":saved.submission,"training_authorized":false,"scheduler_submitted":false,
            "frozen":{"executor":run.executor,"gpus":run.resources.gpus,"cpus":run.resources.cpus,"memory_gb":run.resources.memory_gb,"max_gpu_hours":saved.request.approval.max_gpu_hours}});
        if let Some(code) = &run.code_source_id {
            value["frozen"]["code_source_id"] = json!(code);
        }
        // What this preparation will run as, including one saved before it had a choice.
        if let Some(backend) = self
            .backends
            .get(&run.executor)
            .filter(|backend| matches!(backend.adapter.profile, cloud::adapter::Profile::Pai(_)))
        {
            value["frozen"]["allocation"] = json!(
                run.scheduling
                    .as_ref()
                    .map_or(backend.definition.allocation, |s| s.allocation)
            );
        }
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
        let identity = request
            .runtime
            .as_ref()
            .and_then(|definition| self.bootstrap_identity(&definition.executor));
        self.setup_preflight(project, &request, identity.unwrap_or(&worker), external)?;
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
            runtime_identity: identity.map(str::to_owned),
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
        // The record comes before anything that points at it, and creating it twice is
        // harmless, so a busy lock or a stop at any point here recovers on the next continue.
        let needed = {
            let _lock = self.lock()?;
            let mut saved = self.setup_load(project, id)?;
            if saved.submission.is_some() {
                return self.setup_view(&saved).map(Continue::View);
            }
            if self.follow_current_worker(&mut saved)? {
                saved.request.runtime.clone()
            } else {
                None
            }
        };
        if let Some(definition) = needed {
            self.prepare_runtime(project, definition)?;
        }
        let (saved, still_needed) = {
            let _lock = self.lock()?;
            let mut saved = self.setup_load(project, id)?;
            if saved.submission.is_some() {
                return self.setup_view(&saved).map(Continue::View);
            }
            saved.started = true;
            let still_needed = self.follow_current_worker(&mut saved)?;
            self.slots
                .replace_regular(OsStr::new(&name(id)?), &encode(&saved)?)
                .map_err(io)?;
            (saved, still_needed)
        };
        // Only if the old build became retryable between the two looks.
        if still_needed {
            if let Some(definition) = &saved.request.runtime {
                self.prepare_runtime(project, definition.clone())?;
            }
        }
        let run = saved
            .request
            .resolve(project, saved.frozen())
            .map_err(why!(cloud::Error::evidence()))?;
        self.setup_preflight(project, &saved.request, saved.frozen(), external)?;
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
        // The preflight above has checked this, and a build frozen here keeps its worker.
        self.author_run(project, run.clone(), now)?;
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
