//! Rust owns Runtime identity, dispatch intent and reconciliation. The existing
//! isolated builder may remain Python; its narrow native-image recipe receipt
//! cannot be mistaken for a legacy Python-worker image.
use super::*;
use ml_exp_core::cloud::{build::Definition, run::Runtime};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub socket: std::path::PathBuf,
    pub peer_uid: u32,
    pub worker_sha256: String,
    pub data_base_image: Option<String>,
    #[serde(default)]
    pub slurm: std::collections::BTreeMap<String, Materializer>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Materializer {
    pub ssh: crate::submissions::config::Ssh,
    pub target: helper::Target,
    pub ssh_alias: String,
}
impl Materializer {
    fn connection(&self) -> cloud::Result<crate::backends::slurm::SshConfig> {
        let s = &self.ssh;
        let value = crate::backends::slurm::SshConfig {
            target: self.target.clone(),
            ssh_binary: s.binary.clone(),
            config_file: s.config_file.clone(),
            known_hosts_file: s.known_hosts_file.clone(),
            target_alias: self.ssh_alias.clone(),
            helper_path: s.helper_path.clone(),
            helper_config_path: s.helper_config_path.clone(),
            local_cwd: s.local_cwd.clone(),
            agent_socket: s.agent_socket.clone(),
            budget: crate::backends::slurm::Budget {
                timeout: Duration::from_secs(3600),
                connect_timeout_seconds: s.connect_timeout_seconds,
                termination_grace: Duration::from_secs(3),
                drain_grace: Duration::from_secs(3),
            },
        };
        value
            .validate()
            .map_err(|_| cloud::Error::configuration())?;
        Ok(value)
    }
}
impl Settings {
    pub fn validate(&self) -> cloud::Result<()> {
        if !ml_exp_core::managed::absolute(&self.socket)
            || !helper::valid_digest(&self.worker_sha256)
        {
            return Err(cloud::Error::configuration());
        }
        for (id, target) in &self.slurm {
            if id != &target.target.backend_id {
                return Err(cloud::Error::configuration());
            }
            target.connection()?;
        }
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    project: String,
    runtime_id: String,
    definition: Definition,
    source_receipt_sha256: String,
    worker_sha256: String,
    started: bool,
    #[serde(default)]
    slurm_runtime: Option<ml_exp_core::first_submission::native::Runtime>,
    runtime: Option<Runtime>,
    diagnostic: Option<Value>,
}
struct Activity {
    active: Arc<std::sync::Mutex<std::collections::BTreeSet<String>>>,
    id: String,
}
impl Drop for Activity {
    fn drop(&mut self) {
        if let Ok(mut active) = self.active.lock() {
            active.remove(&self.id);
        }
    }
}
fn name(id: &str) -> cloud::Result<String> {
    if !id
        .strip_prefix("runtime.")
        .is_some_and(|s| helper::valid_digest(&format!("sha256:{s}")))
    {
        return Err(cloud::Error::configuration());
    }
    Ok(format!("{id}.runtime.json"))
}
impl Service {
    pub fn has_builds(&self) -> bool {
        self.builds.is_some()
    }
    pub fn runtimes(&self, project: &str) -> cloud::Result<Value> {
        self.project_hash(project)?;
        let mut rows = Vec::new();
        for name in self.slots.entry_names_limited(32768).map_err(io)? {
            let Some(id) = name.to_str().and_then(|s| s.strip_suffix(".runtime.json")) else {
                continue;
            };
            let value = read(
                &self.slots,
                name.to_str().ok_or_else(cloud::Error::evidence)?,
            )?;
            if value["project"] == project {
                rows.push(self.runtime_status(project, id)?);
            }
        }
        Ok(json!({"project":project,"runtimes":rows}))
    }
    fn build_record(&self, project: &str, id: &str) -> cloud::Result<Record> {
        let record: Record = serde_json::from_value(read(&self.slots, &name(id)?)?)
            .map_err(|_| cloud::Error::evidence())?;
        if record.project != project
            || record.runtime_id != id
            || record
                .definition
                .id(project, &record.worker_sha256)
                .map_err(|_| cloud::Error::evidence())?
                != id
            || canonical::protocol_identity(
                &self
                    .sources
                    .read(project, &record.definition.source_id)
                    .map_err(|_| cloud::Error::evidence())?,
            )
            .map_err(|_| cloud::Error::evidence())?
                != record.source_receipt_sha256
        {
            return Err(cloud::Error::evidence());
        }
        if let Some(runtime) = &record.runtime {
            runtime.validate().map_err(|_| cloud::Error::evidence())?;
            if runtime.runtime_id != id
                || runtime.source_id != record.definition.source_id
                || runtime.worker_sha256 != record.worker_sha256
                || !record.started
            {
                return Err(cloud::Error::evidence());
            }
        }
        Ok(record)
    }
    fn build_view(&self, record: &Record) -> Value {
        let active = self
            .active_builds
            .lock()
            .is_ok_and(|s| s.contains(&record.runtime_id));
        json!({"project":record.project,"runtime_id":record.runtime_id,"source_id":record.definition.source_id,"executor":record.definition.executor,
        "status":if record.runtime.is_some()&&(!self.builds.as_ref().is_some_and(|s|s.slurm.contains_key(&record.definition.executor))||record.slurm_runtime.is_some()){"READY"}else if active{"EXECUTING"}else if record.started{"RECONCILE_REQUIRED"}else{"PREPARED"},"runtime":record.runtime,"slurm_runtime":record.slurm_runtime,"build_error":record.diagnostic,
        "retry_safe":!record.started&&!active,"scheduler_submitted":false,"confirmation":format!("BUILD {}",record.runtime_id)})
    }
    pub fn runtime(&self, project: &str, executor: &str, id: &str) -> cloud::Result<Runtime> {
        if let Some(runtime) = self
            .backends
            .get(executor)
            .and_then(|b| b.definition.runtimes.get(id))
        {
            return Ok(runtime.clone());
        }
        let record = self.build_record(project, id)?;
        if record.definition.executor != executor {
            return Err(cloud::Error::evidence());
        }
        record.runtime.ok_or_else(cloud::Error::evidence)
    }
    pub fn prepare_runtime(&self, project: &str, definition: Definition) -> cloud::Result<Value> {
        let lock = self.lock()?;
        let settings = self
            .builds
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        definition
            .validate()
            .map_err(|_| cloud::Error::configuration())?;
        if !(self.accepts(&definition.executor)
            || settings.slurm.contains_key(&definition.executor))
        {
            return Err(cloud::Error::configuration());
        }
        let source = self
            .sources
            .read(project, &definition.source_id)
            .map_err(|_| cloud::Error::evidence())?;
        let id = definition
            .id(project, &settings.worker_sha256)
            .map_err(|_| cloud::Error::configuration())?;
        if self
            .slots
            .entry_exists(OsStr::new(&name(&id)?))
            .map_err(io)?
        {
            return self.runtime_status(project, &id);
        }
        let record = Record {
            project: project.into(),
            runtime_id: id,
            definition,
            source_receipt_sha256: canonical::protocol_identity(&source)
                .map_err(|_| cloud::Error::evidence())?,
            worker_sha256: settings.worker_sha256.clone(),
            started: false,
            slurm_runtime: None,
            runtime: None,
            diagnostic: None,
        };
        lock.validate()?;
        write(&self.slots, &name(&record.runtime_id)?, &record)?;
        Ok(self.build_view(&record))
    }
    pub fn runtime_status(&self, project: &str, id: &str) -> cloud::Result<Value> {
        Ok(self.build_view(&self.build_record(project, id)?))
    }
    fn builder_call(&self, record: &Record, operation: &str) -> cloud::Result<(u16, Value)> {
        let settings = self
            .builds
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        if record.worker_sha256 != settings.worker_sha256 {
            return Err(cloud::Error::evidence());
        }
        let mirror = self
            .backends
            .get(&record.definition.executor)
            .is_some_and(|b| matches!(b.adapter.profile, cloud::adapter::Profile::Sensecore(_)));
        let payload = json!({"operation":format!("native-{operation}"),"runtime_id":record.runtime_id,"project":record.project,"source_id":record.definition.source_id,
            "base_image":record.definition.base_image,"dockerfile":record.definition.dockerfile,"worker_sha256":record.worker_sha256,"mirror":mirror});
        let timeout = if operation == "build" { 7200 } else { 30 };
        let (status, bytes) = ml_exp_io::unix_http::request_observed(
            &settings.socket,
            settings.peer_uid,
            &encode(&payload)?,
            Duration::from_secs(timeout),
        )
        .map_err(|error| cloud::Error {
            code: if error.may_have_sent {
                "NATIVE_BUILD_RPC_UNCERTAIN"
            } else {
                "NATIVE_BUILD_NOT_DISPATCHED"
            }
            .into(),
            http_status: None,
            uncertain: error.may_have_sent,
        })?;
        let value = Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| cloud::Error::evidence())?,
        );
        Ok((status, value))
    }
    fn accept_build(&self, record: &mut Record, status: u16, result: Value) -> cloud::Result<()> {
        if status == 200 {
            let runtime: Runtime = serde_json::from_value(result["runtime"].clone())
                .map_err(|_| cloud::Error::evidence())?;
            runtime.validate().map_err(|_| cloud::Error::evidence())?;
            if result["runtime_id"] != record.runtime_id
                || result["project"] != record.project
                || result["source_id"] != record.definition.source_id
                || result["recipe"] != ml_exp_core::cloud::build::CONTRACT
                || runtime.runtime_id != record.runtime_id
                || runtime.source_id != record.definition.source_id
                || runtime.worker_sha256 != record.worker_sha256
            {
                return Err(cloud::Error::evidence());
            }
            if record.runtime.as_ref().is_some_and(|old| *old != runtime) {
                return Err(cloud::Error::evidence());
            }
            record.runtime = Some(runtime);
            record.started = true;
            record.diagnostic = None;
        } else {
            // Diagnostics have no commands, URLs, credentials or raw response.
            record.diagnostic = Some(
                json!({"http_status":status,"code":result["code"].as_str().filter(|s|s.len()<=100&&s.bytes().all(|b|b.is_ascii_uppercase()||b.is_ascii_digit()||b==b'_')).unwrap_or("NATIVE_BUILD_UNRESOLVED"),"retry_safe":false}),
            );
        }
        self.slots
            .replace_regular(OsStr::new(&name(&record.runtime_id)?), &encode(record)?)
            .map_err(io)
    }
    pub fn execute_runtime(
        self: &Arc<Self>,
        project: &str,
        id: &str,
        confirmation: &str,
    ) -> cloud::Result<Value> {
        let lock = self.lock()?;
        let mut record = self.build_record(project, id)?;
        if !self.enabled || confirmation != format!("BUILD {id}") {
            return Err(cloud::Error::configuration());
        }
        if record.started || self.active_builds.lock().is_ok_and(|s| s.contains(id)) {
            return Ok(self.build_view(&record));
        }
        record.started = true;
        record.diagnostic = Some(json!({"code":"BUILD_EXECUTING","retry_safe":false}));
        self.slots
            .replace_regular(OsStr::new(&name(id)?), &encode(&record)?)
            .map_err(io)?;
        lock.validate()?;
        drop(lock);
        let response = json!({"project":record.project,"runtime_id":record.runtime_id,"status":"EXECUTING","scheduler_submitted":false,"retry_safe":false});
        let service = Arc::clone(self);
        let project = project.to_owned();
        let id = id.to_owned();
        self.active_builds
            .lock()
            .map_err(|_| cloud::Error::evidence())?
            .insert(id.clone());
        let activity = Activity {
            active: Arc::clone(&self.active_builds),
            id: id.clone(),
        };
        std::thread::Builder::new()
            .name("native-image-build".into())
            .spawn(move || {
                let _activity = activity;
                let result = service.builder_call(&record, "build");
                // A busy metadata lock cannot erase an uncertain effect. The saved
                // intent remains reconcilable if receipt publication cannot commit.
                if let Ok(_lock) = service.lock() {
                    if let Ok(mut current) = service.build_record(&project, &id) {
                        match result {
                            Ok((status, value)) => {
                                let _ = service.accept_build(&mut current, status, value);
                            }
                            Err(error)
                                if error.code == "NATIVE_BUILD_NOT_DISPATCHED"
                                    && !error.uncertain
                                    && current.runtime.is_none() =>
                            {
                                current.started = false;
                                current.diagnostic =
                                    Some(json!({"code":"NATIVE_BUILD_NOT_DISPATCHED",
                                    "retry_safe":true,"scheduler_submitted":false}));
                                if let (Ok(name), Ok(bytes)) = (name(&id), encode(&current)) {
                                    let _ =
                                        service.slots.replace_regular(OsStr::new(&name), &bytes);
                                }
                            }
                            Err(_) => {
                                let _ = service.accept_build(
                                    &mut current,
                                    503,
                                    json!({"code":"NATIVE_BUILD_RPC_UNCERTAIN"}),
                                );
                            }
                        }
                    }
                }
                if service
                    .builds
                    .as_ref()
                    .is_some_and(|s| s.slurm.values().any(|_| true))
                {
                    let _ = service.materialize_runtime(&project, &id, true);
                }
            })
            .map_err(|_| cloud::Error {
                code: "BUILD_DISPATCH_UNCERTAIN".into(),
                http_status: None,
                uncertain: true,
            })?;
        Ok(response)
    }
    pub fn reconcile_runtime(&self, project: &str, id: &str) -> cloud::Result<Value> {
        let mut record = self.build_record(project, id)?;
        if self.active_builds.lock().is_ok_and(|s| s.contains(id)) {
            return Ok(self.build_view(&record));
        }
        if record.runtime.is_none() && record.started {
            let (status, value) = self.builder_call(&record, "get")?;
            let lock = self.lock()?;
            record = self.build_record(project, id)?;
            self.accept_build(&mut record, status, value)?;
            lock.validate()?;
        }
        if self
            .builds
            .as_ref()
            .is_some_and(|s| s.slurm.contains_key(&record.definition.executor))
            && record.runtime.is_some()
            && record.slurm_runtime.is_none()
        {
            return self.materialize_runtime(project, id, false);
        }
        Ok(self.build_view(&record))
    }
    pub fn runtime_logs(&self, project: &str, id: &str, progress: bool) -> cloud::Result<Value> {
        let record = self.build_record(project, id)?;
        let (status, value) =
            self.builder_call(&record, if progress { "progress" } else { "logs" })?;
        if status != 200 || value["runtime_id"] != id {
            return Err(cloud::Error::evidence());
        }
        let text = experiment_redact::redact_line(&value.to_string());
        serde_json::from_str(&text).map_err(|_| cloud::Error::evidence())
    }
}

impl Service {
    pub fn slurm_runtime(
        &self,
        project: &str,
        executor: &str,
        id: &str,
    ) -> cloud::Result<ml_exp_core::first_submission::native::Runtime> {
        let record = self.build_record(project, id)?;
        if record.definition.executor != executor {
            return Err(cloud::Error::evidence());
        }
        record.slurm_runtime.ok_or_else(cloud::Error::evidence)
    }
    pub fn materialize_runtime(
        &self,
        project: &str,
        id: &str,
        build: bool,
    ) -> cloud::Result<Value> {
        let record = self.build_record(project, id)?;
        if record.slurm_runtime.is_some() {
            return Ok(self.build_view(&record));
        }
        let settings = self
            .builds
            .as_ref()
            .and_then(|s| s.slurm.get(&record.definition.executor))
            .ok_or_else(cloud::Error::configuration)?;
        let mut ssh = settings.connection()?;
        let request = ml_exp_core::cloud::materialize::Request {
            operation: if build { "build" } else { "get" }.into(),
            executor: record.definition.executor.clone(),
            runtime: record.runtime.clone().ok_or_else(cloud::Error::evidence)?,
        };
        request
            .validate()
            .map_err(|_| cloud::Error::configuration())?;
        if !build {
            ssh.budget.timeout = Duration::from_secs(90);
        }
        let mut argv = ssh.argv();
        let at = argv
            .iter()
            .position(|s| s == "slurm-helper")
            .ok_or_else(cloud::Error::configuration)?;
        argv[at] = "image-materialize".into();
        argv.remove(at + 1);
        let environment = ssh.environment();
        let bytes = encode(&request)?;
        let result = ml_exp_io::process::execute(ml_exp_io::process::Request {
            argv: &argv,
            cwd: &ssh.local_cwd,
            environment: ml_exp_io::process::Environment::Replace(&environment),
            stdin: ml_exp_io::process::Input::Bytes(&bytes),
            stdout: ml_exp_io::process::Capture::Memory { limit: 65536 },
            stderr: ml_exp_io::process::Capture::Memory { limit: 65536 },
            deadline: std::time::Instant::now() + ssh.budget.timeout,
            termination_grace: ssh.budget.termination_grace,
            drain_grace: ssh.budget.drain_grace,
            cancelled: &AtomicBool::new(false),
        })
        .map_err(io)?;
        if result.exit_code != Some(0)
            || result.timed_out
            || result.cancelled
            || result.failure.is_some()
            || !result.stdin_complete
            || !result.stdout.complete
            || result.stdout.truncated
        {
            return Err(cloud::Error {
                code: "OCI_TO_SIF_UNCONFIRMED".into(),
                http_status: Some(503),
                uncertain: false,
            });
        }
        let value = Value::Object(
            ml_exp_core::json::mapping(&result.stdout.bytes)
                .map_err(|_| cloud::Error::evidence())?,
        );
        let runtime: ml_exp_core::first_submission::native::Runtime =
            serde_json::from_value(value["runtime"].clone())
                .map_err(|_| cloud::Error::evidence())?;
        let expected = record.runtime.as_ref().ok_or_else(cloud::Error::evidence)?;
        if runtime.runtime_id != id
            || runtime.source_id != expected.source_id
            || runtime.oci_image != expected.image
            || runtime.image_id
                != expected
                    .image
                    .rsplit_once('@')
                    .ok_or_else(cloud::Error::evidence)?
                    .1
            || !helper::valid_digest(&runtime.sif_sha256)
            || !ml_exp_core::slurm::absolute_path(&runtime.sif_path)
            || runtime.worker.as_ref().map(|w| &w.program) != Some(&expected.worker_path)
        {
            return Err(cloud::Error::evidence());
        }
        let lock = self.lock()?;
        let mut current = self.build_record(project, id)?;
        if current
            .slurm_runtime
            .as_ref()
            .is_some_and(|r| *r != runtime)
        {
            return Err(cloud::Error::evidence());
        }
        current.slurm_runtime = Some(runtime);
        lock.validate()?;
        self.slots
            .replace_regular(OsStr::new(&name(id)?), &encode(&current)?)
            .map_err(io)?;
        Ok(self.build_view(&current))
    }
}
