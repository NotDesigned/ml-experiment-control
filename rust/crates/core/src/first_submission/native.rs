//! Native admission of an already-authored first Run/Attempt. Effects and
//! materialization receipts are supplied by the service; no caller script is
//! accepted in place of the command frozen in the scientific manifest.
use crate::{canonical, identity, slurm};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use slurm::helper::{self, FrozenIntent, Request, Scope, Target};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("native first submission admission failed: {0}")]
pub struct Error(pub &'static str);
pub type Result<T> = std::result::Result<T, Error>;
const FORMAT: &str = "native-first-submission.v1";
pub mod creation;
pub mod retry;

/// Project committed native submission evidence into an otherwise unsubmitted
/// Run read. This never refreshes a provider or replaces an existing scheduler
/// state. A receipt proves the recorded observation, not current liveness.
pub fn project_run(mut run: Value, submission: &Value) -> Result<Value> {
    let id = field(submission, "submission_id")?;
    crate::operations::validate_submission_id(id).map_err(|_| Error("Submission identity"))?;
    if submission["project"] != run["project"]
        || submission["run_id"] != run["run_id"]
        || submission["attempt_id"] != run["current_attempt_id"]
        || submission["operation"] != "SUBMIT_RUN"
        || submission["status"] != submission["execution"]["status"]
    {
        return Err(Error("Submission scope"));
    }
    run["native_submission"] = json!({"submission_id":id,"status":submission["status"],"source":"native-action-commit","next_action":submission["next_action"]});
    let cancelled = submission["cancellation"]["termination_verified"] == true;
    if !cancelled
        && submission["native_scheduler_observation"] != true
        && !matches!(
            run["evidence"]["scheduler"]["state"].as_str(),
            None | Some("UNKNOWN" | "NOT_SUBMITTED")
        )
    {
        return Ok(run);
    }
    let (state, detail, at) = match submission["status"].as_str() {
        Some("PREPARED" | "AUTHORIZED") => return Ok(run),
        Some("EXECUTING" | "RECONCILE_REQUIRED") => (
            json!("UNKNOWN"),
            json!({"diagnostic":"SUBMISSION_UNCERTAIN","submission_id":id}),
            Value::Null,
        ),
        Some("VERIFIED") => {
            let status: slurm::JobStatus = serde_json::from_value(
                submission["execution"]["result"]["observation"]["status"].clone(),
            )
            .map_err(|_| Error("scheduler receipt"))?;
            status.validate().map_err(|_| Error("scheduler receipt"))?;
            if status.job_id.as_str()
                != field(
                    &submission["execution"]["result"]["submission"],
                    "backend_job_id",
                )?
                || status.scheduler_name
                    != slurm::scheduler_job_name(
                        field(&run, "run_id")?,
                        field(submission, "attempt_id")?,
                    )
                    .map_err(|_| Error("scheduler name"))?
            {
                return Err(Error("scheduler binding"));
            }
            let status = if cancelled {
                let cancellation: slurm::cancellation::State = decode(&submission["cancellation"])?;
                cancellation
                    .validate()
                    .map_err(|_| Error("cancellation receipt"))?;
                let terminal = cancellation
                    .observation
                    .ok_or(Error("cancellation observation"))?
                    .status;
                if terminal.job_id != status.job_id
                    || terminal.ssh_alias != status.ssh_alias
                    || terminal.scheduler_name != status.scheduler_name
                    || terminal.partition != status.partition
                    || !terminal.state.is_terminal()
                    || run["evidence"]["scheduler"]["state"]
                        .as_str()
                        .is_some_and(|state| {
                            matches!(state, "SUCCEEDED" | "FAILED" | "CANCELLED" | "PREEMPTED")
                                && serde_json::to_value(terminal.state).ok().as_ref()
                                    != Some(&json!(state))
                        })
                {
                    return Err(Error("cancellation scope or terminal conflict"));
                }
                terminal
            } else {
                status
            };
            let state = serde_json::to_value(status.state).map_err(|_| Error("scheduler state"))?;
            let at = serde_json::to_value(&status.observed_at)
                .map_err(|_| Error("scheduler timestamp"))?;
            let detail = json!({"backend":submission["execution"]["result"]["submission"]["backend"],"backend_job_id":status.job_id,
                "raw_state":status.raw_state,"partition":status.partition,"elapsed":status.elapsed,"exit_code":status.exit_code,"reason":status.reason,"queue":status.queue,"submission_id":id});
            (state, detail, at)
        }
        _ => return Err(Error("Submission state")),
    };
    run["scheduler_state"] = state.clone();
    run["is_terminal"] = json!(matches!(
        state.as_str(),
        Some("SUCCEEDED" | "FAILED" | "CANCELLED" | "PREEMPTED")
    ));
    let source = if submission["native_scheduler_observation"] == true {
        "native-scheduler-observation"
    } else {
        "native-action-commit"
    };
    run["evidence"]["scheduler"] = json!({"state":state,"attempt_id":submission["attempt_id"],"detail":detail,"as_of":at,
        "source":source,"stale":if at.is_null(){Value::Null}else{json!(true)},"freshness":if at.is_null(){"UNKNOWN"}else{"STALE"},
        "stale_reason":if source=="native-action-commit"{"committed observation; no provider refresh on GET"}else{"cached observation; no provider refresh on GET"},"evidence_availability":{"availability":"PRESENT","source":source,"submission_id":id}});
    Ok(run)
}

/// Exact Attempt views consume the same committed scheduler evidence as Run
/// views. A historical Attempt never borrows another Attempt's receipt, and a
/// scheduler terminal state never becomes process or publication evidence.
pub fn project_attempt(mut value: Value, submission: &Value, scope: &Scope) -> Result<Value> {
    scope.validate().map_err(|_| Error("observation scope"))?;
    let collection = value["project"] == scope.project
        && value["run_id"] == scope.run_id
        && value["attempt_id"] == scope.attempt_id;
    let shown = value["scope"]["project"] == scope.project
        && value["scope"]["scope_type"] == "attempt"
        && value["scope"]["object_id"] == format!("{}::{}", scope.run_id, scope.attempt_id)
        && value["run_id"] == scope.run_id
        && value["attempt"]["attempt_id"] == scope.attempt_id;
    if !collection && !shown {
        return Err(Error("observation scope"));
    }
    let projected = project_run(
        json!({"project":scope.project,"run_id":scope.run_id,
        "current_attempt_id":scope.attempt_id,"evidence":value["evidence"]}),
        submission,
    )?;
    value["evidence"] = projected["evidence"].clone();
    value["native_submission"] = projected["native_submission"].clone();
    if shown
        && projected["scheduler_state"].is_string()
        && (submission["cancellation"]["termination_verified"] == true
            || submission["native_scheduler_observation"] == true
            || matches!(
                value["attempt"]["state"].as_str(),
                None | Some("UNKNOWN" | "NOT_SUBMITTED")
            ))
    {
        value["attempt"]["state"] = projected["scheduler_state"].clone();
        value["attempt"]["backend_job_id"] =
            projected["evidence"]["scheduler"]["detail"]["backend_job_id"].clone();
    }
    Ok(value)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub target: Target,
    pub ssh_alias: String,
    pub partition: String,
    pub account: String,
    pub qos: String,
    pub gpu_type: String,
    pub mount_root: String,
    pub run_root: String,
    pub data_root: String,
    pub max_gpus: u32,
    pub max_cpus: u32,
    pub max_memory_gb: u32,
    pub max_walltime_seconds: u64,
}

/// A published code+environment image, already materialized by the builder
/// shell. This is trusted operator/builder evidence, never an API assertion.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    pub runtime_id: String,
    pub source_id: String,
    pub image_id: String,
    pub sif_path: String,
    pub sif_sha256: String,
    pub oci_image: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<slurm::managed::Runtime>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Approval {
    pub max_gpu_hours: f64,
    #[serde(default)]
    pub reason: String,
}

/// Raw hashes are filesystem observations, not the scientific JSON identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub project: String,
    pub run_id: String,
    pub attempt_id: String,
    pub project_sha256: String,
    pub run_sha256: String,
    pub attempt_sha256: String,
    pub source_receipt_sha256: String,
    pub backend_config_sha256: String,
}

// Manifest contents and private requests deliberately have no Debug output.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prepared {
    pub format: String,
    pub definition_sha256: String,
    pub binding: Binding,
    pub approval: Approval,
    pub plan: Value,
    pub request: Request,
}

fn digest<T: Serialize>(value: &T) -> Result<String> {
    canonical::protocol_identity(&serde_json::to_value(value).map_err(|_| Error("encoding"))?)
        .map_err(|_| Error("encoding"))
}
fn field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(Error("manifest field"))
}
fn decode<T: serde::de::DeserializeOwned>(value: &Value) -> Result<T> {
    serde_json::from_value(value.clone()).map_err(|_| Error("manifest schema"))
}
pub fn slot(project: &str, run: &str) -> Result<String> {
    identity::validate_id(project).map_err(|_| Error("project"))?;
    identity::validate_id(run).map_err(|_| Error("run"))?;
    digest(&json!({"format":FORMAT,"project":project,"run_id":run})).map(|value| value[7..].into())
}
/// Preserve the original first-Attempt filename; every continuation has its own
/// immutable slot so historical reads cannot resolve the current Attempt.
pub fn attempt_slot(project: &str, run: &str, attempt: &str) -> Result<String> {
    retry::number(attempt)?;
    let original = slot(project, run)?;
    Ok(if attempt == "attempt-001" {
        original
    } else {
        format!("{original}.{attempt}")
    })
}
impl Profile {
    pub fn validate(&self) -> Result<()> {
        self.target.validate().map_err(|_| Error("target"))?;
        if [
            &self.ssh_alias,
            &self.partition,
            &self.account,
            &self.qos,
            &self.gpu_type,
        ]
        .iter()
        .any(|value| !slurm::component(value))
            || !slurm::absolute_path(&self.mount_root)
            || self.mount_root == "/"
            || !slurm::path_within(&self.run_root, &self.mount_root)
            || !slurm::path_within(&self.data_root, &self.mount_root)
            || self.run_root == self.data_root
            || self.max_gpus == 0
            || self.max_cpus == 0
            || self.max_memory_gb == 0
            || self.max_walltime_seconds == 0
            || self.max_walltime_seconds > 1000 * 3600
        {
            return Err(Error("profile"));
        }
        Ok(())
    }
}

/// Parse the actual stored manifest. Neither a client nor an outer adapter can
/// replace its argv/resources/backend with a separately supplied JobSpec.
pub fn job(run: &Value, binding: &Binding) -> Result<slurm::JobSpec> {
    Ok(slurm::JobSpec {
        run_id: binding.run_id.clone(),
        attempt_id: binding.attempt_id.clone(),
        image_id: field(run, "image_id")?.into(),
        backend: decode(&run["backend"])?,
        storage: decode(&run["storage"])?,
        resources: decode(&run["resources"])?,
        execution: decode(&run["execution"])?,
        command: decode(&run["command"])?,
    })
}
fn comparable(value: &Value) -> Result<Value> {
    let mut value = value.as_object().cloned().ok_or(Error("manifest"))?;
    for key in ["attempt_id", "created_at", "attempt_created_at"] {
        value.remove(key);
    }
    Ok(Value::Object(value))
}
pub fn definition(
    run: &Value,
    attempt: &Value,
    binding: &Binding,
    profile: &Profile,
    runtime: &Runtime,
    approval: &Approval,
) -> Result<String> {
    profile.validate()?;
    super::validate_gpu_budget(Some(approval.max_gpu_hours), "budget_cap")
        .map_err(|_| Error("budget"))?;
    if approval.reason.chars().count() > 4000 || approval.reason.chars().any(char::is_control) {
        return Err(Error("reason"));
    }
    identity::validate_id(&binding.project).map_err(|_| Error("project"))?;
    slot(&binding.project, &binding.run_id)?;
    retry::number(&binding.attempt_id)?;
    for hash in [
        &binding.project_sha256,
        &binding.run_sha256,
        &binding.attempt_sha256,
        &binding.source_receipt_sha256,
        &binding.backend_config_sha256,
    ] {
        if !helper::valid_digest(hash) {
            return Err(Error("observed hash"));
        }
    }
    for value in [run, attempt] {
        if value["schema_version"] != 1
            || value["identity_version"] != 2
            || field(value, "project")? != binding.project
            || field(value, "run_id")? != binding.run_id
            || !value.get("resume").is_none_or(Value::is_null)
        {
            return Err(Error("manifest identity"));
        }
    }
    let continuation = binding.attempt_id != "attempt-001";
    if continuation {
        let receipt = retry::validate_attempt(run, attempt, binding)?;
        if receipt.backend_id != profile.target.backend_id
            || receipt.reserved_gpu_seconds as f64 > approval.max_gpu_hours * 3600.
            || approval.max_gpu_hours > receipt.request.max_gpu_hours
        {
            return Err(Error("retry cumulative approval"));
        }
    }
    if field(attempt, "attempt_id")? != binding.attempt_id
        || run.get("attempt_id").is_some()
        || !continuation && comparable(run)? != comparable(attempt)?
        || !run
            .get("inputs")
            .is_none_or(|v| v.is_array() && v.as_array().unwrap().is_empty())
        || !run
            .get("assets")
            .is_none_or(|v| v.is_array() && v.as_array().unwrap().is_empty())
    {
        return Err(Error("unsupported first Attempt or unmaterialized input"));
    }
    crate::sources::validate_source_id(field(run, "source_id")?).map_err(|_| Error("source"))?;
    match (
        attempt.get("resume_from").filter(|v| !v.is_null()),
        attempt.get("native_checkpoint"),
    ) {
        (Some(reference), Some(value)) => {
            let reference: crate::checkpoints::Reference = decode(reference)?;
            crate::checkpoints::selected(
                &reference,
                value,
                &binding.project,
                &profile.target.storage_scope,
                &profile.run_root,
            )
            .map_err(|_| Error("checkpoint scope or identity"))?;
            if reference.run_id == binding.run_id && !continuation || runtime.worker.is_none() {
                return Err(Error("checkpoint restore"));
            }
        }
        (None, None) => {}
        _ => return Err(Error("checkpoint evidence")),
    }
    let spec = job(attempt, binding)?;
    spec.validate().map_err(|_| Error("job"))?;
    let datasets = crate::datasets::document_mounts(
        run,
        &binding.project,
        &profile.target,
        &spec.storage.project_data_root,
    )
    .map_err(|_| Error("dataset bindings"))?;
    if !datasets.is_empty() && runtime.worker.is_none() {
        return Err(Error("datasets require managed worker"));
    }
    if let Some(worker) = &runtime.worker {
        slurm::managed::plan_with_inputs(
            &Scope {
                project: binding.project.clone(),
                run_id: binding.run_id.clone(),
                attempt_id: binding.attempt_id.clone(),
            },
            &spec,
            worker,
            attempt.get("native_checkpoint"),
            &datasets,
        )
        .map_err(|_| Error("managed worker runtime or environment"))?;
    }
    let gpu = slurm::parse_gres(&spec.backend.gres).map_err(|_| Error("GPU request"))?;
    let seconds = slurm::walltime_seconds(&spec.backend.time).map_err(|_| Error("walltime"))?;
    if spec.storage.run_dir
        != format!(
            "{}/{}/{}",
            profile.run_root.trim_end_matches('/'),
            binding.project,
            binding.run_id
        )
        || spec.storage.project_data_root
            != format!(
                "{}/{}",
                profile.data_root.trim_end_matches('/'),
                binding.project
            )
        || [
            &spec.backend.apptainer_cache_dir,
            &spec.backend.apptainer_tmp_dir,
        ]
        .into_iter()
        .any(|v| {
            v.as_deref()
                .is_some_and(|v| !slurm::path_within(v, &profile.mount_root))
        })
        || spec.backend.ssh_alias != profile.ssh_alias
        || spec.backend.partition != profile.partition
        || spec.backend.account != profile.account
        || spec.backend.qos != profile.qos
        || spec.backend.mount_root != profile.mount_root
        || gpu.gpu_type != profile.gpu_type
        || spec.resources.gpus > profile.max_gpus
        || spec.resources.cpus > profile.max_cpus
        || spec
            .resources
            .memory_gb
            .is_none_or(|v| v > profile.max_memory_gb)
        || seconds > profile.max_walltime_seconds
        || seconds as f64 * spec.resources.gpus as f64 > approval.max_gpu_hours * 3600.0
    {
        return Err(Error("backend policy or GPU budget"));
    }
    if runtime.source_id != field(run, "source_id")?
        || runtime.image_id != spec.image_id
        || runtime.sif_path != spec.backend.sif_path
        || !helper::valid_digest(&runtime.sif_sha256)
        || spec.backend.oci_image.as_deref() != Some(&runtime.oci_image)
        || field(&run["container"], "runtime_id")? != runtime.runtime_id
        || !runtime
            .runtime_id
            .strip_prefix("runtime.")
            .is_some_and(|v| {
                v.len() == 64
                    && v.bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            })
    {
        return Err(Error("published runtime"));
    }
    digest(
        &json!({"format":FORMAT,"binding":binding,"profile":profile,"runtime":runtime,"approval":approval,
        "run":run,"attempt":attempt}),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prepare(
    run: &Value,
    attempt: &Value,
    binding: Binding,
    profile: &Profile,
    runtime: &Runtime,
    approval: Approval,
    token: &str,
    now: &str,
    expires: &str,
) -> Result<Prepared> {
    let definition_sha256 = definition(run, attempt, &binding, profile, runtime, &approval)?;
    let start = OffsetDateTime::parse(now, &Rfc3339).map_err(|_| Error("time"))?;
    let end = OffsetDateTime::parse(expires, &Rfc3339).map_err(|_| Error("time"))?;
    if end <= start || end - start > time::Duration::hours(24) {
        return Err(Error("gate window"));
    }
    let manifest =
        String::from_utf8(canonical::python_json_bytes(attempt).map_err(|_| Error("encoding"))?)
            .map_err(|_| Error("encoding"))?;
    let spec = job(attempt, &binding)?;
    let datasets = crate::datasets::document_mounts(
        run,
        &binding.project,
        &profile.target,
        &spec.storage.project_data_root,
    )
    .map_err(|_| Error("dataset bindings"))?;
    let scope = Scope {
        project: binding.project.clone(),
        run_id: binding.run_id.clone(),
        attempt_id: binding.attempt_id.clone(),
    };
    let mut launch_spec = spec.clone();
    if let Some(worker) = &runtime.worker {
        let package = slurm::managed::plan_for_document(&scope, &spec, worker, attempt, &datasets)
            .map_err(|_| Error("managed launch"))?;
        launch_spec.command = slurm::managed::command(&package, worker)
            .map_err(|_| Error("managed launch command"))?;
    }
    let script = slurm::render_job_with_datasets(&launch_spec, Some(token), &datasets)
        .map_err(|_| Error("script"))?;
    let attempt_path = format!(
        "{}/attempts/{}/attempt.json",
        spec.storage.run_dir, binding.attempt_id
    );
    let mut environment = String::new();
    for (key, value) in [
        ("PROJECT_NAME", binding.project.as_str()),
        ("RUN_ID", binding.run_id.as_str()),
        ("ATTEMPT_ID", binding.attempt_id.as_str()),
        ("DATA_DIR", spec.storage.project_data_root.as_str()),
    ] {
        environment.push_str(&format!(
            "export {key}={}\n",
            slurm::shell_quote(value).map_err(|_| Error("execution environment"))?
        ));
    }
    if spec.execution.managed_io {
        environment.push_str("export INPUT_DIR=/inputs\nexport OUTPUT_DIR=/outputs\n");
    }
    let verify = format!(
        "set -euo pipefail\n{environment}if ! printf '%s  %s\\n' {} {} | sha256sum --check --status; then\n  printf '%s\\n' 'ML_EXPD_STAGING=FAILED code=ATTEMPT_VERIFICATION_FAILED' >&2\n  exit 78\nfi\nprintf '%s\\n' ML_EXPD_STAGING=READY\nif ! printf '%s  %s\\n' {} {} | sha256sum --check --status; then\n  printf '%s\\n' 'ML_EXPD_RUNTIME=FAILED code=SIF_VERIFICATION_FAILED' >&2\n  exit 78\nfi\nprintf '%s\\n' ML_EXPD_RUNTIME=VERIFIED\n",
        slurm::shell_quote(&helper::sha256_bytes(manifest.as_bytes())[7..])
            .map_err(|_| Error("Attempt hash"))?,
        slurm::shell_quote(&attempt_path).map_err(|_| Error("Attempt path"))?,
        slurm::shell_quote(&runtime.sif_sha256[7..]).map_err(|_| Error("SIF hash"))?,
        slurm::shell_quote(&runtime.sif_path).map_err(|_| Error("SIF path"))?
    );
    let script = script.replacen("set -euo pipefail\n", &verify, 1);
    let request = Request::prepare(
        FrozenIntent {
            target: profile.target.clone(),
            staging: Some(slurm::staging::Requirement {
                runs_root: profile.run_root.clone(),
                worker: runtime.worker.clone(),
            }),
            scope,
            submission_token: token.into(),
            scheduler_name: slurm::scheduler_job_name(&binding.run_id, &binding.attempt_id)
                .map_err(|_| Error("scheduler name"))?,
            manifest_sha256: helper::sha256_bytes(manifest.as_bytes()),
            script_sha256: helper::sha256_bytes(script.as_bytes()),
        },
        manifest,
        script,
    )
    .map_err(|_| Error("helper request"))?;
    let plan=slurm::action::prepare(json!({"operation":"SUBMIT_RUN","scope":{"project":binding.project,"scope_type":"run","object_id":binding.run_id},
        "run_id":binding.run_id,"attempt_id":binding.attempt_id,"execution_campaign_sha256":binding.run_sha256,
        "execution_manifest_sha256":request.intent().manifest_sha256,"ready":true,"gates":[],"created_at":now,"gate_expires_at":expires,
        "native_admission":{"format":FORMAT,"definition_sha256":definition_sha256,"binding":binding,"approval":approval},
        "preflight_summary":{"backend":profile.target.backend_id,"resources":run["resources"],"time":run["backend"]["time"],"max_gpu_hours":approval.max_gpu_hours},
    }),&request).map_err(|_|Error("Action"))?;
    Ok(Prepared {
        format: FORMAT.into(),
        definition_sha256,
        binding,
        approval,
        plan,
        request,
    })
}
impl Prepared {
    pub fn validate(&self) -> Result<()> {
        if self.format != FORMAT
            || !helper::valid_digest(&self.definition_sha256)
            || self.plan["native_admission"]
                != json!({"format":FORMAT,"definition_sha256":self.definition_sha256,"binding":self.binding,"approval":self.approval})
            || self.plan["scope"]["project"] != self.binding.project
            || self.plan["run_id"] != self.binding.run_id
            || self.plan["attempt_id"] != self.binding.attempt_id
        {
            return Err(Error("prepared binding"));
        }
        slurm::action::validate(&self.plan, &self.request).map_err(|_| Error("prepared Action"))
    }
}
