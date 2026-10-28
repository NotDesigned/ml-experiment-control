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
    if !matches!(
        run["evidence"]["scheduler"]["state"].as_str(),
        None | Some("UNKNOWN" | "NOT_SUBMITTED")
    ) {
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
    run["evidence"]["scheduler"] = json!({"state":state,"attempt_id":submission["attempt_id"],"detail":detail,"as_of":at,
        "source":"native-action-commit","stale":true,"freshness":if at.is_null(){"UNKNOWN"}else{"STALE"},
        "stale_reason":"committed observation; no provider refresh on GET","evidence_availability":{"availability":"PRESENT","source":"native-action-commit","submission_id":id}});
    Ok(run)
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
    if binding.attempt_id != "attempt-001" {
        return Err(Error("first Attempt"));
    }
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
            || !value.get("resume_from").is_none_or(Value::is_null)
            || !value.get("resume").is_none_or(Value::is_null)
        {
            return Err(Error("manifest identity"));
        }
    }
    if field(attempt, "attempt_id")? != binding.attempt_id
        || run.get("attempt_id").is_some()
        || comparable(run)? != comparable(attempt)?
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
    let spec = job(run, binding)?;
    spec.validate().map_err(|_| Error("job"))?;
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
    let script =
        slurm::render_job(&job(run, &binding)?, Some(token)).map_err(|_| Error("script"))?;
    let verify = format!(
        "set -euo pipefail\nif ! printf '%s  %s\\n' {} {} | sha256sum --check --status; then\n  printf '%s\\n' 'ML_EXPD_RUNTIME=FAILED code=SIF_VERIFICATION_FAILED' >&2\n  exit 78\nfi\nprintf '%s\\n' ML_EXPD_RUNTIME=VERIFIED\n",
        slurm::shell_quote(&runtime.sif_sha256[7..]).map_err(|_| Error("SIF hash"))?,
        slurm::shell_quote(&runtime.sif_path).map_err(|_| Error("SIF path"))?
    );
    let script = script.replacen("set -euo pipefail\n", &verify, 1);
    let request = Request::prepare(
        FrozenIntent {
            target: profile.target.clone(),
            scope: Scope {
                project: binding.project.clone(),
                run_id: binding.run_id.clone(),
                attempt_id: binding.attempt_id.clone(),
            },
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
