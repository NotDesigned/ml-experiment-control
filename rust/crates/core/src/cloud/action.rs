//! Bind cloud effects to the ordinary reviewed Action lifecycle. Provider
//! requests cannot obtain authority from a bare job name or HTTP response.
use super::*;
use crate::{
    first_submission::native::{Approval, Binding},
    operations::{self, Scope, ScopeType, Transition},
};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
const FORMAT: &str = "native-cloud-action.v1";
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Execution {
    pub backend_id: String,
    pub profile_sha256: String,
    pub binding: Binding,
    pub job: Job,
    pub allocation: pai::Allocation,
}
impl Execution {
    pub fn package(&self) -> Result<super::launch::Package> {
        if self.job.command.len() != 4
            || self.job.command[1] != "cloud-launch"
            || self.job.command[2] != "--package-json"
        {
            return Err(Error("sealed cloud worker command"));
        }
        let package: super::launch::Package = serde_json::from_str(&self.job.command[3])
            .map_err(|_| Error("cloud worker package"))?;
        package.validate()?;
        let b = package
            .config
            .binding()
            .map_err(|_| Error("cloud worker binding"))?;
        if b.identity.project != self.job.project
            || b.identity.run_id != self.job.run_id
            || b.identity.attempt_id != self.job.attempt_id
        {
            return Err(Error("cloud worker scope"));
        }
        Ok(package)
    }
    pub fn validate(&self) -> Result<()> {
        self.job.validate()?;
        identity::validate_id(&self.backend_id).map_err(|_| Error("backend ID"))?;
        let b = &self.binding;
        if b.project != self.job.project
            || b.run_id != self.job.run_id
            || b.attempt_id != self.job.attempt_id
            || [
                &b.project_sha256,
                &b.run_sha256,
                &b.attempt_sha256,
                &b.source_receipt_sha256,
                &b.backend_config_sha256,
                &self.profile_sha256,
            ]
            .into_iter()
            .any(|s| !helper::valid_digest(s))
            || b.backend_config_sha256 != self.profile_sha256
        {
            return Err(Error("frozen cloud authority"));
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        digest(&serde_json::to_value(self).map_err(|_| Error("execution encoding"))?)
    }
}
fn scope(execution: &Execution) -> Scope {
    Scope {
        project: execution.job.project.clone(),
        scope_type: ScopeType::Run,
        object_id: execution.job.run_id.clone(),
    }
}
fn body(plan: &Value) -> Result<Value> {
    let mut value = plan.as_object().cloned().ok_or(Error("Action object"))?;
    for key in [
        "action_id",
        "request_digest",
        "intent_digest",
        "gate_bundle_digest",
        "execution",
        "journal",
    ] {
        value.remove(key);
    }
    Ok(Value::Object(value))
}
pub fn prepare(
    execution: &Execution,
    approval: &Approval,
    now: &str,
    expires: &str,
) -> Result<Value> {
    execution.validate()?;
    let start = OffsetDateTime::parse(now, &Rfc3339).map_err(|_| Error("Action time"))?;
    let end = OffsetDateTime::parse(expires, &Rfc3339).map_err(|_| Error("Action expiry"))?;
    if end <= start
        || end - start > time::Duration::hours(24)
        || !approval.max_gpu_hours.is_finite()
        || approval.max_gpu_hours <= 0.
        || execution.job.gpus as f64 * execution.job.max_time_seconds as f64
            > approval.max_gpu_hours * 3600.
    {
        return Err(Error("GPU budget or gate window"));
    }
    let binding = &execution.binding;
    let mut plan = json!({"operation":"SUBMIT_RUN","scope":scope(execution),"run_id":execution.job.run_id,"attempt_id":execution.job.attempt_id,
        "execution_campaign_sha256":binding.run_sha256,"execution_manifest_sha256":binding.attempt_sha256,"ready":true,"gates":[],"created_at":now,"gate_expires_at":expires,
        "native_cloud":{"format":FORMAT,"execution_sha256":execution.digest()?,"binding":binding,"approval":approval},
        "preflight_summary":{"backend":execution.backend_id,"resources":{"gpus":execution.job.gpus,"cpus":execution.job.cpus,"memory_gb":execution.job.memory_gb},"walltime_seconds":execution.job.max_time_seconds,"max_gpu_hours":approval.max_gpu_hours}});
    let intent = digest(&body(&plan)?)?;
    plan["action_id"] = json!(operations::action_id(&scope(execution), &intent));
    plan["request_digest"] = json!(intent);
    plan["intent_digest"] = json!(intent);
    plan["gate_bundle_digest"] = json!(digest(
        &json!({"intent_digest":intent,"gates":plan["gates"]})
    )?);
    validate(&plan, execution)?;
    Ok(plan)
}
pub fn validate(plan: &Value, execution: &Execution) -> Result<()> {
    execution.validate()?;
    let binding = &execution.binding;
    if plan["operation"] != "SUBMIT_RUN"
        || plan["scope"]
            != serde_json::to_value(scope(execution)).map_err(|_| Error("Action scope"))?
        || plan["run_id"] != execution.job.run_id
        || plan["attempt_id"] != execution.job.attempt_id
        || plan["execution_campaign_sha256"] != binding.run_sha256
        || plan["execution_manifest_sha256"] != binding.attempt_sha256
        || plan["native_cloud"]["format"] != FORMAT
        || plan["native_cloud"]["execution_sha256"] != execution.digest()?
        || plan["native_cloud"]["binding"]
            != serde_json::to_value(binding).map_err(|_| Error("Action binding"))?
    {
        return Err(Error("Action binding"));
    }
    let approval: Approval = serde_json::from_value(plan["native_cloud"]["approval"].clone())
        .map_err(|_| Error("approval"))?;
    if !approval.max_gpu_hours.is_finite()
        || approval.max_gpu_hours <= 0.
        || execution.job.gpus as f64 * execution.job.max_time_seconds as f64
            > approval.max_gpu_hours * 3600.
    {
        return Err(Error("GPU budget"));
    }
    let start = plan["created_at"]
        .as_str()
        .and_then(|s| OffsetDateTime::parse(s, &Rfc3339).ok())
        .ok_or(Error("Action time"))?;
    let end = plan["gate_expires_at"]
        .as_str()
        .and_then(|s| OffsetDateTime::parse(s, &Rfc3339).ok())
        .ok_or(Error("Action expiry"))?;
    if end <= start
        || end - start > time::Duration::hours(24)
        || plan["ready"] != true
        || !plan["gates"].is_array()
    {
        return Err(Error("Action gates"));
    }
    let intent = digest(&body(plan)?)?;
    if plan["request_digest"] != intent
        || plan["intent_digest"] != intent
        || plan["action_id"] != operations::action_id(&scope(execution), &intent)
        || plan["gate_bundle_digest"]
            != digest(&json!({"intent_digest":intent,"gates":plan["gates"]}))?
        || serde_json::to_string(plan)
            .map_err(|_| Error("Action encoding"))?
            .contains(&execution.job.submission_token)
    {
        return Err(Error("Action digest"));
    }
    Ok(())
}
pub enum Outcome {
    Owned {
        job_id: String,
        observation: Box<JobObservation>,
    },
    Uncertain {
        job_id: Option<String>,
    },
}
pub fn finish(
    snapshot: &Value,
    execution: &Execution,
    outcome: Outcome,
    now: &str,
) -> Result<Transition> {
    validate(snapshot, execution)?;
    OffsetDateTime::parse(now, &Rfc3339).map_err(|_| Error("completion time"))?;
    let current = snapshot["execution"]
        .as_object()
        .ok_or(Error("execution state"))?;
    let expected = current["status"]
        .as_str()
        .filter(|s| matches!(*s, "EXECUTING" | "RECONCILE_REQUIRED"))
        .ok_or(Error("execution phase"))?;
    let mut result = match current.get("result") {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(map)) => map.clone(),
        _ => return Err(Error("execution result")),
    };
    let previous = result
        .get("submission")
        .map(|s| {
            if s["backend"] != execution.backend_id {
                return Err(Error("previous backend"));
            }
            s["backend_job_id"]
                .as_str()
                .map(str::to_owned)
                .ok_or(Error("previous job"))
        })
        .transpose()?;
    let (id, observation, verified) = match outcome {
        Outcome::Owned {
            job_id,
            observation,
        } => {
            if job_id != observation.job_id {
                return Err(Error("observation job"));
            }
            (Some(job_id), Some(observation), true)
        }
        Outcome::Uncertain { job_id } => (job_id, None, false),
    };
    let id = id.or_else(|| previous.clone());
    if let Some(id) = id {
        component(&id)?;
        if previous.is_some_and(|p| p != id) {
            return Err(Error("job changed"));
        }
        result.insert(
            "submission".into(),
            json!({"backend":execution.backend_id,"backend_job_id":id}),
        );
    }
    if let Some(observation) = observation {
        result.insert(
            "observation".into(),
            serde_json::to_value(observation).map_err(|_| Error("observation"))?,
        );
    }
    let mut next = current.clone();
    let status = if verified {
        "VERIFIED"
    } else {
        "RECONCILE_REQUIRED"
    };
    next.extend(json!({"status":status,"error":if verified{None}else{Some("cloud submission is unconfirmed; query the original effect")},"result":result,"safe_to_retry":false,
        "resolution":if verified{"APPLIED"}else{"UNKNOWN_DO_NOT_RETRY"},"next_action":if verified{"OBSERVE_RESULT"}else{"RECONCILE"},"last_reconciled_at":now}).as_object().ok_or(Error("execution result"))?.clone());
    if verified {
        next.insert("finished_at".into(), json!(now));
    }
    Ok(Transition {
        execution: Value::Object(next),
        event: json!({"timestamp":now,"event":if verified{"native_cloud_submission_verified"}else{"native_cloud_submission_uncertain"},"payload":{"status":status,"execution_sha256":execution.digest()?}}),
        expected_status: Some(expected.into()),
    })
}
