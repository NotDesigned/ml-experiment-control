//! Backend-neutral cloud Run authorship. Provider configuration and runtime
//! receipts are trusted inputs, never fields supplied by the HTTP caller.
use super::*;
pub use crate::first_submission::native::creation::Request as Create;
use crate::first_submission::native::{
    Binding,
    creation::{Documents, Request},
};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    pub runtime_id: String,
    pub source_id: String,
    pub image: String,
    pub source_dir: String,
    pub worker_path: String,
    pub worker_sha256: String,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    pub worker_features: std::collections::BTreeSet<String>,
}
impl Runtime {
    pub fn validate(&self) -> Result<()> {
        if self
            .worker_features
            .iter()
            .any(|f| !matches!(f.as_str(), "storage-layout.v1" | "script-preparation.v1"))
        {
            return Err(Error("unsupported worker feature"));
        }
        crate::sources::validate_source_id(&self.source_id).map_err(|_| Error("source ID"))?;
        if !self
            .runtime_id
            .strip_prefix("runtime.")
            .is_some_and(|s| helper::valid_digest(&format!("sha256:{s}")))
            || !helper::valid_digest(&self.worker_sha256)
        {
            return Err(Error("runtime receipt"));
        }
        absolute_path(&self.source_dir)?;
        absolute_path(&self.worker_path)?;
        Job {
            project: "check".into(),
            run_id: "check".into(),
            attempt_id: "attempt-001".into(),
            submission_token: "0".repeat(32),
            source_id: self.source_id.clone(),
            image: self.image.clone(),
            command: vec!["true".into()],
            gpus: 0,
            cpus: 1,
            memory_gb: 1,
            max_time_seconds: 1,
            priority: 5,
        }
        .validate()
    }
}
pub struct Authority<'a> {
    pub project_sha256: &'a str,
    pub source_receipt_sha256: &'a str,
    pub profile_sha256: &'a str,
    pub tracking_route: &'a Value,
    pub target: &'a helper::Target,
    pub run_root: &'a str,
    pub data_root: &'a str,
    pub datasets: &'a [crate::datasets::Receipt],
    pub checkpoint: Option<&'a Value>,
}
pub fn documents(
    project: &str,
    request: &Request,
    runtime: &Runtime,
    authority: Authority<'_>,
    now: &str,
) -> Result<Documents> {
    request
        .validate(project)
        .map_err(|_| Error("Run request"))?;
    runtime.validate()?;
    time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
        .map_err(|_| Error("Run time"))?;
    if request.runtime_id != runtime.runtime_id
        || [
            authority.project_sha256,
            authority.source_receipt_sha256,
            authority.profile_sha256,
        ]
        .into_iter()
        .any(|s| !helper::valid_digest(s))
    {
        return Err(Error("cloud Run authority or unsupported input"));
    }
    let datasets = crate::datasets::select(
        &request.data,
        authority.datasets,
        project,
        authority.target,
        authority.data_root,
    )
    .map_err(|_| Error("cloud dataset scope"))?;
    match (&request.resume_from, authority.checkpoint) {
        (None, None) => {}
        (Some(reference), Some(value)) => {
            crate::checkpoints::selected_at(
                reference,
                value,
                project,
                &authority.target.storage_scope,
                &format!(
                    "{}/{}/{}/{}/state",
                    authority.run_root, project, reference.run_id, reference.attempt_id
                ),
            )
            .map_err(|_| Error("cloud checkpoint scope"))?;
        }
        _ => return Err(Error("cloud restore evidence")),
    }
    let mut protocol = json!({"source_id":runtime.source_id,"image":runtime.image,"runtime_id":runtime.runtime_id,
        "command":request.command,"parameters":request.parameters,"seed":request.seed,"evaluation":request.evaluation,"metrics_schema":request.metrics_schema});
    if !request.data.is_empty() {
        protocol["data"] = json!(request.data);
    }
    if !request.env.is_empty() {
        protocol["env"] = json!(request.env);
    }
    if let Some(preparation) = &request.data_preparation {
        protocol["data_preparation"] = json!(preparation);
    }
    if let Some(reference) = &request.resume_from {
        protocol["resume_from"] = json!(reference);
    }
    let protocol_id = digest(&protocol)?;
    // Validate arbitrary project schemas without inventing scientific metrics.
    crate::metrics::MetricSchema::from_value(&request.metrics_schema.to_value())
        .map_err(|_| Error("metric schema"))?;
    let mut run = json!({"schema_version":1,"identity_version":2,"project":project,"run_id":request.run_id,"created_at":now,
        "source_id":runtime.source_id,"image_id":runtime.image,"config_path":"request.json","seed":request.seed,"campaign":format!("run-{}",request.run_id),
        "parameters":request.parameters,"backend":{"kind":"cloud","executor":request.executor},"resources":request.resources,
        "container":{"runtime_id":runtime.runtime_id},"command":request.command,"inputs":[],"assets":[],"native_cloud":{"walltime_seconds":request.walltime_seconds},
        "evaluation":{"protocol":protocol,"protocol_id":protocol_id,"metrics_schema":request.metrics_schema},"tracking_route":authority.tracking_route,
        "native_creation":{"format":"native-cloud-run.v1","request_sha256":request.identity(project).map_err(|_|Error("Run request hash"))?,
            "project_sha256":authority.project_sha256,"source_receipt_sha256":authority.source_receipt_sha256,"backend_config_sha256":authority.profile_sha256,
            "runtime_sha256":digest(&json!(runtime))?}});
    if !datasets.is_empty() {
        run["native_datasets"] = json!(datasets);
    }
    if let Some(checkpoint) = authority.checkpoint {
        run["resume_from"] = json!(request.resume_from);
        run["native_checkpoint"] = checkpoint.clone();
    }
    run["native_cloud"]["data_root"] = json!(authority.data_root);
    run["native_cloud"]["target"] = json!(authority.target);
    if let Some(wandb) = &request.wandb {
        run["wandb"] = wandb.to_value();
    }
    if let Some(policy) = &request.artifacts {
        run["artifacts"] = json!(policy);
    }
    if !request.env.is_empty() {
        run["env"] = json!(request.env);
    }
    if let Some(preparation) = &request.data_preparation {
        run["data_preparation"] = json!(preparation);
    }
    if let Some(policy) = &request.scheduling {
        run["native_cloud"]["scheduling"] = json!(policy);
    }
    let mut attempt = run.clone();
    attempt["attempt_id"] = json!("attempt-001");
    Ok(Documents {
        run,
        attempt,
        status: json!({"project":project,"run_id":request.run_id,"attempt_id":"attempt-001","state":"NOT_SUBMITTED","updated_at":now}),
        request: json!(request),
    })
}
pub fn execution(
    run: &Value,
    attempt: &Value,
    binding: Binding,
    runtime: &Runtime,
    root: &str,
    token: &str,
    allocation: pai::Allocation,
) -> Result<action::Execution> {
    runtime.validate()?;
    let project = &binding.project;
    let run_id = &binding.run_id;
    let attempt_id = &binding.attempt_id;
    let mut parent = attempt.clone();
    parent
        .as_object_mut()
        .ok_or(Error("Attempt object"))?
        .remove("attempt_id");
    if run["native_creation"]["format"] != "native-cloud-run.v1"
        || run["project"] != *project
        || run["run_id"] != *run_id
        || attempt["attempt_id"] != *attempt_id
        || helper::sha256_bytes(&serde_json::to_vec(run).map_err(|_| Error("Run encoding"))?)
            != binding.run_sha256
        || helper::sha256_bytes(
            &serde_json::to_vec(attempt).map_err(|_| Error("Attempt encoding"))?,
        ) != binding.attempt_sha256
        || run["native_creation"]["project_sha256"] != binding.project_sha256
        || run["native_creation"]["source_receipt_sha256"] != binding.source_receipt_sha256
        || (attempt_id == "attempt-001" && parent != *run)
        || run["source_id"] != runtime.source_id
        || run["image_id"] != runtime.image
        || run["container"]["runtime_id"] != runtime.runtime_id
        || run["native_creation"]["backend_config_sha256"] != binding.backend_config_sha256
        || run["native_creation"]["runtime_sha256"] != digest(&json!(runtime))?
    {
        return Err(Error("frozen cloud Run"));
    }
    if attempt_id != "attempt-001" {
        super::retry::validate_attempt(run, attempt, &binding)?;
    }
    let request:Request=serde_json::from_value(json!({"run_id":run_id,"runtime_id":runtime.runtime_id,"executor":run["backend"]["executor"],"command":run["command"],"resources":run["resources"],"walltime_seconds":run["native_cloud"]["walltime_seconds"],"seed":run["seed"]})).map_err(|_|Error("cloud Run resources"))?;
    let mut package = super::launch::compile(super::launch::Input {
        root,
        project,
        run: run_id,
        attempt: attempt_id,
        workdir: &runtime.source_dir,
        argv: &request.command,
        timeout_seconds: request.walltime_seconds,
        worker_sha256: &runtime.worker_sha256,
    })?;
    package.manifest.environment.extend(
        crate::environment::from_document(run).map_err(|_| Error("frozen Run environment"))?,
    );
    package.checkpoint_context = Some(crate::checkpoints::Context {
        project: project.clone(),
        run_id: run_id.clone(),
        attempt_id: attempt_id.clone(),
        source_id: runtime.source_id.clone(),
        image_id: runtime.image.clone(),
        storage_scope: format!(
            "sha256:{}",
            crate::canonical::digest(
                &json!({"profile":binding.backend_config_sha256,"run_root":root})
            )
            .map_err(|_| Error("checkpoint storage"))?
        ),
        state_root: package
            .config
            .attempt_root
            .join("state")
            .to_str()
            .ok_or(Error("checkpoint state path"))?
            .into(),
    });
    package.artifacts = run
        .get("artifacts")
        .map(|p| serde_json::from_value(p.clone()).map_err(|_| Error("artifact policy")))
        .transpose()?;
    let datasets = match run.get("native_datasets") {
        None => Vec::new(),
        Some(v) => serde_json::from_value(v.clone()).map_err(|_| Error("frozen cloud datasets"))?,
    };
    package.attach(
        run["native_cloud"]["data_root"]
            .as_str()
            .ok_or(Error("frozen data root"))?,
        datasets,
        attempt.get("native_checkpoint").cloned(),
    )?;
    if let Some(value) = run.get("data_preparation") {
        let plan = crate::data_preparation::Plan {
            contract: crate::data_preparation::CONTRACT.into(),
            request: serde_json::from_value(value.clone())
                .map_err(|_| Error("frozen data preparation"))?,
            source_id: runtime.source_id.clone(),
            image: runtime.image.clone(),
            cache_root: format!(
                "{}/{}",
                run["native_cloud"]["data_root"]
                    .as_str()
                    .ok_or(Error("frozen data root"))?,
                project
            ),
            mounted_root: run["native_cloud"]["storage"]["mount"]
                .as_str()
                .ok_or(Error("data preparation mount"))?
                .into(),
            workdir: runtime.source_dir.clone(),
            env: crate::environment::from_document(run).map_err(|_| Error("frozen environment"))?,
            require_cached: run["native_cloud"]["data_preparation_before_job"] == true,
        };
        crate::managed::attach_preparation(&mut package.config, &mut package.manifest, plan)
            .map_err(|_| Error("data preparation launch"))?;
    }
    let policy: Option<crate::first_submission::native::creation::Scheduling> = run["native_cloud"]
        .get("scheduling")
        .map(|v| serde_json::from_value(v.clone()).map_err(|_| Error("frozen scheduling policy")))
        .transpose()?;
    if policy
        .as_ref()
        .is_some_and(|p| !(1..=9).contains(&p.priority))
    {
        return Err(Error("frozen scheduling priority"));
    }
    let job = Job {
        project: project.clone(),
        run_id: run_id.clone(),
        attempt_id: attempt_id.clone(),
        submission_token: token.into(),
        source_id: runtime.source_id.clone(),
        image: runtime.image.clone(),
        command: package.command(&runtime.worker_path)?,
        gpus: request.resources.gpus,
        cpus: request.resources.cpus,
        memory_gb: request.resources.memory_gb.ok_or(Error("memory"))?,
        max_time_seconds: request.walltime_seconds,
        priority: policy.as_ref().map(|p| p.priority).unwrap_or(5),
    };
    let result = action::Execution {
        backend_id: request.executor,
        profile_sha256: binding.backend_config_sha256.clone(),
        binding,
        job,
        allocation: policy.map(|p| p.allocation).unwrap_or(allocation),
    };
    result.validate()?;
    Ok(result)
}
