//! Pure authorship of a standalone Run and its first Attempt. The caller owns
//! approved source/runtime observations, time and atomic directory publication.
use super::{Approval, Binding, Error, Profile, Result, Runtime, definition};
use crate::{canonical, identity, metrics::MetricSchema, slurm};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

pub const MAX_REQUEST_BYTES: usize = 64 * 1024;
/// A refusal the caller can act on, so it keeps its own words at the HTTP boundary.
pub const CODE_DELIVERY_UNSUPPORTED: &str =
    "code_source_id is not supported on a Slurm executor yet; keep the code in the image";
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scheduling {
    #[serde(default)]
    pub allocation: crate::cloud::pai::Allocation,
    #[serde(default = "default_priority")]
    pub priority: u8,
}
fn default_priority() -> u8 {
    5
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub run_id: String,
    pub runtime_id: String,
    pub executor: String,
    pub command: Vec<String>,
    pub resources: slurm::Resources,
    pub walltime_seconds: u64,
    pub seed: i64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_preparation: Option<crate::data_preparation::Request>,
    #[serde(default)]
    pub parameters: Map<String, Value>,
    #[serde(default)]
    pub evaluation: Map<String, Value>,
    #[serde(default)]
    pub metrics_schema: MetricSchema,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_from: Option<crate::checkpoints::Reference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<crate::artifacts::publication::Policy>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data: Vec<crate::datasets::Input>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wandb: Option<crate::tracking::WandbOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduling: Option<Scheduling>,
    /// Code delivered to each Attempt instead of baked into the image. Absent, the
    /// Run works from the image and its identity is what it always was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_source_id: Option<String>,
}
impl Request {
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(Error("Run request size"));
        }
        let value = crate::json::mapping(bytes).map_err(|_| Error("Run request JSON"))?;
        serde_json::from_value(Value::Object(value)).map_err(|_| Error("Run request schema"))
    }
    pub fn validate(&self, project: &str) -> Result<()> {
        crate::environment::validate(&self.env).map_err(|_| Error("Run environment"))?;
        if let Some(code) = &self.code_source_id {
            crate::sources::validate_source_id(code).map_err(|_| Error("code source"))?;
            // A preparation job may run before any Attempt exists, so it has no
            // delivered code to read: its script stays in the image.
            if self.data_preparation.is_some() {
                return Err(Error("data_preparation needs its code in the image"));
            }
            // Reserved only here: old stored Runs may hold these names.
            if self.env.contains_key("SOURCE_DIR")
                || self.env.contains_key("PYTHONDONTWRITEBYTECODE")
            {
                return Err(Error(
                    "SOURCE_DIR and PYTHONDONTWRITEBYTECODE are set by the platform for delivered code",
                ));
            }
        }
        if let Some(preparation) = &self.data_preparation {
            preparation
                .validate()
                .map_err(|_| Error("data preparation"))?;
        }
        if self
            .scheduling
            .as_ref()
            .is_some_and(|s| !(1..=9).contains(&s.priority))
        {
            return Err(Error("scheduling priority"));
        }
        crate::datasets::inputs(&self.data).map_err(|_| Error("dataset inputs"))?;
        if let Some(policy) = &self.artifacts {
            policy.validate().map_err(|_| Error("artifact policy"))?;
        }
        if let Some(reference) = &self.resume_from {
            // The server fills a left-out Attempt before the Run is frozen, and the
            // restore check refuses any reference that is still unresolved.
            reference
                .validate_received()
                .map_err(|_| Error("checkpoint reference"))?;
            if reference.run_id == self.run_id {
                return Err(Error("checkpoint source is the new Run"));
            }
        }
        identity::validate_id(project).map_err(|_| Error("project"))?;
        identity::validate_id(&self.run_id).map_err(|_| Error("run_id"))?;
        identity::validate_id(&self.executor).map_err(|_| Error("executor"))?;
        if !self.runtime_id.strip_prefix("runtime.").is_some_and(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }) || self.command.is_empty()
            || self.command.len() > 128
            || self.command[0].is_empty()
            || self
                .command
                .iter()
                .any(|v| v.len() > 8192 || v.contains('\0'))
            || self.resources.memory_gb.is_none()
            || self.walltime_seconds == 0
            || self.walltime_seconds > 1000 * 3600
            || serde_json::to_vec(self)
                .map_err(|_| Error("Run request encoding"))?
                .len()
                > MAX_REQUEST_BYTES
        {
            return Err(Error("Run request bounds"));
        }
        self.resources
            .validate_compute()
            .map_err(|_| Error("Run resources"))
    }
    pub fn identity(&self, project: &str) -> Result<String> {
        self.validate(project)?;
        canonical::protocol_identity(&json!({"project":project,"request":self}))
            .map_err(|_| Error("Run request identity"))
    }
}
pub struct Documents {
    pub run: Value,
    pub attempt: Value,
    pub status: Value,
    pub request: Value,
}
pub struct Authority<'a> {
    pub project_sha256: &'a str,
    pub source_receipt_sha256: &'a str,
    pub backend_config_sha256: &'a str,
    pub checkpoint: Option<&'a Value>,
    pub datasets: &'a [crate::datasets::Receipt],
    pub tracking_route: Option<&'a Value>,
}
pub fn plan(
    project: &str,
    request: &Request,
    profile: &Profile,
    runtime: &Runtime,
    authority: Authority<'_>,
    now: &str,
) -> Result<Documents> {
    request.validate(project)?;
    if request.scheduling.is_some() {
        return Err(Error("cloud scheduling requires a cloud executor"));
    }
    if request.code_source_id.is_some() {
        return Err(Error(CODE_DELIVERY_UNSUPPORTED));
    }
    OffsetDateTime::parse(now, &Rfc3339).map_err(|_| Error("creation time"))?;
    if request.executor != profile.target.backend_id || request.runtime_id != runtime.runtime_id {
        return Err(Error("Run runtime or executor"));
    }
    let command = request.command.clone();
    let datasets = crate::datasets::select(
        &request.data,
        authority.datasets,
        project,
        &profile.target,
        &profile.data_root,
    )
    .map_err(|_| Error("dataset receipt scope"))?;
    crate::datasets::manifest_budget(&datasets)
        .map_err(|_| Error(crate::datasets::ManifestsTooLarge::CAUSE))?;
    let spec = slurm::JobSpec {
        run_id: request.run_id.clone(),
        attempt_id: "attempt-001".into(),
        image_id: runtime.image_id.clone(),
        backend: slurm::SlurmConfig {
            kind: "slurm".into(),
            ssh_alias: profile.ssh_alias.clone(),
            partition: profile.partition.clone(),
            account: profile.account.clone(),
            qos: profile.qos.clone(),
            gres: format!("gpu:{}:{}", profile.gpu_type, request.resources.gpus),
            time: format!(
                "{:02}:{:02}:{:02}",
                request.walltime_seconds / 3600,
                request.walltime_seconds / 60 % 60,
                request.walltime_seconds % 60
            ),
            source_dir: format!("{}/sources/{}", profile.mount_root, runtime.source_id),
            sif_path: runtime.sif_path.clone(),
            mount_root: profile.mount_root.clone(),
            oci_image: Some(runtime.oci_image.clone()),
            apptainer_cache_dir: None,
            apptainer_tmp_dir: None,
            apptainer_unsquash: false,
            image_stage_timeout_seconds: 1080,
        },
        storage: slurm::Storage {
            run_dir: format!("{}/{}/{}", profile.run_root, project, request.run_id),
            project_data_root: format!("{}/{}", profile.data_root, project),
        },
        resources: request.resources.clone(),
        execution: slurm::ExecutionConfig {
            source_mount: runtime.source_dir.clone(),
            workdir: runtime.source_dir.clone(),
            managed_io: true,
            env: request.env.clone(),
        },
        command,
    };
    let mut protocol = json!({"source_id":runtime.source_id,"image":runtime.oci_image,"runtime_id":runtime.runtime_id,
        "command":request.command,"parameters":request.parameters,"seed":request.seed,
        "evaluation":request.evaluation,"metrics_schema":request.metrics_schema});
    if !request.data.is_empty() {
        protocol["data"] = json!(request.data);
    }
    if !request.env.is_empty() {
        protocol["env"] = json!(request.env);
    }
    if let Some(preparation) = &request.data_preparation {
        protocol["data_preparation"] = json!(preparation);
    }
    match (&request.resume_from, authority.checkpoint) {
        (Some(reference), Some(value)) => {
            crate::checkpoints::selected(
                reference,
                value,
                project,
                &profile.target.storage_scope,
                &profile.run_root,
            )
            .map_err(|_| Error("checkpoint scope or identity"))?;
            protocol["resume_from"] = json!(reference);
        }
        (None, None) => {}
        _ => return Err(Error("checkpoint evidence")),
    }
    let mut run = json!({"schema_version":1,"project":project,"run_id":request.run_id,
        "created_at":now,"source_id":runtime.source_id,"image_id":runtime.image_id,
        "config_path":"request.json","seed":request.seed,"campaign":format!("run-{}",request.run_id),
        "parameters":request.parameters,"backend":spec.backend,"storage":spec.storage,
        "resources":spec.resources,"execution":spec.execution,"command":spec.command,
        "container":{"runtime_id":runtime.runtime_id},"inputs":[],"assets":[],
        "evaluation":{"metrics_schema":request.metrics_schema,"protocol":protocol,
            "protocol_id":canonical::protocol_identity(&protocol).map_err(|_|Error("evaluation protocol"))?},
        "native_creation":{"format":"native-run-creation.v1","request_sha256":request.identity(project)?,
            "project_sha256":authority.project_sha256,"source_receipt_sha256":authority.source_receipt_sha256,
            "backend_config_sha256":authority.backend_config_sha256}});
    if let Some(checkpoint) = authority.checkpoint {
        run["resume_from"] = json!(request.resume_from);
        run["native_checkpoint"] = checkpoint.clone();
    }
    if let Some(policy) = &request.artifacts {
        run["artifacts"] = json!(policy);
    }
    if let Some(preparation) = &request.data_preparation {
        run["data_preparation"] = json!(preparation);
    }
    if !datasets.is_empty() {
        run["native_datasets"] = json!(datasets);
    }
    if let Some(options) = &request.wandb {
        run["wandb"] = options.to_value();
    }
    if let Some(route) = authority.tracking_route {
        run["tracking_route"] = route.clone();
    }
    // Bind the full trusted runtime, including its materialization receipt.
    run["native_creation"]["runtime_sha256"] = json!(
        canonical::protocol_identity(&json!(runtime)).map_err(|_| Error("runtime identity"))?
    );
    let mut attempt = run.clone();
    attempt["attempt_id"] = json!("attempt-001");
    let binding = Binding {
        project: project.into(),
        run_id: request.run_id.clone(),
        attempt_id: "attempt-001".into(),
        project_sha256: authority.project_sha256.into(),
        run_sha256: canonical::protocol_identity(&run).map_err(|_| Error("Run hash"))?,
        attempt_sha256: canonical::protocol_identity(&attempt)
            .map_err(|_| Error("Attempt hash"))?,
        source_receipt_sha256: authority.source_receipt_sha256.into(),
        backend_config_sha256: authority.backend_config_sha256.into(),
    };
    definition(
        &run,
        &attempt,
        &binding,
        profile,
        runtime,
        &Approval {
            max_gpu_hours: f64::from(request.resources.gpus) * request.walltime_seconds as f64
                / 3600.,
            reason: String::new(),
        },
    )?;
    Ok(Documents {
        run,
        attempt,
        status: json!({"project":project,"run_id":request.run_id,"attempt_id":"attempt-001",
        "state":"NOT_SUBMITTED","updated_at":now}),
        request: json!(request),
    })
}
/// Compare immutable authorship only; later scheduler/worker state is separate.
/// Canonical bytes decide, as they do for the request identity: a repeat that
/// respells a number has the same digest and is the same creation.
pub fn same_creation(expected: &Documents, run: &Value, attempt: &Value, request: &Value) -> bool {
    let without_time = |value: &Value| {
        let mut v = value.clone();
        if let Some(m) = v.as_object_mut() {
            m.remove("created_at");
        }
        v
    };
    run["created_at"]
        .as_str()
        .is_some_and(|s| OffsetDateTime::parse(s, &Rfc3339).is_ok())
        && canonical::equal(&without_time(&expected.run), &without_time(run))
        && canonical::equal(&without_time(&expected.attempt), &without_time(attempt))
        && canonical::equal(&expected.request, request)
        && run["created_at"] == attempt["created_at"]
}
pub fn receipt(
    project: &str,
    request: &Request,
    run: &Value,
    attempt: &Value,
    reused: bool,
) -> Result<Value> {
    let mut value = json!({"project":project,"run_id":request.run_id,"attempt_id":"attempt-001","creation_state":"CREATED",
        "source_id":run["source_id"],"runtime_id":request.runtime_id,"request_sha256":request.identity(project)?,
        "run_sha256":slurm::helper::sha256_bytes(&serde_json::to_vec(run).map_err(|_|Error("Run hash"))?),
        "attempt_sha256":slurm::helper::sha256_bytes(&serde_json::to_vec(attempt).map_err(|_|Error("Attempt hash"))?),
        "protocol_id":run["evaluation"]["protocol_id"],"created_at":run["created_at"],"reused":reused,
        "scheduler_mutation_performed":false});
    // What was frozen, so a client that left the Attempt out can check its request.
    if let Some(reference) = &request.resume_from {
        value["resume_from"] = json!(reference);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Value {
        json!({"run_id":"trial","runtime_id":format!("runtime.{}","1".repeat(64)),"executor":"slurm-test",
            "command":["python","train.py"],"resources":{"gpus":1,"cpus":2,"memory_gb":4},"walltime_seconds":300,"seed":42})
    }
    #[test]
    fn run_input_refuses_ambiguous_config_and_preserves_metric_and_evaluation_identity() {
        let base = request();
        let parsed: Request = serde_json::from_value(base.clone()).unwrap();
        assert!(parsed.validate("demo").is_ok());
        assert!(serde_json::to_value(&parsed).unwrap().get("env").is_none());
        let mut configured = parsed.clone();
        configured.env.insert("OMP_NUM_THREADS".into(), "4".into());
        assert_ne!(
            parsed.identity("demo").unwrap(),
            configured.identity("demo").unwrap()
        );
        configured
            .env
            .insert("OUTPUT_DIR".into(), "/another/path".into());
        assert!(configured.validate("demo").is_err());
        for field in ["inputs", "resume_from", "backend", "storage"] {
            let mut bad = base.clone();
            bad[field] = json!({});
            assert!(Request::from_slice(&serde_json::to_vec(&bad).unwrap()).is_err());
        }
        let duplicate = base.to_string().replacen("{", "{\"seed\":1,", 1);
        assert!(Request::from_slice(duplicate.as_bytes()).is_err());
        let nested = base
            .to_string()
            .replacen("{", "{\"parameters\":{\"lr\":1,\"lr\":2},", 1);
        assert!(Request::from_slice(nested.as_bytes()).is_err());
        for field in ["cpus", "memory_gb"] {
            let mut bad = base.clone();
            bad["resources"][field] = json!(0);
            let bad: Request = serde_json::from_value(bad).unwrap();
            assert!(bad.validate("demo").is_err());
        }
        let mut cpu = parsed.clone();
        cpu.resources.gpus = 0;
        assert!(cpu.validate("demo").is_ok());
        // The Slurm GPU backend retains its separate positive-GPU constraint.
        assert!(cpu.resources.validate().is_err());
        assert_ne!(
            cpu.identity("demo").unwrap(),
            parsed.identity("demo").unwrap()
        );
        let mut alternate = base.clone();
        alternate["metrics_schema"] = json!({"definitions":{"loss":{"unit":"nats/token"}}});
        alternate["evaluation"] = json!({"variant_id":"v1","D_FT":0,"loss_mask":"prefix"});
        let one: Request = serde_json::from_value(alternate.clone()).unwrap();
        alternate["evaluation"]["variant_id"] = json!("v2");
        let two: Request = serde_json::from_value(alternate.clone()).unwrap();
        assert_ne!(one.identity("demo").unwrap(), two.identity("demo").unwrap());
        alternate["metrics_schema"]["definitions"]["loss"]["unit"] = json!("");
        assert!(Request::from_slice(&serde_json::to_vec(&alternate).unwrap()).is_err());
    }
    #[test]
    fn a_respelled_repeat_is_the_same_creation_and_a_changed_value_is_not() {
        let parse = |text: String| serde_json::from_str::<Value>(&text).unwrap();
        let document = |lr: &str| {
            parse(format!(
                r#"{{"created_at":"2026-01-01T00:00:00Z","parameters":{{"lr":{lr}}}}}"#
            ))
        };
        let original = |lr: &str| parse(format!(r#"{{"parameters":{{"lr":{lr}}}}}"#));
        let expected = Documents {
            run: document("1.5"),
            attempt: document("1.5"),
            status: json!({}),
            request: original("1.5"),
        };
        let same = |lr: &str| same_creation(&expected, &document(lr), &document(lr), &original(lr));
        assert_ne!(expected.run, document("1.50"));
        assert!(same("1.5"));
        assert!(same("1.50"));
        assert!(same("15e-1"));
        assert!(!same("1.6"));
        // A float stays a float, because canonical bytes are also delivered.
        let whole = Documents {
            run: document("2.0"),
            attempt: document("2.0"),
            status: json!({}),
            request: original("2.0"),
        };
        assert!(!same_creation(
            &whole,
            &document("2"),
            &document("2"),
            &original("2")
        ));

        let spelled = |lr: &str| {
            let mut request = request();
            request["parameters"] = json!({});
            let text = request.to_string().replacen(
                r#""parameters":{}"#,
                &format!(r#""parameters":{{"lr":{lr}}}"#),
                1,
            );
            Request::from_slice(text.as_bytes())
                .unwrap()
                .identity("demo")
                .unwrap()
        };
        assert_eq!(spelled("1.5"), spelled("1.50"));
    }
    #[test]
    fn planning_refuses_dataset_manifests_over_their_budget() {
        let digest = format!("sha256:{}", "a".repeat(64));
        let target = slurm::helper::Target {
            backend_id: "slurm-test".into(),
            profile_sha256: digest.clone(),
            storage_scope: digest.clone(),
        };
        let profile = Profile {
            target: target.clone(),
            ssh_alias: "cluster".into(),
            partition: "gpu".into(),
            account: "team".into(),
            qos: "normal".into(),
            gpu_type: "h200".into(),
            mount_root: "/shared".into(),
            run_root: "/shared/runs".into(),
            data_root: "/shared/datasets".into(),
            max_gpus: 8,
            max_cpus: 64,
            max_memory_gb: 512,
            max_walltime_seconds: 86400,
        };
        let runtime = Runtime {
            runtime_id: format!("runtime.{}", "1".repeat(64)),
            source_id: format!("source.{}", "2".repeat(64)),
            image_id: digest.clone(),
            sif_path: "/shared/images/code.sif".into(),
            sif_sha256: digest.clone(),
            oci_image: format!("registry.example/team/code@{digest}"),
            worker: slurm::managed::Runtime {
                program: "/opt/ml-expd/worker".into(),
                base_environment: BTreeMap::new(),
                runtime_environment: Vec::new(),
            },
            source_dir: "/workspace".into(),
        };
        let receipt = crate::datasets::oversized_receipt("demo", &target, &profile.data_root);
        let mut value = request();
        value["data"] = json!([{"name":"train","asset_id":receipt.asset_id}]);
        let request: Request = serde_json::from_value(value).unwrap();
        let refused = plan(
            "demo",
            &request,
            &profile,
            &runtime,
            Authority {
                project_sha256: &digest,
                source_receipt_sha256: &digest,
                backend_config_sha256: &digest,
                checkpoint: None,
                datasets: std::slice::from_ref(&receipt),
                tracking_route: None,
            },
            "2026-10-29T00:00:00Z",
        )
        .err()
        .unwrap();
        assert_eq!(refused.0, crate::datasets::ManifestsTooLarge::CAUSE);
    }
    #[test]
    fn an_absent_code_source_changes_nothing_and_a_present_one_is_bound() {
        let parsed: Request = serde_json::from_value(request()).unwrap();
        assert!(
            serde_json::to_value(&parsed)
                .unwrap()
                .get("code_source_id")
                .is_none()
        );
        // Pinned: this is the identity of a Run stored before the field existed.
        assert_eq!(
            parsed.identity("demo").unwrap(),
            canonical::protocol_identity(&json!({"project":"demo","request":{
                "run_id":"trial","runtime_id":format!("runtime.{}","1".repeat(64)),
                "executor":"slurm-test","command":["python","train.py"],
                "resources":parsed.resources,"walltime_seconds":300,"seed":42,
                "parameters":{},"evaluation":{},"metrics_schema":parsed.metrics_schema}}))
            .unwrap()
        );
        let code = |digit: &str| {
            let mut value = request();
            value["code_source_id"] = json!(format!("source.{}", digit.repeat(64)));
            serde_json::from_value::<Request>(value).unwrap()
        };
        assert_ne!(
            code("a").identity("demo").unwrap(),
            parsed.identity("demo").unwrap()
        );
        assert_ne!(
            code("a").identity("demo").unwrap(),
            code("b").identity("demo").unwrap()
        );
        let mut bad = code("a");
        bad.code_source_id = Some("source.short".into());
        assert!(bad.validate("demo").is_err());
        for name in ["SOURCE_DIR", "PYTHONDONTWRITEBYTECODE"] {
            let mut reserved = code("a");
            reserved.env.insert(name.into(), "x".into());
            assert!(reserved.validate("demo").is_err(), "{name}");
            // Without delivery these stay ordinary names, as in old stored Runs.
            let mut plain = parsed.clone();
            plain.env.insert(name.into(), "x".into());
            assert!(plain.validate("demo").is_ok(), "{name}");
        }
        let mut preparing = code("a");
        preparing.data_preparation = serde_json::from_value(json!({"script":"fetch.py"})).unwrap();
        assert!(preparing.validate("demo").is_err());
    }
    #[test]
    fn a_slurm_run_refuses_delivered_code_with_a_clear_cause() {
        let digest = format!("sha256:{}", "a".repeat(64));
        let target = slurm::helper::Target {
            backend_id: "slurm-test".into(),
            profile_sha256: digest.clone(),
            storage_scope: digest.clone(),
        };
        let profile = Profile {
            target,
            ssh_alias: "cluster".into(),
            partition: "gpu".into(),
            account: "team".into(),
            qos: "normal".into(),
            gpu_type: "h200".into(),
            mount_root: "/shared".into(),
            run_root: "/shared/runs".into(),
            data_root: "/shared/datasets".into(),
            max_gpus: 8,
            max_cpus: 64,
            max_memory_gb: 512,
            max_walltime_seconds: 86400,
        };
        let runtime = Runtime {
            runtime_id: format!("runtime.{}", "1".repeat(64)),
            source_id: format!("source.{}", "2".repeat(64)),
            image_id: digest.clone(),
            sif_path: "/shared/images/code.sif".into(),
            sif_sha256: digest.clone(),
            oci_image: format!("registry.example/team/code@{digest}"),
            worker: slurm::managed::Runtime {
                program: "/opt/ml-expd/worker".into(),
                base_environment: BTreeMap::new(),
                runtime_environment: Vec::new(),
            },
            source_dir: "/workspace".into(),
        };
        let mut value = request();
        value["code_source_id"] = json!(format!("source.{}", "3".repeat(64)));
        let request: Request = serde_json::from_value(value).unwrap();
        let refused = plan(
            "demo",
            &request,
            &profile,
            &runtime,
            Authority {
                project_sha256: &digest,
                source_receipt_sha256: &digest,
                backend_config_sha256: &digest,
                checkpoint: None,
                datasets: &[],
                tracking_route: None,
            },
            "2026-10-29T00:00:00Z",
        )
        .err()
        .unwrap();
        assert_eq!(refused.0, CODE_DELIVERY_UNSUPPORTED);
    }
    #[test]
    fn optional_cloud_policy_is_identity_bound_and_omission_preserves_original_bytes() {
        let base = request();
        let original: Request = serde_json::from_value(base.clone()).unwrap();
        assert!(
            serde_json::to_value(&original)
                .unwrap()
                .get("scheduling")
                .is_none()
        );
        let mut changed = base;
        changed["scheduling"] = json!({"allocation":"allow_idle"});
        let idle: Request = serde_json::from_value(changed.clone()).unwrap();
        assert_eq!(idle.scheduling.as_ref().unwrap().priority, 5);
        assert_ne!(
            idle.identity("demo").unwrap(),
            original.identity("demo").unwrap()
        );
        changed["scheduling"]["priority"] = json!(0);
        assert!(
            serde_json::from_value::<Request>(changed.clone())
                .unwrap()
                .validate("demo")
                .is_err()
        );
        changed["scheduling"]["priority"] = json!(10);
        assert!(
            serde_json::from_value::<Request>(changed)
                .unwrap()
                .validate("demo")
                .is_err()
        );
    }
}
