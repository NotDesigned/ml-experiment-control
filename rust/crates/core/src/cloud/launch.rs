//! A cloud image starts one sealed native worker. Only the outer shell creates
//! directories or starts processes; this module compiles and checks values.
use super::*;
use crate::{launch, managed};
use std::path::PathBuf;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Package {
    pub root: PathBuf,
    pub manifest: launch::Manifest,
    pub config: managed::Config,
    pub worker_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<super::exchange::Channel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_context: Option<crate::checkpoints::Context>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imports: Vec<crate::datasets::import::Config>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<crate::artifacts::publication::Policy>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub result_upload: bool,
}
impl Package {
    /// Attach only exact trusted receipts. Dataset paths point directly at
    /// immutable shared-storage trees, with no symlink or API-host copy.
    pub fn attach(
        &mut self,
        data_root: &str,
        datasets: Vec<crate::datasets::Mount>,
        checkpoint: Option<Value>,
    ) -> Result<()> {
        if !crate::datasets::root_path(data_root) {
            return Err(Error("cloud data root"));
        }
        crate::datasets::mounts(&datasets).map_err(|_| Error("cloud input receipts"))?;
        let project = &self.manifest.identity.project;
        let input_root = format!("{data_root}/{project}");
        self.manifest
            .environment
            .insert("DATA_DIR".into(), input_root.clone());
        self.manifest
            .environment
            .insert("INPUT_DIR".into(), input_root);
        if !datasets.is_empty() {
            let mapping: BTreeMap<_, _> = datasets
                .iter()
                .map(|m| (m.name.clone(), m.receipt.root.clone()))
                .collect();
            self.manifest.environment.insert(
                "DATASETS_JSON".into(),
                serde_json::to_string(&mapping).map_err(|_| Error("dataset mapping"))?,
            );
        }
        if let Some(value) = &checkpoint {
            let plan = crate::checkpoints::restore_plan(value, None)
                .map_err(|_| Error("cloud restore plan"))?;
            self.manifest.environment.extend([
                (
                    "RESUME_DIR".into(),
                    format!("{}/checkpoints/{}", plan.state_root, plan.generation),
                ),
                ("RESUME_CHECKPOINT_ID".into(), plan.checkpoint_id),
                ("RESUME_STEP".into(), plan.step.to_string()),
            ]);
        }
        self.config.datasets = datasets;
        self.config.checkpoint_restore = checkpoint;
        self.config
            .local_binding
            .as_mut()
            .ok_or(Error("local binding"))?
            .sha256 = helper::sha256_bytes(
            &self
                .manifest
                .encode()
                .map_err(|_| Error("launch manifest"))?,
        )[7..]
            .into();
        self.validate()
    }
    pub fn validate(&self) -> Result<()> {
        if let Some(policy) = &self.artifacts {
            policy.validate().map_err(|_| Error("artifact policy"))?;
        }
        self.config
            .validate()
            .map_err(|_| Error("cloud worker config"))?;
        let binding = self
            .config
            .binding()
            .map_err(|_| Error("cloud worker binding"))?;
        let identity = &binding.identity;
        if self.imports.len() > self.config.datasets.len() {
            return Err(Error("cloud imports"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for import in &self.imports {
            import.validate().map_err(|_| Error("cloud import"))?;
            if let crate::datasets::import::Source::ExchangeParts { channel } = &import.source {
                if self.channel.as_ref() != Some(channel) {
                    return Err(Error("dataset exchange scope"));
                }
            }
            let asset = import.definition.asset_id();
            let receipt = self
                .config
                .datasets
                .iter()
                .find(|m| m.receipt.asset_id == asset)
                .ok_or(Error("unbound import"))?;
            receipt
                .receipt
                .bind(&identity.project, &import.target, &import.root)
                .map_err(|_| Error("cloud import binding"))?;
            if import.project != identity.project
                || import.definition != receipt.receipt.definition
                || !seen.insert(asset)
            {
                return Err(Error("cloud import scope"));
            }
        }
        if let Some(channel) = &self.channel {
            channel.validate(identity)?;
        }
        if let Some(context) = &self.checkpoint_context {
            crate::checkpoints::context(
                &serde_json::to_value(context).map_err(|_| Error("checkpoint context"))?,
            )
            .map_err(|_| Error("checkpoint context"))?;
            if context.project != identity.project
                || context.run_id != identity.run_id
                || context.attempt_id != identity.attempt_id
                || std::path::Path::new(&context.state_root)
                    != self.config.attempt_root.join("state")
            {
                return Err(Error("checkpoint context scope"));
            }
        }
        let parent = self.root.join(&identity.project).join(&identity.run_id);
        if !managed::absolute(&self.root)
            || self.root.parent().is_none()
            || self.config.attempt_root != parent.join(&identity.attempt_id)
            || self.config.manifest_path
                != parent.join(format!("{}.launch.json", identity.attempt_id))
            || !helper::valid_digest(&self.worker_sha256)
            || self.config.local_binding.is_none()
            || !self.config.endpoint.is_empty()
            || self.config.output_directory.is_some()
        {
            return Err(Error("cloud worker paths"));
        }
        let bytes = self
            .manifest
            .encode()
            .map_err(|_| Error("cloud launch manifest"))?;
        launch::verify_binding(&bytes, &binding).map_err(|_| Error("cloud launch binding"))?;
        managed::execution_environment(&self.config, &self.manifest)
            .map_err(|_| Error("cloud worker environment"))?;
        if serde_json::to_vec(self)
            .map_err(|_| Error("cloud package"))?
            .len()
            > 64 * 1024
        {
            return Err(Error("cloud package size"));
        }
        Ok(())
    }
    pub fn command(&self, worker: &str) -> Result<Vec<String>> {
        self.validate()?;
        absolute_path(worker)?;
        Ok(vec![
            worker.into(),
            "cloud-launch".into(),
            "--package-json".into(),
            serde_json::to_string(self).map_err(|_| Error("cloud package encoding"))?,
        ])
    }
}
pub struct Input<'a> {
    pub root: &'a str,
    pub project: &'a str,
    pub run: &'a str,
    pub attempt: &'a str,
    pub workdir: &'a str,
    pub argv: &'a [String],
    pub timeout_seconds: u64,
    pub worker_sha256: &'a str,
}
pub fn compile(input: Input<'_>) -> Result<Package> {
    let identity = launch::Identity {
        project: input.project.into(),
        run_id: input.run.into(),
        attempt_id: input.attempt.into(),
    };
    let root = PathBuf::from(input.root);
    let parent = root.join(input.project).join(input.run);
    let attempt_root = parent.join(input.attempt);
    let manifest = launch::Manifest {
        contract: launch::CONTRACT.into(),
        identity: identity.clone(),
        environment: BTreeMap::from([
            (
                "OUTPUT_DIR".into(),
                attempt_root
                    .join("outputs")
                    .to_str()
                    .ok_or(Error("output path"))?
                    .into(),
            ),
            (
                "STATE_DIR".into(),
                attempt_root
                    .join("state")
                    .to_str()
                    .ok_or(Error("state path"))?
                    .into(),
            ),
            (
                "METRICS_PATH".into(),
                attempt_root
                    .join("outputs/metrics.jsonl")
                    .to_str()
                    .ok_or(Error("metrics path"))?
                    .into(),
            ),
        ]),
        argv: input.argv.to_vec(),
        workdir: input.workdir.into(),
        timeout_seconds: input.timeout_seconds,
    };
    let bytes = manifest
        .encode()
        .map_err(|_| Error("cloud launch manifest"))?;
    let config = managed::Config {
        manifest_path: parent.join(format!("{}.launch.json", input.attempt)),
        endpoint: String::new(),
        attempt_root,
        workdir: input.workdir.into(),
        maximum_timeout_seconds: input.timeout_seconds,
        base_environment: BTreeMap::from([(
            "PATH".into(),
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".into(),
        )]),
        local_binding: Some(launch::Binding {
            identity,
            sha256: helper::sha256_bytes(&bytes)[7..].into(),
        }),
        output_directory: None,
        runtime_environment: vec![
            "CUDA_VISIBLE_DEVICES".into(),
            "NVIDIA_VISIBLE_DEVICES".into(),
            "NVIDIA_DRIVER_CAPABILITIES".into(),
            "LD_LIBRARY_PATH".into(),
        ],
        checkpoint_restore: None,
        datasets: Vec::new(),
    };
    let package = Package {
        root,
        manifest,
        config,
        worker_sha256: input.worker_sha256.into(),
        channel: None,
        checkpoint_context: None,
        imports: Vec::new(),
        artifacts: None,
        result_upload: false,
    };
    package.validate()?;
    Ok(package)
}
