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
}
impl Package {
    pub fn validate(&self) -> Result<()> {
        self.config
            .validate()
            .map_err(|_| Error("cloud worker config"))?;
        let binding = self
            .config
            .binding()
            .map_err(|_| Error("cloud worker binding"))?;
        let identity = &binding.identity;
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
    };
    package.validate()?;
    Ok(package)
}
