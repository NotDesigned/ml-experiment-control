//! A data image contains only one verified immutable archive and the native
//! importer. Backend paths and short-lived transfer credentials are excluded.
use super::*;
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub project: String,
    pub executor: String,
    pub definition: crate::datasets::Definition,
    pub base_image: String,
    pub worker_sha256: String,
}
impl Image {
    pub fn validate(&self) -> Result<()> {
        identity::validate_id(&self.project)
            .and_then(|_| identity::validate_id(&self.executor))
            .map_err(|_| Error("data image scope"))?;
        self.definition
            .validate()
            .map_err(|_| Error("data image archive"))?;
        if !helper::valid_digest(&self.worker_sha256) {
            return Err(Error("data image worker"));
        }
        Job {
            project: self.project.clone(),
            run_id: "data-check".into(),
            attempt_id: "attempt-001".into(),
            submission_token: "0".repeat(32),
            source_id: format!("source.{}", self.definition.sha256),
            image: self.base_image.clone(),
            command: vec!["true".into()],
            gpus: 0,
            cpus: 2,
            memory_gb: 4,
            max_time_seconds: 60,
            priority: 5,
            read_only_storage: false,
        }
        .validate()
    }
    pub fn id(&self) -> Result<String> {
        self.validate()?;
        Ok(format!(
            "data-image.{}",
            canonical::digest(&json!(["native-data-image.v1", self]))
                .map_err(|_| Error("data image identity"))?
        ))
    }
}

/// Compile a metadata-only PAI copy using an already published immutable image.
/// Keep the manifest once, and retain the frozen worker's archive/callback path.
pub fn materialization_job(
    runtime: &super::run::Runtime,
    config: &crate::datasets::import::Config,
    channel: &crate::datasets::import::DataChannel,
    storage: &super::launch::StorageLayout,
    request_id: &str,
) -> Result<Job> {
    runtime.validate()?;
    config
        .validate()
        .map_err(|_| Error("materialization config"))?;
    channel
        .validate(
            &config.project,
            &config.target.backend_id,
            &config.definition.asset_id(),
        )
        .map_err(|_| Error("materialization channel"))?;
    storage.validate()?;
    crate::submission::validate_submission_token(request_id)
        .map_err(|_| Error("materialization request"))?;
    if config.root != storage.datasets.to_string_lossy()
        || config.timeout_seconds != 3600
        || !matches!(
            config.source,
            crate::datasets::import::Source::DatasetParts { .. }
        )
    {
        return Err(Error("materialization storage or source"));
    }
    let identity = canonical::digest(&json!([
        "native-pai-data-materialization.v1",
        runtime,
        config.project,
        config.target,
        config.definition.asset_id(),
        storage,
        request_id
    ]))
    .map_err(|_| Error("materialization identity"))?;
    let job = Job {
        project: config.project.clone(),
        run_id: format!("data-{}", &identity[..32]),
        attempt_id: "attempt-001".into(),
        submission_token: identity[..32].into(),
        source_id: runtime.source_id.clone(),
        image: runtime.image.clone(),
        command: vec![
            "python3".into(),
            "-u".into(),
            "-c".into(),
            include_str!("materialize.py").into(),
            runtime.worker_path.clone(),
            serde_json::to_string(config).map_err(|_| Error("materialization encoding"))?,
            serde_json::to_string(channel).map_err(|_| Error("materialization encoding"))?,
            serde_json::to_string(storage).map_err(|_| Error("materialization encoding"))?,
        ],
        gpus: 0,
        cpus: 2,
        memory_gb: 4,
        max_time_seconds: 3600,
        priority: 5,
        read_only_storage: false,
    };
    job.validate()?;
    Ok(job)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn existing_runtime_copy_keeps_a_large_manifest_once_and_has_no_training_command() {
        let runtime = super::super::run::Runtime {
            runtime_id: format!("runtime.{}", "a".repeat(64)),
            source_id: format!("source.{}", "b".repeat(64)),
            image: format!("registry.example/team/image@sha256:{}", "c".repeat(64)),
            source_dir: "/ml-expd/source".into(),
            worker_path: "/usr/local/bin/ml-exp-worker-rs".into(),
            worker_sha256: format!("sha256:{}", "d".repeat(64)),
        };
        let storage = super::super::launch::StorageLayout {
            mount: "/mnt/workspace".into(),
            runs: "/mnt/workspace/ml-expd/runs".into(),
            datasets: "/mnt/workspace/ml-expd/datasets".into(),
        };
        let definition = crate::datasets::Definition {
            sha256: "e".repeat(64),
            archive_bytes: 1024 * 1024,
            files: (0..303)
                .map(|n| crate::artifacts::FileReceipt {
                    path: format!("dataset/tasks/sample-{n:04}.bin"),
                    bytes: 128,
                    sha256: "f".repeat(64),
                })
                .collect(),
        };
        let channel = crate::datasets::import::DataChannel {
            url: format!(
                "https://example.org/api/dataset-exchange/demo/pai/{}",
                definition.asset_id()
            ),
            token: "a".repeat(64),
        };
        let config = crate::datasets::import::Config {
            root: storage.datasets.to_str().unwrap().into(),
            target: crate::slurm::helper::Target {
                backend_id: "pai".into(),
                profile_sha256: format!("sha256:{}", "1".repeat(64)),
                storage_scope: format!("sha256:{}", "2".repeat(64)),
            },
            project: "demo".into(),
            definition: definition.clone(),
            source: crate::datasets::import::Source::DatasetParts {
                channel: channel.clone(),
            },
            timeout_seconds: 3600,
        };
        let job =
            materialization_job(&runtime, &config, &channel, &storage, &"3".repeat(32)).unwrap();
        assert_eq!(
            (job.gpus, job.cpus, job.memory_gb, job.max_time_seconds),
            (0, 2, 4, 3600)
        );
        assert_eq!(job.image, runtime.image);
        assert_eq!(
            serde_json::from_str::<crate::datasets::import::Config>(&job.command[5])
                .unwrap()
                .definition,
            definition
        );
        assert!(
            !job.command
                .iter()
                .any(|s| s == "cloud-launch" || s == "--sync")
        );
        assert!(
            job == materialization_job(&runtime, &config, &channel, &storage, &"3".repeat(32))
                .unwrap()
        );
        assert_ne!(
            job.submission_token,
            materialization_job(&runtime, &config, &channel, &storage, &"4".repeat(32))
                .unwrap()
                .submission_token
        );
    }
}
