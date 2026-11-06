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

const MAX_ARGUMENT_BYTES: usize = 64 * 1024;
/// A dataset config grows with its file inventory, so it travels in argument
/// parts that each fit one argument. Joining them restores the exact text.
pub fn argument_parts(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut rest = text;
    while rest.len() > MAX_ARGUMENT_BYTES {
        let mut end = MAX_ARGUMENT_BYTES;
        while !rest.is_char_boundary(end) {
            end -= 1;
        }
        let (part, tail) = rest.split_at(end);
        parts.push(part.to_owned());
        rest = tail;
    }
    parts.push(rest.to_owned());
    parts
}

/// Launcher arguments are the worker path, the config parts, the channel and the
/// storage layout. This is where the config parts begin in a compiled command.
const CONFIG_ARGUMENT: usize = 5;
/// The frozen config and channel of a job compiled by `materialization_job`.
pub fn materialization_inputs(
    job: &Job,
) -> Result<(
    crate::datasets::import::Config,
    crate::datasets::import::DataChannel,
)> {
    let channel = job
        .command
        .len()
        .checked_sub(2)
        .filter(|at| *at > CONFIG_ARGUMENT)
        .ok_or(Error("materialization command"))?;
    Ok((
        serde_json::from_str(&job.command[CONFIG_ARGUMENT..channel].concat())
            .map_err(|_| Error("materialization config"))?,
        serde_json::from_str(&job.command[channel])
            .map_err(|_| Error("materialization channel"))?,
    ))
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
    let mut command = vec![
        "python3".to_owned(),
        "-u".into(),
        "-c".into(),
        include_str!("materialize.py").into(),
        runtime.worker_path.clone(),
    ];
    command.extend(argument_parts(
        &serde_json::to_string(config).map_err(|_| Error("materialization encoding"))?,
    ));
    command.push(serde_json::to_string(channel).map_err(|_| Error("materialization encoding"))?);
    command.push(serde_json::to_string(storage).map_err(|_| Error("materialization encoding"))?);
    let job = Job {
        project: config.project.clone(),
        run_id: format!("data-{}", &identity[..32]),
        attempt_id: "attempt-001".into(),
        submission_token: identity[..32].into(),
        source_id: runtime.source_id.clone(),
        image: runtime.image.clone(),
        command,
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
    struct Inputs {
        runtime: super::super::run::Runtime,
        storage: super::super::launch::StorageLayout,
        definition: crate::datasets::Definition,
        channel: crate::datasets::import::DataChannel,
        config: crate::datasets::import::Config,
    }
    fn inputs(files: usize) -> Inputs {
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
            files: (0..files)
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
        Inputs {
            runtime,
            storage,
            definition,
            channel,
            config,
        }
    }
    #[test]
    fn existing_runtime_copy_keeps_a_large_manifest_once_and_has_no_training_command() {
        let Inputs {
            runtime,
            storage,
            definition,
            channel,
            config,
        } = inputs(303);
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
    #[test]
    fn a_long_file_inventory_stays_within_each_argument_limit() {
        for files in [1_900, 5_000] {
            let i = inputs(files);
            let job = materialization_job(
                &i.runtime,
                &i.config,
                &i.channel,
                &i.storage,
                &"3".repeat(32),
            )
            .unwrap();
            assert!(job.command.iter().all(|s| s.len() <= 64 * 1024));
            assert!(shell_join(&job.command).unwrap().len() > 128 * 1024);
            let (config, channel) = materialization_inputs(&job).unwrap();
            assert!(config == i.config);
            assert_eq!(channel.url, i.channel.url);
            assert_eq!(
                job.command.last(),
                Some(&serde_json::to_string(&i.storage).unwrap())
            );
        }
    }
    #[test]
    fn argument_parts_never_split_a_character_and_join_to_the_same_text() {
        let text = "\u{1f600}".repeat(40_000);
        let parts = argument_parts(&text);
        assert!(parts.len() > 1);
        assert!(parts.iter().all(|s| s.len() <= 64 * 1024));
        assert_eq!(parts.concat(), text);
        assert_eq!(argument_parts("{}"), ["{}"]);
    }
    #[test]
    fn only_a_cpu_job_may_exceed_one_shell_argument() {
        let i = inputs(1_900);
        let mut job = materialization_job(
            &i.runtime,
            &i.config,
            &i.channel,
            &i.storage,
            &"3".repeat(32),
        )
        .unwrap();
        assert!(job.validate().is_ok());
        job.gpus = 1;
        assert!(job.validate().is_err());
    }
    #[test]
    fn a_command_with_too_few_arguments_has_no_materialization_inputs() {
        let i = inputs(3);
        let mut job = materialization_job(
            &i.runtime,
            &i.config,
            &i.channel,
            &i.storage,
            &"3".repeat(32),
        )
        .unwrap();
        job.command.truncate(6);
        assert!(materialization_inputs(&job).is_err());
    }
}
