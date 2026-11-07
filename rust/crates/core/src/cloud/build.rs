//! Native-image identity: a frozen recipe, source and base, with what the image's worker
//! must do. PAI and SenseCore images hold a worker that can fetch the one its Run froze, so
//! they name the contract it keeps. Slurm images, and every image built before this, name
//! the worker itself, and their identities stay exactly as they were.
use super::*;
pub const CONTRACT: &str = "native-image-recipe.v1";
/// Goes up only when an image's worker can no longer do what a later server asks of it, and
/// that forces one rebuild of every environment. A raised version must go on recognising
/// this one for stored Runtimes. `tests::the_bootstrap_contract_is_pinned` holds what follows.
///
/// A worker in an image is asked to run, by a server of any later release:
/// - `cloud-launch --package-url --package-token --package-sha256 [--inspect|--sync]`, reading
///   only `worker_sha256` from the package, and `GET {channel}/package` and `{channel}/worker`.
/// - `data-import --config --bundle-output`, for PAI data jobs, with `datasets::import::Config`.
///   Its `DatasetParts` and `ExchangeParts` read the exchanges with the channels in that config.
/// - `data-copy --config --channel-json --storage-json`, for the same jobs, with that `Config`,
///   `import::DataChannel` and `launch::StorageLayout`, and `POST {data channel}/ready`.
/// - `script-prepare --plan-json`, for SenseCore's data preparation, with `data_preparation::Plan`.
///
/// Each of those types denies unknown fields, so a new field is a change to this contract
/// unless the worker it reaches is always the frozen one.
pub const BOOTSTRAP: &str = "worker-bootstrap.v1";
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub source_id: String,
    pub executor: String,
    pub base_image: String,
    #[serde(default = "dockerfile")]
    pub dockerfile: String,
}
fn dockerfile() -> String {
    "Dockerfile".into()
}
impl Definition {
    pub fn validate(&self) -> Result<()> {
        identity::validate_id(&self.executor).map_err(|_| Error("build executor"))?;
        crate::sources::validate_source_id(&self.source_id).map_err(|_| Error("build source"))?;
        if !crate::datasets::path(&self.dockerfile) {
            return Err(Error("Dockerfile path"));
        }
        Job {
            project: "check".into(),
            run_id: "check".into(),
            attempt_id: "attempt-001".into(),
            submission_token: "0".repeat(32),
            source_id: self.source_id.clone(),
            image: self.base_image.clone(),
            command: vec!["true".into()],
            gpus: 0,
            cpus: 1,
            memory_gb: 1,
            max_time_seconds: 60,
            priority: 5,
            read_only_storage: false,
        }
        .validate()
    }
    /// `frozen` is `BOOTSTRAP`, or the hash of the worker for an image that cannot bootstrap.
    pub fn id(&self, project: &str, frozen: &str) -> Result<String> {
        self.validate()?;
        identity::validate_id(project).map_err(|_| Error("build project"))?;
        if frozen != BOOTSTRAP && !helper::valid_digest(frozen) {
            return Err(Error("build worker"));
        }
        self.named(project, frozen)
    }
    /// The identity a client looks up to say whether an environment is reused. A server
    /// may offer a later bootstrap version than this client knows; being only a question,
    /// that is worth asking, and a wrong answer says a build is to come.
    pub fn offered_id(&self, project: &str, offered: &str) -> Result<String> {
        self.validate()?;
        identity::validate_id(project).map_err(|_| Error("build project"))?;
        if !offered.starts_with("worker-bootstrap.v") && !helper::valid_digest(offered) {
            return Err(Error("build worker"));
        }
        self.named(project, offered)
    }
    fn named(&self, project: &str, frozen: &str) -> Result<String> {
        Ok(format!(
            "runtime.{}",
            canonical::digest(&json!([CONTRACT, project, self, frozen]))
                .map_err(|_| Error("build identity"))?
        ))
    }
}
