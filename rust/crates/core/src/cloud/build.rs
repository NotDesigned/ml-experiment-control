//! Native-image identity: a frozen recipe, source and base, with what the image's worker
//! must do. PAI and SenseCore images hold a worker that can fetch the one its Run froze, so
//! they name the contract it keeps. Slurm images, and every image built before this, name
//! the worker itself, and their identities stay exactly as they were.
use super::*;
pub const CONTRACT: &str = "native-image-recipe.v1";
/// Goes up only when an image's worker can no longer serve as the bootstrap for a later
/// one, and that forces one rebuild of every environment. The contract is: `cloud-launch`
/// and its arguments, `worker_sha256` at the top of a package, the exchange's `/package`
/// and `/worker`, and the plans of `script-prepare` and `data-import`, which run in the
/// image's own worker. A raised version must go on recognising this one for stored Runtimes.
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
        Ok(format!(
            "runtime.{}",
            canonical::digest(&json!([CONTRACT, project, self, frozen]))
                .map_err(|_| Error("build identity"))?
        ))
    }
}
