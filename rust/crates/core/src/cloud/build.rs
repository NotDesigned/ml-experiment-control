//! Native-image identity is distinct from the published legacy Python recipe.
use super::*;
pub const CONTRACT: &str = "native-image-recipe.v1";
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
        }
        .validate()
    }
    pub fn id(&self, project: &str, worker: &str) -> Result<String> {
        self.validate()?;
        identity::validate_id(project).map_err(|_| Error("build project"))?;
        if !helper::valid_digest(worker) {
            return Err(Error("build worker"));
        }
        Ok(format!(
            "runtime.{}",
            canonical::digest(&json!([CONTRACT, project, self, worker]))
                .map_err(|_| Error("build identity"))?
        ))
    }
}
