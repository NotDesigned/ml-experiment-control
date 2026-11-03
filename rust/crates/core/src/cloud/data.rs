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
