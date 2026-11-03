//! Trusted OCI-to-SIF conversion request. No executable or destination can be
//! chosen by an API client; those belong to the gateway configuration.
use super::*;
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub operation: String,
    pub executor: String,
    pub runtime: run::Runtime,
}
impl Request {
    pub fn validate(&self) -> Result<()> {
        identity::validate_id(&self.executor).map_err(|_| Error("materializer executor"))?;
        self.runtime
            .validate()
            .map_err(|_| Error("materializer runtime"))?;
        if !matches!(self.operation.as_str(), "build" | "get") {
            return Err(Error("materializer operation"));
        }
        Ok(())
    }
}
