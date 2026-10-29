//! Bounded API relay to immutable object parts. Dataset trees materialize only
//! on the backend; an uploaded archive is never reported as a ready NAS tree.
use super::*;
use ml_exp_core::datasets::{self, Command, State};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Saved {
    pub(super) state: State,
    pub(super) archive_verified: bool,
    #[serde(default)]
    pub(super) image_recipe: Option<ml_exp_core::cloud::data::Image>,
    #[serde(default)]
    pub(super) image_started: bool,
    #[serde(default)]
    pub(super) image: Option<String>,
    #[serde(default)]
    pub(super) copy_request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) copy_storage: Option<ml_exp_core::cloud::launch::StorageLayout>,
    #[serde(default)]
    pub(super) copy_status: Option<Value>,
    #[serde(default)]
    pub(super) diagnostic: Option<Value>,
}
impl Service {
    pub fn has_datasets(&self) -> bool {
        self.objects.is_some() && self.exchange.is_some()
    }
    pub(super) fn data_name(project: &str, executor: &str, asset: &str) -> cloud::Result<String> {
        ml_exp_core::identity::validate_id(project)
            .and_then(|_| ml_exp_core::identity::validate_id(executor))
            .map_err(|_| cloud::Error::configuration())?;
        if !datasets::asset_id(asset) {
            return Err(cloud::Error::configuration());
        }
        Ok(format!(
            "dataset-{}.json",
            canonical::digest(&json!([project, executor, asset]))
                .map_err(|_| cloud::Error::evidence())?
        ))
    }
    pub(super) fn data_saved(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
    ) -> cloud::Result<Option<Saved>> {
        let backend = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        self.project_hash(project)?;
        let name = Self::data_name(project, executor, asset)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(None);
        }
        let saved: Saved = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(|_| cloud::Error::evidence())?;
        saved
            .state
            .validate()
            .map_err(|_| cloud::Error::evidence())?;
        if saved.state.project != project
            || saved.state.target != backend.definition.target()?
            || saved.state.asset_id != asset
            || saved.archive_verified
                && saved.state.parts.len() as u64
                    != saved
                        .state
                        .definition
                        .part_count()
                        .map_err(|_| cloud::Error::evidence())?
        {
            return Err(cloud::Error::evidence());
        }
        if saved
            .copy_storage
            .as_ref()
            .is_some_and(|s| backend.definition.storage_layout().as_ref().ok() != Some(s))
        {
            return Err(cloud::Error::evidence());
        }
        Ok(Some(saved))
    }
    pub(super) fn data_view(saved: Saved) -> Value {
        let mut value = json!(saved.state);
        value["archive_verified"] = json!(saved.archive_verified);
        value["materialization"] = json!(if saved.state.ready.is_some() {
            "READY"
        } else if saved.image_recipe.is_some() {
            "CCR_IMAGE_TO_NAS"
        } else {
            "ON_BACKEND_BEFORE_TRAINING"
        });
        value["data_image"] = json!(saved.image);
        value["copy_job"] = json!(saved.copy_status);
        value["diagnostic"] = json!(saved.diagnostic);
        value
    }
    pub fn dataset_state(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
    ) -> cloud::Result<Value> {
        Ok(Self::data_view(
            self.data_saved(project, executor, asset)?
                .ok_or_else(missing)?,
        ))
    }
    pub fn dataset_download(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        part: u64,
    ) -> cloud::Result<Value> {
        let saved = self
            .data_saved(project, executor, asset)?
            .filter(|s| s.archive_verified)
            .ok_or_else(missing)?;
        self.objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .data_ticket(project, executor, &saved.state.definition, part)
            .map_err(io)
    }
    pub fn dataset_transfer(
        &self,
        project: &str,
        executor: &str,
        command: Command,
        payload: &[u8],
    ) -> cloud::Result<Value> {
        if !self.has_datasets() {
            return Err(cloud::Error::configuration());
        }
        let backend = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        self.project_hash(project)?;
        let asset = match &command {
            Command::Begin { definition } => definition.asset_id(),
            Command::Part { asset_id, .. }
            | Command::Complete { asset_id }
            | Command::Status { asset_id } => asset_id.clone(),
        };
        let lock = self.lock()?;
        let mut saved = self.data_saved(project, executor, &asset)?;
        if let Command::Begin { definition } = &command {
            definition
                .validate()
                .map_err(|_| cloud::Error::configuration())?;
            if let Some(saved) = &saved {
                if saved.state.definition != *definition {
                    return Err(cloud::Error::evidence());
                }
            } else {
                saved = Some(Saved {
                    state: State::new(
                        project.into(),
                        backend.definition.target()?,
                        definition.clone(),
                    )
                    .map_err(|_| cloud::Error::configuration())?,
                    archive_verified: false,
                    image_recipe: None,
                    image_started: false,
                    image: None,
                    copy_request_id: None,
                    copy_storage: None,
                    copy_status: None,
                    diagnostic: None,
                });
            }
        }
        let mut saved = saved.ok_or_else(missing)?;
        let name = Self::data_name(project, executor, &asset)?;
        match command {
            Command::Part {
                number,
                sha256,
                bytes,
                ..
            } => {
                if bytes != payload.len() as u64 {
                    return Err(cloud::Error::configuration());
                }
                let record = ml_exp_core::multipart::PartRecord {
                    sha256: sha256.clone(),
                    bytes,
                };
                let state = saved
                    .state
                    .record_part(number, record.clone())
                    .map_err(|_| cloud::Error::evidence())?;
                if saved.archive_verified {
                    return Ok(Self::data_view(saved));
                }
                drop(lock);
                self.objects
                    .as_ref()
                    .ok_or_else(cloud::Error::configuration)?
                    .put_data(
                        project,
                        executor,
                        &state.definition,
                        number,
                        &sha256,
                        payload,
                    )
                    .map_err(io)?;
                let lock = self.lock()?;
                saved = self
                    .data_saved(project, executor, &asset)?
                    .ok_or_else(missing)?;
                saved.state = saved
                    .state
                    .record_part(number, record)
                    .map_err(|_| cloud::Error::evidence())?;
                lock.validate()?;
                self.slots
                    .replace_regular(OsStr::new(&name), &encode(&saved)?)
                    .map_err(io)?;
                return Ok(Self::data_view(saved));
            }
            Command::Complete { .. } => {
                if !payload.is_empty()
                    || saved.state.parts.len() as u64
                        != saved
                            .state
                            .definition
                            .part_count()
                            .map_err(|_| cloud::Error::evidence())?
                {
                    return Err(cloud::Error::evidence());
                }
                if !saved.archive_verified {
                    drop(lock);
                    self.objects
                        .as_ref()
                        .ok_or_else(cloud::Error::configuration)?
                        .verify_data(project, executor, &saved.state.definition)
                        .map_err(io)?;
                    let lock = self.lock()?;
                    let current = self
                        .data_saved(project, executor, &asset)?
                        .ok_or_else(missing)?;
                    if current.state != saved.state {
                        return Err(cloud::Error::evidence());
                    }
                    saved.archive_verified = true;
                    lock.validate()?;
                    self.slots
                        .replace_regular(OsStr::new(&name), &encode(&saved)?)
                        .map_err(io)?;
                }
                return Ok(Self::data_view(saved));
            }
            Command::Begin { .. } | Command::Status { .. } => {}
        }
        if !payload.is_empty() {
            return Err(cloud::Error::configuration());
        }
        lock.validate()?;
        self.slots
            .replace_regular(OsStr::new(&name), &encode(&saved)?)
            .map_err(io)?;
        Ok(Self::data_view(saved))
    }
    pub(super) fn uploaded_data(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
    ) -> cloud::Result<Option<datasets::Definition>> {
        Ok(self
            .data_saved(project, executor, asset)?
            .filter(|s| s.archive_verified)
            .map(|s| s.state.definition))
    }
    pub fn dataset_ticket(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
        asset: &str,
        part: u64,
    ) -> cloud::Result<Value> {
        let prepared = self.worker_authority(project, run, attempt, token)?;
        let package = prepared
            .execution
            .package()
            .map_err(|_| cloud::Error::evidence())?;
        let definition = package
            .config
            .datasets
            .iter()
            .find(|m| m.receipt.asset_id == asset)
            .map(|m| &m.receipt.definition)
            .ok_or_else(cloud::Error::evidence)?;
        if self
            .uploaded_data(project, &prepared.execution.backend_id, asset)?
            .as_ref()
            != Some(definition)
        {
            return Err(cloud::Error::evidence());
        }
        self.objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .data_ticket(project, &prepared.execution.backend_id, definition, part)
            .map_err(io)
    }
}
