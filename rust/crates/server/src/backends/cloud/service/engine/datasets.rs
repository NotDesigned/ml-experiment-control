//! Bounded API relay to immutable object parts. Dataset trees materialize only
//! on the backend; an uploaded archive is never reported as a ready NAS tree.
use super::*;
use ml_exp_core::datasets::{self, Command, State};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Saved {
    pub(super) state: State,
    pub(super) archive_verified: bool,
    pub(super) image_recipe: Option<ml_exp_core::cloud::data::Image>,
    pub(super) image_started: bool,
    pub(super) image: Option<String>,
    pub(super) copy_request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) copy_runtime: Option<ml_exp_core::cloud::run::Runtime>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) copy_intent: Option<ml_exp_core::cloud::Job>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) copy_storage: Option<ml_exp_core::cloud::launch::StorageLayout>,
    pub(super) copy_status: Option<Value>,
    pub(super) diagnostic: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) import_source: Option<datasets::oss::Import>,
}
/// How many wrong parts one call removes while it holds the service lock.
const MAX_DISCARDED_PARTS: usize = 64;
fn refused(code: &str) -> cloud::Error {
    cloud::Error {
        code: code.into(),
        http_status: Some(409),
        uncertain: false,
    }
}
impl Service {
    /// A real socket for tests of the object store requests. Configuration cannot reach this.
    #[doc(hidden)]
    pub fn redirect_objects_for_tests(&mut self, origin: String) {
        if let Some(objects) = &mut self.objects {
            objects.redirect_for_tests(origin);
        }
    }
    pub fn dataset_list(&self, project: &str, executor: &str) -> cloud::Result<Value> {
        self.project_hash(project)?;
        if !self.backends.contains_key(executor) {
            return Err(cloud::Error::configuration());
        }
        let mut rows = Vec::new();
        for name in self.slots.entry_names().map_err(io)? {
            let Some(name) = name
                .to_str()
                .filter(|s| s.starts_with("dataset-") && s.ends_with(".json"))
            else {
                continue;
            };
            let raw = self.slots.read_whole(OsStr::new(name)).map_err(io)?;
            let value =
                ml_exp_core::json::own_mapping(&raw).map_err(why!(cloud::Error::evidence()))?;
            let state = &value["state"];
            if state["project"] != project || state["target"]["backend_id"] != executor {
                continue;
            }
            let asset = state["asset_id"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?;
            if name != Self::data_name(project, executor, asset)? {
                return Err(cloud::Error::evidence());
            }
            let saved = self
                .data_saved(project, executor, asset)?
                .ok_or_else(missing)?;
            rows.push(
                datasets::Summary::from_state(&saved.state, saved.archive_verified)
                    .map_err(why!(cloud::Error::evidence()))?,
            );
        }
        rows.sort_by(|a, b| a.asset_id.cmp(&b.asset_id));
        Ok(json!({"project":project,"executor":executor,"datasets":rows}))
    }
    pub fn has_datasets(&self) -> bool {
        self.objects.is_some() && self.exchange.is_some()
    }
    pub(super) fn data_name(project: &str, executor: &str, asset: &str) -> cloud::Result<String> {
        ml_exp_core::identity::validate_id(project)
            .and_then(|_| ml_exp_core::identity::validate_id(executor))
            .map_err(why!(cloud::Error::configuration()))?;
        if !datasets::asset_id(asset) {
            return Err(cloud::Error::configuration());
        }
        Ok(format!(
            "dataset-{}.json",
            canonical::digest(&json!([project, executor, asset]))
                .map_err(why!(cloud::Error::evidence()))?
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
            .map_err(why!(cloud::Error::evidence()))?;
        saved
            .state
            .validate()
            .map_err(why!(cloud::Error::evidence()))?;
        if saved.state.project != project
            || saved.state.target != backend.definition.target()?
            || saved.state.asset_id != asset
            || saved.archive_verified
                && saved.state.parts.len() as u64
                    != saved
                        .state
                        .definition
                        .part_count()
                        .map_err(why!(cloud::Error::evidence()))?
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
        if let Some(source) = &saved.import_source {
            source
                .matches(&saved.state.definition)
                .map_err(why!(cloud::Error::evidence()))?;
            self.objects
                .as_ref()
                .ok_or_else(cloud::Error::configuration)?
                .validate_import(source)
                .map_err(io)?;
        }
        if let Some(runtime) = &saved.copy_runtime {
            runtime.validate().map_err(why!(cloud::Error::evidence()))?;
            if !matches!(backend.adapter.profile, cloud::adapter::Profile::Pai(_))
                || saved.image_recipe.is_some()
                || saved.image_started
                || saved.image.is_some()
                || saved.copy_request_id.is_none()
                || self.runtime(project, executor, &runtime.runtime_id)? != *runtime
            {
                return Err(cloud::Error::evidence());
            }
            let job = saved
                .copy_intent
                .as_ref()
                .ok_or_else(cloud::Error::evidence)?;
            let (config, channel) = ml_exp_core::cloud::data::materialization_inputs(job)
                .map_err(why!(cloud::Error::evidence()))?;
            let storage = saved
                .copy_storage
                .as_ref()
                .ok_or_else(cloud::Error::evidence)?;
            if config.project != project
                || config.target != backend.definition.target()?
                || config.root != backend.definition.data_root()?
                || config.definition != saved.state.definition
            {
                return Err(cloud::Error::evidence());
            }
            let mut expected = ml_exp_core::cloud::data::materialization_job(
                runtime,
                &config,
                &channel,
                storage,
                saved
                    .copy_request_id
                    .as_deref()
                    .ok_or_else(cloud::Error::evidence)?,
            )
            .map_err(why!(cloud::Error::evidence()))?;
            // Launcher bytes and scoped channels belong to the original intent,
            // not the current API executable or current public URL.
            expected.command[3] = job
                .command
                .get(3)
                .ok_or_else(cloud::Error::evidence)?
                .clone();
            if expected != *job {
                return Err(cloud::Error::evidence());
            }
        } else if saved.copy_intent.is_some() {
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
        } else if saved.copy_runtime.is_some() {
            "OSS_PARTS_TO_CPFS"
        } else {
            "ON_BACKEND_BEFORE_TRAINING"
        });
        value["data_image"] = json!(saved.image);
        value["copy_job"] = json!(saved.copy_status);
        value["new_request_allowed"] = json!(
            saved.state.ready.is_none()
                && saved
                    .copy_status
                    .as_ref()
                    .is_some_and(data_copies::allows_new_request)
        );
        value["diagnostic"] = json!(saved.diagnostic);
        value["import_source"] = json!(saved.import_source);
        value["runtime_id"] = json!(saved.copy_runtime.as_ref().map(|r| &r.runtime_id));
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
        if let Command::Begin { definition } = &command {
            definition.manifest_budget().map_err(|_| cloud::Error {
                code: "DATASET_MANIFESTS_TOO_LARGE".into(),
                http_status: Some(422),
                uncertain: false,
            })?;
        }
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
                .map_err(why!(cloud::Error::configuration()))?;
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
                    .map_err(why!(cloud::Error::configuration()))?,
                    archive_verified: false,
                    image_recipe: None,
                    image_started: false,
                    image: None,
                    copy_request_id: None,
                    copy_storage: None,
                    copy_runtime: None,
                    copy_intent: None,
                    copy_status: None,
                    diagnostic: None,
                    import_source: None,
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
                    .map_err(why!(cloud::Error::evidence()))?;
                if saved.archive_verified {
                    return Ok(Self::data_view(saved));
                }
                drop(lock);
                let objects = self
                    .objects
                    .as_ref()
                    .ok_or_else(cloud::Error::configuration)?;
                let put = || {
                    objects.put_data(
                        project,
                        executor,
                        &state.definition,
                        number,
                        &sha256,
                        payload,
                    )
                };
                match put() {
                    // A direct PUT that was never recorded left other bytes, and nothing
                    // can overwrite them. Without this, --via-server could not recover.
                    Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                        self.dataset_discard_parts(project, executor, &asset, &[number])?;
                        put().map_err(io)?;
                    }
                    other => other.map_err(io)?,
                }
                let _lock = self.lock()?;
                saved = self
                    .data_saved(project, executor, &asset)?
                    .ok_or_else(missing)?;
                saved.state = saved
                    .state
                    .record_part(number, record)
                    .map_err(why!(cloud::Error::evidence()))?;
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
                            .map_err(why!(cloud::Error::evidence()))?
                {
                    return Err(cloud::Error::evidence());
                }
                if !saved.archive_verified {
                    drop(lock);
                    let objects = self
                        .objects
                        .as_ref()
                        .ok_or_else(cloud::Error::configuration)?;
                    if objects
                        .verify_data(project, executor, &saved.state)
                        .map_err(io)?
                    {
                        return self.dataset_archive_verified(
                            project,
                            executor,
                            &asset,
                            &saved.state,
                        );
                    }
                    // Nothing is accepted. Each wrong part is removed so that the repeat
                    // can send it again, since a PUT can never replace it.
                    let wrong = objects
                        .wrong_data_parts(project, executor, &saved.state)
                        .map_err(io)?;
                    if wrong.is_empty() {
                        return Err(refused("DATASET_ARCHIVE_REFUSED"));
                    }
                    self.dataset_discard_parts(project, executor, &asset, &wrong)?;
                    return Err(refused("DATASET_PART_REJECTED"));
                }
                return Ok(Self::data_view(saved));
            }
            Command::Begin { .. } | Command::Status { .. } => {}
        }
        if !payload.is_empty() {
            return Err(cloud::Error::configuration());
        }
        self.slots
            .replace_regular(OsStr::new(&name), &encode(&saved)?)
            .map_err(io)?;
        Ok(Self::data_view(saved))
    }
    /// A signed PUT of one part, after the same checks a part through this server has.
    /// The record comes from `dataset_part_uploaded`, and `complete` proves the bytes.
    pub fn dataset_upload_ticket(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        part: u64,
        sha256: &str,
    ) -> cloud::Result<Value> {
        let saved = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        if saved.archive_verified {
            return Err(cloud::Error::evidence());
        }
        self.part_record(&saved.state, part, sha256)?;
        self.objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .data_upload_ticket(project, executor, &saved.state.definition, part)
            .map_err(io)
    }
    fn part_record(
        &self,
        state: &State,
        part: u64,
        sha256: &str,
    ) -> cloud::Result<ml_exp_core::multipart::PartRecord> {
        let record = ml_exp_core::multipart::PartRecord {
            sha256: sha256.into(),
            bytes: state
                .definition
                .part_size(part)
                .map_err(why!(cloud::Error::configuration()))?,
        };
        state
            .record_part(part, record.clone())
            .map_err(why!(cloud::Error::evidence()))?;
        Ok(record)
    }
    /// A direct PUT finished. A HEAD shows the object exists at the exact size, without
    /// reading it back on this small host; the lock is not held for it. The record is
    /// what the client claims until `complete` reads every byte. A resume skips it, and
    /// nothing else can use it: every reader of the parts needs `archive_verified`.
    pub fn dataset_part_uploaded(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        part: u64,
        sha256: &str,
    ) -> cloud::Result<Value> {
        let saved = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        let record = self.part_record(&saved.state, part, sha256)?;
        if saved.archive_verified || saved.state.parts.contains_key(&part) {
            return Ok(Self::data_view(saved));
        }
        let size = self
            .objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .data_part_size(project, executor, &saved.state.definition, part)
            .map_err(io)?;
        match size {
            None => return Err(refused("DATASET_PART_NOT_STORED")),
            Some(size) if size != record.bytes => {
                self.dataset_discard_parts(project, executor, asset, &[part])?;
                return Err(refused("DATASET_PART_REJECTED"));
            }
            Some(_) => {}
        }
        let _lock = self.lock()?;
        let mut saved = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        saved.state = saved
            .state
            .record_part(part, record)
            .map_err(why!(cloud::Error::evidence()))?;
        self.slots
            .replace_regular(
                OsStr::new(&Self::data_name(project, executor, asset)?),
                &encode(&saved)?,
            )
            .map_err(io)?;
        Ok(Self::data_view(saved))
    }
    /// Remove parts refused as wrong, so that a repeat can send them again. Once the
    /// archive is verified its parts are in use, and no earlier observation removes one.
    /// The lock covers the check and the deletes, so only a few go per call: a repeat
    /// finds the rest, and it keeps verifying until no part is wrong.
    fn dataset_discard_parts(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        wrong: &[u64],
    ) -> cloud::Result<()> {
        let _lock = self.lock()?;
        let mut current = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        if current.archive_verified || current.state.ready.is_some() {
            return Err(cloud::Error::evidence());
        }
        let objects = self
            .objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        for part in wrong.iter().take(MAX_DISCARDED_PARTS) {
            objects
                .delete_data(project, executor, &current.state.definition, *part)
                .map_err(io)?;
            current.state.parts.remove(part);
        }
        self.slots
            .replace_regular(
                OsStr::new(&Self::data_name(project, executor, asset)?),
                &encode(&current)?,
            )
            .map_err(io)
    }
    /// Record a verification that ran with the lock released. Anything else may have
    /// happened meanwhile, including a copy job starting, so only the flag is set, on
    /// the record as it stands now. Writing back the older snapshot would erase that
    /// job's identity and let a second paid copy start with no proof the first ended.
    #[doc(hidden)]
    pub fn dataset_archive_verified(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        verified: &State,
    ) -> cloud::Result<Value> {
        let _lock = self.lock()?;
        let mut current = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        if current.state != *verified {
            return Err(cloud::Error::evidence());
        }
        current.archive_verified = true;
        self.slots
            .replace_regular(
                OsStr::new(&Self::data_name(project, executor, asset)?),
                &encode(&current)?,
            )
            .map_err(io)?;
        Ok(Self::data_view(current))
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
            .map_err(why!(cloud::Error::evidence()))?;
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
