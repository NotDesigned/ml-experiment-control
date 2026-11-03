//! SenseCore delivery uses a CCR data image and a fixed debug CPU ACP profile.
//! Build and copy intents survive response loss; GET never creates a resource.
use super::*;
use ml_exp_core::{
    cloud::{Job, data::Image},
    datasets::{
        Receipt,
        import::{Config, DataChannel, Source},
    },
};
impl Service {
    fn data_channel(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
    ) -> cloud::Result<DataChannel> {
        self.exchange
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .dataset_channel(&self.workspace, project, executor, asset, "download")
    }
    fn data_auth(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        token: &str,
        access: &str,
    ) -> cloud::Result<datasets::Saved> {
        use subtle::ConstantTimeEq;
        let expected = self
            .exchange
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .dataset_channel(&self.workspace, project, executor, asset, access)?;
        if !bool::from(expected.token.as_bytes().ct_eq(token.as_bytes())) {
            return Err(cloud::Error {
                code: "DATASET_AUTH_REQUIRED".into(),
                http_status: Some(401),
                uncertain: false,
            });
        }
        self.data_saved(project, executor, asset)?
            .filter(|s| s.archive_verified)
            .ok_or_else(missing)
    }
    pub fn data_download_ticket(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        token: &str,
        part: u64,
    ) -> cloud::Result<Value> {
        let saved = self.data_auth(project, executor, asset, token, "download")?;
        self.objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .data_ticket(project, executor, &saved.state.definition, part)
            .map_err(io)
    }
    pub fn data_ready(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        token: &str,
        receipt: Receipt,
    ) -> cloud::Result<Value> {
        let lock = self.lock()?;
        let mut saved = self.data_auth(project, executor, asset, token, "ready")?;
        let definition = &self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?
            .definition;
        receipt
            .bind(project, &definition.target()?, &definition.data_root()?)
            .map_err(|_| cloud::Error::evidence())?;
        if !saved.image_started || saved.image.is_none() || saved.copy_request_id.is_none() {
            return Err(cloud::Error::evidence());
        }
        saved.state = saved
            .state
            .publish(receipt)
            .map_err(|_| cloud::Error::evidence())?;
        saved.diagnostic = None;
        lock.validate()?;
        self.slots
            .replace_regular(
                OsStr::new(&Self::data_name(project, executor, asset)?),
                &encode(&saved)?,
            )
            .map_err(io)?;
        Ok(json!({"asset_id":asset,"status":"READY"}))
    }
    fn data_build_call(&self, recipe: &Image, operation: &str) -> cloud::Result<(u16, Value)> {
        let settings = self
            .builds
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let definition = &self
            .backends
            .get(&recipe.executor)
            .ok_or_else(cloud::Error::configuration)?
            .definition;
        let source = Source::DatasetParts {
            channel: self.data_channel(
                &recipe.project,
                &recipe.executor,
                &recipe.definition.asset_id(),
            )?,
        };
        let import = Config {
            root: "/tmp/ml-expd-native-data".into(),
            target: definition.target()?,
            project: recipe.project.clone(),
            definition: recipe.definition.clone(),
            source,
            timeout_seconds: 1200,
        };
        let payload = json!({"operation":operation,"data_image_id":recipe.id().map_err(|_|cloud::Error::evidence())?,"recipe":recipe,"download":import});
        let (status, bytes) = ml_exp_io::unix_http::request(
            &settings.socket,
            settings.peer_uid,
            &encode(&payload)?,
            Duration::from_secs(if operation == "native-data-build" {
                7200
            } else {
                60
            }),
        )
        .map_err(|_| cloud::Error {
            code: "DATA_IMAGE_BUILD_UNCERTAIN".into(),
            http_status: None,
            uncertain: true,
        })?;
        let value = Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| cloud::Error::evidence())?,
        );
        Ok((status, value))
    }
    fn accept_data_build(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        status: u16,
        value: Value,
    ) -> cloud::Result<()> {
        let lock = self.lock()?;
        let mut saved = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        let recipe = saved
            .image_recipe
            .as_ref()
            .ok_or_else(cloud::Error::evidence)?;
        if status == 200 {
            let image = value["image"].as_str().ok_or_else(cloud::Error::evidence)?;
            if value["data_image_id"] != recipe.id().map_err(|_| cloud::Error::evidence())?
                || value["recipe"] != json!(recipe)
            {
                return Err(cloud::Error::evidence());
            }
            let check = Job {
                project: project.into(),
                run_id: "data-check".into(),
                attempt_id: "attempt-001".into(),
                submission_token: "0".repeat(32),
                source_id: format!("source.{}", recipe.definition.sha256),
                image: image.into(),
                command: vec!["true".into()],
                gpus: 0,
                cpus: 2,
                memory_gb: 4,
                max_time_seconds: 60,
                priority: 5,
            };
            check.validate().map_err(|_| cloud::Error::evidence())?;
            if !image
                .split('/')
                .next()
                .is_some_and(|h| h.ends_with(".sensecore.cn"))
                || saved.image.as_ref().is_some_and(|old| old != image)
            {
                return Err(cloud::Error::evidence());
            }
            let features: std::collections::BTreeSet<String> = value
                .get("worker_features")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|_| cloud::Error::evidence())?
                .unwrap_or_default();
            if features.iter().any(|f| {
                !matches!(
                    f.as_str(),
                    "storage-layout.v1" | "script-preparation.v1" | "shared-cache-identity.v2"
                )
            }) {
                return Err(cloud::Error::evidence());
            }
            if saved.copy_request_id.is_none() && features.contains("storage-layout.v1") {
                saved.copy_storage = Some(self.backends[executor].definition.storage_layout()?);
            }
            saved.image = Some(image.into());
            saved.diagnostic = None;
        } else {
            saved.diagnostic =
                Some(json!({"code":"DATA_IMAGE_PUBLICATION_UNCONFIRMED","http_status":status}));
        }
        lock.validate()?;
        self.slots
            .replace_regular(
                OsStr::new(&Self::data_name(project, executor, asset)?),
                &encode(&saved)?,
            )
            .map_err(io)
    }
    pub fn publish_dataset(
        self: &Arc<Self>,
        project: &str,
        executor: &str,
        asset: &str,
        request_id: &str,
    ) -> cloud::Result<Value> {
        let effect = self
            .slots
            .open_lock(OsStr::new(&format!(
                "{}.delivery.lock",
                Self::data_name(project, executor, asset)?
            )))
            .map_err(io)?;
        fs2::FileExt::try_lock_exclusive(&effect).map_err(|_| cloud::Error::evidence())?;
        ml_exp_core::submission::validate_submission_token(request_id)
            .map_err(|_| cloud::Error::configuration())?;
        if !self.enabled {
            return Err(cloud::Error::configuration());
        }
        let backend = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        let copier = backend
            .data_copier
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let mut saved = self
            .data_saved(project, executor, asset)?
            .filter(|s| s.archive_verified)
            .ok_or_else(missing)?;
        if saved.state.ready.is_some() {
            return Ok(Self::data_view(saved));
        }
        if saved.image_recipe.is_none() {
            let builds = self
                .builds
                .as_ref()
                .ok_or_else(cloud::Error::configuration)?;
            let recipe = Image {
                project: project.into(),
                executor: executor.into(),
                definition: saved.state.definition.clone(),
                base_image: builds
                    .data_base_image
                    .clone()
                    .ok_or_else(cloud::Error::configuration)?,
                worker_sha256: builds.worker_sha256.clone(),
            };
            recipe
                .validate()
                .map_err(|_| cloud::Error::configuration())?;
            let lock = self.lock()?;
            saved = self
                .data_saved(project, executor, asset)?
                .ok_or_else(missing)?;
            if saved.image_recipe.is_none() {
                saved.image_recipe = Some(recipe);
                lock.validate()?;
                self.slots
                    .replace_regular(
                        OsStr::new(&Self::data_name(project, executor, asset)?),
                        &encode(&saved)?,
                    )
                    .map_err(io)?;
            }
        }
        if saved.image.is_none() {
            let lock = self.lock()?;
            saved = self
                .data_saved(project, executor, asset)?
                .ok_or_else(missing)?;
            let dispatch = !saved.image_started;
            saved.image_started = true;
            lock.validate()?;
            self.slots
                .replace_regular(
                    OsStr::new(&Self::data_name(project, executor, asset)?),
                    &encode(&saved)?,
                )
                .map_err(io)?;
            drop(lock);
            let recipe = saved
                .image_recipe
                .clone()
                .ok_or_else(cloud::Error::evidence)?;
            if dispatch {
                let service = self.clone();
                let project = project.to_owned();
                let executor = executor.to_owned();
                let asset = asset.to_owned();
                std::thread::Builder::new()
                    .name("native-data-image".into())
                    .spawn(move || {
                        let result = service.data_build_call(&recipe, "native-data-build");
                        let (status, value) = result.unwrap_or((503, json!({})));
                        let _ =
                            service.accept_data_build(&project, &executor, &asset, status, value);
                    })
                    .map_err(|_| cloud::Error::evidence())?;
                return Ok(Self::data_view(saved));
            }
            let (status, value) = self.data_build_call(&recipe, "native-data-get")?;
            self.accept_data_build(project, executor, asset, status, value)?;
            saved = self
                .data_saved(project, executor, asset)?
                .ok_or_else(missing)?;
            if saved.image.is_none() {
                return Ok(Self::data_view(saved));
            }
        }
        let job_for = |request_id: &str| -> cloud::Result<Job> {
            let identity = canonical::digest(&json!([
                "native-data-copy.v1",
                saved.image_recipe,
                request_id
            ]))
            .map_err(|_| cloud::Error::evidence())?;
            let config = Config {
                root: backend.definition.data_root()?,
                target: backend.definition.target()?,
                project: project.into(),
                definition: saved.state.definition.clone(),
                source: Source::File {
                    path: "/ml-expd/data/archive.tar".into(),
                },
                timeout_seconds: 3600,
            };
            let channel = self
                .exchange
                .as_ref()
                .ok_or_else(cloud::Error::configuration)?
                .dataset_channel(&self.workspace, project, executor, asset, "ready")?;
            let mut command = vec![
                "/usr/local/bin/ml-exp-worker-rs".into(),
                "data-copy".into(),
                "--config-json".into(),
                String::from_utf8(encode(&config)?).map_err(|_| cloud::Error::evidence())?,
                "--channel-json".into(),
                String::from_utf8(encode(&channel)?).map_err(|_| cloud::Error::evidence())?,
            ];
            if let Some(storage) = &saved.copy_storage {
                command.extend([
                    "--storage-json".into(),
                    String::from_utf8(encode(storage)?).map_err(|_| cloud::Error::evidence())?,
                ]);
            }
            Ok(Job {
                project: project.into(),
                run_id: format!("data-{}", &identity[..32]),
                attempt_id: "attempt-001".into(),
                submission_token: identity[..32].into(),
                source_id: format!("source.{}", saved.state.definition.sha256),
                image: saved.image.clone().ok_or_else(cloud::Error::evidence)?,
                command,
                gpus: 0,
                cpus: 2,
                memory_gb: 4,
                max_time_seconds: 3600,
                priority: 5,
            })
        };
        if let Some(previous) = saved
            .copy_request_id
            .as_ref()
            .filter(|id| id.as_str() != request_id)
        {
            let job = job_for(previous)?;
            let check = || self.project_hash(project).map(|_| ());
            let state = self.sync_outbox.reconcile(
                copier,
                &job,
                ml_exp_core::cloud::pai::Allocation::Guaranteed,
                &check,
                &AtomicBool::new(false),
            )?;
            if !state["observation"]["state"].as_str().is_some_and(|s| {
                matches!(
                    s,
                    "FAILED" | "CANCELLED" | "TIMEOUT" | "SUCCEEDED" | "PREEMPTED"
                )
            }) {
                return Err(cloud::Error::evidence());
            }
        }
        let job = job_for(request_id)?;
        {
            let lock = self.lock()?;
            saved = self
                .data_saved(project, executor, asset)?
                .ok_or_else(missing)?;
            saved.copy_request_id = Some(request_id.into());
            lock.validate()?;
            self.slots
                .replace_regular(
                    OsStr::new(&Self::data_name(project, executor, asset)?),
                    &encode(&saved)?,
                )
                .map_err(io)?;
        }
        let check = || self.project_hash(project).map(|_| ());
        self.sync_outbox.prepare(
            &copier.profile,
            &job,
            ml_exp_core::cloud::pai::Allocation::Guaranteed,
            &check,
        )?;
        let current = self.sync_outbox.status(&job)?;
        let status = if current["retry_safe"] == true {
            self.sync_outbox.execute(
                copier,
                &job,
                ml_exp_core::cloud::pai::Allocation::Guaranteed,
                &format!("SUBMIT {}", Outbox::slot(&job)?),
                &check,
                &AtomicBool::new(false),
            )?
        } else {
            self.sync_outbox.reconcile(
                copier,
                &job,
                ml_exp_core::cloud::pai::Allocation::Guaranteed,
                &check,
                &AtomicBool::new(false),
            )?
        };
        let lock = self.lock()?;
        saved = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        saved.copy_status = Some(status);
        lock.validate()?;
        self.slots
            .replace_regular(
                OsStr::new(&Self::data_name(project, executor, asset)?),
                &encode(&saved)?,
            )
            .map_err(io)?;
        Ok(Self::data_view(saved))
    }
}
