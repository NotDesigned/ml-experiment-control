//! PAI reuses an existing READY Runtime and the collector's personal CPFS mount.
//! The original asset owns one durable copy intent; response loss only observes it.
use super::*;
use ml_exp_core::datasets::import::{Config, Source};

impl Service {
    pub(super) fn materialize_pai(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        request_id: &str,
        runtime_id: Option<&str>,
    ) -> cloud::Result<Value> {
        let backend = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        let collector = backend
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let saved = {
            let lock = self.lock()?;
            let mut saved = self
                .data_saved(project, executor, asset)?
                .filter(|s| s.archive_verified)
                .ok_or_else(missing)?;
            if let Some(runtime) = &saved.copy_runtime {
                if saved.copy_request_id.as_deref() != Some(request_id)
                    || runtime_id.is_some_and(|id| id != runtime.runtime_id)
                {
                    return Err(cloud::Error {
                        code: "DATA_COPY_IDENTITY_CONFLICT".into(),
                        http_status: Some(409),
                        uncertain: false,
                    });
                }
            } else {
                if saved.image_recipe.is_some() || saved.copy_request_id.is_some() {
                    return Err(cloud::Error::evidence());
                }
                let id = runtime_id.ok_or(cloud::Error {
                    code: "DATA_COPY_RUNTIME_REQUIRED".into(),
                    http_status: Some(422),
                    uncertain: false,
                })?;
                let runtime = self.runtime(project, executor, id)?;
                let storage = backend.definition.storage_layout()?;
                let config = Config {
                    root: backend.definition.data_root()?,
                    target: backend.definition.target()?,
                    project: project.into(),
                    definition: saved.state.definition.clone(),
                    source: Source::DatasetParts {
                        channel: self.data_channel(project, executor, asset)?,
                    },
                    timeout_seconds: 3600,
                };
                let channel = self
                    .exchange
                    .as_ref()
                    .ok_or_else(cloud::Error::configuration)?
                    .dataset_channel(&self.workspace, project, executor, asset, "ready")?;
                saved.copy_intent = Some(
                    ml_exp_core::cloud::data::materialization_job(
                        &runtime, &config, &channel, &storage, request_id,
                    )
                    .map_err(why!(cloud::Error::evidence()))?,
                );
                saved.copy_runtime = Some(runtime);
                saved.copy_request_id = Some(request_id.into());
                saved.copy_storage = Some(storage);
                // Both identity and resource intent precede outbox/provider effects.
                lock.validate()?;
                self.slots
                    .replace_regular(
                        OsStr::new(&Self::data_name(project, executor, asset)?),
                        &encode(&saved)?,
                    )
                    .map_err(io)?;
            }
            saved
        };
        let job = saved.copy_intent.ok_or_else(cloud::Error::evidence)?;
        let check = || self.project_hash(project).map(|_| ());
        let allocation = ml_exp_core::cloud::pai::Allocation::Guaranteed;
        self.sync_outbox
            .prepare(&collector.profile, &job, allocation, &check)?;
        let status = self.sync_outbox.status(&job)?;
        let status = if status["retry_safe"] == true {
            self.sync_outbox.execute(
                collector,
                &job,
                allocation,
                &format!("SUBMIT {}", Outbox::slot(&job)?),
                &check,
                &AtomicBool::new(false),
            )?
        } else {
            self.sync_outbox.reconcile(
                collector,
                &job,
                allocation,
                &check,
                &AtomicBool::new(false),
            )?
        };
        let lock = self.lock()?;
        let mut saved = self
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
