//! Historical receipts are observations, never native submission authority.
use super::*;

fn unavailable() -> Error {
    error(
        409,
        "CHECKPOINT_EVIDENCE_UNAVAILABLE",
        "historical checkpoint, frozen manifest or storage binding cannot be validated",
    )
}
impl Api {
    /// `None` means a native-created Run; its original native service must
    /// validate its submission. Missing/corrupt native slots cannot fall back.
    pub fn historical_checkpoints(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        id: Option<&str>,
    ) -> Result<Option<Value>, Error> {
        self.observe(move |api| api.historical_checkpoints_observed(project, run, attempt, id))
    }
    fn historical_checkpoints_observed(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        id: Option<&str>,
    ) -> Result<Option<Value>, Error> {
        validate_id(project).map_err(|_| invalid_query())?;
        validate_id(run).map_err(|_| invalid_query())?;
        validate_attempt(attempt).map_err(|_| invalid_query())?;
        if let Some(id) = id {
            ml_exp_core::checkpoints::validate_checkpoint_id(id).map_err(|_| invalid_query())?;
        }
        self.validate_registry()?;
        self.ensure_active(project)?;
        let guard = self
            .fence
            .existing_observation_guard(project, run)
            .map_err(|_| unavailable())?;
        let (reference, discovery) = self.locate_run(project, run)?;
        let expected = experiments::ExpectedHashes {
            run: discovery.as_ref().and_then(|d| {
                d.runs
                    .iter()
                    .find(|r| r.reference.run_id == run)
                    .map(|r| r.raw_sha256.clone())
            }),
            attempt: None,
        };
        let record =
            self.store
                .attempt(&reference, attempt, &expected)
                .map_err(|error| match error {
                    experiments::Error::Io(ref cause)
                        if cause.kind() == std::io::ErrorKind::NotFound =>
                    {
                        super::error(404, "ATTEMPT_NOT_FOUND", "the exact Attempt is unavailable")
                    }
                    cause => cause.into(),
                })?;
        if record.run.metadata.contains_key("native_creation") {
            return Ok(None);
        }
        let values = crate::checkpoints::historical::observe(&self.registry, &record, attempt, id)
            .map_err(|_| unavailable())?;
        self.store.attempt(
            &reference,
            attempt,
            &experiments::ExpectedHashes {
                run: Some(record.run.raw_sha256),
                attempt: Some(record.attempt.raw_sha256),
            },
        )?;
        guard
            .require_observed_live(project, run)
            .map_err(|_| unavailable())?;
        if let Some(before) = discovery {
            if !before.same_bindings(&self.store.discover(project, &Default::default())?) {
                return Err(unavailable());
            }
        }
        self.ensure_active(project)?;
        self.validate_registry()?;
        let value = match id {
            Some(_) => values.into_iter().next().ok_or_else(|| {
                error(
                    404,
                    "CHECKPOINT_NOT_FOUND",
                    "the exact registered checkpoint is unavailable",
                )
            })?,
            None => {
                serde_json::json!({"project":project,"run_id":run,"attempt_id":attempt,"checkpoints":values})
            }
        };
        read_model::encoded_size(&value)?;
        Ok(Some(value))
    }
}
