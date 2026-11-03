//! Original result inventories remain readable without creating a native
//! Submission, unsealing a transfer or restoring a payload cache.
use super::*;

fn unavailable() -> Error {
    error(
        409,
        "ARTIFACT_EVIDENCE_UNAVAILABLE",
        "historical result receipt or frozen Attempt cannot be validated",
    )
}

fn transfer(
    root: &Directory,
    project: &str,
    run: &str,
    attempt: &str,
) -> Result<Option<Value>, Error> {
    let mut directory = root.clone();
    for part in ["artifact-transfers", project, run] {
        if !directory
            .entry_exists(OsStr::new(part))
            .map_err(|_| unavailable())?
        {
            return Ok(None);
        }
        directory = directory
            .child(OsStr::new(part), false)
            .map_err(|_| unavailable())?;
        directory
            .verify_private_parent()
            .map_err(|_| unavailable())?;
    }
    let name = format!("{attempt}.json");
    if !directory
        .entry_exists(OsStr::new(&name))
        .map_err(|_| unavailable())?
    {
        return Ok(None);
    }
    let bytes = directory
        .read_regular(OsStr::new(&name), 8 * 1024 * 1024)
        .map_err(|_| unavailable())?;
    let value = ml_exp_core::json::mapping(&bytes).map_err(|_| unavailable())?;
    Ok(Some(Value::Object(value)))
}

impl Api {
    pub fn historical_artifact_ticket(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        now: i64,
    ) -> Result<Option<Value>, Error> {
        validate_id(project).map_err(|_| invalid_query())?;
        validate_observation_id(run).map_err(|_| invalid_query())?;
        validate_attempt(attempt).map_err(|_| invalid_query())?;
        if let Some(history) = self.history_for(project, run)? {
            return history.historical_artifact_ticket(project, run, attempt, now);
        }
        self.observe(|api| {
            let Some(history) = &api.historical else {
                return Ok(None);
            };
            let guard = api
                .fence
                .existing_observation_guard(project, run)
                .map_err(|_| unavailable())?;
            let inventory = api
                .historical_artifacts(project, run, attempt)?
                .ok_or_else(unavailable)?;
            match inventory["storage_status"].as_str() {
                Some("RELEASED" | "EXPIRED") => {
                    return Err(error(
                        410,
                        "ARTIFACT_RELEASED",
                        "the published archive has been released or expired",
                    ));
                }
                Some("AVAILABLE") => {}
                _ => {
                    return Err(error(
                        404,
                        "ARTIFACT_UNAVAILABLE",
                        "this Attempt has no published archive",
                    ));
                }
            }
            let profile = history.artifact_store.as_ref().ok_or(error(
                503,
                "HISTORICAL_OBJECT_STORE_NOT_CONFIGURED",
                "the original object store profile is not configured for read-only download",
            ))?;
            let ticket = profile.ticket(&inventory, now).map_err(|_| unavailable())?;
            if api.historical_artifacts(project, run, attempt)?.as_ref() != Some(&inventory) {
                return Err(unavailable());
            }
            guard
                .require_observed_live(project, run)
                .map_err(|_| unavailable())?;
            Ok(Some(ticket))
        })
    }

    pub fn historical_artifacts(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
    ) -> Result<Option<Value>, Error> {
        if let Some(history) = self.history_for(project, run)? {
            return history.historical_artifacts(project, run, attempt);
        }
        self.observe(|api| {
            validate_id(project).map_err(|_| invalid_query())?;
            validate_observation_id(run).map_err(|_| invalid_query())?;
            validate_attempt(attempt).map_err(|_| invalid_query())?;
            api.validate_registry()?;
            api.ensure_active(project)?;
            let guard = api
                .fence
                .existing_observation_guard(project, run)
                .map_err(|_| unavailable())?;
            let (reference, discovery) = api.locate_run(project, run)?;
            let expected = experiments::ExpectedHashes {
                run: discovery.as_ref().and_then(|d| {
                    d.runs
                        .iter()
                        .find(|r| r.reference.run_id == run)
                        .map(|r| r.raw_sha256.clone())
                }),
                attempt: None,
            };
            let record = api
                .store
                .attempt(&reference, attempt, &expected)
                .map_err(|cause| match cause {
                    experiments::Error::Io(ref cause)
                        if cause.kind() == std::io::ErrorKind::NotFound =>
                    {
                        error(404, "ATTEMPT_NOT_FOUND", "the exact Attempt is unavailable")
                    }
                    cause => cause.into(),
                })?;
            if record.run.metadata.contains_key("native_creation") {
                return Ok(None);
            }
            let before = transfer(&api.registry, project, run, attempt)?;
            let value = ml_exp_core::artifacts::historical::inventory(
                before.as_ref(),
                project,
                run,
                attempt,
                record
                    .run
                    .metadata
                    .get("source_id")
                    .and_then(Value::as_str)
                    .ok_or_else(unavailable)?,
            )
            .map_err(|_| unavailable())?;
            if before != transfer(&api.registry, project, run, attempt)? {
                return Err(unavailable());
            }
            api.store.attempt(
                &reference,
                attempt,
                &experiments::ExpectedHashes {
                    run: Some(record.run.raw_sha256),
                    attempt: Some(record.attempt.raw_sha256),
                },
            )?;
            if let Some(before) = discovery {
                if !before.same_bindings(&api.store.discover(project, &Default::default())?) {
                    return Err(unavailable());
                }
            }
            guard
                .require_observed_live(project, run)
                .map_err(|_| unavailable())?;
            api.ensure_active(project)?;
            api.validate_registry()?;
            read_model::encoded_size(&value)?;
            Ok(Some(value))
        })
    }
}
