//! Explicit post-training collection and acknowledged release. Only a generated
//! archive in the exact backend Attempt is removed; original outputs and state stay.
use super::*;
use ml_exp_core::artifacts::{
    self, RetentionStatus,
    publication::{Bundle, Command, Policy, State},
};
const DIRECTORY: &str = "artifacts";
const ARCHIVE: &str = "archive.tar";
const BUNDLE: &str = "bundle.json";
const STATE: &str = "state.json";
fn read_state(directory: &Directory) -> Result<State> {
    let bytes = directory.read_regular(OsStr::new(STATE), helper::MAX_METADATA_BYTES as u64)?;
    let state: State = serde_json::from_value(serde_json::Value::Object(
        ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Metadata)?,
    ))?;
    state.validate().map_err(|_| Error::Metadata)?;
    let bundle: Bundle = serde_json::from_slice(
        &directory.read_regular(OsStr::new(BUNDLE), helper::MAX_METADATA_BYTES as u64)?,
    )?;
    if bundle != state.bundle {
        return Err(Error::Metadata);
    }
    Ok(state)
}
fn save(directory: &Directory, state: &State) -> Result<()> {
    state.validate().map_err(|_| Error::Metadata)?;
    directory.replace_regular(OsStr::new(STATE), &json(state)?)?;
    Ok(())
}
impl Helper {
    fn result_directories(
        &self,
        request: &Request,
        snapshot: &Snapshot,
    ) -> Result<(ml_exp_core::slurm::staging::Plan, Vec<Directory>)> {
        if request.intent().target != self.config.target
            || request.requested_job_id() != snapshot.receipt.as_ref().map(|r| r.job_id.as_str())
        {
            return Err(Error::Metadata);
        }
        self.verify_issued(request.intent(), &snapshot.directory)?;
        let (plan, directories) = self
            .observe_shared(request.intent(), &snapshot.directory, false)?
            .ok_or(Error::Configuration)?;
        let package = plan.managed.as_ref().ok_or(Error::Configuration)?;
        let worker = crate::managed::sample(&package.config).map_err(|_| Error::Metadata)?;
        if worker.phase != ml_exp_core::managed::evidence::Phase::Finished
            || worker.receipt.is_none()
        {
            return Err(Error::Metadata);
        }
        Ok((plan, directories))
    }
    fn result_state(
        &self,
        request: &Request,
        snapshot: &Snapshot,
        collect: bool,
    ) -> Result<(Directory, State)> {
        let (plan, directories) = self.result_directories(request, snapshot)?;
        let directory = directories[3].child(OsStr::new(DIRECTORY), collect)?;
        let intent = request.intent().digest()?;
        if directory.entry_exists(OsStr::new("owner.json"))? {
            if directory.read_regular(OsStr::new("owner.json"), 1024)? != json(&intent)? {
                return Err(Error::Metadata);
            }
        } else if collect {
            // A partial/foreign directory is not adopted. Only our new empty
            // directory can be given the owner marker.
            if !directory.entry_names_limited(1)?.is_empty() {
                return Err(Error::Metadata);
            }
            directory.publish_new(OsStr::new("owner.json"), &json(&intent)?)?;
        } else {
            return Err(Error::Metadata);
        }
        if !directory.entry_exists(OsStr::new(STATE))? {
            if !collect {
                return Err(Error::Metadata);
            }
            let bundle = if directory.entry_exists(OsStr::new(BUNDLE))? {
                serde_json::from_slice::<Bundle>(
                    &directory
                        .read_regular(OsStr::new(BUNDLE), helper::MAX_METADATA_BYTES as u64)?,
                )?
            } else {
                // Exact owned incomplete archive, under the exclusive helper
                // lock. Never remove a published archive to make space.
                if directory.entry_exists(OsStr::new(ARCHIVE))? {
                    let file = directory.open_regular_read(OsStr::new(ARCHIVE))?;
                    directory.remove_staged_regular(OsStr::new(ARCHIVE), &file)?;
                }
                let payload = directories[5].child(OsStr::new("payload"), false)?;
                let attempt = snapshot
                    .directory
                    .read_regular(OsStr::new(MANIFEST), helper::MAX_METADATA_BYTES as u64)?;
                let value = serde_json::Value::Object(
                    ml_exp_core::json::mapping(&attempt).map_err(|_| Error::Metadata)?,
                );
                let policy = if let Some(value) = value.get("artifacts") {
                    serde_json::from_value::<Policy>(value.clone())?
                } else {
                    Policy::default()
                };
                let required = ml_exp_io::artifacts::estimate(&payload)?;
                let capacity = directory.capacity()?;
                if capacity.available_bytes < required || capacity.available_inodes < 2 {
                    return Err(Error::Storage(std::io::Error::other(
                        "result archive storage is insufficient",
                    )));
                }
                let mut file = directory.create_new_regular(OsStr::new(ARCHIVE))?;
                let packed = ml_exp_io::artifacts::pack(&payload, &mut file, &policy);
                let (sha256, bytes, files) = match packed {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = directory.remove_staged_regular(OsStr::new(ARCHIVE), &file);
                        return Err(e.into());
                    }
                };
                directory.seal_regular(OsStr::new(ARCHIVE), &file, false)?;
                let scope = &request.intent().scope;
                let bundle = Bundle {
                    contract: artifacts::publication::CONTRACT.into(),
                    intent_sha256: intent.clone(),
                    config_sha256: helper::sha256_bytes(
                        &plan
                            .managed
                            .as_ref()
                            .ok_or(Error::Metadata)?
                            .config_bytes()?,
                    ),
                    archive: artifacts::ArchiveReceipt {
                        project: scope.project.clone(),
                        run_id: scope.run_id.clone(),
                        attempt_id: scope.attempt_id.clone(),
                        object_key: format!(
                            "{}/{}/{}/{}.tar",
                            scope.project, scope.run_id, scope.attempt_id, sha256
                        ),
                        sha256,
                        bytes,
                        files,
                    },
                };
                bundle.validate().map_err(|_| Error::Metadata)?;
                directory.publish_new(OsStr::new(BUNDLE), &json(&bundle)?)?;
                let file = directory.open_regular_read(OsStr::new(BUNDLE))?;
                directory.seal_regular(OsStr::new(BUNDLE), &file, false)?;
                bundle
            };
            let state = State::new(bundle).map_err(|_| Error::Metadata)?;
            directory.publish_new(OsStr::new(STATE), &json(&state)?)?;
        }
        let state = read_state(&directory)?;
        state
            .bundle
            .bind(
                request.intent(),
                Some(&helper::sha256_bytes(
                    &plan
                        .managed
                        .as_ref()
                        .ok_or(Error::Metadata)?
                        .config_bytes()?,
                )),
            )
            .map_err(|_| Error::Metadata)?;
        self.verify_issued(request.intent(), &snapshot.directory)?;
        Ok((directory, state))
    }
    pub(super) fn artifact_operation(
        &self,
        request: &Request,
        snapshot: &Snapshot,
        lock: &Lock,
        cancelled: &AtomicBool,
    ) -> Result<ReplyOutcome> {
        lock.validate()?;
        let result = || -> Result<State> {
            let command = request.artifact_command().ok_or(Error::Metadata)?;
            let (directory, mut state) =
                self.result_state(request, snapshot, matches!(command, Command::Collect))?;
            if cancelled.load(Ordering::Relaxed) {
                return Err(Error::Metadata);
            }
            match command {
                Command::Collect => {}
                Command::Download { .. } => return Err(Error::Metadata),
                Command::Acknowledge {
                    proof,
                    verified_at,
                    release,
                } => {
                    state = state
                        .acknowledge(proof.clone(), verified_at, *release)
                        .map_err(|_| Error::Metadata)?;
                    save(&directory, &state)?;
                    if matches!(
                        state.retention.status,
                        RetentionStatus::Scheduled | RetentionStatus::Releasing
                    ) {
                        state.retention =
                            artifacts::begin_release(&state.bundle.archive, &state.retention)
                                .map_err(|_| Error::Metadata)?;
                        save(&directory, &state)?;
                        if directory.entry_exists(OsStr::new(ARCHIVE))? {
                            let file = directory.open_regular_read(OsStr::new(ARCHIVE))?;
                            directory
                                .validate_immutable_regular_entry(OsStr::new(ARCHIVE), &file)?;
                            if file.metadata()?.len() != state.bundle.archive.bytes {
                                return Err(Error::Metadata);
                            }
                            directory.remove_staged_regular(OsStr::new(ARCHIVE), &file)?;
                        }
                        state.retention = artifacts::finish_release(
                            &state.bundle.archive,
                            &state.retention,
                            verified_at,
                        )
                        .map_err(|_| Error::Metadata)?;
                        save(&directory, &state)?;
                    }
                }
            }
            directory.verify_private_parent()?;
            Ok(state)
        };
        let outcome = match result() {
            Ok(state) => ReplyOutcome::Artifacts {
                publication: Box::new(state),
            },
            Err(_) => ReplyOutcome::Unknown {
                reason: UnknownReason::StorageUnavailable,
            },
        };
        lock.validate()?;
        Ok(outcome)
    }
    /// A retained immutable file descriptor outlives the metadata lock. A slow
    /// download cannot block scheduler operations or authorize another submit.
    pub fn artifact_download(
        &self,
        request: &Request,
        cancelled: &AtomicBool,
    ) -> Result<(Reply, File)> {
        request.validate()?;
        let Command::Download { sha256 } = request.artifact_command().ok_or(Error::Metadata)?
        else {
            return Err(Error::Metadata);
        };
        let lock = self.lock(cancelled)?;
        let ReadLedger::Found(snapshot) = self.read_ledger(request.intent())? else {
            return Err(Error::Metadata);
        };
        let (directory, state) = self.result_state(request, &snapshot, false)?;
        if !state.available().map_err(|_| Error::Metadata)?
            || state.bundle.archive.sha256 != *sha256
        {
            return Err(Error::Metadata);
        }
        let file = directory.open_regular_read(OsStr::new(ARCHIVE))?;
        directory.validate_immutable_regular_entry(OsStr::new(ARCHIVE), &file)?;
        if file.metadata()?.len() != state.bundle.archive.bytes {
            return Err(Error::Metadata);
        }
        lock.validate()?;
        Ok((
            Reply::for_request(
                request,
                ReplyOutcome::Artifacts {
                    publication: Box::new(state),
                },
            )?,
            file,
        ))
    }
}
