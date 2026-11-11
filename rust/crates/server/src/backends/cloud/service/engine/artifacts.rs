//! Receipt-only cloud results. CPU collectors read existing persistent output;
//! neither collection nor download can replay the training command.
use super::*;
use ml_exp_core::{
    artifacts::{
        self,
        publication::{Bundle, State},
    },
    cloud::exchange::{ArtifactOperation, ArtifactRequest},
};
use subtle::ConstantTimeEq;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Saved {
    package_sha256: String,
    pub(super) bundle: Bundle,
    pub(super) state: Option<State>,
    pub(super) released_parts: u64,
    pub(super) administrative_release: Option<String>,
    /// The worker's CRC-64 of each part, which matched the object store's at
    /// publication. Empty for an archive published by reading it back.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) part_crc64: Vec<u64>,
}
impl Service {
    pub(super) fn apply_artifacts(
        &self,
        snapshot: &mut crate::experiments::Snapshot,
    ) -> cloud::Result<()> {
        let ids = snapshot
            .attempts
            .iter()
            .map(|a| a.attempt_id.clone())
            .collect::<Vec<_>>();
        for id in ids {
            let Some(prepared) = self.saved_attempt(&snapshot.project, &snapshot.run_id, &id)?
            else {
                continue;
            };
            let Some(saved) = self.artifact_saved(&prepared)? else {
                continue;
            };
            if saved.administrative_release.is_some() {
                continue;
            }
            if let Some(state) = saved.state {
                crate::read_model::project_artifacts(snapshot, &prepared.execution.binding, &state)
                    .map_err(why!(cloud::Error::evidence()))?;
            }
        }
        Ok(())
    }
    pub fn has_artifacts(&self) -> bool {
        self.objects.is_some() && self.has_exchange()
    }
    fn artifact_name(prepared: &Prepared) -> cloud::Result<String> {
        Ok(format!(
            "{}.artifacts.json",
            Self::attempt_slot(
                &prepared.execution.job.project,
                &prepared.execution.job.run_id,
                &prepared.execution.job.attempt_id
            )?
        ))
    }
    fn artifact_binding(&self, prepared: &Prepared, bundle: &Bundle) -> cloud::Result<String> {
        let package = prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        let sha = helper::sha256_bytes(&encode(&package)?);
        bundle.validate().map_err(why!(cloud::Error::evidence()))?;
        let archive = &bundle.archive;
        let job = &prepared.execution.job;
        if bundle.intent_sha256 != sha
            || bundle.config_sha256 != helper::sha256_bytes(&encode(&package.config)?)
            || archive.project != job.project
            || archive.run_id != job.run_id
            || archive.attempt_id != job.attempt_id
            || !package.result_upload
            || crate::objects::parts(archive).is_err()
        {
            return Err(cloud::Error::evidence());
        }
        package
            .artifacts
            .unwrap_or_default()
            .check(&archive.files)
            .map_err(why!(cloud::Error::evidence()))?;
        Ok(sha)
    }
    pub(super) fn artifact_saved(&self, prepared: &Prepared) -> cloud::Result<Option<Saved>> {
        let name = Self::artifact_name(prepared)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(None);
        }
        let saved: Saved = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(why!(cloud::Error::evidence()))?;
        let parts = crate::objects::parts(&saved.bundle.archive).map_err(io)?;
        if self.artifact_binding(prepared, &saved.bundle)? != saved.package_sha256
            || saved.released_parts > parts
            || !saved.part_crc64.is_empty()
                && (saved.part_crc64.len() as u64 != parts || saved.state.is_none())
        {
            return Err(cloud::Error::evidence());
        }
        if let Some(state) = &saved.state {
            state.validate().map_err(why!(cloud::Error::evidence()))?;
            if state.bundle != saved.bundle
                || saved.released_parts > 0
                    && state.retention.status == artifacts::RetentionStatus::Keep
                    && saved.administrative_release.is_none()
            {
                return Err(cloud::Error::evidence());
            }
        } else if saved.released_parts > 0 && saved.administrative_release.is_none() {
            return Err(cloud::Error::evidence());
        }
        Ok(Some(saved))
    }
    pub(super) fn artifact_save(&self, prepared: &Prepared, saved: &Saved) -> cloud::Result<()> {
        self.slots
            .replace_regular(OsStr::new(&Self::artifact_name(prepared)?), &encode(saved)?)
            .map_err(why!(cloud::Error {
                code: "CLOUD_ARTIFACT_COMMIT_UNCERTAIN".into(),
                http_status: None,
                uncertain: true,
            }))
    }
    pub(super) fn worker_authority(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
    ) -> cloud::Result<Prepared> {
        let denied = || cloud::Error {
            code: "WORKER_AUTH_REQUIRED".into(),
            http_status: Some(401),
            uncertain: false,
        };
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(denied)?;
        let channel = self
            .exchange
            .as_ref()
            .ok_or_else(denied)?
            .channel(&self.workspace, &prepared.execution)?;
        if !bool::from(channel.token.as_bytes().ct_eq(token.as_bytes())) {
            return Err(denied());
        }
        self.require_live(project, run)?;
        if self
            .actions
            .observe(
                prepared.plan["action_id"]
                    .as_str()
                    .ok_or_else(cloud::Error::evidence)?,
            )
            .map_err(op)?["execution"]["status"]
            != "VERIFIED"
        {
            return Err(cloud::Error::evidence());
        }
        Ok(prepared)
    }
    pub fn receive_artifacts(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
        request: ArtifactRequest,
        now: &str,
    ) -> cloud::Result<Value> {
        let lock = self.lock("artifacts.receive_artifacts#1")?;
        let prepared = self.worker_authority(project, run, attempt, token)?;
        let objects = self
            .objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let package = prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        if request.package_sha256 != helper::sha256_bytes(&encode(&package)?) {
            return Err(cloud::Error::evidence());
        }
        // A provider success is insufficient. Require the exact worker's sealed
        // terminal receipt before accepting any output identity.
        let observed = self
            .exchange_saved(&prepared)?
            .ok_or_else(cloud::Error::evidence)?;
        if observed.evidence.worker.phase != ml_exp_core::managed::evidence::Phase::Finished
            || observed.evidence.worker.receipt.is_none()
        {
            return Err(cloud::Error::evidence());
        }
        let mut saved = self.artifact_saved(&prepared)?;
        if saved
            .as_ref()
            .is_some_and(|s| s.administrative_release.is_some())
        {
            return Err(cloud::Error {
                code: "ARTIFACTS_ADMINISTRATIVELY_REMOVED".into(),
                http_status: Some(410),
                uncertain: false,
            });
        }
        if let ArtifactOperation::Prepare { bundle } = &request.operation {
            self.artifact_binding(&prepared, bundle)?;
            if let Some(saved) = &saved {
                if &saved.bundle != bundle.as_ref() {
                    return Err(cloud::Error::evidence());
                }
            } else {
                let pending = Saved {
                    package_sha256: request.package_sha256.clone(),
                    bundle: bundle.as_ref().clone(),
                    state: None,
                    released_parts: 0,
                    administrative_release: None,
                    part_crc64: Vec::new(),
                };
                self.artifact_save(&prepared, &pending)?;
                saved = Some(pending);
            }
        }
        let saved = saved.ok_or_else(cloud::Error::evidence)?;
        let published = saved.state.is_some();
        match request.operation {
            ArtifactOperation::Prepare { .. } => Ok(
                json!({"published":published,"released":saved.state.as_ref().is_some_and(|s|s.retention.status!=artifacts::RetentionStatus::Keep),"part_bytes":ml_exp_core::multipart::PART_BYTES,"part_count":crate::objects::parts(&saved.bundle.archive).map_err(io)?}),
            ),
            ArtifactOperation::Ticket { part } => {
                if published {
                    return Err(cloud::Error::evidence());
                }
                let now = time::OffsetDateTime::parse(
                    now,
                    &time::format_description::well_known::Rfc3339,
                )
                .map_err(why!(cloud::Error::configuration()))?
                .unix_timestamp();
                objects
                    .ticket(&saved.bundle.archive, part, true, now)
                    .map_err(io)
            }
            ArtifactOperation::Complete { part_crc64 } => {
                if !published {
                    // No lock while the object store is asked: holding it answered
                    // every other job and command with BUSY.
                    drop(lock);
                    if part_crc64.is_empty() {
                        // A frozen worker from before part CRCs, which cannot be
                        // replaced: read the whole archive back, as it always was.
                        objects.verify(&saved.bundle.archive).map_err(io)?;
                    } else if !objects
                        .wrong_archive_parts(&saved.bundle.archive, &part_crc64)
                        .map_err(io)?
                        .is_empty()
                    {
                        // A part that is missing or differs from what the worker sent.
                        // Nothing is published; the worker's archive and outputs stay.
                        return Err(cloud::Error {
                            code: "ARTIFACT_PART_REJECTED".into(),
                            http_status: Some(409),
                            uncertain: false,
                        });
                    }
                    let _lock = self.lock("artifacts.receive_artifacts#2")?;
                    let mut current = self
                        .artifact_saved(&prepared)?
                        .ok_or_else(cloud::Error::evidence)?;
                    if current.bundle != saved.bundle {
                        return Err(cloud::Error::evidence());
                    }
                    if current.administrative_release.is_some() {
                        return Err(cloud::Error {
                            code: "ARTIFACTS_ADMINISTRATIVELY_REMOVED".into(),
                            http_status: Some(410),
                            uncertain: false,
                        });
                    }
                    if current.state.is_none() {
                        current.state = Some(
                            State::new(current.bundle.clone())
                                .map_err(why!(cloud::Error::evidence()))?,
                        );
                        current.part_crc64 = part_crc64;
                        self.artifact_save(&prepared, &current)?;
                    }
                }
                Ok(
                    json!({"published":true,"sha256":saved.bundle.archive.sha256,"bytes":saved.bundle.archive.bytes}),
                )
            }
        }
    }
    pub fn artifacts(&self, action: &str) -> cloud::Result<Value> {
        let prepared = self.load(action)?;
        let unpublished = || cloud::Error {
            code: "ARTIFACT_NOT_PUBLISHED".into(),
            http_status: Some(409),
            uncertain: false,
        };
        let saved = self.artifact_saved(&prepared)?.ok_or_else(unpublished)?;
        if saved.administrative_release.is_some() {
            return Err(cloud::Error {
                code: "ARTIFACTS_ADMINISTRATIVELY_REMOVED".into(),
                http_status: Some(410),
                uncertain: false,
            });
        }
        let state = saved.state.ok_or_else(unpublished)?;
        Ok(
            json!({"submission_id":action,"state":state,"download_available":state.available().map_err(why!(cloud::Error::evidence()))?,"scheduler_mutation_performed":false,
            "download":{"kind":"oss-parts","part_bytes":ml_exp_core::multipart::PART_BYTES,"part_count":crate::objects::parts(&state.bundle.archive).map_err(io)?},"verification":"SERVER_VERIFIED"}),
        )
    }
    pub fn download_ticket(&self, action: &str, part: u64, now: &str) -> cloud::Result<Value> {
        let prepared = self.load(action)?;
        let saved = self.artifact_saved(&prepared)?.ok_or_else(missing)?;
        if saved.administrative_release.is_some() {
            return Err(cloud::Error {
                code: "ARTIFACTS_ADMINISTRATIVELY_REMOVED".into(),
                http_status: Some(410),
                uncertain: false,
            });
        }
        let state = saved.state.ok_or_else(missing)?;
        if !state.available().map_err(why!(cloud::Error::evidence()))? {
            return Err(missing());
        }
        let timestamp =
            time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
                .map_err(why!(cloud::Error::configuration()))?
                .unix_timestamp();
        let mut ticket = self
            .objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .ticket(&state.bundle.archive, part, false, timestamp)
            .map_err(io)?;
        // Lets the client check each part as it arrives. It checks the archive and
        // every file's SHA256 before any acknowledgement either way.
        if let Some(crc) = saved.part_crc64.get(part as usize) {
            ticket["crc64"] = json!(crc);
        }
        Ok(ticket)
    }
    pub fn collect_artifacts(
        &self,
        action: &str,
        now: &str,
        cancelled: &AtomicBool,
    ) -> cloud::Result<Value> {
        let prepared = self.load(action)?;
        if let Some(saved) = self.artifact_saved(&prepared)? {
            if saved.state.is_some() || saved.administrative_release.is_some() {
                return self.artifacts(action);
            }
        }
        let id = canonical::digest(&json!(["native-cloud-collect.v1", action]))
            .map_err(why!(cloud::Error::evidence()))?;
        let request = ml_exp_core::cloud::exchange::Sync {
            request_id: id[..32].into(),
            confirmation: format!("SYNC {action}"),
        };
        let sync = self.synchronize(action, &request, now, cancelled)?;
        Ok(
            json!({"submission_id":action,"status":"COLLECTING","download_available":false,"scheduler_mutation_performed":true,"training_replayed":false,"synchronization":sync["synchronization"]}),
        )
    }
    pub fn acknowledge_artifacts(
        &self,
        action: &str,
        proof: artifacts::DownloadProof,
        release: bool,
        now: &str,
    ) -> cloud::Result<Value> {
        let lock = self.lock("artifacts.acknowledge_artifacts")?;
        let prepared = self.load(action)?;
        let mut saved = self.artifact_saved(&prepared)?.ok_or_else(missing)?;
        if saved.administrative_release.is_some() {
            return Err(cloud::Error {
                code: "ARTIFACTS_ADMINISTRATIVELY_REMOVED".into(),
                http_status: Some(410),
                uncertain: false,
            });
        }
        let state = saved
            .state
            .as_ref()
            .ok_or_else(missing)?
            .acknowledge(proof, now, release)
            .map_err(why!(cloud::Error::evidence()))?;
        saved.state = Some(state);
        self.artifact_save(&prepared, &saved)?;
        if release
            && saved.state.as_ref().ok_or_else(missing)?.retention.status
                != artifacts::RetentionStatus::Released
        {
            let state = saved.state.as_mut().ok_or_else(missing)?;
            state.retention = artifacts::begin_release(&state.bundle.archive, &state.retention)
                .map_err(why!(cloud::Error::evidence()))?;
            self.artifact_save(&prepared, &saved)?;
            let total = crate::objects::parts(&saved.bundle.archive).map_err(io)?;
            // Bounded work per call; repeated identical ACKs continue the same
            // cursor, including after a lost response or process restart.
            let end = total.min(saved.released_parts.saturating_add(64));
            while saved.released_parts < end {
                self.objects
                    .as_ref()
                    .ok_or_else(cloud::Error::configuration)?
                    .release(&saved.bundle.archive, saved.released_parts)
                    .map_err(io)?;
                saved.released_parts += 1;
                self.artifact_save(&prepared, &saved)?;
            }
            if saved.released_parts == total {
                let state = saved.state.as_mut().ok_or_else(missing)?;
                state.retention =
                    artifacts::finish_release(&state.bundle.archive, &state.retention, now)
                        .map_err(why!(cloud::Error::evidence()))?;
                self.artifact_save(&prepared, &saved)?;
            }
        }
        drop(lock);
        self.artifacts(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frozen worker from before part CRCs still completes; a current one names them.
    #[test]
    fn complete_from_an_old_worker_carries_no_crc_and_a_new_one_does() {
        let old: ArtifactOperation = serde_json::from_value(json!({"kind":"complete"})).unwrap();
        assert!(matches!(old, ArtifactOperation::Complete { part_crc64 } if part_crc64.is_empty()));
        let new: ArtifactOperation =
            serde_json::from_value(json!({"kind":"complete","part_crc64":[1,2]})).unwrap();
        assert!(matches!(new, ArtifactOperation::Complete { part_crc64 } if part_crc64 == [1, 2]));
        assert!(
            serde_json::from_value::<ArtifactOperation>(json!({"kind":"complete","other":1}))
                .is_err()
        );
        // What an old server would be sent by a worker with no CRCs is unchanged.
        assert_eq!(
            serde_json::to_value(ArtifactOperation::Complete { part_crc64: vec![] }).unwrap(),
            json!({"kind":"complete"})
        );
    }
}
