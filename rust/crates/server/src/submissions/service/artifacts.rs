//! Result effects with immutable bundle metadata and durable ACK intent. Payload
//! stays on the backend; download uses a separate bounded SSH byte stream.
use super::*;
use crate::backends::slurm::{CallOutcome, call};
use ml_exp_core::artifacts::{
    self,
    publication::{Command, State},
};
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    request_sha256: String,
    state: State,
}
impl Service {
    fn artifact_name(prepared: &Prepared) -> Result<String> {
        Ok(format!(
            "{}.artifacts.json",
            native::slot(&prepared.binding.project, &prepared.binding.run_id)?
        ))
    }
    fn artifact_saved(&self, prepared: &Prepared) -> Result<Option<State>> {
        let name = Self::artifact_name(prepared)?;
        if !self
            .slots
            .entry_exists(OsStr::new(&name))
            .map_err(|_| Error::Authority)?
        {
            return Ok(None);
        }
        let bytes = self
            .slots
            .read_regular(OsStr::new(&name), helper::MAX_METADATA_BYTES as u64)
            .map_err(|_| Error::Authority)?;
        let saved: Saved = serde_json::from_value(Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Authority)?,
        ))
        .map_err(|_| Error::Authority)?;
        if saved.request_sha256
            != prepared
                .request
                .identity()
                .map_err(|_| Error::Authority)?
                .request_sha256
        {
            return Err(Error::Authority);
        }
        saved.state.validate().map_err(|_| Error::Authority)?;
        saved
            .state
            .bundle
            .bind(prepared.request.intent(), None)
            .map_err(|_| Error::Authority)?;
        Ok(Some(saved.state))
    }
    fn artifact_save(&self, prepared: &Prepared, state: &State) -> Result<()> {
        state.validate().map_err(|_| Error::Authority)?;
        state
            .bundle
            .bind(prepared.request.intent(), None)
            .map_err(|_| Error::Authority)?;
        if let Some(old) = self.artifact_saved(prepared)? {
            old.validate_successor(state).map_err(|_| Error::Conflict)?;
        }
        let saved = Saved {
            request_sha256: prepared
                .request
                .identity()
                .map_err(|_| Error::Authority)?
                .request_sha256,
            state: state.clone(),
        };
        let bytes = serde_json::to_vec(&saved).map_err(|_| Error::Authority)?;
        if bytes.len() > helper::MAX_METADATA_BYTES {
            return Err(Error::Authority);
        }
        self.slots
            .replace_regular(OsStr::new(&Self::artifact_name(prepared)?), &bytes)
            .map_err(|_| Error::Uncertain)?;
        Ok(())
    }
    fn artifact_request(
        &self,
        prepared: &Prepared,
        command: Command,
    ) -> Result<(helper::Request, SshConfig)> {
        self.check(prepared)?;
        let action = prepared.plan["action_id"]
            .as_str()
            .ok_or(Error::Authority)?;
        let snapshot = self.actions.observe(action)?;
        if snapshot["execution"]["status"] != "VERIFIED" {
            return Err(Error::Conflict);
        }
        let job = snapshot["execution"]["result"]["submission"]["backend_job_id"]
            .as_str()
            .ok_or(Error::Authority)?;
        let request =
            helper::Request::artifacts(prepared.request.intent().clone(), job.into(), command)
                .map_err(|_| Error::Authority)?;
        let backend = self
            .backends
            .values()
            .find(|b| b.profile.target == request.intent().target)
            .ok_or(Error::Authority)?;
        Ok((request, backend.ssh.clone()))
    }
    pub fn artifacts(&self, action: &str) -> Result<Value> {
        let prepared = self.load(action)?;
        self.check(&prepared)?;
        let state = self
            .artifact_saved(&prepared)?
            .ok_or(Error::ArtifactsUnavailable)?;
        Ok(
            json!({"submission_id":action,"state":state,"download_available":state.available().map_err(|_|Error::Authority)?,"scheduler_mutation_performed":false}),
        )
    }
    pub fn collect_artifacts(&self, action: &str, cancelled: &AtomicBool) -> Result<Value> {
        self.artifact_change(action, Command::Collect, cancelled)
    }
    pub fn acknowledge_artifacts(
        &self,
        action: &str,
        proof: artifacts::DownloadProof,
        release: bool,
        now: &str,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        // Freeze the local acknowledgement intent before a possibly lost
        // remote release reply. Repeated ACK never regenerates an archive.
        self.artifact_change(
            action,
            Command::Acknowledge {
                proof,
                verified_at: now.into(),
                release,
            },
            cancelled,
        )
    }
    fn artifact_change(
        &self,
        action: &str,
        command: Command,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        let lock = self.lock()?;
        let prepared = self.load(action)?;
        let guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        if let Command::Acknowledge {
            proof,
            verified_at,
            release,
        } = &command
        {
            let state = self
                .artifact_saved(&prepared)?
                .ok_or(Error::ArtifactsUnavailable)?;
            let state = state
                .acknowledge(proof.clone(), verified_at, *release)
                .map_err(|_| Error::Conflict)?;
            self.artifact_save(&prepared, &state)?;
        }
        let (request, ssh) = self.artifact_request(&prepared, command)?;
        let outcome = call(&ssh, &request, cancelled).map_err(|_| Error::ArtifactsUnavailable)?;
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        let CallOutcome::Receipt { reply, .. } = outcome else {
            return Err(Error::ArtifactsUnavailable);
        };
        let helper::ReplyOutcome::Artifacts { publication: state } = reply.outcome else {
            return Err(Error::ArtifactsUnavailable);
        };
        lock.validate()?;
        self.check(&prepared)?;
        self.artifact_save(&prepared, &state)?;
        self.artifacts(action)
    }
    pub(crate) fn artifact_download(
        &self,
        action: &str,
        sha256: &str,
    ) -> Result<crate::backends::slurm::artifact_stream::Download> {
        let prepared = self.load(action)?;
        self.check(&prepared)?;
        let guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        let state = self
            .artifact_saved(&prepared)?
            .ok_or(Error::ArtifactsUnavailable)?;
        if !state.available().map_err(|_| Error::Authority)? {
            return Err(
                if state.retention.status == artifacts::RetentionStatus::Released {
                    Error::ArtifactsReleased
                } else {
                    Error::ArtifactsUnavailable
                },
            );
        }
        if state.bundle.archive.sha256 != sha256 {
            return Err(Error::Conflict);
        }
        let (request, ssh) = self.artifact_request(
            &prepared,
            Command::Download {
                sha256: sha256.into(),
            },
        )?;
        let stream =
            crate::backends::slurm::artifact_stream::Download::open(&ssh, &request, &state)
                .map_err(|_| Error::ArtifactsUnavailable)?;
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        self.check(&prepared)?;
        Ok(stream)
    }
    pub(crate) fn apply_artifacts(
        &self,
        snapshot: &mut crate::experiments::Snapshot,
    ) -> Result<()> {
        let Some(prepared) = self.saved(&snapshot.project, &snapshot.run_id)? else {
            return Ok(());
        };
        let Some(state) = self.artifact_saved(&prepared)? else {
            return Ok(());
        };
        crate::read_model::project_artifacts(snapshot, &prepared.binding, &state)
            .map_err(|_| Error::Authority)
    }
}
