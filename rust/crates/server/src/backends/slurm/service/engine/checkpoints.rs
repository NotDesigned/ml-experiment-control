//! Exact remote verification and metadata-only registration. Checkpoint payloads
//! stay on the backend. Local GET never hashes or downloads remote files.
use super::*;
use crate::backends::slurm::{CallOutcome, call};
use ml_exp_core::checkpoints::{self, Context};

impl Service {
    fn checkpoint_context(&self, prepared: &Prepared) -> Result<Context> {
        let (manifest, _) = prepared
            .request
            .prepared_content()
            .ok_or(Error::Authority)?;
        ml_exp_core::slurm::staging::checkpoint_context(
            prepared.request.intent(),
            manifest.as_bytes(),
        )
        .map_err(why!(Error::CheckpointConflict))
    }
    pub fn checkpoints(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        id: Option<&str>,
    ) -> Result<Value> {
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or(Error::Missing)?;
        self.require_live(project, run)?;
        if prepared.binding.attempt_id != attempt {
            return Err(Error::CheckpointMissing);
        }
        let context = self.checkpoint_context(&prepared)?;
        let store = crate::checkpoints::Store::open_existing(&self.settings.registry_root)
            .map_err(why!(Error::CheckpointConflict))?;
        match id {
            Some(id) => store
                .read(&context, id)
                .map_err(why!(Error::CheckpointConflict))?
                .ok_or(Error::CheckpointMissing),
            None => Ok(
                json!({"project":project,"run_id":run,"attempt_id":attempt,"checkpoints":store.list(&context).map_err(why!(Error::CheckpointConflict))?}),
            ),
        }
    }
    /// The Run's checkpoints: those of its current Attempt, or the one checkpoint `id`
    /// of whichever Attempt registered it. The answer names that Attempt.
    pub fn run_checkpoints(&self, project: &str, run: &str, id: Option<&str>) -> Result<Value> {
        let attempt = match id {
            None => {
                self.saved(project, run)?
                    .ok_or(Error::Missing)?
                    .binding
                    .attempt_id
            }
            Some(id) => {
                let mut reference = checkpoints::Reference {
                    run_id: run.into(),
                    attempt_id: String::new(),
                    checkpoint_id: id.into(),
                };
                if !crate::checkpoints::resolve(
                    &self.settings.registry_root,
                    project,
                    &mut reference,
                )
                .map_err(why!(Error::CheckpointConflict))?
                {
                    return Err(Error::CheckpointMissing);
                }
                reference.attempt_id
            }
        };
        self.checkpoints(project, run, &attempt, id)
    }
    pub fn refresh_checkpoint(
        &self,
        action: &str,
        now: &str,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        let _lock = self.lock()?;
        let prepared = self.load(action)?;
        let guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(why!(Error::Authority))?;
        let snapshot = self.actions.observe(action)?;
        if snapshot["execution"]["status"] != "VERIFIED" {
            return Err(Error::Conflict);
        }
        let job = snapshot["execution"]["result"]["submission"]["backend_job_id"]
            .as_str()
            .ok_or(Error::Authority)?;
        let request =
            helper::Request::observe_checkpoint(prepared.request.intent().clone(), job.into())
                .map_err(why!(Error::Authority))?;
        let backend = self
            .backends
            .values()
            .find(|b| b.profile.target == request.intent().target)
            .ok_or(Error::Authority)?;
        let outcome =
            call(&backend.ssh, &request, cancelled).map_err(why!(Error::CheckpointUnavailable))?;
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(why!(Error::Authority))?;
        let CallOutcome::Receipt { reply, .. } = outcome else {
            return Err(Error::CheckpointUnavailable);
        };
        let helper::ReplyOutcome::CheckpointObserved { document } = reply.outcome else {
            return Err(Error::CheckpointUnavailable);
        };
        let context = self.checkpoint_context(&prepared)?;
        let (_, manifest) = checkpoints::validate_document(&document, Some(&context), None)
            .map_err(why!(Error::CheckpointConflict))?;
        let ready = json!({"step":manifest.step,"files":document["files"]});
        let store = crate::checkpoints::Store::open_existing(&self.settings.registry_root)
            .map_err(why!(Error::CheckpointConflict))?;
        let registered = store
            .register(&context, &ready, now)
            .map_err(why!(Error::CheckpointConflict))?;
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(why!(Error::Authority))?;
        Ok(
            json!({"submission_id":action,"checkpoint":registered,"scheduler_mutation_performed":false}),
        )
    }
}
