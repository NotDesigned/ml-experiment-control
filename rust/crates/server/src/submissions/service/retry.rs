//! Append-only continuation metadata. An explicit retry proves the previous
//! exact job terminal, freezes an explicit checkpoint, then publishes a new
//! Attempt. It never prepares, authorizes or submits that Attempt implicitly.
use super::creation::{clean_files, publish_json, read_json};
use super::*;
use crate::backends::slurm::{CallOutcome, call};
use ml_exp_core::first_submission::native::retry;

impl Service {
    pub fn retry(
        &self,
        action: &str,
        request: retry::Request,
        now: &str,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        let _admission = crate::administration::authorship_guard(&self.settings.registry_root)
            .map_err(|_| Error::Authority)?;
        request.validate(action)?;
        let lock = self.lock()?;
        let parent = self.load(action)?;
        let project = &parent.binding.project;
        let run = &parent.binding.run_id;
        let guard = self
            .fence
            .existing_observation_guard(project, run)
            .map_err(|_| Error::Authority)?;
        let root = self
            .creation_roots
            .get(project)
            .ok_or(Error::RunCreationDisabled)?;
        let reference = self.resolve(project, run)?;
        if reference.root_index != 0 || reference.relative_components != [run.clone()] {
            return Err(Error::Authority);
        }
        let directory = root
            .child(OsStr::new(run), false)
            .map_err(|_| Error::Authority)?;
        let run_document = read_json(&directory, "manifest.json")?;
        if run_document["native_creation"]["format"] != "native-run-creation.v1" {
            return Err(Error::Authority);
        }
        let attempts = directory
            .child(OsStr::new("attempts"), false)
            .map_err(|_| Error::Authority)?;
        let next = retry::next(&parent.binding.attempt_id)?;
        let current = self.facts(project, run)?;
        // Every previous reservation must have its own immutable authority and
        // VERIFIED scheduler identity. Missing slots are never treated as free.
        for n in 1..=retry::number(&parent.binding.attempt_id)? {
            let previous = self
                .saved_attempt(project, run, &format!("attempt-{n:03}"))?
                .ok_or(Error::Authority)?;
            self.check(&previous)?;
            let id = previous.plan["action_id"]
                .as_str()
                .ok_or(Error::Authority)?;
            if self.actions.observe(id)?["execution"]["status"] != "VERIFIED" {
                return Err(Error::Conflict);
            }
        }
        let exists = attempts
            .entry_exists(OsStr::new(&next))
            .map_err(|_| Error::Authority)?;
        let attempt = if exists {
            let target = attempts
                .child(OsStr::new(&next), false)
                .map_err(|_| Error::Authority)?;
            let attempt = read_json(&target, "attempt.json")?;
            let saved: retry::Receipt = serde_json::from_value(attempt["native_retry"].clone())
                .map_err(|_| Error::Authority)?;
            saved.validate(&run_document, &next)?;
            if saved.parent_action != action
                || saved.parent_binding != parent.binding
                || saved.request != request
            {
                return Err(Error::Conflict);
            }
            // A repeat may finish only the original pending mirror update.
            // A later current Attempt remains current.
            attempt
        } else {
            if current.binding != parent.binding {
                return Err(Error::Conflict);
            }
            let snapshot = self.actions.observe(action)?;
            let job = snapshot["execution"]["result"]["submission"]["backend_job_id"]
                .as_str()
                .ok_or(Error::Authority)?;
            let observation =
                helper::Request::observe_job(parent.request.intent().clone(), job.into())
                    .map_err(|_| Error::Authority)?;
            let backend = self
                .backends
                .values()
                .find(|b| b.profile.target == observation.intent().target)
                .ok_or(Error::Authority)?;
            let outcome = call(&backend.ssh, &observation, cancelled)
                .map_err(|_| Error::ObservationUnavailable)?;
            let CallOutcome::Receipt { reply, .. } = outcome else {
                return Err(Error::ObservationUnavailable);
            };
            if reply.identity != observation.identity().map_err(|_| Error::Authority)? {
                return Err(Error::Authority);
            }
            let helper::ReplyOutcome::JobObserved {
                observation: termination,
            } = reply.outcome
            else {
                return Err(Error::ObservationUnavailable);
            };
            let checkpoint = self.checkpoints(
                project,
                &request.resume_from.run_id,
                &request.resume_from.attempt_id,
                Some(&request.resume_from.checkpoint_id),
            )?;
            let attempt = retry::plan(
                &run_document,
                request.clone(),
                retry::Authority {
                    backend_id: &backend.profile.target.backend_id,
                    parent: parent.binding.clone(),
                    action,
                    job,
                    termination: *termination,
                    checkpoint: &checkpoint,
                },
                now,
            )?;
            let mut binding = parent.binding.clone();
            binding.attempt_id = next.clone();
            binding.attempt_sha256 = helper::sha256_bytes(
                &serde_json::to_vec(&attempt).map_err(|_| Error::Configuration)?,
            );
            native::definition(
                &run_document,
                &attempt,
                &binding,
                &backend.profile,
                &self.runtime(&current.binding.project, &current.backend, &current.runtime)?,
                &Approval {
                    max_gpu_hours: request.max_gpu_hours,
                    reason: request.reason.clone(),
                },
            )?;
            guard
                .require_observed_live(project, run)
                .map_err(|_| Error::Authority)?;
            self.check(&parent)?;
            lock.validate()?;
            let mut random = [0u8; 16];
            getrandom::fill(&mut random).map_err(|_| Error::Configuration)?;
            let stage = self
                .slots
                .create_new_child(OsStr::new(&format!(
                    "retry-{:032x}",
                    u128::from_be_bytes(random)
                )))
                .map_err(|_| Error::Uncertain)?;
            let result: Result<()> = (|| {
                publish_json(&stage, "attempt.json", &attempt)?;
                publish_json(
                    &stage,
                    "status.json",
                    &retry_status(project, run, &next, now),
                )?;
                stage
                    .publish_directory(&attempts, OsStr::new(&next))
                    .map_err(|_| Error::Uncertain)?;
                Ok(())
            })();
            if stage.verify_attached().is_ok() {
                clean_files(&stage, &["attempt.json", "status.json"])?;
                self.slots
                    .remove_staged_empty_child(&stage)
                    .map_err(|_| Error::Uncertain)?;
            }
            result?;
            attempt
        };
        lock.validate()?;
        self.check(&parent)?;
        guard
            .require_observed_live(project, run)
            .map_err(|_| Error::Uncertain)?;
        if current.binding == parent.binding {
            let status = retry_status(
                project,
                run,
                &next,
                attempt["attempt_created_at"]
                    .as_str()
                    .ok_or(Error::Authority)?,
            );
            directory
                .replace_regular(
                    OsStr::new("status.json"),
                    &serde_json::to_vec(&status).map_err(|_| Error::Configuration)?,
                )
                .map_err(|_| Error::Uncertain)?;
        }
        let binding = self.facts_for(project, run, Some(&next))?.binding;
        retry::validate_attempt(&run_document, &attempt, &binding)?;
        Ok(
            json!({"project":project,"run_id":run,"attempt_id":next,"parent_submission_id":action,
            "retry_state":"CREATED","reused":exists,"request_sha256":request.identity(action)?,
            "run_sha256":binding.run_sha256,"attempt_sha256":binding.attempt_sha256,
            "resume_from":request.resume_from,"reserved_gpu_seconds":attempt["native_retry"]["reserved_gpu_seconds"],
            "max_gpu_hours":request.max_gpu_hours,"scheduler_mutation_performed":false,"next_action":"PREPARE_NEW_ACTION"}),
        )
    }
}
fn retry_status(project: &str, run: &str, attempt: &str, now: &str) -> Value {
    json!({"project":project,"run_id":run,"attempt_id":attempt,"state":"NOT_SUBMITTED","updated_at":now})
}
