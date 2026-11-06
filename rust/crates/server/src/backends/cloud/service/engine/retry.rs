//! Append an explicit continuation after a fresh exact provider observation.
//! Original Run, Action and Attempt bytes remain immutable.
use super::*;
use ml_exp_core::{cloud::retry, first_submission::native::retry as common};
impl Service {
    pub fn retry(
        &self,
        action: &str,
        request: common::Request,
        now: &str,
        cancelled: &AtomicBool,
    ) -> cloud::Result<Value> {
        let _admission = crate::administration::authorship_guard(&self.settings.registry_root)
            .map_err(why!(cloud::Error::evidence()))?;
        request
            .validate(action)
            .map_err(why!(cloud::Error::configuration()))?;
        if !self.creation {
            return Err(cloud::Error {
                code: "CLOUD_RUN_CREATION_DISABLED".into(),
                http_status: Some(403),
                uncertain: false,
            });
        }
        let _lock = self.lock()?;
        let parent = self.load(action)?;
        let binding = &parent.execution.binding;
        if request
            .parent()
            .map_err(why!(cloud::Error::configuration()))?
            != (binding.run_id.as_str(), binding.attempt_id.as_str())
        {
            return Err(cloud::Error::evidence());
        }
        let project = &binding.project;
        let run = &binding.run_id;
        let guard = self
            .fence
            .existing_observation_guard(project, run)
            .map_err(why!(cloud::Error::evidence()))?;
        let directory = self
            .authority(project, run)?
            .child(OsStr::new(run), false)
            .map_err(io)?;
        let manifest = read(&directory, "manifest.json")?;
        let attempts = directory.child(OsStr::new("attempts"), false).map_err(io)?;
        let next = common::next(&binding.attempt_id).map_err(why!(cloud::Error::evidence()))?;
        let exists = attempts.entry_exists(OsStr::new(&next)).map_err(io)?;
        let attempt = if exists {
            let saved = read(
                &attempts.child(OsStr::new(&next), false).map_err(io)?,
                "attempt.json",
            )?;
            let receipt: retry::Receipt =
                serde_json::from_value(saved["native_cloud_retry"].clone())
                    .map_err(why!(cloud::Error::evidence()))?;
            if receipt.parent_action != action
                || receipt.parent_binding != *binding
                || receipt.request != request
            {
                return Err(cloud::Error::evidence());
            }
            receipt
                .validate(&manifest, &next)
                .map_err(why!(cloud::Error::evidence()))?;
            saved
        } else {
            if self.current_attempt(project, run)? != binding.attempt_id {
                return Err(cloud::Error::evidence());
            }
            // Prepare refuses an Attempt whose Run no longer matches the project, so
            // publishing one here would leave the Run pointing at it unspendable.
            self.check(&parent)?;
            let adapter = self.execution_adapter(&parent.execution)?;
            let state = self.outbox.reconcile(
                &adapter,
                &parent.execution.job,
                parent.execution.allocation,
                &|| self.still_live(&guard, project, run),
                cancelled,
            )?;
            let termination =
                super::sync_jobs::proven_terminal(&state)?.ok_or_else(cloud::Error::evidence)?;
            let checkpoint = match &request.resume_from {
                Some(reference) => {
                    self.restore_reference(project, &parent.execution.backend_id, reference)?
                }
                None => Value::Null,
            };
            let receipt = retry::Receipt {
                contract: retry::CONTRACT.into(),
                parent_action: action.into(),
                parent_binding: binding.clone(),
                parent_job_sha256: cloud::adapter::job_digest(&parent.execution.job)?,
                termination,
                request: request.clone(),
                request_sha256: request
                    .identity(action)
                    .map_err(why!(cloud::Error::evidence()))?,
                reserved_gpu_seconds: u64::from(parent.execution.job.gpus)
                    .checked_mul(parent.execution.job.max_time_seconds)
                    .ok_or_else(cloud::Error::evidence)?,
                created_at: now.into(),
            };
            let attempt = retry::plan(&manifest, receipt, &checkpoint)
                .map_err(why!(cloud::Error::evidence()))?;
            let (_, random) =
                projects::transition_context().map_err(why!(cloud::Error::configuration()))?;
            let stage = self
                .slots
                .create_new_child(OsStr::new(&format!("retry-{random}")))
                .map_err(io)?;
            write(&stage, "attempt.json", &attempt)?;
            write(
                &stage,
                "status.json",
                &json!({"project":project,"run_id":run,"attempt_id":next,"state":"NOT_SUBMITTED","updated_at":now}),
            )?;
            guard
                .require_observed_live(project, run)
                .map_err(why!(cloud::Error::evidence()))?;
            stage
                .publish_directory(&attempts, OsStr::new(&next))
                .map_err(why!(cloud::Error {
                    code: "CLOUD_RETRY_UNCERTAIN".into(),
                    http_status: None,
                    uncertain: true,
                }))?;
            attempt
        };
        let mut next_binding = binding.clone();
        next_binding.attempt_id = next.clone();
        next_binding.attempt_sha256 = helper::sha256_bytes(&encode(&attempt)?);
        retry::validate_attempt(&manifest, &attempt, &next_binding)
            .map_err(why!(cloud::Error::evidence()))?;
        if self.current_attempt(project, run)? == binding.attempt_id {
            directory.replace_regular(OsStr::new("status.json"),&encode(&json!({"project":project,"run_id":run,"attempt_id":next,"state":"NOT_SUBMITTED","updated_at":attempt["attempt_created_at"]}))?).map_err(why!(cloud::Error{code:"CLOUD_RETRY_UNCERTAIN".into(),http_status:None,uncertain:true}))?;
        }
        Ok(
            json!({"project":project,"run_id":run,"attempt_id":next,"parent_submission_id":action,"retry_state":"CREATED","reused":exists,
            "request_sha256":request.identity(action).map_err(why!(cloud::Error::evidence()))?,"run_sha256":binding.run_sha256,"attempt_sha256":next_binding.attempt_sha256,
            "resume_from":request.resume_from,"restart_from":request.restart_from,"reserved_gpu_seconds":attempt["native_cloud_retry"]["reserved_gpu_seconds"],"max_gpu_hours":request.max_gpu_hours,
            "scheduler_mutation_performed":false,"next_action":"PREPARE_NEW_ACTION"}),
        )
    }
}
