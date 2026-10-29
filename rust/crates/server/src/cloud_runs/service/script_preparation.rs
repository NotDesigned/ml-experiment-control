//! One durable zero-GPU preparation per frozen SenseCore Run. The existing
//! outbox reconciles lost replies; a pending/failed copy never dispatches training.
use super::*;
use ml_exp_core::{cloud::Job, data_preparation};

fn view(project: &str, run_id: &str, status: Value) -> cloud::Result<Value> {
    let observation: Option<ml_exp_core::cloud::JobObservation> =
        serde_json::from_value(status["observation"].clone())
            .map_err(|_| cloud::Error::evidence())?;
    let state = data_preparation::job_status(
        observation.as_ref().map(|o| o.state),
        status["retry_safe"] != true,
        !status["handle"].is_null(),
    );
    Ok(
        json!({"project":project,"run_id":run_id,"attempt_id":"attempt-001",
        "operation":"PREPARE_INPUT","status":state,"phase":"DATA_PREPARATION","scheduler_submitted":false,
        "training_authorized":false,"preparation_job_submitted":!status["handle"].is_null(),
        "error_code":if state==data_preparation::Status::Failed{Some("DATA_PREPARATION_FAILED")}else{None},
        "data_preparation":status}),
    )
}

impl Service {
    fn script_job(&self, project: &str, run_id: &str) -> cloud::Result<Option<(String, Job)>> {
        let root = self.authority(project, run_id)?;
        if !root.entry_exists(OsStr::new(run_id)).map_err(io)? {
            return Ok(None);
        }
        let run = read(
            &root.child(OsStr::new(run_id), false).map_err(io)?,
            "manifest.json",
        )?;
        if run["native_cloud"]["data_preparation_before_job"] != true {
            return Ok(None);
        }
        let token = canonical::digest(&json!([
            "native-script-job.v1",
            project,
            run_id,
            run["native_creation"]
        ]))
        .map_err(|_| cloud::Error::evidence())?;
        let execution = self.current_execution(project, run_id, "attempt-001", &token[..32])?;
        let package = execution.package().map_err(|_| cloud::Error::evidence())?;
        let plan = package
            .config
            .data_preparation
            .as_ref()
            .ok_or_else(cloud::Error::evidence)?;
        let backend = self
            .backends
            .get(&execution.backend_id)
            .ok_or_else(cloud::Error::configuration)?;
        let copier = backend
            .data_copier
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let mut job = execution.job.clone();
        job.run_id = format!("script-{}", &token[..32]);
        job.submission_token = token[..32].into();
        job.gpus = 0;
        job.cpus = 2;
        job.memory_gb = 4;
        job.max_time_seconds = plan.request.timeout_seconds + 60;
        let runtime = self.runtime(
            project,
            &execution.backend_id,
            run["container"]["runtime_id"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?,
        )?;
        job.command = vec![
            runtime.worker_path,
            "script-prepare".into(),
            "--plan-json".into(),
            serde_json::to_string(plan).map_err(|_| cloud::Error::evidence())?,
        ];
        match &copier.profile {
            cloud::adapter::Profile::Pai(p) => {
                p.create(&job, ml_exp_core::cloud::pai::Allocation::Guaranteed)
            }
            cloud::adapter::Profile::Sensecore(p) => p.create(&job),
        }
        .map_err(|_| cloud::Error::configuration())?;
        Ok(Some((execution.backend_id, job)))
    }
    /// Read only retained job evidence. This never creates an outbox entry,
    /// starts a copy, refreshes a provider or retries a failed script.
    pub(super) fn script_status(
        &self,
        project: &str,
        run_id: &str,
    ) -> cloud::Result<Option<Value>> {
        let Some((_, job)) = self.script_job(project, run_id)? else {
            return Ok(None);
        };
        self.sync_outbox
            .optional_status(&job)?
            .map(|status| view(project, run_id, status))
            .transpose()
    }
    pub(super) fn prepare_script(
        &self,
        project: &str,
        run_id: &str,
    ) -> cloud::Result<Option<Value>> {
        let Some((executor, job)) = self.script_job(project, run_id)? else {
            return Ok(None);
        };
        if !self.enabled {
            return Err(cloud::Error::configuration());
        }
        let copier = self.backends[&executor]
            .data_copier
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let check = || self.project_hash(project).map(|_| ());
        let allocation = ml_exp_core::cloud::pai::Allocation::Guaranteed;
        self.sync_outbox
            .prepare(&copier.profile, &job, allocation, &check)?;
        let current = self.sync_outbox.status(&job)?;
        let status = if current["retry_safe"] == true {
            self.sync_outbox.execute(
                copier,
                &job,
                allocation,
                &format!("SUBMIT {}", Outbox::slot(&job)?),
                &check,
                &AtomicBool::new(false),
            )?
        } else {
            self.sync_outbox
                .reconcile(copier, &job, allocation, &check, &AtomicBool::new(false))?
        };
        let status = view(project, run_id, status)?;
        if status["status"] == "READY" {
            return Ok(None);
        }
        Ok(Some(status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_cpu_preparation_retains_its_identity_and_never_claims_training_started() {
        for state in [
            "FAILED",
            "CANCELLED",
            "PREEMPTED",
            "SUCCEEDED",
            "RUNNING",
            "UNKNOWN",
        ] {
            let evidence = json!({"retry_safe":false,"handle":{"job_id":"owned-cpu-job"},
                "observation":{"job_id":"owned-cpu-job","provider_state":state,"state":state,"queue":null}});
            let result = view("demo", "train", evidence.clone()).unwrap();
            assert_eq!(result["project"], "demo");
            assert_eq!(result["run_id"], "train");
            assert_eq!(result["data_preparation"], evidence);
            assert_eq!(result["scheduler_submitted"], false);
            assert_eq!(result["training_authorized"], false);
            assert_eq!(result["preparation_job_submitted"], true);
            assert_eq!(
                result["status"],
                match state {
                    "SUCCEEDED" => "READY",
                    "RUNNING" => "PREPARING",
                    "UNKNOWN" => "RECONCILE_REQUIRED",
                    _ => "FAILED",
                }
            );
            assert_eq!(
                result["error_code"].is_string(),
                matches!(state, "FAILED" | "CANCELLED" | "PREEMPTED")
            );
        }
        assert!(view("demo", "train", json!({"observation":{"state":"invalid"}})).is_err());
    }
}
