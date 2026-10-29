//! One durable zero-GPU preparation per frozen SenseCore Run. The existing
//! outbox reconciles lost replies; a pending/failed copy never dispatches training.
use super::*;
impl Service {
    pub(super) fn prepare_script(
        &self,
        project: &str,
        run_id: &str,
    ) -> cloud::Result<Option<Value>> {
        let root = self.authority(project, run_id)?;
        let run = read(
            &root.child(OsStr::new(run_id), false).map_err(io)?,
            "manifest.json",
        )?;
        if run["native_cloud"]["data_preparation_before_job"] != true {
            return Ok(None);
        }
        if !self.enabled {
            return Err(cloud::Error::configuration());
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
        if status["observation"]["state"] == "SUCCEEDED" {
            return Ok(None);
        }
        Ok(Some(
            json!({"status":"PREPARING","phase":"DATA_PREPARATION","scheduler_submitted":false,"training_authorized":false,"data_preparation":status}),
        ))
    }
}
