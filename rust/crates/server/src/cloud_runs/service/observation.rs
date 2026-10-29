use super::*;
impl Service {
    pub fn apply_observation(
        &self,
        snapshot: &mut crate::experiments::Snapshot,
    ) -> cloud::Result<()> {
        if !self.handles_run(&snapshot.project, &snapshot.run_id)? {
            return Ok(());
        }
        let attempts = snapshot
            .attempts
            .iter()
            .map(|a| a.attempt_id.clone())
            .collect::<Vec<_>>();
        for attempt in attempts {
            let Some(prepared) =
                self.saved_attempt(&snapshot.project, &snapshot.run_id, &attempt)?
            else {
                continue;
            };
            self.check(&prepared)?;
            if let Some(saved) = self.exchange_saved(&prepared)? {
                crate::read_model::project_worker(
                    snapshot,
                    &prepared.execution.binding,
                    &saved.evidence,
                    prepared.plan["action_id"]
                        .as_str()
                        .ok_or_else(cloud::Error::evidence)?,
                    &saved.observed_at,
                )
                .map_err(|_| cloud::Error::evidence())?;
            }
        }
        self.apply_artifacts(snapshot)?;
        Ok(())
    }
    pub fn handles_run(&self, project: &str, run: &str) -> cloud::Result<bool> {
        let root = self.authority(project, run)?;
        if !root.entry_exists(OsStr::new(run)).map_err(io)? {
            return Ok(false);
        }
        let directory = root.child(OsStr::new(run), false).map_err(io)?;
        if !directory
            .entry_exists(OsStr::new("manifest.json"))
            .map_err(io)?
        {
            return Ok(false);
        }
        Ok(
            read(&directory, "manifest.json")?["native_creation"]["format"]
                == "native-cloud-run.v1",
        )
    }
    pub fn handles_action(&self, action: &str) -> cloud::Result<bool> {
        Ok(
            self.actions.observe(action).map_err(op)?["native_cloud"]["format"]
                == "native-cloud-action.v1",
        )
    }
    pub(super) fn view(
        &self,
        prepared: &Prepared,
        snapshot: Value,
        now: &str,
    ) -> cloud::Result<Value> {
        self.check(prepared)?;
        let mut result = ml_exp_core::operations::submission_view(&snapshot, now)
            .map_err(|_| cloud::Error::evidence())?;
        if let Some(state) = self.outbox.optional_status(&prepared.execution.job)? {
            result["cloud_observation"] = state;
        }
        Ok(result)
    }
    pub fn get(&self, action: &str, now: &str) -> cloud::Result<Value> {
        let prepared = self.load(action)?;
        self.view(&prepared, self.actions.observe(action).map_err(op)?, now)
    }
    pub fn for_run(&self, project: &str, run: &str, now: &str) -> cloud::Result<Value> {
        self.authority(project, run)?;
        let prepared = self.saved(project, run)?.ok_or_else(missing)?;
        self.get(
            prepared.plan["action_id"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?,
            now,
        )
    }
    /// Project accepted cached scheduler evidence only. Terminal scheduler state
    /// never manufactures training completion, metrics or downloadable results.
    pub fn project_view(
        &self,
        mut value: Value,
        project: &str,
        run: &str,
        attempt: Option<&str>,
        now: &str,
    ) -> cloud::Result<Value> {
        let selected = attempt
            .map(str::to_owned)
            .unwrap_or(self.current_attempt(project, run)?);
        let Some(prepared) = self.saved_attempt(project, run, &selected)? else {
            return Ok(value);
        };
        let submission = self.get(
            prepared.plan["action_id"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?,
            now,
        )?;
        value["native_submission"] = json!({"submission_id":submission["submission_id"],"status":submission["status"],"source":"native-cloud-action","next_action":submission["next_action"]});
        if let Some(saved) = self.exchange_saved(&prepared)? {
            value["worker_observation"] = json!({"status":"OBSERVED","phase":saved.evidence.worker.phase,"received_at":saved.observed_at,"source":"native-cloud-exchange","sequence":saved.sequence,"metrics_complete":saved.complete,"checkpoint_diagnostic":saved.checkpoint_error});
            value["input_delivery"] = json!(saved.inputs);
        }
        let observation = &submission["cloud_observation"]["observation"];
        if observation.is_object() {
            value["evidence"]["scheduler"] = json!({"state":observation["state"],"attempt_id":selected,"detail":observation,"source":"native-cloud-outbox","as_of":null,"stale":true,"freshness":"UNKNOWN"});
            if attempt.is_none() {
                value["scheduler_state"] = observation["state"].clone();
                value["is_terminal"] = json!(
                    serde_json::from_value::<ml_exp_core::state::RunState>(
                        observation["state"].clone()
                    )
                    .is_ok_and(|s| s.is_terminal())
                );
            } else if value["attempt"].is_object() {
                value["attempt"]["state"] = observation["state"].clone();
                value["attempt"]["backend_job_id"] = observation["job_id"].clone();
            }
        } else if matches!(
            submission["status"].as_str(),
            Some("EXECUTING" | "RECONCILE_REQUIRED")
        ) {
            value["evidence"]["scheduler"] = json!({"state":"UNKNOWN","attempt_id":selected,"diagnostic":"SUBMISSION_UNCERTAIN","source":"native-cloud-action","as_of":null,"stale":true,"freshness":"UNKNOWN"});
        }
        Ok(value)
    }
}
