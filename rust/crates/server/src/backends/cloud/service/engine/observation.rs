use super::*;
use ml_exp_core::records::{ActionKind, RunKind};

/// Normalize retained provider evidence to the shared status contract. A read
/// cannot infer an observation time or worker/process completion from it.
fn scheduler_evidence(observation: &Value, backend: &str, attempt: &str) -> Value {
    let mut detail = observation.clone();
    detail["backend"] = json!(backend);
    detail["backend_job_id"] = observation["job_id"].clone();
    detail["raw_state"] = observation["provider_state"].clone();
    json!({"state":observation["state"],"attempt_id":attempt,"detail":detail,
        "source":"native-cloud-outbox","as_of":null,"stale":null,"freshness":"UNKNOWN"})
}

impl Service {
    pub fn apply_observation(
        &self,
        snapshot: &mut crate::experiments::Snapshot,
    ) -> cloud::Result<()> {
        // The caller holds the deletion guard, and this is the Run it already read.
        if RunKind::of(&snapshot.run.metadata) != RunKind::Cloud {
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
                .map_err(why!(cloud::Error::evidence()))?;
            }
        }
        self.apply_artifacts(snapshot)?;
        Ok(())
    }
    pub fn action_kind(&self, action: &str) -> cloud::Result<ActionKind> {
        Ok(ActionKind::of(&self.actions.observe(action).map_err(op)?))
    }
    pub fn handles_action(&self, action: &str) -> cloud::Result<bool> {
        Ok(self.action_kind(action)? == ActionKind::Cloud)
    }
    pub(super) fn view(
        &self,
        prepared: &Prepared,
        snapshot: Value,
        now: &str,
    ) -> cloud::Result<Value> {
        let mut result = ml_exp_core::operations::submission_view(&snapshot, now)
            .map_err(why!(cloud::Error::evidence()))?;
        if let Some(state) = self.outbox.optional_status(&prepared.execution.job)? {
            result["cloud_observation"] = state;
        }
        crate::renewal::decorate(&self.slots, result).map_err(io)
    }
    pub fn get(&self, action: &str, now: &str) -> cloud::Result<Value> {
        let prepared = self.load(action)?;
        self.view(&prepared, self.actions.observe(action).map_err(op)?, now)
    }
    /// Worker-reported evidence retained for one Submission. No provider request.
    /// `None` means the Action belongs to another backend service.
    pub fn submission_progress(&self, action: &str) -> cloud::Result<Option<Value>> {
        if !self.handles_action(action)? {
            return Ok(None);
        }
        let prepared = self.load(action)?;
        let binding = &prepared.execution.binding;
        let mut value = json!({"submission_id":action,"project":binding.project,
            "run_id":binding.run_id,"attempt_id":binding.attempt_id,"availability":"NOT_REPORTED"});
        if let Some(saved) = self.exchange_saved(&prepared)? {
            let inputs: Vec<Value> = saved.inputs.iter().map(|record| json!(record)).collect();
            // A failure outranks later progress; READY rows are the least informative.
            let selected = inputs
                .iter()
                .find(|row| row["phase"] == "FAILED")
                .or_else(|| inputs.iter().find(|row| row["phase"] != "READY"))
                .or_else(|| inputs.first());
            value["availability"] = json!("AVAILABLE");
            value["worker_phase"] =
                json!({"phase":saved.evidence.worker.phase,"observed_at":saved.observed_at});
            value["input_delivery"] = selected.cloned().unwrap_or(Value::Null);
            value["inputs"] = Value::Array(inputs);
        } else if let Some(failure) = self.startup_failure(&prepared)? {
            // Said by a worker that never reported; availability stays NOT_REPORTED.
            value["startup_failure"] = failure;
        }
        Ok(Some(value))
    }
    /// The current Attempt's Submission, as a listing for clients without saved state.
    pub fn run_submissions(&self, project: &str, run: &str) -> cloud::Result<Vec<Value>> {
        self.authority(project, run)?;
        Ok(self
            .saved(project, run)?
            .into_iter()
            .map(|prepared| {
                json!({"project":project,"run_id":run,
                    "attempt_id":prepared.execution.binding.attempt_id,
                    "submission_id":prepared.plan["action_id"]})
            })
            .collect())
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
            value["evidence"]["scheduler"] =
                scheduler_evidence(observation, &prepared.execution.backend_id, &selected);
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
            value["evidence"]["scheduler"] = json!({"state":"UNKNOWN","attempt_id":selected,"diagnostic":"SUBMISSION_UNCERTAIN","source":"native-cloud-action","as_of":null,"stale":null,"freshness":"UNKNOWN"});
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_scheduler_summary_retains_job_native_state_and_queue_without_worker_claims() {
        let observation = json!({"job_id":"dlc123","provider_state":"EnvPreparing","state":"STARTING",
            "queue":{"code":"roleMaximumResource","reason":"GPU allowance exhausted"}});
        let layer = scheduler_evidence(&observation, "pai-cpu", "attempt-002");
        assert_eq!(layer["attempt_id"], "attempt-002");
        assert_eq!(layer["detail"]["backend"], "pai-cpu");
        assert_eq!(layer["detail"]["backend_job_id"], "dlc123");
        assert_eq!(layer["detail"]["raw_state"], "EnvPreparing");
        assert_eq!(layer["detail"]["queue"], observation["queue"]);
        assert!(layer["as_of"].is_null());
        assert!(layer["stale"].is_null());
        assert_eq!(layer["freshness"], "UNKNOWN");
        assert!(layer.get("worker").is_none());
        assert!(layer.get("process").is_none());
    }
}
