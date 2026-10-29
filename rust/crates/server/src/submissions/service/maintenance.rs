use super::*;
use ml_exp_core::maintenance::{Plan, Selection};
impl Service {
    pub fn maintenance_plan(
        &self,
        action: &str,
        selection: Selection,
        request_id: String,
        now: &str,
    ) -> Result<Plan> {
        self.refresh_scheduler(action, now, &AtomicBool::new(false))?;
        let prepared = self.load(action)?;
        let status = self
            .scheduler_observation(&prepared)?
            .ok_or(Error::Authority)?;
        if !status.observation.status.state.is_terminal() {
            return Err(Error::Authority);
        }
        let (manifest, _) = prepared
            .request
            .prepared_content()
            .ok_or(Error::Authority)?;
        let shared =
            ml_exp_core::slurm::staging::plan(prepared.request.intent(), manifest.as_bytes())
                .map_err(|_| Error::Authority)?
                .ok_or(Error::Authority)?;
        let package = shared.managed.ok_or(Error::Authority)?;
        let plan = Plan {
            contract: "native-storage-maintenance.v1".into(),
            config: package.config,
            selection,
            request_id,
        };
        plan.validate().map_err(|_| Error::Authority)?;
        Ok(plan)
    }
    pub fn maintenance_execute(&self, action: &str, plan: &Plan) -> Result<Value> {
        if !self.enabled {
            return Err(Error::Disabled);
        }
        // Recompute the stored managed package, without interpreting a withdrawn
        // checkpoint as permission to rewrite an original scientific manifest.
        let scope = plan
            .config
            .binding()
            .map_err(|_| Error::Authority)?
            .identity;
        let prepared = self
            .saved_attempt(&scope.project, &scope.run_id, &scope.attempt_id)?
            .ok_or(Error::Authority)?;
        if prepared.plan["action_id"] != action {
            return Err(Error::Authority);
        }
        let (manifest, _) = prepared
            .request
            .prepared_content()
            .ok_or(Error::Authority)?;
        let shared =
            ml_exp_core::slurm::staging::plan(prepared.request.intent(), manifest.as_bytes())
                .map_err(|_| Error::Authority)?
                .ok_or(Error::Authority)?;
        if shared.managed.as_ref().map(|p| &p.config) != Some(&plan.config) {
            return Err(Error::Authority);
        }
        let backend = self
            .backends
            .values()
            .find(|b| b.profile.target == prepared.request.intent().target)
            .ok_or(Error::Authority)?;
        let mut ssh = backend.ssh.clone();
        ssh.budget.timeout = std::time::Duration::from_secs(900);
        let mut argv = ssh.argv();
        let at = argv
            .iter()
            .position(|s| s == "slurm-helper")
            .ok_or(Error::Authority)?;
        argv[at] = "storage-maintain".into();
        argv.remove(at + 1);
        let body = serde_json::to_vec(plan).map_err(|_| Error::Authority)?;
        let environment = ssh.environment();
        let result = ml_exp_io::process::execute(ml_exp_io::process::Request {
            argv: &argv,
            cwd: &ssh.local_cwd,
            environment: ml_exp_io::process::Environment::Replace(&environment),
            stdin: ml_exp_io::process::Input::Bytes(&body),
            stdout: ml_exp_io::process::Capture::Memory { limit: 65536 },
            stderr: ml_exp_io::process::Capture::Memory { limit: 65536 },
            deadline: std::time::Instant::now() + ssh.budget.timeout,
            termination_grace: ssh.budget.termination_grace,
            drain_grace: ssh.budget.drain_grace,
            cancelled: &AtomicBool::new(false),
        })
        .map_err(|_| Error::Authority)?;
        if result.exit_code != Some(0)
            || result.timed_out
            || result.cancelled
            || result.failure.is_some()
            || !result.stdin_complete
            || !result.stdout.complete
            || result.stdout.truncated
        {
            return Err(Error::Uncertain);
        }
        let value = Value::Object(
            ml_exp_core::json::mapping(&result.stdout.bytes).map_err(|_| Error::Authority)?,
        );
        if plan.receipt().map_err(|_| Error::Authority)? != value {
            return Err(Error::Authority);
        }
        Ok(value)
    }
}
