use super::*;
use ml_exp_core::maintenance::{Plan, Selection};
impl Service {
    pub fn maintenance_plan(
        &self,
        action: &str,
        selection: Selection,
        request_id: String,
        now: &str,
    ) -> cloud::Result<Plan> {
        self.refresh(action, now, &AtomicBool::new(false))?;
        let prepared = self.load(action)?;
        let state = self.outbox.status(&prepared.execution.job)?;
        if !state["observation"]["state"].as_str().is_some_and(|s| {
            matches!(
                s,
                "SUCCEEDED" | "FAILED" | "CANCELLED" | "TIMEOUT" | "PREEMPTED"
            )
        }) {
            return Err(cloud::Error::evidence());
        }
        let package = prepared
            .execution
            .package()
            .map_err(|_| cloud::Error::evidence())?;
        let plan = Plan {
            contract: "native-storage-maintenance.v1".into(),
            config: package.config,
            selection,
            request_id,
        };
        plan.validate().map_err(|_| cloud::Error::evidence())?;
        Ok(plan)
    }
    fn maintenance_original(&self, action: &str, plan: &Plan) -> cloud::Result<Prepared> {
        let snapshot = self.actions.observe(action).map_err(op)?;
        let scope = &plan
            .config
            .binding()
            .map_err(|_| cloud::Error::evidence())?
            .identity;
        let prepared = self
            .saved_attempt(&scope.project, &scope.run_id, &scope.attempt_id)?
            .ok_or_else(missing)?;
        if snapshot["action_id"] != prepared.plan["action_id"]
            || snapshot["action_id"] != action
            || prepared
                .execution
                .package()
                .map_err(|_| cloud::Error::evidence())?
                .config
                != plan.config
        {
            return Err(cloud::Error::evidence());
        }
        self.check(&prepared)?;
        Ok(prepared)
    }
    pub fn maintenance_execute(&self, action: &str, plan: &Plan) -> cloud::Result<Value> {
        if !self.enabled {
            return Err(cloud::Error::configuration());
        }
        let original = self.maintenance_original(action, plan)?;
        let execution = &original.execution;
        let package = execution.package().map_err(|_| cloud::Error::evidence())?;
        let channel = package.channel.ok_or_else(cloud::Error::configuration)?;
        let collector = self.backends[&execution.backend_id]
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let mut job = execution.job.clone();
        let id = plan.id().map_err(|_| cloud::Error::evidence())?;
        job.submission_token = id[..32].into();
        job.gpus = 0;
        job.cpus = 2;
        job.memory_gb = 4;
        job.max_time_seconds = 600;
        job.priority = 5;
        let worker = execution
            .job
            .command
            .first()
            .ok_or_else(cloud::Error::evidence)?;
        job.command = vec![
            worker.into(),
            "storage-maintain".into(),
            "--plan-json".into(),
            String::from_utf8(encode(plan)?).map_err(|_| cloud::Error::evidence())?,
            "--channel-json".into(),
            String::from_utf8(encode(&channel)?).map_err(|_| cloud::Error::evidence())?,
        ];
        let check = || self.maintenance_original(action, plan).map(|_| ());
        self.sync_outbox.prepare(
            &collector.profile,
            &job,
            ml_exp_core::cloud::pai::Allocation::Guaranteed,
            &check,
        )?;
        let current = self.sync_outbox.status(&job)?;
        if current["retry_safe"] == true {
            self.sync_outbox.execute(
                collector,
                &job,
                ml_exp_core::cloud::pai::Allocation::Guaranteed,
                &format!("SUBMIT {}", Outbox::slot(&job)?),
                &check,
                &AtomicBool::new(false),
            )
        } else {
            self.sync_outbox.reconcile(
                collector,
                &job,
                ml_exp_core::cloud::pai::Allocation::Guaranteed,
                &check,
                &AtomicBool::new(false),
            )
        }
    }
    pub fn maintenance_auth(&self, action: &str, plan: &Plan, token: &str) -> cloud::Result<()> {
        use subtle::ConstantTimeEq;
        let prepared = self.maintenance_original(action, plan)?;
        let channel = self
            .exchange
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .channel(&self.workspace, &prepared.execution)?;
        if !bool::from(channel.token.as_bytes().ct_eq(token.as_bytes())) {
            return Err(cloud::Error {
                code: "WORKER_AUTH_REQUIRED".into(),
                http_status: Some(401),
                uncertain: false,
            });
        }
        Ok(())
    }
    pub fn maintenance_release_object(
        &self,
        action: &str,
        plan: &Plan,
        cursor: u64,
    ) -> cloud::Result<(bool, u64)> {
        let lock = self.lock()?;
        let prepared = self.maintenance_original(action, plan)?;
        let Some(mut saved) = self.artifact_saved(&prepared)? else {
            return Ok((true, 0));
        };
        let id = plan.id().map_err(|_| cloud::Error::evidence())?;
        if !matches!(plan.selection, Selection::Outputs)
            || saved
                .administrative_release
                .as_ref()
                .is_some_and(|old| old != &id)
            || cursor > saved.released_parts
        {
            return Err(cloud::Error::evidence());
        }
        // Administrative deletion is separate from verified download receipts.
        saved.administrative_release = Some(id);
        self.artifact_save(&prepared, &saved)?;
        let count = crate::objects::parts(&saved.bundle.archive).map_err(io)?;
        if saved.released_parts >= count {
            return Ok((true, saved.released_parts));
        }
        self.objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .release(&saved.bundle.archive, saved.released_parts)
            .map_err(io)?;
        saved.released_parts += 1;
        lock.validate()?;
        self.artifact_save(&prepared, &saved)?;
        Ok((saved.released_parts == count, saved.released_parts))
    }
}

impl Service {
    pub fn for_attempt_admin(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        now: &str,
    ) -> cloud::Result<Value> {
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(missing)?;
        self.get(
            prepared.plan["action_id"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?,
            now,
        )
    }
}
