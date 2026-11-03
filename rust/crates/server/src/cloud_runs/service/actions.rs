use super::*;
impl Service {
    pub fn prepare(
        &self,
        project: &str,
        run: &str,
        approval: Approval,
        now: &str,
        expires: &str,
        transition: &str,
    ) -> cloud::Result<Value> {
        if let Some(status) = self.prepare_script(project, run)? {
            return Ok(status);
        }
        let lock = self.lock()?;
        let attempt_id = self.current_attempt(project, run)?;
        let saved = self.saved_attempt(project, run, &attempt_id)?;
        let reused = saved.is_some();
        let prepared = if let Some(saved) = saved {
            if saved.approval != approval {
                return Err(cloud::Error::evidence());
            }
            self.check(&saved)?;
            saved
        } else {
            let root = self.authority(project, run)?;
            let directory = root.child(OsStr::new(run), false).map_err(io)?;
            if read(&directory, "status.json")?["state"] != "NOT_SUBMITTED" {
                return Err(cloud::Error::evidence());
            }
            let scope = ml_exp_core::operations::Scope {
                project: project.into(),
                scope_type: ml_exp_core::operations::ScopeType::Run,
                object_id: run.into(),
            };
            if attempt_id == "attempt-001" && self.actions.has_scope_history(&scope).map_err(op)? {
                return Err(cloud::Error::evidence());
            }
            let mut token = [0u8; 16];
            getrandom::fill(&mut token).map_err(|_| cloud::Error::configuration())?;
            let execution = self.current_execution(
                project,
                run,
                &attempt_id,
                &format!("{:032x}", u128::from_be_bytes(token)),
            )?;
            if attempt_id != "attempt-001" {
                let run_dir = root.child(OsStr::new(run), false).map_err(io)?;
                let attempt = read(
                    &run_dir
                        .child(OsStr::new("attempts"), false)
                        .map_err(io)?
                        .child(OsStr::new(&attempt_id), false)
                        .map_err(io)?,
                    "attempt.json",
                )?;
                if approval.max_gpu_hours
                    != attempt["native_cloud_retry"]["request"]["max_gpu_hours"]
                        .as_f64()
                        .ok_or_else(cloud::Error::evidence)?
                {
                    return Err(cloud::Error::evidence());
                }
            }
            // Compile provider resource limits before publishing any Action.
            let adapter = self.execution_adapter(&execution)?;
            match &adapter.profile {
                cloud::adapter::Profile::Pai(p) => p.create(&execution.job, execution.allocation),
                cloud::adapter::Profile::Sensecore(p) => p.create(&execution.job),
            }
            .map_err(|_| cloud::Error::configuration())?;
            let plan = action::prepare(&execution, &approval, now, expires)
                .map_err(|_| cloud::Error::evidence())?;
            let prepared = Prepared {
                execution,
                plan,
                approval,
            };
            write(
                &self.slots,
                &Self::attempt_slot(project, run, &attempt_id)?,
                &prepared,
            )?;
            prepared
        };
        lock.validate()?;
        self.check(&prepared)?;
        let snapshot = self
            .actions
            .save_plan(&prepared.plan, now, transition)
            .map_err(op)?;
        self.actions
            .bind_cloud_request(
                snapshot["action_id"]
                    .as_str()
                    .ok_or_else(cloud::Error::evidence)?,
                &prepared.execution,
            )
            .map_err(op)?;
        let mut result = self.view(&prepared, snapshot, now)?;
        result["reused"] = json!(reused);
        Ok(result)
    }
    pub fn authorize(
        &self,
        action: &str,
        reason: &str,
        now: &str,
        transition: &str,
    ) -> cloud::Result<Value> {
        let lock = self.lock()?;
        let prepared = self.load(action)?;
        self.check(&prepared)?;
        let snapshot = self
            .actions
            .authorize(action, "bearer-client", reason, now, transition)
            .map_err(op)?;
        lock.validate()?;
        self.view(&prepared, snapshot, now)
    }
    pub fn execute(
        &self,
        action: &str,
        confirmation: &str,
        now: &str,
        transition: &str,
        cancelled: &AtomicBool,
    ) -> cloud::Result<Value> {
        if !self.enabled {
            return Err(cloud::Error {
                code: "CLOUD_SUBMISSION_DISABLED".into(),
                http_status: Some(403),
                uncertain: false,
            });
        }
        let lock = self.lock()?;
        let prepared = self.load(action)?;
        let execution = &prepared.execution;
        let adapter = self.execution_adapter(execution)?;
        if self.actions.observe(action).map_err(op)?["execution"]["status"] == "AUTHORIZED" {
            let root = self.authority(&execution.job.project, &execution.job.run_id)?;
            let run = root
                .child(OsStr::new(&execution.job.run_id), false)
                .map_err(io)?;
            let attempts = run.child(OsStr::new("attempts"), false).map_err(io)?;
            let attempt = attempts
                .child(OsStr::new(&execution.job.attempt_id), false)
                .map_err(io)?;
            for directory in [&run, &attempt] {
                let status = read(directory, "status.json")?;
                if status["project"] != execution.job.project
                    || status["run_id"] != execution.job.run_id
                    || status["attempt_id"] != execution.job.attempt_id
                    || status["state"] != "NOT_SUBMITTED"
                    || status.get("backend_job_id").is_some_and(|v| !v.is_null())
                {
                    return Err(cloud::Error::evidence());
                }
                for name in ["submission.json", "backend.json", "collection.json"] {
                    if directory.entry_exists(OsStr::new(name)).map_err(io)? {
                        return Err(cloud::Error::evidence());
                    }
                }
            }
        }
        let guard = self
            .fence
            .existing_observation_guard(&execution.job.project, &execution.job.run_id)
            .map_err(|_| cloud::Error::evidence())?;
        let check = || {
            lock.validate()?;
            guard
                .require_observed_live(&execution.job.project, &execution.job.run_id)
                .map_err(|_| cloud::Error::evidence())?;
            self.check(&prepared)
        };
        let policy = Policy {
            allow_scheduler_mutations: true,
            ..Default::default()
        };
        let files = FrozenFiles {
            execution_campaign_sha256: Some(execution.binding.run_sha256.clone()),
            execution_manifest_sha256: Some(execution.binding.attempt_sha256.clone()),
            ..Default::default()
        };
        let snapshot = Runner {
            actions: &self.actions,
            adapter: &adapter,
            outbox: &self.outbox,
        }
        .execute(
            execution,
            Execute {
                action_id: action,
                confirmation,
                policy: &policy,
                files: &files,
                claim: ClaimContext {
                    now,
                    transition_id: transition,
                    pid: std::process::id(),
                },
            },
            &check,
            cancelled,
        )?;
        self.view(&prepared, snapshot, now)
    }
    pub fn reconcile(
        &self,
        action: &str,
        now: &str,
        transition: &str,
        cancelled: &AtomicBool,
    ) -> cloud::Result<Value> {
        let lock = self.lock()?;
        let prepared = self.load(action)?;
        let snapshot = self.actions.observe(action).map_err(op)?;
        if let Some(transition_value) =
            ml_exp_core::operations::interrupted(&snapshot["execution"], now)
                .map_err(|_| cloud::Error::evidence())?
        {
            self.actions
                .commit(action, &transition_value, transition, now)
                .map_err(op)?;
        }
        let adapter = self.execution_adapter(&prepared.execution)?;
        let check = || {
            lock.validate()?;
            self.check(&prepared)
        };
        let snapshot = Runner {
            actions: &self.actions,
            adapter: &adapter,
            outbox: &self.outbox,
        }
        .reconcile(action, &check, cancelled)?;
        self.view(&prepared, snapshot, now)
    }
    pub fn refresh(&self, action: &str, now: &str, cancelled: &AtomicBool) -> cloud::Result<Value> {
        let prepared = self.load(action)?;
        let execution = &prepared.execution;
        let guard = self
            .fence
            .existing_observation_guard(&execution.job.project, &execution.job.run_id)
            .map_err(|_| cloud::Error::evidence())?;
        let lock = self.lock()?;
        if self.actions.observe(action).map_err(op)?["execution"]["status"] != "VERIFIED" {
            return Err(cloud::Error::evidence());
        }
        lock.validate()?;
        // The outbox serializes its own observation commits and the Run fence
        // prevents deletion. A slow provider GET must not hold the global
        // authorship/worker-ingress lock across every other Run.
        drop(lock);
        self.outbox.reconcile(
            &self.execution_adapter(execution)?,
            &execution.job,
            execution.allocation,
            &|| {
                guard
                    .require_observed_live(&execution.job.project, &execution.job.run_id)
                    .map_err(|_| cloud::Error::evidence())?;
                self.check(&prepared)
            },
            cancelled,
        )?;
        self.get(action, now)
    }
    pub fn cancel(
        &self,
        action: &str,
        confirmation: &str,
        now: &str,
        cancelled: &AtomicBool,
    ) -> cloud::Result<Value> {
        if !self.enabled || confirmation != format!("CANCEL {action}") {
            return Err(cloud::Error::configuration());
        }
        let lock = self.lock()?;
        let prepared = self.load(action)?;
        let execution = &prepared.execution;
        if self.actions.observe(action).map_err(op)?["execution"]["status"] != "VERIFIED" {
            return Err(cloud::Error::evidence());
        }
        let check = || {
            lock.validate()?;
            self.check(&prepared)
        };
        let current = self.outbox.status(&execution.job)?;
        if current["cancellation_started"] == true {
            self.outbox.reconcile(
                &self.execution_adapter(execution)?,
                &execution.job,
                execution.allocation,
                &check,
                cancelled,
            )?;
        } else {
            self.outbox.stop(
                &self.execution_adapter(execution)?,
                &execution.job,
                execution.allocation,
                &format!("STOP {}", Outbox::slot(&execution.job)?),
                &check,
                cancelled,
            )?;
        }
        self.get(action, now)
    }
}
