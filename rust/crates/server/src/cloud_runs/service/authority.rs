use super::*;
impl Service {
    pub(super) fn authority(&self, project: &str, run: &str) -> cloud::Result<Directory> {
        ml_exp_core::identity::validate_id(project)
            .and_then(|_| ml_exp_core::identity::validate_id(run))
            .map_err(|_| cloud::Error::configuration())?;
        let (_, workspace) = read_config::workspace(&self.settings.registry_root)
            .map_err(|_| cloud::Error::evidence())?;
        if workspace != self.workspace {
            return Err(cloud::Error::evidence());
        }
        self.fence
            .existing_observation_guard(project, run)
            .map_err(|_| cloud::Error::evidence())?;
        if self
            .lifecycle
            .as_ref()
            .is_some_and(|l| !l.active_projects().is_ok_and(|p| p.contains(project)))
        {
            return Err(cloud::Error::evidence());
        }
        let project = self
            .settings
            .projects
            .iter()
            .find(|p| p.project == project && p.discovery_enabled() && p.run_roots.len() == 1)
            .ok_or_else(cloud::Error::configuration)?;
        let root = Directory::open(&project.run_roots[0], false).map_err(io)?;
        root.verify_private_parent().map_err(io)?;
        Ok(root)
    }
    pub(super) fn project_hash(&self, project: &str) -> cloud::Result<String> {
        experiments::Store::open(self.settings.bindings(), self.fence.clone())
            .map_err(|_| cloud::Error::evidence())?
            .project(project, None)
            .map(|p| p.raw_sha256)
            .map_err(|_| cloud::Error::evidence())
    }
    pub(super) fn documents(
        &self,
        project: &str,
        request: &run::Create,
        route: &Value,
        now: &str,
    ) -> cloud::Result<creation::Documents> {
        let backend = self
            .backends
            .get(&request.executor)
            .ok_or_else(cloud::Error::configuration)?;
        let runtime = backend
            .definition
            .runtimes
            .get(&request.runtime_id)
            .ok_or_else(cloud::Error::configuration)?;
        let source = self
            .sources
            .read(project, &runtime.source_id)
            .map_err(|_| cloud::Error::evidence())?;
        run::documents(
            project,
            request,
            runtime,
            run::Authority {
                project_sha256: &self.project_hash(project)?,
                source_receipt_sha256: &canonical::protocol_identity(&source)
                    .map_err(|_| cloud::Error::evidence())?,
                profile_sha256: &backend.adapter.profile.digest()?,
                tracking_route: route,
            },
            now,
        )
        .map_err(|_| cloud::Error::evidence())
    }
    pub(super) fn current_execution(
        &self,
        project: &str,
        run_id: &str,
        token: &str,
    ) -> cloud::Result<Execution> {
        let root = self.authority(project, run_id)?;
        let directory = root.child(OsStr::new(run_id), false).map_err(io)?;
        let run = read(&directory, "manifest.json")?;
        let original = read(&directory, "request.json")?;
        let request: run::Create =
            serde_json::from_value(original.clone()).map_err(|_| cloud::Error::evidence())?;
        let attempt = read(
            &directory
                .child(OsStr::new("attempts"), false)
                .map_err(io)?
                .child(OsStr::new("attempt-001"), false)
                .map_err(io)?,
            "attempt.json",
        )?;
        let docs = self.documents(
            project,
            &request,
            &run["tracking_route"],
            run["created_at"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?,
        )?;
        if !creation::same_creation(&docs, &run, &attempt, &original) {
            return Err(cloud::Error::evidence());
        }
        let backend = &self.backends[&request.executor];
        let runtime = &backend.definition.runtimes[&request.runtime_id];
        let binding = Binding {
            project: project.into(),
            run_id: run_id.into(),
            attempt_id: "attempt-001".into(),
            project_sha256: self.project_hash(project)?,
            run_sha256: helper::sha256_bytes(&encode(&run)?),
            attempt_sha256: helper::sha256_bytes(&encode(&attempt)?),
            source_receipt_sha256: run["native_creation"]["source_receipt_sha256"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?
                .into(),
            backend_config_sha256: backend.adapter.profile.digest()?,
        };
        run::execution(
            &run,
            &attempt,
            binding,
            runtime,
            &backend.definition.run_root,
            token,
            backend.definition.allocation,
        )
        .map_err(|_| cloud::Error::evidence())
    }
    pub(super) fn check(&self, prepared: &Prepared) -> cloud::Result<()> {
        let current = self.current_execution(
            &prepared.execution.job.project,
            &prepared.execution.job.run_id,
            &prepared.execution.job.submission_token,
        )?;
        if current != prepared.execution {
            return Err(cloud::Error::evidence());
        }
        action::validate(&prepared.plan, &current).map_err(|_| cloud::Error::evidence())
    }
}
