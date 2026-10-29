use super::*;
impl Service {
    fn selected_profile(
        &self,
        project: &str,
        request: &run::Create,
        selected: Option<&cloud::adapter::Profile>,
    ) -> cloud::Result<Option<cloud::adapter::Profile>> {
        let backend = self
            .backends
            .get(&request.executor)
            .ok_or_else(cloud::Error::configuration)?;
        let frozen = if let Some(profile) = selected {
            Some(profile.clone())
        } else {
            let root = self.authority(project, &request.run_id)?;
            if root.entry_exists(OsStr::new(&request.run_id)).map_err(io)? {
                read(
                    &root.child(OsStr::new(&request.run_id), false).map_err(io)?,
                    "manifest.json",
                )?["native_cloud"]
                    .get("profile")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|_| cloud::Error::evidence())?
            } else {
                None
            }
        };
        if let Some(profile) = &frozen {
            profile.validate()?;
            match (&backend.adapter.profile, profile) {
                (
                    cloud::adapter::Profile::Sensecore(base),
                    cloud::adapter::Profile::Sensecore(chosen),
                ) => {
                    let mut expected = base.clone();
                    expected.cluster = chosen.cluster.clone();
                    if expected != *chosen
                        || !backend
                            .definition
                            .spot_clusters
                            .as_ref()
                            .is_some_and(|s| s.contains(&chosen.cluster))
                    {
                        return Err(cloud::Error::evidence());
                    }
                }
                _ => return Err(cloud::Error::evidence()),
            }
        }
        Ok(frozen)
    }
    pub(super) fn execution_adapter(
        &self,
        execution: &Execution,
    ) -> cloud::Result<Adapter<&Client>> {
        let request: run::Create = serde_json::from_value(read(
            &self
                .authority(&execution.job.project, &execution.job.run_id)?
                .child(OsStr::new(&execution.job.run_id), false)
                .map_err(io)?,
            "request.json",
        )?)
        .map_err(|_| cloud::Error::evidence())?;
        let backend = self
            .backends
            .get(&execution.backend_id)
            .ok_or_else(cloud::Error::configuration)?;
        let profile = self
            .selected_profile(&execution.job.project, &request, None)?
            .unwrap_or_else(|| backend.adapter.profile.clone());
        if profile.digest()? != execution.profile_sha256 {
            return Err(cloud::Error::evidence());
        }
        backend.adapter.with_profile(profile)
    }
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
        selected: Option<&cloud::adapter::Profile>,
    ) -> cloud::Result<creation::Documents> {
        let backend = self
            .backends
            .get(&request.executor)
            .ok_or_else(cloud::Error::configuration)?;
        if matches!(
            backend.adapter.profile,
            cloud::adapter::Profile::Sensecore(_)
        ) && request
            .scheduling
            .as_ref()
            .is_some_and(|p| p.allocation != ml_exp_core::cloud::pai::Allocation::Guaranteed)
        {
            return Err(cloud::Error::configuration());
        }
        let runtime = self.runtime(project, &request.executor, &request.runtime_id)?;
        let selected = self.selected_profile(project, request, selected)?;
        let profile = selected.as_ref().unwrap_or(&backend.adapter.profile);
        let source = self
            .sources
            .read(project, &runtime.source_id)
            .map_err(|_| cloud::Error::evidence())?;
        let target = backend.definition.target()?;
        let data_root = backend.definition.data_root()?;
        let datasets =
            request
                .data
                .iter()
                .map(|input| {
                    if let Some(receipt) = backend
                        .definition
                        .datasets
                        .iter()
                        .find(|r| r.project == project && r.asset_id == input.asset_id)
                    {
                        return Ok(receipt.clone());
                    }
                    if matches!(
                        backend.adapter.profile,
                        cloud::adapter::Profile::Sensecore(_)
                    ) {
                        if let Some(saved) =
                            self.data_saved(project, &request.executor, &input.asset_id)?
                        {
                            return saved.state.ready.ok_or(cloud::Error {
                                code: "INPUT_NOT_READY".into(),
                                http_status: Some(409),
                                uncertain: false,
                            });
                        }
                    }
                    let definition = if let Some(recipe) =
                        backend.definition.imports.iter().find(|r| {
                            r.project == project && r.definition.asset_id() == input.asset_id
                        }) {
                        recipe.definition.clone()
                    } else {
                        self.uploaded_data(project, &request.executor, &input.asset_id)?
                            .ok_or_else(cloud::Error::evidence)?
                    };
                    Ok(ml_exp_core::datasets::Receipt {
                        contract: ml_exp_core::datasets::CONTRACT.into(),
                        project: project.into(),
                        target: target.clone(),
                        asset_id: input.asset_id.clone(),
                        definition,
                        root: format!("{data_root}/{project}/{}/tree", input.asset_id),
                    })
                })
                .collect::<cloud::Result<Vec<_>>>()?;
        let checkpoint = if request.resume_from.is_some() {
            let root = self.authority(project, &request.run_id)?;
            if root.entry_exists(OsStr::new(&request.run_id)).map_err(io)? {
                let value = read(
                    &root.child(OsStr::new(&request.run_id), false).map_err(io)?,
                    "manifest.json",
                )?["native_checkpoint"]
                    .clone();
                if value.is_null() {
                    return Err(cloud::Error::evidence());
                }
                Some(value)
            } else {
                request
                    .resume_from
                    .as_ref()
                    .map(|reference| self.restore_reference(project, &request.executor, reference))
                    .transpose()?
            }
        } else {
            None
        };
        let mut documents = run::documents(
            project,
            request,
            &runtime,
            run::Authority {
                project_sha256: &self.project_hash(project)?,
                source_receipt_sha256: &canonical::protocol_identity(&source)
                    .map_err(|_| cloud::Error::evidence())?,
                profile_sha256: &profile.digest()?,
                tracking_route: route,
                target: &target,
                run_root: &backend.definition.run_root,
                data_root: &data_root,
                datasets: &datasets,
                checkpoint: checkpoint.as_ref(),
            },
            now,
        )
        .map_err(|_| cloud::Error::evidence())?;
        if let Some(selected) = selected {
            documents.run["native_cloud"]["profile"] = json!(selected);
            documents.attempt["native_cloud"]["profile"] =
                documents.run["native_cloud"]["profile"].clone();
        }
        Ok(documents)
    }
    pub(super) fn current_execution(
        &self,
        project: &str,
        run_id: &str,
        attempt_id: &str,
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
                .child(OsStr::new(attempt_id), false)
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
            None,
        )?;
        let mut first = run.clone();
        first["attempt_id"] = json!("attempt-001");
        if !creation::same_creation(&docs, &run, &first, &original) {
            return Err(cloud::Error::evidence());
        }
        let backend = &self.backends[&request.executor];
        let runtime = self.runtime(project, &request.executor, &request.runtime_id)?;
        let binding = Binding {
            project: project.into(),
            run_id: run_id.into(),
            attempt_id: attempt_id.into(),
            project_sha256: self.project_hash(project)?,
            run_sha256: helper::sha256_bytes(&encode(&run)?),
            attempt_sha256: helper::sha256_bytes(&encode(&attempt)?),
            source_receipt_sha256: run["native_creation"]["source_receipt_sha256"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?
                .into(),
            backend_config_sha256: self
                .selected_profile(project, &request, None)?
                .as_ref()
                .unwrap_or(&backend.adapter.profile)
                .digest()?,
        };
        let mut execution = run::execution(
            &run,
            &attempt,
            binding,
            &runtime,
            &backend.definition.run_root,
            token,
            backend.definition.allocation,
        )
        .map_err(|_| cloud::Error::evidence())?;
        let mut package = execution.package().map_err(|_| cloud::Error::evidence())?;
        package.result_upload = self.objects.is_some();
        package.imports = backend
            .definition
            .imports
            .iter()
            .filter(|i| {
                i.project == project
                    && package
                        .config
                        .datasets
                        .iter()
                        .any(|m| m.receipt.asset_id == i.definition.asset_id())
            })
            .cloned()
            .collect();
        if let Some(authority) = &self.exchange {
            package.channel = Some(authority.channel(&self.workspace, &execution)?);
            for mount in &package.config.datasets {
                if !matches!(
                    backend.adapter.profile,
                    cloud::adapter::Profile::Sensecore(_)
                ) && !package
                    .imports
                    .iter()
                    .any(|i| i.definition.asset_id() == mount.receipt.asset_id)
                    && self
                        .uploaded_data(project, &request.executor, &mount.receipt.asset_id)?
                        .is_some()
                {
                    package.imports.push(ml_exp_core::datasets::import::Config {
                        root: backend.definition.data_root()?,
                        target: backend.definition.target()?,
                        project: project.into(),
                        definition: mount.receipt.definition.clone(),
                        source: ml_exp_core::datasets::import::Source::ExchangeParts {
                            channel: package
                                .channel
                                .clone()
                                .ok_or_else(cloud::Error::configuration)?,
                        },
                        timeout_seconds: 3600,
                    });
                }
            }
        }
        execution.job.command = package
            .command(&runtime.worker_path)
            .map_err(|_| cloud::Error::evidence())?;
        Ok(execution)
    }
    pub(in crate::cloud_runs) fn check(&self, prepared: &Prepared) -> cloud::Result<()> {
        let current = self.current_execution(
            &prepared.execution.job.project,
            &prepared.execution.job.run_id,
            &prepared.execution.job.attempt_id,
            &prepared.execution.job.submission_token,
        )?;
        if current != prepared.execution {
            return Err(cloud::Error::evidence());
        }
        action::validate(&prepared.plan, &current).map_err(|_| cloud::Error::evidence())
    }
}
