use super::*;
fn launch_error(error: ml_exp_core::cloud::Error) -> cloud::Error {
    if error.0 == "cloud package size" {
        cloud::Error {
            code: "LAUNCH_METADATA_TOO_LARGE".into(),
            http_status: Some(422),
            uncertain: false,
        }
    } else {
        cloud::Error::evidence()
    }
}
/// A planning refusal the caller can act on keeps its code. Anything else about the
/// frozen authority stays the generic evidence refusal.
fn planning_error(error: ml_exp_core::cloud::Error) -> cloud::Error {
    if error.0 == ml_exp_core::datasets::ManifestsTooLarge::CAUSE {
        cloud::Error {
            code: "DATASET_MANIFESTS_TOO_LARGE".into(),
            http_status: Some(422),
            uncertain: false,
        }
    } else {
        crate::why::record(file!(), line!(), &error);
        cloud::Error::evidence()
    }
}
impl Service {
    /// The receipt an undelivered input will have: its import recipe, or else its
    /// verified upload, under this backend's data root. A missing input is evidence.
    pub(super) fn undelivered_receipt(
        &self,
        backend: &Backend,
        project: &str,
        executor: &str,
        asset: &str,
    ) -> cloud::Result<ml_exp_core::datasets::Receipt> {
        let definition = if let Some(recipe) = backend
            .definition
            .imports
            .iter()
            .find(|r| r.project == project && r.definition.asset_id() == asset)
        {
            recipe.definition.clone()
        } else {
            self.uploaded_data(project, executor, asset)?
                .ok_or_else(cloud::Error::evidence)?
        };
        Ok(ml_exp_core::datasets::Receipt {
            contract: ml_exp_core::datasets::CONTRACT.into(),
            project: project.into(),
            target: backend.definition.target()?,
            asset_id: asset.into(),
            definition,
            root: format!("{}/{project}/{asset}/tree", backend.definition.data_root()?),
        })
    }
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
                    .map_err(why!(cloud::Error::evidence()))?
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
    ) -> cloud::Result<Adapter<&Shared>> {
        let request: run::Create = serde_json::from_value(read(
            &self
                .authority(&execution.job.project, &execution.job.run_id)?
                .child(OsStr::new(&execution.job.run_id), false)
                .map_err(io)?,
            "request.json",
        )?)
        .map_err(why!(cloud::Error::evidence()))?;
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
    /// The project lifecycle gate. An archived project takes no further effect.
    fn require_active(&self, project: &str) -> cloud::Result<()> {
        if self
            .lifecycle
            .as_ref()
            .is_some_and(|l| !l.active_projects().is_ok_and(|p| p.contains(project)))
        {
            return Err(cloud::Error::evidence());
        }
        Ok(())
    }
    /// Fence and lifecycle at acquisition: the Run has no tombstone and its project
    /// is active. These stay enforced on every operation, unlike `check`.
    pub(super) fn require_live(&self, project: &str, run: &str) -> cloud::Result<()> {
        let _guard = self
            .fence
            .existing_observation_guard(project, run)
            .map_err(why!(cloud::Error::evidence()))?;
        self.require_active(project)
    }
    /// The same two gates under a guard already held across an effect.
    pub(super) fn still_live(
        &self,
        guard: &crate::deletion::Guard<'_>,
        project: &str,
        run: &str,
    ) -> cloud::Result<()> {
        guard
            .require_observed_live(project, run)
            .map_err(why!(cloud::Error::evidence()))?;
        self.require_active(project)
    }
    pub(super) fn authority(&self, project: &str, run: &str) -> cloud::Result<Directory> {
        ml_exp_core::identity::validate_id(project)
            .and_then(|_| ml_exp_core::identity::validate_id(run))
            .map_err(why!(cloud::Error::configuration()))?;
        let (_, workspace) = read_config::workspace(&self.settings.registry_root)
            .map_err(why!(cloud::Error::evidence()))?;
        if workspace != self.workspace {
            return Err(cloud::Error::evidence());
        }
        self.require_live(project, run)?;
        let project = self
            .settings
            .projects
            .iter()
            .find(|p| p.project == project && p.discovery_enabled() && p.run_roots.len() == 1)
            .ok_or_else(cloud::Error::configuration)?;
        Directory::open(&project.run_roots[0], false).map_err(io)
    }
    pub(super) fn project_hash(&self, project: &str) -> cloud::Result<String> {
        experiments::Store::open(self.settings.bindings(), self.fence.clone())
            .map_err(why!(cloud::Error::evidence()))?
            .project(project, None)
            .map(|p| p.raw_sha256)
            .map_err(why!(cloud::Error::evidence()))
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
            .map_err(why!(cloud::Error::evidence()))?;
        let target = backend.definition.target()?;
        let data_root = backend.definition.data_root()?;
        let datasets = request
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
                if let Some(saved) = self.data_saved(project, &request.executor, &input.asset_id)? {
                    if let Some(receipt) = saved.state.ready {
                        return Ok(receipt);
                    }
                    if matches!(
                        backend.adapter.profile,
                        cloud::adapter::Profile::Sensecore(_)
                    ) {
                        return Err(cloud::Error {
                            code: "INPUT_NOT_READY".into(),
                            http_status: Some(409),
                            uncertain: false,
                        });
                    }
                }
                self.undelivered_receipt(backend, project, &request.executor, &input.asset_id)
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
                    .map_err(why!(cloud::Error::evidence()))?,
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
        .map_err(planning_error)?;
        let storage = backend.definition.storage_layout()?;
        if request.data_preparation.is_some() {
            let before_job = matches!(
                backend.adapter.profile,
                cloud::adapter::Profile::Sensecore(_)
            );
            if before_job && backend.data_copier.is_none() {
                return Err(cloud::Error::configuration());
            }
            documents.run["native_cloud"]["data_preparation_before_job"] = json!(before_job);
            documents.attempt["native_cloud"]["data_preparation_before_job"] = json!(before_job);
        }
        let root = self.authority(project, &request.run_id)?;
        // Authorship freezes this provisioning plan. A missing or conflicting plan
        // in an existing Run is evidence of a problem and is never repaired.
        if root.entry_exists(OsStr::new(&request.run_id)).map_err(io)?
            && read(
                &root.child(OsStr::new(&request.run_id), false).map_err(io)?,
                "manifest.json",
            )?["native_cloud"]
                .get("storage")
                != Some(&json!(storage))
        {
            return Err(cloud::Error::evidence());
        }
        documents.run["native_cloud"]["storage"] = json!(storage);
        documents.attempt["native_cloud"]["storage"] = json!(storage);
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
            serde_json::from_value(original.clone()).map_err(why!(cloud::Error::evidence()))?;
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
        let channel = self
            .exchange
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .channel_for(&self.workspace, &binding, token)?;
        let mut execution = run::execution(
            &run,
            &attempt,
            binding,
            &runtime,
            &backend.definition.run_root,
            token,
            backend.definition.allocation,
            channel,
        )
        .map_err(launch_error)?;
        let mut package = execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        package.result_upload = self.objects.is_some();
        let mut materialized = std::collections::BTreeSet::new();
        for mount in &package.config.datasets {
            if let Some(receipt) = self
                .data_saved(project, &request.executor, &mount.receipt.asset_id)?
                .and_then(|s| s.state.ready)
            {
                if receipt != mount.receipt {
                    return Err(cloud::Error::evidence());
                }
                materialized.insert(mount.receipt.asset_id.clone());
            }
        }
        package.imports = backend
            .definition
            .imports
            .iter()
            .filter(|i| {
                i.project == project
                    && !materialized.contains(&i.definition.asset_id())
                    && package
                        .config
                        .datasets
                        .iter()
                        .any(|m| m.receipt.asset_id == i.definition.asset_id())
            })
            .cloned()
            .collect();
        for mount in &package.config.datasets {
            if !matches!(
                backend.adapter.profile,
                cloud::adapter::Profile::Sensecore(_)
            ) && !materialized.contains(&mount.receipt.asset_id)
                && !package
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
                        channel: package.channel.clone(),
                    },
                    timeout_seconds: 3600,
                });
            }
        }
        execution.job.command = package
            .download_command(&runtime.worker_path)
            .map_err(launch_error)?;
        execution.launch_package = Box::new(package);
        Ok(execution)
    }
    /// The first-spend gate: the Attempt about to be spent still matches its frozen
    /// Run, source and profile. It runs before reuse, authorization and the claim,
    /// never afterwards, and before a retry publishes the Attempt that would be spent
    /// next. A running Attempt stays readable and cancellable after an edit; a drifted
    /// one that has not started stays unspendable, and the remedy is a new Run.
    pub(super) fn check(&self, prepared: &Prepared) -> cloud::Result<()> {
        let mut current = self.current_execution(
            &prepared.execution.job.project,
            &prepared.execution.job.run_id,
            &prepared.execution.job.attempt_id,
            &prepared.execution.job.submission_token,
        )?;
        // The submitted command belongs to the saved Action. A change to the
        // service's public URL only affects future submissions; it cannot change
        // whether the original job's evidence is readable.
        current.job.command = prepared.execution.job.command.clone();
        current.launch_package = prepared.execution.launch_package.clone();
        if current != prepared.execution {
            return Err(cloud::Error::evidence());
        }
        action::validate(&prepared.plan, &prepared.execution)
            .map_err(why!(cloud::Error::evidence()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_core::cloud::launch;
    #[test]
    fn dataset_manifests_over_their_budget_are_a_typed_refusal_not_missing_evidence() {
        let refused = planning_error(ml_exp_core::cloud::Error(
            ml_exp_core::datasets::ManifestsTooLarge::CAUSE,
        ));
        assert_eq!(refused.code, "DATASET_MANIFESTS_TOO_LARGE");
        assert_eq!(refused.http_status, Some(422));
        assert!(!refused.uncertain);
        let public = failure(refused);
        assert_eq!(
            (public.status, public.code),
            (422, "DATASET_MANIFESTS_TOO_LARGE")
        );
        assert_eq!(
            planning_error(ml_exp_core::cloud::Error("cloud dataset scope")).code,
            cloud::Error::evidence().code
        );
    }
    /// Request, manifest and receipt bounds keep an ordinary Run far below 8 MiB, so
    /// this compiles the oversized package directly rather than through a Run.
    #[test]
    fn a_package_over_its_download_budget_is_a_typed_refusal_not_missing_evidence() {
        let mut package = launch::compile(launch::Input {
            root: "/mnt/workspace/runs",
            project: "demo",
            run: "train",
            attempt: "attempt-001",
            workdir: "/workspace",
            argv: &["train".into()],
            timeout_seconds: 300,
            worker_sha256: &format!("sha256:{}", "a".repeat(64)),
            storage: launch::StorageLayout {
                mount: "/mnt/workspace".into(),
                runs: "/mnt/workspace/runs".into(),
                datasets: "/mnt/workspace/datasets".into(),
            },
            channel: ml_exp_core::cloud::exchange::Channel {
                url: "https://api.example/api/worker-exchange/demo/train/attempt-001".into(),
                token: "b".repeat(64),
            },
        })
        .unwrap();
        package.validate().unwrap();
        package.artifacts = Some(ml_exp_core::artifacts::publication::Policy {
            required_files: (0..20_000)
                .map(|i| format!("{}/{i:05}", "x".repeat(500)))
                .collect(),
        });
        assert!(
            serde_json::to_vec(&package).unwrap().len() > launch::MAX_PACKAGE_BYTES,
            "the fixture must exceed the budget"
        );
        let refused = launch_error(package.validate().unwrap_err());
        assert_eq!(refused.code, "LAUNCH_METADATA_TOO_LARGE");
        assert_eq!(refused.http_status, Some(422));
        assert!(!refused.uncertain);
        let command = package.download_command("/worker").err().unwrap();
        assert_eq!(launch_error(command).code, "LAUNCH_METADATA_TOO_LARGE");
        let public = failure(refused);
        assert_eq!(
            (public.status, public.code),
            (422, "LAUNCH_METADATA_TOO_LARGE")
        );
        // Any other compile failure stays the generic evidence refusal.
        assert_eq!(
            launch_error(ml_exp_core::cloud::Error("cloud storage binding")).code,
            cloud::Error::evidence().code
        );
    }
}
