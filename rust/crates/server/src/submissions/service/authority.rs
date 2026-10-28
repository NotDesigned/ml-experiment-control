//! Authoritative Run/source/profile checks shared by every submission phase.
use super::*;

impl Service {
    pub(super) fn validate_workspace(&self, project: &str, run: &str) -> Result<()> {
        let (_, workspace) =
            read_config::workspace(&self.settings.registry_root).map_err(|_| Error::Authority)?;
        if workspace != self.workspace {
            return Err(Error::Authority);
        }
        self.directory
            .verify_private_parent()
            .map_err(|_| Error::Authority)?;
        self.slots
            .verify_private_parent()
            .map_err(|_| Error::Authority)?;
        self.directory
            .validate_regular_entry(OsStr::new(LOCK), &self.lock_anchor)
            .map_err(|_| Error::Authority)?;
        self.fence
            .existing_observation_guard(project, run)
            .map_err(|_| Error::Authority)?;
        if let Some(lifecycle) = &self.lifecycle {
            if !lifecycle
                .active_projects()
                .map_err(|_| Error::Authority)?
                .contains(project)
            {
                return Err(Error::Authority);
            }
        }
        Ok(())
    }
    pub(super) fn resolve(&self, project: &str, run: &str) -> Result<RunRef> {
        let approved = self
            .settings
            .projects
            .iter()
            .find(|p| p.project == project)
            .ok_or(Error::Authority)?;
        if approved.discovery_enabled() {
            let discovery = self
                .evidence
                .discover(project, &experiments::DiscoveryLimits::default())
                .map_err(|_| Error::Authority)?;
            discovery
                .runs
                .into_iter()
                .find(|r| r.reference.run_id == run)
                .map(|r| r.reference)
                .ok_or(Error::Authority)
        } else {
            self.settings
                .runs()
                .into_iter()
                .find(|r| r.project == project && r.run_id == run)
                .ok_or(Error::Authority)
        }
    }
    pub(super) fn facts(&self, project: &str, run: &str) -> Result<Facts> {
        self.validate_workspace(project, run)?;
        let reference = self.resolve(project, run)?;
        let project_document = self
            .evidence
            .project(project, None)
            .map_err(|_| Error::Authority)?;
        let snapshot = self
            .evidence
            .snapshot_metadata(&reference)
            .map_err(|_| Error::Authority)?;
        if snapshot.run.legacy_observation_only
            || snapshot.attempts.len() != 1
            || snapshot.attempts[0].attempt_id != "attempt-001"
            || !snapshot.diagnostics.is_empty()
        {
            return Err(Error::Authority);
        }
        let attempt = &snapshot.attempts[0];
        if attempt.manifest.legacy_observation_only
            || !attempt.provenance.compared_fields_complete
            || matches!(attempt.status, EvidenceDocument::Unavailable { .. })
            || matches!(attempt.submission, EvidenceDocument::Unavailable { .. })
            || matches!(attempt.backend, EvidenceDocument::Unavailable { .. })
            || matches!(attempt.collection, EvidenceDocument::Unavailable { .. })
        {
            return Err(Error::Authority);
        }
        let scheduler_state = snapshot
            .current_status
            .metadata()
            .and_then(|v| v.get("state"))
            .and_then(Value::as_str)
            .ok_or(Error::Authority)?
            .to_owned();
        let run_value = Value::Object(snapshot.run.metadata);
        let backend = self
            .backends
            .iter()
            .find(|(_, backend)| {
                run_value["backend"]["ssh_alias"] == backend.profile.ssh_alias
                    && run_value["backend"]["partition"] == backend.profile.partition
            })
            .map(|(name, _)| name.clone())
            .ok_or(Error::Authority)?;
        let runtime = run_value["container"]["runtime_id"]
            .as_str()
            .ok_or(Error::Authority)?
            .to_owned();
        if !self.backends[&backend].runtimes.contains_key(&runtime) {
            return Err(Error::Authority);
        }
        let source = run_value["source_id"].as_str().ok_or(Error::Authority)?;
        let source_receipt = self
            .sources
            .read(project, source)
            .map_err(|_| Error::Authority)?;
        let binding = Binding {
            project: project.into(),
            run_id: run.into(),
            attempt_id: attempt.attempt_id.clone(),
            project_sha256: project_document.raw_sha256,
            run_sha256: snapshot.run.raw_sha256,
            attempt_sha256: attempt.manifest.raw_sha256.clone(),
            source_receipt_sha256: canonical::protocol_identity(&source_receipt)
                .map_err(|_| Error::Authority)?,
            backend_config_sha256: self.backends[&backend].fingerprint()?,
        };
        let facts = Facts {
            reference,
            run: run_value,
            attempt: Value::Object(attempt.manifest.metadata.clone()),
            binding,
            scheduler_state,
            submitted: !matches!(attempt.submission, EvidenceDocument::Missing)
                || !matches!(attempt.backend, EvidenceDocument::Missing)
                || !matches!(attempt.collection, EvidenceDocument::Missing)
                || !matches!(snapshot.current_collection, EvidenceDocument::Missing)
                || attempt
                    .status
                    .metadata()
                    .and_then(|v| v.get("state"))
                    .and_then(Value::as_str)
                    != Some("NOT_SUBMITTED"),
            backend,
            runtime,
        };
        self.validate_workspace(project, run)?;
        if self.resolve(project, run)? != facts.reference {
            return Err(Error::Authority);
        }
        Ok(facts)
    }
    pub(super) fn check(&self, prepared: &Prepared) -> Result<()> {
        prepared.validate()?;
        let facts = self.facts(&prepared.binding.project, &prepared.binding.run_id)?;
        let backend = &self.backends[&facts.backend];
        if facts.binding != prepared.binding
            || native::definition(
                &facts.run,
                &facts.attempt,
                &facts.binding,
                &backend.profile,
                &backend.runtimes[&facts.runtime],
                &prepared.approval,
            )? != prepared.definition_sha256
        {
            return Err(Error::Conflict);
        }
        let saved = self
            .saved(&facts.binding.project, &facts.binding.run_id)?
            .ok_or(Error::Authority)?;
        if saved.plan != prepared.plan || saved.request != prepared.request {
            return Err(Error::Authority);
        }
        Ok(())
    }
    pub(super) fn load(&self, action: &str) -> Result<Prepared> {
        let snapshot = self.actions.observe(action)?;
        let project = snapshot["scope"]["project"]
            .as_str()
            .ok_or(Error::Authority)?;
        let run = snapshot["run_id"].as_str().ok_or(Error::Authority)?;
        let prepared = self.saved(project, run)?.ok_or(Error::Authority)?;
        if prepared.plan["action_id"] != action {
            return Err(Error::Authority);
        }
        self.check(&prepared)?;
        Ok(prepared)
    }
}
