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
        self.facts_for(project, run, None)
    }
    pub(super) fn facts_for(&self, project: &str, run: &str, exact: Option<&str>) -> Result<Facts> {
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
            || snapshot.attempts.is_empty()
            || snapshot.attempts.len() > 256
            || !snapshot.diagnostics.is_empty()
        {
            return Err(Error::Authority);
        }
        let selected = exact
            .or(snapshot.current_attempt_id.as_deref())
            .ok_or(Error::Authority)?;
        let attempt = snapshot
            .attempts
            .iter()
            .find(|v| v.attempt_id == selected)
            .ok_or(Error::Authority)?;
        native::retry::number(selected)?;
        if attempt.manifest.legacy_observation_only
            || !attempt.provenance.compared_fields_complete
            || matches!(attempt.status, EvidenceDocument::Unavailable { .. })
            || matches!(attempt.submission, EvidenceDocument::Unavailable { .. })
            || matches!(attempt.backend, EvidenceDocument::Unavailable { .. })
            || matches!(attempt.collection, EvidenceDocument::Unavailable { .. })
        {
            return Err(Error::Authority);
        }
        let status = if snapshot.current_attempt_id.as_deref() == Some(selected) {
            &snapshot.current_status
        } else {
            &attempt.status
        };
        let scheduler_state = status
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
        self.runtime(project, &backend, &runtime)?;
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
                || snapshot.current_attempt_id.as_deref() == Some(selected)
                    && !matches!(snapshot.current_collection, EvidenceDocument::Missing)
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
        let facts = self.facts_for(
            &prepared.binding.project,
            &prepared.binding.run_id,
            Some(&prepared.binding.attempt_id),
        )?;
        let backend = &self.backends[&facts.backend];
        if facts.binding != prepared.binding
            || native::definition(
                &facts.run,
                &facts.attempt,
                &facts.binding,
                &backend.profile,
                &self.runtime(&facts.binding.project, &facts.backend, &facts.runtime)?,
                &prepared.approval,
            )? != prepared.definition_sha256
        {
            return Err(Error::Conflict);
        }
        let saved = self
            .saved_attempt(
                &facts.binding.project,
                &facts.binding.run_id,
                &facts.binding.attempt_id,
            )?
            .ok_or(Error::Authority)?;
        if saved.plan != prepared.plan || saved.request != prepared.request {
            let action = prepared.plan["action_id"]
                .as_str()
                .ok_or(Error::Authority)?;
            let name = ml_exp_core::renewal::archive(action).map_err(|_| Error::Authority)?;
            let archived: Prepared = crate::renewal::read(&self.slots, &name)
                .map_err(|_| Error::Authority)?
                .ok_or(Error::Authority)?;
            if archived.plan != prepared.plan
                || archived.request != prepared.request
                || archived.binding != prepared.binding
                || saved.request != prepared.request
            {
                return Err(Error::Authority);
            }
        }
        if prepared.binding.attempt_id != "attempt-001" {
            let receipt =
                native::retry::validate_attempt(&facts.run, &facts.attempt, &facts.binding)?;
            let parent = self
                .saved_attempt(
                    &facts.binding.project,
                    &facts.binding.run_id,
                    &receipt.parent_binding.attempt_id,
                )?
                .ok_or(Error::Authority)?;
            let parent_facts = self.facts_for(
                &facts.binding.project,
                &facts.binding.run_id,
                Some(&receipt.parent_binding.attempt_id),
            )?;
            if parent.binding != receipt.parent_binding
                || parent_facts.binding != parent.binding
                || parent.plan["action_id"] != receipt.parent_action
            {
                return Err(Error::Authority);
            }
        }
        Ok(())
    }
    pub(super) fn load(&self, action: &str) -> Result<Prepared> {
        let snapshot = self.actions.observe(action)?;
        let project = snapshot["scope"]["project"]
            .as_str()
            .ok_or(Error::Authority)?;
        let run = snapshot["run_id"].as_str().ok_or(Error::Authority)?;
        let attempt = snapshot["attempt_id"].as_str().ok_or(Error::Authority)?;
        let name = ml_exp_core::renewal::archive(action).map_err(|_| Error::Authority)?;
        let prepared =
            match crate::renewal::read(&self.slots, &name).map_err(|_| Error::Authority)? {
                Some(value) => value,
                None => self
                    .saved_attempt(project, run, attempt)?
                    .ok_or(Error::Authority)?,
            };
        if prepared.plan["action_id"] != action {
            return Err(Error::Authority);
        }
        self.check(&prepared)?;
        Ok(prepared)
    }
}
