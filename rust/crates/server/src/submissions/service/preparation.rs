//! Bounded immutable preparation slots and at-most-once admission publication.
use super::*;

pub(super) struct PreparationLock<'a> {
    service: &'a Service,
    file: File,
}
impl PreparationLock<'_> {
    pub(super) fn validate(&self) -> Result<()> {
        self.service
            .directory
            .validate_regular_entry(OsStr::new(LOCK), &self.service.lock_anchor)
            .map_err(|_| Error::Authority)?;
        self.service
            .directory
            .validate_regular_entry(OsStr::new(LOCK), &self.file)
            .map_err(|_| Error::Authority)
    }
}
impl Drop for PreparationLock<'_> {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}
impl Service {
    pub(super) fn lock(&self) -> Result<PreparationLock<'_>> {
        self.directory
            .validate_regular_entry(OsStr::new(LOCK), &self.lock_anchor)
            .map_err(|_| Error::Authority)?;
        let file = self
            .directory
            .open_existing_lock(OsStr::new(LOCK))
            .map_err(|_| Error::Authority)?;
        file.try_lock_exclusive().map_err(|error| {
            if error.kind() == std::io::ErrorKind::WouldBlock {
                Error::Busy
            } else {
                Error::Authority
            }
        })?;
        let lock = PreparationLock {
            service: self,
            file,
        };
        lock.validate()?;
        Ok(lock)
    }
    fn slot_name(project: &str, run: &str) -> Result<String> {
        Ok(format!("{}.json", native::slot(project, run)?))
    }
    pub(super) fn saved(&self, project: &str, run: &str) -> Result<Option<Prepared>> {
        let name = Self::slot_name(project, run)?;
        match self
            .slots
            .regular_identity(OsStr::new(&name))
            .map_err(|_| Error::Authority)?
        {
            None => return Ok(None),
            Some(identity)
                if identity.is_private_for(effective_uid()) && identity.mode & 0o077 == 0 => {}
            Some(_) => return Err(Error::Authority),
        }
        let value: Prepared = serde_json::from_slice(
            &self
                .slots
                .read_regular(OsStr::new(&name), MAX_SLOT_BYTES)
                .map_err(|_| Error::Authority)?,
        )
        .map_err(|_| Error::Authority)?;
        value.validate()?;
        if value.binding.project != project || value.binding.run_id != run {
            return Err(Error::Authority);
        }
        Ok(Some(value))
    }
    /// A single immutable slot per Run survives response loss and restart.
    /// Token generation occurs only after proving the slot does not exist.
    pub fn prepare(
        &self,
        project: &str,
        run: &str,
        approval: Approval,
        context: Context<'_>,
    ) -> Result<Value> {
        let _guard = self
            .fence
            .existing_observation_guard(project, run)
            .map_err(|_| Error::Authority)?;
        let lock = self.lock()?;
        let facts = self.facts(project, run)?;
        let backend = &self.backends[&facts.backend];
        let definition = native::definition(
            &facts.run,
            &facts.attempt,
            &facts.binding,
            &backend.profile,
            &backend.runtimes[&facts.runtime],
            &approval,
        )?;
        let reused;
        let prepared = if let Some(saved) = self.saved(project, run)? {
            reused = true;
            if saved.definition_sha256 != definition || saved.binding != facts.binding {
                return Err(Error::Conflict);
            }
            saved
        } else {
            reused = false;
            if facts.scheduler_state != "NOT_SUBMITTED" || facts.submitted {
                return Err(Error::Authority);
            }
            let scope = ml_exp_core::operations::Scope {
                project: project.into(),
                scope_type: ml_exp_core::operations::ScopeType::Run,
                object_id: run.into(),
            };
            if self.actions.has_scope_history(&scope)? {
                return Err(Error::Conflict);
            }
            let mut token = [0u8; 16];
            getrandom::fill(&mut token).map_err(|_| Error::Configuration)?;
            let token = format!("{:032x}", u128::from_be_bytes(token));
            let prepared = native::prepare(
                &facts.run,
                &facts.attempt,
                facts.binding.clone(),
                &backend.profile,
                &backend.runtimes[&facts.runtime],
                approval,
                &token,
                context.now,
                context.expires,
            )?;
            lock.validate()?;
            let bytes = serde_json::to_vec(&prepared).map_err(|_| Error::Configuration)?;
            if bytes.len() as u64 > MAX_SLOT_BYTES {
                return Err(Error::Configuration);
            }
            self.slots
                .publish_new(OsStr::new(&Self::slot_name(project, run)?), &bytes)
                .map_err(|_| Error::Uncertain)?;
            prepared
        };
        self.check(&prepared)?;
        let snapshot =
            self.actions
                .save_plan(&prepared.plan, context.now, context.transition_id)?;
        self.actions.bind_slurm_request(
            snapshot["action_id"].as_str().ok_or(Error::Authority)?,
            &prepared.request,
        )?;
        lock.validate().map_err(|_| Error::Uncertain)?;
        self.check(&prepared).map_err(|_| Error::Uncertain)?;
        let mut view = self.public(snapshot, context.now)?;
        view["reused"] = json!(reused);
        Ok(view)
    }
}
