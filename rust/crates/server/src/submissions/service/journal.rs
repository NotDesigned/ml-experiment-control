//! Canonical observation ingress stays independent from optional publication.
use super::*;
impl Service {
    pub(super) fn journal_terminal_recorded(&self, prepared: &Prepared) -> bool {
        let store = crate::records::Store::read_only(&self.settings.registry_root);
        let Ok(guard) = store.guard(&prepared.binding.project, &prepared.binding.run_id) else {
            return false;
        };
        let Ok(scope) = ml_exp_core::records::Scope::new(
            &prepared.binding.project,
            &prepared.binding.run_id,
            &prepared.binding.attempt_id,
        )
        .encoded() else {
            return false;
        };
        store
            .latest_lifecycle(&guard, &scope)
            .is_ok_and(|value| value["phase"] == "METRICS_DRAINED")
    }
    pub(super) fn journal_training_end(
        &self,
        prepared: &Prepared,
        receipt: &ml_exp_core::managed::evidence::Receipt,
    ) -> Result<()> {
        self.journal_lifecycle(
            prepared,
            json!({"phase":"METRICS_DRAINED","state":match receipt.training {
            ml_exp_core::managed::evidence::Training::Completed => "SUCCEEDED",
            ml_exp_core::managed::evidence::Training::Cancelled => "CANCELLED",
            ml_exp_core::managed::evidence::Training::TimedOut => "TIMEOUT",
            ml_exp_core::managed::evidence::Training::Failed => "FAILED",
            ml_exp_core::managed::evidence::Training::Unknown => "UNKNOWN",
        },"exit_code":receipt.process.exit_code}),
        )
    }
    pub fn tracking_status(&self, project: &str, run: &str) -> Result<Value> {
        self.validate_workspace(project, run)?;
        let facts = self.facts(project, run)?;
        let route = facts
            .run
            .get("tracking_route")
            .cloned()
            .unwrap_or_else(|| json!({"enabled":false,"reason":"NATIVE_LOCAL_ONLY"}));
        let root =
            Directory::open(&self.settings.registry_root, false).map_err(|_| Error::Authority)?;
        if !root
            .entry_exists(OsStr::new("tracking"))
            .map_err(|_| Error::Authority)?
        {
            return Ok(
                json!({"project":project,"run_id":run,"wandb":route,"scopes":[],"status":"NOT_STARTED"}),
            );
        }
        let store = crate::records::Store::read_only(&self.settings.registry_root);
        let guard = store.guard(project, run).map_err(|_| Error::Authority)?;
        let binding = store.binding(&guard).map_err(|_| Error::Authority)?;
        let snapshot = self
            .evidence
            .snapshot_metadata(&facts.reference)
            .map_err(|_| Error::Authority)?;
        if let Some(binding) = &binding {
            if !crate::read_api::journal_binding_matches(&snapshot, Some(binding)) {
                return Err(Error::ObservationConflict);
            }
        }
        let mut scopes = store.run_scopes(&guard).map_err(|_| Error::Authority)?;
        let frozen = binding
            .as_ref()
            .map(|b| b["route"].clone())
            .unwrap_or(route);
        for scope in &mut scopes {
            scope["pending"] = json!(
                scope["total"].as_i64().ok_or(Error::Authority)?
                    - scope["confirmed"].as_i64().ok_or(Error::Authority)?
            );
            if frozen["enabled"] == true {
                scope["url"] = json!(format!(
                    "https://wandb.ai/{}/{}/runs/{}",
                    frozen["entity"].as_str().ok_or(Error::Authority)?,
                    frozen["project"].as_str().ok_or(Error::Authority)?,
                    scope["wandb_id"].as_str().ok_or(Error::Authority)?
                ));
            }
        }
        Ok(
            json!({"project":project,"run_id":run,"wandb":frozen,"scopes":scopes,"source":"local-canonical-journal","cloud_query_performed":false}),
        )
    }
    pub fn publisher_store(&self) -> Result<crate::records::Store> {
        crate::records::Store::open(
            &self.settings.registry_root,
            self.workspace
                .strip_prefix("rust-preview-")
                .ok_or(Error::Authority)?,
        )
        .map_err(|_| Error::ObservationConflict)
    }
    pub fn tracking_journal(&self, project: &str, run: &str) -> Result<crate::records::Store> {
        self.validate_workspace(project, run)?;
        self.journal(&self.saved(project, run)?.ok_or(Error::Missing)?)
    }
    pub fn publisher_leader(&self) -> Result<File> {
        let file = self
            .directory
            .open_lock(OsStr::new(".publisher.lock"))
            .map_err(|_| Error::Authority)?;
        file.try_lock_exclusive().map_err(|_| Error::Busy)?;
        self.directory
            .validate_regular_entry(OsStr::new(".publisher.lock"), &file)
            .map_err(|_| Error::Authority)?;
        Ok(file)
    }
    pub fn validate_publisher_leader(&self, file: &File) -> Result<()> {
        self.directory
            .validate_regular_entry(OsStr::new(".publisher.lock"), file)
            .map_err(|_| Error::Authority)
    }
    pub(super) fn journal(&self, prepared: &Prepared) -> Result<crate::records::Store> {
        let seed = self
            .workspace
            .strip_prefix("rust-preview-")
            .ok_or(Error::Authority)?;
        let records = crate::records::Store::open(&self.settings.registry_root, seed)
            .map_err(|_| Error::ObservationConflict)?;
        let guard = records
            .guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        if records.workspace_id(&guard).map_err(|_| Error::Authority)? != seed {
            return Err(Error::Authority);
        }
        let facts = self.facts_for(
            &prepared.binding.project,
            &prepared.binding.run_id,
            Some(&prepared.binding.attempt_id),
        )?;
        if records
            .binding(&guard)
            .map_err(|_| Error::Authority)?
            .is_none()
        {
            let binding =
                ml_exp_core::records::observed_binding(&facts.run, &facts.binding.run_sha256)
                    .map_err(|_| Error::Authority)?;
            records
                .bind(&guard, &binding["route"], &binding["config"])
                .map_err(|_| Error::ObservationConflict)?;
        }
        let snapshot = self
            .evidence
            .snapshot_metadata(&facts.reference)
            .map_err(|_| Error::Authority)?;
        let binding = records.binding(&guard).map_err(|_| Error::Authority)?;
        if !crate::read_api::journal_binding_matches(&snapshot, binding.as_ref()) {
            return Err(Error::ObservationConflict);
        }
        drop(guard);
        Ok(records)
    }
    pub(super) fn journal_lifecycle(&self, prepared: &Prepared, data: Value) -> Result<()> {
        let records = self.journal(prepared)?;
        let guard = records
            .guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        let scope = records
            .scope(&guard, &prepared.binding.attempt_id)
            .map_err(|_| Error::ObservationConflict)?;
        let payload = json!({"kind":"lifecycle","data":data});
        let event = ml_exp_core::records::Event {
            event_id: format!(
                "native.{}",
                canonical::digest(&payload).map_err(|_| Error::Authority)?
            ),
            payload,
        };
        records
            .append(&guard, &scope, &[event])
            .map_err(|_| Error::ObservationConflict)?;
        Ok(())
    }
}
