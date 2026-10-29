//! Compose namespaces by immutable identity, never by whether a read failed.
//! Corrupt, deleted, and ambiguous native evidence cannot fall back to history.
use super::*;

impl Api {
    pub fn with_history(
        mut self,
        history: Arc<crate::history::Bridge>,
    ) -> Result<Self, read_config::Error> {
        if self.is_historical()
            || self.legacy.is_some()
            || self.cloud.is_some()
            || self.submissions.is_some()
            || self.administration.is_some()
        {
            return Err(read_config::Error);
        }
        history.validate_separation(&self.approved_roots)?;
        self.legacy = Some(history);
        Ok(self)
    }
    pub fn history(&self) -> Option<Arc<crate::history::Bridge>> {
        self.legacy.clone()
    }

    /// Claims include configured exact locators and deletion tombstones, even
    /// when the payload directory is gone or its Project has been archived.
    pub(crate) fn claims(&self, project: &str, run: &str) -> Result<bool, Error> {
        validate_id(project).map_err(|_| invalid_query())?;
        validate_observation_id(run).map_err(|_| invalid_query())?;
        self.observe(|api| {
            api.validate_registry()?;
            if let Some(history) = &api.historical {
                history.snapshot()?;
            }
            if api.fence.is_deleted(project, run).map_err(|_| {
                error(
                    409,
                    "RUN_NAMESPACE_UNAVAILABLE",
                    "Run deletion evidence is unavailable",
                )
            })? {
                return Ok(true);
            }
            if !api.project_files.contains_key(project) {
                return Ok(false);
            }
            if api.discovery_projects.contains(project) {
                let found = api.store.discover(project, &Default::default())?;
                Ok(found
                    .runs
                    .iter()
                    .chain(&found.deleted)
                    .any(|r| r.reference.run_id == run))
            } else {
                Ok(api.runs.contains_key(&(project.into(), run.into())))
            }
        })
    }
    pub(super) fn history_for(&self, project: &str, run: &str) -> Result<Option<&Api>, Error> {
        let Some(history) = &self.legacy else {
            return Ok(None);
        };
        let native = self.claims(project, run)?;
        let old = history.claims(project, run)?;
        if native && old {
            return Err(crate::history::identity_conflict());
        }
        Ok((!native && old).then_some(history.api()))
    }
    pub(super) fn combined_projects(&self, native: Value, now: f64) -> Result<Value, Error> {
        let Some(history) = &self.legacy else {
            return Ok(native);
        };
        // Check all native identities, including inactive Projects, before
        // combining counters. Never hide a duplicate behind an aggregate.
        for project in &self.projects {
            let references = if self.discovery_projects.contains(project) {
                let d = self.store.discover(project, &Default::default())?;
                d.runs
                    .into_iter()
                    .chain(d.deleted)
                    .map(|r| r.reference)
                    .collect::<Vec<_>>()
            } else {
                self.runs
                    .values()
                    .filter(|r| &r.project == project)
                    .cloned()
                    .collect()
            };
            for run in references {
                history.require_available(project, &run.run_id)?;
            }
        }
        let old = history.api().projects(now)?;
        let mut rows = BTreeMap::<String, Value>::new();
        for row in native
            .as_array()
            .ok_or_else(invalid_query)?
            .iter()
            .chain(old.as_array().ok_or_else(invalid_query)?)
        {
            let project = row["project"]
                .as_str()
                .ok_or_else(invalid_query)?
                .to_owned();
            if let Some(current) = rows.get_mut(&project) {
                for (state, count) in row["run_counts"].as_object().ok_or_else(invalid_query)? {
                    let value = current["run_counts"][state]
                        .as_u64()
                        .unwrap_or(0)
                        .checked_add(count.as_u64().ok_or_else(invalid_query)?)
                        .ok_or_else(invalid_query)?;
                    current["run_counts"][state] = Value::from(value);
                }
                current["attention_count"] = Value::from(
                    current["attention_count"]
                        .as_u64()
                        .ok_or_else(invalid_query)?
                        .checked_add(row["attention_count"].as_u64().ok_or_else(invalid_query)?)
                        .ok_or_else(invalid_query)?,
                );
            } else {
                rows.insert(project, row.clone());
            }
        }
        self.validate_registry()?;
        let result = Value::Array(rows.into_values().collect());
        read_model::encoded_size(&result)?;
        Ok(result)
    }
}
