//! Historical W&B routes/cursors are observed, never republished or reset.
use super::*;
use serde_json::json;

impl Api {
    pub fn historical_tracking(&self, project: &str, run: &str) -> Result<Option<Value>, Error> {
        if let Some(history) = self.history_for(project, run)? {
            return history.historical_tracking(project, run);
        }
        if !self.is_historical() {
            return Ok(None);
        }
        self.observe(|api| {
            validate_id(project).map_err(|_|invalid_query())?;
            validate_observation_id(run).map_err(|_|invalid_query())?;
            api.validate_registry()?;
            api.ensure_active(project)?;
            let unavailable=||error(409,"TRACKING_EVIDENCE_UNAVAILABLE","historical tracking identity or journal cannot be validated");
            let fence=api.fence.existing_observation_guard(project,run).map_err(|_|unavailable())?;
            let (reference,discovery)=api.locate_run(project,run)?;
            let snapshot=api.store.snapshot_metadata(&reference)?;
            if discovery.as_ref().is_some_and(|d|d.runs.iter().find(|r|r.reference.run_id==run).is_none_or(|r|r.raw_sha256!=snapshot.run.raw_sha256)) {return Err(unavailable());}
            let history=api.historical.as_ref().ok_or_else(unavailable)?;
            history.validate_journal(&api.registry_root).map_err(|_|unavailable())?;
            let mut route=snapshot.run.metadata.get("tracking_route").cloned().unwrap_or(json!({"enabled":false,"reason":"NO_HISTORICAL_JOURNAL"}));
            let mut scopes=Vec::new();
            if history.journal_workspace_id.is_some() {
                let records=crate::records::Store::read_only(&api.registry_root);
                let guard=records.guard(project,run).map_err(|_|unavailable())?;
                if let Some(binding)=records.binding(&guard).map_err(|_|unavailable())? {
                    if !journal_binding_matches(&snapshot,Some(&binding)) {return Err(unavailable());}
                    route=binding["route"].clone();
                    scopes=records.run_scopes(&guard).map_err(|_|unavailable())?;
                    for scope in &mut scopes {
                        let total=scope["total"].as_i64().ok_or_else(unavailable)?;
                        let confirmed=scope["confirmed"].as_i64().filter(|v|*v>=0&&*v<=total).ok_or_else(unavailable)?;
                        scope["pending"]=json!(total-confirmed);
                    }
                }
            }
            let value=experiment_redact::redact_json(json!({"project":project,"run_id":run,"wandb":route,"scopes":scopes,"source":"historical-canonical-journal","read_only":true,"cloud_query_performed":false}));
            read_model::encoded_size(&value)?;
            if let Some(before)=discovery {if !before.same_bindings(&api.store.discover(project,&Default::default())?) {return Err(unavailable());}}
            api.store.snapshot_metadata(&reference).and_then(|current|if current.run.raw_sha256==snapshot.run.raw_sha256 {Ok(current)}else{Err(experiments::Error::Binding)})?;
            fence.require_observed_live(project,run).map_err(|_|unavailable())?;
            api.ensure_active(project)?;
            api.validate_registry()?;
            Ok(Some(value))
        })
    }
}
