//! Explicit bounded history ingestion. Events commit before the cursor; replay
//! after a crash emits the same event IDs and cannot lose an accepted prefix.
//! GET remains local-only. Cloud publication is a separate effect boundary.
use super::*;
use crate::backends::slurm::{CallOutcome, call};
use ml_exp_core::{
    managed::history::{Cursor, Page},
    records::{ImportCounts, JsonlImporter},
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    request_sha256: String,
    cursor: Option<Cursor>,
    importer: Option<JsonlImporter>,
    counts: ImportCounts,
    complete: bool,
}
impl Service {
    pub fn ingest_history(&self, action: &str, cancelled: &AtomicBool) -> Result<Value> {
        let lock = self.lock()?;
        let prepared = self.load(action)?;
        let guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        let snapshot = self.actions.observe(action)?;
        if snapshot["execution"]["status"] != "VERIFIED" {
            return Err(Error::Conflict);
        }
        let job = snapshot["execution"]["result"]["submission"]["backend_job_id"]
            .as_str()
            .ok_or(Error::Authority)?;
        let (manifest, _) = prepared
            .request
            .prepared_content()
            .ok_or(Error::Authority)?;
        let plan =
            ml_exp_core::slurm::staging::plan(prepared.request.intent(), manifest.as_bytes())
                .map_err(|_| Error::Authority)?
                .ok_or(Error::Authority)?;
        let package = plan.managed.ok_or(Error::Authority)?;
        let name = format!(
            "{}.history.json",
            native::attempt_slot(
                &prepared.binding.project,
                &prepared.binding.run_id,
                &prepared.binding.attempt_id
            )?
        );
        let request_sha256 = prepared
            .request
            .identity()
            .map_err(|_| Error::Authority)?
            .request_sha256;
        // First use is an explicit metadata mutation in this isolated registry.
        // Existing bindings win: history import cannot enable or reroute W&B.
        let records = crate::records::Store::open(
            &self.settings.registry_root,
            self.workspace
                .strip_prefix("rust-preview-")
                .ok_or(Error::Authority)?,
        )
        .map_err(|_| Error::ObservationConflict)?;
        let journal_guard = records
            .guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        if records
            .workspace_id(&journal_guard)
            .map_err(|_| Error::Authority)?
            != self
                .workspace
                .strip_prefix("rust-preview-")
                .ok_or(Error::Authority)?
        {
            return Err(Error::Authority);
        }
        let facts = self.facts_for(
            &prepared.binding.project,
            &prepared.binding.run_id,
            Some(&prepared.binding.attempt_id),
        )?;
        if records
            .binding(&journal_guard)
            .map_err(|_| Error::Authority)?
            .is_none()
        {
            let binding =
                ml_exp_core::records::native_binding(&facts.run, &facts.binding.run_sha256)
                    .map_err(|_| Error::Authority)?;
            records
                .bind(&journal_guard, &binding["route"], &binding["config"])
                .map_err(|_| Error::ObservationConflict)?;
        }
        let snapshot = self
            .evidence
            .snapshot_metadata(&facts.reference)
            .map_err(|_| Error::Authority)?;
        let binding = records
            .binding(&journal_guard)
            .map_err(|_| Error::Authority)?;
        if !crate::read_api::journal_binding_matches(&snapshot, binding.as_ref()) {
            return Err(Error::ObservationConflict);
        }
        let document = Value::Object(
            ml_exp_core::json::mapping(manifest.as_bytes()).map_err(|_| Error::Authority)?,
        );
        let (scope, config) = records
            .prepare_import(&journal_guard, &prepared.binding.attempt_id, &document)
            .map_err(|_| Error::ObservationConflict)?;
        let saved = if self
            .slots
            .entry_exists(OsStr::new(&name))
            .map_err(|_| Error::Authority)?
        {
            let bytes = self
                .slots
                .read_regular(OsStr::new(&name), 2 * 1024 * 1024)
                .map_err(|_| Error::Authority)?;
            let value: Saved = serde_json::from_value(Value::Object(
                ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Authority)?,
            ))
            .map_err(|_| Error::Authority)?;
            if value.request_sha256 != request_sha256
                || value.complete != value.importer.is_none()
                || value.cursor.as_ref().is_none_or(|v| v.validate().is_err())
                || value
                    .importer
                    .as_ref()
                    .is_some_and(|v| !v.matches("metrics.jsonl", &config))
            {
                return Err(Error::Authority);
            }
            value
        } else {
            Saved {
                request_sha256,
                cursor: None,
                importer: Some(
                    JsonlImporter::new("metrics.jsonl", config)
                        .map_err(|_| Error::ObservationConflict)?,
                ),
                counts: ImportCounts::default(),
                complete: false,
            }
        };
        if saved.complete {
            return Ok(history_view(action, &saved));
        }
        let request = helper::Request::metric_history(
            prepared.request.intent().clone(),
            job.into(),
            saved.cursor.clone(),
        )
        .map_err(|_| Error::Authority)?;
        let backend = self
            .backends
            .values()
            .find(|v| v.profile.target == request.intent().target)
            .ok_or(Error::Authority)?;
        let outcome =
            call(&backend.ssh, &request, cancelled).map_err(|_| Error::ObservationUnavailable)?;
        let CallOutcome::Receipt { reply, .. } = outcome else {
            return Err(Error::ObservationUnavailable);
        };
        let helper::ReplyOutcome::MetricHistory { page } = reply.outcome else {
            return Err(Error::ObservationUnavailable);
        };
        let page: Page = *page;
        page.validate(saved.cursor.as_ref())
            .map_err(|_| Error::ObservationConflict)?;
        if page.config_sha256
            != helper::sha256_bytes(&package.config_bytes().map_err(|_| Error::Authority)?)
        {
            return Err(Error::Authority);
        }
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        self.check(&prepared)?;
        lock.validate()?;
        let mut importer = saved.importer.ok_or(Error::Authority)?;
        let mut counts = saved.counts;
        for chunk in page.bytes.chunks(JsonlImporter::INPUT_BOUND) {
            let (next, emission) = importer
                .feed(chunk)
                .map_err(|_| Error::ObservationConflict)?;
            records
                .append_import(&journal_guard, &scope, &emission)
                .map_err(|_| Error::ObservationConflict)?;
            counts = emission.counts;
            importer = next;
        }
        let complete = page.complete();
        let importer = if complete {
            let emission = importer.finish().map_err(|_| Error::ObservationConflict)?;
            records
                .append_import(&journal_guard, &scope, &emission)
                .map_err(|_| Error::ObservationConflict)?;
            counts = emission.counts;
            None
        } else {
            Some(importer)
        };
        let next = Saved {
            request_sha256: saved.request_sha256,
            cursor: Some(page.next),
            importer,
            counts,
            complete,
        };
        let bytes = serde_json::to_vec(&next).map_err(|_| Error::Authority)?;
        if bytes.len() > 2 * 1024 * 1024 {
            return Err(Error::Authority);
        }
        guard
            .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Uncertain)?;
        lock.validate()?;
        self.slots
            .replace_regular(OsStr::new(&name), &bytes)
            .map_err(|_| Error::Uncertain)?;
        Ok(history_view(action, &next))
    }
}
fn history_view(action: &str, saved: &Saved) -> Value {
    json!({"submission_id":action,"status":if saved.complete{"COMPLETE"}else{"PARTIAL"},
        "received_bytes":saved.cursor.as_ref().map(|v|v.offset_bytes).unwrap_or(0),"records":saved.counts.records,
        "invalid_lines":saved.counts.invalid_lines,"scheduler_mutation_performed":false,"source":"native-shared-metric-history"})
}
