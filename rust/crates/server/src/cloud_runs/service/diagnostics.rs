//! One persisted, read-only CPU diagnostic per original synchronization lineage.
//! A lost submit response only observes that same job. GET never probes a cloud.
use super::*;
use ml_exp_core::cloud::{Job, exchange::Sync, pai::Allocation};

const SCRIPT: &str = include_str!("sync_diagnostic.py");
const PREFIX: &str = "ML_EXPD_DIAGNOSTIC=";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    lineage: Value,
    parent: Job,
    parent_job_id: String,
    job: Job,
    created_at: String,
    report: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    logs_error: Option<Value>,
}
impl Saved {
    fn validate(&self, lineage: &Value, parent: &Job, config: &Value) -> cloud::Result<()> {
        // Reuse the original launcher across upgrades, including a crash before
        // the outbox was prepared. Never reconstruct a different provider job.
        let script = self.job.command.get(2).ok_or_else(cloud::Error::evidence)?;
        if self.lineage != *lineage
            || self.parent != *parent
            || self.job != compiled(parent, lineage, config, script)?
        {
            return Err(cloud::Error::evidence());
        }
        Ok(())
    }
}
fn name(lineage: &Value) -> cloud::Result<String> {
    Ok(format!(
        "diagnostic-{}.json",
        canonical::digest(lineage).map_err(why!(cloud::Error::evidence()))?
    ))
}
fn unavailable(code: &str) -> cloud::Error {
    cloud::Error {
        code: code.into(),
        http_status: Some(409),
        uncertain: false,
    }
}
fn compiled(parent: &Job, lineage: &Value, config: &Value, script: &str) -> cloud::Result<Job> {
    let identity = canonical::digest(&json!([
        "readonly-diagnostic.v1",
        lineage,
        parent.submission_token
    ]))
    .map_err(why!(cloud::Error::evidence()))?;
    let mut job = parent.clone();
    job.run_id = format!("diagnostic-{}", &identity[..32]);
    job.submission_token = identity[..32].into();
    job.gpus = 0;
    job.cpus = 2;
    job.memory_gb = 4;
    job.max_time_seconds = 300;
    job.read_only_storage = true;
    job.command = vec![
        "python3".into(),
        "-c".into(),
        script.into(),
        serde_json::to_string(config).map_err(why!(cloud::Error::evidence()))?,
    ];
    job.validate().map_err(why!(cloud::Error::evidence()))?;
    Ok(job)
}
fn report(value: &Value, config: &Value) -> Option<Value> {
    match value {
        Value::String(text) => text
            .lines()
            .filter_map(|line| {
                let raw = line.split_once(PREFIX)?.1;
                if raw.len() > 32768 {
                    return None;
                }
                let value: Value = serde_json::from_slice(raw.as_bytes()).ok()?;
                if value["contract"] != config["contract"]
                    || value["request_id"] != config["request_id"]
                {
                    return None;
                }
                for key in ["action", "project", "asset_id"] {
                    if value[key] != config[key] {
                        return None;
                    }
                }
                Some(value)
            })
            .next(),
        Value::Array(rows) => rows.iter().find_map(|v| report(v, config)),
        Value::Object(rows) => rows.values().find_map(|v| report(v, config)),
        _ => None,
    }
}
impl Service {
    fn diagnostic_saved(
        &self,
        lineage: &Value,
        parent: &Job,
        config: &Value,
    ) -> cloud::Result<Option<Saved>> {
        let name = name(lineage)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Ok(None);
        }
        let saved: Saved = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(why!(cloud::Error::evidence()))?;
        saved.validate(lineage, parent, config)?;
        Ok(Some(saved))
    }
    #[allow(clippy::too_many_arguments)]
    fn diagnostic(
        &self,
        collector: &Adapter<Shared>,
        parent: &Job,
        lineage: Value,
        config: Value,
        execute: bool,
        now: &str,
        check: &dyn Fn() -> cloud::Result<()>,
    ) -> cloud::Result<Value> {
        if execute && !self.enabled {
            return Err(cloud::Error::configuration());
        }
        let existing = self.diagnostic_saved(&lineage, parent, &config)?;
        let mut saved = if let Some(saved) = existing {
            saved
        } else {
            if !execute {
                return Err(missing());
            }
            let cloud::adapter::Profile::Pai(profile) = &collector.profile else {
                return Err(unavailable("READONLY_DIAGNOSTIC_UNSUPPORTED"));
            };
            if profile.storage.is_none() {
                return Err(unavailable("READONLY_DIAGNOSTIC_STORAGE_REQUIRED"));
            }
            let original = self.sync_outbox.reconcile(
                collector,
                parent,
                Allocation::Guaranteed,
                check,
                &AtomicBool::new(false),
            )?;
            let evidence: ml_exp_core::cloud::JobObservation =
                serde_json::from_value(original["observation"].clone())
                    .map_err(why!(cloud::Error::evidence()))?;
            if !original["diagnostic"].is_null()
                || original["handle"]["job_id"] != evidence.job_id
                || !evidence.state.is_terminal()
            {
                return Err(unavailable("DIAGNOSTIC_PARENT_NOT_TERMINAL"));
            }
            let saved = Saved {
                lineage: lineage.clone(),
                parent: parent.clone(),
                parent_job_id: evidence.job_id,
                job: compiled(parent, &lineage, &config, SCRIPT)?,
                created_at: now.into(),
                report: None,
                logs_error: None,
            };
            check()?;
            // This intent must survive a crash before Outbox::prepare/create.
            write(&self.slots, &name(&lineage)?, &saved)?;
            saved
        };
        if execute {
            let status = self.submit_once_or_reconcile(
                collector,
                &saved.job,
                Allocation::Guaranteed,
                check,
                &AtomicBool::new(false),
            )?;
            if saved.report.is_none() && status["handle"].is_object() {
                let handle = serde_json::from_value(status["handle"].clone())
                    .map_err(why!(cloud::Error::evidence()))?;
                // Delayed offline logs remain pending and can be observed again.
                match collector.pai_diagnostic_logs(&saved.job, &handle, &AtomicBool::new(false)) {
                    Ok(logs) => {
                        saved.report = report(&logs, &config);
                        saved.logs_error = None;
                    }
                    Err(error) => {
                        saved.logs_error =
                            Some(json!({"code":error.code,"http_status":error.http_status}))
                    }
                }
                check()?;
                self.slots
                    .replace_regular(OsStr::new(&name(&lineage)?), &encode(&saved)?)
                    .map_err(io)?;
            }
        }
        let status = self.sync_outbox.optional_status(&saved.job)?;
        Ok(
            json!({"contract":"readonly-diagnostic.v1", "lineage":saved.lineage,
            "parent_job_id":saved.parent_job_id,"created_at":saved.created_at,
            "resources":{"gpus":0,"cpus":2,"memory_gb":4},"max_time_seconds":300,
            "storage_read_only":true,"training_replayed":false,"job":status,"report":saved.report,
            "report_status":if saved.report.is_some(){"AVAILABLE"}else{"AWAITING_OFFLINE_LOGS"},"logs_error":saved.logs_error,
            "remote_refresh_performed":execute}),
        )
    }
    fn provider_read(
        &self,
        collector: &Adapter<Shared>,
        job: &Job,
        lines: u32,
    ) -> cloud::Result<Value> {
        let handle = self
            .sync_outbox
            .optional_status(job)?
            .map(|status| status["handle"].clone())
            .filter(Value::is_object);
        let Some(handle) = handle else {
            return Ok(json!({"diagnostic":"NO_PROVIDER_JOB_RECORDED"}));
        };
        let handle = serde_json::from_value(handle).map_err(why!(cloud::Error::evidence()))?;
        collector.pai_provider_diagnostics(job, &handle, lines, &AtomicBool::new(false))
    }
    /// The provider's account of a data copy and its read-only diagnostic. These
    /// jobs are not Attempts, so `provider_diagnostics` cannot reach their logs.
    pub fn data_provider_diagnostics(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        request: &str,
        lines: u32,
    ) -> cloud::Result<Value> {
        ml_exp_core::submission::validate_submission_token(request)
            .map_err(why!(cloud::Error::configuration()))?;
        let saved = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        if saved.copy_request_id.as_deref() != Some(request) {
            return Err(unavailable("DATA_COPY_IDENTITY_CONFLICT"));
        }
        let copy = saved.copy_intent.ok_or_else(cloud::Error::evidence)?;
        let collector = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let mut result = json!({"project":project,"executor":executor,"asset_id":asset,
            "request_id":request,"provider_read":false});
        if !matches!(collector.profile, cloud::adapter::Profile::Pai(_)) {
            result["diagnostic"] = json!("PROVIDER_DIAGNOSTICS_UNSUPPORTED");
            return Ok(result);
        }
        result["copy"] = self.provider_read(collector, &copy, lines)?;
        let lineage =
            json!({"project":project,"executor":executor,"asset_id":asset,"request_id":request});
        let name = name(&lineage)?;
        if self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            let diagnostic: Saved = serde_json::from_value(read(&self.slots, &name)?)
                .map_err(why!(cloud::Error::evidence()))?;
            if diagnostic.lineage != lineage {
                return Err(cloud::Error::evidence());
            }
            result["read_only_diagnostic"] =
                self.provider_read(collector, &diagnostic.job, lines)?;
        }
        result["provider_read"] = json!(true);
        Ok(result)
    }
    /// Ask the provider why one exact Attempt is in its current state. This
    /// reads and redacts provider evidence; nothing is stored or mutated.
    pub fn provider_diagnostics(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        lines: u32,
    ) -> cloud::Result<Value> {
        self.authority(project, run)?;
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(missing)?;
        let backend = self
            .backends
            .get(&prepared.execution.backend_id)
            .ok_or_else(cloud::Error::configuration)?;
        let mut result = json!({"project":project,"run_id":run,"attempt_id":attempt,
            "backend":prepared.execution.backend_id,"provider_read":false});
        if !matches!(backend.adapter.profile, cloud::adapter::Profile::Pai(_)) {
            result["diagnostic"] = json!("PROVIDER_DIAGNOSTICS_UNSUPPORTED");
            return Ok(result);
        }
        let handle = self
            .outbox
            .optional_status(&prepared.execution.job)?
            .map(|status| status["handle"].clone())
            .filter(Value::is_object);
        let Some(handle) = handle else {
            result["diagnostic"] = json!("NO_PROVIDER_JOB_RECORDED");
            return Ok(result);
        };
        let handle = serde_json::from_value(handle).map_err(why!(cloud::Error::evidence()))?;
        let provider = backend.adapter.pai_provider_diagnostics(
            &prepared.execution.job,
            &handle,
            lines,
            &AtomicBool::new(false),
        )?;
        result["provider_read"] = json!(true);
        result["provider"] = provider;
        Ok(result)
    }
    pub fn sync_diagnostic(
        &self,
        action: &str,
        request: &str,
        execute: bool,
        now: &str,
    ) -> cloud::Result<Value> {
        Sync {
            request_id: request.into(),
            confirmation: format!("SYNC {action}"),
        }
        .validate(action)
        .map_err(why!(cloud::Error::configuration()))?;
        let _lock = self.lock()?;
        let prepared = self.load(action)?;
        let collector = self.backends[&prepared.execution.backend_id]
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let parent = self.sync_recovery_job(&prepared, action, request)?;
        let package = prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        let lineage = json!({"action":action,"request_id":request});
        let config = json!({"contract":"sync-diagnostic.v1","action":action,"request_id":request,
            "channel":package.channel,"package_sha256":helper::sha256_bytes(&encode(&package)?),
            "config_sha256":helper::sha256_bytes(&encode(&package.config)?),
            "config_binding_sha256":package.config.digest().map_err(why!(cloud::Error::evidence()))?,"package":package});
        let guard = self
            .fence
            .existing_observation_guard(&parent.project, &prepared.execution.job.run_id)
            .map_err(why!(cloud::Error::evidence()))?;
        let check = || self.still_live(&guard, &parent.project, &prepared.execution.job.run_id);
        self.diagnostic(collector, &parent, lineage, config, execute, now, &check)
    }
    pub fn data_diagnostic(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        request: &str,
        execute: bool,
        now: &str,
    ) -> cloud::Result<Value> {
        ml_exp_core::submission::validate_submission_token(request)
            .map_err(why!(cloud::Error::configuration()))?;
        let _lock = self.lock()?;
        let saved = self
            .data_saved(project, executor, asset)?
            .ok_or_else(missing)?;
        if saved.copy_request_id.as_deref() != Some(request) {
            return Err(unavailable("DATA_COPY_IDENTITY_CONFLICT"));
        }
        let parent = saved.copy_intent.ok_or_else(cloud::Error::evidence)?;
        let (import, _) = ml_exp_core::cloud::data::materialization_inputs(&parent)
            .map_err(why!(cloud::Error::evidence()))?;
        let lineage =
            json!({"project":project,"executor":executor,"asset_id":asset,"request_id":request});
        let config = json!({"contract":"data-diagnostic.v1","project":project,"asset_id":asset,"request_id":request,
            "root":import.root,"target":import.target,"definition":import.definition});
        let collector = self.backends[executor]
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let check = || {
            self.data_saved(project, executor, asset)?
                .ok_or_else(missing)
                .map(|_| ())
        };
        self.diagnostic(collector, &parent, lineage, config, execute, now, &check)
    }
    pub fn worker_diagnostic_status(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
    ) -> cloud::Result<Value> {
        let _lock = self.lock()?;
        let prepared = self.worker_authority(project, run, attempt, token)?;
        let exchange = self.exchange_saved(&prepared)?;
        let artifacts = self.artifact_saved(&prepared)?;
        Ok(json!({"contract":"sync-diagnostic-status.v1",
            "exchange":{"observed":exchange.is_some(),"sequence":exchange.as_ref().map(|e|e.sequence),
                "complete":exchange.as_ref().map(|e|e.complete),"phase":exchange.as_ref().map(|e|&e.evidence.worker.phase)},
            "artifacts":{"prepared":artifacts.is_some(),"published":artifacts.as_ref().is_some_and(|a|a.state.is_some()),
                "released":artifacts.as_ref().is_some_and(|a|a.released_parts>0 || a.administrative_release.is_some())}}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parent() -> Job {
        serde_json::from_value(
            json!({"project":"demo","run_id":"train","attempt_id":"attempt-001",
            "submission_token":"a".repeat(32),"source_id":format!("source.{}","b".repeat(64)),
            "image":format!("registry.example/team/image@sha256:{}","c".repeat(64)),
            "command":["/worker","cloud-launch","--package-url","https://api.example/package","--sync"],
            "gpus":1,"cpus":8,"memory_gb":64,"max_time_seconds":900,"priority":5}),
        )
        .unwrap()
    }
    #[test]
    fn deterministic_readonly_cpu_job_preserves_image_and_never_runs_worker_or_training() {
        let parent = parent();
        let lineage = json!({"action":"action-0123456789abcdef","request_id":"d".repeat(32)});
        let config = json!({"contract":"sync-diagnostic.v1"});
        let job = compiled(&parent, &lineage, &config, SCRIPT).unwrap();
        assert!(job == compiled(&parent, &lineage, &config, SCRIPT).unwrap());
        assert_eq!(
            (job.gpus, job.cpus, job.memory_gb, job.max_time_seconds),
            (0, 2, 4, 300)
        );
        assert!(job.read_only_storage);
        assert_eq!(job.image, parent.image);
        assert_eq!(job.source_id, parent.source_id);
        assert_eq!(&job.command[..2], &["python3", "-c"]);
        assert!(
            !job.command
                .iter()
                .any(|s| s == "--sync" || s == "cloud-launch")
        );
        assert_ne!(job.submission_token, parent.submission_token);
        let changed = compiled(&parent, &lineage, &config, "future launcher").unwrap();
        assert!(changed != job);
        assert!(
            compiled(&parent, &lineage, &config, &job.command[2]).unwrap() == job,
            "saved intent uses its original launcher after an API upgrade"
        );
    }
    #[test]
    fn persisted_intent_survives_restart_before_outbox_creation_and_keeps_original_launcher() {
        let temp = tempfile::tempdir().unwrap();
        let parent = parent();
        let lineage = json!({"action":"action-0123456789abcdef","request_id":"d".repeat(32)});
        let config = json!({"contract":"sync-diagnostic.v1"});
        let saved = Saved {
            lineage: lineage.clone(),
            parent: parent.clone(),
            parent_job_id: "dlc-original".into(),
            job: compiled(&parent, &lineage, &config, "original launcher").unwrap(),
            created_at: "2026-11-05T00:00:00Z".into(),
            report: None,
            logs_error: None,
        };
        let filename = name(&lineage).unwrap();
        {
            let directory = Directory::open(temp.path(), false).unwrap();
            write(&directory, &filename, &saved).unwrap();
        }
        // Reopen only the retained intent: no outbox or provider was created.
        let directory = Directory::open(temp.path(), false).unwrap();
        let recovered: Saved =
            serde_json::from_value(read(&directory, &filename).unwrap()).unwrap();
        recovered.validate(&lineage, &parent, &config).unwrap();
        assert!(recovered.job == saved.job);
        assert_eq!(recovered.job.command[2], "original launcher");
        assert_eq!(
            Outbox::slot(&recovered.job).unwrap(),
            Outbox::slot(&saved.job).unwrap()
        );
        assert!(recovered.job != compiled(&parent, &lineage, &config, SCRIPT).unwrap());
        assert!(
            recovered
                .validate(&json!({"request_id":"e".repeat(32)}), &parent, &config)
                .is_err()
        );
        let mut changed = parent.clone();
        changed.image = format!("registry.example/team/image@sha256:{}", "f".repeat(64));
        assert!(recovered.validate(&lineage, &changed, &config).is_err());
        assert!(
            recovered
                .validate(&lineage, &parent, &json!({"contract":"changed"}))
                .is_err()
        );
        assert!(
            write(&directory, &filename, &saved).is_err(),
            "intent cannot be replaced"
        );
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
    }
    #[test]
    fn only_bounded_matching_diagnostic_line_is_accepted_from_delayed_nested_logs() {
        let config = json!({"contract":"sync-diagnostic.v1","action":"action-0123456789abcdef","request_id":"a".repeat(32)});
        assert!(report(&json!({"Logs":["no report yet"]}), &config).is_none());
        let text = format!(
            "timestamp {PREFIX}{}",
            serde_json::to_string(&config).unwrap()
        );
        assert_eq!(
            report(&json!({"Logs":[text]}), &config),
            Some(config.clone())
        );
        let mut foreign = config.clone();
        foreign["request_id"] = json!("b".repeat(32));
        assert!(report(&json!(format!("{PREFIX}{foreign}")), &config).is_none());
        assert!(report(&json!(format!("{PREFIX}{}", "x".repeat(32769))), &config).is_none());
    }
}
