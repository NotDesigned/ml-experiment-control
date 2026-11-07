//! One persisted, read-only CPU diagnostic per original synchronization lineage.
//! A lost submit response only observes that same job. GET never probes a cloud.
use super::*;
use ml_exp_core::cloud::{Job, exchange::Sync, pai::Allocation};

const SCRIPT: &str = include_str!("sync_diagnostic.py");
const PART_PREFIX: &str = "ML_EXPD_DIAGNOSTIC_PART=";
const MAX_REPORT_BYTES: usize = 32768;
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
/// The provider's account of one job, read through the adapter and outbox that own
/// it. `None` says no provider job is recorded, so there is nothing to ask about.
fn provider_read(
    adapter: &Adapter<Shared>,
    outbox: &Outbox,
    job: &Job,
    lines: u32,
) -> cloud::Result<Option<Value>> {
    let handle = outbox
        .optional_status(job)?
        .map(|status| status["handle"].clone())
        .filter(Value::is_object);
    let Some(handle) = handle else {
        return Ok(None);
    };
    let handle = serde_json::from_value(handle).map_err(why!(cloud::Error::evidence()))?;
    let cancelled = AtomicBool::new(false);
    match adapter.profile {
        cloud::adapter::Profile::Pai(_) => {
            adapter.pai_provider_diagnostics(job, &handle, lines, &cancelled)
        }
        cloud::adapter::Profile::Sensecore(_) => {
            adapter.sensecore_provider_diagnostics(job, &handle, lines, &cancelled)
        }
    }
    .map(Some)
}
fn job_view(read: Option<Value>) -> Value {
    read.unwrap_or_else(|| json!({"diagnostic":"NO_PROVIDER_JOB_RECORDED"}))
}
/// Every string in a provider's log response, however its pods and Logs arrays nest them.
fn log_texts<'a>(value: &'a Value, texts: &mut Vec<&'a str>) {
    match value {
        Value::String(text) => texts.push(text),
        Value::Array(rows) => {
            for row in rows {
                log_texts(row, texts);
            }
        }
        Value::Object(rows) => {
            for row in rows.values() {
                log_texts(row, texts);
            }
        }
        _ => {}
    }
}
/// One `<sha256>:<i>:<n>:<chunk>` line of a report. A provider timestamp may lead it.
struct Part<'a> {
    digest: &'a str,
    index: usize,
    count: usize,
    chunk: &'a str,
}
fn part(line: &str) -> Option<Part<'_>> {
    let mut fields = line.split_once(PART_PREFIX)?.1.splitn(4, ':');
    let part = Part {
        digest: fields.next()?,
        index: fields.next()?.parse::<usize>().ok()?,
        count: fields.next()?.parse::<usize>().ok()?,
        chunk: fields.next()?,
    };
    (part.index < part.count).then_some(part)
}
/// The report one complete set of parts spells, if it is within its cap, matches its
/// digest and belongs to this exact diagnostic.
fn complete_report(
    digest: &str,
    count: usize,
    chunks: &BTreeMap<usize, &str>,
    config: &Value,
) -> Option<Value> {
    if chunks.len() != count {
        return None;
    }
    let text: String = chunks.values().copied().collect();
    if text.len() > MAX_REPORT_BYTES || helper::sha256_bytes(text.as_bytes())[7..] != *digest {
        return None;
    }
    let value: Value = serde_json::from_str(&text).ok()?;
    if value["contract"] != config["contract"] || value["request_id"] != config["request_id"] {
        return None;
    }
    for key in ["action", "project", "asset_id"] {
        if value[key] != config[key] {
            return None;
        }
    }
    Some(value)
}
/// The first report whose parts have all arrived. Parts may come in any order, from
/// any pod and more than once. An incomplete set is no report yet, so it stays pending.
fn report(value: &Value, config: &Value) -> Option<Value> {
    let mut texts = Vec::new();
    log_texts(value, &mut texts);
    let mut sets: Vec<(&str, usize, BTreeMap<usize, &str>)> = Vec::new();
    for found in texts.into_iter().flat_map(str::lines).filter_map(part) {
        match sets
            .iter_mut()
            .find(|set| set.0 == found.digest && set.1 == found.count)
        {
            Some(set) => {
                set.2.entry(found.index).or_insert(found.chunk);
            }
            None => sets.push((
                found.digest,
                found.count,
                BTreeMap::from([(found.index, found.chunk)]),
            )),
        }
    }
    sets.iter()
        .find_map(|(digest, count, chunks)| complete_report(digest, *count, chunks, config))
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
            let Some(evidence) = super::sync_jobs::proven_terminal(&original)? else {
                return Err(unavailable("DIAGNOSTIC_PARENT_NOT_TERMINAL"));
            };
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
        let backend = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        let mut result = json!({"project":project,"executor":executor,"asset_id":asset,
            "request_id":request,"provider_read":false});
        // SenseCore keeps no copy intent and has no collector, so this comes first.
        if !matches!(backend.adapter.profile, cloud::adapter::Profile::Pai(_)) {
            result["diagnostic"] = json!("PROVIDER_DIAGNOSTICS_UNSUPPORTED");
            return Ok(result);
        }
        let copy = self.copy_job_of_request(saved, project, executor, asset, request)?;
        let collector = backend
            .collector
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        result["copy"] = job_view(provider_read(collector, &self.sync_outbox, &copy, lines)?);
        let lineage =
            json!({"project":project,"executor":executor,"asset_id":asset,"request_id":request});
        let name = name(&lineage)?;
        if self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            let diagnostic: Saved = serde_json::from_value(read(&self.slots, &name)?)
                .map_err(why!(cloud::Error::evidence()))?;
            if diagnostic.lineage != lineage {
                return Err(cloud::Error::evidence());
            }
            result["read_only_diagnostic"] = job_view(provider_read(
                collector,
                &self.sync_outbox,
                &diagnostic.job,
                lines,
            )?);
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
        let read = provider_read(
            &backend.adapter,
            &self.outbox,
            &prepared.execution.job,
            lines,
        )?;
        let Some(provider) = read else {
            result["diagnostic"] = json!("NO_PROVIDER_JOB_RECORDED");
            return Ok(result);
        };
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
        let backend = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        // SenseCore keeps no copy intent and has no collector, so this comes first.
        if !matches!(backend.adapter.profile, cloud::adapter::Profile::Pai(_)) {
            return Err(unavailable("READONLY_DIAGNOSTIC_UNSUPPORTED"));
        }
        let parent = self.copy_job_of_request(saved, project, executor, asset, request)?;
        let (import, _) = ml_exp_core::cloud::data::materialization_inputs(&parent)
            .map_err(why!(cloud::Error::evidence()))?;
        let lineage =
            json!({"project":project,"executor":executor,"asset_id":asset,"request_id":request});
        let config = json!({"contract":"data-diagnostic.v1","project":project,"asset_id":asset,"request_id":request,
            "root":import.root,"target":import.target,"definition":import.definition});
        let collector = backend
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
    /// The lines `sync_diagnostic.py` prints for a report of this text.
    fn lines_of(text: &str) -> Vec<String> {
        let digest = helper::sha256_bytes(text.as_bytes());
        let chunks: Vec<_> = text.as_bytes().chunks(1800).collect();
        chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                format!(
                    "{PART_PREFIX}{}:{index}:{}:{}",
                    &digest[7..],
                    chunks.len(),
                    std::str::from_utf8(chunk).unwrap()
                )
            })
            .collect()
    }
    fn config() -> Value {
        json!({"contract":"sync-diagnostic.v1","action":"action-0123456789abcdef","request_id":"a".repeat(32)})
    }
    fn large(config: &Value) -> Value {
        let mut value = config.clone();
        value["filler"] = json!("x".repeat(30000));
        value
    }
    #[test]
    fn the_line_the_launcher_prints_is_the_line_this_reads() {
        // The same line is pinned in tests/test_sync_diagnostic.py.
        let line = "ML_EXPD_DIAGNOSTIC_PART=265ee923a61584279face2967b07f430b70f20d8d36ab06bc1ccf8d66cde6446:0:1:{\"contract\":\"sync-diagnostic.v1\"}";
        let config = json!({"contract":"sync-diagnostic.v1"});
        assert_eq!(report(&json!(line), &config), Some(config));
    }
    #[test]
    fn a_report_is_read_from_parts_in_any_order_repeated_and_across_pods() {
        let config = config();
        let whole = large(&config);
        let lines = lines_of(&whole.to_string());
        assert!(lines.len() > 10);
        assert!(report(&json!({"Logs":["no report yet"]}), &config).is_none());
        let mut first_pod: Vec<String> = lines
            .iter()
            .rev()
            .map(|line| format!("2026-11-06T00:00:00Z {line}"))
            .collect();
        let second_pod = first_pod.split_off(5);
        let logs = json!([{"Logs":first_pod},{"Logs":second_pod.join("\n")},{"Logs":[lines[0]]}]);
        assert_eq!(report(&logs, &config), Some(whole));
    }
    #[test]
    fn an_incomplete_set_is_no_report_until_its_last_part_arrives() {
        let config = config();
        let whole = large(&config);
        let mut lines = lines_of(&whole.to_string());
        let last = lines.pop().unwrap();
        assert!(report(&json!({"Logs":lines}), &config).is_none());
        lines.push(last);
        assert_eq!(report(&json!({"Logs":lines}), &config), Some(whole));
    }
    #[test]
    fn corrupt_foreign_oversized_or_malformed_parts_are_no_report() {
        let config = config();
        let whole = large(&config);
        let text = whole.to_string();
        let good = lines_of(&text);
        // A changed chunk under the original digest.
        let mut corrupt = good.clone();
        corrupt[1] = corrupt[1].replacen("xxxx", "xxxy", 1);
        assert!(report(&json!(corrupt), &config).is_none());
        // Another diagnostic's complete report, with a valid digest.
        for key in ["request_id", "action", "contract"] {
            let mut foreign = whole.clone();
            foreign[key] = json!("foreign");
            assert!(report(&json!(lines_of(&foreign.to_string())), &config).is_none());
        }
        // Valid parts and digest for a report over the cap.
        let mut over = config.clone();
        over["filler"] = json!("x".repeat(MAX_REPORT_BYTES));
        assert!(report(&json!(lines_of(&over.to_string())), &config).is_none());
        // A foreign set beside the right one does not hide it, and order does not matter.
        let mut foreign = whole.clone();
        foreign["request_id"] = json!("b".repeat(32));
        let mut both = lines_of(&foreign.to_string());
        both.extend(good.clone());
        assert_eq!(report(&json!(both), &config), Some(whole));
        // Lines that are not a part.
        let digest = helper::sha256_bytes(text.as_bytes());
        let digest = &digest[7..];
        for line in [
            format!("{PART_PREFIX}{digest}"),
            format!("{PART_PREFIX}{digest}:0"),
            format!("{PART_PREFIX}{digest}:0:0:{text}"),
            format!("{PART_PREFIX}{digest}:1:1:{text}"),
            format!("{PART_PREFIX}{digest}:x:1:{text}"),
            format!("ML_EXPD_DIAGNOSTIC={text}"),
        ] {
            assert!(report(&json!(line), &config).is_none(), "{line}");
        }
        // A count that disagrees is another set, so it cannot complete this one.
        let mut mixed = good;
        mixed[0] = mixed[0].replacen(&format!(":0:{}:", mixed.len()), ":0:1:", 1);
        assert!(report(&json!(mixed), &config).is_none());
    }
}
