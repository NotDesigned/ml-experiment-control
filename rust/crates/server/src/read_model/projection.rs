use super::{Error, MetricQuery, Result, bounded, merge_metrics, metric_payload};
use crate::experiments::{
    AttemptSnapshot, EvidenceDocument, EvidenceProblem, LogTail, ManifestSource, Snapshot,
};
use ml_exp_core::{
    failures::{self, AssessmentContext},
    metrics::python_string,
    observation,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;

/// Already-acquired journal observations from an explicitly guarded Attempt.
/// Receive order is retained; no maximum-step/scientific selection is made.
pub struct LiveMetrics<'a> {
    pub attempt_id: &'a str,
    pub records: &'a [Value],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogStream {
    #[default]
    Both,
    Stdout,
    Stderr,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogQuery {
    #[serde(default)]
    pub stream: LogStream,
    #[serde(default = "default_lines")]
    pub lines: usize,
}
const fn default_lines() -> usize {
    80
}
impl Default for LogQuery {
    fn default() -> Self {
        Self {
            stream: LogStream::Both,
            lines: default_lines(),
        }
    }
}
impl LogQuery {
    pub fn validate(&self) -> Result<()> {
        if !(1..=10000).contains(&self.lines) {
            return Err(Error::Query("INVALID_LINE_COUNT"));
        }
        Ok(())
    }
}

fn truth(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Bool(true) => true,
        Value::Number(v) => v.as_f64().is_none_or(|v| v != 0.0),
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
    }
}
fn object(value: &Value) -> Value {
    if value.is_object() {
        value.clone()
    } else {
        json!({})
    }
}
fn map(document: &EvidenceDocument) -> Value {
    document
        .metadata()
        .cloned()
        .map(Value::Object)
        .unwrap_or_else(|| json!({}))
}
fn first(values: &[Value]) -> Value {
    values
        .iter()
        .find(|v| truth(v))
        .cloned()
        .unwrap_or(Value::Null)
}
fn first_present(values: &[Value]) -> Value {
    values
        .iter()
        .find(|v| !v.is_null())
        .cloned()
        .unwrap_or(Value::Null)
}
fn field(document: &EvidenceDocument, name: &str) -> Value {
    document
        .metadata()
        .and_then(|v| v.get(name))
        .cloned()
        .unwrap_or(Value::Null)
}
fn without_producer_class(value: Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .filter(|(key, _)| key != "failure_class")
                .map(|(key, value)| (key, without_producer_class(value)))
                .collect(),
        ),
        Value::Array(values) => {
            Value::Array(values.into_iter().map(without_producer_class).collect())
        }
        value => value,
    }
}
fn sanitize(value: &Value) -> Value {
    // Keep the shared outward contract and credential policy. The second walk
    // also strips a class smuggled inside an operational-decision allowed value.
    experiment_redact::redact_json(without_producer_class(observation::sanitized_outward(
        value,
    )))
}
pub fn outward_metadata(value: &Value) -> Result<Value> {
    bounded(sanitize(value))
}
fn finish(value: Value) -> Result<Value> {
    bounded(sanitize(&value))
}
fn finish_projection(value: Value) -> Result<Value> {
    bounded(experiment_redact::redact_json(without_producer_class(
        value,
    )))
}
fn clean_record_metadata(record: &mut Value) {
    for key in ["source", "sources", "error", "invalid_value"] {
        if let Some(value) = record.get_mut(key) {
            *value = experiment_redact::redact_json(value.take());
        }
    }
}
fn clean_metric_result(result: &mut Value) {
    if let Some(records) = result.get_mut("records").and_then(Value::as_array_mut) {
        for record in records {
            clean_record_metadata(record);
        }
    }
    if let Some(contexts) = result.get_mut("contexts").and_then(Value::as_array_mut) {
        for context in contexts {
            if let Some(records) = context.get_mut("records").and_then(Value::as_array_mut) {
                for record in records {
                    clean_record_metadata(record);
                }
            }
        }
    }
    if let Some(records) = result
        .get_mut("current")
        .and_then(|c| c.get_mut("records"))
        .and_then(Value::as_array_mut)
    {
        for record in records {
            clean_record_metadata(record);
        }
    }
}
fn clock(now: f64) -> Result<()> {
    if now.is_finite() {
        Ok(())
    } else {
        Err(Error::Query("INVALID_OBSERVATION_TIME"))
    }
}

fn checked(snapshot: &Snapshot) -> Result<()> {
    crate::experiments::validate_run(&snapshot.run.metadata, &snapshot.project, &snapshot.run_id)
        .map_err(|_| Error::Identity)?;
    let mut attempts = BTreeSet::new();
    for attempt in &snapshot.attempts {
        if !attempts.insert(attempt.attempt_id.as_str()) {
            return Err(Error::Identity);
        }
        crate::experiments::validate_attempt(
            &snapshot.run,
            &attempt.manifest,
            &snapshot.project,
            &snapshot.run_id,
            &attempt.attempt_id,
        )
        .map_err(|_| Error::Identity)?;
        for document in [
            &attempt.status,
            &attempt.backend,
            &attempt.collection,
            &attempt.decision,
            &attempt.submission,
        ] {
            if let Some(metadata) = document.metadata() {
                for (key, expected) in [
                    ("project", snapshot.project.as_str()),
                    ("run_id", snapshot.run_id.as_str()),
                    ("attempt_id", attempt.attempt_id.as_str()),
                ] {
                    if metadata
                        .get(key)
                        .is_some_and(|v| !v.is_null() && v.as_str() != Some(expected))
                    {
                        return Err(Error::Identity);
                    }
                }
            }
        }
    }
    if snapshot
        .current_attempt_id
        .as_ref()
        .is_some_and(|id| !attempts.contains(id.as_str()))
    {
        return Err(Error::Identity);
    }
    for document in [
        &snapshot.current_status,
        &snapshot.current_collection,
        &snapshot.current_decision,
    ] {
        if let Some(metadata) = document.metadata() {
            for (key, expected) in [
                ("project", snapshot.project.as_str()),
                ("run_id", snapshot.run_id.as_str()),
            ] {
                if metadata
                    .get(key)
                    .is_some_and(|v| !v.is_null() && v.as_str() != Some(expected))
                {
                    return Err(Error::Identity);
                }
            }
        }
    }
    if let Some(id) = snapshot.current_attempt_id.as_deref() {
        if ![
            &snapshot.current_status,
            &snapshot.current_collection,
            &snapshot.current_decision,
        ]
        .iter()
        .any(|doc| field(doc, "attempt_id").as_str() == Some(id))
        {
            return Err(Error::Identity);
        }
    }
    Ok(())
}
fn legacy_process_record(attempt: &AttemptSnapshot) -> bool {
    attempt.manifest.source == ManifestSource::AttemptJson
        && !attempt.manifest.metadata.contains_key("native_creation")
}
fn effective_collection(snapshot: &Snapshot, attempt: &AttemptSnapshot) -> Value {
    let document = collection(snapshot, attempt);
    let value = map(document);
    if !value.as_object().is_some_and(Map::is_empty)
        || matches!(document, EvidenceDocument::Unavailable { .. })
        || !legacy_process_record(attempt)
    {
        return value;
    }
    let record = Value::Object(attempt.manifest.metadata.clone());
    json!({"attempt_id":attempt.attempt_id,"worker_state":"LOCAL_GPU","process_state":record["state"],
        "failure_class":record["failure_class"],"observed_at":first_present(&[record["finished_at"].clone(),record["observed_at"].clone(),record["updated_at"].clone()]),
        "process_evidence":{"return_code":record["return_code"],"launch_error":record["launch_error"]}})
}
fn collection_provenance(snapshot: &Snapshot, attempt: &AttemptSnapshot) -> Value {
    let document = collection(snapshot, attempt);
    if !matches!(document, EvidenceDocument::Unavailable { .. })
        && document.metadata().is_none_or(Map::is_empty)
        && legacy_process_record(attempt)
    {
        json!({"availability":"PRESENT","source":format!("attempts/{}/attempt.json",attempt.attempt_id),
            "raw_sha256":attempt.manifest.raw_sha256,"kind":"legacy_process_record"})
    } else {
        evidence_view(document)
    }
}
fn exact<'a>(snapshot: &'a Snapshot, id: &str) -> Result<&'a AttemptSnapshot> {
    checked(snapshot)?;
    snapshot
        .attempts
        .iter()
        .find(|a| a.attempt_id == id)
        .ok_or(Error::UnknownAttempt)
}
fn current(snapshot: &Snapshot) -> Result<Option<&AttemptSnapshot>> {
    checked(snapshot)?;
    snapshot
        .current_attempt_id
        .as_deref()
        .map(|id| exact(snapshot, id))
        .transpose()
}
fn evidence_view(document: &EvidenceDocument) -> Value {
    match document {
        EvidenceDocument::Missing => json!({"availability":"MISSING"}),
        EvidenceDocument::Unavailable { source, code } => {
            json!({"availability":"UNAVAILABLE","source":source,"diagnostic":code})
        }
        EvidenceDocument::Present {
            source,
            raw_sha256,
            modified_at_unix,
            ..
        } => json!({"availability":"PRESENT",
            "source":source,"raw_sha256":raw_sha256,"file_modified_at_unix":modified_at_unix}),
    }
}
fn source(document: &EvidenceDocument) -> Value {
    match document {
        EvidenceDocument::Missing => Value::Null,
        EvidenceDocument::Present { source, .. } | EvidenceDocument::Unavailable { source, .. } => {
            json!(source)
        }
    }
}
fn observed(document: &EvidenceDocument, names: &[&str]) -> Value {
    names
        .iter()
        .find_map(|name| {
            document
                .metadata()
                .and_then(|m| m.get(*name))
                .filter(|v| !v.is_null())
        })
        .cloned()
        .unwrap_or(Value::Null)
}
fn state(attempt: &AttemptSnapshot) -> Value {
    let status = field(&attempt.status, "state");
    if truth(&status) {
        status
    } else if attempt.manifest.source == ManifestSource::AttemptJson {
        attempt
            .manifest
            .metadata
            .get("state")
            .cloned()
            .unwrap_or(Value::Null)
    } else {
        Value::Null
    }
}
fn decision<'a>(snapshot: &'a Snapshot, attempt: &'a AttemptSnapshot) -> &'a EvidenceDocument {
    if !matches!(attempt.decision, EvidenceDocument::Missing) {
        return &attempt.decision;
    }
    if snapshot.current_attempt_id.as_deref() == Some(&attempt.attempt_id)
        && field(&snapshot.current_decision, "attempt_id").as_str() == Some(&attempt.attempt_id)
    {
        &snapshot.current_decision
    } else {
        &attempt.decision
    }
}
fn collection<'a>(snapshot: &'a Snapshot, attempt: &'a AttemptSnapshot) -> &'a EvidenceDocument {
    if !matches!(attempt.collection, EvidenceDocument::Missing) {
        return &attempt.collection;
    }
    if snapshot.current_attempt_id.as_deref() == Some(&attempt.attempt_id)
        && field(&snapshot.current_collection, "attempt_id").as_str() == Some(&attempt.attempt_id)
    {
        &snapshot.current_collection
    } else {
        &attempt.collection
    }
}
fn scheduler<'a>(snapshot: &'a Snapshot, attempt: &'a AttemptSnapshot) -> &'a EvidenceDocument {
    if !matches!(attempt.status, EvidenceDocument::Missing) {
        return &attempt.status;
    }
    if snapshot.current_attempt_id.as_deref() == Some(&attempt.attempt_id)
        && field(&snapshot.current_status, "attempt_id").as_str() == Some(&attempt.attempt_id)
    {
        &snapshot.current_status
    } else {
        &attempt.status
    }
}
fn summary(snapshot: &Snapshot, attempt: &AttemptSnapshot) -> Value {
    let status = scheduler(snapshot, attempt);
    let state = first(&[state(attempt), field(status, "state")]);
    let backend = first(&[
        field(&attempt.backend, "backend"),
        field(status, "backend"),
        if attempt
            .submission
            .metadata()
            .is_some_and(|m| m.get("gpu").is_some_and(|v| !v.is_null()))
        {
            json!("local-cuda")
        } else {
            Value::Null
        },
    ]);
    json!({"attempt_id":attempt.attempt_id,"state":state,"backend":backend,
        "backend_job_id":first(&[field(&attempt.backend,"backend_job_id"),field(status,"backend_job_id")]),
        "decision":observation::operational_decision(&map(decision(snapshot,attempt))),
        "has_submission":match attempt.submission {EvidenceDocument::Present{..}=>json!(true),EvidenceDocument::Missing=>json!(false),EvidenceDocument::Unavailable{..}=>Value::Null},
        "submission_evidence":evidence_view(&attempt.submission)})
}

pub fn run_attempts(snapshot: &Snapshot) -> Result<Value> {
    checked(snapshot)?;
    let attempts = snapshot.attempts.iter().map(|attempt| {
        let view = summary(snapshot, attempt);
        json!({"identity":format!("{}::{}",snapshot.run_id,attempt.attempt_id),"attempt_id":attempt.attempt_id,
            "current":snapshot.current_attempt_id.as_deref()==Some(&attempt.attempt_id),
            "backend":view["backend"],"backend_job_id":view["backend_job_id"],"state":view["state"],
            "decision":view["decision"]["action"],"has_submission":view["has_submission"]})
    }).collect::<Vec<_>>();
    finish_projection(
        json!({"project":snapshot.project,"run_id":snapshot.run_id,"current_attempt_id":snapshot.current_attempt_id,
        "current_selection_source":snapshot.current_selection_source,"count":attempts.len(),"attempts":attempts,
        "diagnostics":snapshot.diagnostics}),
    )
}

struct LayerScope<'a> {
    attempt_id: &'a str,
    scheduler_state: &'a Value,
    now: f64,
}
fn layer(
    state: Value,
    observed_at: Value,
    source: Value,
    detail: Value,
    name: &str,
    scope: &LayerScope<'_>,
) -> Value {
    let LayerScope {
        attempt_id,
        scheduler_state,
        now,
    } = scope;
    let timestamp = failures::evidence_timestamp(&observed_at, None);
    let active = matches!(
        scheduler_state.as_str().map(str::to_uppercase).as_deref(),
        Some("SUBMITTING" | "QUEUED" | "RUNNING" | "EVALUATING" | "STARTING")
    );
    let stale = timestamp.map(|ts| {
        active && (*now - ts).max(0.0) > if name == "scheduler" { 3600.0 } else { 1800.0 }
    });
    json!({"state":state,"attempt_id":attempt_id,"as_of":timestamp,"reported_at":observed_at,
        "source":source,"detail":detail,"stale":stale,"freshness":if timestamp.is_none(){"UNKNOWN"}else if stale==Some(true){"STALE"}else{"OBSERVED"},
        "stale_reason":if stale==Some(true){json!(format!("{name} evidence exceeds its observation age threshold"))}else{Value::Null}})
}
fn empty_layer() -> Value {
    json!({"state":null,"attempt_id":null,"as_of":null,"reported_at":null,"source":null,"detail":{},
        "stale":null,"freshness":"UNKNOWN","stale_reason":null})
}
fn take(fields: &Map<String, Value>, keys: &[&str]) -> Value {
    Value::Object(
        keys.iter()
            .filter_map(|key| {
                fields
                    .get(*key)
                    .filter(|v| !v.is_null())
                    .map(|v| ((*key).into(), v.clone()))
            })
            .collect(),
    )
}
fn metric_last(
    attempt: &AttemptSnapshot,
    live: Option<LiveMetrics<'_>>,
) -> Result<Option<(Value, Value, bool)>> {
    if let Some(live) = live {
        if live.attempt_id != attempt.attempt_id {
            return Err(Error::Identity);
        }
        if let Some(record) = live.records.last() {
            return Ok(Some((
                record.clone(),
                json!("canonical_record_journal"),
                true,
            )));
        }
    }
    if attempt
        .metrics
        .source_attempt_id
        .as_deref()
        .is_some_and(|id| id != attempt.attempt_id)
    {
        return Err(Error::Identity);
    }
    Ok(attempt
        .metrics
        .records
        .last()
        .map(|record| (record.clone(), json!(attempt.metrics.source), false)))
}
fn model_detail(record: &Value, normalized: bool, contract: &Value) -> Result<Value> {
    let normalized_record;
    let empty = json!({});
    let record = if normalized {
        record
    } else if record.get("name").is_some() {
        normalized_record = ml_exp_core::metrics::observations(
            record,
            contract.get("metrics_schema").unwrap_or(&empty),
            contract["protocol_id"].as_str(),
            None,
        )?
        .into_iter()
        .next()
        .ok_or(Error::Evidence)?;
        &normalized_record
    } else {
        record
    };
    let fields = record.as_object().ok_or(Error::Evidence)?;
    let mut detail = Map::new();
    for (key, value) in fields {
        if key == "timestamp" {
            continue;
        }
        if value.is_number() && ml_exp_core::metrics::finite(value)
            || [
                "name",
                "unit",
                "value",
                "status",
                "step",
                "epoch",
                "protocol_id",
                "checkpoint_id",
                "dataset_id",
                "variant_id",
                "numerator",
                "denominator",
                "summary",
            ]
            .contains(&key.as_str())
                && (value.is_null() || value.is_boolean() || value.is_string() || value.is_number())
        {
            detail.insert(key.clone(), value.clone());
        }
    }
    let mut detail = Value::Object(detail);
    clean_record_metadata(&mut detail);
    Ok(detail)
}
fn layers(
    snapshot: &Snapshot,
    attempt: &AttemptSnapshot,
    now: f64,
    live: Option<LiveMetrics<'_>>,
) -> Result<Value> {
    if let Some(live) = live.as_ref() {
        let contract = snapshot
            .run
            .metadata
            .get("evaluation")
            .filter(|value| value.is_object())
            .cloned()
            .unwrap_or_else(|| json!({}));
        super::validate_journal_observations(live.records, &contract)?;
    }
    let status_doc = scheduler(snapshot, attempt);
    let status = map(status_doc);
    let scheduler_state = status["state"].clone();
    let scope = LayerScope {
        attempt_id: &attempt.attempt_id,
        scheduler_state: &scheduler_state,
        now,
    };
    let mut detail = take(
        status.as_object().ok_or(Error::Evidence)?,
        &[
            "raw_state",
            "backend",
            "backend_job_id",
            "partition",
            "elapsed",
            "exit_code",
            "reason",
            "pending_reason",
        ],
    );
    if status["detail"]["queue"].is_object() {
        detail["queue"] = status["detail"]["queue"].clone();
    }
    for key in ["backend", "backend_job_id"] {
        if detail[key].is_null() {
            detail[key] = field(&attempt.backend, key);
        }
    }
    detail["backend_evidence"] = evidence_view(&attempt.backend);
    let mut scheduler_layer = layer(
        scheduler_state.clone(),
        observed(status_doc, &["observed_at", "updated_at", "finished_at"]),
        source(status_doc),
        detail,
        "scheduler",
        &scope,
    );
    let collection_doc = collection(snapshot, attempt);
    let collected = effective_collection(snapshot, attempt);
    let collection_time = first_present(&[
        collected["observed_at"].clone(),
        collected["collected_at"].clone(),
    ]);
    let worker_state = collected["worker_state"].clone();
    let process_state = collected["process_state"].clone();
    scheduler_layer["evidence_availability"] = evidence_view(status_doc);
    let mut worker = layer(
        worker_state,
        first_present(&[
            collected["worker_observed_at"].clone(),
            collection_time.clone(),
        ]),
        collection_provenance(snapshot, attempt)["source"].clone(),
        json!({}),
        "worker",
        &scope,
    );
    worker["evidence_availability"] = collection_provenance(snapshot, attempt);
    let mut process = layer(
        process_state,
        first_present(&[
            collected["process_evidence"]["observed_at"].clone(),
            collected["process_observed_at"].clone(),
            collection_time,
        ]),
        collection_provenance(snapshot, attempt)["source"].clone(),
        take(
            collected["process_evidence"]
                .as_object()
                .unwrap_or(&Map::new()),
            &[
                "exit_code",
                "return_code",
                "observed",
                "pid",
                "process_running",
            ],
        ),
        "process",
        &scope,
    );
    process["evidence_availability"] = collection_provenance(snapshot, attempt);
    let model = if let Some((record, source, normalized)) = metric_last(attempt, live)? {
        let detail = model_detail(
            &record,
            normalized,
            snapshot
                .run
                .metadata
                .get("evaluation")
                .unwrap_or(&json!({})),
        )?;
        let state = record
            .get("step")
            .filter(|v| !v.is_null())
            .map(|v| json!(format!("step {}", python_string(v))))
            .unwrap_or_else(|| json!("present"));
        let mut value = layer(
            state,
            record.get("timestamp").cloned().unwrap_or(Value::Null),
            source,
            detail,
            "model",
            &scope,
        );
        value["not_a_heartbeat"] = json!(true);
        value["history_truncated"] = json!(attempt.metrics.truncated);
        value["history_diagnostics"] = json!(attempt.metrics.diagnostics);
        value
    } else {
        layer(
            collected["model_state"].clone(),
            observed(collection_doc, &["model_observed_at"]),
            source(collection_doc),
            json!({}),
            "model",
            &scope,
        )
    };
    Ok(
        json!({"scheduler":scheduler_layer,"worker":worker,"process":process,"model":model,"evaluation":empty_layer()}),
    )
}
fn assessment(snapshot: &Snapshot, attempt: &AttemptSnapshot) -> Result<Value> {
    let collected = effective_collection(snapshot, attempt);
    let decision_doc = decision(snapshot, attempt);
    let decision = map(decision_doc);
    let status_doc = scheduler(snapshot, attempt);
    let mut context = AssessmentContext::new(&attempt.attempt_id);
    context.attempt_state = first(&[state(attempt), field(status_doc, "state")])
        .as_str()
        .map(str::to_owned);
    context.attempt_started_at = first_present(&[
        field(status_doc, "started_at"),
        attempt
            .manifest
            .metadata
            .get("started_at")
            .cloned()
            .unwrap_or(Value::Null),
    ]);
    context.observed_at = first_present(&[
        collected["observed_at"].clone(),
        collected["collected_at"].clone(),
    ]);
    for domain in ["process", "worker", "scheduler"] {
        let time = if domain == "process" {
            first_present(&[
                collected["process_evidence"]["observed_at"].clone(),
                collected["process_observed_at"].clone(),
                context.observed_at.clone(),
            ])
        } else {
            first_present(&[
                collected[format!("{domain}_observed_at")].clone(),
                context.observed_at.clone(),
            ])
        };
        context.domain_observed_at.insert(domain.into(), time);
        context
            .domain_attempt_ids
            .insert(domain.into(), collected["attempt_id"].clone());
        context.domain_evidence_sources.insert(
            domain.into(),
            collection_provenance(snapshot, attempt)["source"].clone(),
        );
    }
    if field(status_doc, "attempt_id").as_str() == Some(&attempt.attempt_id) {
        context.domain_observed_at.insert(
            "scheduler".into(),
            observed(status_doc, &["observed_at", "updated_at", "finished_at"]),
        );
        context
            .domain_attempt_ids
            .insert("scheduler".into(), json!(attempt.attempt_id));
        context
            .domain_evidence_sources
            .insert("scheduler".into(), source(status_doc));
    }
    context.decision_observed_at = observed(decision_doc, &["observed_at", "created_at"]);
    context.decision_source_binding = if decision["attempt_id"] == attempt.attempt_id {
        "EXACT_ATTEMPT"
    } else if decision["attempt_id"].is_null() {
        "EXACT_ATTEMPT_PATH_UNSCOPED"
    } else {
        "ATTEMPT_MISMATCH"
    }
    .into();
    context.decision_evidence_source = source(decision_doc).as_str().map(str::to_owned);
    // Classification is computed before outward removal, never copied from a
    // producer field into the authoritative response.
    Ok(failures::attempt_failure_evidence_assessment(
        &collected,
        Some(&decision),
        &context,
    )?)
}

pub fn run_detail(
    snapshot: &Snapshot,
    now_unix: f64,
    live: Option<LiveMetrics<'_>>,
) -> Result<Value> {
    clock(now_unix)?;
    let current = current(snapshot)?;
    if live
        .as_ref()
        .is_some_and(|live| snapshot.current_attempt_id.as_deref() != Some(live.attempt_id))
    {
        return Err(Error::Identity);
    }
    let evidence = if let Some(attempt) = current {
        layers(snapshot, attempt, now_unix, live)?
    } else {
        json!({"scheduler":empty_layer(),"worker":empty_layer(),"process":empty_layer(),"model":empty_layer(),"evaluation":empty_layer()})
    };
    let state = evidence["scheduler"]["state"].clone();
    let failure_assessment = if let Some(attempt) = current {
        assessment(snapshot, attempt)?
    } else {
        json!({"failure_summary":null,"diagnostic_evidence":[]})
    };
    let listing = run_attempts(snapshot)?;
    let mut value = json!({"project":snapshot.project,"run_id":snapshot.run_id,"current_attempt_id":snapshot.current_attempt_id,
        "current_selection_source":snapshot.current_selection_source,"scheduler_state":state,
        "is_terminal":matches!(state.as_str().map(str::to_uppercase).as_deref(),Some("SUCCEEDED"|"FAILED"|"CANCELLED"|"PREEMPTED")),
        "evidence":evidence,"attempts":listing["attempts"],"manifest":snapshot.run,
        "decision":current.map(|a|observation::operational_decision(&map(decision(snapshot,a)))).unwrap_or_else(||json!({})),
        "provenance":{"manifest_raw_sha256":snapshot.run.raw_sha256,"manifest_source":snapshot.run.source},
        "latest_metrics":{},"eval_metrics":{},"diagnostics":snapshot.diagnostics,"scanned_at":now_unix});
    for key in ["campaign", "role"] {
        value[key] = snapshot
            .run
            .metadata
            .get(key)
            .cloned()
            .unwrap_or(Value::Null);
    }
    // Only daemon assessment may carry a failure_class after sanitizing all
    // producer-controlled metadata. Never re-sanitize and lose this distinction.
    let mut clean = sanitize(&value);
    clean["manifest"]["metadata_redacted"] = json!(true);
    clean["attempts"] = listing["attempts"].clone();
    clean["evidence"]["model"]["detail"] = value["evidence"]["model"]["detail"].clone();
    clean["failure_assessment"] = experiment_redact::redact_json(failure_assessment);
    bounded(clean)
}

pub fn attempt_show(snapshot: &Snapshot, attempt_id: &str, now_unix: f64) -> Result<Value> {
    clock(now_unix)?;
    let attempt = exact(snapshot, attempt_id)?;
    let evidence = layers(snapshot, attempt, now_unix, None)?;
    let detail = evidence["model"]["detail"].clone();
    let mut value = sanitize(
        &json!({"scope":{"project":snapshot.project,"scope_type":"attempt","object_id":format!("{}::{attempt_id}",snapshot.run_id)},
        "run_id":snapshot.run_id,"attempt":summary(snapshot,attempt),"run_dir":".","attempt_dir":format!("attempts/{attempt_id}"),
        "manifest":attempt.manifest,"provenance":attempt.provenance,
        "collection":failures::compact_evidence(&effective_collection(snapshot,attempt)),
        "evidence":evidence,"diagnostics":snapshot.diagnostics,
        "evidence_availability":{"status":evidence_view(&attempt.status),"backend":evidence_view(&attempt.backend),
            "collection":evidence_view(collection(snapshot,attempt)),"decision":evidence_view(decision(snapshot,attempt)),"submission":evidence_view(&attempt.submission)}}),
    );
    value["evidence"]["model"]["detail"] = detail;
    value["manifest"]["metadata_redacted"] = json!(true);
    value["failure_assessment"] = experiment_redact::redact_json(assessment(snapshot, attempt)?);
    bounded(value)
}

fn loaded(diagnostics: &[crate::experiments::Diagnostic]) -> Result<()> {
    if diagnostics
        .iter()
        .any(|d| d.code == EvidenceProblem::NotLoaded)
    {
        Err(Error::Evidence)
    } else {
        Ok(())
    }
}
fn stream_view(tail: &LogTail, process: &Value, name: &str, lines: usize) -> Result<Value> {
    loaded(&tail.diagnostics)?;
    let local = tail.source.is_some();
    let archived = process
        .get(format!("{name}_tail"))
        .and_then(Value::as_array);
    let selected = if local {
        tail.lines
            .iter()
            .rev()
            .take(lines)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
    } else {
        archived
            .map(|items| {
                items
                    .iter()
                    .rev()
                    .take(lines)
                    .rev()
                    .map(python_string)
                    .collect()
            })
            .unwrap_or_default()
    };
    let cleaned = selected
        .iter()
        .map(|line| experiment_redact::redact_lines(line))
        .collect::<Vec<_>>();
    Ok(
        json!({"source":if local{json!(tail.source)}else{process["sources"][name].clone()},
        "mode":if local{"local_file"}else{"collected_tail"},"lines":cleaned,
        "available_lines":if local{tail.available_lines}else{archived.map_or(0,Vec::len)},"truncated":tail.truncated,
        "offset_bytes":tail.offset_bytes,"sampled_bytes":tail.sampled_bytes,
        "diagnostics":tail.diagnostics,"file_modified_at_unix":tail.modified_at_unix}),
    )
}
pub fn attempt_logs(snapshot: &Snapshot, attempt_id: &str, query: &LogQuery) -> Result<Value> {
    query.validate()?;
    let attempt = exact(snapshot, attempt_id)?;
    let collected = effective_collection(snapshot, attempt);
    let mut streams = Map::new();
    for (name, tail) in [("stdout", &attempt.stdout), ("stderr", &attempt.stderr)] {
        if query.stream == LogStream::Both
            || query.stream
                == if name == "stdout" {
                    LogStream::Stdout
                } else {
                    LogStream::Stderr
                }
        {
            streams.insert(
                name.into(),
                stream_view(tail, &collected["process_evidence"], name, query.lines)?,
            );
        }
    }
    finish(
        json!({"project":snapshot.project,"run_id":snapshot.run_id,"attempt_id":attempt_id,"streams":streams,
        "follow_supported":false,"read_mode":"bounded_exact_attempt","diagnostics":snapshot.diagnostics}),
    )
}

pub fn attempt_metrics(
    snapshot: &Snapshot,
    attempt_id: &str,
    query: &MetricQuery,
    live: Option<LiveMetrics<'_>>,
) -> Result<Value> {
    query.validate()?;
    let attempt = exact(snapshot, attempt_id)?;
    loaded(&attempt.metrics.diagnostics)?;
    if attempt
        .metrics
        .source_attempt_id
        .as_deref()
        .is_some_and(|id| id != attempt_id)
        || live.as_ref().is_some_and(|v| v.attempt_id != attempt_id)
    {
        return Err(Error::Identity);
    }
    let contract = snapshot
        .run
        .metadata
        .get("evaluation")
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let live = live.map(|v| v.records).unwrap_or_default();
    let merged = if !live.is_empty() {
        merge_metrics(
            &attempt.metrics.records,
            live,
            &contract,
            attempt.metrics.source.as_deref(),
        )?
    } else {
        Vec::new()
    };
    let records = if live.is_empty() {
        attempt.metrics.records.as_slice()
    } else {
        merged.as_slice()
    };
    let mut value = metric_payload(
        records,
        query,
        &contract,
        if live.is_empty() {
            attempt.metrics.source.as_deref()
        } else {
            Some("canonical_record_journal")
        },
        Some(attempt_id),
        !live.is_empty(),
    )?;
    value["project"] = json!(snapshot.project);
    value["run_id"] = json!(snapshot.run_id);
    value["attempt_id"] = json!(attempt_id);
    let incomplete = attempt.metrics.truncated || !attempt.metrics.diagnostics.is_empty();
    if incomplete {
        value["validation"]["coverage"] = json!("bounded_partial_history");
        if value["validation"]["state"] != "FAILED" {
            value["validation"]["state"] = json!("PARTIAL");
        }
    }
    value["validation"]["complete_history"] = json!(!incomplete);
    value["truncated"] = json!(attempt.metrics.truncated);
    value["diagnostics"] = json!(attempt.metrics.diagnostics);
    if let Some(records) = value.get_mut("records").and_then(Value::as_array_mut) {
        for record in records {
            clean_record_metadata(record);
        }
    }
    if let Some(result) = value.get_mut("metrics") {
        clean_metric_result(result);
    }
    if let Some(source) = value.get_mut("source") {
        *source = experiment_redact::redact_json(source.take());
    }
    // Arbitrary scientific names, units, contexts and values remain intact.
    // Raw manifests use the full metadata policy; this normalized metric view
    // only redacts explicit source/error metadata, never schema or metric data.
    bounded(value)
}

/// Process and native artifact evidence remain independent. Only a validated
/// exact bundle registry may establish publication or verified downloadability.
pub fn attempt_collection(snapshot: &Snapshot, attempt_id: &str, now_unix: f64) -> Result<Value> {
    clock(now_unix)?;
    let attempt = exact(snapshot, attempt_id)?;
    let collected_doc = collection(snapshot, attempt);
    let collected = effective_collection(snapshot, attempt);
    let process = object(&collected["process_evidence"]);
    let process_state = collected["process_state"].clone();
    let exit_code = first_present(&[
        process["exit_code"].clone(),
        process["return_code"].clone(),
        if attempt.manifest.source == ManifestSource::AttemptJson {
            attempt
                .manifest
                .metadata
                .get("return_code")
                .cloned()
                .unwrap_or(Value::Null)
        } else {
            Value::Null
        },
    ]);
    let exit_code = exit_code.as_i64().filter(|v| (-255..=255).contains(v));
    let terminal = matches!(
        process_state.as_str().map(str::to_uppercase).as_deref(),
        Some("SUCCEEDED" | "COMPLETED" | "FAILED" | "PREEMPTED" | "CANCELLED")
    );
    let normalized_state = process_state.as_str().map(str::to_uppercase);
    let conflicting = match (normalized_state.as_deref(), exit_code) {
        (Some("SUCCEEDED"), Some(code)) => code != 0,
        (Some("FAILED" | "PREEMPTED" | "CANCELLED"), Some(0)) => true,
        (Some("RUNNING"), Some(_)) => true,
        _ => false,
    };
    let managed_training =
        if process["managed_contract"] == ml_exp_core::managed::evidence::CONTRACT {
            serde_json::from_value::<ml_exp_core::managed::evidence::Training>(
                process["managed_training"].clone(),
            )
            .ok()
        } else {
            None
        };
    let training = if let Some(managed) = managed_training {
        match managed {
            ml_exp_core::managed::evidence::Training::Completed
                if exit_code == Some(0) && !conflicting =>
            {
                "COMPLETED"
            }
            ml_exp_core::managed::evidence::Training::Failed => "FAILED",
            ml_exp_core::managed::evidence::Training::TimedOut => "TIMED_OUT",
            ml_exp_core::managed::evidence::Training::Cancelled => "CANCELLED",
            _ => "UNKNOWN",
        }
    } else if conflicting {
        "UNKNOWN"
    } else if terminal && exit_code.is_some() {
        if exit_code == Some(0) {
            "COMPLETED"
        } else {
            "FAILED"
        }
    } else if process_state == "RUNNING" {
        "RUNNING"
    } else {
        "UNKNOWN"
    };
    let evidence = layers(snapshot, attempt, now_unix, None)?;
    let detail = evidence["model"]["detail"].clone();
    let mut value = sanitize(
        &json!({"contract":"result-collection-read.v1","project":snapshot.project,"run_id":snapshot.run_id,"attempt_id":attempt_id,
        "status":"NOT_OBSERVED","phase":"NOT_OBSERVED","training":{"status":training,"exit_code":exit_code,
            "process_state":process_state,"diagnostic":if conflicting{json!("PROCESS_STATE_EXIT_CODE_CONFLICT")}else{Value::Null},
            "evidence":if collected_doc.metadata().is_some(){json!("exact_attempt_process_evidence")}else{Value::Null}},
        "verification":"PENDING","download_available":false,"publication":null,
        "diagnostic":if snapshot.run.metadata.contains_key("native_creation"){"ARTIFACT_PUBLICATION_NOT_OBSERVED"}else{"ARTIFACT_PUBLICATION_NOT_MIGRATED"},
        "evidence_availability":collection_provenance(snapshot,attempt),"evidence":evidence,
        "process":process,"diagnostics":snapshot.diagnostics}),
    );
    value["evidence"]["model"]["detail"] = detail;
    if let Some(native) = effective_collection(snapshot, attempt).get("native_artifacts") {
        let state: ml_exp_core::artifacts::publication::State =
            serde_json::from_value(native.clone()).map_err(|_| Error::Evidence)?;
        state.validate().map_err(|_| Error::Identity)?;
        if state.bundle.archive.project != snapshot.project
            || state.bundle.archive.run_id != snapshot.run_id
            || state.bundle.archive.attempt_id != attempt_id
        {
            return Err(Error::Identity);
        }
        let available = state.available().map_err(|_| Error::Identity)?;
        let status = if available {
            "AVAILABLE"
        } else if state.retention.status == ml_exp_core::artifacts::RetentionStatus::Released {
            "RELEASED"
        } else {
            "RELEASING"
        };
        value["status"] = json!(status);
        value["phase"] = json!(status);
        value["verification"] = json!("SHA256_VERIFIED");
        value["download_available"] = json!(available);
        value["publication"] = json!({"status":status,"archive_sha256":state.bundle.archive.sha256,"archive_bytes":state.bundle.archive.bytes,"files":state.bundle.archive.files,"retention":state.retention});
        value["diagnostic"] = Value::Null;
    }
    bounded(value)
}

/// A lightweight metadata projection for project aggregation. No scientific
/// payload or indexed collector state is substituted for missing evidence.
pub fn run_summary(snapshot: &Snapshot, now_unix: f64) -> Result<Value> {
    clock(now_unix)?;
    let evidence = match current(snapshot)? {
        Some(attempt) => layers(snapshot, attempt, now_unix, None)?,
        None => json!({}),
    };
    let state = &evidence["scheduler"]["state"];
    if !state.is_null() && !state.is_string() {
        return Err(Error::Evidence);
    }
    let stale = ["scheduler", "worker", "process", "model", "evaluation"]
        .iter()
        .any(|name| evidence[name]["stale"] == true);
    bounded(
        json!({"project":snapshot.project,"run_id":snapshot.run_id,"scheduler_state":state,"evidence_stale":stale}),
    )
}

/// Public project-list DTO matches the established REST array member shape.
/// Collector errors must be supplied explicitly once the native collector is
/// migrated; this read service has no collector and makes no such observations.
pub fn project_summary(project: &str, title: &str, rows: &[Value]) -> Result<Value> {
    ml_exp_core::identity::validate_id(project).map_err(|_| Error::Identity)?;
    if title.chars().count() > 2000 {
        return Err(Error::OutputLimit);
    }
    let mut counts = Map::new();
    let mut ids = BTreeSet::new();
    let mut attention = 0usize;
    for row in rows {
        if row["project"] != project || !ids.insert(row["run_id"].as_str().ok_or(Error::Identity)?)
        {
            return Err(Error::Identity);
        }
        let state = match row["scheduler_state"].as_str() {
            Some(s) if !s.is_empty() => s,
            None if !row["scheduler_state"].is_null() => return Err(Error::Evidence),
            _ => "UNKNOWN",
        };
        let count = counts.get(state).and_then(Value::as_u64).unwrap_or(0) + 1;
        counts.insert(state.into(), json!(count));
        if matches!(state.to_uppercase().as_str(), "FAILED" | "PREEMPTED") {
            attention += 1;
        }
        match row["evidence_stale"].as_bool() {
            Some(true) => attention += 1,
            Some(false) => {}
            None => return Err(Error::Evidence),
        }
    }
    bounded(experiment_redact::redact_json(
        json!({"project":project,"title":title,"run_counts":counts,"attention_count":attention}),
    ))
}
