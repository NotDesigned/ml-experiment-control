//! Read-only readiness projection. Configuration permits submission; cached
//! provider reads and successful GPU execution are separate evidence.
use crate::read_api::Api;
use serde_json::{Value, json};

pub(crate) fn observe(api: Option<&Api>) -> Value {
    let mut rows = Vec::new();
    let mut errors = Vec::new();
    if let Some(api) = api {
        if let Some(slurm) = api.submissions() {
            rows.extend(slurm.catalogue());
        }
        if let Some(cloud) = api.cloud() {
            match cloud.catalogue() {
                Ok(values) => rows.extend(values),
                Err(_) => {
                    errors.push(json!({"code":"BACKEND_CATALOGUE_UNAVAILABLE", "status":"BLOCKED"}))
                }
            }
        }
    }
    let now = crate::projects::transition_context()
        .ok()
        .map(|(now, _)| now);
    let backends = rows
        .into_iter()
        .map(|row| {
            let executor = row["executor_id"].as_str().unwrap_or("");
            let cached = api.and_then(|api| {
                let now = now.as_deref()?;
                if let Some(cloud) = api.cloud().filter(|service| service.accepts(executor)) {
                    cloud.capacity(executor, false, now).ok()
                } else {
                    api.submissions()?.capacity(executor, false, now).ok()
                }
            });
            backend(row, cached.as_ref())
        })
        .collect::<Vec<_>>();
    document(
        api.is_some(),
        api.is_some_and(|api| api.sources().is_some()),
        api.is_some_and(|api| api.is_historical() || api.history().is_some()),
        backends,
        errors,
    )
}

fn backend(row: Value, capacity: Option<&Value>) -> Value {
    let mut blockers = Vec::new();
    for (field, code) in [
        ("enabled", "SCHEDULER_MUTATIONS_DISABLED"),
        ("run_creation", "RUN_CREATION_DISABLED"),
    ] {
        if row[field] != true {
            blockers.push(json!({"code":code,"status":"BLOCKED"}));
        }
    }
    let provider = match capacity {
        Some(value) => json!({"status":value["status"],"diagnostic":value["diagnostic"],
            "observed_at":value["observed_at"],"checked_at":value["checked_at"],
            "source":"cached_capacity_query", "live_query_performed":false}),
        None => json!({"status":"UNAVAILABLE","diagnostic":"CAPACITY_CACHE_UNAVAILABLE",
            "observed_at":null,"checked_at":null,"live_query_performed":false}),
    };
    json!({"executor_id":row["executor_id"],"kind":row["kind"],
        "submission_available":blockers.is_empty(),"submission_blockers":blockers,
        "configuration":{"status":"VALIDATED_AT_STARTUP"},"capabilities":row["capabilities"],
        "provider_connectivity":provider,
        "gpu_end_to_end":{"status":"NOT_RECORDED","code":"GPU_ACCEPTANCE_EVIDENCE_NOT_CONFIGURED",
            "detail":"Submission permission and capacity observations do not prove a successful GPU training and checkpoint restore."},
        "admission_guaranteed":false})
}

fn document(
    reads: bool,
    sources: bool,
    history: bool,
    backends: Vec<Value>,
    mut checks: Vec<Value>,
) -> Value {
    checks.push(
        json!({"id":"local_reads","status":if reads{"PASS"}else{"BLOCKED"},
        "code":if reads{Value::Null}else{json!("LOCAL_READS_NOT_CONFIGURED")}}),
    );
    checks.push(
        json!({"id":"source_imports","status":if sources{"PASS"}else{"BLOCKED"},
        "code":if sources{Value::Null}else{json!("SOURCE_IMPORTS_NOT_CONFIGURED")}}),
    );
    if backends.is_empty() {
        checks.push(json!({"id":"training_backends","status":"BLOCKED","code":"NO_TRAINING_BACKENDS_CONFIGURED"}));
    }
    checks.push(json!({"id":"gpu_acceptance","status":"NOT_RECORDED",
        "code":"GPU_ACCEPTANCE_EVIDENCE_NOT_CONFIGURED",
        "detail":"No machine-readable end-to-end acceptance registry is configured; inspect per-backend submission availability."}));
    if history {
        checks.push(
            json!({"id":"historical_metadata_audit","status":"NOT_RECORDED",
            "code":"FULL_HISTORICAL_METADATA_AUDIT_NOT_RECORDED"}),
        );
        checks.push(json!({"id":"historical_mutations","status":"UNSUPPORTED",
            "code":"HISTORICAL_NAMESPACE_READ_ONLY",
            "detail":"Historical logs, metrics and receipts are readable; old submissions are not replayed or rewritten."}));
    }
    let available = backends
        .iter()
        .filter(|row| row["submission_available"] == true)
        .map(|row| row["executor_id"].clone())
        .collect::<Vec<_>>();
    json!({"api":"READY","training":"INCOMPLETE","migration":if history{"IN_PROGRESS"}else{"NOT_REQUIRED"},
        "submission_available":!available.is_empty(),"submission_available_backends":available,
        "scheduler_mutations":backends.iter().any(|row| !row["submission_blockers"].as_array().unwrap().iter().any(|b|b["code"]=="SCHEDULER_MUTATIONS_DISABLED")),
        "providers_configured":!backends.is_empty(),"checks":checks,"backends":backends,
        "provider_queries_performed":false,"training_status_scope":"acceptance_evidence"})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn permissions_failures_and_stale_evidence_are_not_gpu_acceptance() {
        let row = json!({"executor_id":"pai","kind":"pai-dlc","enabled":true,"run_creation":true,"capabilities":{}});
        let allowed = backend(
            row.clone(),
            Some(
                &json!({"status":"STALE","diagnostic":"SCHEDULER_QUERY_UNAVAILABLE","observed_at":"2026-10-01T00:00:00Z"}),
            ),
        );
        assert_eq!(allowed["submission_available"], true);
        assert_eq!(allowed["provider_connectivity"]["status"], "STALE");
        assert_eq!(allowed["gpu_end_to_end"]["status"], "NOT_RECORDED");
        let mut disabled = row;
        disabled["enabled"] = json!(false);
        let blocked = backend(disabled, None);
        assert_eq!(blocked["submission_available"], false);
        let value = document(true, true, true, vec![allowed, blocked], vec![]);
        assert_eq!(value["submission_available_backends"], json!(["pai"]));
        assert_eq!(value["training"], "INCOMPLETE");
        assert_eq!(value["provider_queries_performed"], false);
        assert!(
            document(false, false, false, vec![], vec![])["checks"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["code"] == "NO_TRAINING_BACKENDS_CONFIGURED")
        );
    }
}
