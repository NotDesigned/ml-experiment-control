use ml_exp_server::experiments::{
    AttemptSnapshot, Document, EvidenceDocument, LogTail, ManifestSource, MetricTail, Provenance,
    Snapshot,
};
use serde_json::{Value, json};

pub fn document(value: Value, source: ManifestSource) -> Document {
    Document {
        source,
        metadata: value.as_object().unwrap().clone(),
        raw_sha256: format!("sha256:{}", "a".repeat(64)),
        legacy_observation_only: false,
    }
}
pub fn evidence(id: &str, name: &str, mut value: Value) -> EvidenceDocument {
    let map = value.as_object_mut().unwrap();
    map.insert("project".into(), json!("demo"));
    map.insert("run_id".into(), json!("trial"));
    map.insert("attempt_id".into(), json!(id));
    EvidenceDocument::Present {
        source: format!("attempts/{id}/{name}.json"),
        metadata: map.clone(),
        raw_sha256: format!("sha256:{}", "b".repeat(64)),
        modified_at_unix: Some(1_000_000.0),
    }
}
pub fn empty_log() -> LogTail {
    LogTail {
        source: None,
        lines: vec![],
        available_lines: 0,
        truncated: false,
        offset_bytes: 0,
        sampled_bytes: 0,
        modified_at_unix: None,
        diagnostics: vec![],
    }
}
pub fn attempt(id: &str) -> AttemptSnapshot {
    AttemptSnapshot {
        attempt_id: id.into(),
        manifest: document(
            json!({"identity_version":2,"project":"demo","run_id":"trial","attempt_id":id}),
            ManifestSource::AttemptYaml,
        ),
        provenance: Provenance {
            compared_fields_complete: false,
            compared_fields: vec![],
            missing_fields: vec![],
            project_inherited_from_json_run: false,
        },
        status: EvidenceDocument::Missing,
        backend: EvidenceDocument::Missing,
        collection: EvidenceDocument::Missing,
        decision: EvidenceDocument::Missing,
        submission: EvidenceDocument::Missing,
        stdout: empty_log(),
        stderr: empty_log(),
        metrics: MetricTail {
            source: None,
            records: vec![],
            source_attempt_id: Some(id.into()),
            truncated: false,
            diagnostics: vec![],
        },
    }
}
pub fn snapshot() -> Snapshot {
    let mut current_status = evidence(
        "attempt-001",
        "status",
        json!({"state":"QUEUED","detail":{"queue":{"reason":"Resources","code":"WAITING_GPU"}}}),
    );
    if let EvidenceDocument::Present { source, .. } = &mut current_status {
        *source = "status.json".into();
    }
    Snapshot {
        project: "demo".into(),
        run_id: "trial".into(),
        run: document(
            json!({"identity_version":2,"project":"demo","run_id":"trial"}),
            ManifestSource::RunJson,
        ),
        current_status,
        current_collection: EvidenceDocument::Missing,
        current_decision: EvidenceDocument::Missing,
        current_attempt_id: Some("attempt-001".into()),
        current_selection_source: Some("status.json".into()),
        attempts: vec![attempt("attempt-001")],
        diagnostics: vec![],
    }
}
