//! Pure projection of accepted exact-worker samples onto existing read views.
//! No path, clock, storage, provider or publication effects occur here.
use super::{Error, Result};
use crate::experiments::{
    Diagnostic, EvidenceDocument, EvidenceProblem, LogTail, MetricTail, Snapshot,
};
use ml_exp_core::{
    first_submission::native::Binding,
    managed::{
        evidence::Phase,
        remote::{Evidence, Problem, Window},
    },
    slurm::helper,
};
use serde_json::{Value, json};
pub(crate) fn project_worker(
    snapshot: &mut Snapshot,
    binding: &Binding,
    evidence: &Evidence,
    action: &str,
    observed_at: &str,
) -> Result<()> {
    if snapshot.project != binding.project
        || snapshot.run_id != binding.run_id
        || snapshot.run.raw_sha256 != binding.run_sha256
    {
        return Err(Error::Identity);
    }
    let Some(attempt) = snapshot
        .attempts
        .iter_mut()
        .find(|a| a.attempt_id == binding.attempt_id)
    else {
        return Ok(());
    };
    if attempt.manifest.raw_sha256 != binding.attempt_sha256 {
        return Err(Error::Identity);
    }
    let source = format!("native-worker-observation:{}", action);
    let receipt = &evidence.worker.receipt;
    let metadata = json!({"project":snapshot.project,"run_id":snapshot.run_id,"attempt_id":attempt.attempt_id,
            "worker_state":evidence.worker.phase,"process_state":if evidence.worker.phase == Phase::Finished { receipt.as_ref().map(|r|json!(r.training)).unwrap_or(Value::Null) } else {json!("UNKNOWN")},
            "observed_at":observed_at,"observation_kind":"shared-storage-sample","heartbeat_proven":false,
            "process_evidence":receipt.as_ref().map(|r|json!({"return_code":r.process.exit_code,"exit_code":r.process.exit_code,"termination_signal":r.process.termination_signal,"timed_out":r.process.timed_out,"cancelled":r.process.cancelled,"launch_error":r.process.error,"stdout_dropped_bytes":r.process.stdout_dropped_bytes,"stderr_dropped_bytes":r.process.stderr_dropped_bytes,"managed_training":r.training,"managed_contract":r.contract}))});
    attempt.collection = EvidenceDocument::Present {
        source: source.clone(),
        raw_sha256: helper::sha256_bytes(
            &serde_json::to_vec(&metadata).map_err(why!(Error::Identity))?,
        ),
        metadata: metadata.as_object().ok_or(Error::Identity)?.clone(),
        modified_at_unix: None,
    };
    attempt.stdout = log_tail(&evidence.stdout, &format!("{source}/stdout.log"));
    attempt.stderr = log_tail(&evidence.stderr, &format!("{source}/stderr.log"));
    let (records, incomplete, invalid, foreign) = evidence.metric_records();
    let mut diagnostics = diagnostic(&evidence.metrics, &format!("{source}/metrics.jsonl"));
    if invalid > 0 {
        diagnostics.push(Diagnostic {
            source: source.clone(),
            code: EvidenceProblem::MalformedRecord,
            count: invalid,
        });
    }
    if foreign > 0 {
        diagnostics.push(Diagnostic {
            source: source.clone(),
            code: EvidenceProblem::ForeignRecord,
            count: foreign,
        });
    }
    if !evidence.metrics.text.is_empty() && !evidence.metrics.text.ends_with('\n') {
        diagnostics.push(Diagnostic {
            source: source.clone(),
            code: EvidenceProblem::PartialFinalLine,
            count: 1,
        });
    }
    attempt.metrics = MetricTail {
        source: Some(format!("{source}/metrics.jsonl")),
        records,
        source_attempt_id: Some(attempt.attempt_id.clone()),
        truncated: incomplete,
        diagnostics,
    };
    Ok(())
}

fn diagnostic(window: &Window, source: &str) -> Vec<Diagnostic> {
    window
        .problem
        .as_ref()
        .map(|problem| Diagnostic {
            source: source.into(),
            count: 1,
            code: match problem {
                Problem::Truncated => EvidenceProblem::Truncated,
                Problem::InvalidUtf8 => EvidenceProblem::InvalidUtf8,
                Problem::Changed | Problem::HashMismatch => EvidenceProblem::IdentityConflict,
                Problem::Missing | Problem::Unavailable => EvidenceProblem::Io,
            },
        })
        .into_iter()
        .collect()
}
fn log_tail(window: &Window, source: &str) -> LogTail {
    let lines: Vec<_> = window
        .text
        .lines()
        .skip(usize::from(window.offset_bytes > 0))
        .map(str::to_owned)
        .collect();
    LogTail {
        source: Some(source.into()),
        available_lines: lines.len(),
        lines,
        truncated: window.problem.is_some(),
        offset_bytes: window.offset_bytes,
        sampled_bytes: window.text.len() as u64,
        modified_at_unix: None,
        diagnostics: diagnostic(window, source),
    }
}

pub(crate) fn project_artifacts(
    snapshot: &mut Snapshot,
    binding: &Binding,
    state: &ml_exp_core::artifacts::publication::State,
) -> Result<()> {
    state.validate().map_err(why!(Error::Identity))?;
    let receipt = &state.bundle.archive;
    if snapshot.project != binding.project
        || snapshot.run_id != binding.run_id
        || snapshot.run.raw_sha256 != binding.run_sha256
        || receipt.project != binding.project
        || receipt.run_id != binding.run_id
        || receipt.attempt_id != binding.attempt_id
    {
        return Err(Error::Identity);
    }
    let Some(attempt) = snapshot
        .attempts
        .iter_mut()
        .find(|a| a.attempt_id == binding.attempt_id)
    else {
        return Ok(());
    };
    if attempt.manifest.raw_sha256 != binding.attempt_sha256 {
        return Err(Error::Identity);
    }
    let mut metadata = attempt.collection.metadata().cloned().unwrap_or_default();
    metadata.insert("project".into(), json!(binding.project));
    metadata.insert("run_id".into(), json!(binding.run_id));
    metadata.insert("attempt_id".into(), json!(binding.attempt_id));
    metadata.insert("native_artifacts".into(), json!(state));
    let raw_sha256 =
        helper::sha256_bytes(&serde_json::to_vec(&metadata).map_err(why!(Error::Identity))?);
    attempt.collection = EvidenceDocument::Present {
        source: "native-artifact-registry".into(),
        raw_sha256,
        metadata,
        modified_at_unix: None,
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::{AttemptSnapshot, Document, ManifestSource, Provenance};
    use ml_exp_core::{
        launch,
        managed::{
            evidence::{CONTRACT, Observation, ProcessEvidence, Receipt, Training},
            remote::OBSERVATION_CONTRACT,
        },
    };

    #[test]
    fn process_evidence_counts_what_each_log_dropped() {
        let digest = "sha256:a".to_owned();
        let document = |source: ManifestSource| Document {
            source,
            raw_sha256: digest.clone(),
            metadata: Default::default(),
        };
        let log = || LogTail {
            source: None,
            lines: vec![],
            available_lines: 0,
            truncated: false,
            offset_bytes: 0,
            sampled_bytes: 0,
            modified_at_unix: None,
            diagnostics: vec![],
        };
        let mut snapshot = Snapshot {
            project: "demo".into(),
            run_id: "train".into(),
            run: document(ManifestSource::RunJson),
            current_status: EvidenceDocument::Missing,
            current_attempt_id: None,
            current_selection_source: None,
            attempts: vec![AttemptSnapshot {
                attempt_id: "attempt-001".into(),
                manifest: document(ManifestSource::AttemptJson),
                provenance: Provenance {
                    compared_fields_complete: false,
                    compared_fields: vec![],
                    missing_fields: vec![],
                },
                status: EvidenceDocument::Missing,
                collection: EvidenceDocument::Missing,
                stdout: log(),
                stderr: log(),
                metrics: MetricTail {
                    source: None,
                    records: vec![],
                    source_attempt_id: None,
                    truncated: false,
                    diagnostics: vec![],
                },
            }],
            diagnostics: vec![],
        };
        let binding = Binding {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
            project_sha256: digest.clone(),
            run_sha256: digest.clone(),
            attempt_sha256: digest.clone(),
            source_receipt_sha256: digest.clone(),
            backend_config_sha256: digest.clone(),
        };
        let identity = launch::Identity {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
        };
        let evidence = Evidence {
            contract: OBSERVATION_CONTRACT.into(),
            config_sha256: digest.clone(),
            job_id: digest.clone(),
            worker: Observation {
                contract: CONTRACT.into(),
                identity: identity.clone(),
                manifest_sha256: digest.clone(),
                phase: Phase::Finished,
                safe_to_retry: false,
                receipt: Some(Receipt {
                    contract: CONTRACT.into(),
                    identity,
                    manifest_sha256: digest.clone(),
                    process: ProcessEvidence {
                        exit_code: Some(0),
                        termination_signal: None,
                        timed_out: false,
                        cancelled: false,
                        error: None,
                        stdout_sha256: None,
                        stderr_sha256: None,
                        stdout_dropped_bytes: 7,
                        stderr_dropped_bytes: 0,
                    },
                    training: Training::Completed,
                    download_available: false,
                    runtime_environment_sha256: None,
                }),
                runtime_environment_sha256: None,
            },
            stdout: Window::unavailable(Problem::Missing),
            stderr: Window::unavailable(Problem::Missing),
            metrics: Window::unavailable(Problem::Missing),
        };
        project_worker(&mut snapshot, &binding, &evidence, "action", "now").unwrap();
        let EvidenceDocument::Present { metadata, .. } = &snapshot.attempts[0].collection else {
            panic!("worker evidence was not projected");
        };
        assert_eq!(metadata["process_evidence"]["stdout_dropped_bytes"], 7);
        assert_eq!(metadata["process_evidence"]["stderr_dropped_bytes"], 0);
    }
}
