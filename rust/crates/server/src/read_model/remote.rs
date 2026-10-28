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
            "process_evidence":receipt.as_ref().map(|r|json!({"return_code":r.process.exit_code,"exit_code":r.process.exit_code,"termination_signal":r.process.termination_signal,"timed_out":r.process.timed_out,"cancelled":r.process.cancelled,"launch_error":r.process.error,"managed_training":r.training,"managed_contract":r.contract}))});
    attempt.collection = EvidenceDocument::Present {
        source: source.clone(),
        raw_sha256: helper::sha256_bytes(
            &serde_json::to_vec(&metadata).map_err(|_| Error::Identity)?,
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
