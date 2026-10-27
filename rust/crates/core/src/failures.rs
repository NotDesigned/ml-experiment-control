//! Pure failure evidence assessment. A producer's label alone is diagnostic;
//! no retry, execution permission or provider fact is inferred from it.
use crate::{
    canonical,
    metrics::{numeric_compare, python_string},
};
use regex::Regex;
use serde::Deserialize;
use serde_json::{Map, Number, Value, json};
use std::{cmp::Ordering, collections::BTreeSet, str::FromStr, sync::OnceLock};
use time::{
    OffsetDateTime, PrimitiveDateTime, UtcOffset,
    format_description::well_known::{Iso8601, Rfc3339},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FailureError {
    #[error("failure evidence has an unsupported structure")]
    Evidence,
    #[error("failure memory evidence is not a finite numeric quantity")]
    Memory,
}

/// Filesystem/source paths and observation times are already-acquired inputs.
/// No timestamp falls back to a clock, mtime or local timezone in this module.
#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AssessmentContext {
    pub attempt_id: String,
    pub attempt_state: Option<String>,
    pub attempt_started_at: Value,
    pub observed_at: Value,
    pub domain_observed_at: Map<String, Value>,
    pub domain_attempt_ids: Map<String, Value>,
    pub domain_evidence_sources: Map<String, Value>,
    pub decision_observed_at: Value,
    pub decision_source_binding: String,
    pub decision_evidence_source: Option<String>,
    /// Required to interpret a supported timezone-less ISO date/time. Without
    /// this explicit choice its freshness is unknown, not implicitly local.
    pub naive_utc_offset_seconds: Option<i32>,
}
impl Default for AssessmentContext {
    fn default() -> Self {
        Self {
            attempt_id: String::new(),
            attempt_state: None,
            attempt_started_at: Value::Null,
            observed_at: Value::Null,
            domain_observed_at: Map::new(),
            domain_attempt_ids: Map::new(),
            domain_evidence_sources: Map::new(),
            decision_observed_at: Value::Null,
            decision_source_binding: "EXACT_ATTEMPT".into(),
            decision_evidence_source: None,
            naive_utc_offset_seconds: None,
        }
    }
}
impl AssessmentContext {
    pub fn new(attempt_id: impl Into<String>) -> Self {
        Self {
            attempt_id: attempt_id.into(),
            ..Self::default()
        }
    }
}
fn truth(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Bool(true) => true,
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Number(value) => value.as_f64().is_none_or(|value| value != 0.0),
    }
}
fn get<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}
fn or_string(value: &Value, default: &str) -> String {
    if truth(value) {
        python_string(value)
    } else {
        default.into()
    }
}
fn strip(value: &str) -> &str {
    // Python's whitespace predicate additionally includes four C0 separators.
    value.trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'))
}
fn normalized_class(value: &Value) -> Option<String> {
    let normalized = strip(&or_string(value, "")).to_owned();
    (!matches!(normalized.to_lowercase().as_str(), "" | "none" | "null")).then_some(normalized)
}

/// Omit raw tails/scripts, cap records and preserve all remaining JSON values.
/// This is compaction, not a scientific normalization or a secret-redaction API.
pub fn compact_evidence(value: &Value) -> Value {
    fn at(value: &Value, depth: usize) -> Value {
        if depth >= 8 {
            return json!("[nested evidence omitted]");
        }
        match value {
            Value::Object(value) => Value::Object(
                value
                    .iter()
                    .filter(|(key, _)| {
                        !matches!(
                            key.as_str(),
                            "stdout_tail"
                                | "stderr_tail"
                                | "raw_stdout"
                                | "raw_stderr"
                                | "startup_script"
                        )
                    })
                    .map(|(key, value)| (key.clone(), at(value, depth + 1)))
                    .collect(),
            ),
            Value::Array(value) => {
                let mut output: Vec<_> = value
                    .iter()
                    .take(40)
                    .map(|value| at(value, depth + 1))
                    .collect();
                if value.len() > 40 {
                    output.push(json!(format!(
                        "[{} additional records omitted]",
                        value.len() - 40
                    )));
                }
                Value::Array(output)
            }
            Value::String(value) if value.chars().count() > 2000 => json!(format!(
                "{}…[truncated]",
                value.chars().take(2000).collect::<String>()
            )),
            _ => value.clone(),
        }
    }
    at(value, 0)
}
pub fn evidence_digest(payload: &Value) -> Result<String, canonical::CanonicalError> {
    canonical::protocol_identity(payload)
}

/// The legacy helper permits review of an unknown classification; this result
/// is never authorization to retry or to execute any action.
pub fn reviewable_unknown_retry(
    decision: &Value,
    resource_approval: &str,
    operator_reason: Option<&str>,
) -> bool {
    fn integer(value: &Value) -> bool {
        value
            .as_number()
            .is_some_and(|value| !value.to_string().contains(['.', 'e', 'E']))
    }
    let allowed = get(decision, "retries_allowed");
    let used = decision.get("retries_used").cloned().unwrap_or(json!(0));
    matches!(
        or_string(get(decision, "action"), "")
            .to_uppercase()
            .as_str(),
        "REVIEW_RETRY" | "DO_NOT_RETRY"
    ) && or_string(get(decision, "failure_class"), "").to_lowercase() == "unknown"
        && integer(allowed)
        && integer(&used)
        && matches!(numeric_compare(&used, allowed), Ok(Ordering::Less))
        && resource_approval == "review_exact"
        && operator_reason.is_none_or(|reason| !strip(reason).is_empty())
}

/// Numeric timestamps must be finite. Common ISO timestamps are parsed without
/// consulting the host timezone. Unsupported legacy date grammars remain unknown.
pub fn evidence_timestamp(value: &Value, naive_offset_seconds: Option<i32>) -> Option<f64> {
    if let Value::Number(number) = value {
        return number.as_f64().filter(|number| number.is_finite());
    }
    let value = value.as_str()?;
    if value.is_empty() || value.len() > 256 {
        return None;
    }
    static ISO: OnceLock<Regex> = OnceLock::new();
    let captures=ISO.get_or_init(||Regex::new(r"^(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})(?:[Tt ](?P<time>[0-9]{2}:[0-9]{2}:[0-9]{2})(?:[.,](?P<fraction>[0-9]+))?(?P<offset>Z|[+-][0-9]{2}:[0-9]{2})?)?$").expect("fixed timestamp pattern")).captures(value)?;
    let time = captures
        .name("time")
        .map_or("00:00:00", |value| value.as_str());
    if &time[6..] == "60" {
        return None;
    } // datetime.fromisoformat rejects leap seconds.
    let mut normalized = format!("{}T{time}", &captures["date"]);
    if let Some(fraction) = captures.name("fraction") {
        normalized.push('.');
        normalized.extend(fraction.as_str().chars().take(6));
    }
    let timestamp = if let Some(offset) = captures.name("offset") {
        normalized.push_str(offset.as_str());
        OffsetDateTime::parse(&normalized, &Rfc3339).ok()?
    } else {
        let offset = UtcOffset::from_whole_seconds(naive_offset_seconds?).ok()?;
        PrimitiveDateTime::parse(&normalized, &Iso8601::DEFAULT)
            .ok()?
            .assume_offset(offset)
    };
    // Parse the exact decimal microsecond ratio once. Summing -1 + .999999
    // would lose precision around the epoch and differ from datetime.timestamp.
    let micros =
        i128::from(timestamp.unix_timestamp()) * 1_000_000 + i128::from(timestamp.microsecond());
    let magnitude = micros.unsigned_abs();
    let seconds = format!(
        "{}{}.{:06}",
        if micros < 0 { "-" } else { "" },
        magnitude / 1_000_000,
        magnitude % 1_000_000
    )
    .parse::<f64>()
    .ok()?;
    seconds.is_finite().then_some(seconds)
}

fn joined_tail(value: &Value) -> Result<String, FailureError> {
    if !truth(value) {
        return Ok(String::new());
    }
    Ok(match value {
        Value::Array(values) => values
            .iter()
            .map(python_string)
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(values) => values.keys().cloned().collect::<Vec<_>>().join("\n"),
        Value::String(value) => value
            .chars()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return Err(FailureError::Evidence),
    })
}
fn memory_bytes(value: &str, unit: &str) -> Result<Value, FailureError> {
    let factor = if unit.eq_ignore_ascii_case("gib") {
        1024f64.powi(3)
    } else {
        1024f64.powi(2)
    };
    let quantity = value.parse::<f64>().map_err(|_| FailureError::Memory)? * factor;
    if !quantity.is_finite() {
        return Err(FailureError::Memory);
    }
    Number::from_str(&format!("{:.0}", quantity.round_ties_even()))
        .map(Value::Number)
        .map_err(|_| FailureError::Memory)
}
fn raw_process_summary(collection: &Value) -> Result<Option<Value>, FailureError> {
    let process = get(collection, "process_evidence");
    let stderr = joined_tail(get(process, "stderr_tail"))?;
    let stdout = joined_tail(get(process, "stdout_tail"))?;
    let combined = format!("{stdout}\n{stderr}");
    let class = normalized_class(get(collection, "failure_class"));
    static OOM: OnceLock<Regex> = OnceLock::new();
    let pattern=OOM.get_or_init(||Regex::new(r"(?i)Tried to allocate (?P<requested>[\d.]+) (?P<requested_unit>[GM]iB).*?total capacity of (?P<total>[\d.]+) (?P<total_unit>[GM]iB) of which (?P<free>[\d.]+) (?P<free_unit>[GM]iB) is free.*?allocated memory (?P<allocated>[\d.]+) (?P<allocated_unit>[GM]iB) is allocated").expect("fixed OOM pattern"));
    if let Some(oom) = pattern.captures(&combined) {
        static RANK: OnceLock<Regex> = OnceLock::new();
        let ranks: BTreeSet<_> = RANK
            .get_or_init(|| {
                Regex::new(r"\[rank([0-9]+)\].*?OutOfMemoryError").expect("fixed rank pattern")
            })
            .captures_iter(&combined)
            .map(|value| value[1].trim_start_matches('0').to_owned())
            .collect();
        static WORLD: OnceLock<Regex> = OnceLock::new();
        let world = WORLD
            .get_or_init(|| {
                Regex::new(r"(?:world_size|nproc_per_node)=([0-9]+)")
                    .expect("fixed world-size pattern")
            })
            .captures(&stdout)
            .and_then(|value| {
                Number::from_str(if value[1].trim_start_matches('0').is_empty() {
                    "0"
                } else {
                    value[1].trim_start_matches('0')
                })
                .ok()
            })
            .map(Value::Number)
            .filter(truth);
        let ranks = (!ranks.is_empty())
            .then(|| json!(ranks.len()))
            .unwrap_or(Value::Null);
        let phase = if combined.contains("Performing initial training step")
            && combined.contains(".backward()")
        {
            "first_backward"
        } else if combined.contains(".backward()") {
            "backward"
        } else {
            "unknown"
        };
        return Ok(Some(
            json!({"failure_signature":"CUDA_OOM","failure_class":"resource","phase":phase,
            "requested_bytes":memory_bytes(&oom["requested"],&oom["requested_unit"])?,
            "free_bytes":memory_bytes(&oom["free"],&oom["free_unit"])?,
            "total_bytes":memory_bytes(&oom["total"],&oom["total_unit"])?,
            "allocated_bytes":memory_bytes(&oom["allocated"],&oom["allocated_unit"])?,
            "rank_count":world.unwrap_or_else(||ranks.clone()),"observed_oom_rank_count":ranks,"source":"collection.process_evidence"}),
        ));
    }
    let lowered = combined.to_lowercase();
    for (needle, signature, default) in [
        ("OutOfMemoryError", "CUDA_OOM", "resource"),
        ("CUDA out of memory", "CUDA_OOM", "resource"),
        (
            "ModuleNotFoundError",
            "MISSING_PYTHON_MODULE",
            "configuration",
        ),
        (
            "no kernel image is available",
            "UNSUPPORTED_CUDA_KERNEL",
            "configuration",
        ),
        ("TIMEOUT", "TIMEOUT", "timeout"),
    ] {
        if lowered.contains(&needle.to_lowercase()) {
            return Ok(Some(
                json!({"failure_signature":signature,"failure_class":if signature=="CUDA_OOM"{default}else{class.as_deref().unwrap_or(default)},"phase":"unknown","source":"collection.process_evidence"}),
            ));
        }
    }
    if class.is_some() || or_string(get(collection, "process_state"), "").to_uppercase() == "FAILED"
    {
        return Ok(Some(
            json!({"failure_signature":"UNCLASSIFIED_PROCESS_FAILURE","failure_class":class.as_deref().unwrap_or("unknown"),"phase":"unknown","source":"collection.process_evidence"}),
        ));
    }
    Ok(None)
}
struct Candidate {
    domain: &'static str,
    value: Value,
    applicable: &'static str,
    reason: String,
    observed_id: Option<String>,
    observed_ts: Option<f64>,
}
fn assess(
    domain: &'static str,
    value: Value,
    terminal: bool,
    collection: &Value,
    context: &AssessmentContext,
    state: &str,
) -> Candidate {
    let observed = context
        .domain_attempt_ids
        .get(domain)
        .unwrap_or_else(|| get(collection, "attempt_id"));
    let observed = or_string(observed, "");
    let observed_id = (!observed.is_empty()).then_some(observed);
    let observed_ts = evidence_timestamp(
        context
            .domain_observed_at
            .get(domain)
            .unwrap_or(&context.observed_at),
        context.naive_utc_offset_seconds,
    );
    let started = evidence_timestamp(
        &context.attempt_started_at,
        context.naive_utc_offset_seconds,
    );
    let (applicable, reason) = if observed_id.as_deref() != Some(&context.attempt_id) {
        (
            "ATTEMPT_MISMATCH",
            format!(
                "{domain} evidence names {} instead of {}",
                observed_id.as_deref().unwrap_or("no Attempt"),
                context.attempt_id
            ),
        )
    } else if started.is_none() || observed_ts.is_none() {
        (
            "UNKNOWN_APPLICABILITY",
            format!("actual Attempt start or {domain} observation time is unavailable"),
        )
    } else if observed_ts < started {
        (
            "STALE",
            format!("{domain} evidence predates the exact Attempt start"),
        )
    } else if !matches!(state, "FAILED" | "PREEMPTED") {
        (
            "NON_APPLICABLE",
            format!("exact Attempt state {state} is not a failed terminal state"),
        )
    } else if !terminal {
        (
            "NON_APPLICABLE",
            format!("{domain} layer state is not terminal failed"),
        )
    } else {
        (
            "APPLICABLE",
            format!("exact, fresh terminal {domain} evidence"),
        )
    };
    Candidate {
        domain,
        value,
        applicable,
        reason,
        observed_id,
        observed_ts,
    }
}

/// Match each domain to the exact Attempt and its actual start. Source names
/// remain explicit provenance, not permission to refresh providers or retry.
pub fn attempt_failure_evidence_assessment(
    collection: &Value,
    decision: Option<&Value>,
    context: &AssessmentContext,
) -> Result<Value, FailureError> {
    if !collection.is_object() {
        return Err(FailureError::Evidence);
    }
    let state = context
        .attempt_state
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("UNKNOWN")
        .to_uppercase();
    let mut candidates = Vec::new();
    let process_state = or_string(get(collection, "process_state"), "UNKNOWN").to_uppercase();
    let raw = raw_process_summary(collection)?;
    let process_terminal = matches!(process_state.as_str(), "FAILED" | "PREEMPTED");
    if process_terminal || raw.is_some() {
        let mut value=raw.unwrap_or_else(||json!({"failure_signature":"UNCLASSIFIED_PROCESS_FAILURE","failure_class":normalized_class(get(collection,"failure_class")).unwrap_or_else(||"unknown".into()),"phase":"unknown","source":"collection.process_evidence"}));
        value["failure_domain"] = json!("process");
        candidates.push(assess(
            "process",
            value,
            process_terminal,
            collection,
            context,
            &state,
        ));
    }
    for (domain, state_field, signature, failed, default, source) in [
        (
            "scheduler",
            "scheduler_state",
            "SCHEDULER_TERMINAL_FAILURE",
            &[
                "FAILED",
                "PREEMPTED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "TIMEOUT",
            ][..],
            "scheduler",
            "collection.scheduler_state",
        ),
        (
            "worker",
            "worker_state",
            "WORKER_TERMINAL_FAILURE",
            &["FAILED", "LOST", "ERROR", "NODE_FAIL"][..],
            "worker",
            "collection.worker_state",
        ),
    ] {
        let layer = or_string(get(collection, state_field), "UNKNOWN").to_uppercase();
        if failed.contains(&layer.as_str()) {
            let mut value = json!({"failure_signature":signature,"failure_class":normalized_class(get(collection,"failure_class")).unwrap_or_else(||default.into()),"failure_domain":domain,"phase":"unknown","source":source});
            value[state_field] = json!(layer);
            candidates.push(assess(domain, value, true, collection, context, &state));
        }
    }
    let selected = candidates
        .iter()
        .position(|candidate| candidate.applicable == "APPLICABLE");
    let mut summary = Value::Null;
    if let Some(index) = selected {
        let candidate = &candidates[index];
        summary = candidate.value.clone();
        summary["attempt_id"] = json!(context.attempt_id);
        summary["observed_at"] = json!(candidate.observed_ts);
        summary["applicability"] = json!("APPLICABLE");
        summary["evidence_source"] = context
            .domain_evidence_sources
            .get(candidate.domain)
            .cloned()
            .unwrap_or(Value::Null);
    }
    let mut diagnostics = Vec::new();
    for (index, candidate) in candidates.into_iter().enumerate() {
        if selected == Some(index) || candidate.applicable == "APPLICABLE" {
            continue;
        }
        let mut value = candidate.value;
        for (key, item) in [
            ("kind", json!("failure_signal")),
            ("attempt_id", json!(candidate.observed_id)),
            ("expected_attempt_id", json!(context.attempt_id)),
            ("attempt_state", json!(state)),
            ("observed_at", json!(candidate.observed_ts)),
            ("applicability", json!(candidate.applicable)),
            (
                "evidence_source",
                context
                    .domain_evidence_sources
                    .get(candidate.domain)
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            ("reason", json!(candidate.reason)),
        ] {
            value[key] = item;
        }
        diagnostics.push(value);
    }
    if let Some(class) = decision
        .filter(|value| value.is_object())
        .and_then(|value| normalized_class(get(value, "failure_class")))
        .filter(|_| summary.is_null())
    {
        diagnostics.push(json!({"kind":"preliminary_failure_classification","failure_class":class,"source":"decision.failure_class","attempt_id":context.attempt_id,"attempt_state":state,"observed_at":evidence_timestamp(&context.decision_observed_at,context.naive_utc_offset_seconds),"applicability":"NON_APPLICABLE","source_binding":context.decision_source_binding,"evidence_source":context.decision_evidence_source,"reason":"decision metadata is contextual only and is not exact, fresh, terminal failure evidence"}));
    }
    Ok(
        json!({"failure_summary":summary,"diagnostic_evidence":compact_evidence(&Value::Array(diagnostics))}),
    )
}

/// Without applicability context the original wrapper returns no failure. Its
/// legacy contract uses only common Attempt/start/observation fields; callers
/// needing per-domain bindings use `attempt_failure_evidence_assessment`.
pub fn structured_failure_summary(
    collection: &Value,
    decision: Option<&Value>,
    context: Option<&AssessmentContext>,
) -> Result<Option<Value>, FailureError> {
    let Some(context) = context else {
        return Ok(None);
    };
    let common = AssessmentContext {
        attempt_id: context.attempt_id.clone(),
        attempt_state: context.attempt_state.clone(),
        attempt_started_at: context.attempt_started_at.clone(),
        observed_at: context.observed_at.clone(),
        naive_utc_offset_seconds: context.naive_utc_offset_seconds,
        ..AssessmentContext::default()
    };
    let value = attempt_failure_evidence_assessment(collection, decision, &common)?;
    Ok(value
        .get("failure_summary")
        .filter(|value| !value.is_null())
        .cloned())
}
