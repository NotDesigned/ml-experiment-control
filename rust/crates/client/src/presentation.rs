//! Pure human summaries of already-bound evidence. No clock, configuration,
//! provider reads, scientific selection or mutation is hidden in rendering.
use ml_exp_core::metrics::{python_is_printable, python_string};
use serde_json::Value;
use time::OffsetDateTime;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("cannot render malformed experiment evidence")]
pub struct PresentationError;
pub type Result<T> = std::result::Result<T, PresentationError>;
fn get<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}
fn object(value: &Value) -> &Value {
    if value.is_object() {
        value
    } else {
        &Value::Null
    }
}
fn truth(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Bool(true) => true,
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        Value::Number(v) => v.as_f64().is_none_or(|n| n != 0.0),
    }
}
fn first<'a>(values: &[&'a Value]) -> &'a Value {
    values
        .iter()
        .copied()
        .find(|value| truth(value))
        .unwrap_or(&Value::Null)
}
fn text(value: &Value, default: &str) -> String {
    if !matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_)) {
        return default.into();
    }
    let raw = python_string(value)
        .chars()
        .map(|c| if python_is_printable(c) { c } else { ' ' })
        .collect::<String>();
    let clean = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if clean.is_empty() {
        return default.into();
    }
    let mut result = clean.chars().take(400).collect::<String>();
    if clean.chars().count() > 400 {
        result.push('…');
    }
    result
}
fn show(value: &Value) -> String {
    text(value, "UNKNOWN")
}
fn number(value: &Value) -> Result<Option<f64>> {
    let Some(value) = value.as_number() else {
        return Ok(None);
    };
    let finite = value.as_f64().filter(|v| v.is_finite());
    if finite.is_none() && !value.to_string().contains(['.', 'e', 'E']) {
        return Err(PresentationError);
    }
    Ok(finite.filter(|v| *v >= 0.0))
}
fn bytes(value: &Value) -> Result<String> {
    let Some(mut value) = number(value)? else {
        return Ok("unknown".into());
    };
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut index = 0;
    while value >= 1024.0 && index < units.len() - 1 {
        value /= 1024.0;
        index += 1;
    }
    Ok(format!("{value:.1} {}", units[index]))
}
fn divided(digits: &str, divisor: u32) -> (String, u32) {
    let mut result = String::new();
    let mut remainder = 0;
    for byte in digits.bytes() {
        let value = remainder * 10 + u32::from(byte - b'0');
        let next = value / divisor;
        remainder = value % divisor;
        if next != 0 || !result.is_empty() {
            result.push(char::from(b'0' + next as u8));
        }
    }
    if result.is_empty() {
        result.push('0');
    }
    (result, remainder)
}
fn age(value: &Value) -> Result<Option<String>> {
    let Some(number) = number(value)? else {
        return Ok(None);
    };
    let raw = value.as_number().expect("number checked").to_string();
    let seconds = if raw.contains(['.', 'e', 'E']) {
        format!("{:.0}", number.trunc())
            .trim_start_matches('-')
            .to_owned()
    } else {
        raw.trim_start_matches('-').into()
    };
    let (minutes, remainder) = divided(&seconds, 60);
    Ok(Some(if minutes == "0" {
        format!("{remainder}s ago")
    } else {
        let (hours, minute) = divided(&minutes, 60);
        if hours == "0" {
            format!("{minutes}m {remainder}s ago")
        } else {
            format!("{hours}h {minute}m ago")
        }
    }))
}
fn observed(value: &Value) -> Result<String> {
    let mut timestamp = ["observed_at", "as_of", "at"]
        .into_iter()
        .map(|key| get(value, key))
        .find(|v| !v.is_null())
        .cloned()
        .unwrap_or(Value::Null);
    if let Some(number) = number(&timestamp)? {
        // datetime.fromtimestamp rounds to microseconds before timespec=seconds.
        let mut seconds = number.floor();
        if (number.fract() * 1_000_000.0).round_ties_even() >= 1_000_000.0 {
            seconds += 1.0;
        }
        timestamp = if seconds <= 253_402_300_799.0 {
            OffsetDateTime::from_unix_timestamp(seconds as i64)
                .ok()
                .map(|v| {
                    Value::String(format!(
                        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}+00:00",
                        v.year(),
                        u8::from(v.month()),
                        v.day(),
                        v.hour(),
                        v.minute(),
                        v.second()
                    ))
                })
                .unwrap_or(Value::Null)
        } else {
            Value::Null
        };
    }
    let mut parts = Vec::new();
    if !timestamp.is_null() {
        parts.push(format!("observed {}", show(&timestamp)));
    }
    if get(value, "stale") == true {
        parts.push("stale evidence".into());
    }
    Ok(parts.join("; "))
}
fn diagnostic(value: &Value) -> String {
    if value.is_object() {
        ["code", "reason", "message", "error_code"]
            .into_iter()
            .map(|key| get(value, key))
            .filter(|v| truth(v))
            .map(show)
            .collect::<Vec<_>>()
            .join("; ")
    } else {
        text(value, "")
    }
}
fn command(argv: &Value) -> Option<String> {
    let args = argv.as_array()?;
    if args.is_empty() || args.len() > 64 {
        return None;
    }
    let mut result = Vec::new();
    for arg in args {
        let arg = arg.as_str()?;
        if arg.is_empty() || !arg.chars().all(python_is_printable) {
            return None;
        }
        result.push(
            if arg
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&c))
            {
                arg.into()
            } else {
                format!("'{}'", arg.replace('\'', "'\"'\"'"))
            },
        );
    }
    let joined = result.join(" ");
    (joined.chars().count() <= 2048).then_some(joined)
}
fn next_lines(value: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(actions) = get(value, "next_actions").as_array() {
        for item in actions.iter().take(4).filter(|v| v.is_object()) {
            let mut label = format!("Next: {}", show(get(item, "kind")));
            if get(item, "requires_execution") == true {
                label.push_str(" (explicit execution required)");
            }
            if truth(get(item, "reason")) {
                label.push_str(&format!(" — {}", show(get(item, "reason"))));
            }
            lines.push(label);
            if !get(item, "argv").is_null() {
                lines.push(format!(
                    "  {}",
                    command(get(item, "argv"))
                        .unwrap_or_else(|| "Command unavailable; inspect JSON details.".into())
                ));
            }
        }
    }
    if lines.is_empty() {
        lines.push(format!(
            "Next: {}",
            text(
                get(value, "next_action"),
                "inspect existing evidence; no action specified"
            )
        ));
    }
    lines
}
fn member(value: &Value, names: &[&str]) -> Result<bool> {
    if value.is_array() || value.is_object() {
        return Err(PresentationError);
    }
    Ok(value.as_str().is_some_and(|value| names.contains(&value)))
}
fn queue_lines(queue: &Value, reason: &Value, waiting: bool) -> Result<Vec<String>> {
    let blocker = object(get(queue, "blocker"));
    let reason = first(&[get(queue, "reason"), get(blocker, "Reason"), reason]);
    let mut details = [
        first(&[get(queue, "code"), get(blocker, "Code")]),
        get(queue, "provider_code"),
    ]
    .into_iter()
    .filter(|v| truth(v))
    .map(show)
    .collect::<Vec<_>>();
    if number(get(queue, "position"))?.is_some() {
        details.push(format!("position {}", show(get(queue, "position"))));
    }
    if !truth(reason) && details.is_empty() && !waiting {
        return Ok(Vec::new());
    }
    let observed = observed(queue)?;
    if !observed.is_empty() {
        details.push(observed);
    }
    Ok(vec![format!(
        "{}{}{}",
        if waiting {
            "Waiting: "
        } else {
            "Scheduler reason: "
        },
        text(reason, "reason not observed"),
        if details.is_empty() {
            String::new()
        } else {
            format!(" ({})", details.join("; "))
        }
    )])
}
fn progress_lines(value: &Value) -> Result<Vec<String>> {
    let age = age(get(value, "seconds_since_progress"))?;
    let observed = observed(
        &serde_json::json!({"observed_at":get(value,"last_progress_unix"),"stale":get(value,"stale")}),
    )?;
    let mut parts = Vec::new();
    if let Some(age) = age {
        parts.push(format!("last recorded {age}"));
    }
    if !observed.is_empty() {
        parts.push(observed);
    }
    let mut lines = vec![format!(
        "Progress: {}",
        if parts.is_empty() {
            "recent progress not observed".into()
        } else {
            parts.join("; ")
        }
    )];
    if get(value, "no_progress_warning") == true {
        lines.push("Warning: no recorded progress; this does not establish failure.".into());
    }
    Ok(lines)
}
fn stage_line(phase: &Value, worker: &Value) -> String {
    format!(
        "Stage: {}{}",
        show(phase),
        if truth(get(worker, "phase"))
            && ml_exp_core::observation::python_equal(phase, get(worker, "phase"))
        {
            " (last reported worker phase; not a heartbeat)"
        } else {
            ""
        }
    )
}
fn worker_lines(worker: &Value) -> Result<Vec<String>> {
    if !truth(worker) {
        return Ok(Vec::new());
    }
    let mut parts = vec![
        show(get(worker, "phase")),
        "last reported; not a heartbeat".into(),
    ];
    let observed = observed(worker)?;
    if !observed.is_empty() {
        parts.push(observed);
    }
    if let Some(age) = age(get(worker, "age_seconds"))? {
        parts.push(format!("report {age}"));
    }
    Ok(vec![format!("Worker phase: {}", parts.join("; "))])
}
fn training_line(training: &Value, process: bool) -> Result<String> {
    let mut parts = vec![show(get(training, "status"))];
    if process {
        parts.push(format!(
            "process {}",
            text(get(training, "process_state"), "not observed")
        ));
    }
    if !get(training, "exit_code").is_null() {
        parts.push(format!("exit {}", show(get(training, "exit_code"))));
    }
    if process {
        let observed = observed(training)?;
        if !observed.is_empty() {
            parts.push(format!("process evidence: {observed}"));
        }
    }
    Ok(format!("Training: {}", parts.join("; ")))
}
fn result_lines(publication: &Value, verification: &Value) -> Vec<String> {
    let verified = if verification.is_object() {
        get(verification, "status")
    } else {
        verification
    };
    let available = get(publication, "status") == "AVAILABLE"
        && matches!(
            verified.as_str(),
            Some("SHA256_VERIFIED" | "HISTORICAL_RECEIPT")
        )
        && get(publication, "download_available") == true;
    vec![
        format!(
            "Results: {}; download {}",
            text(get(publication, "status"), "NOT_OBSERVED"),
            if available {
                "available"
            } else {
                "unavailable"
            }
        ),
        format!("Verification: {}", text(verified, "NOT_OBSERVED")),
    ]
}
fn input_line(value: &Value) -> Result<Option<String>> {
    if !truth(value) {
        return Ok(None);
    }
    let mut parts = vec![show(get(value, "phase"))];
    if number(get(value, "received_bytes"))?.is_some() {
        parts.push(format!(
            "{} / {}",
            bytes(get(value, "received_bytes"))?,
            bytes(get(value, "expected_bytes"))?
        ));
    }
    if number(get(value, "bytes_per_second"))?.is_some() {
        parts.push(format!("{}/s", bytes(get(value, "bytes_per_second"))?));
    }
    if let Some(age) = age(get(value, "seconds_since_progress"))? {
        parts.push(format!("last progress {age}"));
    }
    let observed = observed(value)?;
    if !observed.is_empty() {
        parts.push(observed);
    }
    for field in [
        "code",
        "failed_phase",
        "http_status",
        "errno",
        "retries",
        "retry_count",
    ] {
        if !get(value, field).is_null() {
            parts.push(format!("{field}={}", show(get(value, field))));
        }
    }
    Ok(Some(format!("Input evidence: {}", parts.join("; "))))
}
fn root(value: &Value) -> Result<()> {
    if value.is_object() {
        Ok(())
    } else {
        Err(PresentationError)
    }
}
/// Render reported scheduler, process/model and publication states separately.
/// Absolute timestamps and already-computed ages come only from `value`.
pub fn status_summary(value: &Value) -> Result<String> {
    root(value)?;
    let preparation = object(get(value, "preparation"));
    let scheduler = object(get(value, "scheduler"));
    let training = object(get(value, "training"));
    let publication = object(get(value, "publication"));
    let verification = object(get(value, "verification"));
    let progress = object(get(value, "progress"));
    let model = object(get(value, "model"));
    let worker = first(&[
        object(get(value, "worker_phase")),
        object(get(progress, "worker_phase")),
    ]);
    let mut lines = vec![
        format!(
            "Run: {}/{} · {}",
            show(get(value, "project")),
            show(get(value, "run_id")),
            text(get(value, "attempt_id"), "Attempt not observed")
        ),
        stage_line(
            first(&[
                get(progress, "phase"),
                get(preparation, "phase"),
                get(scheduler, "state"),
            ]),
            worker,
        ),
    ];
    if !get(preparation, "status").is_null() && get(preparation, "status") != "NOT_OBSERVED" {
        lines.push(format!("Preparation: {}", show(get(preparation, "status"))));
    }
    let observed_scheduler = observed(scheduler)?;
    lines.push(format!(
        "Scheduler: {}{}",
        show(get(scheduler, "state")),
        if observed_scheduler.is_empty() {
            String::new()
        } else {
            format!("; {observed_scheduler}")
        }
    ));
    lines.push(format!(
        "Backend: {}; job {}; native {}",
        text(get(scheduler, "backend"), "not observed"),
        text(get(scheduler, "job_id"), "not observed"),
        text(get(scheduler, "native_state"), "not observed")
    ));
    lines.extend(queue_lines(
        first(&[
            object(get(value, "queue")),
            object(get(scheduler, "queue")),
            object(get(progress, "queue")),
        ]),
        get(scheduler, "reason"),
        member(get(scheduler, "state"), &["QUEUED", "PENDING"])?,
    )?);
    lines.push(training_line(training, true)?);
    lines.extend(worker_lines(worker)?);
    let step = model
        .get("step")
        .unwrap_or_else(|| get(object(get(model, "detail")), "step"));
    let measurement = if number(step)?.is_some() {
        format!("step {}", show(step))
    } else {
        text(get(model, "state"), "NOT_OBSERVED")
    };
    let observed_model = observed(model)?;
    lines.push(format!(
        "Model: {measurement}{}",
        if observed_model.is_empty() {
            String::new()
        } else {
            format!("; {observed_model}")
        }
    ));
    let metric = object(get(model, "last_metric"));
    if truth(metric) {
        lines.push(format!(
            "Last metric: {}={} {}; {}",
            show(get(metric, "name")),
            text(get(metric, "value"), "missing"),
            show(get(metric, "unit")),
            text(get(metric, "status"), "status not recorded")
        ));
        if get(model, "observed_at").is_null() {
            lines.push("Metric time not recorded; freshness is unknown.".into());
        }
    }
    lines.extend(progress_lines(progress)?);
    if let Some(line) = input_line(first(&[
        object(get(value, "input_delivery")),
        object(get(progress, "input_delivery")),
    ]))? {
        lines.push(line);
    }
    lines.extend(result_lines(publication, verification));
    for item in [
        get(progress, "diagnostic"),
        get(publication, "diagnostic"),
        get(publication, "error"),
    ] {
        let diagnostic = diagnostic(item);
        if !diagnostic.is_empty() {
            lines.push(format!("Diagnostic: {diagnostic}"));
        }
    }
    lines.extend(next_lines(value));
    Ok(lines.join("\n"))
}
/// A polling update never treats scheduler RUNNING as a training heartbeat.
pub fn progress_summary(value: &Value) -> Result<String> {
    root(value)?;
    let progress = object(value.get("progress").unwrap_or(value));
    let worker = first(&[
        object(get(value, "worker_phase")),
        object(get(progress, "worker_phase")),
    ]);
    let mut lines = vec![stage_line(
        progress
            .get("phase")
            .unwrap_or_else(|| get(progress, "status")),
        worker,
    )];
    if truth(get(progress, "message")) {
        lines.push(show(get(progress, "message")));
    }
    lines.extend(queue_lines(
        first(&[object(get(value, "queue")), object(get(progress, "queue"))]),
        &Value::Null,
        member(get(progress, "phase"), &["QUEUED", "PENDING"])?,
    )?);
    lines.extend(progress_lines(progress)?);
    lines.extend(worker_lines(worker)?);
    if let Some(line) = input_line(object(get(progress, "input_delivery")))? {
        lines.push(line);
    }
    let message = diagnostic(get(progress, "diagnostic"));
    if !message.is_empty() {
        lines.push(format!("Diagnostic: {message}"));
    }
    let mut collection = first(&[
        object(get(progress, "result_collection")),
        object(get(value, "result_collection")),
    ]);
    if !truth(collection)
        && ["training", "verification", "download_available"]
            .iter()
            .any(|key| progress.get(*key).is_some())
    {
        collection = progress;
    }
    if truth(collection) {
        lines.push(training_line(object(get(collection, "training")), false)?);
        lines.extend(result_lines(collection, get(collection, "verification")));
        let diagnostic = diagnostic(get(collection, "diagnostic"));
        if !diagnostic.is_empty() {
            lines.push(format!("Diagnostic: {diagnostic}"));
        }
    }
    Ok(lines.join("\n"))
}
fn gates(value: &Value) -> Vec<&Value> {
    get(value, "gates")
        .as_array()
        .map(|items| items.iter().filter(|v| v.is_object()).collect())
        .unwrap_or_default()
}
/// Review display only. An argv is shell-quoted, bounded and never executed.
pub fn operation_summary(value: &Value) -> Result<String> {
    root(value)?;
    let action = object(get(value, "action"));
    let execution = object(get(action, "execution"));
    let request = object(get(value, "request"));
    let preflight = object(get(action, "preflight_summary"));
    let mut lines = vec![
        format!(
            "Operation: {} — {}/{}",
            show(get(value, "operation")),
            show(get(value, "project")),
            show(get(value, "run_id"))
        ),
        format!(
            "Attempts: {} → {}",
            show(get(value, "source_attempt_id")),
            show(get(action, "attempt_id"))
        ),
        format!(
            "Action: {}; {}",
            show(get(action, "action_id")),
            show(get(execution, "status"))
        ),
        format!(
            "Ready: {}",
            if get(action, "ready") == true {
                "yes"
            } else if get(action, "ready") == false {
                "no"
            } else {
                "unknown"
            }
        ),
    ];
    let cap = get(request, "max_gpu_hours");
    let requested = get(preflight, "requested_gpu_hours");
    if number(cap)?.is_some() || number(requested)?.is_some() {
        lines.push(format!(
            "GPU hours: requested {}; cap {}",
            if number(requested)?.is_some() {
                show(requested)
            } else {
                "unknown".into()
            },
            if number(cap)?.is_some() {
                show(cap)
            } else {
                "unknown".into()
            }
        ));
    }
    let gates = gates(action);
    let notable = gates
        .iter()
        .filter(|v| get(v, "status") != "PASS")
        .copied()
        .collect::<Vec<_>>();
    lines.push(format!(
        "Gates: {}",
        if gates.is_empty() {
            "NOT_OBSERVED"
        } else if notable.is_empty() {
            "PASS"
        } else {
            "inspect"
        }
    ));
    for gate in notable.into_iter().take(6) {
        lines.push(format!(
            "  {}: {}{}",
            show(get(gate, "name")),
            show(get(gate, "status")),
            if truth(get(gate, "detail")) {
                format!(" — {}", show(get(gate, "detail")))
            } else {
                String::new()
            }
        ));
    }
    if truth(get(value, "retry_semantics")) {
        lines.push(format!("Retry: {}", show(get(value, "retry_semantics"))));
    }
    lines.extend(next_lines(value));
    if !get(value, "execute_argv").is_null() {
        lines.push(format!(
            "Execute after review: {}",
            command(get(value, "execute_argv"))
                .unwrap_or_else(|| "Command unavailable; inspect JSON details.".into())
        ));
    }
    Ok(lines.join("\n"))
}
/// Workflow review with explicit state/config paths; never loads either file.
pub fn experiment_summary(
    value: &Value,
    state_file: &str,
    executed: bool,
    client_config: Option<&str>,
) -> Result<String> {
    root(value)?;
    let submission = object(get(value, "submission"));
    let preflight = object(get(submission, "preflight_summary"));
    let resources = object(get(preflight, "resources"));
    let mut lines = vec![
        format!(
            "Experiment: {}/{}",
            show(get(value, "project")),
            show(get(value, "run_id"))
        ),
        format!(
            "State: {}",
            text(&Value::String(state_file.into()), "UNKNOWN")
        ),
        format!(
            "Submission: {}",
            text(get(submission, "status"), "NOT_OBSERVED")
        ),
    ];
    if truth(resources) {
        lines.push(format!(
            "Resources: {}",
            ["gpus", "cpus", "memory_gb", "max_time"]
                .into_iter()
                .filter_map(|key| resources.get(key).map(|v| format!("{key}={}", show(v))))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !get(preflight, "requested_gpu_hours").is_null() {
        lines.push(format!(
            "GPU hours requested: {}",
            show(get(preflight, "requested_gpu_hours"))
        ));
    }
    for gate in gates(submission)
        .into_iter()
        .filter(|v| get(v, "status") != "PASS")
        .take(6)
    {
        lines.push(format!(
            "{}{}: {}{}",
            if member(get(gate, "status"), &["FAIL", "BLOCKED"])? {
                "Blocked: "
            } else {
                "Gate: "
            },
            show(get(gate, "name")),
            show(get(gate, "status")),
            if truth(get(gate, "detail")) {
                format!(" — {}", show(get(gate, "detail")))
            } else {
                String::new()
            }
        ));
    }
    if truth(get(submission, "attempt_id")) {
        lines.push(format!("Attempt: {}", show(get(submission, "attempt_id"))));
    }
    let result = object(get(value, "result"));
    if truth(result) {
        lines.push(format!(
            "Scheduler: {}",
            show(get(result, "scheduler_state"))
        ));
    }
    let proof = object(get(value, "download"));
    if truth(proof) {
        lines.push(format!(
            "Results: {}",
            text(
                proof
                    .get("verification")
                    .unwrap_or_else(|| get(proof, "status")),
                "downloaded; see verification.json"
            )
        ));
    }
    let mut argv = vec![Value::String("ml-exp".into())];
    if let Some(config) = client_config {
        argv.extend([
            Value::String("--config".into()),
            Value::String(config.into()),
        ]);
    }
    argv.extend(
        ["run", "status", "--state", state_file]
            .into_iter()
            .map(|v| Value::String(v.into())),
    );
    lines.push(format!(
        "Inspect: {}",
        command(&Value::Array(argv))
            .unwrap_or_else(|| "Command unavailable; inspect JSON details.".into())
    ));
    if !executed {
        lines.push("Training has not been authorized by this command. Review the saved preparation before --execute.".into());
    }
    Ok(lines.join("\n"))
}
