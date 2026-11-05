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
fn training_line(training: &Value) -> Result<String> {
    let mut parts = vec![
        show(get(training, "status")),
        format!(
            "process {}",
            text(get(training, "process_state"), "not observed")
        ),
    ];
    if !get(training, "exit_code").is_null() {
        parts.push(format!("exit {}", show(get(training, "exit_code"))));
    }
    let observed = observed(training)?;
    if !observed.is_empty() {
        parts.push(format!("process evidence: {observed}"));
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
        && verified == "SHA256_VERIFIED"
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
        "error_class",
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
    lines.push(training_line(training)?);
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

/// Provider text is shown as evidence, not as a conclusion about the training code.
pub fn provider_diagnostics(value: &Value) -> String {
    let word = |value: &Value| value.as_str().unwrap_or("not reported").to_owned();
    let mut lines = vec![format!(
        "Run: {}/{} · {} · {}",
        word(&value["project"]),
        word(&value["run_id"]),
        word(&value["attempt_id"]),
        word(&value["backend"])
    )];
    if !value["diagnostic"].is_null() {
        lines.push(format!("Provider not read: {}", word(&value["diagnostic"])));
        return lines.join("\n");
    }
    let provider = &value["provider"];
    lines.push(format!(
        "Provider job: {} · {}{}",
        word(&provider["job_id"]),
        word(&provider["provider_state"]),
        provider["sub_status"]
            .as_str()
            .filter(|sub| !sub.is_empty())
            .map(|sub| format!(" ({sub})"))
            .unwrap_or_default()
    ));
    lines.push(format!(
        "Reason: {} {}",
        word(&provider["reason_code"]),
        provider["reason_message"].as_str().unwrap_or_default()
    ));
    let list = |value: &Value| value.as_array().cloned().unwrap_or_default();
    for pod in list(&provider["pods"]) {
        lines.push(format!(
            "Pod {} ({}): {}",
            word(&pod["pod_id"]),
            word(&pod["type"]),
            word(&pod["status"])
        ));
        if !pod["logs_error"].is_null() {
            lines.push(format!("  logs unavailable: {}", pod["logs_error"]));
        }
        for line in list(&pod["logs"]) {
            lines.push(format!("  {}", line.as_str().unwrap_or_default()));
        }
    }
    if provider["pods_total"].as_u64().unwrap_or(0) as usize > list(&provider["pods"]).len() {
        lines.push(format!("Pods shown: first of {}", provider["pods_total"]));
    }
    if !provider["events_error"].is_null() {
        lines.push(format!("Events unavailable: {}", provider["events_error"]));
    }
    for event in list(&provider["events"]) {
        lines.push(format!("Event: {}", event.as_str().unwrap_or_default()));
    }
    lines.join("\n")
}
