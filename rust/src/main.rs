use regex::Regex;
use serde_json::{Map, Value, json};
use std::env;
use std::io::{self, Read};
use std::process::ExitCode;
use std::sync::LazyLock;

const HELP: &str = "Sanitize SCO output and normalize SenseCore scheduler states.\n\
Only explicitly allowlisted job fields are emitted. Error and log text can be\n\
passed through redact-lines without exposing the raw response to a controller.\n\n\
Usage: experiment-safe-sco <MODE> [VALUE]\n\n\
Modes:\n\
  job-summary     Sanitize one JSON job object\n\
  job-list        Sanitize a JSON job array\n\
  worker-list     Sanitize the SCO worker table\n\
  redact-lines    Redact credential forms in text\n\
  normalize-state Normalize a SenseCore state value\n\n\
Options:\n\
  -h, --help      Print help";

static SECRET_ASSIGNMENT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(\b(?:secret|token|password|passwd|credential|access[_-]?key(?:[_-]?(?:id|secret))?|api[_-]?key|proxy|authorization|cookie)[\w.-]*\b[\s"']*[=:][\s"']*)([^\s,;"']+)"#,
    )
    .expect("secret assignment regex is valid")
});
static BEARER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(\b(?:authorization\s*:\s*)?bearer\s+)[^\s,;]+")
        .expect("bearer regex is valid")
});
static URL_USERINFO_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@").expect("URL userinfo regex is valid")
});
static SENSITIVE_QUERY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)([?&](?:access[_-]?key(?:[_-]?(?:id|secret))?|api[_-]?key|secret|token|signature)=)[^&#\s]+",
    )
    .expect("sensitive query regex is valid")
});
static WORKER_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z0-9._-]+$").expect("worker regex is valid"));
const STRUCTURED_EVIDENCE_PREFIX: &str = "EXPERIMENT_EVIDENCE_JSON=";
const STRUCTURED_EVIDENCE_MAX_CHARS: usize = 131_072;

fn redact_line(input: &str) -> String {
    let value = URL_USERINFO_RE.replace_all(input, "$1<redacted>@");
    let value = BEARER_RE.replace_all(&value, "$1<redacted>");
    let value = SENSITIVE_QUERY_RE.replace_all(&value, "$1<redacted>");
    SECRET_ASSIGNMENT_RE
        .replace_all(&value, "$1<redacted>")
        .into_owned()
}

fn normalized_key(key: &str) -> String {
    let mut normalized = String::with_capacity(key.len());
    let mut previous_was_lowercase_or_digit = false;
    for character in key.chars() {
        if character == '-' || character == '.' {
            normalized.push('_');
            previous_was_lowercase_or_digit = false;
            continue;
        }
        if character.is_ascii_uppercase() && previous_was_lowercase_or_digit {
            normalized.push('_');
        }
        normalized.push(character.to_ascii_lowercase());
        previous_was_lowercase_or_digit =
            character.is_ascii_lowercase() || character.is_ascii_digit();
    }
    normalized
}

fn sensitive_json_key(key: &str) -> bool {
    let key = normalized_key(key);
    let sensitive_suffix = [
        "token",
        "secret",
        "password",
        "passwd",
        "credential",
        "authorization",
        "cookie",
        "proxy",
        "signature",
        "api_key",
        "access_key",
        "access_key_id",
        "access_key_secret",
        "private_key",
    ]
    .iter()
    .any(|suffix| key == *suffix || key.ends_with(&format!("_{suffix}")));
    sensitive_suffix || key.starts_with("authorization_")
}

fn redact_json_value(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if sensitive_json_key(key) {
                    *child = Value::String("<redacted>".to_owned());
                } else {
                    redact_json_value(child);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_json_value(item);
            }
        }
        Value::String(text) => *text = redact_line(text),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn line_parts(line: &str) -> (&str, &str) {
    match line.strip_suffix('\n') {
        Some(body) => (body.strip_suffix('\r').unwrap_or(body), &line[body.len()..]),
        None => (line, ""),
    }
}

fn redact_lines(input: &str) -> String {
    let lines = input.split_inclusive('\n').collect::<Vec<_>>();
    let mut output = String::with_capacity(input.len());
    let mut index = 0;
    while index < lines.len() {
        let (body, newline) = line_parts(lines[index]);
        let Some(marker) = body.find(STRUCTURED_EVIDENCE_PREFIX) else {
            output.push_str(&redact_line(body));
            output.push_str(newline);
            index += 1;
            continue;
        };
        output.push_str(&redact_line(&body[..marker]));
        output.push_str(STRUCTURED_EVIDENCE_PREFIX);
        let mut payload = body[marker + STRUCTURED_EVIDENCE_PREFIX.len()..].to_owned();
        let mut final_newline = newline;
        loop {
            if payload.len() > STRUCTURED_EVIDENCE_MAX_CHARS {
                output.push_str("<redacted-malformed>");
                output.push_str(final_newline);
                return output;
            }
            match serde_json::from_str::<Value>(&payload) {
                Ok(mut value) if value.is_object() => {
                    redact_json_value(&mut value);
                    match serde_json::to_string(&value) {
                        Ok(payload) => output.push_str(&payload),
                        Err(_) => output.push_str("<redacted-malformed>"),
                    }
                    output.push_str(final_newline);
                    index += 1;
                    break;
                }
                Ok(_) => {
                    output.push_str("<redacted-malformed>");
                    output.push_str(final_newline);
                    return output;
                }
                Err(error) if error.is_eof() && index + 1 < lines.len() => {
                    index += 1;
                    let (fragment, newline) = line_parts(lines[index]);
                    payload.push_str(fragment);
                    final_newline = newline;
                }
                Err(_) => {
                    output.push_str("<redacted-malformed>");
                    output.push_str(final_newline);
                    return output;
                }
            }
        }
    }
    output
}

fn normalize_state(input: &str) -> &'static str {
    match input.to_ascii_uppercase().as_str() {
        "WAITING" | "INIT" | "QUEUEING" | "PENDING" | "CREATING" => "QUEUED",
        "STARTING" | "RECOVERING" => "STARTING",
        "RUNNING" | "RESTARTING" => "RUNNING",
        "SUCCEEDED" | "COMPLETED" => "SUCCEEDED",
        "SUSPENDING" | "SUSPENDED" => "PREEMPTED",
        "FAILED" | "ERROR" => "FAILED",
        "DELETING" | "DELETED" | "CANCELLED" | "CANCELED" => "CANCELLED",
        _ => "UNKNOWN",
    }
}

fn safe_value(value: Option<&Value>) -> Value {
    match value {
        Some(Value::String(text)) => Value::String(redact_line(text)),
        Some(value) => value.clone(),
        None => Value::Null,
    }
}

fn first_object(value: Option<&Value>) -> Option<&Map<String, Value>> {
    value?.as_array()?.first()?.as_object()
}

fn job_summary(job: &Map<String, Value>) -> Value {
    let first_role = first_object(job.get("roles"));
    let first_spec = first_object(first_role.and_then(|role| role.get("resource_spec")));
    let pool = job.get("resource_pool").and_then(Value::as_object);
    let mounts = job
        .get("mount")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .map(|mount| {
            json!({
                "id": safe_value(mount.get("id")),
                "subdir": safe_value(mount.get("subdir")),
                "mount_path": safe_value(mount.get("mount_path")),
            })
        })
        .collect::<Vec<_>>();
    let state = job.get("state");
    json!({
        "name": safe_value(job.get("name")),
        "display_name": safe_value(job.get("display_name")),
        "state": state.cloned().unwrap_or(Value::Null),
        "normalized_state": normalize_state(state.and_then(Value::as_str).unwrap_or("")),
        "create_time": job.get("create_time").cloned().unwrap_or(Value::Null),
        "pool": safe_value(pool.and_then(|value| value.get("name"))),
        "image": safe_value(first_role.and_then(|value| value.get("image_path"))),
        "spec": safe_value(first_spec.and_then(|value| value.get("name"))),
        "mounts": mounts,
    })
}

fn worker_list(input: &str) -> Result<Value, &'static str> {
    let rows = input
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('|'))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Ok(json!([]));
    }
    fn cells(line: &str) -> Vec<&str> {
        line.trim_matches('|').split('|').map(str::trim).collect()
    }
    let expected = ["WORKER_NAME", "RESOURCE", "HOST_IP", "POD_IP", "PHASE"];
    if cells(rows[0]) != expected {
        return Err("safe_sco: unexpected worker table schema; raw response suppressed");
    }
    let mut workers = Vec::new();
    for row in &rows[1..] {
        let row = cells(row);
        if row.len() != expected.len() {
            return Err("safe_sco: malformed worker table; raw response suppressed");
        }
        let [worker_name, resource, host_ip, pod_ip, phase] = row.as_slice() else {
            unreachable!("worker cell count was checked")
        };
        if worker_name.is_empty() {
            if !resource.is_empty() && host_ip.is_empty() && pod_ip.is_empty() && phase.is_empty() {
                continue;
            }
            return Err("safe_sco: malformed worker continuation; raw response suppressed");
        }
        if !WORKER_NAME_RE.is_match(worker_name) {
            return Err("safe_sco: unsafe worker identity; raw response suppressed");
        }
        workers.push(json!({
            "worker_name": worker_name,
            "phase": redact_line(phase),
        }));
    }
    Ok(Value::Array(workers))
}

fn parse_json(input: &str, empty_list: bool) -> Result<Value, &'static str> {
    let trimmed = input.trim();
    if empty_list && (trimmed.is_empty() || trimmed.eq_ignore_ascii_case("no jobs found")) {
        return Ok(json!([]));
    }
    serde_json::from_str(input)
        .map_err(|_| "safe_sco: input was not valid JSON; raw response suppressed")
}

fn sanitize(mode: &str, value: Option<&str>, input: &str) -> Result<String, &'static str> {
    match mode {
        "normalize-state" => value
            .map(|state| format!("{}\n", normalize_state(state)))
            .ok_or("safe_sco: normalize-state requires a state value"),
        "redact-lines" => Ok(redact_lines(input)),
        "worker-list" => serde_json::to_string(&worker_list(input)?)
            .map(|output| format!("{output}\n"))
            .map_err(|_| "safe_sco: could not serialize sanitized worker output"),
        "job-summary" => {
            let payload = parse_json(input, false)?;
            let job = payload
                .as_object()
                .ok_or("safe_sco: expected one JSON job object")?;
            serde_json::to_string(&job_summary(job))
                .map(|output| format!("{output}\n"))
                .map_err(|_| "safe_sco: could not serialize sanitized job output")
        }
        "job-list" => {
            let payload = parse_json(input, true)?;
            let jobs = payload
                .as_array()
                .ok_or("safe_sco: expected a JSON job array")?;
            let summaries = jobs
                .iter()
                .filter_map(Value::as_object)
                .map(job_summary)
                .collect::<Vec<_>>();
            serde_json::to_string(&summaries)
                .map(|output| format!("{output}\n"))
                .map_err(|_| "safe_sco: could not serialize sanitized job list")
        }
        _ => Err("safe_sco: unsupported sanitizer mode"),
    }
}

fn run() -> Result<(), &'static str> {
    let mut arguments = env::args().skip(1);
    let Some(mode) = arguments.next() else {
        eprintln!("{HELP}");
        return Err("safe_sco: a sanitizer mode is required");
    };
    if mode == "-h" || mode == "--help" {
        println!("{HELP}");
        return Ok(());
    }
    let value = arguments.next();
    if arguments.next().is_some() {
        return Err("safe_sco: too many arguments");
    }
    let input = match mode.as_str() {
        "normalize-state" => String::new(),
        "redact-lines" | "worker-list" | "job-summary" | "job-list" => {
            let mut input = String::new();
            io::stdin()
                .read_to_string(&mut input)
                .map_err(|_| "safe_sco: could not read standard input")?;
            input
        }
        _ => return Err("safe_sco: unsupported sanitizer mode"),
    };
    print!("{}", sanitize(&mode, value.as_deref(), &input)?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_every_supported_credential_form() {
        let output = redact_line(
            "token=alpha Authorization: Bearer bravo proxy=https://user:pass@example.test/?signature=charlie",
        );
        for secret in ["alpha", "bravo", "user:pass", "charlie"] {
            assert!(!output.contains(secret));
        }
        assert_eq!(output.matches("<redacted>").count(), 4);
    }

    #[test]
    fn structured_evidence_preserves_scientific_token_fields() {
        let input = concat!(
            "2026-07-16T12:00:00Z EXPERIMENT_EVIDENCE_JSON=",
            r#"{"token_recon_ppl":23.3,"oracle_plan_token_denoising_l2":2.1,"sampled_plan_num_samples":16,"tokenizer_path":"/data/tokenizer","proxy_loss":0.4}"#,
            "\n"
        );
        let output = sanitize("redact-lines", None, input).expect("valid evidence");
        let payload = output
            .trim_end()
            .split_once(STRUCTURED_EVIDENCE_PREFIX)
            .unwrap()
            .1;
        let evidence: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(evidence["token_recon_ppl"], 23.3);
        assert_eq!(evidence["oracle_plan_token_denoising_l2"], 2.1);
        assert_eq!(evidence["sampled_plan_num_samples"], 16);
        assert_eq!(evidence["tokenizer_path"], "/data/tokenizer");
        assert_eq!(evidence["proxy_loss"], 0.4);
    }

    #[test]
    fn structured_evidence_redacts_sensitive_keys_and_string_values() {
        let input = concat!(
            "EXPERIMENT_EVIDENCE_JSON=",
            r#"{"submission_token":"alpha","refreshToken":"bravo","WANDB_API_KEY":"echo","nested":{"access_key_secret":"foxtrot","metric_url":"https://user:pass@example.test/?token=charlie"},"message":"Authorization: Bearer delta"}"#,
            "\n"
        );
        let output = sanitize("redact-lines", None, input).expect("valid evidence");
        for secret in [
            "alpha",
            "bravo",
            "echo",
            "foxtrot",
            "user:pass",
            "charlie",
            "delta",
        ] {
            assert!(!output.contains(secret));
        }
        let payload = output
            .trim_end()
            .strip_prefix(STRUCTURED_EVIDENCE_PREFIX)
            .unwrap();
        let evidence: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(evidence["submission_token"], "<redacted>");
        assert_eq!(evidence["refreshToken"], "<redacted>");
        assert_eq!(evidence["WANDB_API_KEY"], "<redacted>");
        assert_eq!(evidence["nested"]["access_key_secret"], "<redacted>");
    }

    #[test]
    fn malformed_structured_evidence_is_suppressed() {
        let secret = "must-not-echo";
        let input = format!("{STRUCTURED_EVIDENCE_PREFIX}{{\"token\":\"{secret}\"\n");
        let output = sanitize("redact-lines", None, &input).expect("fail closed output");
        assert_eq!(
            output,
            format!("{STRUCTURED_EVIDENCE_PREFIX}<redacted-malformed>\n")
        );
        assert!(!output.contains(secret));
    }

    #[test]
    fn fragmented_structured_evidence_is_reassembled_without_physical_newlines() {
        let input = concat!(
            "before\n",
            "EXPERIMENT_EVIDENCE_JSON={\"run_id\":\"long-\n",
            "run\",\"token_recon_ppl\":\n",
            "23.3}\n",
            "after token=alpha\n"
        );
        let output = sanitize("redact-lines", None, input).expect("fragmented evidence");
        let lines = output.lines().collect::<Vec<_>>();
        assert_eq!(lines[0], "before");
        let payload = lines[1].strip_prefix(STRUCTURED_EVIDENCE_PREFIX).unwrap();
        let evidence: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(evidence["run_id"], "long-run");
        assert_eq!(evidence["token_recon_ppl"], 23.3);
        assert_eq!(lines[2], "after token=<redacted>");
    }

    #[test]
    fn job_summary_is_allowlisted() {
        let input = r#"{"name":"job","state":"SUSPENDED","token":"secret","roles":[{"image_path":"https://user:pass@registry/image"}]}"#;
        let output = sanitize("job-summary", None, input).expect("valid job");
        assert!(!output.contains("secret"));
        assert!(!output.contains("user:pass"));
        assert_eq!(
            serde_json::from_str::<Value>(&output).unwrap()["normalized_state"],
            "PREEMPTED"
        );
    }

    #[test]
    fn worker_parser_accepts_resource_continuation() {
        let input = "| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |\n\
                     | worker-0 | 4 GPUs | 1 | 2 | Running |\n\
                     | | 56 CPUs | | | |\n";
        let output = worker_list(input).expect("valid table");
        assert_eq!(output[0]["worker_name"], "worker-0");
        assert_eq!(output[0]["phase"], "Running");
    }

    #[test]
    fn malformed_input_fails_without_echoing_it() {
        let secret = "must-not-echo";
        let error = sanitize("job-summary", None, secret).unwrap_err();
        assert!(!error.contains(secret));
        assert!(error.contains("raw response suppressed"));
    }

    #[test]
    fn normalizes_known_and_unknown_states() {
        assert_eq!(normalize_state("completed"), "SUCCEEDED");
        assert_eq!(normalize_state("future"), "UNKNOWN");
    }
}
