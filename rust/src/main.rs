use regex::Regex;
use serde_json::Value;
use std::env;
use std::io::{self, Read};
use std::process::ExitCode;
use std::sync::LazyLock;

const HELP: &str = "Redact credentials in experiment log text from standard input.\n\nUsage: experiment-redact\n\nOptions:\n  -h, --help      Print help";

static SECRET_ASSIGNMENT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(\b(?:[\w.-]*[_-])?(?:secret|token|password|passwd|credential|access[_-]?key(?:[_-]?(?:id|secret))?|api[_-]?key|private[_-]?key|proxy|authorization|cookie)[\w.-]*\b[\s"']*[=:][\s"']*)([^\s,;"']+)"#,
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
        r"(?i)([?&](?:access[_-]?key(?:[_-]?(?:id|secret))?|api[_-]?key|secret|token|signature|x-amz-(?:credential|signature|security-token))=)[^&#\s]+",
    )
    .expect("sensitive query regex is valid")
});
const STRUCTURED_EVIDENCE_PREFIX: &str = "EXPERIMENT_EVIDENCE_JSON=";
const STRUCTURED_EVIDENCE_MAX_CHARS: usize = 1_048_576;

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

fn run() -> Result<(), &'static str> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments == ["-h"] || arguments == ["--help"] {
        println!("{HELP}");
        return Ok(());
    }
    if !arguments.is_empty() {
        return Err("experiment-redact: unexpected arguments; input suppressed");
    }
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|_| "experiment-redact: could not read standard input")?;
    print!("{}", redact_lines(&input));
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
        let output = redact_lines(input);
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
        let output = redact_lines(input);
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
        let output = redact_lines(&input);
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
        let output = redact_lines(input);
        let lines = output.lines().collect::<Vec<_>>();
        assert_eq!(lines[0], "before");
        let payload = lines[1].strip_prefix(STRUCTURED_EVIDENCE_PREFIX).unwrap();
        let evidence: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(evidence["run_id"], "long-run");
        assert_eq!(evidence["token_recon_ppl"], 23.3);
        assert_eq!(lines[2], "after token=<redacted>");
    }

    #[test]
    fn structured_evidence_can_exceed_the_legacy_128_kib_limit() {
        let payload = "x".repeat(131_073);
        let input = format!("{STRUCTURED_EVIDENCE_PREFIX}{{\"large_summary\":\"{payload}\"}}\n");
        let output = redact_lines(&input);
        let encoded = output
            .trim_end()
            .strip_prefix(STRUCTURED_EVIDENCE_PREFIX)
            .unwrap();
        let evidence: Value = serde_json::from_str(encoded).unwrap();
        assert_eq!(evidence["large_summary"].as_str().unwrap().len(), 131_073);
    }

    #[test]
    fn oversized_structured_evidence_is_suppressed() {
        let payload = "x".repeat(STRUCTURED_EVIDENCE_MAX_CHARS + 1);
        let input = format!("{STRUCTURED_EVIDENCE_PREFIX}{{\"large_summary\":\"{payload}\"}}\n");
        let output = redact_lines(&input);
        assert_eq!(
            output,
            format!("{STRUCTURED_EVIDENCE_PREFIX}<redacted-malformed>\n")
        );
    }
}
