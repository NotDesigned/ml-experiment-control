use ml_exp_core::diagnostic::{self, HttpFailure};
use serde_json::{Value, json};

const CREDENTIAL: &str = "synthetic-credential-ZZZ";
const NOW_UNIX: i64 = 1_700_000_000;

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

#[test]
fn text_redacts_credentials_assignments_urls_and_control_characters() {
    for (input, expected) in [
        ("plain failure", "plain failure"),
        ("secret=synthetic-credential-ZZZ", "secret=[REDACTED]"),
        ("Bearer synthetic-credential-ZZZ", "Bearer [REDACTED]"),
        ("api_key=\"a long quoted secret\"", "api_key=[REDACTED]"),
        ("https://some.test/a?signature=signed-secret", "[REDACTED]"),
        (
            "-----BEGIN RSA PRIVATE KEY-----\nsecret\n-----END RSA PRIVATE KEY-----",
            "[REDACTED]",
        ),
        ("line\r\n\u{0}\tbreak", "line    break"),
    ] {
        assert_eq!(diagnostic::text(input, CREDENTIAL), expected);
    }
}

#[test]
fn json_redacts_sensitive_keys_truncates_depth_and_keeps_scientific_fields() {
    for (input, expected) in [
        (
            r#"{"detail":{"code":"NO_SPACE","required":100,"available":50}}"#,
            r#"{"detail":{"code":"NO_SPACE","required":100,"available":50}}"#,
        ),
        (
            r#"{"Authorization":"Bearer secret","nested":{"api_url":"https://private.test","input":{"secret":"no"}}}"#,
            r#"{"Authorization":"[REDACTED]","nested":{"api_url":"[REDACTED]","input":"[REDACTED]"}}"#,
        ),
        (
            r#"{"token_count":77,"message":"BearER abc; stable"}"#,
            r#"{"token_count":"[REDACTED]","message":"Bearer [REDACTED]; stable"}"#,
        ),
        (
            r#"{"depth":[[[[[[[[{"name":"deep"}]]]]]]]]}"#,
            r#"{"depth":[[[[[["[TRUNCATED]"]]]]]]}"#,
        ),
        (
            r#"{"detail":"credential=synthetic-credential-ZZZ"}"#,
            r#"{"detail":"credential=[REDACTED]"}"#,
        ),
    ] {
        assert_eq!(
            diagnostic::json(&literal(input), CREDENTIAL),
            literal(expected)
        );
    }
}

#[test]
fn retry_after_accepts_seconds_and_http_dates_and_caps_or_rejects_the_rest() {
    assert_eq!(diagnostic::retry_after(Some("60"), NOW_UNIX), Some(60));
    assert_eq!(
        diagnostic::retry_after(Some("999999999999999999999999"), NOW_UNIX),
        Some(86400)
    );
    assert_eq!(
        diagnostic::retry_after(Some("Tue, 14 Nov 2023 22:13:30 GMT"), NOW_UNIX),
        Some(10)
    );
    assert_eq!(
        diagnostic::retry_after(Some("Tue, 14 Nov 2023 20:13:30 +0000"), NOW_UNIX),
        Some(0)
    );
    assert_eq!(diagnostic::retry_after(Some("-1"), NOW_UNIX), None);
    assert_eq!(diagnostic::retry_after(Some("nope"), NOW_UNIX), None);
    assert_eq!(diagnostic::retry_after(None, NOW_UNIX), None);
}

#[test]
fn http_failures_keep_status_and_safe_code_and_drop_unparseable_bodies() {
    for (status, code, retry, body, expected) in [
        (
            409,
            None,
            None,
            "{\"detail\": {\"code\": \"NO_SPACE\", \"required\": 100, \"available\": 50}}",
            r#"{"status":409,"retryable":false,"error_code":"API_ERROR","retry_after":null,"details":{"code":"NO_SPACE","required":100,"available":50},"response_body":{"detail":{"code":"NO_SPACE","required":100,"available":50}},"body_truncated":false}"#,
        ),
        (
            422,
            Some("INPUT_NOT_READY"),
            Some("10"),
            "{\"detail\":{\"phase\":\"DATA\",\"url\":\"https://private.test\"},\"diagnostic\":\"WAIT\"}",
            r#"{"status":422,"retryable":false,"error_code":"INPUT_NOT_READY","retry_after":10,"details":{"phase":"DATA","url":"[REDACTED]"},"response_body":{"detail":{"phase":"DATA","url":"[REDACTED]"},"diagnostic":"WAIT"},"body_truncated":false}"#,
        ),
        (
            507,
            Some("synthetic-credential-ZZZ"),
            None,
            "{\"detail\":\"disk full\"}",
            r#"{"status":507,"retryable":false,"error_code":"API_ERROR","retry_after":null,"details":"disk full","response_body":{"detail":"disk full"},"body_truncated":false}"#,
        ),
        (
            502,
            Some("bad"),
            None,
            "<html>do not retain</html>",
            r#"{"status":502,"retryable":false,"error_code":"API_ERROR","retry_after":null,"details":null,"response_body":null,"body_truncated":false}"#,
        ),
    ] {
        let failure =
            HttpFailure::from_response(status, code, retry, body.as_bytes(), CREDENTIAL, NOW_UNIX);
        assert_eq!(serde_json::to_value(failure).unwrap(), literal(expected));
    }
}

#[test]
fn nonstandard_or_unparseable_bodies_do_not_escape_as_plaintext_and_shared_policy_is_stricter() {
    for body in [
        br#"{"detail":NaN}"#.as_slice(),
        br#"{"detail":"\ud800"}"#,
        b"<html>Bearer secret</html>",
    ] {
        let value = HttpFailure::from_response(502, None, None, body, "secret", 0);
        assert_eq!(value.details, None);
        assert_eq!(value.response_body, None);
        assert!(!value.to_string().contains("secret"));
    }
    assert_eq!(
        diagnostic::text("proxy=secret passwd='secret'", ""),
        "proxy=[REDACTED] passwd=[REDACTED]"
    );
    let value = diagnostic::json(
        &json!({"n":serde_json::from_str::<Value>("1e400").unwrap(),"large":serde_json::from_str::<Value>("123456789012345678901234567890").unwrap()}),
        "",
    );
    assert_eq!(value["n"], Value::Null);
    assert_eq!(value["large"].to_string(), "123456789012345678901234567890");
    assert_eq!(diagnostic::retry_after(Some("١٢"), 0), None);
}

#[test]
fn shared_sensitive_keys_quoted_credentials_and_numeric_token_echo_never_enter_diagnostics() {
    let credential = "1234567890123456789012345678901234567890123456789012345678901234";
    let body = format!(
        r#"{{"detail":{{"passwd":"ssh-password-synthetic","proxy":"proxy-password-synthetic","custom_proxy":"opaque secret","value":{credential},"other":123456789012345678901234567890}}}}"#
    );
    let failure = HttpFailure::from_response(409, None, None, body.as_bytes(), credential, 0);
    let serialized = serde_json::to_string(&failure).unwrap();
    let debug = format!("{failure:?}");
    for forbidden in [
        credential,
        "ssh-password-synthetic",
        "proxy-password-synthetic",
        "opaque secret",
    ] {
        assert!(!serialized.contains(forbidden));
        assert!(!debug.contains(forbidden));
    }
    assert_eq!(
        failure.details.as_ref().unwrap()["other"].to_string(),
        "123456789012345678901234567890"
    );
    for assignment in [
        "passwd='a long quoted secret'",
        "proxy=\"a long quoted secret\"",
    ] {
        let result = diagnostic::text(assignment, "");
        assert!(!result.contains("long quoted secret"));
        assert!(result.contains("[REDACTED]"));
    }
}
