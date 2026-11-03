use ml_exp_core::diagnostic::{self, HttpFailure};
use serde_json::{Value, json};

#[test]
fn bounded_http_diagnostics_match_frozen_python_reference() {
    let x: Value =
        serde_json::from_str(include_str!("../../../compatibility/diagnostic.json")).unwrap();
    let credential = x["credential"].as_str().unwrap();
    for (i, c) in x["text"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            diagnostic::text(c["input"].as_str().unwrap(), credential),
            c["expected"],
            "text {i}"
        );
    }
    for (i, c) in x["json"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            diagnostic::json(&c["input"], credential),
            c["expected"],
            "json {i}"
        );
    }
    for (i, c) in x["retry"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            serde_json::to_value(diagnostic::retry_after(
                c["input"].as_str(),
                x["now_unix"].as_i64().unwrap()
            ))
            .unwrap(),
            c["expected"],
            "retry {i}"
        );
    }
    for (i, c) in x["http"].as_array().unwrap().iter().enumerate() {
        let failure = HttpFailure::from_response(
            c["status"].as_u64().unwrap() as u16,
            c["headers"]["X-ML-Expd-Error-Code"].as_str(),
            c["headers"]["Retry-After"].as_str(),
            c["body"].as_str().unwrap().as_bytes(),
            credential,
            x["now_unix"].as_i64().unwrap(),
        );
        assert_eq!(
            serde_json::to_value(failure).unwrap(),
            c["expected"],
            "http {i}"
        );
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
