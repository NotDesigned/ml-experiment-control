use ml_exp_core::launch::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CONTRACT_ERROR: &str = "invalid launch manifest contract";
const IDENTITY_ERROR: &str = "invalid launch manifest identity";
const ENVIRONMENT_ERROR: &str = "invalid launch manifest environment";
const ARGV_ERROR: &str = "invalid launch manifest argv";
const WORKDIR_ERROR: &str = "invalid launch manifest workdir";
const TIMEOUT_ERROR: &str = "invalid launch manifest timeout";

fn manifest() -> Value {
    json!({"contract":CONTRACT,"identity":{"project":"p","run_id":"r","attempt_id":"attempt-001"},
        "environment":{"OUTPUT_DIR":"/output","MODEL":"λ😀"},"argv":["--","python","train.py"],
        "workdir":"/app","timeout_seconds":300})
}
fn with(path: &[&str], value: Value) -> Value {
    let mut document = manifest();
    let (last, parents) = path.split_last().unwrap();
    let mut target = &mut document;
    for key in parents {
        target = &mut target[*key];
    }
    target[*last] = value;
    document
}
fn without(key: &str) -> Value {
    let mut document = manifest();
    document.as_object_mut().unwrap().remove(key);
    document
}
fn refused(document: &Value, message: &str) {
    assert_eq!(
        encode(document).unwrap_err().to_string(),
        message,
        "{document}"
    );
}
fn accepted(document: &Value) {
    let bytes = encode(document).unwrap();
    assert_eq!(&serde_json::from_slice::<Value>(&bytes).unwrap(), document);
    assert_eq!(encode(document).unwrap(), bytes);
}

#[test]
fn an_accepted_ascii_manifest_encodes_to_sorted_compact_bytes() {
    accepted(&manifest());
    let document = with(&["environment"], json!({"OUTPUT_DIR":"/output"}));
    assert_eq!(
        std::str::from_utf8(&encode(&document).unwrap()).unwrap(),
        r#"{"argv":["--","python","train.py"],"contract":"launcher-manifest.v1","environment":{"OUTPUT_DIR":"/output"},"identity":{"attempt_id":"attempt-001","project":"p","run_id":"r"},"timeout_seconds":300,"workdir":"/app"}"#
    );
}

#[test]
fn each_field_refuses_a_missing_or_wrongly_typed_value_with_its_own_error() {
    for (field, error) in [
        ("contract", CONTRACT_ERROR),
        ("identity", IDENTITY_ERROR),
        ("environment", ENVIRONMENT_ERROR),
        ("argv", ARGV_ERROR),
        ("workdir", WORKDIR_ERROR),
        ("timeout_seconds", TIMEOUT_ERROR),
    ] {
        refused(&without(field), CONTRACT_ERROR);
        for value in [json!(null), json!(false), json!(""), json!([]), json!({})] {
            refused(&with(&[field], value), error);
        }
        if field != "timeout_seconds" {
            refused(&with(&[field], json!(1)), error);
        }
    }
    refused(&with(&["extra"], json!(true)), CONTRACT_ERROR);
}

#[test]
fn timeout_is_a_whole_number_of_seconds_within_its_range() {
    for value in [1, 3_599_999] {
        accepted(&with(&["timeout_seconds"], json!(value)));
    }
    let too_large: Value = serde_json::from_str("18446744073709551616").unwrap();
    for value in [
        json!(-1),
        json!(0),
        json!(3_600_000),
        json!(3.0),
        json!(true),
        json!("3"),
        too_large,
    ] {
        refused(&with(&["timeout_seconds"], value), TIMEOUT_ERROR);
    }
}

#[test]
fn identity_fields_are_validated_and_the_attempt_has_its_own_format() {
    for field in ["project", "run_id", "attempt_id"] {
        for value in [
            json!(""),
            json!(null),
            json!(false),
            json!("../x"),
            json!("模型"),
            json!("x".repeat(129)),
        ] {
            refused(&with(&["identity", field], value), IDENTITY_ERROR);
        }
        accepted(&with(&["identity", field], json!("attempt-99999")));
    }
    accepted(&with(&["identity", "project"], json!("attempt-01")));
    accepted(&with(&["identity", "run_id"], json!("attempt-01")));
    refused(
        &with(&["identity", "attempt_id"], json!("attempt-01")),
        IDENTITY_ERROR,
    );
    refused(&with(&["identity", "extra"], json!(true)), IDENTITY_ERROR);
}

#[test]
fn environment_refuses_bad_names_reserved_tokens_and_non_text_values() {
    let values = || {
        [
            json!(""),
            json!(null),
            json!(0),
            json!("secret"),
            json!("\0"),
            json!("α"),
        ]
    };
    for key in [
        "",
        "1KEY",
        "bad-key",
        "ML_EXPD_UPLOAD_TOKEN",
        "ML_EXPD_BOOTSTRAP_TOKEN",
    ] {
        for value in values() {
            refused(&with(&["environment", key], value), ENVIRONMENT_ERROR);
        }
    }
    for key in ["GOOD_KEY", "Z"] {
        for value in values() {
            let document = with(&["environment", key], value.clone());
            if value.as_str().is_some_and(|text| !text.contains('\0')) {
                accepted(&document);
            } else {
                refused(&document, ENVIRONMENT_ERROR);
            }
        }
    }
    for path in [json!("relative"), json!("/bad\0"), json!(null)] {
        refused(
            &with(&["environment", "OUTPUT_DIR"], path),
            ENVIRONMENT_ERROR,
        );
    }
    for path in ["/", "/mnt/checkpoints"] {
        accepted(&with(&["environment", "OUTPUT_DIR"], json!(path)));
    }
}

#[test]
fn argv_workdir_and_size_are_bounded() {
    for argv in [json!([""]), json!(["echo", "\0"]), json!(["echo", null])] {
        refused(&with(&["argv"], argv), ARGV_ERROR);
    }
    for argv in [json!(["echo", ""]), json!(["echo", "α😀"])] {
        accepted(&with(&["argv"], argv));
    }
    refused(
        &with(&["argv"], json!(["a".repeat(270_000)])),
        "launch manifest exceeds size limit",
    );
    for workdir in ["relative", "/bad\0"] {
        refused(&with(&["workdir"], json!(workdir)), WORKDIR_ERROR);
    }
    for workdir in ["/", "/mnt/checkpoints"] {
        accepted(&with(&["workdir"], json!(workdir)));
    }
}

#[test]
fn launch_endpoint_and_frozen_body_are_exact() {
    let document = json!({"contract":CONTRACT,"identity":{"project":"p","run_id":"r","attempt_id":"attempt-001"},
        "environment":{"OUTPUT_DIR":"/output"},"argv":["--","python","train.py"],"workdir":"/app","timeout_seconds":300});
    let bytes = encode(&document).unwrap();
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let url = format!("https://example.test/ml-expd/api/launch-transfers/p/r/attempt-001/{digest}");
    let target = endpoint(&url).unwrap();
    assert_eq!(verify(&bytes, &target).unwrap().encode().unwrap(), bytes);
    for bad in [
        url.replace("https:", "http:"),
        url.replace("example.test", "user:secret@example.test"),
        url.replace("example.test", "@example.test"),
        format!("{url}?"),
        format!("{url}#fragment"),
        url.replace("/ml-expd/", "/../"),
        url.replace("/ml-expd/", "/%2e%2e/"),
        url.replace("/attempt-001/", "/attempt-01/"),
        url.replace(&digest, "ABCDEF"),
        format!(" {url}"),
    ] {
        assert!(endpoint(&bad).is_err(), "bad endpoint accepted");
    }
    let mut foreign = target.clone();
    foreign.identity.run_id = "other".into();
    assert!(verify(&bytes, &foreign).is_err());
    assert!(verify(b"wrong", &target).is_err());
    let mut noncanonical = bytes.clone();
    noncanonical.push(b' ');
    let mut matching = target;
    matching.sha256 = format!("{:x}", Sha256::digest(&noncanonical));
    assert!(verify(&noncanonical, &matching).is_err());
}
