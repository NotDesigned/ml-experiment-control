use ml_exp_core::{canonical::python_json_bytes, launch::*};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[test]
fn sealed_launcher_encoding_matches_python() {
    let reference: Value =
        serde_json::from_str(include_str!("../../../compatibility/launch.json")).unwrap();
    for (index, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        match encode(&case["input"]) {
            Ok(bytes) => assert_eq!(
                std::str::from_utf8(&bytes).unwrap(),
                case["expected"].as_str().unwrap(),
                "case {index}"
            ),
            Err(error) => assert_eq!(
                error.to_string(),
                case["error"].as_str().unwrap(),
                "case {index}"
            ),
        }
    }
}

#[test]
fn launch_endpoint_capability_and_frozen_body_are_exact() {
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
    let mut noncanonical = python_json_bytes(&document).unwrap();
    noncanonical.push(b' ');
    let mut matching = target;
    matching.sha256 = format!("{:x}", Sha256::digest(&noncanonical));
    assert!(verify(&noncanonical, &matching).is_err());
    for token in [
        "",
        "token with space",
        "token\n",
        "token:bad",
        "token/other",
    ] {
        assert!(validate_capability(token).is_err());
    }
    assert!(validate_capability("test-token_123").is_ok());
}
