//! The client-only platform jobs run ml-exp-client and ml-exp-io, not ml-exp-core,
//! so the golden vectors also run here on every client target.
use ml_exp_core::canonical;
use serde_json::Value;

#[test]
fn golden_vectors_give_exact_bytes_and_digests_on_this_platform() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../core/tests/fixtures/canonical-vectors.json"
    ))
    .unwrap();
    let vectors = fixture["vectors"].as_array().unwrap();
    assert!(vectors.len() > 40);
    for vector in vectors {
        let input = vector["input"].as_str().unwrap();
        let value: Value = serde_json::from_str(input).unwrap();
        assert_eq!(
            canonical::text(&value).unwrap(),
            vector["expected"].as_str().unwrap(),
            "{input}"
        );
        assert_eq!(
            canonical::digest(&value).unwrap(),
            vector["sha256"].as_str().unwrap(),
            "{input}"
        );
    }
    for input in fixture["refused"].as_array().unwrap() {
        let value: Value = serde_json::from_str(input.as_str().unwrap()).unwrap();
        assert!(canonical::bytes(&value).is_err(), "{input}");
    }
}
