//! Golden canonical bytes and the determinism that digests and sealed manifests rely on.
use ml_exp_core::canonical;
use serde_json::{Value, json};

/// Parsed from text, because a Rust float literal could not carry these spellings.
const HARD_FLOATS: &[&str] = &[
    "5e-324",
    "2.225073858507201e-308",
    "2.2250738585072014e-308",
    "1.7976931348623157e308",
    "0.30000000000000004",
    "1.2345678901234567",
    "9007199254740992.0",
    "8.41e21",
    "1e23",
    "1e-7",
    "123456.789",
];

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/canonical-vectors.json")).unwrap()
}

#[test]
fn golden_vectors_give_exact_bytes_and_digests() {
    let fixture = fixture();
    let vectors = fixture["vectors"].as_array().unwrap();
    assert!(vectors.len() > 40);
    for vector in vectors {
        let input = vector["input"].as_str().unwrap();
        let expected = vector["expected"].as_str().unwrap();
        let sha256 = vector["sha256"].as_str().unwrap();
        let value: Value = serde_json::from_str(input).unwrap();
        assert_eq!(canonical::text(&value).unwrap(), expected, "{input}");
        assert_eq!(canonical::bytes(&value).unwrap(), expected.as_bytes());
        assert_eq!(canonical::digest(&value).unwrap(), sha256, "{input}");
        assert_eq!(
            canonical::protocol_identity(&value).unwrap(),
            format!("sha256:{sha256}")
        );
    }
    for input in fixture["refused"].as_array().unwrap() {
        let value: Value = serde_json::from_str(input.as_str().unwrap()).unwrap();
        assert!(canonical::bytes(&value).is_err(), "{input}");
    }
}

#[test]
fn key_order_never_reaches_the_bytes() {
    let orders = [
        r#"{"b":1,"a":{"y":[1,{"q":1.5,"p":"模型"}],"x":null},"é":true}"#,
        r#"{"é":true,"a":{"x":null,"y":[1,{"p":"模型","q":1.5}]},"b":1}"#,
        r#"{"a":{"y":[1,{"q":1.5,"p":"模型"}],"x":null},"é":true,"b":1}"#,
    ];
    let values: Vec<Value> = orders
        .iter()
        .map(|text| serde_json::from_str(text).unwrap())
        .collect();
    // preserve_order keeps the differences visible to an order-sensitive encoder.
    assert_ne!(
        serde_json::to_vec(&values[0]).unwrap(),
        serde_json::to_vec(&values[1]).unwrap()
    );
    for value in &values {
        assert_eq!(
            canonical::text(value).unwrap(),
            r#"{"a":{"x":null,"y":[1,{"p":"模型","q":1.5}]},"b":1,"é":true}"#
        );
    }
}

/// launch::verify_binding re-encodes the sealed bytes and refuses any difference.
#[test]
fn encoding_is_a_fixed_point_through_a_parse() {
    let fixture = fixture();
    let inputs = fixture["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|vector| vector["input"].as_str().unwrap())
        .chain(HARD_FLOATS.iter().copied());
    for input in inputs {
        let value: Value = serde_json::from_str(input).unwrap();
        let bytes = canonical::bytes(&value).unwrap();
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(canonical::bytes(&parsed).unwrap(), bytes, "{input}");
    }
}

/// max_gpu_hours is re-hashed on a money path, so a float cannot move through JSON.
#[test]
fn hard_floats_round_trip_through_every_json_route() {
    for text in HARD_FLOATS {
        let number: f64 = text.parse().unwrap();
        let canonical_text = canonical::float_text(number).unwrap();
        assert_eq!(
            canonical_text.parse::<f64>().unwrap().to_bits(),
            number.to_bits(),
            "{text}"
        );
        let typed: f64 = serde_json::from_slice(&serde_json::to_vec(&number).unwrap()).unwrap();
        assert_eq!(typed.to_bits(), number.to_bits(), "{text}");
        let parsed: f64 = serde_json::from_str(&canonical_text).unwrap();
        assert_eq!(parsed.to_bits(), number.to_bits(), "{text}");
        let value: Value = serde_json::from_slice(text.as_bytes()).unwrap();
        assert_eq!(
            value.as_f64().unwrap().to_bits(),
            number.to_bits(),
            "{text}"
        );
        assert_eq!(canonical::text(&value).unwrap(), canonical_text);
        assert_eq!(canonical::text(&json!(number)).unwrap(), canonical_text);
    }
}

#[test]
fn integers_and_floats_stay_distinct_and_nonfinite_values_refuse() {
    assert_eq!(canonical::text(&json!(1)).unwrap(), "1");
    assert_eq!(canonical::text(&json!(1.0)).unwrap(), "1.0");
    assert_eq!(canonical::text(&json!(-0.0)).unwrap(), "0.0");
    assert_ne!(
        canonical::digest(&json!({"lr":1})).unwrap(),
        canonical::digest(&json!({"lr":1.0})).unwrap()
    );
    for number in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(canonical::float_text(number).is_err());
    }
}
