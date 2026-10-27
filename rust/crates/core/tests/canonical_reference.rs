use ml_exp_core::canonical::{JsonOptions, python_float, python_json_from_slice};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    floats: Vec<Float>,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Float {
    bits: String,
    expected: String,
}
#[derive(Deserialize)]
struct Options {
    sort_keys: bool,
    ensure_ascii: bool,
    compact: bool,
}
#[derive(Deserialize)]
struct Case {
    raw: String,
    options: Options,
    expected: Option<String>,
    sha256: Option<String>,
    error: bool,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../../../compatibility/canonical.json")).unwrap()
}
#[test]
fn python_shortest_float_repr_matches_ieee_patterns() {
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.floats.len(), 10012);
    for f in fixture.floats {
        let bits = u64::from_str_radix(&f.bits, 16).unwrap();
        let value = f64::from_bits(bits);
        assert_eq!(
            python_float(value).unwrap(),
            f.expected,
            "IEEE f64 bits {}",
            f.bits
        );
    }
}
#[test]
fn python_json_encoding_and_sha_match_exactly() {
    let fixture = fixture();
    assert!(fixture.cases.len() > 900);
    for (index, case) in fixture.cases.into_iter().enumerate() {
        let options = JsonOptions {
            sort_keys: case.options.sort_keys,
            ensure_ascii: case.options.ensure_ascii,
            compact: case.options.compact,
        };
        let result = python_json_from_slice(case.raw.as_bytes(), options);
        if case.error {
            assert!(result.is_err(), "case {index}");
        } else {
            let bytes = result.unwrap_or_else(|error| panic!("case {index}: {error}"));
            assert_eq!(
                std::str::from_utf8(&bytes).unwrap(),
                case.expected.unwrap(),
                "case {index}"
            );
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                case.sha256.unwrap(),
                "case {index}"
            );
        }
    }
}

#[test]
fn wide_objects_keep_first_key_position_and_last_duplicate_value() {
    let mut raw = String::from("{");
    for i in 0..20_000 {
        if i > 0 {
            raw.push(',');
        }
        raw.push_str(&format!("\"k{i}\":{i}"));
    }
    raw.push_str(",\"k0\":-1}");
    let options = JsonOptions {
        sort_keys: false,
        ensure_ascii: true,
        compact: true,
    };
    let result = python_json_from_slice(raw.as_bytes(), options).unwrap();
    let text = std::str::from_utf8(&result).unwrap();
    assert!(text.starts_with("{\"k0\":-1,\"k1\":1,"));
    assert!(text.ends_with("\"k19999\":19999}"));
    assert_eq!(text.matches("\"k0\"").count(), 1);
}
