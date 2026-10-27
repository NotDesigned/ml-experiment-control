use ml_exp_core::{
    canonical,
    manifest::{RunManifestInput, build_run_manifest, comparable_manifest},
};
use serde_json::Value;

#[test]
fn scientific_manifest_and_canonical_identity_match_python() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/run-manifest.json")).unwrap();
    assert_eq!(fixture["schema_version"], 1);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 36);
    for (index, case) in cases.iter().enumerate() {
        let input: RunManifestInput = serde_json::from_value(case["input"].clone()).unwrap();
        let manifest = build_run_manifest(&input);
        assert_eq!(manifest, case["manifest"], "manifest {index}");
        assert_eq!(
            comparable_manifest(manifest.as_object().unwrap()),
            case["comparable"],
            "compare {index}"
        );
        assert_eq!(
            String::from_utf8(canonical::python_json_bytes(&manifest).unwrap()).unwrap(),
            case["canonical"].as_str().unwrap(),
            "bytes {index}"
        );
        assert_eq!(
            canonical::digest(&manifest).unwrap(),
            case["sha256"].as_str().unwrap(),
            "SHA {index}"
        );
        assert!(
            !manifest["resolved_config"]
                .as_object()
                .unwrap()
                .contains_key("resume")
        );
    }
    for case in fixture["comparisons"].as_array().unwrap() {
        assert_eq!(
            comparable_manifest(case["input"].as_object().unwrap()),
            case["expected"]
        );
    }
}
