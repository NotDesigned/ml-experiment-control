//! Frozen behavior obtained by evaluating the original Python implementation.
//! These tests do not import Python or contact any scheduler at test runtime.

use std::collections::BTreeSet;

use ml_exp_core::identity::{submission_marker, validate_id};
use ml_exp_core::state::RunState;
use serde::Deserialize;

const BASELINE: &str = "e725faec689fb8395e692e27d00a8896fa7fe5a5";
const MANIFEST_SHA: &str = "2b51c7ccb21ef21f98f395de2487aa901a0b470331ad1b13e4173f2c80369ba6";
const SUBMISSION_SHA: &str = "ec5b38bb27ea71d818deda7babdaa3229058c8f1e01a278a21c83ba94486ebad";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    baseline_commit: String,
    source_path: String,
    source_sha256: String,
    symbols: Vec<String>,
    generation: String,
}

impl Reference {
    fn assert_provenance(&self, source: &str, sha: &str, symbols: &[&str]) {
        assert_eq!(self.baseline_commit, BASELINE);
        assert_eq!(self.source_path, source);
        assert_eq!(self.source_sha256, sha);
        assert_eq!(self.symbols, symbols);
        assert!(!self.generation.is_empty());
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Transition {
    from: RunState,
    to: RunState,
    allowed: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StateFixture {
    schema_version: u32,
    reference: Reference,
    states: Vec<RunState>,
    terminal_states: Vec<RunState>,
    transitions: Vec<Transition>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Cases<T> {
    schema_version: u32,
    reference: Reference,
    input_scope: String,
    cases: Vec<T>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityCase {
    value: String,
    valid: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MarkerCase {
    token: String,
    valid: bool,
    marker: Option<String>,
}

#[test]
fn every_state_pair_and_terminal_state_matches_python_reference() {
    let fixture: StateFixture =
        serde_json::from_str(include_str!("../../../compatibility/state-machine.json")).unwrap();
    assert_eq!(fixture.schema_version, 1);
    fixture.reference.assert_provenance(
        "src/experiment_control/manifest.py",
        MANIFEST_SHA,
        &[
            "RunState",
            "ALLOWED_STATE_TRANSITIONS",
            "TERMINAL_RUN_STATES",
        ],
    );

    // The fixture must retain every positive AND negative transition, including
    // sampled scheduler skips, requeue and terminal self-transitions.
    let names: Vec<_> = fixture
        .states
        .iter()
        .map(|state| serde_json::to_value(state).unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "NOT_SUBMITTED",
            "CREATED",
            "SUBMITTING",
            "QUEUED",
            "STARTING",
            "RUNNING",
            "EVALUATING",
            "SUCCEEDED",
            "FAILED",
            "PREEMPTED",
            "CANCELLED",
            "UNKNOWN",
        ]
        .map(serde_json::Value::from)
    );
    assert_eq!(fixture.terminal_states.len(), 4);
    assert_eq!(fixture.transitions.len(), 144);
    let mut pairs = BTreeSet::new();
    for case in &fixture.transitions {
        assert!(fixture.states.contains(&case.from));
        assert!(fixture.states.contains(&case.to));
        assert!(pairs.insert(format!("{:?}->{:?}", case.from, case.to)));
        assert_eq!(
            case.from.allows(case.to),
            case.allowed,
            "Python transition {:?} -> {:?}",
            case.from,
            case.to
        );
    }
    for state in fixture.states {
        assert_eq!(
            state.is_terminal(),
            fixture.terminal_states.contains(&state),
            "Python terminal classification of {state:?}"
        );
    }
}

#[test]
fn identity_ascii_and_length_boundaries_match_python_reference() {
    let fixture: Cases<IdentityCase> =
        serde_json::from_str(include_str!("../../../compatibility/identity.json")).unwrap();
    assert_eq!(fixture.schema_version, 1);
    fixture.reference.assert_provenance(
        "src/experiment_control/manifest.py",
        MANIFEST_SHA,
        &["validate_identity", "IDENTITY_RE"],
    );
    assert!(!fixture.input_scope.is_empty());
    assert_eq!(fixture.cases.len(), 279);
    let mut values = BTreeSet::new();
    for case in fixture.cases {
        assert!(values.insert(case.value.clone()));
        let actual = validate_id(&case.value);
        assert_eq!(actual.is_ok(), case.valid, "Python ID {:?}", case.value);
        if let Ok(value) = actual {
            assert_eq!(value, case.value);
        }
    }
}

#[test]
fn submission_token_and_marker_match_python_reference() {
    let fixture: Cases<MarkerCase> = serde_json::from_str(include_str!(
        "../../../compatibility/submission-marker.json"
    ))
    .unwrap();
    assert_eq!(fixture.schema_version, 1);
    fixture.reference.assert_provenance(
        "src/experiment_control/submission.py",
        SUBMISSION_SHA,
        &[
            "submission_marker",
            "validate_submission_token",
            "SUBMISSION_TOKEN_RE",
        ],
    );
    assert!(!fixture.input_scope.is_empty());
    assert_eq!(fixture.cases.len(), 135);
    let mut tokens = BTreeSet::new();
    for case in fixture.cases {
        assert!(tokens.insert(case.token.clone()));
        assert_eq!(case.marker.is_some(), case.valid);
        let actual = submission_marker(&case.token);
        assert_eq!(actual.is_ok(), case.valid, "Python token {:?}", case.token);
        assert_eq!(actual.ok(), case.marker, "Python marker {:?}", case.token);
    }
}
