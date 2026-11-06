//! Native boundaries of identities and submission tokens, and the Run state
//! machine. Nothing here touches a scheduler.

use ml_exp_core::identity::{submission_marker, validate_id};
use ml_exp_core::state::RunState;

/// Every state pair, including sampled scheduler skips, requeue and terminal
/// self-transitions. A pair missing from this table is refused.
const ALLOWED_TRANSITIONS: [(&str, &[&str]); 12] = [
    ("NOT_SUBMITTED", &["CREATED"]),
    (
        "CREATED",
        &[
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
        ],
    ),
    (
        "SUBMITTING",
        &[
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
        ],
    ),
    (
        "QUEUED",
        &[
            "QUEUED",
            "STARTING",
            "RUNNING",
            "EVALUATING",
            "SUCCEEDED",
            "FAILED",
            "PREEMPTED",
            "CANCELLED",
            "UNKNOWN",
        ],
    ),
    (
        "STARTING",
        &[
            "QUEUED",
            "STARTING",
            "RUNNING",
            "EVALUATING",
            "SUCCEEDED",
            "FAILED",
            "PREEMPTED",
            "CANCELLED",
            "UNKNOWN",
        ],
    ),
    (
        "RUNNING",
        &[
            "QUEUED",
            "RUNNING",
            "EVALUATING",
            "SUCCEEDED",
            "FAILED",
            "PREEMPTED",
            "CANCELLED",
            "UNKNOWN",
        ],
    ),
    (
        "EVALUATING",
        &[
            "EVALUATING",
            "SUCCEEDED",
            "FAILED",
            "PREEMPTED",
            "CANCELLED",
            "UNKNOWN",
        ],
    ),
    ("SUCCEEDED", &["SUCCEEDED"]),
    ("FAILED", &["FAILED"]),
    ("PREEMPTED", &["PREEMPTED"]),
    ("CANCELLED", &["CANCELLED"]),
    (
        "UNKNOWN",
        &[
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
        ],
    ),
];

#[test]
fn run_state_transitions_and_terminal_states_are_exact() {
    let state =
        |name: &str| -> RunState { serde_json::from_value(serde_json::Value::from(name)).unwrap() };
    let terminal = ["SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED"];
    for (from, allowed) in ALLOWED_TRANSITIONS {
        assert_eq!(
            state(from).is_terminal(),
            terminal.contains(&from),
            "{from}"
        );
        for (to, _) in ALLOWED_TRANSITIONS {
            assert_eq!(
                state(from).allows(state(to)),
                allowed.contains(&to),
                "{from} -> {to}"
            );
        }
    }
}

fn ascii() -> impl Iterator<Item = char> {
    (0u8..=127).map(char::from)
}

#[test]
fn identity_accepts_only_ascii_names_within_the_length_limit() {
    let accepted = [
        "a".to_owned(),
        "A".to_owned(),
        "0".to_owned(),
        "project-2026.10_27".to_owned(),
        "attempt-001".to_owned(),
        format!("source.{}", "a".repeat(64)),
        "a".repeat(127),
        "a".repeat(128),
        format!("0{}", ".".repeat(127)),
        format!("Z{}", "_".repeat(127)),
        format!("1{}", "-".repeat(127)),
    ];
    for value in &accepted {
        assert_eq!(validate_id(value).ok(), Some(value.as_str()));
    }
    let refused = [
        String::new(),
        "a".repeat(129),
        "a".repeat(256),
        ".hidden".to_owned(),
        "..".to_owned(),
        "../run".to_owned(),
        "a/b".to_owned(),
        "a\\b".to_owned(),
        "\u{e9}".to_owned(),
        "a\u{e9}".to_owned(),
        "\u{6a21}\u{578b}".to_owned(),
        "a\u{6a21}\u{578b}".to_owned(),
        "a\u{a0}".to_owned(),
        "\u{a0}a".to_owned(),
        "a\u{1f642}".to_owned(),
    ];
    for value in &refused {
        assert!(validate_id(value).is_err(), "{value:?}");
    }
}

#[test]
fn identity_checks_every_ascii_character_in_first_and_later_positions() {
    for c in ascii() {
        let first = c.to_string();
        assert_eq!(
            validate_id(&first).is_ok(),
            c.is_ascii_alphanumeric(),
            "{first:?}"
        );
        let later = format!("a{c}");
        assert_eq!(
            validate_id(&later).is_ok(),
            c.is_ascii_alphanumeric() || "._-".contains(c),
            "{later:?}"
        );
    }
}

#[test]
fn submission_token_is_exactly_32_lowercase_hex_characters_and_names_its_marker() {
    for token in [
        "0".repeat(32),
        "0123456789abcdef0123456789abcdef".to_owned(),
        "f".repeat(32),
    ] {
        assert_eq!(
            submission_marker(&token).ok(),
            Some(format!("ml-exp-{token}"))
        );
    }
    for token in [
        String::new(),
        "0".repeat(31),
        "0".repeat(33),
        format!("{}\n", "0".repeat(32)),
        format!("{}\u{e9}", "0".repeat(31)),
    ] {
        assert!(submission_marker(&token).is_err(), "{token:?}");
    }
    for c in ascii() {
        let token = format!("{}{c}", "0".repeat(31));
        let valid = c.is_ascii_digit() || ('a'..='f').contains(&c);
        assert_eq!(
            submission_marker(&token).ok(),
            valid.then(|| format!("ml-exp-{token}")),
            "{token:?}"
        );
    }
}
