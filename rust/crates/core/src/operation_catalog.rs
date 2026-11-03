//! Pure catalog of daemon-owned operations and scope constraints.
//!
//! The catalog is renderer- and transport-neutral.  It does not authorize,
//! persist or execute an Action; those effects remain in the server shell.
//! Keeping the catalog here prevents clients and HTTP renderers from growing
//! separate copies of the operation/intent policy.
use crate::operations::ScopeType;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Intent,
    Direct,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParameterKind {
    Text,
    Number,
    Enum,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub key: String,
    pub label: String,
    pub kind: ParameterKind,
    pub required: bool,
    pub default: Value,
    pub placeholder: String,
    pub positive: bool,
    pub choices: Vec<(String, Value)>,
}

impl Parameter {
    pub fn text(key: &str, label: &str, required: bool, placeholder: &str) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            kind: ParameterKind::Text,
            required,
            default: Value::Null,
            placeholder: placeholder.into(),
            positive: false,
            choices: Vec::new(),
        }
    }

    pub fn number(key: &str, label: &str, default: f64, positive: bool) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            kind: ParameterKind::Number,
            required: true,
            default: serde_json::Number::from_f64(default)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            placeholder: String::new(),
            positive,
            choices: Vec::new(),
        }
    }

    pub fn enumeration(
        key: &str,
        label: &str,
        required: bool,
        default: &str,
        choices: &[(&str, &str)],
    ) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            kind: ParameterKind::Enum,
            required,
            default: Value::String(default.into()),
            placeholder: String::new(),
            positive: false,
            choices: choices
                .iter()
                .map(|(label, value)| ((*label).into(), Value::String((*value).into())))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Definition {
    pub operation_id: String,
    pub label: String,
    pub description: String,
    pub category: String,
    pub scopes: Vec<ScopeType>,
    pub mode: Mode,
    pub expected_effect: String,
    pub intent_kind: Option<String>,
    pub parameters: Vec<Parameter>,
    pub priority: u16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Availability {
    pub operation: Definition,
    pub scope: Scope,
    pub status: AvailabilityStatus,
    pub reasons: Vec<String>,
    pub expected_effect: String,
    pub metadata: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AvailabilityStatus {
    Available,
    Blocked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scope {
    pub project: String,
    pub scope_type: ScopeType,
    pub object_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultRecord {
    pub operation_id: String,
    pub scope: Scope,
    pub message: String,
    pub action_id: Option<String>,
    pub affected_identity: Option<String>,
}

fn request() -> Parameter {
    Parameter::text(
        "request",
        "Operation request",
        true,
        "Describe the requested change",
    )
}

fn reason() -> Parameter {
    Parameter::text(
        "reason",
        "Reason",
        true,
        "Record why this operation is needed",
    )
}

#[allow(clippy::too_many_arguments)]
fn operation(
    operation_id: &str,
    label: &str,
    description: &str,
    category: &str,
    scopes: &[ScopeType],
    mode: Mode,
    expected_effect: &str,
    intent_kind: Option<&str>,
    parameters: Vec<Parameter>,
    priority: u16,
) -> Definition {
    Definition {
        operation_id: operation_id.into(),
        label: label.into(),
        description: description.into(),
        category: category.into(),
        scopes: scopes.to_vec(),
        mode,
        expected_effect: expected_effect.into(),
        intent_kind: intent_kind.map(str::to_owned),
        parameters,
        priority,
    }
}

/// Return the immutable operation catalog. A fresh vector avoids exposing a
/// mutable global and keeps callers free to sort or serialize the result.
pub fn definitions() -> Vec<Definition> {
    let gpu_budget = Parameter::number("max_gpu_hours", "Maximum GPU-hours", 1.0, true);
    let resource_approval = Parameter::enumeration(
        "resource_approval",
        "Resource approval",
        false,
        "budget_cap",
        &[
            ("Agent/CLI budget cap", "budget_cap"),
            ("Review exact dry-run resources", "review_exact"),
        ],
    );
    let new_attempt = Parameter::text(
        "new_attempt_id",
        "New Attempt ID",
        false,
        "Optional attempt-NNN identity",
    );
    let new_run = Parameter::text(
        "new_run_id",
        "New Run ID",
        true,
        "Use {profile} when cloning more than one fallback profile",
    );
    let profiles = Parameter::text(
        "profiles",
        "Ordered profiles",
        false,
        "Comma-separated profiles, for example h100,h200,l40s",
    );
    let overrides = Parameter::text(
        "config_overrides",
        "Config overrides",
        false,
        r#"JSON list, for example ["train_steps=10000"]"#,
    );
    vec![
        operation(
            "campaign.create",
            "Create Campaign",
            "Prepare a Campaign file and catalog update from a client-authored definition.",
            "Project code",
            &[ScopeType::Project],
            Mode::Intent,
            "Prepare a reviewable Campaign write",
            Some("CREATE_CAMPAIGN_DRAFT"),
            vec![request()],
            20,
        ),
        operation(
            "campaign.update",
            "Update Campaign",
            "Prepare a new Campaign revision without deriving a scientific conclusion.",
            "Project code",
            &[ScopeType::Campaign],
            Mode::Intent,
            "Prepare a reviewable Campaign write",
            Some("UPDATE_CAMPAIGN_DRAFT"),
            vec![request()],
            20,
        ),
        operation(
            "run.derive",
            "Derive Run definition",
            "Prepare a client-authored Campaign/Run definition with immutable identities.",
            "Project code",
            &[ScopeType::Campaign, ScopeType::Run],
            Mode::Intent,
            "Prepare a reviewable Campaign write",
            Some("DERIVE_RUN_DRAFT"),
            vec![request()],
            30,
        ),
        operation(
            "run.clone",
            "Clone Run definition",
            "Deterministically clone an authored Run in its current Campaign.",
            "Project code",
            &[ScopeType::Run],
            Mode::Direct,
            "Prepare one reviewable Campaign update",
            Some("DERIVE_RUN_DRAFT"),
            vec![new_run, profiles, overrides, reason()],
            29,
        ),
        operation(
            "object.archive",
            "Archive record",
            "Append an archive record while retaining immutable evidence.",
            "Lifecycle",
            &[ScopeType::Campaign, ScopeType::Run, ScopeType::Attempt],
            Mode::Direct,
            "Prepare an archive Action",
            None,
            vec![reason()],
            40,
        ),
        operation(
            "evidence.rebuild_local",
            "Rebuild local Attempt evidence",
            "Recompute one terminal Attempt collection from already-local durable artifacts.",
            "Evidence",
            &[ScopeType::Attempt],
            Mode::Direct,
            "Prepare an audited local-evidence rebuild Action",
            Some("REBUILD_LOCAL_EVIDENCE"),
            vec![reason()],
            31,
        ),
        operation(
            "run.submit",
            "Submit first Attempt",
            "Prepare the first scheduler submission for an authored Run.",
            "Execution",
            &[ScopeType::Run],
            Mode::Direct,
            "Prepare a scheduler Action",
            Some("SUBMIT_RUN"),
            vec![gpu_budget.clone(), resource_approval.clone()],
            30,
        ),
        operation(
            "attempt.retry",
            "Retry as new Attempt",
            "Prepare a retry without reusing an immutable Attempt identity.",
            "Execution",
            &[ScopeType::Attempt],
            Mode::Direct,
            "Prepare a scheduler Action",
            Some("RETRY_ATTEMPT"),
            vec![
                reason(),
                gpu_budget.clone(),
                resource_approval.clone(),
                new_attempt,
            ],
            30,
        ),
        operation(
            "attempt.cancel",
            "Cancel active Attempt",
            "Prepare cancellation of the exact observed backend job.",
            "Execution",
            &[ScopeType::Attempt],
            Mode::Direct,
            "Prepare a scheduler Action",
            Some("CANCEL_RUN"),
            vec![reason()],
            30,
        ),
        operation(
            "run.evaluate",
            "Run evaluation",
            "Prepare an immutable evaluation-as-run supplied by the client.",
            "Execution",
            &[ScopeType::Run, ScopeType::Attempt],
            Mode::Intent,
            "Prepare an evaluation scheduler Action",
            Some("RUN_EVALUATION"),
            vec![request(), gpu_budget],
            35,
        ),
    ]
}

/// Validate that an intent kind is usable for the requested scope.
pub fn intent_scope_error(kind: &str, scope: ScopeType) -> Option<String> {
    let allowed = match kind {
        "CREATE_CAMPAIGN_DRAFT" => vec![ScopeType::Project],
        "UPDATE_CAMPAIGN_DRAFT" => vec![ScopeType::Campaign],
        "DERIVE_RUN_DRAFT" => vec![ScopeType::Campaign, ScopeType::Run],
        "SUBMIT_RUN" | "ARCHIVE_RUN" | "DELETE_RUN_RESOURCES" => vec![ScopeType::Run],
        "RETRY_ATTEMPT" | "CANCEL_RUN" | "ARCHIVE_ATTEMPT" | "REBUILD_LOCAL_EVIDENCE" => {
            vec![ScopeType::Attempt]
        }
        "RUN_EVALUATION" => vec![ScopeType::Run, ScopeType::Attempt],
        "ARCHIVE_CAMPAIGN" => vec![ScopeType::Campaign],
        _ => return Some(format!("unsupported intent kind {kind:?}")),
    };
    if allowed.contains(&scope) {
        return None;
    }
    let names = allowed
        .iter()
        .map(|item| item.name())
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "intent kind {kind} is not valid in {} scope; expected {names}",
        scope.name()
    ))
}

/// Return definitions in the same stable order as the Python catalog.
pub fn for_scope(scope: ScopeType) -> Vec<Definition> {
    let mut values: Vec<_> = definitions()
        .into_iter()
        .filter(|value| value.scopes.contains(&scope))
        .collect();
    values.sort_by(|left, right| {
        (left.priority, left.category.as_str(), left.label.as_str()).cmp(&(
            right.priority,
            right.category.as_str(),
            right.label.as_str(),
        ))
    });
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_stable_ids_and_scope_order() {
        let values = definitions();
        assert_eq!(values.len(), 10);
        assert_eq!(values[0].operation_id, "campaign.create");
        assert_eq!(
            for_scope(ScopeType::Attempt)[0].operation_id,
            "attempt.cancel"
        );
        assert_eq!(for_scope(ScopeType::Run)[0].operation_id, "run.clone");
        assert!(values.iter().all(|value| !value.operation_id.is_empty()));
    }

    #[test]
    fn intent_scope_validation_is_explicit_and_fail_closed() {
        assert_eq!(intent_scope_error("SUBMIT_RUN", ScopeType::Run), None);
        assert!(
            intent_scope_error("SUBMIT_RUN", ScopeType::Attempt)
                .is_some_and(|value| value.contains("expected run"))
        );
        assert!(
            intent_scope_error("UNKNOWN", ScopeType::Run)
                .is_some_and(|value| value.contains("unsupported"))
        );
    }

    #[test]
    fn parameters_preserve_defaults_and_choices() {
        let submit = definitions()
            .into_iter()
            .find(|value| value.operation_id == "run.submit")
            .unwrap();
        assert_eq!(submit.parameters[0].default, serde_json::json!(1.0));
        assert!(submit.parameters[0].positive);
        assert_eq!(submit.parameters[1].choices.len(), 2);
        assert!(!submit.parameters[1].required);
    }
}
