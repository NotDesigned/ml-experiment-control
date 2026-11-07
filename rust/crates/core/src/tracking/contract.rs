use std::collections::BTreeSet;

use super::{Error, Result};
use crate::canonical;
use serde_json::{Value, json};

/// A missing preparation option stays absent from the intent JSON; an explicitly
/// supplied false must remain present and override the server default.
#[derive(Clone, Debug, PartialEq)]
pub struct WandbOptions {
    pub enabled: bool,
    pub preparation_enabled: Option<bool>,
    pub entity: Option<String>,
    pub project: Option<String>,
}
impl Default for WandbOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            preparation_enabled: None,
            entity: None,
            project: None,
        }
    }
}
impl serde::Serialize for WandbOptions {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}
impl<'de> serde::Deserialize<'de> for WandbOptions {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
        Self::from_value(&value).map_err(serde::de::Error::custom)
    }
}
fn boolean(value: &Value) -> Result<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Number(value) => match value.as_f64() {
            Some(0.0) => Ok(false),
            Some(1.0) => Ok(true),
            _ => Err(Error::Contract("boolean")),
        },
        Value::String(value) => match value.to_ascii_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
            _ => Err(Error::Contract("boolean")),
        },
        _ => Err(Error::Contract("boolean")),
    }
}
fn target(value: Option<&Value>) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value))
            if !value.is_empty()
                && value.len() <= 128
                && value
                    .bytes()
                    .enumerate()
                    .all(|(i, c)| c.is_ascii_alphanumeric() || i > 0 && b"_.-".contains(&c)) =>
        {
            Ok(Some(value.clone()))
        }
        _ => Err(Error::Contract("entity/project identifier")),
    }
}
impl WandbOptions {
    pub fn from_value(value: &Value) -> Result<Self> {
        let object = value.as_object().ok_or(Error::Contract("options object"))?;
        if object.keys().any(|key| {
            !["enabled", "preparation_enabled", "entity", "project"].contains(&key.as_str())
        }) {
            return Err(Error::Contract("unknown option"));
        }
        let preparation_enabled = object
            .get("preparation_enabled")
            .map(|value| {
                value
                    .as_bool()
                    .ok_or(Error::Contract("strict preparation boolean"))
            })
            .transpose()?;
        Ok(Self {
            enabled: object
                .get("enabled")
                .map(boolean)
                .transpose()?
                .unwrap_or(true),
            preparation_enabled,
            entity: target(object.get("entity"))?,
            project: target(object.get("project"))?,
        })
    }
    pub fn to_value(&self) -> Value {
        let mut value = json!({"enabled":self.enabled,"entity":self.entity,"project":self.project});
        if let Some(preparation) = self.preparation_enabled {
            value["preparation_enabled"] = json!(preparation);
        }
        value
    }
}

/// Validation-only request data. It intentionally has no Debug or Serialize
/// implementation; accessing a credential requires an explicit accessor. The
/// effect owner, rather than core, decides whether/where to store the secret.
#[derive(Clone)]
pub struct WandbSettings {
    pub options: WandbOptions,
    pub clear_credentials: bool,
    api_key: Option<String>,
    provided: BTreeSet<String>,
}
fn key_whitespace(value: char) -> bool {
    matches!(value,'\t'..='\r'|'\u{1c}'..='\u{20}'|'\u{85}'|'\u{a0}'|'\u{1680}'|'\u{2000}'..='\u{200a}'|'\u{2028}'|'\u{2029}'|'\u{202f}'|'\u{205f}'|'\u{3000}')
}
impl WandbSettings {
    pub fn from_value(value: &Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or(Error::Contract("settings object"))?;
        if object.keys().any(|key| {
            ![
                "enabled",
                "preparation_enabled",
                "entity",
                "project",
                "api_key",
                "clear_credentials",
            ]
            .contains(&key.as_str())
        }) {
            return Err(Error::Contract("unknown setting"));
        }
        let mut options = object.clone();
        options.remove("api_key");
        options.remove("clear_credentials");
        let api_key = match object.get("api_key") {
            None | Some(Value::Null) => None,
            Some(Value::String(key))
                if (1..=512).contains(&key.chars().count())
                    && !key.chars().any(|c| c < '!' || key_whitespace(c)) =>
            {
                Some(key.clone())
            }
            _ => return Err(Error::Contract("private API key")),
        };
        Ok(Self {
            options: WandbOptions::from_value(&Value::Object(options))?,
            clear_credentials: object
                .get("clear_credentials")
                .map(boolean)
                .transpose()?
                .unwrap_or(false),
            api_key,
            provided: object.keys().cloned().collect(),
        })
    }
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
    pub fn provided(&self, field: &str) -> bool {
        self.provided.contains(field)
    }
}

/// Already validated server settings without any credential material. SDK/network
/// availability is deliberately absent: it must not rewrite a frozen opt-in route.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingSettings {
    pub enabled: bool,
    pub preparation_enabled: bool,
    pub credentials_configured: bool,
    pub resolved_entity: Option<String>,
    pub project: Option<String>,
    /// Present only when the operator configured MLflow. Absent, every route is
    /// exactly what it was before MLflow support.
    pub mlflow: Option<MlflowRouting>,
}
impl Default for RoutingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            preparation_enabled: false,
            credentials_configured: false,
            resolved_entity: None,
            project: None,
            mlflow: None,
        }
    }
}

/// Server MLflow defaults without credentials or the server address. A Run
/// freezes the decision and the experiment name, as it freezes its W&B route.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MlflowRouting {
    pub enabled: bool,
    pub experiment_prefix: String,
    /// When set, only these ML-Expd projects publish to MLflow.
    pub projects: Option<BTreeSet<String>>,
}
/// The frozen MLflow part of a Run's route: one MLflow experiment per project.
pub fn mlflow_route(settings: &MlflowRouting, project: &str) -> Value {
    let reason = if !settings.enabled {
        Some("DISABLED_BY_SERVER")
    } else if settings
        .projects
        .as_ref()
        .is_some_and(|projects| !projects.contains(project))
    {
        Some("PROJECT_NOT_SELECTED")
    } else {
        None
    };
    json!({"enabled":reason.is_none(),"reason":reason,"experiment":format!("{}{project}",settings.experiment_prefix)})
}
pub fn resolve(options: Option<&WandbOptions>, settings: &RoutingSettings, project: &str) -> Value {
    let default = WandbOptions::default();
    let options = options.unwrap_or(&default);
    let entity = options
        .entity
        .as_ref()
        .filter(|value| !value.is_empty())
        .or(settings.resolved_entity.as_ref());
    let reason = if !options.enabled {
        Some("DISABLED_BY_REQUEST")
    } else if !settings.enabled {
        Some("DISABLED_BY_SERVER")
    } else if !settings.credentials_configured {
        Some("CREDENTIALS_NOT_CONFIGURED")
    } else if entity.is_none_or(|value| value.is_empty()) {
        Some("DEFAULT_ENTITY_UNAVAILABLE")
    } else {
        None
    };
    let mut route = json!({"requested":options.enabled,"enabled":reason.is_none(),"reason":reason,"entity":entity,"project":options.project.as_deref().filter(|value|!value.is_empty()).or(settings.project.as_deref().filter(|value|!value.is_empty())).unwrap_or(project)});
    // W&B options never change the MLflow decision: each destination stands alone.
    if let Some(mlflow) = &settings.mlflow {
        route["mlflow"] = mlflow_route(mlflow, project);
    }
    route
}
pub fn resolve_preparation(
    options: Option<&WandbOptions>,
    settings: &RoutingSettings,
    project: &str,
) -> Value {
    let selected = WandbOptions {
        enabled: options
            .and_then(|v| v.preparation_enabled)
            .unwrap_or(settings.preparation_enabled),
        preparation_enabled: None,
        entity: options.and_then(|v| v.entity.clone()),
        project: options.and_then(|v| v.project.clone()),
    };
    // A preparation Run is build and data progress, never an MLflow run.
    let mut route = resolve(Some(&selected), settings, project);
    if let Some(route) = route.as_object_mut() {
        route.remove("mlflow");
    }
    route
}

pub fn finite_parameters(value: &Value) -> Result<&Value> {
    if !value.is_null() {
        canonical::bytes(value)?;
    }
    Ok(value)
}
