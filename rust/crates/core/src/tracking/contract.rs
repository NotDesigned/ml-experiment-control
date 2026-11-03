use std::collections::BTreeSet;

use super::{Error, Result};
use crate::canonical::{self, JsonOptions};
use serde_json::{Value, json};

/// A missing preparation option retains historical intent JSON; an explicitly
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
fn python_whitespace(value: char) -> bool {
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
                    && !key.chars().any(|c| c < '!' || python_whitespace(c)) =>
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
    pub fn provided_fields(&self) -> impl Iterator<Item = &str> {
        self.provided.iter().map(String::as_str)
    }
    /// Equivalent to the reference's JSON-mode request serialization. Never
    /// exposes the raw credential, including on failed validation paths.
    pub fn to_public_value(&self) -> Value {
        let mut value = self.options.to_value();
        value["api_key"] = self
            .api_key
            .as_ref()
            .map(|_| json!("**********"))
            .unwrap_or(Value::Null);
        value["clear_credentials"] = json!(self.clear_credentials);
        value
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
}
impl Default for RoutingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            preparation_enabled: false,
            credentials_configured: false,
            resolved_entity: None,
            project: None,
        }
    }
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
    json!({"requested":options.enabled,"enabled":reason.is_none(),"reason":reason,"entity":entity,"project":options.project.as_deref().filter(|value|!value.is_empty()).or(settings.project.as_deref().filter(|value|!value.is_empty())).unwrap_or(project)})
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
    resolve(Some(&selected), settings, project)
}

fn parameter_json() -> JsonOptions {
    JsonOptions {
        sort_keys: false,
        ensure_ascii: true,
        compact: false,
    }
}
pub fn finite_parameters(value: &Value) -> Result<&Value> {
    if !value.is_null() && canonical::encode(value, parameter_json())?.len() > 16384 {
        return Err(Error::Contract("parameters exceed 16 KiB"));
    }
    Ok(value)
}
/// Raw JSON boundary so NaN/Infinity can be rejected without replacing them with
/// valid JSON null/zero. Returned bytes use Python's ordinary spaced JSON format.
pub fn finite_parameters_json(bytes: &[u8]) -> Result<Vec<u8>> {
    let encoded = canonical::python_json_from_slice(bytes, parameter_json())?;
    if encoded.len() > 16384 {
        return Err(Error::Contract("parameters exceed 16 KiB"));
    }
    Ok(encoded)
}
