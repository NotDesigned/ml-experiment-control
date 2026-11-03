//! Project-authored metric schemas; omitted `required_when` remains omitted.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};

use super::MetricError;

#[derive(Clone, Debug, Default, PartialEq)]
pub enum RequiredWhen {
    #[default]
    Omitted,
    Null,
    Context(Map<String, Value>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricDefinition {
    pub unit: String,
    pub required: bool,
    pub description: String,
    pub aggregation: String,
    pub numerator_unit: Option<String>,
    pub denominator_unit: Option<String>,
    pub required_when: RequiredWhen,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetricSchema {
    /// Input order matters for published required/missing metric lists.
    pub definitions: Vec<(String, MetricDefinition)>,
}

fn string(
    value: Option<&Value>,
    min: usize,
    max: usize,
    printable: bool,
) -> Result<String, MetricError> {
    let value = value.and_then(Value::as_str).ok_or(MetricError::Schema)?;
    if value.chars().count() < min
        || value.chars().count() > max
        || printable && (value.trim().is_empty() || value.chars().any(|c| c < ' '))
    {
        return Err(MetricError::Schema);
    }
    Ok(value.to_owned())
}

fn optional_unit(value: Option<&Value>) -> Result<Option<String>, MetricError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        value => string(value, 1, 128, true).map(Some),
    }
}

fn python_bool(value: &Value) -> Result<bool, MetricError> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Number(number) => match number.as_f64() {
            Some(0.0) => Ok(false),
            Some(1.0) => Ok(true),
            _ => Err(MetricError::Schema),
        },
        Value::String(value) => match value.to_ascii_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
            _ => Err(MetricError::Schema),
        },
        _ => Err(MetricError::Schema),
    }
}

impl MetricDefinition {
    pub fn from_value(value: &Value) -> Result<Self, MetricError> {
        let fields = value.as_object().ok_or(MetricError::Schema)?;
        if fields.keys().any(|key| {
            ![
                "unit",
                "required",
                "description",
                "aggregation",
                "numerator_unit",
                "denominator_unit",
                "required_when",
            ]
            .contains(&key.as_str())
        }) {
            return Err(MetricError::Schema);
        }
        let required_when = match fields.get("required_when") {
            None => RequiredWhen::Omitted,
            Some(Value::Null) => RequiredWhen::Null,
            Some(Value::Object(context)) if !context.is_empty() => {
                for (key, value) in context {
                    if !["dataset_id", "variant_id", "checkpoint_id"].contains(&key.as_str()) {
                        return Err(MetricError::Schema);
                    }
                    string(Some(value), 1, 2000, true)?;
                }
                RequiredWhen::Context(context.clone())
            }
            _ => return Err(MetricError::Schema),
        };
        Ok(Self {
            unit: string(fields.get("unit"), 1, 128, true)?,
            required: fields
                .get("required")
                .map(python_bool)
                .transpose()?
                .unwrap_or(false),
            description: match fields.get("description") {
                None => String::new(),
                value => string(value, 0, 2000, false)?,
            },
            aggregation: match fields.get("aggregation") {
                None => "scalar".to_owned(),
                value => string(value, 1, 128, false)?,
            },
            numerator_unit: optional_unit(fields.get("numerator_unit"))?,
            denominator_unit: optional_unit(fields.get("denominator_unit"))?,
            required_when,
        })
    }

    pub fn to_value(&self) -> Value {
        let mut value = json!({
            "unit": self.unit,
            "required": self.required,
            "description": self.description,
            "aggregation": self.aggregation,
            "numerator_unit": self.numerator_unit,
            "denominator_unit": self.denominator_unit,
        });
        match &self.required_when {
            RequiredWhen::Omitted => {}
            RequiredWhen::Null => {
                value["required_when"] = Value::Null;
            }
            RequiredWhen::Context(context) => {
                value["required_when"] = Value::Object(context.clone());
            }
        }
        value
    }
}

impl MetricSchema {
    pub fn from_value(value: &Value) -> Result<Self, MetricError> {
        let fields = value.as_object().ok_or(MetricError::Schema)?;
        if fields
            .keys()
            .any(|key| !["schema_version", "definitions"].contains(&key.as_str()))
        {
            return Err(MetricError::Schema);
        }
        if let Some(version) = fields.get("schema_version") {
            if !matches!(version, Value::Bool(true)) && version.as_f64() != Some(1.0) {
                return Err(MetricError::Schema);
            }
        }
        let definitions = match fields.get("definitions") {
            None => return Ok(Self::default()),
            Some(Value::Object(definitions)) if definitions.len() <= 256 => definitions,
            _ => return Err(MetricError::Schema),
        };
        let mut result = Vec::with_capacity(definitions.len());
        for (name, definition) in definitions {
            string(Some(&Value::String(name.clone())), 1, 128, true)?;
            result.push((name.clone(), MetricDefinition::from_value(definition)?));
        }
        Ok(Self {
            definitions: result,
        })
    }

    pub fn to_value(&self) -> Value {
        let definitions: Map<_, _> = self
            .definitions
            .iter()
            .map(|(name, definition)| (name.clone(), definition.to_value()))
            .collect();
        json!({"schema_version": 1, "definitions": definitions})
    }
}

impl Serialize for MetricDefinition {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for MetricDefinition {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::from_value(&Value::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}
impl Serialize for MetricSchema {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for MetricSchema {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::from_value(&Value::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}
