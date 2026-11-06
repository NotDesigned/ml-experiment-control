//! Explicit metric units and context. Scientific meanings remain project-owned.
use serde::{Deserialize, Serialize};

mod observation;
mod python_printability;
mod result;
mod schema;

pub use observation::python_string;

/// Python-reference Unicode printability, shared by pure human presentation.
pub fn python_is_printable(value: char) -> bool {
    python_printability::is_printable(value)
}

pub(crate) use observation::python_repr;
pub use observation::{MetricValue, ObservationInput, finite, observations, observations_typed};
pub use result::{metric_result, metric_result_typed, numeric_compare, numeric_equal};
pub use schema::{MetricDefinition, MetricSchema, RequiredWhen};

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum MetricError {
    #[error("invalid metric schema")]
    Schema,
    #[error("metric record or normalized identity is invalid")]
    Record,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MetricStatus {
    Valid,
    Partial,
    Failed,
    Missing,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricRecord {
    pub name: String,
    pub unit: String,
    pub value: Option<f64>,
    #[serde(default = "valid")]
    pub status: MetricStatus,
    pub step: Option<f64>,
    pub epoch: Option<f64>,
    pub timestamp: Option<String>,
    pub protocol_id: Option<String>,
    pub checkpoint_id: Option<String>,
    pub dataset_id: Option<String>,
    pub variant_id: Option<String>,
    pub numerator: Option<f64>,
    pub denominator: Option<f64>,
    #[serde(default)]
    pub summary: bool,
    pub error: Option<String>,
}
fn valid() -> MetricStatus {
    MetricStatus::Valid
}

impl MetricRecord {
    pub fn validate(&self) -> Result<(), &'static str> {
        for s in [&self.name, &self.unit] {
            if s.trim().is_empty() || s.chars().count() > 128 || s.chars().any(|c| c < ' ') {
                return Err("metric needs a printable name and explicit unit");
            }
        }
        if self.status == MetricStatus::Valid && !self.value.is_some_and(f64::is_finite) {
            return Err("valid metric needs a finite value");
        }
        for n in [
            self.value,
            self.step,
            self.epoch,
            self.numerator,
            self.denominator,
        ]
        .into_iter()
        .flatten()
        {
            if !n.is_finite() {
                return Err("metric numeric fields must be finite");
            }
        }
        for s in [
            &self.timestamp,
            &self.protocol_id,
            &self.checkpoint_id,
            &self.dataset_id,
            &self.variant_id,
            &self.error,
        ]
        .into_iter()
        .flatten()
        {
            if s.trim().is_empty() || s.chars().count() > 2000 || s.chars().any(|c| c < ' ') {
                return Err("invalid metric context");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonfinite_values_and_missing_units_are_rejected() {
        let mut a: MetricRecord = serde_json::from_value(serde_json::json!({"name":"custom/loss", "unit":"nats/token", "value": 2.0, "step": 3, "variant_id":"k0"})).unwrap();
        a.validate().unwrap();
        a.value = Some(f64::NAN);
        assert!(a.validate().is_err());
        a.value = None;
        a.status = MetricStatus::Missing;
        a.validate().unwrap();
        a.unit.clear();
        assert!(a.validate().is_err());
    }
}
