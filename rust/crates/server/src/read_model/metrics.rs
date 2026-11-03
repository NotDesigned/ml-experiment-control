use super::{Error, MAX_METRIC_OBSERVATIONS, Result, bounded};
use ml_exp_core::{
    canonical,
    metrics::{MetricRecord, MetricStatus, finite, metric_result, observations, python_string},
    observation::python_equal,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricFormat {
    #[default]
    Legacy,
    Records,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricQuery {
    #[serde(default)]
    pub format: MetricFormat,
    #[serde(default = "default_points")]
    pub max_points: usize,
    #[serde(default)]
    pub keys: Option<String>,
    #[serde(default)]
    pub names: Option<Vec<String>>,
}
const fn default_points() -> usize {
    2000
}
impl Default for MetricQuery {
    fn default() -> Self {
        Self {
            format: MetricFormat::Legacy,
            max_points: default_points(),
            keys: None,
            names: None,
        }
    }
}
impl MetricQuery {
    pub fn validate(&self) -> Result<()> {
        if self.max_points == 0 {
            return Err(Error::Query("INVALID_MAX_POINTS"));
        }
        if self.format == MetricFormat::Records && self.keys.is_some()
            || self.format == MetricFormat::Legacy && self.names.is_some()
        {
            return Err(Error::Query("INVALID_METRIC_FORMAT"));
        }
        if self.names.as_ref().is_some_and(|names| {
            names.len() > 256
                || names.iter().any(|name| {
                    name.trim().is_empty()
                        || name.chars().count() > 128
                        || name.chars().any(|c| c < ' ')
                })
        }) {
            return Err(Error::Query("INVALID_METRIC_NAMES"));
        }
        Ok(())
    }
}
fn schema(contract: &Value) -> Value {
    contract
        .get("metrics_schema")
        .cloned()
        .unwrap_or_else(|| json!({}))
}
fn protocol(contract: &Value) -> Option<&str> {
    contract.get("protocol_id").and_then(Value::as_str)
}
fn array<'a>(value: &'a Value, key: &str) -> Result<&'a [Value]> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or(Error::Evidence)
}
fn string_set(values: impl IntoIterator<Item = String>) -> Vec<Value> {
    values
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(Value::String)
        .collect()
}
fn indices(total: usize, count: usize) -> Vec<usize> {
    if total <= count {
        return (0..total).collect();
    }
    if count == 1 {
        return vec![total - 1];
    }
    (0..count)
        .map(|i| ((i as u128 * (total - 1) as u128) / (count - 1) as u128) as usize)
        .collect()
}
fn expansion(records: &[Value], normalized: bool) -> Result<usize> {
    let mut total = 0usize;
    for record in records {
        let fields = record.as_object().ok_or(Error::Evidence)?;
        let count = if normalized || fields.contains_key("name") {
            1
        } else {
            fields
                .iter()
                .filter(|(key, value)| {
                    value.is_number()
                        && ![
                            "step",
                            "epoch",
                            "timestamp",
                            "checkpoint_id",
                            "dataset_id",
                            "protocol_id",
                            "variant_id",
                        ]
                        .contains(&key.as_str())
                })
                .count()
        };
        total = total
            .checked_add(count)
            .filter(|total| *total <= MAX_METRIC_OBSERVATIONS)
            .ok_or(Error::OutputLimit)?;
    }
    Ok(total)
}

/// Journal append is a durable acceptance boundary, not proof that its caller
/// used the metric normalizer. Verify the canonical shape before trusting a
/// claimed status. Diagnostic records are retained unchanged, never repaired
/// into VALID records while reading.
pub fn validate_journal_observations(records: &[Value], contract: &Value) -> Result<()> {
    expansion(records, true)?;
    let schema = schema(contract);
    for record in records {
        let fields = record.as_object().ok_or(Error::Evidence)?;
        const ALLOWED: &[&str] = &[
            "name",
            "unit",
            "value",
            "status",
            "source",
            "sources",
            "step",
            "epoch",
            "timestamp",
            "protocol_id",
            "checkpoint_id",
            "dataset_id",
            "variant_id",
            "numerator",
            "denominator",
            "summary",
            "error",
            "errors",
            "invalid_value",
        ];
        if fields.keys().any(|key| !ALLOWED.contains(&key.as_str()))
            || ["name", "unit", "value", "status", "source"]
                .iter()
                .any(|key| !fields.contains_key(*key))
        {
            return Err(Error::Evidence);
        }
        let status = record["status"].as_str().ok_or(Error::Evidence)?;
        if ![
            "VALID",
            "PARTIAL",
            "MISSING",
            "FAILED",
            "UNDECLARED",
            "CONFLICTING",
        ]
        .contains(&status)
        {
            return Err(Error::Evidence);
        }
        let invalid_name = record["name"].is_null()
            && status == "FAILED"
            && record["error"] == "INVALID_METRIC_NAME";
        let name = if invalid_name {
            "<invalid-metric-name>"
        } else {
            record["name"].as_str().ok_or(Error::Evidence)?
        };
        let unit = if record["unit"].is_null() && status != "VALID" {
            "<undeclared-unit>"
        } else {
            record["unit"].as_str().ok_or(Error::Evidence)?
        };
        for key in ["value", "step", "epoch", "numerator", "denominator"] {
            if record
                .get(key)
                .is_some_and(|value| !value.is_null() && !finite(value))
            {
                return Err(Error::Evidence);
            }
        }
        for key in ["timestamp", "checkpoint_id", "dataset_id"] {
            if record
                .get(key)
                .is_some_and(|value| !value.is_null() && !value.is_string() && !finite(value))
            {
                return Err(Error::Evidence);
            }
        }
        for key in ["protocol_id", "variant_id"] {
            if record
                .get(key)
                .is_some_and(|value| !value.is_null() && !value.is_string())
            {
                return Err(Error::Evidence);
            }
        }
        if record
            .get("summary")
            .is_some_and(|value| !value.is_boolean())
            || record.get("error").is_some_and(|value| {
                !value.is_null()
                    && value
                        .as_str()
                        .is_none_or(|text| text.chars().count() > 2000)
            })
            || record.get("invalid_value").is_some_and(|value| {
                !value.is_null() && value.as_str().is_none_or(|text| text.chars().count() > 128)
            })
            || record["source"].as_str().is_none() && !record["source"].is_null()
            || record.get("sources").is_some_and(|value| {
                !value
                    .as_array()
                    .is_some_and(|items| items.iter().all(Value::is_string))
            })
            || record.get("errors").is_some_and(|value| {
                !value.as_array().is_some_and(|items| {
                    items.iter().all(|item| {
                        item.as_str().is_some_and(|code| {
                            !code.is_empty()
                                && code.len() <= 128
                                && code.bytes().all(|c| {
                                    c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_'
                                })
                        })
                    })
                })
            })
            || status == "VALID"
                && (record.get("invalid_value").is_some_and(|v| !v.is_null())
                    || record
                        .get("errors")
                        .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty)))
            || status == "CONFLICTING" && !record["value"].is_null()
        {
            return Err(Error::Evidence);
        }
        let context_text = |key: &str| {
            record
                .get(key)
                .filter(|value| !value.is_null())
                .map(python_string)
        };
        // Reuse the typed domain's text, finite numeric and context limits.
        // The two diagnostic-only placeholders are validation inputs only;
        // original nulls remain untouched in the returned scientific evidence.
        MetricRecord {
            name: name.into(),
            unit: unit.into(),
            value: record["value"].as_f64(),
            status: if status == "VALID" {
                MetricStatus::Valid
            } else {
                MetricStatus::Failed
            },
            step: record["step"].as_f64(),
            epoch: record["epoch"].as_f64(),
            timestamp: context_text("timestamp"),
            protocol_id: context_text("protocol_id"),
            checkpoint_id: context_text("checkpoint_id"),
            dataset_id: context_text("dataset_id"),
            variant_id: context_text("variant_id"),
            numerator: record["numerator"].as_f64(),
            denominator: record["denominator"].as_f64(),
            summary: record["summary"].as_bool().unwrap_or(false),
            error: None,
        }
        .validate()
        .map_err(|_| Error::Evidence)?;
        let normalized = observations(
            record,
            &schema,
            protocol(contract),
            record["source"].as_str(),
        )?;
        let [candidate] = normalized.as_slice() else {
            return Err(Error::Evidence);
        };
        if status == "VALID" && candidate["status"] != "VALID" {
            return Err(Error::Evidence);
        }
        // Missing/null optional identities are equivalent. Any coercion,
        // omitted invalid field, defaulted frozen protocol or aggregation value
        // change means this was not a canonical observation.
        for key in [
            "name",
            "unit",
            "value",
            "step",
            "epoch",
            "timestamp",
            "protocol_id",
            "checkpoint_id",
            "dataset_id",
            "variant_id",
            "numerator",
            "denominator",
            "summary",
        ] {
            if !python_equal(
                record.get(key).unwrap_or(&Value::Null),
                candidate.get(key).unwrap_or(&Value::Null),
            ) {
                return Err(Error::Evidence);
            }
        }
    }
    Ok(())
}

/// Merge exact-Attempt file observations with its journal observations. The
/// content identity excludes provenance only; protocol, variant, unit, dataset,
/// checkpoint and summary intent are never removed from deduplication.
pub fn merge_metrics(
    records: &[Value],
    live: &[Value],
    contract: &Value,
    source: Option<&str>,
) -> Result<Vec<Value>> {
    validate_journal_observations(live, contract)?;
    expansion(records, false)?
        .checked_add(expansion(live, true)?)
        .filter(|count| *count <= MAX_METRIC_OBSERVATIONS)
        .ok_or(Error::OutputLimit)?;
    let schema = schema(contract);
    let mut all = Vec::new();
    for record in records {
        all.extend(observations(record, &schema, protocol(contract), source)?);
    }
    all.extend_from_slice(live);
    let mut keys: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut merged = Vec::new();
    for record in all {
        let mut identity = record.as_object().cloned().ok_or(Error::Evidence)?;
        identity.shift_remove("source");
        let key = canonical::python_json_bytes(&Value::Object(identity))?;
        if let Some(index) = keys.get(&key) {
            merged[*index] = record;
        } else {
            keys.insert(key, merged.len());
            merged.push(record);
        }
    }
    Ok(merged)
}

/// Existing Python `_metric_payload` semantics, including complete-history
/// validation before filtering/downsampling. Source paths are explicit inputs.
/// This function retains the legacy payload; `records` is the compact format.
pub fn metric_payload(
    records: &[Value],
    query: &MetricQuery,
    contract: &Value,
    source: Option<&str>,
    source_attempt_id: Option<&str>,
    normalized: bool,
) -> Result<Value> {
    query.validate()?;
    if normalized {
        validate_journal_observations(records, contract)?;
    }
    expansion(records, normalized)?;
    let schema = schema(contract);
    let complete = metric_result(
        records,
        Some(&schema),
        protocol(contract),
        source,
        normalized,
    )?;
    let normalized_records = array(&complete, "records")?;
    let contexts = array(&complete, "contexts")?;
    let failed = normalized_records
        .iter()
        .filter(|r| r["status"] == "FAILED")
        .count();
    let conflicting = normalized_records
        .iter()
        .filter(|r| r["status"] == "CONFLICTING")
        .count();
    let incomplete = contexts.iter().filter(|c| c["state"] == "PARTIAL").count();
    let mut validation = json!({"coverage":"complete_history","total_records":records.len(),
        "failed_records":failed,"conflicting_records":conflicting,"incomplete_contexts":incomplete,
        "state":if failed > 0 || conflicting > 0 { json!("FAILED") } else if incomplete > 0 { json!("PARTIAL") } else { complete["state"].clone() }});
    if query.format == MetricFormat::Records {
        validation["total_records"] = json!(normalized_records.len());
        let missing = array(&complete, "missing_metrics")?
            .iter()
            .chain(
                contexts
                    .iter()
                    .filter_map(|c| c["missing_metrics"].as_array())
                    .flatten(),
            )
            .map(|v| v.as_str().map(str::to_owned).ok_or(Error::Evidence))
            .collect::<Result<Vec<_>>>()?;
        validation["missing_metrics"] = json!(string_set(missing));
        let available = normalized_records
            .iter()
            .filter_map(|r| r["name"].as_str())
            .collect::<BTreeSet<_>>();
        let selected = normalized_records
            .iter()
            .filter(|r| {
                query.names.as_ref().is_none_or(|names| {
                    names.is_empty() || names.iter().any(|name| r["name"] == *name)
                })
            })
            .collect::<Vec<_>>();
        let total = selected.len();
        let sampled = indices(total, query.max_points)
            .into_iter()
            .map(|i| {
                let mut record = selected[i].as_object().cloned().ok_or(Error::Evidence)?;
                record.shift_remove("source");
                record.shift_remove("sources");
                Ok(Value::Object(record))
            })
            .collect::<Result<Vec<_>>>()?;
        let missing_names = string_set(
            query
                .names
                .iter()
                .flatten()
                .filter(|n| !available.contains(n.as_str()))
                .cloned(),
        );
        return bounded(
            json!({"records":sampled,"validation":validation,"total_records":total,
            "downsampled":total > query.max_points,"missing_names":missing_names,
            "protocol_id":complete["protocol_id"],"source_attempt_id":source_attempt_id}),
        );
    }
    let wanted = query
        .keys
        .as_ref()
        .filter(|keys| !keys.is_empty())
        .map(|keys| {
            keys.split(',')
                .map(str::trim)
                .filter(|key| !key.is_empty())
                .collect::<Vec<_>>()
        });
    let mut available = BTreeSet::new();
    let mut numeric = BTreeSet::new();
    for record in records {
        for (key, value) in record.as_object().ok_or(Error::Evidence)? {
            if key == "step" || key == "timestamp" {
                continue;
            }
            if value.is_string() || value.is_number() || value.is_boolean() {
                available.insert(key.clone());
            }
            if value.is_number() || value.is_boolean() {
                numeric.insert(key.clone());
            }
        }
    }
    let sampled = indices(records.len(), query.max_points)
        .into_iter()
        .map(|i| records[i].clone())
        .collect::<Vec<_>>();
    let mut points = Vec::new();
    for record in &sampled {
        let fields = record.as_object().ok_or(Error::Evidence)?;
        let mut point = Map::new();
        point.insert(
            "step".into(),
            record.get("step").cloned().unwrap_or(Value::Null),
        );
        point.insert(
            "timestamp".into(),
            record.get("timestamp").cloned().unwrap_or(Value::Null),
        );
        let keys = wanted
            .as_ref()
            .filter(|keys| !keys.is_empty())
            .cloned()
            .unwrap_or_else(|| {
                fields
                    .keys()
                    .filter(|key| *key != "step" && *key != "timestamp")
                    .map(String::as_str)
                    .collect()
            });
        for key in keys {
            if let Some(value) = record.get(key).filter(|value| {
                finite(value) || wanted.is_some() && (value.is_string() || value.is_boolean())
            }) {
                point.insert(key.to_owned(), value.clone());
            }
        }
        points.push(Value::Object(point));
    }
    let result = metric_result(
        &sampled,
        Some(&schema),
        protocol(contract),
        source,
        normalized,
    )?;
    let keys = wanted
        .as_ref()
        .filter(|v| !v.is_empty())
        .map(|v| json!(v))
        .unwrap_or_else(|| json!(numeric));
    let missing = wanted
        .iter()
        .flatten()
        .filter(|key| !available.contains(**key))
        .map(|v| json!(v))
        .collect::<Vec<_>>();
    bounded(
        json!({"metrics":result,"evaluation":contract,"validation":validation,"points":points,
        "keys":keys,"missing_keys":missing,"total_records":records.len(),"downsampled":records.len() > sampled.len(),
        "source":source,"source_attempt_id":source_attempt_id}),
    )
}
