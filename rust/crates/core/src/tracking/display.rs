//! Derived charts, never scientific selection. Checkpoint provenance stays on
//! each point; a checkpoint change does not create a new training curve.
use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::{Error, Result};
use crate::{
    canonical::{self, JsonOptions},
    metrics::numeric_compare,
    observation::python_equal,
};

const FIELDS: [&str; 4] = ["name", "unit", "protocol_id", "dataset_id"];
const VERSION: i64 = 2;
fn object(value: &Value) -> Result<&Map<String, Value>> {
    value.as_object().ok_or(Error::Display("object required"))
}
fn get<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}
fn truth(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::Number(v) => v.as_f64() != Some(0.0),
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
    }
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(Error::Display("string field required"))
}

pub fn identity(value: &Value, variant: bool) -> Result<String> {
    object(value)?;
    let mut fields = Map::new();
    for field in FIELDS.into_iter().chain(variant.then_some("variant_id")) {
        fields.insert(field.into(), get(value, field).clone());
    }
    Ok(
        String::from_utf8(canonical::python_json_bytes(&Value::Object(fields))?)
            .expect("canonical JSON is UTF-8"),
    )
}
pub fn fingerprint(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))[..16].into()
}
pub fn stem(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "_.-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}
pub fn number(value: &Value) -> Result<bool> {
    let Some(number) = value.as_number() else {
        return Ok(false);
    };
    let finite = number.as_f64().is_some_and(f64::is_finite);
    // Python math.isfinite raises on an integer too large for conversion. Such
    // integers never arise as valid normalized metrics, but fail rather than
    // silently treating malformed legacy values as a successful empty display.
    if !finite && !number.to_string().contains(['.', 'e', 'E']) {
        return Err(Error::Display("integer exceeds finite float range"));
    }
    Ok(finite)
}
fn axis(item: &Value) -> Result<Option<&'static str>> {
    Ok(if number(get(item, "step"))? {
        Some("step")
    } else if number(get(item, "epoch"))? {
        Some("epoch")
    } else {
        None
    })
}
fn valid_curve(item: &Value) -> Result<bool> {
    let status = get(item, "status");
    if status.is_array() || status.is_object() {
        return Err(Error::Display("unhashable metric status"));
    }
    Ok(matches!(status.as_str(), Some("VALID" | "PARTIAL")))
}

#[derive(Clone, Debug)]
struct Series {
    key: String,
    context: Value,
    latest: Option<Value>,
    summary: Option<Value>,
    positions: Map<String, Value>,
}
impl Series {
    fn state(&self) -> Value {
        json!({"context":self.context,"latest":self.latest,"summary":self.summary,"positions":self.positions})
    }
}
/// Immutable-to-the-caller state transition: consuming `add` returns the next
/// disposable view. It never changes journal records or chooses a best loss.
#[derive(Clone, Debug, Default)]
pub struct Projection {
    cursor: i64,
    series: Vec<Series>,
}
impl Projection {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cursor(&self) -> i64 {
        self.cursor
    }
    /// The shell must replay from zero when requesting an older cutoff. This
    /// only selects a read position; core never reads or invents missing events.
    pub fn replay_after(&self, through: i64) -> Result<i64> {
        if through < 0 {
            return Err(Error::Display("negative journal cutoff"));
        }
        Ok(if self.cursor > through {
            0
        } else {
            self.cursor
        })
    }
    /// Prepare the disposable state for a requested cutoff. The shell still
    /// supplies accepted journal events and validates `through` against its
    /// durable MAX(sequence); this transition never invents a journal receipt.
    pub fn for_cutoff(self, through: i64) -> Result<(Self, i64)> {
        let after = self.replay_after(through)?;
        Ok((
            if self.cursor > through {
                Self::new()
            } else {
                self
            },
            after,
        ))
    }
    /// Mark an actually replayed cutoff. The effect owner must validate it
    /// against the accepted journal prefix before calling this pure function.
    pub fn at(mut self, through: i64) -> Result<Self> {
        if through < 0 {
            return Err(Error::Display("negative journal cutoff"));
        }
        self.cursor = through;
        Ok(self)
    }
    pub fn add(mut self, item: &Value, sequence: i64) -> Result<Self> {
        object(item)?;
        if !truth(get(item, "name")) || !truth(get(item, "unit")) {
            return Ok(self);
        }
        let key = identity(item, true)?;
        let index = match self.series.iter().position(|series| series.key == key) {
            Some(index) => index,
            None => {
                if self.series.len() >= 512 {
                    return Err(Error::Display("DISPLAY_CONTEXT_LIMIT"));
                }
                let context = FIELDS
                    .into_iter()
                    .chain(["variant_id"])
                    .map(|key| (key.to_owned(), get(item, key).clone()))
                    .collect::<Map<_, _>>();
                self.series.push(Series {
                    key,
                    context: Value::Object(context),
                    latest: None,
                    summary: None,
                    positions: Map::new(),
                });
                self.series.len() - 1
            }
        };
        let series = &mut self.series[index];
        let axis = axis(item)?;
        let mut point = item.clone();
        point["axis"] = json!(axis);
        point["sequence"] = json!(sequence);
        let summary = get(item, "summary") == &Value::Bool(true);
        if !summary && axis.is_some() && number(get(item, "value"))? {
            if get(item, "status").is_null() && item.get("status").is_none() {
                return Err(Error::Display("missing metric status"));
            }
            if valid_curve(item)? {
                let axis = axis.unwrap();
                let positions = series
                    .positions
                    .entry(axis)
                    .or_insert_with(|| json!([]))
                    .as_array_mut()
                    .expect("internal positions array");
                if positions.len() < 2
                    && !positions.iter().any(|v| python_equal(v, get(item, axis)))
                {
                    positions.push(get(item, axis).clone());
                }
            }
        }
        let previous = if summary {
            &series.summary
        } else {
            &series.latest
        };
        if let Some(old) = previous {
            let old_axis = get(old, "axis").as_str();
            if axis.is_some() && axis == old_axis {
                let axis = axis.unwrap();
                match numeric_compare(get(item, axis), get(old, axis))
                    .map_err(|_| Error::Display("numeric axis"))?
                {
                    Ordering::Less => return Ok(self),
                    Ordering::Equal
                        if [
                            "value",
                            "status",
                            "checkpoint_id",
                            "numerator",
                            "denominator",
                        ]
                        .iter()
                        .any(|key| !python_equal(get(old, key), get(item, key))) =>
                    {
                        point["status"] = json!("CONFLICTING");
                        point["value"] = Value::Null;
                        point["error"] = json!("CONFLICTING_LATEST_OBSERVATIONS");
                    }
                    _ => {}
                }
            } else if axis.is_none() && old_axis.is_none() {
                let error = if !python_equal(get(old, "checkpoint_id"), get(item, "checkpoint_id"))
                {
                    Some("UNORDERED_CHECKPOINTS")
                } else if ["value", "status", "numerator", "denominator"]
                    .iter()
                    .any(|key| !python_equal(get(old, key), get(item, key)))
                {
                    Some("CONFLICTING_LATEST_OBSERVATIONS")
                } else {
                    None
                };
                if let Some(error) = error {
                    point["status"] = json!("CONFLICTING");
                    point["value"] = Value::Null;
                    point["error"] = json!(error);
                }
            }
        }
        if summary {
            series.summary = Some(point);
        } else {
            series.latest = Some(point);
        }
        Ok(self)
    }
    pub fn state_value(&self) -> Value {
        let series = self
            .series
            .iter()
            .map(|series| (series.key.clone(), series.state()))
            .collect::<Map<_, _>>();
        json!({"cursor":self.cursor,"series":series})
    }
    /// Names are the existing destination-wide SQLite reservations. Allocation
    /// and collision locking belong to the server's `display_metrics` table.
    pub fn document(&self, names: &Map<String, Value>) -> Result<Value> {
        let mut variants: Vec<(String, Vec<Value>)> = Vec::new();
        for series in &self.series {
            let base = identity(&series.context, false)?;
            let variant = get(&series.context, "variant_id");
            if variant.is_array() || variant.is_object() {
                return Err(Error::Display("unhashable variant"));
            }
            let index = match variants.iter().position(|(key, _)| key == &base) {
                Some(i) => i,
                None => {
                    variants.push((base, Vec::new()));
                    variants.len() - 1
                }
            };
            if !variants[index].1.iter().any(|v| python_equal(v, variant)) {
                variants[index].1.push(variant.clone());
            }
        }
        let mut result = Map::new();
        for series in &self.series {
            let base = identity(&series.context, false)?;
            let name = names
                .get(&base)
                .and_then(Value::as_str)
                .ok_or(Error::Display("missing public metric name"))?;
            let multiple = variants
                .iter()
                .find(|(key, _)| key == &base)
                .is_some_and(|(_, v)| v.len() > 1);
            let suffix = format!("/variants/{}", fingerprint(&series.key));
            let mut row = series.state();
            row["latest"] = series
                .summary
                .as_ref()
                .or(series.latest.as_ref())
                .cloned()
                .unwrap_or(Value::Null);
            row["selection"] = json!(if series.summary.is_some() {
                "explicit_summary"
            } else {
                "latest_observation"
            });
            row["name"] = json!(name);
            row["multiple_variants"] = json!(multiple);
            row["curve"] = json!(format!("curves/{name}"));
            row["shadow"] = json!(format!("curves/{name}{suffix}"));
            row["result"] = json!(format!(
                "results/{name}{}",
                if multiple { &suffix } else { "" }
            ));
            result.insert(series.key.clone(), row);
        }
        Ok(json!({"contract":"metric-display.v2","through":self.cursor,"series":result}))
    }
}

pub fn curve_record(event: &Value, display: &Value) -> Result<Value> {
    let payload = event
        .get("payload")
        .ok_or(Error::Display("event payload"))?;
    let kind = object(payload)?
        .get("kind")
        .ok_or(Error::Display("event kind"))?;
    if kind != "metrics" {
        return Ok(json!({}));
    }
    let observations = payload
        .get("observations")
        .and_then(Value::as_array)
        .ok_or(Error::Display("metric observations"))?;
    let series = object(get(display, "series"))?;
    let mut record = Map::new();
    for item in observations {
        let Some(row) = series.get(&identity(item, true)?) else {
            continue;
        };
        if get(item, "summary") == &Value::Bool(true) || !number(get(item, "value"))? {
            continue;
        }
        if item.get("status").is_none() {
            return Err(Error::Display("missing metric status"));
        }
        if !valid_curve(item)? {
            continue;
        }
        let Some(axis) = axis(item)? else {
            continue;
        };
        let mut keys = vec![text(row, "shadow")?];
        if !truth(get(row, "multiple_variants")) {
            keys.push(text(row, "curve")?);
        }
        for key in keys {
            let key = format!("{key}{}", if axis == "epoch" { "/by_epoch" } else { "" });
            record.insert(key.clone(), get(item, "value").clone());
            record.insert(
                format!(
                    "axes/{}/{axis}",
                    key.strip_prefix("curves/").unwrap_or(&key)
                ),
                get(item, axis).clone(),
            );
        }
    }
    Ok(Value::Object(record))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DefinitionSpec {
    pub name: String,
    pub options: Value,
}
fn definition(name: impl Into<String>, options: Value) -> DefinitionSpec {
    DefinitionSpec {
        name: name.into(),
        options,
    }
}
pub fn definition_specs(display: &Value) -> Result<Vec<DefinitionSpec>> {
    let mut specs = vec![
        definition(
            "metrics/*",
            json!({"hidden":true,"summary":"none","overwrite":true}),
        ),
        definition(
            "axes/*",
            json!({"hidden":true,"summary":"none","overwrite":true}),
        ),
        definition("ml_expd/*", json!({"hidden":true})),
    ];
    for series in object(get(display, "series"))?.values() {
        for axis in ["step", "epoch"] {
            let multiple = truth(get(series, "multiple_variants"));
            for (key, visible) in [
                (text(series, "curve")?, !multiple),
                (text(series, "shadow")?, multiple),
            ] {
                let key = format!("{key}{}", if axis == "epoch" { "/by_epoch" } else { "" });
                let axis_key = format!(
                    "axes/{}/{axis}",
                    key.strip_prefix("curves/").unwrap_or(&key)
                );
                let two = get(get(series, "positions"), axis)
                    .as_array()
                    .map_or(0, Vec::len)
                    == 2;
                specs.push(definition(
                    &axis_key,
                    json!({"hidden":true,"summary":"none","overwrite":true}),
                ));
                specs.push(definition(key,json!({"step_metric":axis_key,"step_sync":false,"hidden":!(visible&&two),"summary":"none","overwrite":true})));
            }
        }
        specs.push(definition(
            text(series, "result")?,
            json!({"hidden":false,"overwrite":true}),
        ));
    }
    Ok(specs)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryPatch {
    pub delete: Vec<String>,
    pub set: Map<String, Value>,
}
/// `remote_keys` is an explicit snapshot, not an SDK handle. Applying this patch
/// never acknowledges history or metadata publication; remote reads must prove it.
pub fn summary_patch(
    display: &Value,
    remote_keys: &[String],
    terminal: bool,
    coverage: &str,
) -> Result<SummaryPatch> {
    let series = object(get(display, "series"))?;
    let mut values = Map::new();
    for row in series.values() {
        let latest = get(row, "latest");
        if terminal {
            if latest.get("status").is_none() {
                return Err(Error::Display("missing latest status"));
            }
            if get(latest, "status") == "VALID" && number(get(latest, "value"))? {
                values.insert(text(row, "result")?.into(), get(latest, "value").clone());
            }
        }
    }
    let delete = remote_keys
        .iter()
        .filter(|key| {
            ["metrics/", "axes/", "curves/", "results/"]
                .iter()
                .any(|prefix| key.starts_with(prefix))
                && !values.contains_key(*key)
        })
        .cloned()
        .collect();
    let mut definitions = Map::new();
    for row in series.values() {
        let latest = get(row, "latest");
        let mut context = object(get(row, "context"))?.clone();
        for key in [
            "checkpoint_id",
            "step",
            "epoch",
            "numerator",
            "denominator",
            "status",
            "error",
        ] {
            context.insert(key.into(), get(latest, key).clone());
        }
        if latest.get("status").is_none() {
            return Err(Error::Display("missing latest status"));
        }
        context.insert("selection".into(), get(row, "selection").clone());
        definitions.insert(text(row, "result")?.into(), Value::Object(context));
    }
    let json = canonical::encode(
        &Value::Object(definitions),
        JsonOptions {
            compact: false,
            ..JsonOptions::default()
        },
    )?;
    values.insert(
        "ml_expd/display_definitions_json".into(),
        json!(String::from_utf8(json).expect("canonical JSON is UTF-8")),
    );
    values.insert("ml_expd/display_coverage".into(), json!(coverage));
    values.insert(
        "ml_expd/display_through".into(),
        get(display, "through").clone(),
    );
    values.insert("ml_expd/display_version".into(), json!(VERSION));
    Ok(SummaryPatch {
        delete,
        set: values,
    })
}
