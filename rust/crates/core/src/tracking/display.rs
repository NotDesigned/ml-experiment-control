//! Derived charts, never scientific selection. Checkpoint provenance stays on
//! each point; a checkpoint change does not create a new training curve.
use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{Error, Result};
use crate::{canonical, metrics::numeric_compare, observation::numeric_tolerant_equal};

const FIELDS: [&str; 5] = ["name", "unit", "protocol_id", "dataset_id", "variant_id"];
pub const DISPLAY_VERSION: i64 = 4;
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

pub fn identity(value: &Value) -> Result<String> {
    object(value)?;
    let mut fields = Map::new();
    for field in FIELDS {
        fields.insert(field.into(), get(value, field).clone());
    }
    Ok(canonical::text(&Value::Object(fields))?)
}
pub fn number(value: &Value) -> Result<bool> {
    let Some(number) = value.as_number() else {
        return Ok(false);
    };
    let finite = number.as_f64().is_some_and(f64::is_finite);
    // An integer too large for a float has no finite conversion. Such integers
    // never arise as valid normalized metrics, but fail rather than
    // silently treating malformed values as a successful empty display.
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
        let key = identity(item)?;
        let index = match self.series.iter().position(|series| series.key == key) {
            Some(index) => index,
            None => {
                let context = FIELDS
                    .into_iter()
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
                    && !positions
                        .iter()
                        .any(|v| numeric_tolerant_equal(v, get(item, axis)))
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
                        .any(|key| !numeric_tolerant_equal(get(old, key), get(item, key))) =>
                    {
                        point["status"] = json!("CONFLICTING");
                        point["value"] = Value::Null;
                        point["error"] = json!("CONFLICTING_LATEST_OBSERVATIONS");
                    }
                    _ => {}
                }
            } else if axis.is_none() && old_axis.is_none() {
                let error = if !numeric_tolerant_equal(
                    get(old, "checkpoint_id"),
                    get(item, "checkpoint_id"),
                ) {
                    Some("UNORDERED_CHECKPOINTS")
                } else if ["value", "status", "numerator", "denominator"]
                    .iter()
                    .any(|key| !numeric_tolerant_equal(get(old, key), get(item, key)))
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
    /// W&B keys belong to the caller. Scientific contexts stay in the journal;
    /// sharing a name intentionally shares one W&B metric, including variants.
    pub fn client_document(&self) -> Result<Value> {
        let mut series = Map::new();
        for item in &self.series {
            let name = text(&item.context, "name")?;
            let mut row = item.state();
            row["latest"] = item
                .summary
                .as_ref()
                .or(item.latest.as_ref())
                .cloned()
                .unwrap_or(Value::Null);
            row["selection"] = json!(if item.summary.is_some() {
                "explicit_summary"
            } else {
                "latest_observation"
            });
            row["name"] = json!(name);
            row["curve"] = json!(name);
            row["result"] = json!(name);
            series.insert(item.key.clone(), row);
        }
        Ok(json!({"contract":"metric-display.v4","through":self.cursor,"series":series}))
    }
}

fn require_contract(display: &Value) -> Result<()> {
    if get(display, "contract").as_str() != Some("metric-display.v4") {
        return Err(Error::Display("unsupported metric display contract"));
    }
    Ok(())
}

pub fn curve_record(event: &Value, display: &Value) -> Result<Value> {
    require_contract(display)?;
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
        let Some(row) = series.get(&identity(item)?) else {
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
        let name = text(row, "curve")?;
        if record
            .insert(name.into(), get(item, "value").clone())
            .is_some()
        {
            return Err(Error::Display("DISPLAY_NAME_CONFLICT"));
        }
        for axis in ["step", "epoch"] {
            if number(get(item, axis))? {
                record.insert(
                    format!("ml_expd/axes/{name}/{axis}"),
                    get(item, axis).clone(),
                );
            }
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
    require_contract(display)?;
    let series = object(get(display, "series"))?;
    let mut names = BTreeSet::new();
    let mut specs = vec![definition(
        "ml_expd/*",
        json!({"hidden":true,"summary":"none","overwrite":true}),
    )];
    for row in series.values() {
        let name = text(row, "curve")?;
        if !names.insert(name) {
            continue;
        }
        let mut options = json!({"hidden":false,"summary":"none","overwrite":true});
        if let Some(axis) = ["step", "epoch"].into_iter().find(|axis| {
            series.values().any(|r| {
                get(r, "curve").as_str() == Some(name)
                    && get(get(r, "positions"), axis)
                        .as_array()
                        .is_some_and(|a| !a.is_empty())
            })
        }) {
            let axis_key = format!("ml_expd/axes/{name}/{axis}");
            specs.push(definition(
                &axis_key,
                json!({"hidden":true,"summary":"none","overwrite":true}),
            ));
            options["step_metric"] = json!(axis_key);
            options["step_sync"] = json!(false);
        }
        specs.push(definition(name, options));
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
    coverage: &str,
) -> Result<SummaryPatch> {
    require_contract(display)?;
    let series = object(get(display, "series"))?;
    // Several contexts can intentionally use the same client name. Choose its
    // explicit summary, otherwise the last accepted observation, in journal order.
    let mut selected = Map::new();
    for row in series.values() {
        let name = text(row, "result")?;
        let rank = |row: &Value| {
            (
                get(row, "selection") == "explicit_summary",
                get(get(row, "latest"), "sequence").as_i64().unwrap_or(0),
            )
        };
        if selected.get(name).is_none_or(|old| rank(row) > rank(old)) {
            selected.insert(name.into(), row.clone());
        }
    }
    let mut values = Map::new();
    for row in selected.values() {
        let latest = get(row, "latest");
        if latest.get("status").is_none() {
            return Err(Error::Display("missing latest status"));
        }
        if get(latest, "status") == "VALID" && number(get(latest, "value"))? {
            values.insert(text(row, "result")?.into(), get(latest, "value").clone());
        }
    }
    let delete = remote_keys
        .iter()
        .filter(|key| selected.contains_key(key.as_str()) && !values.contains_key(*key))
        .cloned()
        .collect();
    let mut definitions = Map::new();
    for row in selected.values() {
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
        context.insert("selection".into(), get(row, "selection").clone());
        definitions.insert(text(row, "result")?.into(), Value::Object(context));
    }
    values.insert(
        "ml_expd/display_definitions_json".into(),
        json!(canonical::text(&Value::Object(definitions))?),
    );
    values.insert("ml_expd/display_coverage".into(), json!(coverage));
    values.insert(
        "ml_expd/display_through".into(),
        get(display, "through").clone(),
    );
    values.insert("ml_expd/display_version".into(), json!(DISPLAY_VERSION));
    Ok(SummaryPatch {
        delete,
        set: values,
    })
}
