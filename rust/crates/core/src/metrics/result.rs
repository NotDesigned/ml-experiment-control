//! Context-local aggregation and conflict evidence; no scientific ranking.
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use super::{MetricError, ObservationInput, finite, observations, observations_typed};

const IDENTITY: [&str; 6] = [
    "protocol_id",
    "checkpoint_id",
    "dataset_id",
    "variant_id",
    "epoch",
    "step",
];
const NULL: Value = Value::Null;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Atom {
    Null,
    Number(String),
    Text(String),
}

fn atom(value: &Value) -> Result<Atom, MetricError> {
    Ok(match value {
        Value::Null => Atom::Null,
        Value::Bool(value) => Atom::Number(if *value { "1" } else { "0" }.to_owned()),
        Value::String(value) => Atom::Text(value.clone()),
        Value::Number(number) => {
            let text = number.to_string();
            let value = if !text.contains(['.', 'e', 'E']) {
                if text == "-0" { "0".to_owned() } else { text }
            } else {
                let value = number
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or(MetricError::Record)?;
                if value == 0.0 {
                    "0".to_owned()
                } else if value.fract() == 0.0 {
                    format!("{value:.0}")
                } else {
                    format!("fraction:{:016x}", value.to_bits())
                }
            };
            Atom::Number(value)
        }
        _ => return Err(MetricError::Record),
    })
}

fn integer_cmp(left: &str, right: &str) -> Ordering {
    let negative_left = left.starts_with('-');
    let negative_right = right.starts_with('-');
    if negative_left != negative_right {
        return if negative_left {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let left = left.trim_start_matches('-');
    let right = right.trim_start_matches('-');
    let result = left.len().cmp(&right.len()).then_with(|| left.cmp(right));
    if negative_left {
        result.reverse()
    } else {
        result
    }
}

/// Python number ordering, including bools and exact integer/float comparison.
/// Integer magnitudes are not rounded through f64.
pub fn numeric_compare(left: &Value, right: &Value) -> Result<Ordering, MetricError> {
    let Atom::Number(left_key) = atom(left)? else {
        return Err(MetricError::Record);
    };
    let Atom::Number(right_key) = atom(right)? else {
        return Err(MetricError::Record);
    };
    let left_fraction = left_key.starts_with("fraction:");
    let right_fraction = right_key.starts_with("fraction:");
    Ok(match (left_fraction, right_fraction) {
        (false, false) => integer_cmp(&left_key, &right_key),
        (true, true) => left
            .as_f64()
            .unwrap()
            .partial_cmp(&right.as_f64().unwrap())
            .unwrap(),
        (true, false) => {
            let floor = format!("{:.0}", left.as_f64().unwrap().floor());
            let result = integer_cmp(&floor, &right_key);
            if result == Ordering::Equal {
                Ordering::Greater
            } else {
                result
            }
        }
        (false, true) => numeric_compare(right, left)?.reverse(),
    })
}

fn field<'a>(item: &'a Value, key: &str) -> &'a Value {
    item.get(key).unwrap_or(&NULL)
}

fn identity(item: &Value, fields: &[&str]) -> Result<Vec<Atom>, MetricError> {
    fields.iter().map(|key| atom(field(item, key))).collect()
}

fn equal(a: &Value, b: &Value) -> bool {
    if (a.is_number() || a.is_boolean()) && (b.is_number() || b.is_boolean()) {
        return numeric_equal(a, b);
    }
    a == b
}

pub fn numeric_equal(left: &Value, right: &Value) -> bool {
    if !(left.is_number() || left.is_boolean()) || !(right.is_number() || right.is_boolean()) {
        return false;
    }
    match (atom(left), atom(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn sources(previous: Option<&Value>, current: &Value) -> Result<Value, MetricError> {
    let mut values = Vec::new();
    if let Some(previous) = previous.and_then(|item| item.get("sources")) {
        values.extend(
            previous
                .as_array()
                .ok_or(MetricError::Record)?
                .iter()
                .cloned(),
        );
    }
    let source = current.get("source").ok_or(MetricError::Record)?;
    if truthy(source) {
        values.push(source.clone());
    }
    let mut seen = BTreeSet::new();
    let mut unique = Vec::new();
    for value in values {
        if seen.insert(atom(&value)?) {
            unique.push(value);
        }
    }
    if unique.iter().all(Value::is_string) {
        unique.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
    } else if unique
        .iter()
        .all(|value| value.is_number() || value.is_boolean())
    {
        unique.sort_by(|a, b| numeric_compare(a, b).unwrap());
    } else if unique.len() > 1 {
        return Err(MetricError::Record);
    }
    Ok(Value::Array(unique))
}

fn result_from_observations(
    records: Vec<Value>,
    schema: &Value,
    protocol_id: Option<&str>,
) -> Result<Value, MetricError> {
    let mut by_key: BTreeMap<Vec<Atom>, usize> = BTreeMap::new();
    let mut items: Vec<Value> = Vec::new();
    for mut item in records {
        if !item.is_object()
            || ["name", "unit", "status", "source"]
                .iter()
                .any(|key| item.get(key).is_none())
        {
            return Err(MetricError::Record);
        }
        let mut key = identity(
            &item,
            &[
                "name",
                "unit",
                "protocol_id",
                "checkpoint_id",
                "dataset_id",
                "variant_id",
                "epoch",
                "step",
            ],
        )?;
        key.push(atom(item.get("summary").unwrap_or(&Value::Bool(false)))?);
        let previous = by_key.get(&key).copied();
        item["sources"] = sources(previous.map(|index| &items[index]), &item)?;
        if previous.is_some_and(|index| {
            ["value", "status", "numerator", "denominator"]
                .iter()
                .any(|key| !equal(field(&items[index], key), field(&item, key)))
        }) {
            item["status"] = json!("CONFLICTING");
            item["error"] = json!("CONFLICTING_OBSERVATIONS");
            item["value"] = Value::Null;
        }
        if let Some(index) = previous {
            items[index] = item;
        } else {
            by_key.insert(key, items.len());
            items.push(item);
        }
    }
    let unit_fields = [
        "name",
        "protocol_id",
        "checkpoint_id",
        "dataset_id",
        "variant_id",
        "epoch",
        "step",
    ];
    let mut units: BTreeMap<Vec<Atom>, BTreeSet<Atom>> = BTreeMap::new();
    for item in &items {
        units
            .entry(identity(item, &unit_fields)?)
            .or_default()
            .insert(atom(field(item, "unit"))?);
    }
    for item in &mut items {
        if units[&identity(item, &unit_fields)?].len() > 1 {
            item["status"] = json!("CONFLICTING");
            item["error"] = json!("CONFLICTING_UNITS");
            item["value"] = Value::Null;
        }
    }
    let empty = Map::new();
    let definitions = match schema.get("definitions") {
        None => &empty,
        Some(Value::Object(value)) => value,
        _ => return Err(MetricError::Schema),
    };
    let mut required = Vec::new();
    for (name, definition) in definitions {
        let definition = definition.as_object().ok_or(MetricError::Schema)?;
        if definition.get("required").is_some_and(truthy) {
            required.push(name.clone());
        }
    }
    let mut context_indices: BTreeMap<Vec<Atom>, usize> = BTreeMap::new();
    let mut contexts: Vec<(Vec<Atom>, Vec<Value>)> = Vec::new();
    for item in &items {
        let key = identity(item, &IDENTITY)?;
        if let Some(index) = context_indices.get(&key) {
            contexts[*index].1.push(item.clone());
        } else {
            context_indices.insert(key.clone(), contexts.len());
            contexts.push((key, vec![item.clone()]));
        }
    }
    let mut groups = Vec::new();
    for (_, members) in &contexts {
        let context: Map<_, _> = IDENTITY
            .iter()
            .map(|key| ((*key).to_owned(), field(&members[0], key).clone()))
            .collect();
        let mut missing = Vec::new();
        for name in &required {
            if members
                .iter()
                .any(|item| item["status"] == "VALID" && item["name"].as_str() == Some(name))
            {
                continue;
            }
            let requirement = definitions[name]
                .get("required_when")
                .filter(|value| truthy(value));
            let applies = match requirement {
                None => true,
                Some(Value::Object(values)) => values
                    .iter()
                    .all(|(key, value)| equal(context.get(key).unwrap_or(&NULL), value)),
                _ => return Err(MetricError::Schema),
            };
            if applies {
                missing.push(name.clone());
            }
        }
        let failed = members
            .iter()
            .any(|item| item["status"] == "FAILED" || item["status"] == "CONFLICTING");
        let state = if failed {
            "FAILED"
        } else if !missing.is_empty() || members.iter().any(|item| item["status"] != "VALID") {
            "PARTIAL"
        } else {
            "COMPLETE"
        };
        groups.push(
            json!({"identity":context,"state":state,"missing_metrics":missing,"records":members}),
        );
    }
    let bases: BTreeSet<_> = contexts.iter().map(|(key, _)| key[..4].to_vec()).collect();
    let current = if bases.len() == 1 {
        if groups
            .iter()
            .all(|group| finite(&group["identity"]["step"]))
        {
            let mut selected = 0;
            for index in 1..groups.len() {
                let epoch = |group: &Value| {
                    if truthy(&group["identity"]["epoch"]) {
                        group["identity"]["epoch"].clone()
                    } else {
                        json!(0)
                    }
                };
                let order = numeric_compare(&epoch(&groups[index]), &epoch(&groups[selected]))?;
                if order == Ordering::Greater
                    || order == Ordering::Equal
                        && numeric_compare(
                            &groups[index]["identity"]["step"],
                            &groups[selected]["identity"]["step"],
                        )? == Ordering::Greater
                {
                    selected = index;
                }
            }
            Some(groups[selected].clone())
        } else if groups.len() == 1 {
            Some(groups[0].clone())
        } else {
            None
        }
    } else {
        None
    };
    let aggregate = if groups.iter().any(|group| group["state"] == "FAILED") {
        "FAILED"
    } else if groups.iter().any(|group| group["state"] == "PARTIAL") {
        "PARTIAL"
    } else {
        "COMPLETE"
    };
    let missing: Vec<_> = required
        .iter()
        .filter(|name| {
            groups.iter().any(|group| {
                group["missing_metrics"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|value| value.as_str() == Some(name))
            })
        })
        .cloned()
        .collect();
    let state = current
        .as_ref()
        .map(|current| current["state"].clone())
        .unwrap_or_else(|| {
            json!(if items.is_empty() {
                "NOT_OBSERVED"
            } else {
                aggregate
            })
        });
    let missing = current
        .as_ref()
        .map(|current| current["missing_metrics"].clone())
        .unwrap_or_else(|| {
            json!(if items.is_empty() {
                &required
            } else {
                &missing
            })
        });
    Ok(
        json!({"state":state,"required_metrics":required,"missing_metrics":missing,"records":items,"contexts":groups,"current":current,"protocol_id":protocol_id}),
    )
}

pub fn metric_result(
    records: &[Value],
    schema: Option<&Value>,
    protocol_id: Option<&str>,
    source: Option<&str>,
    normalized: bool,
) -> Result<Value, MetricError> {
    let empty = json!({"definitions":{}});
    let schema = schema.filter(|schema| truthy(schema)).unwrap_or(&empty);
    if !schema.is_object() {
        return Err(MetricError::Schema);
    }
    let mut items = Vec::new();
    for record in records {
        if normalized {
            items.push(record.clone());
        } else {
            items.extend(observations(record, schema, protocol_id, source)?);
        }
    }
    result_from_observations(items, schema, protocol_id)
}

pub fn metric_result_typed(
    records: &[ObservationInput],
    schema: Option<&Value>,
    protocol_id: Option<&str>,
    source: Option<&str>,
) -> Result<Value, MetricError> {
    let empty = json!({"definitions":{}});
    let schema = schema.filter(|schema| truthy(schema)).unwrap_or(&empty);
    let mut items = Vec::new();
    for record in records {
        items.extend(observations_typed(record, schema, protocol_id, source)?);
    }
    result_from_observations(items, schema, protocol_id)
}
