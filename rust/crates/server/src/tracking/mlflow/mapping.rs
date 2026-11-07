//! Journal records as MLflow metrics and params. Pure: the same events always
//! give the same batch, so a batch sent again after an uncertain outcome is an
//! exact no-op on MLflow's SQL store, which keeps one row per identical point.
use crate::records::EventRow;
use serde_json::{Value, json};
use std::collections::BTreeSet;

/// MLflow's limits for one `runs/log-batch`.
pub const MAX_METRICS: usize = 1000;
pub const MAX_PARAMS: usize = 100;
pub const MAX_KEY: usize = 250;
pub const MAX_PARAM_VALUE: usize = 6000;
/// Below MLflow's 1 MB request limit, with room for the JSON around the points.
const BODY_BUDGET: usize = 900 * 1024;
/// Bytes of one encoded point besides its key: value, timestamp, step and syntax.
const POINT_BYTES: usize = 96;
/// The most params published for one Attempt; the rest are listed as skipped.
pub const PARAMS_LIMIT: usize = 1000;
/// How many skipped names a state keeps to show; the counts are complete.
pub const SHOWN_SKIPPED: usize = 20;

/// MLflow accepts letters, digits, `_ - . /` and spaces in a name, and refuses a
/// path that normalizes to something else (`a//b`, `./a`, `a/..`, `/a`, `a/`).
/// Colons are accepted only by some MLflow versions, so they are refused here.
pub fn valid_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= MAX_KEY
        && key
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ' '))
        && !key.starts_with('/')
        && !key.ends_with('/')
        && key.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

/// The x-axis MLflow draws: the record's `step`, otherwise its `epoch`, otherwise 0.
/// A fractional coordinate is rounded down, since MLflow steps are integers.
fn step(observation: &Value) -> i64 {
    ["step", "epoch"]
        .iter()
        .find_map(|axis| match &observation[*axis] {
            Value::Number(number) => number.as_i64().or_else(|| {
                number
                    .as_f64()
                    .filter(|value| value.is_finite() && value.abs() < 9.0e18)
                    .map(|value| value.floor() as i64)
            }),
            _ => None,
        })
        .unwrap_or(0)
}

/// The published points of one event, without repeats, and the names skipped.
fn points(event: &EventRow, base: i64, skipped: &mut Skipped) -> Vec<Value> {
    let kind = &event.payload["kind"];
    if kind != "metrics" && kind != ml_exp_core::records::DERIVED_KIND {
        return Vec::new();
    }
    let Some(observations) = event.payload["observations"].as_array() else {
        return Vec::new();
    };
    let timestamp = base.saturating_add(event.seq);
    let mut seen = BTreeSet::new();
    let mut points = Vec::new();
    for item in observations {
        // As on W&B: only a valid or partial finite measurement is a point.
        if !matches!(item["status"].as_str(), Some("VALID" | "PARTIAL")) {
            continue;
        }
        let (Some(name), Some(value)) = (
            item["name"].as_str(),
            item["value"].as_f64().filter(|v| v.is_finite()),
        ) else {
            continue;
        };
        let key = if item["summary"] == true {
            format!("summary/{name}")
        } else {
            name.to_owned()
        };
        if !valid_key(&key) {
            skipped.name(name);
            continue;
        }
        let step = step(item);
        if seen.insert((key.clone(), step, value.to_bits())) {
            points.push(json!({"key":key,"value":value,"timestamp":timestamp,"step":step}));
        }
    }
    points
}

#[derive(Default, Debug, PartialEq)]
pub struct Skipped {
    pub count: u64,
    pub names: BTreeSet<String>,
}
impl Skipped {
    fn name(&mut self, name: &str) {
        self.count += 1;
        if self.names.len() < SHOWN_SKIPPED {
            self.names.insert(name.to_owned());
        }
    }
}

/// The next delivery after `confirmed`: one or more `log-batch` bodies and the
/// journal sequence they complete. Usually one body; one event holding more
/// points than a batch may is split, and only its last body completes it.
#[derive(Debug, PartialEq)]
pub struct Batch {
    pub chunks: Vec<Vec<Value>>,
    pub through: i64,
    pub skipped: Skipped,
}

pub fn next_batch(events: &[EventRow], confirmed: i64, base: i64) -> Batch {
    let mut metrics = Vec::new();
    let mut bytes = 0;
    let mut through = confirmed;
    let mut skipped = Skipped::default();
    for event in events {
        let mut here = Skipped::default();
        let points = points(event, base, &mut here);
        let size: usize = points
            .iter()
            .map(|p| p["key"].as_str().map_or(0, str::len) + POINT_BYTES)
            .sum();
        if !metrics.is_empty()
            && (metrics.len() + points.len() > MAX_METRICS || bytes + size > BODY_BUDGET)
        {
            break;
        }
        if metrics.is_empty() && points.len() > MAX_METRICS {
            return Batch {
                chunks: points.chunks(MAX_METRICS).map(<[Value]>::to_vec).collect(),
                through: event.seq,
                skipped: here,
            };
        }
        metrics.extend(points);
        bytes += size;
        through = event.seq;
        skipped.count += here.count;
        for name in here.names {
            if skipped.names.len() < SHOWN_SKIPPED {
                skipped.names.insert(name);
            }
        }
    }
    Batch {
        chunks: if metrics.is_empty() {
            Vec::new()
        } else {
            vec![metrics]
        },
        through,
        skipped,
    }
}

/// Research parameters as MLflow params: nested objects become dotted keys,
/// strings stay as they are, and anything else is its compact JSON. A key MLflow
/// refuses, a value over its length limit or a param past the limit is skipped
/// and named in the result.
pub fn params(config: &Value) -> (Vec<(String, String)>, Vec<String>) {
    let mut params = Vec::new();
    let mut skipped = Vec::new();
    let data = &config["ml_expd"];
    flatten("", &data["parameters"], &mut params);
    if !data["seed"].is_null() {
        params.push(("ml_expd.seed".into(), compact(&data["seed"])));
    }
    let mut kept = Vec::new();
    for (key, value) in params {
        if valid_key(&key) && value.chars().count() <= MAX_PARAM_VALUE && kept.len() < PARAMS_LIMIT
        {
            kept.push((key, value));
        } else {
            skipped.push(key);
        }
    }
    (kept, skipped)
}
fn flatten(prefix: &str, value: &Value, out: &mut Vec<(String, String)>) {
    match value {
        Value::Null if prefix.is_empty() => {}
        Value::Object(map) if map.is_empty() && prefix.is_empty() => {}
        Value::Object(map) if !map.is_empty() => {
            for (key, value) in map {
                let key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten(&key, value, out);
            }
        }
        _ if prefix.is_empty() => out.push(("parameters".into(), compact(value))),
        _ => out.push((prefix.to_owned(), compact(value))),
    }
}
fn compact(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}
