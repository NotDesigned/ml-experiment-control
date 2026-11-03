//! Canonical journal decisions and bounded import framing. No files, clock,
//! environment, UUID generation, network or database access.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{canonical, metrics};

mod python_json;

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid canonical record: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Canonical(#[from] canonical::CanonicalError),
    #[error(transparent)]
    Metric(#[from] metrics::MetricError),
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scope {
    pub project: String,
    pub run: String,
    /// Legacy observation scopes were not restricted to managed Attempt IDs.
    pub attempt: Value,
}
impl Scope {
    pub fn new(project: &str, run: &str, attempt: &str) -> Self {
        Self {
            project: project.into(),
            run: run.into(),
            attempt: json!(attempt),
        }
    }
    pub fn encoded(&self) -> Result<String> {
        encoded(&json!([self.project, self.run, self.attempt]))
    }
    pub fn parse(value: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(value).map_err(|_| Error::Invalid("scope JSON"))?;
        let fields = value
            .as_array()
            .filter(|fields| fields.len() == 3)
            .ok_or(Error::Invalid("scope triple"))?;
        if fields[2].is_array() || fields[2].is_object() {
            return Err(Error::Invalid("Attempt scalar"));
        }
        Ok(Self {
            project: fields[0]
                .as_str()
                .ok_or(Error::Invalid("scope project"))?
                .into(),
            run: fields[1]
                .as_str()
                .ok_or(Error::Invalid("scope Run"))?
                .into(),
            attempt: fields[2].clone(),
        })
    }
}

pub fn encoded(value: &Value) -> Result<String> {
    String::from_utf8(canonical::python_json_bytes(value)?)
        .map_err(|_| Error::Invalid("encoded JSON UTF-8"))
}
pub fn digest(value: &Value) -> Result<String> {
    Ok(canonical::digest(value)?)
}

pub fn wandb_id(workspace: &str, scope: &str, frozen_route: &Value) -> Result<String> {
    Ok(digest(&json!([workspace, scope, frozen_route]))?[..32].to_owned())
}

pub fn run_identities(run: &str) -> [String; 3] {
    [
        run.to_owned(),
        format!("{run}--preparation"),
        format!("{run}--backfill"),
    ]
}

/// A suffix alone never proves that a Run is a publication companion.
/// The I/O caller must supply its immutable stored configuration (or the exact
/// configuration being atomically bound for a new Run).
pub fn fence_names(run: &str, frozen_config: &Value) -> Result<Vec<String>> {
    let config = frozen_config
        .as_object()
        .ok_or(Error::Invalid("bound config object"))?;
    let empty = Map::new();
    let data = match config.get("ml_expd") {
        None => &empty,
        Some(Value::Object(data)) => data,
        _ => return Err(Error::Invalid("ml_expd config object")),
    };
    let mut names = vec![run.to_owned()];
    if let (Some(kind @ ("preparation" | "backfill")), Some(parent)) = (
        data.get("record_type").and_then(Value::as_str),
        data.get("run_id").and_then(Value::as_str),
    ) {
        if run == format!("{parent}--{kind}") {
            names.push(parent.to_owned());
        }
    }
    Ok(names)
}

/// A verified archive can be imported for an historical Run without opting it
/// into cloud publication. An already bound Run must always retain its route.
pub fn historical_binding(manifest: &Value) -> Result<Value> {
    let manifest = manifest
        .as_object()
        .ok_or(Error::Invalid("import manifest object"))?;
    let definition = manifest.get("resolved_config").unwrap_or(&Value::Null);
    let definition = if manifest.contains_key("resolved_config") {
        definition
            .as_object()
            .ok_or(Error::Invalid("resolved import config object"))?
    } else {
        manifest
    };
    let evaluation = definition
        .get("evaluation")
        .cloned()
        .unwrap_or_else(|| json!({}));
    Ok(
        json!({"route":{"enabled":false,"reason":"HISTORICAL_RUN_NOT_BOUND"},"config":{"ml_expd":{"evaluation":evaluation}}}),
    )
}

/// Native local ingestion freezes the entire Run document identity. It cannot
/// turn an absent W&B configuration into cloud publication authorization.
pub fn native_binding(run: &Value, raw_sha256: &str) -> Result<Value> {
    if !matches!(
        run["native_creation"]["format"].as_str(),
        Some("native-run-creation.v1" | "native-cloud-run.v1")
    ) || !crate::slurm::helper::valid_digest(raw_sha256)
    {
        return Err(Error::Invalid("native journal Run authority"));
    }
    frozen_binding(
        run,
        raw_sha256,
        run.get("tracking_route")
            .cloned()
            .unwrap_or_else(|| json!({"enabled":false,"reason":"NATIVE_LOCAL_ONLY"})),
    )
}

/// Previously authored managed Runs can be collected into a local journal.
/// Their lack of a native creation record never authorizes a cloud route.
pub fn observed_binding(run: &Value, raw_sha256: &str) -> Result<Value> {
    if matches!(
        run["native_creation"]["format"].as_str(),
        Some("native-run-creation.v1" | "native-cloud-run.v1")
    ) {
        return native_binding(run, raw_sha256);
    }
    let mut definition = run
        .as_object()
        .cloned()
        .ok_or(Error::Invalid("Run mapping"))?;
    if let Some(resolved) = run.get("resolved_config") {
        definition.extend(
            resolved
                .as_object()
                .ok_or(Error::Invalid("resolved Run mapping"))?
                .clone(),
        );
    }
    frozen_binding(
        &Value::Object(definition),
        raw_sha256,
        json!({"enabled":false,"reason":"OBSERVED_LEGACY_LOCAL_ONLY"}),
    )
}
fn frozen_binding(run: &Value, raw_sha256: &str, route: Value) -> Result<Value> {
    if !crate::slurm::helper::valid_digest(raw_sha256) {
        return Err(Error::Invalid("Run raw identity"));
    }
    for key in ["project", "run_id"] {
        crate::identity::validate_id(run[key].as_str().ok_or(Error::Invalid("Run identity"))?)
            .map_err(|_| Error::Invalid("Run identity"))?;
    }
    let original = run
        .as_object()
        .ok_or(Error::Invalid("native Run mapping"))?;
    let mut config: Map<String, Value> = [
        "project",
        "run_id",
        "source_id",
        "image_id",
        "resources",
        "evaluation",
        "parameters",
        "resume_from",
        "seed",
        "native_datasets",
    ]
    .iter()
    .filter_map(|k| original.get(*k).map(|v| ((*k).into(), v.clone())))
    .collect();
    config.insert("native_run_sha256".into(), json!(raw_sha256));
    config.insert("runtime_id".into(), run["container"]["runtime_id"].clone());
    config.insert("backend".into(), run["backend"]["kind"].clone());
    let empty_inputs = Vec::new();
    let inputs = run
        .get("inputs")
        .map(|value| value.as_array().ok_or(Error::Invalid("Run inputs")))
        .transpose()?
        .unwrap_or(&empty_inputs)
        .iter()
        .map(|input| {
            Value::Object(
                ["asset_id", "sha256", "mount_path"]
                    .iter()
                    .filter_map(|k| input.get(*k).map(|v| ((*k).into(), v.clone())))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    config.insert("inputs".into(), json!(inputs));
    if route.get("enabled").and_then(Value::as_bool).is_none() {
        return Err(Error::Invalid("native frozen tracking route"));
    }
    Ok(json!({"route":route,"config":{"ml_expd":config}}))
}

pub fn normalized_metric(record: &Value, config: &Value) -> Result<Value> {
    normalized_metric_typed(&metrics::ObservationInput::from_json(record)?, config)
}
pub fn normalized_metric_typed(
    record: &metrics::ObservationInput,
    config: &Value,
) -> Result<Value> {
    let empty_contract = Map::new();
    let contract = evaluation(config)?.unwrap_or(&empty_contract);
    let empty = json!({});
    let schema = contract.get("metrics_schema").unwrap_or(&empty);
    let protocol = match contract.get("protocol_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.as_str()),
        _ => return Err(Error::Invalid("protocol ID string")),
    };
    Ok(
        json!({"kind":"metrics","observations":metrics::observations_typed(record,schema,protocol,Some("exact-attempt-worker"))?}),
    )
}
fn evaluation(config: &Value) -> Result<Option<&Map<String, Value>>> {
    let config = config
        .as_object()
        .ok_or(Error::Invalid("bound config object"))?;
    let data = match config.get("ml_expd") {
        None => return Ok(None),
        Some(Value::Object(data)) => data,
        _ => return Err(Error::Invalid("ml_expd config object")),
    };
    match data.get("evaluation") {
        None => Ok(None),
        Some(Value::Object(value)) => Ok(Some(value)),
        _ => Err(Error::Invalid("evaluation object")),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub event_id: String,
    pub payload: Value,
}
#[derive(Clone, Debug)]
pub struct PreparedEvent {
    pub event_id: String,
    pub body: String,
    pub checksum: String,
    pub dedup: Option<String>,
}

pub fn prepare_event(event: &Event) -> Result<PreparedEvent> {
    let payload = event
        .payload
        .as_object()
        .ok_or(Error::Invalid("payload object"))?;
    let kind = payload.get("kind").ok_or(Error::Invalid("payload kind"))?;
    let body = encoded(&event.payload)?;
    let checksum = digest(&event.payload)?;
    Ok(PreparedEvent {
        event_id: event.event_id.clone(),
        body,
        checksum: checksum.clone(),
        dedup: (kind == "metrics").then_some(checksum),
    })
}

/// Validate an already decoded typed batch. HTTP/JSONL shells should use
/// `worker_batch_json` to preserve Python float and nonfinite JSON semantics.
pub fn worker_batch(body: &Value, config: &Value) -> Result<Vec<Event>> {
    let body = body
        .as_object()
        .filter(|body| body.len() == 1)
        .ok_or(Error::Invalid("record batch object"))?;
    let records = body
        .get("records")
        .and_then(Value::as_array)
        .filter(|records| !records.is_empty() && records.len() <= 64)
        .ok_or(Error::Invalid("record batch size"))?;
    let mut result = Vec::with_capacity(records.len());
    for item in records {
        let item = item
            .as_object()
            .filter(|item| {
                item.len() == 3
                    && ["event_id", "kind", "data"]
                        .iter()
                        .all(|key| item.contains_key(*key))
            })
            .ok_or(Error::Invalid("worker record fields"))?;
        let event_id = item["event_id"]
            .as_str()
            .filter(|id| {
                !id.is_empty()
                    && id.len() <= 128
                    && id
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
            })
            .ok_or(Error::Invalid("worker event ID"))?;
        let data = item["data"]
            .as_object()
            .ok_or(Error::Invalid("worker data object"))?;
        let payload = match item["kind"].as_str() {
            Some("metrics") => normalized_metric(&item["data"], config)?,
            Some(kind @ ("lifecycle" | "checkpoints")) => {
                if data.iter().any(|(key, value)| {
                    ![
                        "phase",
                        "state",
                        "exit_code",
                        "pid",
                        "diagnostic",
                        "checkpoint_id",
                        "step",
                        "timestamp",
                    ]
                    .contains(&key.as_str())
                        || !matches!(value, Value::Null | Value::String(_) | Value::Bool(_))
                            && value
                                .as_number()
                                .is_none_or(|value| value.to_string().contains(['.', 'e', 'E']))
                }) {
                    return Err(Error::Invalid("worker lifecycle fields"));
                }
                json!({"kind":kind,"data":data})
            }
            _ => return Err(Error::Invalid("worker record kind")),
        };
        result.push(Event {
            event_id: event_id.into(),
            payload,
        });
    }
    Ok(result)
}

/// Raw JSON boundary with Python-compatible number and encoding handling.
/// Body size and authorization belong to the outer transport boundary.
pub fn worker_batch_json(body: &[u8], config: &Value) -> Result<Vec<Event>> {
    let original = python_json::parse(body).ok_or(Error::Invalid("worker JSON"))?;
    let mut events = worker_batch(&original.json(), config)?;
    let records = original
        .get("records")
        .and_then(|value| value.array())
        .ok_or(Error::Invalid("worker records"))?;
    for (event, record) in events.iter_mut().zip(records) {
        let data = record.get("data").ok_or(Error::Invalid("worker data"))?;
        if event.payload["kind"] == "metrics" {
            event.payload = normalized_metric_typed(
                &data
                    .observation()
                    .ok_or(Error::Invalid("worker metric object"))?,
                config,
            )?;
        } else if data.has_nonfinite() {
            return Err(Error::Invalid("worker lifecycle scalar types"));
        }
    }
    Ok(events)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImportCounts {
    pub records: u64,
    pub invalid_lines: u64,
}
impl ImportCounts {
    pub fn document(&self) -> Value {
        json!({"status":"COMPLETE","records":self.records,"invalid_lines":self.invalid_lines})
    }
}
#[derive(Clone, Debug)]
pub struct ImportEmission {
    pub events: Vec<Event>,
    pub counts: ImportCounts,
}

/// Pure state transition for verified output bytes. The shell commits emitted
/// batches before retaining the returned state. Replaying a transition is safe:
/// all event IDs derive from frozen content, never a clock or random value.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonlImporter {
    name: String,
    config: Value,
    number: u64,
    chunk: Vec<u8>,
    oversized: bool,
    counts: ImportCounts,
}
impl JsonlImporter {
    pub const INPUT_BOUND: usize = 4096;
    const READLINE_BOUND: usize = 65537;
    pub fn new(name: &str, config: Value) -> Result<Self> {
        if !["metrics.jsonl", "train_metrics.jsonl"].contains(&name) {
            return Err(Error::Invalid("verified metrics filename"));
        }
        Ok(Self {
            name: name.into(),
            config,
            number: 0,
            chunk: Vec::new(),
            oversized: false,
            counts: ImportCounts::default(),
        })
    }
    /// Validate a private continuation against the original frozen schema.
    pub fn matches(&self, name: &str, config: &Value) -> bool {
        self.name == name && self.config == *config && self.chunk.len() <= Self::READLINE_BOUND
    }
    fn line(&mut self, bytes: &[u8], oversized: bool) -> Result<Event> {
        self.number += 1;
        let record = if oversized {
            None
        } else {
            python_json::parse(bytes).and_then(|record| record.observation())
        };
        if let Some(record) = record {
            self.counts.records += 1;
            let payload = normalized_metric_typed(&record, &self.config)?;
            Ok(Event {
                event_id: format!("archive.{}", digest(&payload)?),
                payload,
            })
        } else {
            self.counts.invalid_lines += 1;
            let mut hex = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                const DIGITS: &[u8; 16] = b"0123456789abcdef";
                hex.push(DIGITS[(byte >> 4) as usize] as char);
                hex.push(DIGITS[(byte & 15) as usize] as char);
            }
            Ok(Event {
                event_id: format!("invalid.{}", digest(&json!([self.name, self.number, hex]))?),
                payload: json!({"kind":"diagnostic","data":{"diagnostic":"INVALID_METRIC_JSONL"}}),
            })
        }
    }
    pub fn feed(mut self, bytes: &[u8]) -> Result<(Self, ImportEmission)> {
        if bytes.len() > Self::INPUT_BOUND {
            return Err(Error::Invalid("import input chunk bound"));
        }
        let mut events = Vec::new();
        for &byte in bytes {
            self.chunk.push(byte);
            if byte == b'\n' || self.chunk.len() == Self::READLINE_BOUND {
                let chunk = std::mem::take(&mut self.chunk);
                let oversized = self.oversized || chunk.len() > 65536;
                if oversized && byte != b'\n' {
                    self.oversized = true;
                } else {
                    events.push(self.line(&chunk, oversized)?);
                    self.oversized = false;
                }
            }
        }
        let counts = self.counts.clone();
        Ok((self, ImportEmission { events, counts }))
    }
    pub fn finish(mut self) -> Result<ImportEmission> {
        let mut events = Vec::new();
        if self.oversized {
            events.push(self.line(&[], true)?);
        } else if !self.chunk.is_empty() {
            let chunk = std::mem::take(&mut self.chunk);
            events.push(self.line(&chunk, false)?);
        }
        Ok(ImportEmission {
            events,
            counts: self.counts,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemoteObservation {
    pub sequence: i64,
    pub event_id: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemoteHistoryEvidence {
    pub route: Value,
    pub wandb_id: String,
    pub through: i64,
    pub observed: Vec<RemoteObservation>,
    pub display_version: Option<i64>,
    pub display_through: Option<i64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct CursorAdvance {
    pub confirmed: i64,
    pub display_version: Option<i64>,
}

/// Validate explicit observations supplied by a trusted publisher reader. This
/// function itself makes no claim that a remote read happened. The HTTP/public
/// worker ingress must never expose a cursor advancement API.
pub fn confirmation_plan(
    state: &Value,
    frozen_route: &Value,
    events: &[(i64, String)],
    proof: &RemoteHistoryEvidence,
    error: Option<&str>,
) -> Result<CursorAdvance> {
    let current = state["confirmed"]
        .as_i64()
        .filter(|value| *value >= 0)
        .ok_or(Error::Invalid("confirmed cursor"))?;
    let total = state["total"]
        .as_i64()
        .filter(|value| *value >= current)
        .ok_or(Error::Invalid("journal total"))?;
    let enabled = frozen_route
        .get("enabled")
        .is_some_and(|value| match value {
            Value::Bool(value) => *value,
            Value::Number(value) => value.as_f64() == Some(1.0),
            _ => false,
        });
    if !enabled
        || encoded(&proof.route)? != encoded(frozen_route)?
        || state["wandb_id"].as_str() != Some(proof.wandb_id.as_str())
        || proof.through < current
        || proof.through > total
    {
        return Err(Error::Invalid("remote confirmation identity or range"));
    }
    let mut seen = std::collections::BTreeMap::new();
    for item in &proof.observed {
        if item.sequence <= 0
            || item.sequence > proof.through
            || seen.insert(item.sequence, item.event_id.as_str()).is_some()
        {
            return Err(Error::Invalid("remote sequence evidence"));
        }
    }
    let first = if current == 0 { 1 } else { current };
    let expected_count = if proof.through < first {
        0
    } else {
        proof.through - first + 1
    };
    let expected: std::collections::BTreeMap<_, _> = events
        .iter()
        .filter(|(seq, _)| *seq >= first && *seq <= proof.through)
        .map(|(seq, id)| (*seq, id.as_str()))
        .collect();
    if expected.len() as i64 != expected_count
        || seen.len() as i64 != expected_count
        || expected.iter().any(|(seq, id)| seen.get(seq) != Some(id))
    {
        return Err(Error::Invalid(
            "remote history is not a complete exact prefix",
        ));
    }
    match (proof.display_version, proof.display_through) {
        (None, None) => {}
        (Some(2 | 3), Some(through)) if through == proof.through && error.is_none() => {}
        _ => return Err(Error::Invalid("remote display acknowledgement")),
    }
    Ok(CursorAdvance {
        confirmed: proof.through,
        display_version: proof.display_version,
    })
}
