//! Canonical journal decisions and bounded import framing. No files, clock,
//! environment, UUID generation, network or database access.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{canonical, metrics};

mod metric_line;

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
    /// Observation scopes are not restricted to managed Attempt IDs.
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
    Ok(canonical::text(value)?)
}
pub fn digest(value: &Value) -> Result<String> {
    Ok(canonical::digest(value)?)
}

pub fn wandb_id(workspace: &str, scope: &str, frozen_route: &Value) -> Result<String> {
    Ok(digest(&json!([workspace, scope, frozen_route]))?[..32].to_owned())
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

/// Native local ingestion freezes the entire Run document identity. It cannot
/// turn an absent W&B configuration into cloud publication authorization.
pub fn native_binding(run: &Value, raw_sha256: &str) -> Result<Value> {
    if !matches!(
        run["native_creation"]["format"].as_str(),
        Some("native-run-creation.v1" | "native-cloud-run.v1" | "native-local-run.v1")
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
        "local_execution",
        "code_source_id",
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
    normalized_metric_source(record, config, "exact-attempt-worker")
}
fn normalized_metric_source(
    record: &metrics::ObservationInput,
    config: &Value,
    source: &str,
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
        json!({"kind":"metrics","observations":metrics::observations_typed(record,schema,protocol,Some(source))?}),
    )
}
/// The journal kind of a post-hoc record. Training-time readers select `metrics`,
/// so a derived record never joins a training history, a result or a validation.
pub const DERIVED_KIND: &str = "derived_metrics";
/// One derived batch: far above the few hundred records an evaluation writes,
/// and well inside the 8 MiB HTTP body once the text is JSON-escaped.
pub const DERIVED_MAX_RECORDS: usize = 5000;
pub const DERIVED_MAX_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivedRefusal {
    /// More records or bytes than one batch may hold.
    TooLarge,
    /// No record, or a line that is not one complete explicit metric record.
    Invalid,
}

/// One complete derived JSONL file, all or nothing. Each line is one explicit
/// record with `name`, `value` and `unit`, validated under the frozen schema as
/// worker records are. Unlike a worker stream, a bad line refuses the batch.
pub fn derived_observations(
    bytes: &[u8],
    config: &Value,
) -> std::result::Result<Vec<Value>, DerivedRefusal> {
    if bytes.len() > DERIVED_MAX_BYTES {
        return Err(DerivedRefusal::TooLarge);
    }
    let mut observations = Vec::new();
    for line in bytes.split(|byte| *byte == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if observations.len() >= DERIVED_MAX_RECORDS {
            return Err(DerivedRefusal::TooLarge);
        }
        if line.len() > 65536 {
            return Err(DerivedRefusal::Invalid);
        }
        let parsed = metric_line::parse(line).ok_or(DerivedRefusal::Invalid)?;
        if ["name", "value", "unit"]
            .iter()
            .any(|key| !parsed.has_field(key))
        {
            return Err(DerivedRefusal::Invalid);
        }
        let record = parsed.observation().ok_or(DerivedRefusal::Invalid)?;
        let payload = normalized_metric_source(&record, config, "derived-post-hoc")
            .map_err(|_| DerivedRefusal::Invalid)?;
        let items = payload["observations"]
            .as_array()
            .filter(|items| items.len() == 1 && items[0]["name"].is_string())
            .ok_or(DerivedRefusal::Invalid)?;
        observations.extend(items.iter().cloned());
    }
    if observations.is_empty() {
        return Err(DerivedRefusal::Invalid);
    }
    Ok(observations)
}

/// Whether a finished Run succeeded, for W&B and MLflow. A terminal lifecycle state
/// from the worker's sealed receipt wins over the provider's. Whether the metric
/// stream drained is a separate question: an undrained stream does not make a
/// successful job a failure.
pub fn succeeded(lifecycle: &Value, provider: Option<&str>) -> bool {
    match lifecycle["state"].as_str() {
        Some(state @ ("SUCCEEDED" | "FAILED" | "CANCELLED" | "TIMEOUT")) => state == "SUCCEEDED",
        _ => provider == Some("SUCCEEDED"),
    }
}

/// The journal's own end of a Run: the lifecycle the W&B publisher finishes on.
pub fn ended(lifecycle: &Value) -> bool {
    lifecycle["phase"] == "METRICS_DRAINED"
        && matches!(
            lifecycle["state"].as_str(),
            Some("SUCCEEDED" | "FAILED" | "CANCELLED" | "TIMEOUT")
        )
}

/// One exclusion list for a W&B rebuild: far above the few hundred metric names
/// a Run declares, and small enough to read back on every publication visit.
pub const REBUILD_MAX_NAMES: usize = 5000;
pub const REBUILD_MAX_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebuildRefusal {
    /// More names or bytes than one list may hold.
    TooLarge,
    /// No name, text that is not UTF-8, or a line with a control character.
    Invalid,
}

/// The exact metric names one exclusion list removes. One full name per line,
/// compared literally (`*` is an ordinary character); surrounding whitespace and
/// blank lines are ignored, and a repeated name counts once.
pub fn rebuild_names(
    text: &[u8],
) -> std::result::Result<std::collections::BTreeSet<String>, RebuildRefusal> {
    if text.len() > REBUILD_MAX_BYTES {
        return Err(RebuildRefusal::TooLarge);
    }
    let text = std::str::from_utf8(text).map_err(|_| RebuildRefusal::Invalid)?;
    let mut names = std::collections::BTreeSet::new();
    for line in text.split('\n') {
        let name = line.trim();
        if name.is_empty() {
            continue;
        }
        if name.chars().any(char::is_control) {
            return Err(RebuildRefusal::Invalid);
        }
        names.insert(name.to_owned());
        if names.len() > REBUILD_MAX_NAMES {
            return Err(RebuildRefusal::TooLarge);
        }
    }
    if names.is_empty() {
        return Err(RebuildRefusal::Invalid);
    }
    Ok(names)
}

/// The W&B id a rebuild publishes to. It follows from the id it replaces and the
/// list's SHA256, so a restart resumes the same new Run, and another list, or the
/// same list after another, gives another id.
pub fn rebuild_wandb_id(
    workspace: &str,
    scope: &str,
    previous_wandb_id: &str,
    sha256: &str,
) -> Result<String> {
    Ok(digest(&json!([
        "wandb-rebuild",
        workspace,
        scope,
        previous_wandb_id,
        sha256
    ]))?[..32]
        .to_owned())
}

/// Remove every observation named in `excluded` from one journal payload, as a
/// rebuilt W&B Run publishes it. Other kinds are unchanged. A metric row left with
/// no observation stays, so its sequence still proves the Run's history.
pub fn exclude_observations(
    payload: &mut Value,
    excluded: &std::collections::BTreeSet<String>,
) -> usize {
    let kind = &payload["kind"];
    if kind != "metrics" && kind != DERIVED_KIND {
        return 0;
    }
    let Some(observations) = payload
        .get_mut("observations")
        .and_then(Value::as_array_mut)
    else {
        return 0;
    };
    let before = observations.len();
    observations.retain(|item| {
        !item["name"]
            .as_str()
            .is_some_and(|name| excluded.contains(name))
    });
    before - observations.len()
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
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    canonical_only: bool,
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
            canonical_only: false,
        })
    }
    /// Local attachment takes explicit named observations. Wide logs
    /// need a caller-authored mapping; never guess units or turn global_step
    /// into a scientific result.
    pub fn canonical_metrics(config: Value) -> Result<Self> {
        Ok(Self {
            canonical_only: true,
            ..Self::new("metrics.jsonl", config)?
        })
    }
    /// Validate a private continuation against the original frozen schema.
    pub fn matches(&self, name: &str, config: &Value) -> bool {
        self.name == name && self.config == *config && self.chunk.len() <= Self::READLINE_BOUND
    }
    fn line(&mut self, bytes: &[u8], oversized: bool) -> Result<Event> {
        self.number += 1;
        if self.canonical_only && !oversized {
            if let Some(parsed) = metric_line::parse(bytes) {
                if ["name", "value", "unit"]
                    .iter()
                    .any(|key| !parsed.has_field(key))
                {
                    return Err(Error::Invalid(
                        "local JSONL requires explicit name, value and unit",
                    ));
                }
            }
        }
        let record = if oversized {
            None
        } else {
            metric_line::parse(bytes).and_then(|record| record.observation())
        };
        if let Some(record) = record {
            self.counts.records += 1;
            let payload = normalized_metric_source(
                &record,
                &self.config,
                if self.canonical_only {
                    "local-jsonl"
                } else {
                    "exact-attempt-worker"
                },
            )?;
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

#[derive(Clone, Debug, PartialEq)]
pub struct CursorAdvance {
    pub confirmed: i64,
    pub display_version: Option<i64>,
}

/// Check a cursor advance the trusted publisher reports after the W&B SDK
/// accepted rows through `through`. W&B is a best-effort mirror: this proves
/// only that the advance belongs to this scope's frozen route and current W&B
/// id and stays inside its journal. The HTTP/public worker ingress must never
/// expose a cursor advancement API.
pub fn cursor_advance(
    state: &Value,
    frozen_route: &Value,
    wandb_id: &str,
    through: i64,
    display: bool,
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
        || state["wandb_id"].as_str() != Some(wandb_id)
        || through < current
        || through > total
    {
        return Err(Error::Invalid("publication cursor identity or range"));
    }
    Ok(CursorAdvance {
        confirmed: through,
        display_version: display.then_some(crate::tracking::DISPLAY_VERSION),
    })
}

/// Who created a Run, read from the format its frozen manifest declares. Each
/// creator writes its own exclusive format, and anything else is authored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunKind {
    Slurm,
    Cloud,
    Local,
    Authored,
}
impl RunKind {
    pub fn of(manifest: &Map<String, Value>) -> Self {
        let format = manifest
            .get("native_creation")
            .and_then(|creation| creation.get("format"))
            .and_then(Value::as_str);
        match format {
            Some("native-run-creation.v1") => Self::Slurm,
            Some("native-cloud-run.v1") => Self::Cloud,
            Some(crate::local::FORMAT) => Self::Local,
            _ => Self::Authored,
        }
    }
}

/// Which protocol an Action's plan is bound to. A plan naming neither, or both,
/// is no backend's Submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionKind {
    Slurm,
    Cloud,
    Other,
}
impl ActionKind {
    pub fn of(snapshot: &Value) -> Self {
        let slurm = snapshot["native_slurm"]["format"] == "native-slurm-action.v1";
        let cloud = snapshot["native_cloud"]["format"] == "native-cloud-action.v1";
        match (slurm, cloud) {
            (true, false) => Self::Slurm,
            (false, true) => Self::Cloud,
            _ => Self::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_follows_the_job_not_whether_metrics_drained() {
        let done = |phase: &str, state: &str| json!({"phase":phase,"state":state});
        // A successful job whose stream did not drain (the 60 s drain ran out).
        assert!(succeeded(&done("TRAINING_COMPLETED", "SUCCEEDED"), None));
        assert!(succeeded(&done("METRICS_DRAINED", "SUCCEEDED"), Some("FAILED")));
        assert!(!succeeded(&done("TRAINING_COMPLETED", "FAILED"), Some("SUCCEEDED")));
        assert!(!succeeded(&done("METRICS_DRAINED", "TIMEOUT"), None));
        // No terminal lifecycle yet: the provider's end decides.
        assert!(succeeded(&done("TRAINING", "RUNNING"), Some("SUCCEEDED")));
        assert!(!succeeded(&done("TRAINING", "RUNNING"), Some("FAILED")));
        assert!(!succeeded(&json!({}), None));
    }

    fn run(creation: Value) -> Map<String, Value> {
        json!({"native_creation":creation})
            .as_object()
            .unwrap()
            .clone()
    }
    #[test]
    fn a_run_has_the_kind_its_format_names_and_anything_else_is_authored() {
        for (format, kind) in [
            ("native-run-creation.v1", RunKind::Slurm),
            ("native-cloud-run.v1", RunKind::Cloud),
            ("native-local-run.v1", RunKind::Local),
            ("native-run-creation.v2", RunKind::Authored),
            ("", RunKind::Authored),
        ] {
            assert_eq!(
                RunKind::of(&run(json!({"format":format}))),
                kind,
                "{format}"
            );
        }
        for creation in [
            json!(null),
            json!("native-cloud-run.v1"),
            json!({}),
            json!({"format":1}),
        ] {
            assert_eq!(RunKind::of(&run(creation)), RunKind::Authored);
        }
        assert_eq!(RunKind::of(&Map::new()), RunKind::Authored);
    }
    #[test]
    fn an_action_has_one_kind_and_a_plan_naming_neither_or_both_has_none() {
        let slurm = json!({"format":"native-slurm-action.v1"});
        let cloud = json!({"format":"native-cloud-action.v1"});
        for (snapshot, kind) in [
            (json!({"native_slurm":slurm}), ActionKind::Slurm),
            (json!({"native_cloud":cloud}), ActionKind::Cloud),
            (
                json!({"native_slurm":slurm,"native_cloud":cloud}),
                ActionKind::Other,
            ),
            (json!({"native_slurm":cloud}), ActionKind::Other),
            (json!({"native_cloud":slurm}), ActionKind::Other),
            (json!({"operation":"SUBMIT_RUN"}), ActionKind::Other),
            (json!(null), ActionKind::Other),
        ] {
            assert_eq!(ActionKind::of(&snapshot), kind, "{snapshot}");
        }
    }
    /// The creators keep these strings private, and two of those files are sealed.
    #[test]
    fn each_format_is_the_one_its_creator_writes() {
        for (source, written) in [
            (
                include_str!("slurm/action.rs"),
                r#"const FORMAT: &str = "native-slurm-action.v1";"#,
            ),
            (
                include_str!("cloud/action.rs"),
                r#"const FORMAT: &str = "native-cloud-action.v1";"#,
            ),
            (
                include_str!("cloud/run.rs"),
                r#""native_creation":{"format":"native-cloud-run.v1""#,
            ),
            (
                include_str!("first_submission/native/creation.rs"),
                r#""native_creation":{"format":"native-run-creation.v1""#,
            ),
        ] {
            assert!(source.contains(written), "{written}");
        }
    }
    #[test]
    fn a_derived_file_is_all_explicit_records_or_refused() {
        let config = json!({});
        let line = r#"{"name":"eval/fid","value":3.5,"unit":"fid","step":100}"#;
        let text = format!("{line}\n\n{line}\n");
        let observations = derived_observations(text.as_bytes(), &config).unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0]["name"], "eval/fid");
        assert_eq!(observations[0]["step"], 100);
        assert_eq!(observations[0]["status"], "VALID");
        assert_eq!(observations[0]["source"], "derived-post-hoc");
        for bad in [
            "",
            "\n",
            "not json\n",
            r#"{"name":"x","value":1}"#,
            r#"{"name":"","value":1,"unit":"fid"}"#,
        ] {
            assert_eq!(
                derived_observations(bad.as_bytes(), &config),
                Err(DerivedRefusal::Invalid),
                "{bad}"
            );
        }
        let many = format!("{line}\n").repeat(DERIVED_MAX_RECORDS + 1);
        assert_eq!(
            derived_observations(many.as_bytes(), &config),
            Err(DerivedRefusal::TooLarge)
        );
        let exact = format!("{line}\n").repeat(DERIVED_MAX_RECORDS);
        assert_eq!(
            derived_observations(exact.as_bytes(), &config)
                .unwrap()
                .len(),
            DERIVED_MAX_RECORDS
        );
        let wide = vec![b' '; DERIVED_MAX_BYTES + 1];
        assert_eq!(
            derived_observations(&wide, &config),
            Err(DerivedRefusal::TooLarge)
        );
    }
    #[test]
    fn an_exclusion_list_is_exact_names_one_per_line() {
        let names = rebuild_names(b"loss/old\n\n  eval/* \r\nloss/old\n").unwrap();
        assert_eq!(
            names.into_iter().collect::<Vec<_>>(),
            vec!["eval/*".to_owned(), "loss/old".to_owned()]
        );
        let bad: [&[u8]; 4] = [b"", b"\n \n", b"a\x01b\n", b"\xff\n"];
        for bad in bad {
            assert_eq!(rebuild_names(bad), Err(RebuildRefusal::Invalid));
        }
        let many = (0..=REBUILD_MAX_NAMES)
            .map(|i| format!("m{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            rebuild_names(many.as_bytes()),
            Err(RebuildRefusal::TooLarge)
        );
        let wide = vec![b'a'; REBUILD_MAX_BYTES + 1];
        assert_eq!(rebuild_names(&wide), Err(RebuildRefusal::TooLarge));

        let excluded = rebuild_names(b"loss/old\n").unwrap();
        let mut payload = json!({"kind":"metrics","observations":[{"name":"loss/old","value":1},{"name":"loss","value":2}]});
        assert_eq!(exclude_observations(&mut payload, &excluded), 1);
        assert_eq!(payload["observations"], json!([{"name":"loss","value":2}]));
        let mut lifecycle = json!({"kind":"lifecycle","data":{"name":"loss/old"}});
        assert_eq!(exclude_observations(&mut lifecycle, &excluded), 0);
        let a = rebuild_wandb_id("w", "s", "old", &"a".repeat(64)).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(
            a,
            rebuild_wandb_id("w", "s", "old", &"a".repeat(64)).unwrap()
        );
        assert_ne!(a, rebuild_wandb_id("w", "s", &a, &"a".repeat(64)).unwrap());
        assert_ne!(
            a,
            rebuild_wandb_id("w", "s", "old", &"b".repeat(64)).unwrap()
        );
    }
    #[test]
    fn only_a_drained_terminal_lifecycle_has_ended() {
        let lifecycle = |phase: &str, state: &str| json!({"phase":phase,"state":state});
        assert!(ended(&lifecycle("METRICS_DRAINED", "SUCCEEDED")));
        assert!(ended(&lifecycle("METRICS_DRAINED", "FAILED")));
        assert!(!ended(&lifecycle("TRAINING", "RUNNING")));
        assert!(!ended(&lifecycle("METRICS_DRAINED", "UNKNOWN")));
        assert!(!ended(&json!({})));
    }
}
