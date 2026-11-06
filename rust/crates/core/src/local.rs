//! Attached execution metadata and ordered JSONL ingestion. No process control,
//! scheduler identity, filesystem, network, clock or W&B SDK effects.
use crate::{
    canonical,
    metrics::MetricSchema,
    records::{Event, ImportCounts, JsonlImporter},
    tracking::WandbOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const MAX_BATCH: usize = 65536;
pub const INITIAL_CHAIN: &str = "0000000000000000000000000000000000000000000000000000000000000000";
pub const FORMAT: &str = "native-local-run.v1";
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Error(pub &'static str);
type Result<T> = std::result::Result<T, Error>;
fn object() -> Value {
    json!({})
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registration {
    pub run_id: String,
    pub execution_id: String,
    #[serde(default)]
    pub metrics_schema: MetricSchema,
    #[serde(default = "object")]
    pub evaluation: Value,
    #[serde(default = "object")]
    pub parameters: Value,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub wandb: Option<WandbOptions>,
}
impl Registration {
    pub fn validate(&self, project: &str) -> Result<()> {
        crate::identity::validate_id(project)
            .and_then(|_| crate::identity::validate_id(&self.run_id))
            .map_err(|_| Error("LOCAL_IDENTITY_INVALID"))?;
        crate::submission::validate_submission_token(&self.execution_id)
            .map_err(|_| Error("LOCAL_IDENTITY_INVALID"))?;
        if !self.evaluation.is_object()
            || !self.parameters.is_object()
            || serde_json::to_vec(self)
                .map_err(|_| Error("LOCAL_DEFINITION_INVALID"))?
                .len()
                > MAX_BATCH
        {
            return Err(Error("LOCAL_DEFINITION_INVALID"));
        }
        MetricSchema::from_value(&self.metrics_schema.to_value())
            .map_err(|_| Error("LOCAL_METRICS_SCHEMA_INVALID"))?;
        crate::tracking::finite_parameters(&self.parameters)
            .map_err(|_| Error("LOCAL_PARAMETERS_INVALID"))?;
        Ok(())
    }
    pub fn manifest(&self, project: &str, route: Value, now: &str) -> Result<Value> {
        self.validate(project)?;
        let mut evaluation = self.evaluation.clone();
        evaluation["metrics_schema"] = self.metrics_schema.to_value();
        if evaluation.get("protocol_id").is_none() {
            evaluation["protocol_id"] = json!(
                canonical::digest(&evaluation).map_err(|_| Error("LOCAL_DEFINITION_INVALID"))?
            );
        }
        if !evaluation["protocol_id"].as_str().is_some_and(|s| {
            !s.is_empty() && s.chars().count() <= 2000 && !s.chars().any(char::is_control)
        }) {
            return Err(Error("LOCAL_PROTOCOL_INVALID"));
        }
        Ok(
            json!({"schema_version":1,"project":project,"run_id":self.run_id,"created_at":now,
            "backend":{"kind":"local"},"local_execution":{"execution_id":self.execution_id,"mode":"attached","process_control":false},
            "parameters":self.parameters,"seed":self.seed,"evaluation":evaluation,"tracking_route":route,
            "native_creation":{"format":FORMAT,"request_sha256":canonical::digest(&json!(self)).map_err(|_| Error("LOCAL_DEFINITION_INVALID"))?}}),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Finish {
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Packet {
    pub execution_id: String,
    pub offset: u64,
    pub previous_chain_sha256: String,
    pub bytes: Vec<u8>,
    #[serde(default)]
    pub finish: Option<Finish>,
}
impl Packet {
    pub fn digest(&self) -> Result<String> {
        crate::submission::validate_submission_token(&self.execution_id)
            .map_err(|_| Error("LOCAL_IDENTITY_INVALID"))?;
        crate::sources::validate_archive_digest(&self.previous_chain_sha256)
            .map_err(|_| Error("LOCAL_STREAM_INVALID"))?;
        if self.bytes.len() > MAX_BATCH || self.bytes.is_empty() && self.finish.is_none() {
            return Err(Error("LOCAL_BATCH_INVALID"));
        }
        canonical::digest(&json!(self)).map_err(|_| Error("LOCAL_BATCH_INVALID"))
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cursor {
    pub offset: u64,
    pub chain_sha256: String,
    pub finished: Option<Finish>,
    pub importer: JsonlImporter,
    pub counts: ImportCounts,
}
impl Cursor {
    pub fn new(config: Value) -> Result<Self> {
        Ok(Self {
            offset: 0,
            chain_sha256: INITIAL_CHAIN.into(),
            finished: None,
            importer: JsonlImporter::canonical_metrics(config)
                .map_err(|_| Error("LOCAL_DEFINITION_INVALID"))?,
            counts: ImportCounts::default(),
        })
    }
    /// Replay is handled by the shell's exact packet receipt. This reducer only
    /// accepts the next bytes; its event IDs remain stable across crash replay.
    pub fn advance(&self, packet: &Packet) -> Result<(Self, Vec<Event>)> {
        let digest = packet.digest()?;
        if self.finished.is_some() {
            return Err(Error("LOCAL_STREAM_FINISHED"));
        }
        if packet.offset != self.offset || packet.previous_chain_sha256 != self.chain_sha256 {
            return Err(Error("LOCAL_STREAM_ORDER_MISMATCH"));
        }
        let mut next = self.clone();
        let mut events = Vec::new();
        for chunk in packet.bytes.chunks(JsonlImporter::INPUT_BOUND) {
            let (importer, emission) = next
                .importer
                .feed(chunk)
                .map_err(|_| Error("LOCAL_METRIC_IMPORT_FAILED"))?;
            next.importer = importer;
            next.counts = emission.counts;
            events.extend(emission.events);
        }
        if let Some(finish) = packet.finish {
            let emission = next
                .importer
                .clone()
                .finish()
                .map_err(|_| Error("LOCAL_METRIC_IMPORT_FAILED"))?;
            next.counts = emission.counts;
            events.extend(emission.events);
            events.push(Event {event_id:format!("local-finish.{digest}"),payload:json!({"kind":"lifecycle","data":{"phase":"METRICS_DRAINED","state":finish,"exit_code":if finish==Finish::Succeeded {Some(0)}else{None}}})});
            next.finished = Some(finish);
        }
        next.offset = next
            .offset
            .checked_add(packet.bytes.len() as u64)
            .ok_or(Error("LOCAL_STREAM_INVALID"))?;
        next.chain_sha256 = digest;
        Ok((next, events))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packet(cursor: &Cursor, bytes: &[u8], finish: Option<Finish>) -> Packet {
        Packet {
            execution_id: "1".repeat(32),
            offset: cursor.offset,
            previous_chain_sha256: cursor.chain_sha256.clone(),
            bytes: bytes.into(),
            finish,
        }
    }
    #[test]
    fn append_partial_and_finish_preserve_scientific_coordinates_and_replay_identity() {
        let c = Cursor::new(json!({"ml_expd":{"evaluation":{}}})).unwrap();
        let p=packet(&c,b"{\"name\":\"train_loss\",\"unit\":\"nats/token\",\"step\":37,\"epoch\":2.5,\"value\":4.2}",None);
        let (n, events) = c.advance(&p).unwrap();
        assert!(events.is_empty());
        let p = packet(&n, b"\n", Some(Finish::Succeeded));
        let (done, events) = n.advance(&p).unwrap();
        assert_eq!(events.len(), 2);
        let data = &events[0].payload;
        assert_eq!(data["observations"][0]["step"], 37);
        assert_eq!(data["observations"][0]["epoch"], 2.5);
        assert_eq!(data["observations"][0]["source"], "local-jsonl");
        assert_eq!(events[0].event_id, n.advance(&p).unwrap().1[0].event_id);
        assert!(done.advance(&p).is_err());
        let mut bad = packet(&n, b"\n", None);
        bad.previous_chain_sha256 = INITIAL_CHAIN.into();
        assert!(n.advance(&bad).is_err());
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    fn request() -> Registration {
        serde_json::from_value(json!({"run_id":"local","execution_id":"a".repeat(32),"wandb":{"enabled":false},"parameters":{"lr":0.01}})).unwrap()
    }
    fn packet(c: &Cursor, bytes: &[u8], finish: Option<Finish>) -> Packet {
        Packet {
            execution_id: "a".repeat(32),
            offset: c.offset,
            previous_chain_sha256: c.chain_sha256.clone(),
            bytes: bytes.into(),
            finish,
        }
    }
    #[test]
    fn protocol_tracks_evaluation_not_training_hyperparameters_and_local_is_explicit() {
        let a = request();
        let mut b = a.clone();
        b.parameters = json!({"lr":0.02});
        b.seed = 7;
        let ma = a.manifest("demo", json!({"enabled":false}), "now").unwrap();
        let mb = b.manifest("demo", json!({"enabled":false}), "now").unwrap();
        assert_eq!(
            ma["evaluation"]["protocol_id"],
            mb["evaluation"]["protocol_id"]
        );
        assert_eq!(ma["backend"]["kind"], "local");
        assert!(ma.get("source_id").is_none());
        assert!(ma.get("container").is_none());
        b.evaluation = json!({"window":256});
        assert_ne!(
            ma["evaluation"]["protocol_id"],
            b.manifest("demo", json!({"enabled":false}), "now").unwrap()["evaluation"]["protocol_id"]
        );
    }
    #[test]
    fn bounded_packets_and_canonical_names_units_failures_and_finish() {
        let c = Cursor::new(json!({"ml_expd":{"evaluation":{}}})).unwrap();
        assert!(
            c.advance(&packet(&c, &vec![b'x'; MAX_BATCH + 1], None))
                .is_err()
        );
        assert!(c.advance(&packet(&c, b"", None)).is_err());
        assert!(
            c.advance(&packet(
                &c,
                b"{\"global_step\":10,\"train/loss\":4.2}\n",
                None
            ))
            .is_err()
        );
        let bytes=b"{\"name\":\"train/loss\",\"unit\":\"nats/token\",\"value\":NaN,\"step\":3}\nnot-json\n";
        let (next, events) = c.advance(&packet(&c, bytes, Some(Finish::Failed))).unwrap();
        assert_eq!(next.counts.records, 1);
        assert_eq!(next.counts.invalid_lines, 1);
        assert_eq!(events[0].payload["observations"][0]["name"], "train/loss");
        assert_eq!(events[0].payload["observations"][0]["status"], "FAILED");
        assert!(events[0].payload["observations"][0]["value"].is_null());
        assert_eq!(
            events.last().unwrap().payload["data"]["phase"],
            "METRICS_DRAINED"
        );
        assert!(next.advance(&packet(&next, b"x", None)).is_err());
    }
}
