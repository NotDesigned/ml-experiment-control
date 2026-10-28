//! Optional publication decisions and disposable metric displays. No filesystem,
//! environment, clock, SDK, HTTP or database access; accepted journal records
//! remain authoritative regardless of any publication/display failure.
mod contract;
mod display;
pub mod publication;

pub use contract::{
    RoutingSettings, WandbOptions, WandbSettings, finite_parameters, finite_parameters_json,
    resolve, resolve_preparation,
};
pub use display::{
    DefinitionSpec, Projection, SummaryPatch, curve_record, definition_specs, fingerprint,
    identity, number, stem, summary_patch,
};

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid tracking contract: {0}")]
    Contract(&'static str),
    #[error("invalid metric display: {0}")]
    Display(&'static str),
    #[error(transparent)]
    Canonical(#[from] crate::canonical::CanonicalError),
}
pub type Result<T> = std::result::Result<T, Error>;

/// Audit payload and display coordinates are derived from the same accepted
/// event. Journal sequence remains transport order, never a training x-axis.
pub fn history_record(
    event: &serde_json::Value,
    display: &serde_json::Value,
) -> Result<serde_json::Value> {
    use serde_json::{Value, json};
    let mut record = curve_record(event, display)?
        .as_object()
        .cloned()
        .ok_or(Error::Display("curve mapping"))?;
    record.insert("ml_expd/sequence".into(), event["seq"].clone());
    record.insert("ml_expd/event_id".into(), event["event_id"].clone());
    record.insert("ml_expd/kind".into(), event["payload"]["kind"].clone());
    record.insert(
        "ml_expd/record".into(),
        json!(
            crate::records::encoded(&event["payload"])
                .map_err(|_| Error::Display("event encoding"))?
        ),
    );
    if event["payload"]["kind"] == "lifecycle" {
        for (key, value) in event["payload"]["data"]
            .as_object()
            .ok_or(Error::Display("lifecycle mapping"))?
        {
            record.insert(format!("ml_expd/{key}"), value.clone());
        }
    }
    Ok(Value::Object(record))
}
