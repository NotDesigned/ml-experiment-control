//! Cached read-only scheduler evidence. Arithmetic headroom is never an
//! admission promise, and a failed refresh cannot make old data look fresh.
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub executor_id: String,
    pub status: String,
    pub data: Option<Value>,
    pub observed_at: Option<String>,
    pub checked_at: Option<String>,
    pub diagnostic: Option<String>,
    pub admission_guaranteed: bool,
}
pub fn unknown(executor: &str) -> Result<Observation, &'static str> {
    crate::identity::validate_id(executor).map_err(|_| "capacity executor")?;
    Ok(Observation {
        executor_id: executor.into(),
        status: "UNKNOWN".into(),
        data: None,
        observed_at: None,
        checked_at: None,
        diagnostic: None,
        admission_guaranteed: false,
    })
}
pub fn refresh(
    previous: Observation,
    data: Result<Value, &str>,
    now: &str,
) -> Result<Observation, &'static str> {
    crate::identity::validate_id(&previous.executor_id).map_err(|_| "capacity executor")?;
    time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
        .map_err(|_| "capacity timestamp")?;
    let mut next = previous;
    next.checked_at = Some(now.into());
    next.admission_guaranteed = false;
    match data {
        Ok(value) => {
            if !value.is_object() {
                return Err("capacity data");
            }
            next.status = "OBSERVED".into();
            next.data = Some(value);
            next.observed_at = Some(now.into());
            next.diagnostic = None;
        }
        Err(_) => {
            next.status = if next.data.is_some() {
                "STALE"
            } else {
                "UNKNOWN"
            }
            .into();
            next.diagnostic = Some("SCHEDULER_QUERY_UNAVAILABLE".into());
        }
    }
    Ok(next)
}
pub fn view(mut saved: Observation, now: &str) -> Result<Observation, &'static str> {
    let now = time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
        .map_err(|_| "capacity timestamp")?;
    if let Some(observed) = &saved.observed_at {
        let observed =
            time::OffsetDateTime::parse(observed, &time::format_description::well_known::Rfc3339)
                .map_err(|_| "capacity timestamp")?;
        if observed > now || now - observed > time::Duration::seconds(25) {
            saved.status = "STALE".into();
        }
    }
    Ok(saved)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_or_old_queries_preserve_original_evidence_without_new_freshness() {
        let first = refresh(
            unknown("slurm").unwrap(),
            Ok(serde_json::json!({"gpus":4})),
            "2026-10-29T01:00:00Z",
        )
        .unwrap();
        let failed = refresh(first.clone(), Err("timeout"), "2026-10-29T01:00:10Z").unwrap();
        assert_eq!(failed.status, "STALE");
        assert_eq!(failed.observed_at, first.observed_at);
        assert_eq!(failed.data, first.data);
        assert!(!failed.admission_guaranteed);
        assert_eq!(
            view(first.clone(), "2026-10-29T01:00:26Z").unwrap().status,
            "STALE"
        );
        assert_eq!(
            view(first, "2026-10-29T01:00:25Z").unwrap().status,
            "OBSERVED"
        );
        assert_eq!(
            refresh(
                unknown("slurm").unwrap(),
                Err("timeout"),
                "2026-10-29T01:00:10Z"
            )
            .unwrap()
            .status,
            "UNKNOWN"
        );
    }
}
