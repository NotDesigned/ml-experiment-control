//! Logs and metrics as a consumed read-only continuation. Saved state follows
//! its exact Preparation/Submission/Action; only an unsaved request may resolve
//! the current Attempt once. No continuation can switch to another Attempt.
use crate::{
    Client, Error,
    observation::{self, Reference, Selection},
};
use ml_exp_core::identity::validate_attempt;
use serde::Deserialize;
use serde_json::{Value, json};

/// Inputs to the pure read plan. Query values are encoded once, not interpreted
/// as an API path or transformed into a different metric schema.
#[derive(Clone, Deserialize)]
#[serde(tag = "view", rename_all = "lowercase", deny_unknown_fields)]
pub enum View {
    Logs {
        stream: String,
        lines: i64,
    },
    Metrics {
        max_points: i64,
        keys: Option<String>,
        names: Option<Vec<String>>,
    },
}

pub enum Step {
    Read(Read),
    Complete(Value),
}
pub struct Read {
    path: String,
    pending: Pending,
}
enum Pending {
    Identity {
        read: Box<observation::Read>,
        view: View,
        records: bool,
        saved: bool,
    },
    Current {
        reference: Box<Reference>,
        query: Query,
    },
    Attempt {
        project: String,
        run: String,
        attempt: String,
    },
}
struct Query {
    view: &'static str,
    encoded: String,
}
fn fail(message: &'static str) -> Error {
    Error::Observation(message)
}

// Python urllib.parse.urlencode uses quote_plus with RFC3986 unreserved bytes:
// unlike the WHATWG form encoder it escapes '*' and preserves '~'. Existing
// legacy keys and arbitrary Unicode metric names retain that wire spelling.
fn quote_plus(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || b"_.-~".contains(&byte) {
            encoded.push(char::from(byte));
        } else if byte == b' ' {
            encoded.push('+');
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 15)]));
        }
    }
    encoded
}
fn query(view: View, records: bool) -> Result<Query, Error> {
    let (view, pairs) = match view {
        View::Logs { stream, lines } => {
            if !matches!(stream.as_str(), "both" | "stdout" | "stderr")
                || !(1..=10000).contains(&lines)
            {
                return Err(fail("logs require stdout/stderr/both and 1..10000 lines"));
            }
            (
                "logs",
                vec![("stream", stream), ("lines", lines.to_string())],
            )
        }
        View::Metrics {
            max_points,
            keys,
            names,
        } => {
            if max_points <= 0 {
                return Err(fail("max-points must be positive"));
            }
            let mut pairs = vec![("max_points", max_points.to_string())];
            if records {
                let mut selected = names.unwrap_or_default();
                if let Some(keys) = keys {
                    selected.extend(
                        keys.split(',')
                            .map(str::trim)
                            .filter(|name| !name.is_empty())
                            .map(str::to_owned),
                    );
                }
                if selected.len() > 256
                    || selected.iter().any(|name| {
                        name.trim().is_empty()
                            || name.chars().count() > 128
                            || name.chars().any(|c| u32::from(c) < 32)
                    })
                {
                    return Err(fail("name filters must be valid metric names"));
                }
                pairs.push(("format", "records".into()));
                pairs.extend(selected.into_iter().map(|name| ("name", name)));
            } else if names.is_some() {
                return Err(fail(
                    "server lacks metric-records.v1; --keys retains the legacy field-filter behavior",
                ));
            } else if let Some(keys) = keys {
                pairs.push(("keys", keys));
            }
            ("metrics", pairs)
        }
    };
    Ok(Query {
        view,
        encoded: pairs
            .into_iter()
            .map(|(key, value)| format!("{key}={}", quote_plus(&value)))
            .collect::<Vec<_>>()
            .join("&"),
    })
}
fn attempt_read(reference: Reference, attempt: String, query: Query) -> Result<Step, Error> {
    validate_attempt(&attempt).map_err(|_| fail("choose an exact attempt-NNN identity"))?;
    let path = format!(
        "/api/attempts/{}/{}/{}?{}",
        observation::segment(&reference.project)?,
        observation::segment(&format!("{}::{attempt}", reference.run_id))?,
        query.view,
        query.encoded
    );
    Ok(Step::Read(Read {
        path,
        pending: Pending::Attempt {
            project: reference.project,
            run: reference.run_id,
            attempt,
        },
    }))
}
fn identified(
    step: observation::Step,
    view: View,
    records: bool,
    saved: bool,
) -> Result<Step, Error> {
    match step {
        observation::Step::Read(read) => Ok(Step::Read(Read {
            path: read.path().into(),
            pending: Pending::Identity {
                read: Box::new(read),
                view,
                records,
                saved,
            },
        })),
        observation::Step::Complete(reference) => {
            if saved && reference.attempt_id.is_none() {
                return Err(fail(
                    "saved state has no bound Attempt yet; use run status --state to inspect preparation",
                ));
            }
            let query = query(view, records)?;
            if let Some(attempt) = reference.attempt_id.clone() {
                attempt_read(reference, attempt, query)
            } else {
                Ok(Step::Read(Read {
                    path: format!(
                        "/api/runs/{}/{}/attempts",
                        observation::segment(&reference.project)?,
                        observation::segment(&reference.run_id)?
                    ),
                    pending: Pending::Current {
                        reference: Box::new(reference),
                        query,
                    },
                }))
            }
        }
    }
}

/// Build a pure GET plan. Saved state never falls back to a current/latest
/// Attempt; an unsaved selection resolves current once and then pins its ID.
pub fn begin(
    base: &str,
    health: &Value,
    saved: Option<&Value>,
    selection: Selection<'_>,
    view: View,
) -> Result<Step, Error> {
    let records = health
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|values| {
            values
                .iter()
                .any(|v| v.as_str() == Some("metric-records.v1"))
        });
    identified(
        observation::begin(base, saved, selection)?,
        view,
        records,
        saved.is_some(),
    )
}
impl Read {
    pub fn path(&self) -> &str {
        &self.path
    }
    /// Consume this exact continuation. HTTP failures propagate unchanged;
    /// response identities must match the original selected Run and Attempt.
    pub fn accept(self, response: Result<Value, Error>) -> Result<Step, Error> {
        let value = response?;
        match self.pending {
            Pending::Identity {
                read,
                view,
                records,
                saved,
            } => identified(read.accept(value)?, view, records, saved),
            Pending::Current { reference, query } => {
                observation::bound_response(
                    &value,
                    &[
                        ("project", json!(reference.project)),
                        ("run_id", json!(reference.run_id)),
                    ],
                    "Attempt listing differs from the requested Run",
                )?;
                let attempt = value
                    .get("current_attempt_id")
                    .and_then(Value::as_str)
                    .filter(|value| validate_attempt(value).is_ok())
                    .ok_or_else(|| fail("choose an exact attempt-NNN identity"))?
                    .to_owned();
                let identity = format!("{}::{attempt}", reference.run_id);
                let matches = value
                    .get("attempts")
                    .and_then(Value::as_array)
                    .map(|rows| {
                        rows.iter()
                            .filter(|row| {
                                row.get("attempt_id").and_then(Value::as_str) == Some(&attempt)
                                    && row.get("identity").and_then(Value::as_str)
                                        == Some(&identity)
                            })
                            .count()
                    })
                    .unwrap_or(0);
                if matches == 0 {
                    return Err(fail("current Attempt is absent from the Run listing"));
                }
                if matches != 1 {
                    return Err(fail("Run listing contains duplicate exact Attempts"));
                }
                attempt_read(*reference, attempt, query)
            }
            Pending::Attempt {
                project,
                run,
                attempt,
            } => {
                observation::bound_response(
                    &value,
                    &[
                        ("project", json!(project)),
                        ("run_id", json!(run)),
                        ("attempt_id", json!(attempt)),
                    ],
                    "read response differs from the requested exact Run/Attempt",
                )?;
                Ok(Step::Complete(value))
            }
        }
    }
}

/// The only effect shell in this module: bounded authenticated GETs, using the
/// existing transport's no-redirect/no-retry policy. Metrics/logs stay intact.
pub fn observe(
    client: &Client,
    health: &Value,
    saved: Option<&Value>,
    selection: Selection<'_>,
    view: View,
) -> Result<Value, Error> {
    let mut step = begin(client.base_url(), health, saved, selection, view)?;
    loop {
        match step {
            Step::Complete(value) => return Ok(value),
            Step::Read(read) => {
                let result = client.get(read.path());
                step = read.accept(result)?;
            }
        }
    }
}
