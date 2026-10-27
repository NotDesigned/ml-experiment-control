//! Polling policy as pure state. The shell only calls the existing read-only
//! status flow, reports an update and sleeps; it never stops or retries a job.
use crate::{Client, Error, observation::Selection, status};
use ml_exp_core::identity::validate_attempt;
use serde_json::Value;
use std::time::{Duration, Instant};

const INTERVAL: Duration = Duration::from_secs(5);

pub struct Watch {
    duration: Duration,
    elapsed: Duration,
    project: Option<String>,
    run: Option<String>,
    attempt: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Settled,
    ObservationWindowEnded,
}
pub enum Decision {
    Wait {
        watch: Watch,
        delay: Duration,
        value: Value,
    },
    Complete {
        value: Value,
        reason: End,
    },
}
fn fail(message: &'static str) -> Error {
    Error::Observation(message)
}
fn selected(current: &mut Option<String>, incoming: &str) -> Result<(), Error> {
    if current.as_deref().is_some_and(|old| old != incoming) {
        return Err(fail(
            "watch response differs from the pinned exact Run/Attempt",
        ));
    }
    *current = Some(incoming.into());
    Ok(())
}
fn reported_state<'a>(value: &'a Value, layer: &str, key: &str) -> Result<Option<&'a str>, Error> {
    match value.get(layer).and_then(|value| value.get(key)) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        _ => Err(fail("watch received malformed status evidence")),
    }
}
impl Watch {
    /// Duration is finite by construction. All identities and elapsed time are
    /// explicit inputs; no state file, environment or clock is read here.
    pub fn new(duration: Duration, selection: Selection<'_>) -> Result<Self, Error> {
        for id in [selection.project, selection.run].into_iter().flatten() {
            crate::observation::segment(id)?;
        }
        if let Some(attempt) = selection.attempt {
            validate_attempt(attempt).map_err(|_| fail("choose an exact attempt-NNN identity"))?;
        }
        Ok(Self {
            duration,
            elapsed: Duration::ZERO,
            project: selection.project.map(str::to_owned),
            run: selection.run.map(str::to_owned),
            attempt: selection.attempt.map(str::to_owned),
        })
    }
    pub fn selection(&self) -> Selection<'_> {
        Selection {
            project: self.project.as_deref(),
            run: self.run.as_deref(),
            attempt: self.attempt.as_deref(),
        }
    }
    fn at(mut self, elapsed: Duration) -> Result<Self, Error> {
        if elapsed < self.elapsed {
            return Err(fail("watch elapsed time must not move backwards"));
        }
        self.elapsed = elapsed;
        Ok(self)
    }
    fn expired(&self) -> bool {
        self.elapsed >= self.duration
    }
    fn delay(&self) -> Duration {
        self.duration.saturating_sub(self.elapsed).min(INTERVAL)
    }
    /// Bind the first returned identity, then preserve it across every update.
    /// An unbound saved Preparation may wait; after its Attempt appears, neither
    /// a different Attempt nor a return to `null` can replace the pinned ID.
    pub fn accept(self, value: Value, elapsed: Duration) -> Result<Decision, Error> {
        let mut next = self.at(elapsed)?;
        for (key, slot) in [("project", &mut next.project), ("run_id", &mut next.run)] {
            let incoming = value
                .get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| fail("watch received malformed Run identity"))?;
            crate::observation::segment(incoming)?;
            selected(slot, incoming)?;
        }
        match value.get("attempt_id") {
            Some(Value::String(incoming)) => {
                validate_attempt(incoming)
                    .map_err(|_| fail("choose an exact attempt-NNN identity"))?;
                selected(&mut next.attempt, incoming)?;
            }
            Some(Value::Null) if next.attempt.is_none() => {}
            Some(Value::Null) => {
                return Err(fail(
                    "watch response differs from the pinned exact Run/Attempt",
                ));
            }
            _ => return Err(fail("watch received malformed Attempt identity")),
        }
        let terminal = matches!(
            reported_state(&value, "scheduler", "state")?,
            Some("SUCCEEDED" | "FAILED" | "CANCELLED" | "PREEMPTED" | "TIMEOUT")
        );
        let published = matches!(
            reported_state(&value, "publication", "status")?,
            Some("AVAILABLE" | "RELEASED" | "EXPIRED" | "FAILED")
        );
        if terminal && published {
            Ok(Decision::Complete {
                value,
                reason: End::Settled,
            })
        } else if next.expired() {
            Ok(Decision::Complete {
                value,
                reason: End::ObservationWindowEnded,
            })
        } else {
            Ok(Decision::Wait {
                delay: next.delay(),
                watch: next,
                value,
            })
        }
    }
}

/// A small injectable shell. `read` must remain read-only and return a complete
/// bound status. Its finite request timeout is separate from this observation
/// window; an already in-flight read is not interrupted or retried here.
/// `elapsed` must be monotonic and use the same origin throughout this call.
pub fn observe_with(
    duration: Duration,
    selection: Selection<'_>,
    mut read: impl FnMut(Selection<'_>) -> Result<Value, Error>,
    mut elapsed: impl FnMut() -> Duration,
    mut sleep: impl FnMut(Duration),
    mut on_update: impl FnMut(&Value) -> Result<(), Error>,
) -> Result<Value, Error> {
    let mut watch = Watch::new(duration, selection)?.at(elapsed())?;
    loop {
        // As in a one-shot status call, even a zero-length watch returns one
        // complete observation. It does not authorize any provider operation.
        let value = read(watch.selection())?;
        match watch.accept(value, elapsed())? {
            Decision::Complete { value, .. } => return Ok(value),
            Decision::Wait {
                watch: next, value, ..
            } => {
                on_update(&value)?;
                watch = next.at(elapsed())?;
                if watch.expired() {
                    return Ok(value);
                }
                sleep(watch.delay());
                watch = watch.at(elapsed())?;
                if watch.expired() {
                    // Do not start another HTTP observation after the budget.
                    return Ok(value);
                }
            }
        }
    }
}

/// Production clock/sleep boundary, sharing the existing authenticated status
/// reader. Completion/timeout only ends observation; it sends no mutation.
pub fn observe(
    client: &Client,
    health: &Value,
    saved: Option<&Value>,
    selection: Selection<'_>,
    details: bool,
    duration: Duration,
    on_update: impl FnMut(&Value) -> Result<(), Error>,
) -> Result<Value, Error> {
    let start = Instant::now();
    start
        .checked_add(duration)
        .ok_or_else(|| fail("status observation duration is out of range"))?;
    observe_with(
        duration,
        selection,
        |selection| status::observe(client, health, saved, selection, details),
        || start.elapsed(),
        std::thread::sleep,
        on_update,
    )
}
