//! What a cloud worker says when it could not start, for providers whose logs are empty.
//! It is evidence only: it changes no Run state, starts no retry and needs no Attempt
//! storage. Only the latest report is kept, and only while the Attempt has sent no
//! worker packet; once one exists, a report is ignored and none is shown.
use super::*;

/// The largest request body a startup failure report may have.
pub const MAX_BODY_BYTES: usize = 4096;
/// The longest message kept, in characters, after redaction.
const MAX_MESSAGE_CHARS: usize = 1000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub message: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    message: String,
    received_at: String,
}

/// One redacted line, clipped. An empty result is no report.
fn clean(message: &str) -> String {
    let line: String = message
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    experiment_redact::redact_line(line.trim())
        .chars()
        .take(MAX_MESSAGE_CHARS)
        .collect()
}

impl Service {
    fn startup_failure_name(prepared: &Prepared) -> cloud::Result<String> {
        Ok(format!(
            "{}.startup-failure.json",
            Self::attempt_slot(
                &prepared.execution.job.project,
                &prepared.execution.job.run_id,
                &prepared.execution.job.attempt_id
            )?
        ))
    }
    /// Keep the latest report of an Attempt that has not reported through the exchange.
    /// Returns whether it was kept.
    pub fn receive_startup_failure(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
        report: &Report,
        now: &str,
    ) -> cloud::Result<Value> {
        let invalid = || cloud::Error {
            code: "INVALID_STARTUP_FAILURE".into(),
            http_status: Some(422),
            uncertain: false,
        };
        // The same lock as a worker packet, so a report never lands after one.
        let _lock = self.lock("startup_failure.receive_startup_failure")?;
        // Authentication precedes every read of what the report says.
        self.worker_authorized(project, run, attempt, token)?;
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(cloud::Error::evidence)?;
        // The token itself is not a pattern redaction knows.
        let message = clean(&report.message.replace(token, "<redacted>"));
        if message.is_empty() {
            return Err(invalid());
        }
        if self.exchange_saved(&prepared)?.is_some() {
            return Ok(json!({"recorded":false}));
        }
        let saved = Saved {
            message,
            received_at: now.into(),
        };
        self.slots
            .replace_regular(
                OsStr::new(&Self::startup_failure_name(&prepared)?),
                &encode(&saved)?,
            )
            .map_err(io)?;
        Ok(json!({"recorded":true}))
    }
    /// The kept report, while the Attempt still has no worker packet.
    pub(super) fn startup_failure(&self, prepared: &Prepared) -> cloud::Result<Option<Value>> {
        let name = Self::startup_failure_name(prepared)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)?
            || self.exchange_saved(prepared)?.is_some()
        {
            return Ok(None);
        }
        let saved: Saved = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(why!(cloud::Error::evidence()))?;
        Ok(Some(
            json!({"message":saved.message,"received_at":saved.received_at}),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_is_one_redacted_bounded_line() {
        let cleaned =
            clean("  storage failed\nat x: Authorization: Bearer abcdefabcdef0123456789 \t");
        assert!(!cleaned.contains('\n'));
        assert!(!cleaned.contains("abcdefabcdef0123456789"), "{cleaned}");
        assert!(cleaned.starts_with("storage failed at x"));
        assert_eq!(clean(&"x".repeat(5000)).chars().count(), MAX_MESSAGE_CHARS);
        assert!(clean(" \n\t").is_empty());
    }
}
