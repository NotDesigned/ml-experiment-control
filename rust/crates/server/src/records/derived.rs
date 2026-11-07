//! Post-hoc derived batches of an ended Attempt. They continue the Attempt's
//! sequence, so the publisher proves each row as it proves any other, but they
//! live in their own tables: training records, results and validation never
//! read them, and an older server sees only the training journal.
use super::*;
use std::collections::BTreeSet;

const LATEST_LIFECYCLE: &str = "SELECT payload FROM events WHERE scope=? AND json_extract(payload,'$.kind')='lifecycle' ORDER BY seq DESC LIMIT 1";
const TRAINING_NAMES: &str = "SELECT DISTINCT json_extract(j.value,'$.name') FROM events AS e,json_each(e.payload,'$.observations') AS j WHERE e.scope=? AND json_extract(e.payload,'$.kind')='metrics'";

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl Store {
    /// Append one derived file to an ended Attempt, once. The same SHA256 again
    /// returns the original batch and changes nothing. A record never takes the
    /// name of a training-time metric, so it cannot change a training curve or
    /// its W&B summary.
    pub fn append_derived(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        sha256: &str,
        submitted_at: &str,
        observations: &[Value],
    ) -> Result<DerivedBatch> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_scope(guard, scope)?;
        if !digest(sha256)
            || observations.is_empty()
            || observations.len() > records::DERIVED_MAX_RECORDS
        {
            return Err(Error::Invalid("derived batch"));
        }
        let names: BTreeSet<&str> = observations
            .iter()
            .map(|item| item["name"].as_str().ok_or(Error::Invalid("derived name")))
            .collect::<Result<_>>()?;
        let rows = observations
            .iter()
            .enumerate()
            .map(|(index, observation)| {
                let payload = json!({"kind":records::DERIVED_KIND,"batch_sha256":sha256,
                    "submitted_at":submitted_at,"observations":[observation]});
                Ok((
                    format!("derived.{sha256}.{}", index + 1),
                    records::encoded(&payload)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let count = i64::try_from(rows.len()).map_err(why!(Error::Invalid("derived batch")))?;
        self.database.transaction(
            || self.require_guard(guard),
            |tx| {
                self.require_guard(guard)?;
                let existing = tx
                    .query_row(
                        "SELECT submitted_at,first,last,records FROM derived_batches WHERE scope=? AND sha256=?",
                        params![scope, sha256],
                        |row| {
                            Ok(DerivedBatch {
                                sha256: sha256.into(),
                                submitted_at: row.get(0)?,
                                first: row.get(1)?,
                                last: row.get(2)?,
                                records: row.get(3)?,
                            })
                        },
                    )
                    .optional()?;
                if let Some(batch) = existing {
                    return Ok(batch);
                }
                let lifecycle: Option<String> = tx
                    .query_row(LATEST_LIFECYCLE, [scope], |row| row.get(0))
                    .optional()?;
                let lifecycle = match lifecycle {
                    Some(payload) => serde_json::from_str::<Value>(&payload)?["data"].clone(),
                    None => Value::Null,
                };
                if !records::ended(&lifecycle) {
                    return Err(Error::NotEnded);
                }
                let training: BTreeSet<String> = {
                    let mut statement = tx.prepare(TRAINING_NAMES)?;
                    let found = statement
                        .query_map([scope], |row| row.get::<_, Option<String>>(0))?
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    found.into_iter().flatten().collect()
                };
                if names.iter().any(|name| training.contains(*name)) {
                    return Err(Error::DerivedNameConflict);
                }
                let total: i64 =
                    tx.query_row(&format!("SELECT {}", self.total_sql("?1")), [scope], |row| {
                        row.get(0)
                    })?;
                let batch = DerivedBatch {
                    sha256: sha256.into(),
                    submitted_at: submitted_at.into(),
                    first: total + 1,
                    last: total + count,
                    records: count,
                };
                for (seq, (event_id, body)) in (batch.first..).zip(&rows) {
                    let taken: bool = tx.query_row(
                        "SELECT EXISTS(SELECT 1 FROM event_ids WHERE scope=? AND event_id=?)",
                        params![scope, event_id],
                        |row| row.get(0),
                    )?;
                    if taken {
                        return Err(Error::EventConflict);
                    }
                    tx.execute(
                        "INSERT INTO derived_events VALUES (?,?,?,?,?)",
                        params![scope, seq, event_id, body, sha256],
                    )?;
                }
                tx.execute(
                    "INSERT INTO derived_batches VALUES (?,?,?,?,?,?)",
                    params![scope, sha256, submitted_at, batch.first, batch.last, count],
                )?;
                // A finished publication starts again for the new rows.
                tx.execute("UPDATE scopes SET status='PENDING' WHERE confirmed<? AND error IS NULL AND status NOT IN ('LOCAL_ONLY','PENDING') AND id=?",params![batch.last,scope])?;
                self.require_guard(guard)?;
                Ok(batch)
            },
        )
    }
    /// Every derived batch of `scope`, oldest first.
    pub fn derived_batches(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
    ) -> Result<Vec<DerivedBatch>> {
        self.require_scope(guard, scope)?;
        if self.read_only {
            return Ok(Vec::new());
        }
        let batches = self.database.read_only(|connection| -> Result<_> {
            let mut statement = connection.prepare(
                "SELECT sha256,submitted_at,first,last,records FROM derived_batches WHERE scope=? ORDER BY first",
            )?;
            let rows = statement
                .query_map([scope], |row| {
                    Ok(DerivedBatch {
                        sha256: row.get(0)?,
                        submitted_at: row.get(1)?,
                        first: row.get(2)?,
                        last: row.get(3)?,
                        records: row.get(4)?,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        self.require_scope(guard, scope)?;
        Ok(batches)
    }
    /// Whether a derived row follows `after`: the one reason a finished W&B Run
    /// is resumed.
    pub fn derived_after(&self, guard: &RecordGuard<'_>, scope: &str, after: i64) -> Result<bool> {
        self.require_scope(guard, scope)?;
        if self.read_only {
            return Ok(false);
        }
        let value = self.database.read_only(|connection| -> Result<_> {
            Ok(connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM derived_events WHERE scope=? AND seq>?)",
                params![scope, after],
                |row| row.get(0),
            )?)
        })?;
        self.require_scope(guard, scope)?;
        Ok(value)
    }
}
