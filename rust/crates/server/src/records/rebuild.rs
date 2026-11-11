//! W&B rebuilds of a finished Attempt. W&B cannot delete one key of a Run, so a
//! rebuild binds the scope to a new W&B id, which the publisher fills from the
//! training journal and derived batches without the names of one exclusion list.
//! Nothing in the journal changes and no W&B Run is deleted: the old id, and the
//! publication state it reached, stay here beside the list and its SHA256.
use super::*;
use std::collections::BTreeSet;

/// The W&B id a rebuilt scope publishes to: the latest rebuild's.
pub(super) const CURRENT_ID: &str =
    "SELECT new_wandb_id FROM wandb_rebuilds WHERE scope=? ORDER BY ordinal DESC LIMIT 1";
const COLUMNS: &str =
    "ordinal,sha256,audit,old_wandb_id,new_wandb_id,requested_at,finished_at FROM wandb_rebuilds";
const LATEST_LIFECYCLE: &str = "SELECT payload FROM events WHERE scope=? AND json_extract(payload,'$.kind')='lifecycle' ORDER BY seq DESC LIMIT 1";

/// One accepted exclusion list for one scope, oldest first by `ordinal`.
#[derive(Clone, Debug, PartialEq)]
pub struct WandbRebuild {
    pub ordinal: i64,
    pub sha256: String,
    /// Names in the list, the excluded ones with their observation counts, the
    /// names no record had, and the rows left with only ml_expd metadata.
    pub audit: Value,
    pub previous_wandb_id: String,
    pub wandb_id: String,
    pub requested_at: String,
    pub finished_at: Option<String>,
}
impl WandbRebuild {
    pub fn document(&self) -> Value {
        let mut document = json!({"ordinal":self.ordinal,"sha256":self.sha256,
            "previous_wandb_id":self.previous_wandb_id,"wandb_id":self.wandb_id,
            "requested_at":self.requested_at,"finished_at":self.finished_at,
            "status":if self.finished_at.is_some() {"FINISHED"} else {"PUBLISHING"},
            "previous_wandb_run_deleted":false});
        if let (Some(document), Some(audit)) = (document.as_object_mut(), self.audit.as_object()) {
            document.extend(audit.clone());
        }
        document
    }
    fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(Self, String)> {
        Ok((
            Self {
                ordinal: row.get(0)?,
                sha256: row.get(1)?,
                audit: Value::Null,
                previous_wandb_id: row.get(3)?,
                wandb_id: row.get(4)?,
                requested_at: row.get(5)?,
                finished_at: row.get(6)?,
            },
            row.get(2)?,
        ))
    }
    fn parsed((mut rebuild, audit): (Self, String)) -> Result<Self> {
        rebuild.audit = serde_json::from_str(&audit)?;
        Ok(rebuild)
    }
}

fn hex64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl Store {
    /// Bind a finished scope to a new W&B id published without `names`, the
    /// verbatim exclusion list whose SHA256 the HTTP boundary checked. The list
    /// of the latest rebuild again returns that rebuild and changes nothing;
    /// `false` then says no new rebuild began. Another list starts another
    /// rebuild from the current id, once that one's publication has finished.
    /// `provider_ended` is the provider's proof that a Run without its own end in
    /// the journal can no longer write.
    pub fn rebuild_wandb(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        sha256: &str,
        names: &str,
        requested_at: &str,
        provider_ended: bool,
    ) -> Result<(WandbRebuild, bool)> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_scope(guard, scope)?;
        if !hex64(sha256) {
            return Err(Error::Invalid("rebuild digest"));
        }
        let excluded = records::rebuild_names(names.as_bytes())
            .map_err(|_| Error::Invalid("rebuild names"))?;
        let listed = serde_json::to_string(&excluded)?;
        let workspace = self.workspace()?;
        self.database.transaction(
            || self.require_guard(guard),
            |tx| {
                self.require_guard(guard)?;
                let latest = tx
                    .query_row(
                        &format!("SELECT {COLUMNS} WHERE scope=? ORDER BY ordinal DESC LIMIT 1"),
                        [scope],
                        WandbRebuild::from_row,
                    )
                    .optional()?
                    .map(WandbRebuild::parsed)
                    .transpose()?;
                if let Some(latest) = latest.as_ref().filter(|latest| latest.sha256 == sha256) {
                    return Ok((latest.clone(), false));
                }
                let route: String = tx.query_row(
                    "SELECT route FROM runs WHERE project=? AND run=?",
                    params![guard.project, guard.run],
                    |row| row.get(0),
                )?;
                let state: Option<String> = tx
                    .query_row(
                        "SELECT state FROM native_publications WHERE scope=?",
                        [scope],
                        |row| row.get(0),
                    )
                    .optional()?;
                let state: Value = match &state {
                    Some(state) => serde_json::from_str(state)?,
                    None => Value::Null,
                };
                let (previous, confirmed): (String, i64) = tx.query_row(
                    "SELECT wandb_id,confirmed FROM scopes WHERE id=?",
                    [scope],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )?;
                let total: i64 =
                    tx.query_row(&format!("SELECT {}", self.total_sql("?1")), [scope], |row| {
                        row.get(0)
                    })?;
                // Every record, derived ones too, is on a W&B Run that was finished.
                if serde_json::from_str::<Value>(&route)?["enabled"] != true
                    || state["remote_finished"] != true
                    || state["remote_missing"] == true
                    || confirmed != total
                {
                    return Err(Error::RebuildNotFinished);
                }
                let lifecycle: Option<String> = tx
                    .query_row(LATEST_LIFECYCLE, [scope], |row| row.get(0))
                    .optional()?;
                let lifecycle = match lifecycle {
                    Some(payload) => serde_json::from_str::<Value>(&payload)?["data"].clone(),
                    None => Value::Null,
                };
                if !records::ended(&lifecycle) && !provider_ended {
                    return Err(Error::NotEnded);
                }
                let wandb_id = records::rebuild_wandb_id(&workspace, scope, &previous, sha256)?;
                let kinds = format!(
                    "json_extract(e.payload,'$.kind') IN ('metrics','{}')",
                    records::DERIVED_KIND
                );
                let counted: Vec<(String, i64)> = {
                    let mut statement = tx.prepare(&format!(
                        "SELECT json_extract(j.value,'$.name'),COUNT(*) FROM {PUBLISHED} AS e,json_each(e.payload,'$.observations') AS j WHERE e.scope=?1 AND {kinds} AND json_extract(j.value,'$.name') IN (SELECT value FROM json_each(?2)) GROUP BY 1 ORDER BY 1"
                    ))?;
                    statement
                        .query_map(params![scope, listed], |row| {
                            Ok((row.get(0)?, row.get(1)?))
                        })?
                        .collect::<std::result::Result<_, _>>()?
                };
                let emptied: i64 = tx.query_row(
                    &format!(
                        "SELECT COUNT(*) FROM {PUBLISHED} AS e WHERE e.scope=?1 AND {kinds} AND json_array_length(e.payload,'$.observations')>0 AND NOT EXISTS (SELECT 1 FROM json_each(e.payload,'$.observations') AS j WHERE json_extract(j.value,'$.name') IS NULL OR json_extract(j.value,'$.name') NOT IN (SELECT value FROM json_each(?2)))"
                    ),
                    params![scope, listed],
                    |row| row.get(0),
                )?;
                let found: BTreeSet<&str> = counted.iter().map(|(name, _)| name.as_str()).collect();
                let counts: Map<String, Value> = counted
                    .iter()
                    .map(|(name, count)| (name.clone(), json!(count)))
                    .collect();
                let audit = json!({"names":excluded.len(),"excluded":counts,
                    "excluded_observations":counted.iter().map(|(_, count)| count).sum::<i64>(),
                    "unmatched_names":excluded.iter().filter(|name| !found.contains(name.as_str())).collect::<Vec<_>>(),
                    "emptied_rows":emptied});
                let rebuild = WandbRebuild {
                    ordinal: latest.map_or(1, |latest| latest.ordinal + 1),
                    sha256: sha256.into(),
                    audit,
                    previous_wandb_id: previous,
                    wandb_id,
                    requested_at: requested_at.into(),
                    finished_at: None,
                };
                tx.execute(
                    "INSERT INTO wandb_rebuilds VALUES (?,?,?,?,?,?,?,?,NULL,?)",
                    params![
                        scope,
                        rebuild.ordinal,
                        sha256,
                        names,
                        records::encoded(&rebuild.audit)?,
                        rebuild.previous_wandb_id,
                        rebuild.wandb_id,
                        requested_at,
                        state.is_object().then(|| records::encoded(&state)).transpose()?,
                    ],
                )?;
                // A new W&B Run, published from the first record as any new scope
                // is. A server that predates rebuilds cannot load this state, so
                // it never publishes the excluded names to the new id.
                let fresh = json!({"state":serde_json::to_value(ml_exp_core::tracking::publication::PublicationState::default())?,
                    "remote_finished":false,"ever_opened":false,"wandb_rebuild":sha256});
                tx.execute(
                    "INSERT INTO native_publications VALUES (?,?) ON CONFLICT(scope) DO UPDATE SET state=excluded.state",
                    params![scope, records::encoded(&fresh)?],
                )?;
                tx.execute(
                    "UPDATE scopes SET wandb_id=?,confirmed=0,status='PENDING',error=NULL,failures=0,last_confirmed_at=NULL,display_version=0,display_confirmed=0 WHERE id=?",
                    params![rebuild.wandb_id, scope],
                )?;
                self.require_guard(guard)?;
                Ok((rebuild, true))
            },
        )
    }
    /// Every W&B rebuild of `scope`, oldest first.
    pub fn wandb_rebuilds(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
    ) -> Result<Vec<WandbRebuild>> {
        self.require_scope(guard, scope)?;
        if self.read_only {
            return Ok(Vec::new());
        }
        let rows = self.database.read_only(|connection| -> Result<_> {
            let mut statement =
                connection.prepare(&format!("SELECT {COLUMNS} WHERE scope=? ORDER BY ordinal"))?;
            let rows = statement
                .query_map([scope], WandbRebuild::from_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        let rebuilds = rows
            .into_iter()
            .map(WandbRebuild::parsed)
            .collect::<Result<Vec<_>>>()?;
        self.require_scope(guard, scope)?;
        Ok(rebuilds)
    }
    /// Remove what the latest rebuild of `scope` excludes from rows read for
    /// publication. A read-only Store may predate rebuilds and changes nothing.
    pub(super) fn as_published(&self, scope: &str, rows: &mut [EventRow]) -> Result<()> {
        if self.read_only || rows.is_empty() {
            return Ok(());
        }
        let names: Option<String> = self.database.read_only(|connection| -> Result<_> {
            Ok(connection
                .query_row(
                    "SELECT names FROM wandb_rebuilds WHERE scope=? ORDER BY ordinal DESC LIMIT 1",
                    [scope],
                    |row| row.get(0),
                )
                .optional()?)
        })?;
        let Some(names) = names else {
            return Ok(());
        };
        let excluded = records::rebuild_names(names.as_bytes())
            .map_err(|_| Error::Invalid("stored rebuild names"))?;
        for row in rows {
            records::exclude_observations(&mut row.payload, &excluded);
        }
        Ok(())
    }
}
