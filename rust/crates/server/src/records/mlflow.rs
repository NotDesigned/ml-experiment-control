//! Private MLflow publication state: one document per scope beside the W&B
//! state. Its cursor and diagnostics never touch the W&B columns of `scopes`,
//! so either destination can fail or lag without moving the other.
use super::*;
use std::collections::BTreeMap;

/// A read-only Store may predate the table: it then has no MLflow state.
const TABLE_EXISTS: &str =
    "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='mlflow_publications')";
/// One state document stays far below this; it bounds a corrupted one.
const STATE_BYTES: usize = 256 * 1024;

impl Store {
    /// Runs whose frozen route publishes to MLflow.
    pub fn mlflow_runs(&self) -> Result<Vec<(String, String)>> {
        self.database.read_only(|connection| -> Result<_> {
            let mut statement = connection.prepare("SELECT project,run FROM runs WHERE json_extract(route,'$.mlflow.enabled')=1 ORDER BY project,run")?;
            Ok(statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<std::result::Result<Vec<_>, _>>()?)
        })
    }
    pub fn mlflow_state(&self, guard: &RecordGuard<'_>, scope: &str) -> Result<Option<Value>> {
        self.require_scope(guard, scope)?;
        let value: Option<String> = self.database.read_only(|connection| -> Result<_> {
            if !connection.query_row(TABLE_EXISTS, [], |row| row.get::<_, bool>(0))? {
                return Ok(None);
            }
            Ok(connection
                .query_row(
                    "SELECT state FROM mlflow_publications WHERE scope=?",
                    [scope],
                    |row| row.get(0),
                )
                .optional()?)
        })?;
        self.require_scope(guard, scope)?;
        value
            .map(|v| serde_json::from_str(&v).map_err(Error::from))
            .transpose()
    }
    pub fn save_mlflow_state(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        value: &Value,
    ) -> Result<()> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_scope(guard, scope)?;
        let encoded = records::encoded(value)?;
        if encoded.len() > STATE_BYTES {
            return Err(Error::Invalid("MLflow publisher state bound"));
        }
        self.database.transaction(
            || self.require_guard(guard),
            |tx| {
                self.require_guard(guard)?;
                tx.execute("INSERT INTO mlflow_publications VALUES (?,?) ON CONFLICT(scope) DO UPDATE SET state=excluded.state",params![scope,encoded])?;
                self.require_guard(guard)
            },
        )
    }
    /// Each scope of a Run whose route publishes to MLflow shows its MLflow
    /// publication beside the W&B columns, with its own `confirmed` and `pending`.
    pub(super) fn attach_mlflow(
        &self,
        guard: &RecordGuard<'_>,
        scopes: &mut [Value],
    ) -> Result<()> {
        let Some(binding) = self.binding_raw(&guard.project, &guard.run)? else {
            return Ok(());
        };
        let route = &binding["route"]["mlflow"];
        if route["enabled"] != true {
            return Ok(());
        }
        let states: BTreeMap<String, String> =
            self.database.read_only(|connection| -> Result<_> {
                if !connection.query_row(TABLE_EXISTS, [], |row| row.get::<_, bool>(0))? {
                    return Ok(BTreeMap::new());
                }
                let mut statement = connection.prepare("SELECT m.scope,m.state FROM mlflow_publications m JOIN scopes s ON s.id=m.scope WHERE s.project=? AND s.run=?")?;
                Ok(statement
                    .query_map(params![guard.project, guard.run], |row| {
                        Ok((row.get(0)?, row.get(1)?))
                    })?
                    .collect::<std::result::Result<_, _>>()?)
            })?;
        for scope in scopes {
            let id = scope["id"]
                .as_str()
                .ok_or(Error::Invalid("scope id"))?
                .to_owned();
            let total = scope["total"]
                .as_i64()
                .ok_or(Error::Invalid("scope total"))?;
            let mut view = match states.get(&id) {
                Some(state) => serde_json::from_str::<Value>(state)?,
                None => json!({"status":"MLFLOW_PENDING"}),
            };
            let confirmed = view["confirmed"].as_i64().unwrap_or(0);
            view["confirmed"] = json!(confirmed);
            view["total"] = json!(total);
            view["pending"] = json!(total - confirmed);
            view["experiment"] = route["experiment"].clone();
            scope["mlflow"] = view;
        }
        Ok(())
    }
}
