//! Private durable publication state and disposable display projections.
//! Only trusted publisher code can save these states, never worker HTTP input.
use super::*;
use ml_exp_core::tracking;

impl Store {
    pub fn publication_runs(&self) -> Result<Vec<(String, String)>> {
        self.database.read_only(|connection| -> Result<_> {
            let mut statement = connection.prepare("SELECT project,run FROM runs WHERE json_extract(route,'$.enabled')=1 ORDER BY project,run")?;
            Ok(statement.query_map([],|row|Ok((row.get(0)?,row.get(1)?)))?.collect::<std::result::Result<Vec<_>,_>>()?)
        })
    }
    pub fn publication_state(&self, guard: &RecordGuard<'_>, scope: &str) -> Result<Option<Value>> {
        self.require_scope(guard, scope)?;
        let value: Option<String> = self.database.read_only(|connection| -> Result<_> {
            Ok(connection
                .query_row(
                    "SELECT state FROM native_publications WHERE scope=?",
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
    pub fn save_publication_state(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        value: &Value,
        status: &str,
        error: Option<&str>,
        at: &str,
    ) -> Result<()> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        self.require_scope(guard, scope)?;
        let encoded = records::encoded(value)?;
        if encoded.len() > 1024 * 1024 {
            return Err(Error::Invalid("publisher state bound"));
        }
        self.database.transaction(||self.require_guard(guard),|tx| {
            self.require_guard(guard)?;
            tx.execute("INSERT INTO native_publications VALUES (?,?) ON CONFLICT(scope) DO UPDATE SET state=excluded.state",params![scope,encoded])?;
            tx.execute("UPDATE scopes SET status=?,error=?,last_attempt_at=?,failures=CASE WHEN ? IS NULL THEN 0 ELSE failures+1 END WHERE id=?",params![status,error,at,error,scope])?;
            self.require_guard(guard)
        })
    }
    pub fn display(&self, guard: &RecordGuard<'_>, scope: &str, through: i64) -> Result<Value> {
        self.require_scope(guard, scope)?;
        if self.read_only || through < 0 || through > self.total(guard, scope)? || through > 100_000
        {
            return Err(Error::Invalid("publisher display bound"));
        }
        let mut projection = tracking::Projection::new();
        let mut after = 0;
        while after < through {
            let rows = self.events(guard, scope, after, (through - after).min(256))?;
            if rows.is_empty() {
                return Err(Error::Invalid("publisher journal prefix"));
            }
            for row in rows {
                if row.payload["kind"] == "metrics" {
                    for item in row.payload["observations"]
                        .as_array()
                        .ok_or(Error::Invalid("metric array"))?
                    {
                        projection = projection
                            .add(item, row.seq)
                            .map_err(why!(Error::Invalid("display projection")))?;
                    }
                }
                after = row.seq;
            }
        }
        projection
            .at(through)
            .map_err(why!(Error::Invalid("display cutoff")))?
            .client_document()
            .map_err(why!(Error::Invalid("display document")))
    }
}
