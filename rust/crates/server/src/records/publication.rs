//! Private durable publication reservations and disposable display projections.
//! Only trusted publisher code can save these states, never worker HTTP input.
use super::*;
use ml_exp_core::tracking;

impl Store {
    /// Handoff reservations do not reset the scope cursor, status or failure
    /// counter. Existing native reservations always win across restarts.
    pub(crate) fn initialize_publication_state(
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
        if encoded.len() > 8 * 1024 * 1024 {
            return Err(Error::Invalid("handoff reservation bound"));
        }
        self.database.transaction(
            || self.require_guard(guard),
            |tx| {
                tx.execute(
                    "INSERT OR IGNORE INTO native_publications VALUES (?,?)",
                    params![scope, encoded],
                )?;
                self.require_guard(guard)
            },
        )
    }
    pub fn publication_runs(&self, maximum: usize) -> Result<Vec<(String, String)>> {
        let rows = self.database.read_only(|connection| -> Result<_> {
            let mut statement = connection.prepare("SELECT project,run FROM runs WHERE json_extract(route,'$.enabled')=1 ORDER BY project,run LIMIT ?")?;
            Ok(statement.query_map([maximum as i64 + 1],|row|Ok((row.get(0)?,row.get(1)?)))?.collect::<std::result::Result<Vec<_>,_>>()?)
        })?;
        if rows.len() > maximum {
            return Err(Error::Invalid("publisher Run limit"));
        }
        Ok(rows)
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
    pub fn display(
        &self,
        guard: &RecordGuard<'_>,
        scope: &str,
        through: i64,
        version: i64,
    ) -> Result<Value> {
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
                            .map_err(|_| Error::Invalid("display projection"))?;
                    }
                }
                after = row.seq;
            }
        }
        let projection = projection
            .at(through)
            .map_err(|_| Error::Invalid("display cutoff"))?;
        if version == tracking::STABLE_DISPLAY_VERSION {
            return projection.stable_document().map_err(|error| match error {
                tracking::Error::Display(code) => Error::Invalid(code),
                _ => Error::Invalid("display document"),
            });
        }
        if version != 2 {
            return Err(Error::Invalid("unsupported display version"));
        }
        // Only already-started v2 publications retain destination reservations.
        let binding = self.binding(guard)?.ok_or(Error::Unbound)?;
        let destination = records::encoded(&json!([
            binding["route"]["entity"],
            binding["route"]["project"]
        ]))?;
        let mut names = Map::new();
        let state = projection.state_value();
        let series = state["series"]
            .as_object()
            .ok_or(Error::Invalid("display series"))?;
        self.database.transaction(||self.require_guard(guard),|tx| {
            for series in series.values() {
                let context: Map<_,_> = ["name","unit","protocol_id","dataset_id"].iter().map(|key|((*key).into(),series["context"][*key].clone())).collect();
                let context = Value::Object(context);
                let signature = tracking::identity(&context,false).map_err(|_|Error::Invalid("display identity"))?;
                let existing: Option<String> = tx.query_row("SELECT public_name FROM display_metrics WHERE destination=? AND signature=?",params![destination,signature],|row|row.get(0)).optional()?;
                let name = if let Some(existing) = existing { existing } else {
                    let stem = tracking::stem(context["name"].as_str().ok_or(Error::Invalid("metric name"))?);
                    let used: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM display_metrics WHERE destination=? AND public_name=?)",params![destination,stem],|row|row.get(0))?;
                    let name = if used {format!("{stem}/{}",tracking::fingerprint(&signature))} else {stem};
                    tx.execute("INSERT INTO display_metrics VALUES (?,?,?,?)",params![destination,signature,name,records::encoded(&context)?])?;
                    name
                };
                names.insert(signature,json!(name));
            }
            self.require_guard(guard)
        })?;
        projection
            .at(through)
            .and_then(|p| p.document(&names))
            .map_err(|_| Error::Invalid("display document"))
    }
}
