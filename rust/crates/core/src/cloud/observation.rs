use super::*;
use crate::state::RunState;
use experiment_redact::redact_line;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueDiagnostic {
    pub source: String,
    pub code: Option<String>,
    pub code_type: Option<String>,
    pub reason: Option<String>,
    pub position: Option<u64>,
    pub priority: Option<u64>,
    pub enqueued_at: Option<String>,
    pub blocker: Option<Value>,
}
fn safe(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(|s| {
        redact_line(&s.chars().take(8192).collect::<String>())
            .chars()
            .take(2048)
            .collect()
    })
}
impl QueueDiagnostic {
    pub fn pai(row: &Value) -> Result<Self> {
        if !row.is_object() {
            return Err(Error("queue row"));
        }
        let mut result = Self {
            source: "pai-quota-workload".into(),
            code: safe(row.get("Code")),
            code_type: safe(row.get("CodeType")),
            reason: safe(row.get("Reason")),
            position: row["Position"].as_u64(),
            priority: row["Priority"].as_u64(),
            enqueued_at: safe(row.get("GmtEnqueuedTime")),
            blocker: None,
        };
        if let Some(blocker) = row["Reason"]
            .as_str()
            .filter(|s| s.len() <= 16384)
            .and_then(|s| crate::json::mapping(s.as_bytes()).ok())
        {
            let fields: serde_json::Map<String, Value> = ["CodeType", "Code", "Reason"]
                .into_iter()
                .filter_map(|k| safe(blocker.get(k)).map(|v| (k.into(), json!(v))))
                .collect();
            result.code = safe(fields.get("Code")).or(result.code);
            result.code_type = safe(fields.get("CodeType")).or(result.code_type);
            result.reason = safe(fields.get("Reason"));
            result.blocker = Some(Value::Object(fields));
        }
        Ok(result)
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobObservation {
    pub job_id: String,
    pub provider_state: String,
    pub state: RunState,
    pub queue: Option<QueueDiagnostic>,
}
/// Complete provider pagination is required before absence can be asserted.
#[derive(Clone, PartialEq)]
pub struct Page {
    pub rows: Vec<Value>,
    pub total: Option<u64>,
    pub next: Option<String>,
}
impl Page {
    pub fn pai(value: &Value, field: &str, page: u32, page_size: u32) -> Result<Self> {
        let rows = value[field].as_array().ok_or(Error("list rows"))?;
        let total = value["TotalCount"].as_u64().ok_or(Error("list total"))?;
        if page == 0
            || page_size == 0
            || page_size > 100
            || rows.len() > page_size as usize
            || total > 10000
        {
            return Err(Error("list bounds"));
        }
        let offset = u64::from(page - 1) * u64::from(page_size);
        if offset > total
            || offset + rows.len() as u64 > total
            || offset + (rows.len() as u64) < total && rows.len() != page_size as usize
        {
            return Err(Error("incomplete page"));
        }
        let next = (offset + (rows.len() as u64) < total).then(|| (page + 1).to_string());
        Ok(Self {
            rows: rows.clone(),
            total: Some(total),
            next,
        })
    }
    pub fn sensecore(value: &Value, field: &str, token: &str) -> Result<Self> {
        let rows = value[field].as_array().ok_or(Error("list rows"))?;
        let total = match value.get("total_size") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_u64().ok_or(Error("list total"))?),
        };
        let next = match value.get("next_page_token") {
            None => None,
            Some(Value::String(s)) if s.is_empty() || s == "0" => None,
            Some(Value::String(s)) => {
                component(s)?;
                Some(s.clone())
            }
            _ => return Err(Error("page token")),
        };
        if rows.len() > 100
            || total.is_some_and(|n| n < rows.len() as u64 || n > 10000)
            || next.as_deref() == Some(token)
            || rows.is_empty() && next.is_some()
        {
            return Err(Error("incomplete page"));
        }
        Ok(Self {
            rows: rows.clone(),
            total,
            next,
        })
    }
}
