//! The few MLflow REST calls the publisher makes, over one blocking client. The
//! password goes only into the Authorization header: no URL, body or logged
//! cause carries it, and response text never reaches a diagnostic.
use super::Error;
use serde_json::{Value, json};
use std::{io::Read, time::Duration};

/// MLflow answers these calls with small documents; a history page is the largest.
const RESPONSE_LIMIT: u64 = 16 * 1024 * 1024;

/// What the publisher keeps of an MLflow run.
#[derive(Clone, Debug, PartialEq)]
pub struct RunInfo {
    pub run_id: String,
    pub start_time: i64,
}

pub struct Client {
    http: reqwest::blocking::Client,
    /// `https://host:port[/prefix]/api/2.0/mlflow/`
    api: String,
    auth: Option<(String, String)>,
}

impl Client {
    pub fn new(
        base: &str,
        auth: Option<(String, String)>,
        ca_pem: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<Self, Error> {
        let mut builder = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            // The Tokyo-Shenzhen link drops SYNs; retransmits take 1 s and 3 s,
            // so a 5 s limit failed several connects an hour.
            .connect_timeout(Duration::from_secs(15))
            // uvicorn closes an idle keep-alive connection after 5 s; reusing one it
            // has just closed fails with "connection closed before message completed".
            .pool_idle_timeout(Duration::from_secs(3))
            .timeout(timeout);
        if let Some(pem) = ca_pem {
            builder = builder.add_root_certificate(
                reqwest::Certificate::from_pem(pem).map_err(why!(Error::Configuration))?,
            );
        }
        Ok(Self {
            http: builder.build().map_err(why!(Error::Configuration))?,
            api: format!("{base}/api/2.0/mlflow/"),
            auth,
        })
    }

    fn call(
        &self,
        method: reqwest::Method,
        endpoint: &str,
        query: &[(&str, &str)],
        body: Option<&Value>,
    ) -> Result<Value, Error> {
        let mut url = url::Url::parse(&format!("{}{endpoint}", self.api))
            .map_err(why!(Error::Configuration))?;
        if !query.is_empty() {
            url.query_pairs_mut().extend_pairs(query);
        }
        let mut request = self.http.request(method, url);
        if let Some((username, password)) = &self.auth {
            request = request.basic_auth(username, Some(password));
        }
        if let Some(body) = body {
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(serde_json::to_vec(body).map_err(why!(Error::Identity))?);
        }
        let response = request.send().map_err(why_chain!(Error::Unreachable))?;
        let status = response.status().as_u16();
        let mut bytes = Vec::new();
        response
            .take(RESPONSE_LIMIT + 1)
            .read_to_end(&mut bytes)
            .map_err(why!(Error::Unreachable))?;
        if bytes.len() as u64 > RESPONSE_LIMIT {
            return Err(Error::ResponseInvalid);
        }
        let value = if bytes.is_empty() {
            json!({})
        } else {
            serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null)
        };
        match status {
            200 if value.is_object() => Ok(value),
            200 => Err(Error::ResponseInvalid),
            401 | 403 => Err(Error::AuthFailed),
            _ if value["error_code"] == "RESOURCE_DOES_NOT_EXIST" => Err(Error::NotFound),
            _ if value["error_code"] == "RESOURCE_ALREADY_EXISTS" => Err(Error::AlreadyExists),
            status if status >= 500 => Err(Error::ServerError),
            _ => Err(Error::Rejected),
        }
    }
    fn get(&self, endpoint: &str, query: &[(&str, &str)]) -> Result<Value, Error> {
        self.call(reqwest::Method::GET, endpoint, query, None)
    }
    fn post(&self, endpoint: &str, body: &Value) -> Result<Value, Error> {
        self.call(reqwest::Method::POST, endpoint, &[], Some(body))
    }

    /// The experiment id and lifecycle stage, or None when no experiment has the name.
    pub fn experiment_by_name(&self, name: &str) -> Result<Option<(String, String)>, Error> {
        match self.get("experiments/get-by-name", &[("experiment_name", name)]) {
            Ok(value) => {
                let experiment = &value["experiment"];
                Ok(Some((
                    text(&experiment["experiment_id"])?,
                    experiment["lifecycle_stage"]
                        .as_str()
                        .unwrap_or("active")
                        .to_owned(),
                )))
            }
            Err(Error::NotFound) => Ok(None),
            Err(error) => Err(error),
        }
    }
    /// `Error::AlreadyExists` when another writer created it first.
    pub fn create_experiment(&self, name: &str) -> Result<String, Error> {
        text(&self.post("experiments/create", &json!({"name":name}))?["experiment_id"])
    }
    /// The active run of `experiment` whose tag `key` is exactly `value`.
    pub fn search_run(
        &self,
        experiment: &str,
        key: &str,
        value: &str,
    ) -> Result<Option<RunInfo>, Error> {
        let found = self.post(
            "runs/search",
            &json!({"experiment_ids":[experiment],"filter":format!("tags.`{key}` = '{value}'"),
                "run_view_type":"ACTIVE_ONLY","max_results":2}),
        )?;
        match found["runs"].as_array().map(Vec::as_slice) {
            None | Some([]) => Ok(None),
            Some([run]) => run_info(&run["info"]).map(Some),
            // Two runs for one Attempt: something other than this publisher made one.
            Some(_) => Err(Error::Identity),
        }
    }
    pub fn create_run(
        &self,
        experiment: &str,
        name: &str,
        start_time: i64,
        tags: &[(String, String)],
    ) -> Result<RunInfo, Error> {
        let tags: Vec<Value> = tags
            .iter()
            .map(|(key, value)| json!({"key":key,"value":value}))
            .collect();
        let created = self.post(
            "runs/create",
            &json!({"experiment_id":experiment,"run_name":name,"start_time":start_time,"tags":tags}),
        )?;
        run_info(&created["run"]["info"])
    }
    /// One committed write: MLflow stores the whole batch or none of it.
    pub fn log_batch(
        &self,
        run: &str,
        metrics: &[Value],
        params: &[(String, String)],
    ) -> Result<(), Error> {
        let params: Vec<Value> = params
            .iter()
            .map(|(key, value)| json!({"key":key,"value":value}))
            .collect();
        self.post(
            "runs/log-batch",
            &json!({"run_id":run,"metrics":metrics,"params":params}),
        )
        .map(drop)
    }
    pub fn update_run(&self, run: &str, status: &str, end_time: i64) -> Result<(), Error> {
        self.post(
            "runs/update",
            &json!({"run_id":run,"status":status,"end_time":end_time}),
        )
        .map(drop)
    }
    /// The whole run document, or None when MLflow has no such run.
    pub fn get_run(&self, run: &str) -> Result<Option<Value>, Error> {
        match self.get("runs/get", &[("run_id", run)]) {
            Ok(value) => Ok(Some(value["run"].clone())),
            Err(Error::NotFound) => Ok(None),
            Err(error) => Err(error),
        }
    }
    /// Every stored point of one metric, following MLflow's pages.
    #[cfg(test)]
    pub fn history(&self, run: &str, key: &str) -> Result<Vec<Value>, Error> {
        let mut points = Vec::new();
        let mut token = String::new();
        loop {
            let mut query = vec![("run_id", run), ("metric_key", key)];
            if !token.is_empty() {
                query.push(("page_token", token.as_str()));
            }
            let page = self.get("metrics/get-history", &query)?;
            if let Some(items) = page["metrics"].as_array() {
                points.extend(items.iter().cloned());
            }
            match page["next_page_token"].as_str() {
                Some(next) if !next.is_empty() => token = next.to_owned(),
                _ => return Ok(points),
            }
        }
    }
}

fn text(value: &Value) -> Result<String, Error> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or(Error::ResponseInvalid)
}
fn run_info(info: &Value) -> Result<RunInfo, Error> {
    Ok(RunInfo {
        run_id: text(&info["run_id"])?,
        // MLflow's JSON encodes int64 fields as strings or numbers.
        start_time: match &info["start_time"] {
            Value::Number(number) => number.as_i64(),
            Value::String(text) => text.parse().ok(),
            _ => None,
        }
        .ok_or(Error::ResponseInvalid)?,
    })
}
