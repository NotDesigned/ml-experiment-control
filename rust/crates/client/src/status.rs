//! Read-only status as a consumed continuation. Pure steps return exact GETs;
//! the final `observe` shell alone executes them. No repair or mutation plans.
use crate::{
    Client, Error,
    observation::{self, Reference, Selection},
};
use ml_exp_core::identity::validate_attempt;
use serde_json::{Map, Value, json};

fn fail(message: &'static str) -> Error {
    Error::Observation(message)
}
fn truth(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Number(n) => n.as_f64().is_none_or(|v| v != 0.0),
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        Value::Bool(true) => true,
    }
}
fn mapping(value: &Value) -> Value {
    if value.is_object() {
        value.clone()
    } else {
        json!({})
    }
}
fn otherwise(value: &Value, default: Value) -> Value {
    if truth(value) { value.clone() } else { default }
}
fn get_or(value: &Value, key: &str, default: Value) -> Value {
    value.get(key).cloned().unwrap_or(default)
}
fn checked(value: &Value, project: &str, run: &str, attempt: Option<&str>) -> Result<(), Error> {
    let mut expected = vec![("project", json!(project)), ("run_id", json!(run))];
    if let Some(attempt) = attempt {
        expected.push(("attempt_id", json!(attempt)));
    }
    observation::bound_response(
        value,
        &expected,
        "status response differs from the requested exact Run/Attempt",
    )?;
    Ok(())
}
fn capability(health: &Value, name: &str) -> bool {
    health
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|values| values.iter().any(|v| v.as_str() == Some(name)))
}

pub enum Step {
    Read(Read),
    Complete(Value),
}
pub struct Read {
    path: String,
    state: Pending,
}
enum Pending {
    Identity {
        read: Box<observation::Read>,
        health: Value,
        details: bool,
        saved: bool,
    },
    Flow {
        flow: Box<Flow>,
        phase: Phase,
    },
}
enum Phase {
    Preparation,
    InitialProgress(String),
    Run,
    Attempts,
    Discover,
    DiscoveredProgress(String),
    Collection,
    Expanded,
}
struct Flow {
    reference: Reference,
    evidence: Map<String, Value>,
    preparation: Value,
    run: Value,
    listing: Value,
    selected: Value,
    attempt: Option<String>,
    materialized: bool,
    progress: Value,
    collection: Value,
    collection_error: Option<Value>,
    progress_enabled: bool,
    collection_enabled: bool,
    details: bool,
    saved: bool,
}

pub fn begin(
    base: &str,
    health: &Value,
    saved: Option<&Value>,
    selection: Selection<'_>,
    details: bool,
) -> Result<Step, Error> {
    let step = observation::begin(base, saved, selection)?;
    identity_step(step, health.clone(), details, saved.is_some())
}
fn identity_step(
    step: observation::Step,
    health: Value,
    details: bool,
    saved: bool,
) -> Result<Step, Error> {
    match step {
        observation::Step::Read(read) => Ok(Step::Read(Read {
            path: read.path().into(),
            state: Pending::Identity {
                read: Box::new(read),
                health,
                details,
                saved,
            },
        })),
        observation::Step::Complete(reference) => {
            let attempt = reference.attempt_id.clone();
            Flow {
                reference,
                attempt,
                evidence: Map::new(),
                preparation: json!({"status":"NOT_OBSERVED","phase":null}),
                run: json!({}),
                listing: json!({}),
                selected: json!({}),
                materialized: false,
                progress: json!({"availability":"NOT_OBSERVED","diagnostic":null}),
                collection: json!({}),
                collection_error: None,
                progress_enabled: capability(&health, "execution-progress.v1"),
                collection_enabled: capability(&health, "result-collection.v1"),
                details,
                saved,
            }
            .start()
        }
    }
}
impl Read {
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn accept(self, response: Result<Value, Error>) -> Result<Step, Error> {
        match self.state {
            Pending::Identity {
                read,
                health,
                details,
                saved,
            } => identity_step((*read).accept(response?)?, health, details, saved),
            Pending::Flow { flow, phase } => flow.accept(phase, response),
        }
    }
}
impl Flow {
    fn request(self, phase: Phase, path: String) -> Result<Step, Error> {
        Ok(Step::Read(Read {
            path,
            state: Pending::Flow {
                flow: Box::new(self),
                phase,
            },
        }))
    }
    fn endpoint(&self) -> Result<String, Error> {
        Ok(format!(
            "/api/runs/{}/{}",
            observation::segment(&self.reference.project)?,
            observation::segment(&self.reference.run_id)?
        ))
    }
    fn start(mut self) -> Result<Step, Error> {
        if self.saved && truth(&self.reference.saved["preparation"]) {
            if let Some(value) = self.reference.preparation.clone() {
                self.preparation(value)?;
            } else {
                let id = self.reference.saved["preparation"]["preparation_id"]
                    .as_str()
                    .ok_or_else(|| fail("invalid saved Preparation identity"))?;
                let path = format!(
                    "/api/projects/{}/experiment-preparations/{}",
                    observation::segment(&self.reference.project)?,
                    observation::segment(id)?
                );
                return self.request(Phase::Preparation, path);
            }
        }
        self.reference_evidence()
    }
    fn preparation(&mut self, value: Value) -> Result<(), Error> {
        checked(
            &value,
            &self.reference.project,
            &self.reference.run_id,
            None,
        )?;
        if value["preparation_id"] != self.reference.saved["preparation"]["preparation_id"] {
            return Err(fail(
                "Preparation identity differs from the saved experiment",
            ));
        }
        self.preparation = json!({"status":value["status"],"phase":value["phase"],"error_code":value["error_code"]});
        self.evidence.insert("preparation".into(), value);
        Ok(())
    }
    fn reference_evidence(mut self) -> Result<Step, Error> {
        let mut progress = None;
        if self.saved {
            if let Some(value) = self.reference.submission.clone() {
                self.evidence.insert("submission".into(), value.clone());
                if self.preparation["status"] == "NOT_OBSERVED" {
                    self.preparation = json!({"status":value["status"],"phase":null});
                }
                if self.progress_enabled {
                    progress = value["submission_id"].as_str().map(str::to_owned);
                }
            }
            if let Some(action) = self.reference.action.clone() {
                self.evidence.insert("action".into(), action.clone());
                if action["operation"] == "RETRY_ATTEMPT" {
                    self.preparation = json!({"status":action["execution"]["status"],"phase":null});
                    if self.progress_enabled {
                        progress = action["action_id"].as_str().map(str::to_owned);
                    }
                }
            }
        }
        if let Some(id) = progress {
            let path = format!("/api/submissions/{}/progress", observation::segment(&id)?);
            return self.request(Phase::InitialProgress(id), path);
        }
        self.read_run()
    }
    fn read_run(self) -> Result<Step, Error> {
        let path = self.endpoint()?;
        self.request(Phase::Run, path)
    }
    fn before_submission(&self) -> bool {
        let action = mapping(self.evidence.get("action").unwrap_or(&Value::Null));
        let mut submission = mapping(self.evidence.get("submission").unwrap_or(&Value::Null));
        let execution = mapping(&action["execution"]);
        if !truth(&submission) && action["operation"] == "RETRY_ATTEMPT" {
            submission = json!({"status":execution["status"],"execution":execution});
        }
        action["operation"] == "RETRY_ATTEMPT"
            && matches!(
                execution["status"].as_str(),
                Some("BLOCKED" | "PREPARED" | "AUTHORIZED")
            )
            || submission["status"] == "EXECUTING"
                && matches!(
                    self.progress["phase"].as_str(),
                    Some("VALIDATING_SOURCE" | "STAGING")
                )
            || submission["status"] == "FAILED"
                && submission["execution"]["resolution"] == "FAILED_BEFORE_SUBMISSION"
    }
    fn progress(&mut self, id: &str, value: Value) -> Result<(), Error> {
        observation::bound_response(
            &value,
            &[
                ("submission_id", json!(id)),
                ("run_id", json!(self.reference.run_id)),
                ("attempt_id", json!(self.attempt)),
            ],
            "progress identity differs from the saved exact Submission",
        )?;
        self.progress = value.clone();
        self.evidence.insert("progress".into(), value);
        Ok(())
    }
    fn accept(
        mut self: Box<Self>,
        phase: Phase,
        response: Result<Value, Error>,
    ) -> Result<Step, Error> {
        match phase {
            Phase::Preparation => {
                self.preparation(response?)?;
                (*self).reference_evidence()
            }
            Phase::InitialProgress(id) => {
                self.progress(&id, response?)?;
                (*self).read_run()
            }
            Phase::Run => {
                match response {
                    Ok(value) => {
                        checked(
                            &value,
                            &self.reference.project,
                            &self.reference.run_id,
                            None,
                        )?;
                        self.run = value;
                    }
                    Err(Error::Http(error))
                        if error.status == 404
                            && (self.before_submission()
                                || self.evidence.contains_key("preparation")
                                    && !truth(&self.reference.saved["submission"])) => {}
                    Err(error) => return Err(error),
                }
                if truth(&self.run) {
                    let path = self.endpoint()? + "/attempts";
                    (*self).request(Phase::Attempts, path)
                } else {
                    (*self).after_attempts()
                }
            }
            Phase::Attempts => {
                let value = response?;
                checked(
                    &value,
                    &self.reference.project,
                    &self.reference.run_id,
                    None,
                )?;
                let entries = value["attempts"]
                    .as_array()
                    .ok_or_else(|| fail("invalid Run Attempt listing"))?;
                if self.attempt.is_none() && self.reference.state_kind.is_none() {
                    self.attempt = value["current_attempt_id"].as_str().map(str::to_owned);
                    if !value["current_attempt_id"].is_null() && self.attempt.is_none() {
                        return Err(fail("choose an exact attempt-NNN identity"));
                    }
                }
                if let Some(attempt) = &self.attempt {
                    validate_attempt(attempt)
                        .map_err(|_| fail("choose an exact attempt-NNN identity"))?;
                    let identity = format!("{}::{attempt}", self.reference.run_id);
                    let selected = entries
                        .iter()
                        .filter(|v| {
                            v.is_object()
                                && v["attempt_id"] == attempt.as_str()
                                && v["identity"] == identity
                        })
                        .collect::<Vec<_>>();
                    if selected.len() > 1 {
                        return Err(fail("Run listing contains duplicate exact Attempts"));
                    }
                    if let Some(row) = selected.first() {
                        self.selected = (*row).clone();
                        self.materialized = true;
                    } else if !self.before_submission()
                        || value["current_attempt_id"] == attempt.as_str()
                        || entries
                            .iter()
                            .any(|v| v.is_object() && v["attempt_id"] == attempt.as_str())
                    {
                        return Err(fail(
                            "requested exact Attempt is absent from the Run listing",
                        ));
                    }
                } else if !entries.is_empty() && self.reference.state_kind.is_none() {
                    return Err(fail(
                        "Run has Attempts but no current Attempt; choose --attempt",
                    ));
                }
                self.listing = value;
                (*self).after_attempts()
            }
            Phase::Discover => {
                let value = match response {
                    Ok(v) => v,
                    Err(Error::Http(e)) if ![401, 403].contains(&e.status) => {
                        self.unavailable_progress(&e);
                        return (*self).collection();
                    }
                    Err(e) => return Err(e),
                };
                checked(
                    &value,
                    &self.reference.project,
                    &self.reference.run_id,
                    None,
                )?;
                let rows = value["submissions"]
                    .as_array()
                    .ok_or_else(|| fail("invalid Submission listing"))?;
                let mut matches = Vec::new();
                for row in rows {
                    if !row.is_object()
                        || row["project"] != self.reference.project
                        || row["run_id"] != self.reference.run_id
                    {
                        return Err(fail("Submission listing contains a different Run identity"));
                    }
                    if row["attempt_id"] == json!(self.attempt) {
                        matches.push(row);
                    }
                }
                if matches.len() != 1 {
                    self.progress = json!({"availability":"UNAVAILABLE","diagnostic":if matches.is_empty(){"NO_EXACT_SUBMISSION"}else{"AMBIGUOUS_SUBMISSIONS"}});
                    return (*self).collection();
                }
                let selected = matches[0];
                let id = selected["submission_id"]
                    .as_str()
                    .ok_or_else(|| fail("invalid API identity"))?
                    .to_owned();
                let path = format!("/api/submissions/{}/progress", observation::segment(&id)?);
                self.evidence.insert("submission".into(), selected.clone());
                (*self).request(Phase::DiscoveredProgress(id), path)
            }
            Phase::DiscoveredProgress(id) => {
                match response {
                    Ok(v) => self.progress(&id, v)?,
                    Err(Error::Http(e)) if ![401, 403].contains(&e.status) => {
                        self.unavailable_progress(&e)
                    }
                    Err(e) => return Err(e),
                };
                (*self).collection()
            }
            Phase::Collection => {
                match response {
                    Ok(value) => {
                        checked(
                            &value,
                            &self.reference.project,
                            &self.reference.run_id,
                            self.attempt.as_deref(),
                        )?;
                        self.collection = value;
                    }
                    Err(Error::Http(error)) if [404, 409].contains(&error.status) => {
                        let brief = json!({"status":error.status,"error_code":error.error_code});
                        self.collection_error = Some(brief.clone());
                        self.evidence.insert("collection_error".into(),json!({"status":brief["status"],"error_code":brief["error_code"],"details":error.details}));
                    }
                    Err(error) => return Err(error),
                }
                (*self).expanded()
            }
            Phase::Expanded => {
                let value = response?;
                let identity = format!(
                    "{}::{}",
                    self.reference.run_id,
                    self.attempt.as_deref().unwrap_or("")
                );
                if !value.is_object()
                    || value["run_id"] != self.reference.run_id
                    || value["attempt"]["attempt_id"] != json!(self.attempt)
                    || value["scope"]
                        != json!({"project":self.reference.project,"scope_type":"attempt","object_id":identity})
                {
                    return Err(fail(
                        "expanded evidence differs from the requested exact Attempt",
                    ));
                }
                self.evidence.insert("attempt".into(), value);
                Ok(Step::Complete(self.finish()))
            }
        }
    }
    fn unavailable_progress(&mut self, error: &ml_exp_core::diagnostic::HttpFailure) {
        self.progress = json!({"availability":"UNAVAILABLE","diagnostic":"SUBMISSION_PROGRESS_UNAVAILABLE","error":{"status":error.status,"error_code":error.error_code}});
    }
    fn after_attempts(self) -> Result<Step, Error> {
        if self.materialized && self.progress_enabled && !self.evidence.contains_key("progress") {
            let path = format!(
                "/api/experiments/{}/{}/submissions",
                observation::segment(&self.reference.project)?,
                observation::segment(&self.reference.run_id)?
            );
            self.request(Phase::Discover, path)
        } else {
            self.collection()
        }
    }
    fn collection(self) -> Result<Step, Error> {
        if self.materialized && self.collection_enabled {
            let path = format!(
                "{}/attempts/{}/collection",
                self.endpoint()?,
                observation::segment(
                    self.attempt
                        .as_deref()
                        .ok_or_else(|| fail("choose an exact attempt-NNN identity"))?
                )?
            );
            self.request(Phase::Collection, path)
        } else {
            self.expanded()
        }
    }
    fn expanded(mut self) -> Result<Step, Error> {
        if self.details {
            self.evidence.insert("run".into(), self.run.clone());
            self.evidence
                .insert("attempts".into(), self.listing.clone());
            self.evidence
                .insert("collection".into(), self.collection.clone());
            if self.materialized {
                let identity = format!(
                    "{}::{}",
                    self.reference.run_id,
                    self.attempt.as_deref().unwrap_or("")
                );
                let path = format!(
                    "/api/attempts/{}/{}/show",
                    observation::segment(&self.reference.project)?,
                    observation::segment(&identity)?
                );
                return self.request(Phase::Expanded, path);
            }
        }
        Ok(Step::Complete(self.finish()))
    }
    fn layer(&self, name: &str) -> Value {
        if !self.materialized {
            return json!({});
        }
        let layer = mapping(&self.run["evidence"][name]);
        if layer["attempt_id"] == json!(self.attempt) {
            layer
        } else {
            json!({})
        }
    }
    fn finish(self) -> Value {
        let scheduler = self.layer("scheduler");
        let process = self.layer("process");
        let worker = self.layer("worker");
        let model = self.layer("model");
        let scheduler_detail = mapping(&scheduler["detail"]);
        let queue = otherwise(
            &mapping(&self.progress["queue"]),
            mapping(&scheduler_detail["queue"]),
        );
        let training = mapping(&self.collection["training"]);
        let mut model_view = json!({"state":model["state"],"step":mapping(&model["detail"])["step"],"last_metric":otherwise(&model["detail"],Value::Null),
            "observed_at":model["as_of"],"stale":get_or(&model,"stale",json!(true)),
            "metric_record":if truth(&model["detail"]){json!({"source":"indexed-model","sequence":null,"event_id":null})}else{Value::Null}});
        let metric = mapping(&self.progress["last_metric"]);
        let observation = mapping(&metric["observation"]);
        if model["state"].is_null() && truth(&observation) {
            let step = &observation["step"];
            let timestamp = &observation["timestamp"];
            model_view = json!({"state":if step.is_null(){"present".into()}else{format!("step {}",ml_exp_core::metrics::python_string(step))},"step":step,
                "last_metric":observation,"metric_record":{"sequence":metric["sequence"],"event_id":metric["event_id"],"source":metric["source"]},
                "observed_at":timestamp,"stale":if timestamp.is_null(){json!(true)}else{Value::Null}});
        }
        let progress_keys = [
            "availability",
            "phase",
            "message",
            "phase_started_at",
            "last_progress_unix",
            "seconds_since_progress",
            "no_progress_warning",
            "diagnostic",
            "error",
        ];
        let progress = Value::Object(
            progress_keys
                .into_iter()
                .filter_map(|key| self.progress.get(key).map(|v| (key.to_owned(), v.clone())))
                .collect(),
        );
        let mut value = json!({"project":self.reference.project,"run_id":self.reference.run_id,"attempt_id":self.attempt,"preparation":self.preparation,
            "scheduler":{"state":otherwise(&scheduler["state"],otherwise(&self.selected["state"],json!(if self.materialized{"UNKNOWN"}else{"NOT_SUBMITTED"}))),
                "backend":scheduler_detail["backend"],"job_id":scheduler_detail["backend_job_id"],"native_state":scheduler_detail["raw_state"],
                "observed_at":scheduler["as_of"],"stale":get_or(&scheduler,"stale",json!(true)),"reason":otherwise(&queue["reason"],otherwise(&scheduler_detail["reason"],scheduler_detail["pending_reason"].clone()))},
            "training":{"status":get_or(&training,"status",json!("UNKNOWN")),"exit_code":training["exit_code"],"process_state":process["state"],"observed_at":process["as_of"],"stale":get_or(&process,"stale",json!(true))},
            "publication":{"status":get_or(&self.collection,"status",json!("NOT_OBSERVED")),"phase":self.collection["phase"],"download_available":self.collection["status"]=="AVAILABLE"&&self.collection["verification"]=="SHA256_VERIFIED"&&self.collection["download_available"]==true,
                "diagnostic":self.collection["diagnostic"],"error":self.collection_error},
            "verification":{"status":get_or(&self.collection,"verification",json!("NOT_OBSERVED"))},"progress":progress,
            "queue":otherwise(&queue,Value::Null),"worker":{"state":worker["state"],"observed_at":worker["as_of"],"stale":get_or(&worker,"stale",json!(true))},
            "model":model_view,"input_delivery":self.progress["input_delivery"],"worker_phase":self.progress["worker_phase"]});
        if self.details {
            value["evidence"] = Value::Object(self.evidence);
        }
        value
    }
}

/// Explicit effect shell: every planned operation is a GET. No provider refresh,
/// result collection, retry, cancellation or acknowledgement is dispatched.
pub fn observe(
    client: &Client,
    health: &Value,
    saved: Option<&Value>,
    selection: Selection<'_>,
    details: bool,
) -> Result<Value, Error> {
    let mut step = begin(client.base_url(), health, saved, selection, details)?;
    loop {
        step = match step {
            Step::Complete(value) => return Ok(value),
            Step::Read(read) => {
                let response = client.get(read.path());
                read.accept(response)?
            }
        };
    }
}
