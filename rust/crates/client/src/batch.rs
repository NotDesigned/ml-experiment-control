//! A batch is a collection of the ordinary recoverable preparation workflow.
//! Every child identity is saved before sending its first request. No batch
//! Preparation never authorizes training. Explicit batch dispatch is bounded
//! and never replays an uncertain scheduler submission.
use crate::{Client, Error, preparations, submissions};
use ml_exp_core::preparation::{Batch, Request};
use serde_json::{Value, json};
use std::{io::Read, path::Path};
fn invalid() -> Error {
    Error::Observation("invalid bounded batch configuration or private recovery state")
}
pub fn load(project: &str, path: &Path) -> Result<Batch, Error> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|_| invalid())?
        .take(131073)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid())?;
    if bytes.len() > 131072 {
        return Err(invalid());
    }
    let batch: Batch = serde_json::from_value(Value::Object(
        ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?,
    ))
    .map_err(|_| invalid())?;
    batch.expand(project).map_err(|_| invalid())?;
    Ok(batch)
}
pub fn prepare(
    client: &Client,
    project: &str,
    batch: &Batch,
    directory: &Path,
    health: &Value,
) -> Result<Value, Error> {
    let requests = batch.expand(project).map_err(|_| invalid())?;
    let absolute = if directory.is_absolute() {
        directory.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|_| invalid())?
            .join(directory)
    };
    let root = ml_exp_io::directory::Directory::open(&absolute, true).map_err(|_| invalid())?;
    root.verify_private_parent().map_err(|_| invalid())?;
    let manifest = json!({"contract":"native-batch.v1","server":client.base_url(),"project":project,"batch":batch});
    let file = std::ffi::OsStr::new("batch.json");
    let bytes = serde_json::to_vec(&manifest).map_err(|_| invalid())?;
    if root.entry_exists(file).map_err(|_| invalid())? {
        if root.read_regular(file, 131072).map_err(|_| invalid())? != bytes {
            return Err(invalid());
        }
    } else {
        root.publish_new(file, &bytes).map_err(|_| invalid())?;
    }
    let mut result = Vec::new();
    for request in requests {
        let path = root.path().join(format!("{}.json", request.request_id));
        let value = preparations::prepare(client, project, &request, &path, health)?;
        result.push(value);
    }
    Ok(
        json!({"project":project,"batch_id":batch.request_id,"confirmation":format!("SUBMIT BATCH {}",batch.request_id),"prepared":result,"training_authorized":false,"scheduler_submitted":false,"state":directory}),
    )
}
pub fn status(client: &Client, directory: &Path) -> Result<Value, Error> {
    let bytes = ml_exp_io::read_private(&directory.join("batch.json"), 131072, true)
        .map_err(|_| invalid())?;
    let value = Value::Object(ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?);
    if value["contract"] != "native-batch.v1" || value["server"] != client.base_url() {
        return Err(invalid());
    }
    let project = value["project"].as_str().ok_or_else(invalid)?;
    let batch: Batch = serde_json::from_value(value["batch"].clone()).map_err(|_| invalid())?;
    let requests: Vec<Request> = batch.expand(project).map_err(|_| invalid())?;
    let mut rows = Vec::new();
    for request in requests {
        let id = request.id(project).map_err(|_| invalid())?;
        let path = format!("/api/projects/{project}/experiment-preparations/{id}");
        match client.get(&path){
            Ok(mut value)=>{if value["project"]!=project||value["preparation_id"]!=id||value["run_id"]!=request.run["run_id"]{return Err(Error::Response);}
                if value["submission"]["submission_id"].is_string(){
                    let run=request.run["run_id"].as_str().ok_or_else(invalid)?;
                    let plan=submissions::plan(submissions::Command::Existing{project,run})?;
                    let current=submissions::response(&plan,client.get(&format!("/api/runs/{project}/{run}/submission"))?)?;
                    if current["project"]!=project||current["run_id"]!=request.run["run_id"]||current["attempt_id"]!="attempt-001"{return Err(Error::Response);}
                    value["submission"]=current;
                }
                rows.push(value);},
            Err(_)=>rows.push(json!({"project":project,"run_id":request.run["run_id"],"preparation_id":id,"status":"UNKNOWN","recovery_path":path})),
        }
    }
    Ok(json!({"project":project,"experiments":rows}))
}
/// One explicit preparation pass: build/reconcile environments, stage data and
/// freeze each ordinary Submission. It cannot authorize or start training.
pub fn advance(client: &Client, directory: &Path, confirmation: &str) -> Result<Value, Error> {
    let bytes = ml_exp_io::read_private(&directory.join("batch.json"), 131072, true)
        .map_err(|_| invalid())?;
    let manifest = Value::Object(ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?);
    let batch: Batch = serde_json::from_value(manifest["batch"].clone()).map_err(|_| invalid())?;
    if manifest["contract"] != "native-batch.v1"
        || manifest["server"] != client.base_url()
        || confirmation != format!("PREPARE BATCH {}", batch.request_id)
    {
        return Err(invalid());
    }
    let project = manifest["project"].as_str().ok_or_else(invalid)?;
    let mut rows = Vec::new();
    for request in batch.expand(project).map_err(|_| invalid())? {
        let id = request.id(project).map_err(|_| invalid())?;
        let value = preparations::advance(client, project, &id, &format!("PREPARE {id}"))?;
        rows.push(value);
    }
    Ok(
        json!({"project":project,"batch_id":batch.request_id,"experiments":rows,"training_authorized":false,"scheduler_submitted":false}),
    )
}
/// One explicit scheduling pass. The cap covers this saved batch; unknown
/// executions consume a slot. A subsequent pass can submit the next children.
pub fn dispatch(
    client: &Client,
    directory: &Path,
    max_active: usize,
    confirmation: &str,
    health: &Value,
) -> Result<Value, Error> {
    if !(1..=ml_exp_core::preparation::MAX_BATCH).contains(&max_active) {
        return Err(invalid());
    }
    let root = if directory.is_absolute() {
        directory.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|_| invalid())?
            .join(directory)
    };
    let directory_cap =
        ml_exp_io::directory::Directory::open(&root, false).map_err(|_| invalid())?;
    let lock = directory_cap
        .open_lock(std::ffi::OsStr::new(".dispatch.lock"))
        .map_err(|_| invalid())?;
    fs2::FileExt::try_lock_exclusive(&lock).map_err(|_| invalid())?;
    let value = Value::Object(
        ml_exp_core::json::mapping(
            &directory_cap
                .read_regular(std::ffi::OsStr::new("batch.json"), 131072)
                .map_err(|_| invalid())?,
        )
        .map_err(|_| invalid())?,
    );
    let batch: Batch = serde_json::from_value(value["batch"].clone()).map_err(|_| invalid())?;
    if value["server"] != client.base_url()
        || confirmation != format!("SUBMIT BATCH {}", batch.request_id)
    {
        return Err(invalid());
    }
    let mut state = status(client, &root)?;
    let rows = state["experiments"].as_array_mut().ok_or_else(invalid)?;
    let mut active = rows
        .iter()
        .filter(|row| ml_exp_core::preparation::occupied(row))
        .count();
    for row in rows.iter_mut() {
        if active >= max_active {
            break;
        }
        let s = &row["submission"];
        if !s["status"]
            .as_str()
            .is_some_and(|v| matches!(v, "PREPARED" | "AUTHORIZED"))
        {
            continue;
        }
        let action = s["submission_id"].as_str().ok_or_else(invalid)?.to_owned();
        let mut current = s.clone();
        if s["status"] == "PREPARED" {
            current = submissions::execute(
                client,
                &submissions::plan(submissions::Command::Authorize {
                    submission: &action,
                    reason: confirmation,
                })?,
                health,
            )?;
        }
        if current["status"] != "AUTHORIZED" {
            return Err(Error::Observation(
                "batch authorization is unresolved; inspect the original Submission",
            ));
        }
        let confirmation = current["confirmation"].as_str().ok_or_else(invalid)?;
        let claim = std::ffi::OsString::from(format!("{action}.dispatch.json"));
        if directory_cap.entry_exists(&claim).map_err(|_| invalid())? {
            row["diagnostic"] =
                json!("DISPATCH_ALREADY_REQUESTED; inspect or reconcile the original Submission");
            active += 1;
            continue;
        }
        directory_cap
            .publish_new(
                &claim,
                &serde_json::to_vec(&json!({"submission_id":action,"confirmation":confirmation}))
                    .map_err(|_| invalid())?,
            )
            .map_err(|_| invalid())?;
        row["submission"] = submissions::execute(
            client,
            &submissions::plan(submissions::Command::Execute {
                submission: &action,
                confirmation,
            })?,
            health,
        )?;
        active += 1;
    }
    state["max_active"] = json!(max_active);
    state["occupied_slots"] = json!(active);
    state["automatic_resubmission"] = json!(false);
    Ok(state)
}
