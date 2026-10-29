//! One reviewed configuration, a deterministic recovery URL, and private state.
//! A lost response never causes an automatic authorize/execute or new Run ID.
use crate::{Client, Error};
use ml_exp_core::preparation::Request;
use serde_json::{Value, json};
use std::{io::Read, path::Path};
fn invalid() -> Error {
    Error::Observation("invalid experiment preparation configuration or saved identity")
}
pub fn load(project: &str, path: &Path) -> Result<Request, Error> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|_| invalid())?
        .take(131073)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid())?;
    if bytes.len() > 131072 {
        return Err(invalid());
    }
    let request: Request = serde_json::from_value(Value::Object(
        ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?,
    ))
    .map_err(|_| invalid())?;
    request.validate(project).map_err(|_| invalid())?;
    Ok(request)
}
pub fn prepare(
    client: &Client,
    project: &str,
    request: &Request,
    state: &Path,
    health: &Value,
) -> Result<Value, Error> {
    if !health["capabilities"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v == "native-experiment-preparations.v1"))
    {
        return Err(Error::Observation(
            "server has not configured experiment preparations",
        ));
    }
    let id = request.id(project).map_err(|_| invalid())?;
    let recovery = format!("/api/projects/{project}/experiment-preparations/{id}");
    let seed = json!({"server":client.base_url(),"config_sha256":ml_exp_core::canonical::protocol_identity(&json!(request)).map_err(|_|invalid())?,"project":project,"run_id":request.run.get("run_id"),"preparation":{"preparation_id":id},"request":request,"recovery_path":recovery});
    if state.exists() {
        let existing: Value = serde_json::from_slice(
            &ml_exp_io::read_private(state, 256 * 1024, true).map_err(|_| invalid())?,
        )
        .map_err(|_| invalid())?;
        if existing["server"] != seed["server"]
            || existing["project"] != seed["project"]
            || existing["request"] != seed["request"]
        {
            return Err(invalid());
        }
    } else {
        ml_exp_io::write_atomic(
            state,
            &serde_json::to_vec(&seed).map_err(|_| invalid())?,
            true,
        )
        .map_err(|_| invalid())?;
    }
    let value = client.post(
        &format!("/api/projects/{project}/experiment-preparations"),
        &json!(request),
    )?;
    if value["project"] != project || value["preparation_id"] != id {
        return Err(Error::Response);
    }
    let mut saved = seed;
    saved["preparation"] = value.clone();
    ml_exp_io::write_atomic(
        state,
        &serde_json::to_vec(&saved).map_err(|_| invalid())?,
        false,
    )
    .map_err(|_| invalid())?;
    Ok(value)
}
pub fn advance(
    client: &Client,
    project: &str,
    id: &str,
    confirmation: &str,
) -> Result<Value, Error> {
    ml_exp_core::identity::validate_id(project).map_err(|_| invalid())?;
    if !id.strip_prefix("preparation-").is_some_and(|s| {
        s.len() == 32
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    }) || confirmation != format!("PREPARE {id}")
    {
        return Err(invalid());
    }
    let value = client.post(
        &format!("/api/projects/{project}/experiment-preparations/{id}/continue"),
        &json!({"confirmation":confirmation}),
    )?;
    if value["project"] != project || value["preparation_id"] != id {
        return Err(Error::Response);
    }
    Ok(value)
}
