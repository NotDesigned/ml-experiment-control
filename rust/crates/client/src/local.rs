//! Attach an existing append-only JSONL file. No process signals or scheduler calls.
//! The exact pending packet is durable before HTTP; only an exact ACK advances it.
use crate::{Client, Error};
use ml_exp_core::local::{Finish, INITIAL_CHAIN, MAX_BATCH, Packet, Registration};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    ffi::OsString,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Duration,
};

fn invalid() -> Error {
    Error::Observation(
        "local attachment identity, saved cursor or accepted file prefix differs; keep the original state",
    )
}
fn source() -> Error {
    Error::Observation(
        "metrics input was replaced, truncated or cannot be read as a regular append-only file",
    )
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pending {
    packet: Packet,
    prefix_sha256: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    format: String,
    server: String,
    project: String,
    request: Registration,
    offset: u64,
    chain_sha256: String,
    prefix_sha256: String,
    finished: Option<Finish>,
    pending: Option<Pending>,
}
fn save(path: &Path, saved: &Saved) -> Result<(), Error> {
    ml_exp_io::write_atomic(
        path,
        &serde_json::to_vec(saved).map_err(|_| invalid())?,
        false,
    )
    .map_err(|_| invalid())
}
fn definition(path: &Path, execution: &str) -> Result<Registration, Error> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|_| invalid())?
        .take(MAX_BATCH as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid())?;
    if bytes.len() > MAX_BATCH {
        return Err(invalid());
    }
    let mut value = Value::Object(ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?);
    if value.get("execution_id").is_none() {
        value["execution_id"] = json!(execution);
    }
    serde_json::from_value(value).map_err(|_| invalid())
}
fn sha(hasher: &Sha256) -> String {
    format!("{:x}", hasher.clone().finalize())
}
fn prefix(file: &mut File, length: u64) -> Result<Sha256, Error> {
    file.seek(SeekFrom::Start(0)).map_err(|_| source())?;
    let mut hasher = Sha256::new();
    let mut remaining = length;
    let mut buffer = [0u8; MAX_BATCH];
    while remaining > 0 {
        let count = file
            .read(&mut buffer[..remaining.min(MAX_BATCH as u64) as usize])
            .map_err(|_| source())?;
        if count == 0 {
            return Err(source());
        }
        hasher.update(&buffer[..count]);
        remaining -= count as u64;
    }
    Ok(hasher)
}
fn current(file: &File, path: &Path, offset: u64) -> Result<(), Error> {
    let selected = File::open(path).map_err(|_| source())?;
    if !file.metadata().map_err(|_| source())?.is_file()
        || !ml_exp_io::same_file(file, &selected).map_err(|_| source())?
        || selected.metadata().map_err(|_| source())?.len() < offset
    {
        return Err(source());
    }
    Ok(())
}
fn accepted(saved: &Saved, pending: &Pending, ack: &Value) -> Result<(), Error> {
    let digest = pending.packet.digest().map_err(|_| invalid())?;
    let end = saved
        .offset
        .checked_add(pending.packet.bytes.len() as u64)
        .ok_or_else(invalid)?;
    if ack["project"] != saved.project
        || ack["run_id"] != saved.request.run_id
        || ack["execution_id"] != saved.request.execution_id
        || ack["packet_sha256"] != digest
        || ack["chain_sha256"] != digest
        || ack["offset"] != end
        || ack["finished"] != json!(pending.packet.finish)
    {
        return Err(Error::Response);
    }
    Ok(())
}
fn deliver(client: &Client, saved: &mut Saved, state: &Path) -> Result<Value, Error> {
    let pending = saved.pending.as_ref().ok_or_else(invalid)?;
    let ack = client.post(
        &format!(
            "/api/local-runs/{}/{}/metrics",
            saved.project, saved.request.run_id
        ),
        &json!(pending.packet),
    )?;
    accepted(saved, pending, &ack)?;
    saved.offset = ack["offset"].as_u64().ok_or(Error::Response)?;
    saved.chain_sha256 = ack["chain_sha256"].as_str().ok_or(Error::Response)?.into();
    saved.finished = pending.packet.finish;
    saved.prefix_sha256 = pending.prefix_sha256.clone();
    saved.pending = None;
    save(state, saved)?;
    Ok(ack)
}
fn bound_status(saved: &Saved, value: &Value) -> Result<(), Error> {
    if value["project"] != saved.project
        || value["run_id"] != saved.request.run_id
        || value["execution_id"] != saved.request.execution_id
        || value["execution_kind"] != "local"
        || !value["submission_id"].is_null()
        || !value["provider_job_id"].is_null()
        || value["cursor"]["offset"] != saved.offset
        || value["cursor"]["chain_sha256"] != saved.chain_sha256
        || value["cursor"]["finished"] != json!(saved.finished)
    {
        return Err(invalid());
    }
    Ok(())
}
pub fn status(client: &Client, project: &str, run: &str) -> Result<Value, Error> {
    ml_exp_core::identity::validate_id(project)
        .and_then(|_| ml_exp_core::identity::validate_id(run))
        .map_err(|_| invalid())?;
    client.get(&format!("/api/local-runs/{project}/{run}"))
}
/// A missing execution ID is generated once and saved before registration. The
/// definition is scientific metadata only: no Runtime, cloud Job or process PID.
pub enum Mode {
    Once,
    Watch,
    Finish(Finish),
}
pub fn attach(
    client: &Client,
    project: &str,
    definition_path: &Path,
    metrics: &Path,
    state: &Path,
    mode: Mode,
    mut progress: impl FnMut(&Value),
) -> Result<Value, Error> {
    let watch = matches!(mode, Mode::Watch);
    let finish = if let Mode::Finish(finish) = mode {
        Some(finish)
    } else {
        None
    };
    let parent = state
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = std::fs::canonicalize(parent).map_err(|_| invalid())?;
    let directory = ml_exp_io::directory::Directory::open(&parent, false).map_err(|_| invalid())?;
    let name = state.file_name().ok_or_else(invalid)?;
    let state = parent.join(name);
    let mut lock_name = OsString::from(name);
    lock_name.push(".lock");
    let _lock = directory
        .lock(
            &lock_name,
            ml_exp_io::directory::LockOptions {
                create: true,
                ..Default::default()
            },
        )
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::WouldBlock {
                Error::Observation("another uploader holds this attachment state")
            } else {
                invalid()
            }
        })?;
    let mut saved = if state.exists() {
        let value: Saved = serde_json::from_slice(
            &ml_exp_io::read_private(&state, 1024 * 1024, true).map_err(|_| invalid())?,
        )
        .map_err(|_| invalid())?;
        if value.format != "native-local-attach.v1"
            || value.server != client.base_url()
            || value.project != project
            || json!(definition(definition_path, &value.request.execution_id)?)
                != json!(value.request)
        {
            return Err(invalid());
        }
        value
    } else {
        let mut random = [0u8; 16];
        getrandom::fill(&mut random).map_err(|_| invalid())?;
        let execution = format!("{:032x}", u128::from_be_bytes(random));
        Saved {
            format: "native-local-attach.v1".into(),
            server: client.base_url().into(),
            project: project.into(),
            request: definition(definition_path, &execution)?,
            offset: 0,
            chain_sha256: INITIAL_CHAIN.into(),
            prefix_sha256: sha(&Sha256::new()),
            finished: None,
            pending: None,
        }
    };
    saved.request.validate(project).map_err(|_| invalid())?;
    let mut file = File::open(metrics).map_err(|_| source())?;
    current(&file, metrics, saved.offset)?;
    let mut hasher = prefix(&mut file, saved.offset)?;
    if sha(&hasher) != saved.prefix_sha256 {
        return Err(source());
    }
    if let Some(pending) = &saved.pending {
        if pending.packet.execution_id != saved.request.execution_id
            || pending.packet.offset != saved.offset
            || pending.packet.previous_chain_sha256 != saved.chain_sha256
        {
            return Err(invalid());
        }
        pending.packet.digest().map_err(|_| invalid())?;
        let mut bytes = vec![0u8; pending.packet.bytes.len()];
        file.read_exact(&mut bytes).map_err(|_| source())?;
        if bytes != pending.packet.bytes {
            return Err(source());
        }
        hasher.update(&bytes);
        if sha(&hasher) != pending.prefix_sha256 {
            return Err(source());
        }
    }
    if saved.finished.is_some()
        && (file.metadata().map_err(|_| source())?.len() != saved.offset
            || finish.is_some_and(|f| Some(f) != saved.finished))
    {
        return Err(invalid());
    }
    save(&state, &saved)?;
    let health = client.health()?;
    if !health["capabilities"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v == "native-local-metrics.v1"))
    {
        return Err(Error::Observation(
            "server does not support native-local-metrics.v1",
        ));
    }
    let mut value = client.post(
        &format!("/api/projects/{project}/local-runs"),
        &json!(saved.request),
    )?;
    if saved.pending.is_some() {
        progress(&deliver(client, &mut saved, &state)?);
        value = status(client, project, &saved.request.run_id)?;
    }
    bound_status(&saved, &value)?;
    if saved.finished.is_some() {
        return Ok(value);
    }
    loop {
        current(&file, metrics, saved.offset)?;
        let mut bytes = vec![0u8; MAX_BATCH];
        let count = file.read(&mut bytes).map_err(|_| source())?;
        bytes.truncate(count);
        if count == 0 && finish.is_none() {
            if !watch {
                return status(client, project, &saved.request.run_id);
            }
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }
        hasher.update(&bytes);
        let packet = Packet {
            execution_id: saved.request.execution_id.clone(),
            offset: saved.offset,
            previous_chain_sha256: saved.chain_sha256.clone(),
            bytes,
            finish: if count == 0 { finish } else { None },
        };
        saved.pending = Some(Pending {
            packet,
            prefix_sha256: sha(&hasher),
        });
        save(&state, &saved)?;
        let ack = deliver(client, &mut saved, &state)?;
        progress(&ack);
        if saved.finished.is_some() {
            return status(client, project, &saved.request.run_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acknowledgements_must_match_packet_scope_bytes_order_and_completion() {
        let request: Registration =
            serde_json::from_value(json!({"run_id":"train","execution_id":"a".repeat(32)}))
                .unwrap();
        let packet = Packet {
            execution_id: request.execution_id.clone(),
            offset: 0,
            previous_chain_sha256: INITIAL_CHAIN.into(),
            bytes: b"abc".into(),
            finish: None,
        };
        let digest = packet.digest().unwrap();
        let saved = Saved {
            format: "native-local-attach.v1".into(),
            server: "https://example.com".into(),
            project: "demo".into(),
            request,
            offset: 0,
            chain_sha256: INITIAL_CHAIN.into(),
            prefix_sha256: sha(&Sha256::new()),
            finished: None,
            pending: None,
        };
        let pending = Pending {
            packet,
            prefix_sha256: "unused".into(),
        };
        let ack = json!({"project":"demo","run_id":"train","execution_id":"a".repeat(32),"packet_sha256":digest,"chain_sha256":digest,"offset":3,"finished":null});
        accepted(&saved, &pending, &ack).unwrap();
        for (key, value) in [
            ("project", json!("other")),
            ("run_id", json!("other")),
            ("execution_id", json!("b".repeat(32))),
            ("packet_sha256", json!("b".repeat(64))),
            ("chain_sha256", json!(INITIAL_CHAIN)),
            ("offset", json!(4)),
            ("finished", json!("SUCCEEDED")),
        ] {
            let mut wrong = ack.clone();
            wrong[key] = value;
            assert!(accepted(&saved, &pending, &wrong).is_err(), "{key}");
        }
    }
    #[test]
    fn saved_server_scope_mismatch_fails_before_any_network_request() {
        let temp = tempfile::tempdir().unwrap();
        let definition = temp.path().join("local.json");
        let metrics = temp.path().join("metrics");
        let state = temp.path().join("state");
        let request: Registration =
            serde_json::from_value(json!({"run_id":"train","execution_id":"a".repeat(32)}))
                .unwrap();
        std::fs::write(&definition, serde_json::to_vec(&request).unwrap()).unwrap();
        std::fs::write(&metrics, b"").unwrap();
        let saved = Saved {
            format: "native-local-attach.v1".into(),
            server: "https://different.example".into(),
            project: "demo".into(),
            request,
            offset: 0,
            chain_sha256: INITIAL_CHAIN.into(),
            prefix_sha256: sha(&Sha256::new()),
            finished: None,
            pending: None,
        };
        save(&state, &saved).unwrap();
        let before = std::fs::read(&state).unwrap();
        let client = Client::new(
            "http://127.0.0.1:9",
            "synthetic-00000000000000000000000000000000".into(),
        )
        .unwrap();
        assert!(matches!(
            attach(
                &client,
                "demo",
                &definition,
                &metrics,
                &state,
                Mode::Once,
                |_| {}
            ),
            Err(Error::Observation(_))
        ));
        assert_eq!(std::fs::read(state).unwrap(), before);
    }
}
