//! Exact result requests plus a streaming local verification shell. Complete
//! files are durable before ACK; GET/download never authorizes deletion alone.
use crate::{Client, Error, session};
use ml_exp_core::{
    artifacts::{self, publication::State},
    operations::validate_submission_id,
};
use serde_json::{Value, json};
use std::{
    fs::File,
    path::{Path, PathBuf},
};
mod files;
pub use files::{download as download_file, list as files};
#[derive(Clone)]
pub struct Plan {
    path: String,
    body: Option<Vec<u8>>,
    action: String,
}
impl Plan {
    pub(crate) fn path(&self) -> &str {
        &self.path
    }
    pub(crate) fn body(&self) -> Option<&[u8]> {
        self.body.as_deref()
    }
}
fn invalid() -> Error {
    Error::Observation("invalid exact artifact identity or archive proof; no acknowledgement sent")
}
fn enabled(health: &Value) -> Result<(), Error> {
    session::validate_health(health)?;
    if !health["capabilities"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v == "native-artifacts.v1"))
    {
        return Err(Error::Observation(
            "server has not enabled native-artifacts.v1",
        ));
    }
    Ok(())
}
pub fn plan(action: &str, operation: &str, body: Option<Value>) -> Result<Plan, Error> {
    validate_submission_id(action).map_err(|_| invalid())?;
    if !matches!(
        (operation, body.is_some()),
        ("artifacts", false)
            | ("collect", true)
            | ("acknowledge", true)
            | ("download-ticket", true)
    ) {
        return Err(invalid());
    }
    let body = body
        .map(|v| serde_json::to_vec(&v).map_err(|_| invalid()))
        .transpose()?;
    if body.as_ref().is_some_and(|b| b.len() > 8 * 1024 * 1024) {
        return Err(invalid());
    }
    Ok(Plan {
        path: format!("/api/submissions/{action}/{operation}"),
        body,
        action: action.into(),
    })
}
pub fn validate(plan: &Plan, value: Value) -> Result<(Value, State), Error> {
    if value["submission_id"] != plan.action || value["scheduler_mutation_performed"] != false {
        return Err(Error::Response);
    }
    let state: State =
        serde_json::from_value(value["state"].clone()).map_err(|_| Error::Response)?;
    state.validate().map_err(|_| Error::Response)?;
    if value["download_available"] != state.available().map_err(|_| Error::Response)? {
        return Err(Error::Response);
    }
    Ok((value, state))
}
pub fn observe(
    client: &Client,
    action: &str,
    collect: bool,
    health: &Value,
) -> Result<Value, Error> {
    enabled(health)?;
    let plan = plan(
        action,
        if collect { "collect" } else { "artifacts" },
        if collect { Some(json!({})) } else { None },
    )?;
    let result = client.send_artifact(&plan)?;
    if collect
        && result["submission_id"] == action
        && result["status"] == "COLLECTING"
        && result["download_available"] == false
        && result["training_replayed"] == false
        && result["synchronization"]["resources"]["gpus"] == 0
    {
        return Ok(result);
    }
    Ok(validate(&plan, result)?.0)
}
pub struct DownloadPlan {
    path: String,
    bytes: u64,
}
impl DownloadPlan {
    pub(crate) fn path(&self) -> &str {
        &self.path
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}
fn destination(path: &Path) -> Result<PathBuf, Error> {
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir().map_err(|_| invalid())?.join(path)
    };
    if path.file_name().is_none()
        || path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(invalid());
    }
    let parent = path.parent().ok_or_else(invalid)?;
    ml_exp_io::directory::Directory::open(parent, false).map_err(|_| invalid())?;
    Ok(path)
}
/// A repeated command verifies the existing complete local tar and sends only
/// the same proof. It never overwrites a file or redownloads after release.
pub fn download(
    client: &Client,
    action: &str,
    output: &Path,
    release: bool,
    health: &Value,
) -> Result<Value, Error> {
    enabled(health)?;
    let output = destination(output)?;
    let metadata = observe(client, action, false, health)?;
    let metadata_plan = plan(action, "artifacts", None)?;
    let direct = metadata["download"]["kind"] == "oss-parts";
    let (_, state) = validate(&metadata_plan, metadata.clone())?;
    let receipt = &state.bundle.archive;
    let reused = match std::fs::symlink_metadata(&output) {
        Ok(m) => {
            if !m.is_file() || m.file_type().is_symlink() {
                return Err(invalid());
            }
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => return Err(invalid()),
    };
    let proof = if reused {
        let mut file = File::open(&output).map_err(|_| invalid())?;
        ml_exp_io::artifacts::verify(&mut file, receipt).map_err(|_| {
            Error::Observation(
                "existing local archive differs; it was not overwritten and no ACK was sent",
            )
        })?
    } else {
        if !state.available().map_err(|_| invalid())? {
            return Err(Error::Observation(
                "archive was released; no remote checkpoint or original output is deleted",
            ));
        }
        let mut temporary = tempfile::NamedTempFile::new_in(output.parent().ok_or_else(invalid)?)
            .map_err(|_| invalid())?;
        let plan = DownloadPlan {
            path: format!("/api/submissions/{action}/artifacts/{}", receipt.sha256),
            bytes: receipt.bytes,
        };
        if direct {
            let part_bytes = 16 * 1024 * 1024u64;
            let count = receipt.bytes.div_ceil(part_bytes);
            if metadata["download"]["part_bytes"] != part_bytes
                || metadata["download"]["part_count"] != count
                || count > 65536
            {
                return Err(Error::Response);
            }
            for part in 0..count {
                let request = self::plan(action, "download-ticket", Some(json!({"part":part})))?;
                let ticket = client.send_artifact(&request)?;
                let bytes = (receipt.bytes - part * part_bytes).min(part_bytes);
                if ticket["part"] != part
                    || ticket["bytes"] != bytes
                    || ticket["headers"] != json!({})
                {
                    return Err(Error::Response);
                }
                client.download_object(
                    ticket["url"].as_str().ok_or(Error::Response)?,
                    bytes,
                    temporary.as_file_mut(),
                )?;
            }
        } else {
            client.download_artifact(&plan, temporary.as_file_mut())?;
        }
        temporary.as_file().sync_all().map_err(|_| invalid())?;
        let proof =
            ml_exp_io::artifacts::verify(temporary.as_file_mut(), receipt).map_err(|_| {
                Error::Observation("download hash or file manifest differs; no ACK was sent")
            })?;
        temporary.persist_noclobber(&output).map_err(|_| {
            Error::Observation("destination changed; no existing file was overwritten or ACK sent")
        })?;
        File::open(output.parent().ok_or_else(invalid)?)
            .and_then(|f| f.sync_all())
            .map_err(|_| invalid())?;
        proof
    };
    artifacts::proof_matches(receipt, &proof).map_err(|_| invalid())?;
    let ack = plan(
        action,
        "acknowledge",
        Some(json!({"proof":proof,"release":release})),
    )?;
    let reply = client
        .send_artifact(&ack)
        .map_err(|error| Error::ArtifactAcknowledgementUnconfirmed(Box::new(error)))?;
    let (_, mut after) = validate(&ack, reply)?;
    if after.bundle != state.bundle
        || after.acknowledgement.as_ref().map(|a| &a.proof) != Some(&proof)
    {
        return Err(Error::Response);
    }
    if direct && release {
        for _ in 0..1024 {
            if after.retention.status == artifacts::RetentionStatus::Released {
                break;
            }
            let reply = client
                .send_artifact(&ack)
                .map_err(|error| Error::ArtifactAcknowledgementUnconfirmed(Box::new(error)))?;
            let (_, next) = validate(&ack, reply)?;
            after
                .validate_successor(&next)
                .map_err(|_| Error::Response)?;
            after = next;
        }
        if after.retention.status != artifacts::RetentionStatus::Released {
            return Err(Error::Observation(
                "archive verified locally; release remains pending",
            ));
        }
    }
    Ok(
        json!({"submission_id":action,"local_archive":output,"local_archive_verified":true,"reused_existing_archive":reused,"release_requested":release,"retention":after.retention,"scheduler_mutation_performed":false}),
    )
}
