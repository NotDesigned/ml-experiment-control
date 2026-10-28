//! Deterministic local packing and exact-identity resumable transfer. Only the
//! backend gateway keeps dataset payloads; the API acknowledges durable parts.
use crate::{Client, Error};
use ml_exp_core::{
    datasets::{self, Definition, State},
    multipart::PartRecord,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Duration,
};
pub(crate) struct Plan {
    pub(crate) path: String,
    pub(crate) method: reqwest::Method,
    pub(crate) body: Vec<u8>,
    pub(crate) timeout: Duration,
}
pub fn validate(project: &str, executor: &str, asset: Option<&str>) -> Result<(), Error> {
    ml_exp_core::identity::validate_id(project).map_err(|_| Error::Path)?;
    ml_exp_core::identity::validate_id(executor).map_err(|_| Error::Path)?;
    if asset.is_some_and(|v| !datasets::asset_id(v)) {
        return Err(Error::Path);
    }
    Ok(())
}
fn capability(health: &Value) -> Result<(), Error> {
    if !health["capabilities"]
        .as_array()
        .is_some_and(|v| v.iter().any(|s| s == "native-datasets.v1"))
    {
        return Err(Error::Observation(
            "native dataset delivery is not configured on this server",
        ));
    }
    Ok(())
}
fn state(value: Value, project: &str, executor: &str, asset: &str) -> Result<State, Error> {
    let state: State = serde_json::from_value(value).map_err(|_| Error::Response)?;
    state.validate().map_err(|_| Error::Response)?;
    if state.project != project || state.target.backend_id != executor || state.asset_id != asset {
        return Err(Error::Response);
    }
    Ok(state)
}
fn send(
    client: &Client,
    path: String,
    method: reqwest::Method,
    body: Vec<u8>,
    complete: bool,
) -> Result<Value, Error> {
    client.send_dataset(&Plan {
        path,
        method,
        body,
        timeout: Duration::from_secs(if complete { 1230 } else { 120 }),
    })
}
pub fn show(
    client: &Client,
    project: &str,
    executor: &str,
    asset: &str,
    health: &Value,
) -> Result<Value, Error> {
    validate(project, executor, Some(asset))?;
    capability(health)?;
    let value = client.get(&format!(
        "/api/projects/{project}/datasets/{executor}/{asset}"
    ))?;
    serde_json::to_value(state(value, project, executor, asset)?).map_err(|_| Error::Response)
}
#[cfg(unix)]
pub fn upload(
    client: &Client,
    project: &str,
    executor: &str,
    path: &Path,
    health: &Value,
) -> Result<Value, Error> {
    validate(project, executor, None)?;
    capability(health)?;
    let path = std::fs::canonicalize(path)
        .map_err(|_| Error::Observation("cannot open dataset directory"))?;
    let directory = ml_exp_io::directory::Directory::open(&path, false)
        .map_err(|_| Error::Observation("cannot open safe dataset directory"))?;
    let mut archive = tempfile::tempfile()
        .map_err(|_| Error::Observation("cannot create local dataset packing file"))?;
    let (sha256, archive_bytes, files) = ml_exp_io::artifacts::pack_data(&directory, &mut archive)
        .map_err(|_| Error::Observation("cannot pack safe regular dataset files"))?;
    let definition = Definition {
        sha256,
        archive_bytes,
        files,
    };
    definition
        .validate()
        .map_err(|_| Error::Observation("dataset exceeds metadata or multipart bounds"))?;
    let asset = definition.asset_id();
    eprintln!(
        "Dataset {asset}: {archive_bytes} bytes; resume by repeating this upload with the same directory."
    );
    let base = format!("/api/projects/{project}/datasets/{executor}");
    let transfer = (|| {
        let value = send(
            client,
            base.clone(),
            reqwest::Method::POST,
            serde_json::to_vec(&definition).map_err(|_| Error::Response)?,
            false,
        )?;
        let mut current = state(value, project, executor, &asset)?;
        if current.definition != definition {
            return Err(Error::Response);
        }
        if current.ready.is_none() {
            for number in 0..definition.part_count().map_err(|_| Error::Response)? {
                let count = definition.part_size(number).map_err(|_| Error::Response)?;
                archive
                    .seek(SeekFrom::Start(number * datasets::PART_BYTES))
                    .map_err(|_| Error::Response)?;
                let mut part = vec![0u8; count as usize];
                archive.read_exact(&mut part).map_err(|_| Error::Response)?;
                let sha256 = format!("{:x}", Sha256::digest(&part));
                let record = PartRecord {
                    sha256: sha256.clone(),
                    bytes: count,
                };
                if let Some(old) = current.parts.get(&number) {
                    if old != &record {
                        return Err(Error::Response);
                    }
                    continue;
                }
                eprintln!(
                    "Uploading part {}/{} ({} bytes)",
                    number + 1,
                    definition.part_count().map_err(|_| Error::Response)?,
                    count
                );
                let value = send(
                    client,
                    format!("{base}/{asset}/parts/{number}/{sha256}"),
                    reqwest::Method::PUT,
                    part,
                    false,
                )?;
                let next = state(value, project, executor, &asset)?;
                current
                    .validate_successor(&next)
                    .map_err(|_| Error::Response)?;
                if next.parts.get(&number) != Some(&record) {
                    return Err(Error::Response);
                }
                current = next;
            }
        }
        let value = send(
            client,
            format!("{base}/{asset}/complete"),
            reqwest::Method::POST,
            Vec::new(),
            true,
        )?;
        let ready = state(value, project, executor, &asset)?;
        current
            .validate_successor(&ready)
            .map_err(|_| Error::Response)?;
        if ready.ready.is_none() {
            return Err(Error::Response);
        }
        serde_json::to_value(ready).map_err(|_| Error::Response)
    })();
    transfer.map_err(|source| Error::DatasetOutcomeUnknown {
        asset,
        source: Box::new(source),
    })
}
#[cfg(not(unix))]
pub fn upload(_: &Client, _: &str, _: &str, _: &Path, _: &Value) -> Result<Value, Error> {
    Err(Error::Observation(
        "native dataset packing currently requires Unix",
    ))
}
