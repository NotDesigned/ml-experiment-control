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
fn state(mut value: Value, project: &str, executor: &str, asset: &str) -> Result<State, Error> {
    if let Some(m) = value.as_object_mut() {
        for field in [
            "archive_verified",
            "materialization",
            "data_image",
            "copy_job",
            "diagnostic",
            "import_source",
        ] {
            m.remove(field);
        }
    }
    let state: State = serde_json::from_value(value).map_err(|_| Error::Response)?;
    state.validate().map_err(|_| Error::Response)?;
    if state.project != project || state.target.backend_id != executor || state.asset_id != asset {
        return Err(Error::Response);
    }
    Ok(state)
}
pub fn import_oss(
    client: &Client,
    project: &str,
    executor: &str,
    source: datasets::oss::Import,
    health: &Value,
) -> Result<Value, Error> {
    validate(project, executor, None)?;
    source.validate().map_err(|_| Error::Path)?;
    capability(health)?;
    if !health["capabilities"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v == "native-oss-dataset-import.v1"))
    {
        return Err(Error::Observation(
            "server has not enabled OSS dataset import",
        ));
    }
    let asset = source.asset_id();
    let outcome = (|| {
        let value = send(
            client,
            format!("/api/projects/{project}/datasets/{executor}/import-oss"),
            reqwest::Method::POST,
            serde_json::to_vec(&source).map_err(|_| Error::Response)?,
            true,
        )?;
        let saved = state(value.clone(), project, executor, &asset)?;
        source
            .matches(&saved.definition)
            .map_err(|_| Error::Response)?;
        if value["archive_verified"] != true {
            return Err(Error::Response);
        }
        Ok(value)
    })();
    outcome.map_err(|source| Error::DatasetOutcomeUnknown {
        asset,
        source: Box::new(source),
    })
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
    state(value.clone(), project, executor, asset)?;
    Ok(value)
}
pub fn list(
    client: &Client,
    project: &str,
    executor: &str,
    health: &Value,
) -> Result<Value, Error> {
    validate(project, executor, None)?;
    capability(health)?;
    let value = client.get(&format!("/api/projects/{project}/datasets/{executor}"))?;
    if value["project"] != project || value["executor"] != executor {
        return Err(Error::Response);
    }
    let rows: Vec<datasets::Summary> =
        serde_json::from_value(value["datasets"].clone()).map_err(|_| Error::Response)?;
    if rows.len() > 1024 {
        return Err(Error::Response);
    }
    let mut previous: Option<&str> = None;
    for row in &rows {
        row.validate().map_err(|_| Error::Response)?;
        if previous.is_some_and(|s| s >= row.asset_id.as_str()) {
            return Err(Error::Response);
        }
        previous = Some(&row.asset_id);
    }
    Ok(value)
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
        let archived = value["archive_verified"] == true;
        let ready = state(value.clone(), project, executor, &asset)?;
        current
            .validate_successor(&ready)
            .map_err(|_| Error::Response)?;
        if ready.ready.is_none() && !archived {
            return Err(Error::Response);
        }
        Ok(value)
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
/// Immutable object parts go directly to the client. Verify the archive and
/// each member before no-clobber publication. Downloading datasets sends no ACK.
pub fn download(
    client: &Client,
    project: &str,
    executor: &str,
    asset: &str,
    output: &Path,
    health: &Value,
) -> Result<Value, Error> {
    let value = show(client, project, executor, asset, health)?;
    if value["archive_verified"] != true {
        return Err(Error::Observation(
            "dataset object download is unavailable for this backend",
        ));
    }
    let metadata = state(value, project, executor, asset)?;
    let output = if output.is_absolute() {
        output.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|_| Error::Response)?
            .join(output)
    };
    let parent = output.parent().ok_or(Error::Path)?;
    ml_exp_io::directory::Directory::open(parent, false).map_err(|_| Error::Path)?;
    if output.exists() {
        return Err(Error::Observation(
            "dataset output already exists; choose a new output path",
        ));
    }
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|_| Error::Response)?;
    for part in 0..metadata
        .definition
        .part_count()
        .map_err(|_| Error::Response)?
    {
        let ticket = client.post(
            &format!("/api/projects/{project}/datasets/{executor}/{asset}/download-ticket"),
            &serde_json::json!({"part":part}),
        )?;
        let expected = metadata
            .definition
            .part_size(part)
            .map_err(|_| Error::Response)?;
        if ticket["part"] != part || ticket["bytes"] != expected {
            return Err(Error::Response);
        }
        client.download_object(
            ticket["url"].as_str().ok_or(Error::Response)?,
            expected,
            temporary.as_file_mut(),
        )?;
    }
    temporary
        .as_file_mut()
        .seek(SeekFrom::Start(0))
        .map_err(|_| Error::Response)?;
    ml_exp_io::artifacts::verify_data_stream(temporary.as_file_mut(), &metadata.definition)
        .map_err(|_| {
            Error::Observation("dataset archive or member hash differs; output was not published")
        })?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|_| Error::Response)?;
    temporary.persist_noclobber(&output).map_err(|_| {
        Error::Observation("verified dataset could not be published without replacing a file")
    })?;
    std::fs::File::open(parent)
        .and_then(|f| f.sync_all())
        .map_err(|_| Error::Observation("dataset saved; directory sync could not be confirmed"))?;
    Ok(
        serde_json::json!({"asset_id":asset,"sha256":metadata.definition.sha256,"bytes":metadata.definition.archive_bytes,"output":output,"verified":true,"acknowledged":false}),
    )
}
