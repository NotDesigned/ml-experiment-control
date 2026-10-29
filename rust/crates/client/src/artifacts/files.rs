//! File selection from verified result archives. This path makes no release
//! request: a partial local result is never a complete archive receipt.
use super::*;
use std::io::{self, Read};

pub fn list(client: &Client, action: &str, health: &Value) -> Result<Value, Error> {
    let value = observe(client, action, false, health)?;
    let (_, state) = validate(&plan(action, "artifacts", None)?, value)?;
    Ok(
        json!({"submission_id":action,"source":"verified_result_archive", "files":state.bundle.archive.files,
        "archive_sha256":state.bundle.archive.sha256,"download_available":state.available().map_err(|_| invalid())?,
        "acknowledgement_required":false,"scheduler_mutation_performed":false}),
    )
}

struct Parts<'a> {
    client: &'a Client,
    action: &'a str,
    bytes: u64,
    count: u64,
    number: u64,
    remaining: u64,
    response: Option<reqwest::blocking::Response>,
}
fn stream_error() -> io::Error {
    io::Error::other("result archive part is unavailable or changed")
}
impl Read for Parts<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            if let Some(response) = &mut self.response {
                let count = buffer.len().min(self.remaining.saturating_add(1) as usize);
                let n = response.read(&mut buffer[..count])?;
                if n as u64 > self.remaining {
                    return Err(stream_error());
                }
                if n > 0 {
                    self.remaining -= n as u64;
                    return Ok(n);
                }
                if self.remaining != 0 {
                    return Err(stream_error());
                }
                self.response = None;
            }
            if self.number >= self.count {
                return Ok(0);
            }
            let request = plan(
                self.action,
                "download-ticket",
                Some(json!({"part":self.number})),
            )
            .map_err(|_| stream_error())?;
            let ticket = self
                .client
                .send_artifact(&request)
                .map_err(|_| stream_error())?;
            let bytes = (self.bytes - self.number * 16 * 1024 * 1024).min(16 * 1024 * 1024);
            if ticket["part"] != self.number
                || ticket["bytes"] != bytes
                || ticket["headers"] != json!({})
            {
                return Err(stream_error());
            }
            self.response = Some(
                self.client
                    .object_reader(ticket["url"].as_str().ok_or_else(stream_error)?, bytes)
                    .map_err(|_| stream_error())?,
            );
            self.remaining = bytes;
            self.number += 1;
        }
    }
}
pub fn download(
    client: &Client,
    action: &str,
    path: &str,
    output: &Path,
    health: &Value,
) -> Result<Value, Error> {
    validate_submission_id(action).map_err(|_| invalid())?;
    if !artifacts::publication::export_path(path) {
        return Err(invalid());
    }
    let output = destination(output)?;
    let value = observe(client, action, false, health)?;
    let (_, state) = validate(&plan(action, "artifacts", None)?, value.clone())?;
    let receipt = &state.bundle.archive;
    let record = receipt
        .files
        .iter()
        .find(|f| f.path == path)
        .ok_or(Error::Observation(
            "requested file is absent from the verified archive inventory",
        ))?;
    let reused = match std::fs::symlink_metadata(&output) {
        Ok(meta) => {
            if !meta.is_file() || meta.file_type().is_symlink() {
                return Err(invalid());
            }
            let (sha, size) =
                ml_exp_io::artifacts::hash(&mut File::open(&output).map_err(|_| invalid())?)
                    .map_err(|_| invalid())?;
            if sha != record.sha256 || size != record.bytes {
                return Err(Error::Observation(
                    "existing local file differs; it was not overwritten",
                ));
            }
            true
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => false,
        Err(_) => return Err(invalid()),
    };
    if !reused {
        if !state.available().map_err(|_| invalid())? {
            return Err(Error::Observation(
                "result archive is unavailable; original backend-file recovery is a separate operation",
            ));
        }
        let mut temporary = tempfile::NamedTempFile::new_in(output.parent().ok_or_else(invalid)?)
            .map_err(|_| invalid())?;
        if value["download"]["kind"] == "oss-parts" {
            let count = receipt.bytes.div_ceil(16 * 1024 * 1024);
            if count == 0
                || count > 65536
                || value["download"]["part_bytes"] != 16 * 1024 * 1024u64
                || value["download"]["part_count"] != count
            {
                return Err(Error::Response);
            }
            let mut parts = Parts {
                client,
                action,
                bytes: receipt.bytes,
                count,
                number: 0,
                remaining: 0,
                response: None,
            };
            ml_exp_io::artifacts::copy_result_file(
                &mut parts,
                receipt,
                path,
                temporary.as_file_mut(),
            )
            .map_err(|_| invalid())?;
        } else {
            let mut response = client.artifact_reader(&DownloadPlan {
                path: format!("/api/submissions/{action}/artifacts/{}", receipt.sha256),
                bytes: receipt.bytes,
            })?;
            ml_exp_io::artifacts::copy_result_file(
                &mut response,
                receipt,
                path,
                temporary.as_file_mut(),
            )
            .map_err(|_| invalid())?;
        }
        temporary.as_file().sync_all().map_err(|_| invalid())?;
        temporary
            .persist_noclobber(&output)
            .map_err(|_| Error::Observation("destination changed; no file was overwritten"))?;
        File::open(output.parent().ok_or_else(invalid)?)
            .and_then(|f| f.sync_all())
            .map_err(|_| invalid())?;
    }
    Ok(
        json!({"submission_id":action,"source":"verified_result_archive","file":record,"local_file":output,
        "local_file_verified":true,"reused_existing_file":reused,"acknowledgement_sent":false,"scheduler_mutation_performed":false}),
    )
}
