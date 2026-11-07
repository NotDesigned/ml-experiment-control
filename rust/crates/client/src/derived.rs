//! `ml-exp run append-derived`: one complete post-hoc JSONL file for an Attempt
//! whose Run has ended. One POST and no automatic retry; repeating the same file
//! returns the same batch and adds nothing, so a lost response is safe to repeat.
use crate::{
    Client, Error,
    observation::{self, Selection},
};
use ml_exp_core::records::DERIVED_MAX_BYTES;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

pub fn append(
    client: &Client,
    saved: Option<&Value>,
    selection: Selection<'_>,
    metrics: &Path,
    confirmation: &str,
) -> Result<Value, Error> {
    let mut bytes = Vec::new();
    std::fs::File::open(metrics)
        .and_then(|file| {
            file.take(DERIVED_MAX_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
        })
        .map_err(|_| {
            Error::Observation("cannot read the derived metrics file; nothing was sent")
        })?;
    if bytes.len() > DERIVED_MAX_BYTES {
        return Err(Error::Observation(
            "a derived batch holds at most 2 MiB and 5000 records; split the file",
        ));
    }
    let records = String::from_utf8(bytes)
        .map_err(|_| Error::Observation("derived metrics must be UTF-8 JSONL; nothing was sent"))?;
    let reference = observation::resolve(client, saved, selection)?;
    let project = observation::segment(&reference.project)?;
    let run = observation::segment(&reference.run_id)?;
    let attempt = match reference.attempt_id {
        Some(attempt) => attempt,
        None => client.get(&format!("/api/runs/{project}/{run}"))?["current_attempt_id"]
            .as_str()
            .ok_or(Error::Observation("this Run has no current Attempt"))?
            .to_owned(),
    };
    let expected = format!(
        "APPEND {}/{}/{attempt}",
        reference.project, reference.run_id
    );
    if confirmation != expected {
        return Err(Error::Submit(format!(
            "confirm with --confirmation '{expected}'; nothing was sent"
        )));
    }
    let path = format!(
        "/api/runs/{project}/{run}/attempts/{}/derived-metrics",
        observation::segment(&attempt)?
    );
    let sha256 = format!("{:x}", Sha256::digest(records.as_bytes()));
    client.post(
        &path,
        &json!({"sha256":sha256,"records":records,"confirmation":confirmation}),
    )
}
