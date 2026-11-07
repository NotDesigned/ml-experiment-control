//! `ml-exp run rebuild-wandb`: publish one finished Attempt again to a new W&B
//! Run without the metric names of one exclusion list. One POST and no automatic
//! retry; repeating the same list returns the same rebuild and starts nothing,
//! so a lost response is safe to repeat. No W&B Run is ever deleted.
use crate::{
    Client, Error,
    observation::{self, Selection},
};
use ml_exp_core::records::REBUILD_MAX_BYTES;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

pub fn rebuild(
    client: &Client,
    saved: Option<&Value>,
    selection: Selection<'_>,
    exclude_names: &Path,
    confirmation: &str,
) -> Result<Value, Error> {
    let mut bytes = Vec::new();
    std::fs::File::open(exclude_names)
        .and_then(|file| {
            file.take(REBUILD_MAX_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
        })
        .map_err(|_| Error::Observation("cannot read the exclusion list; nothing was sent"))?;
    if bytes.len() > REBUILD_MAX_BYTES {
        return Err(Error::Observation(
            "an exclusion list holds at most 5000 names and 256 KiB",
        ));
    }
    // The file is sent verbatim, so its SHA256 is the one the server keeps.
    let names = String::from_utf8(bytes).map_err(|_| {
        Error::Observation("the exclusion list must be UTF-8 text; nothing was sent")
    })?;
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
        "REBUILD {}/{}/{attempt}",
        reference.project, reference.run_id
    );
    if confirmation != expected {
        return Err(Error::Submit(format!(
            "confirm with --confirmation '{expected}'; nothing was sent"
        )));
    }
    let path = format!(
        "/api/runs/{project}/{run}/attempts/{}/wandb-rebuilds",
        observation::segment(&attempt)?
    );
    let sha256 = format!("{:x}", Sha256::digest(names.as_bytes()));
    client.post(
        &path,
        &json!({"sha256":sha256,"names":names,"confirmation":confirmation}),
    )
}
