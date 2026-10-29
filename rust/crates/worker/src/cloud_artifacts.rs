//! Pack on persistent backend storage and send immutable 16MiB objects directly
//! to OSS. Only metadata and short-lived scoped tickets pass through the API.
use crate::{
    cloud,
    managed::{self, Error, Result},
};
use ml_exp_core::{
    artifacts::{self, publication::Bundle},
    cloud::{
        exchange::{ArtifactOperation, ArtifactRequest},
        launch::Package,
    },
    slurm::helper,
};
use ml_exp_io::directory::Directory;
use serde_json::{Value, json};
use std::{
    ffi::OsStr,
    io::{Read, Seek, SeekFrom},
    time::Duration,
};
const PART_BYTES: u64 = 16 * 1024 * 1024;
fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(3600))
        .build()
        .map_err(|_| Error::Configuration)
}
fn send(
    http: &reqwest::blocking::Client,
    package: &Package,
    operation: ArtifactOperation,
) -> Result<Value> {
    let channel = package.channel.as_ref().ok_or(Error::Configuration)?;
    let request = ArtifactRequest {
        package_sha256: cloud::package_sha(package)?,
        operation,
    };
    let body = serde_json::to_vec(&request).map_err(|_| Error::Configuration)?;
    if body.len() > 8 * 1024 * 1024 {
        return Err(Error::Binding);
    }
    let response = http
        .post(format!("{}/artifacts", channel.url))
        .bearer_auth(&channel.token)
        .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .map_err(|_| Error::Uncertain)?;
    if response.status() != 200 {
        return Err(Error::Uncertain);
    }
    let mut bytes = Vec::new();
    response
        .take(65537)
        .read_to_end(&mut bytes)
        .map_err(|_| Error::Uncertain)?;
    if bytes.len() > 65536 {
        return Err(Error::Binding);
    }
    Ok(Value::Object(
        ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Binding)?,
    ))
}
pub fn publish(package: &Package) -> Result<()> {
    if !package.result_upload {
        return Ok(());
    }
    package.validate().map_err(|_| Error::Configuration)?;
    let observed = managed::sample(&package.config)?;
    if observed.phase != ml_exp_core::managed::evidence::Phase::Finished
        || observed.receipt.is_none()
    {
        return Err(Error::Binding);
    }
    let attempt =
        Directory::open(&package.config.attempt_root, false).map_err(|_| Error::Storage)?;
    let dir = if attempt
        .entry_exists(OsStr::new(".cloud-artifacts"))
        .map_err(|_| Error::Storage)?
    {
        attempt.child(OsStr::new(".cloud-artifacts"), false)
    } else {
        attempt.create_new_child(OsStr::new(".cloud-artifacts"))
    }
    .map_err(|_| Error::Storage)?;
    let package_sha = cloud::package_sha(package)?;
    if dir
        .entry_exists(OsStr::new("owner"))
        .map_err(|_| Error::Storage)?
    {
        if dir
            .read_regular(OsStr::new("owner"), 128)
            .map_err(|_| Error::Storage)?
            != package_sha.as_bytes()
        {
            return Err(Error::Binding);
        }
    } else {
        if !dir
            .entry_names_limited(1)
            .map_err(|_| Error::Storage)?
            .is_empty()
        {
            return Err(Error::Binding);
        }
        dir.publish_new(OsStr::new("owner"), package_sha.as_bytes())
            .map_err(|_| Error::Storage)?;
    }
    let lock = dir
        .open_lock(OsStr::new(".lock"))
        .map_err(|_| Error::Storage)?;
    fs2::FileExt::try_lock_exclusive(&lock).map_err(|_| Error::Uncertain)?;
    let bundle: Bundle = if dir
        .entry_exists(OsStr::new("bundle.json"))
        .map_err(|_| Error::Storage)?
    {
        serde_json::from_slice(
            &dir.read_regular(OsStr::new("bundle.json"), 8 * 1024 * 1024)
                .map_err(|_| Error::Storage)?,
        )
        .map_err(|_| Error::Binding)?
    } else {
        // A previous interrupted unpublished pack may be removed under the
        // exact package owner and exclusive lock. Published archives never are.
        if dir
            .entry_exists(OsStr::new("archive.tar"))
            .map_err(|_| Error::Storage)?
        {
            let file = dir
                .open_regular_read(OsStr::new("archive.tar"))
                .map_err(|_| Error::Storage)?;
            dir.remove_staged_regular(OsStr::new("archive.tar"), &file)
                .map_err(|_| Error::Storage)?;
        }
        let outputs =
            Directory::open(&package.config.output_path(), false).map_err(|_| Error::Storage)?;
        if dir.capacity().map_err(|_| Error::Storage)?.available_bytes
            < ml_exp_io::artifacts::estimate(&outputs).map_err(|_| Error::Storage)?
        {
            return Err(Error::Storage);
        }
        let mut file = dir
            .create_new_regular(OsStr::new("archive.tar"))
            .map_err(|_| Error::Storage)?;
        let (sha256, bytes, files) = ml_exp_io::artifacts::pack(
            &outputs,
            &mut file,
            &package.artifacts.clone().unwrap_or_default(),
        )
        .map_err(|_| Error::Storage)?;
        file.sync_all().map_err(|_| Error::Storage)?;
        dir.seal_regular(OsStr::new("archive.tar"), &file, false)
            .map_err(|_| Error::Storage)?;
        let id = &package.manifest.identity;
        let bundle = Bundle {
            contract: artifacts::publication::CONTRACT.into(),
            intent_sha256: package_sha.clone(),
            config_sha256: helper::sha256_bytes(
                &serde_json::to_vec(&package.config).map_err(|_| Error::Binding)?,
            ),
            archive: artifacts::ArchiveReceipt {
                project: id.project.clone(),
                run_id: id.run_id.clone(),
                attempt_id: id.attempt_id.clone(),
                object_key: format!(
                    "{}/{}/{}/{}.tar",
                    id.project, id.run_id, id.attempt_id, sha256
                ),
                sha256,
                bytes,
                files,
            },
        };
        bundle.validate().map_err(|_| Error::Binding)?;
        dir.publish_new(
            OsStr::new("bundle.json"),
            &serde_json::to_vec(&bundle).map_err(|_| Error::Binding)?,
        )
        .map_err(|_| Error::Storage)?;
        bundle
    };
    bundle.validate().map_err(|_| Error::Binding)?;
    if bundle.intent_sha256 != package_sha {
        return Err(Error::Binding);
    }
    let http = client()?;
    let prepared = send(
        &http,
        package,
        ArtifactOperation::Prepare {
            bundle: Box::new(bundle.clone()),
        },
    )?;
    let count = bundle.archive.bytes.div_ceil(PART_BYTES);
    if prepared["part_count"] != count || prepared["part_bytes"] != PART_BYTES || count > 65536 {
        return Err(Error::Binding);
    }
    if prepared["published"] != true {
        let mut file = dir
            .open_regular_read(OsStr::new("archive.tar"))
            .map_err(|_| Error::Storage)?;
        ml_exp_io::artifacts::verify(&mut file, &bundle.archive).map_err(|_| Error::Binding)?;
        let mut buffer = vec![0u8; PART_BYTES as usize];
        for part in 0..count {
            let ticket = send(&http, package, ArtifactOperation::Ticket { part })?;
            let bytes = (bundle.archive.bytes - part * PART_BYTES).min(PART_BYTES);
            if ticket["part"] != part
                || ticket["bytes"] != bytes
                || ticket["headers"] != json!({"x-oss-forbid-overwrite":"true"})
            {
                return Err(Error::Binding);
            }
            let url = ticket["url"].as_str().ok_or(Error::Binding)?;
            let uri = reqwest::Url::parse(url).map_err(|_| Error::Binding)?;
            if uri.scheme() != "https"
                || !uri.host_str().is_some_and(|h| h.ends_with(".aliyuncs.com"))
                || !uri.username().is_empty()
                || uri.password().is_some()
            {
                return Err(Error::Binding);
            }
            file.seek(SeekFrom::Start(part * PART_BYTES))
                .map_err(|_| Error::Storage)?;
            file.read_exact(&mut buffer[..bytes as usize])
                .map_err(|_| Error::Storage)?;
            let response = http
                .put(uri)
                .header("x-oss-forbid-overwrite", "true")
                .body(buffer[..bytes as usize].to_vec())
                .send()
                .map_err(|_| Error::Uncertain)?;
            // A duplicate immutable part is verified in the full server-side
            // read before publication; it is never overwritten on retry.
            if !matches!(response.status().as_u16(), 200 | 409) {
                return Err(Error::Uncertain);
            }
        }
        let complete = send(&http, package, ArtifactOperation::Complete)?;
        if complete["published"] != true
            || complete["sha256"] != bundle.archive.sha256
            || complete["bytes"] != bundle.archive.bytes
        {
            return Err(Error::Binding);
        }
    }
    // Remote durable receipt is now known. Keep original outputs/checkpoints;
    // remove only this owned generated archive, with metadata for lost replies.
    if dir
        .entry_exists(OsStr::new("archive.tar"))
        .map_err(|_| Error::Storage)?
    {
        let file = dir
            .open_regular_read(OsStr::new("archive.tar"))
            .map_err(|_| Error::Storage)?;
        dir.remove_staged_regular(OsStr::new("archive.tar"), &file)
            .map_err(|_| Error::Storage)?;
    }
    dir.verify_private_parent().map_err(|_| Error::Storage)?;
    Ok(())
}
