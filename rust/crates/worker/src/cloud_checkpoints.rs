//! Object-store copies of this Attempt's registered checkpoints, and the delivery of
//! the generation a restoring Attempt resumes from.
//!
//! Bytes go straight between backend storage and the object store with short-lived
//! tickets on the Attempt's own capability; this worker holds no other credential. An
//! upload runs beside training on its own thread, one part at a time under the rate the
//! server names, and the server reads every object back before it records a copy. A
//! failed upload is tried again on the next pass and never touches training. A restore
//! gives the Attempt its own copy before training starts: a hard link to the original
//! Attempt's sealed file when that file still has its registered bytes, otherwise the
//! verified object-store copy. Any other outcome stops the start; nothing is guessed.
use crate::{
    cloud,
    managed::{Error, Result},
};
use ml_exp_core::{
    checkpoints::{
        self, FileRecord,
        archive::{self, Operation, Request},
    },
    cloud::launch::Package,
    multipart::PART_BYTES,
    retry,
};
use ml_exp_io::directory::Directory;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const RESPONSE_LIMIT: u64 = 4 * 1024 * 1024;
/// Between two looks for newly registered checkpoints while training runs.
const PASS_SECONDS: u64 = 60;
const PARTIAL_SUFFIX: &str = ".ml-exp-partial";

fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .connect_timeout(Duration::from_secs(5))
        // Completion reads a whole copy back on the server.
        .timeout(Duration::from_secs(4 * 3600))
        .build()
        .map_err(|_| Error::Configuration)
}
fn restore_failed(code: &'static str) -> Error {
    Error::CheckpointRestore { code }
}

/// One request on the Attempt's capability. Anything that may pass is tried again;
/// a refusal is final for this call.
fn send(
    http: &reqwest::blocking::Client,
    package: &Package,
    operation: Operation,
) -> Result<Value> {
    let body = serde_json::to_vec(&Request {
        package_sha256: cloud::package_sha(package)?,
        operation,
    })
    .map_err(|_| Error::Configuration)?;
    let mut count = 0;
    loop {
        count += 1;
        let result = (|| -> std::result::Result<Vec<u8>, Option<String>> {
            let response = http
                .post(format!("{}/checkpoints", package.channel.url))
                .bearer_auth(&package.channel.token)
                .header("content-type", "application/json")
                .body(body.clone())
                .send()
                .map_err(|error| Some(error.without_url().to_string()))?;
            let status = response.status().as_u16();
            if status != 200 {
                if retry::may_pass(status) {
                    return Err(Some(format!("HTTP {status}")));
                }
                eprintln!("ML_EXPD_CHECKPOINT_ARCHIVE=REFUSED HTTP {status}");
                return Err(None);
            }
            let mut bytes = Vec::new();
            response
                .take(RESPONSE_LIMIT + 1)
                .read_to_end(&mut bytes)
                .map_err(|error| Some(error.to_string()))?;
            Ok(bytes)
        })();
        match result {
            Ok(bytes) if bytes.len() as u64 > RESPONSE_LIMIT => return Err(Error::Binding),
            Ok(bytes) => {
                return Ok(Value::Object(
                    ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Binding)?,
                ));
            }
            Err(None) => return Err(Error::Uncertain),
            Err(Some(failure)) => {
                let failure = experiment_redact::redact_line(&failure);
                if count >= retry::ATTEMPTS {
                    eprintln!("ML_EXPD_CHECKPOINT_ARCHIVE=GAVE_UP {failure}");
                    return Err(Error::Uncertain);
                }
                std::thread::sleep(retry::delay(count));
            }
        }
    }
}

/// An object-store URL, never another origin with the platform's token.
fn object_url(ticket: &Value) -> Result<reqwest::Url> {
    let uri = reqwest::Url::parse(ticket["url"].as_str().ok_or(Error::Binding)?)
        .map_err(|_| Error::Binding)?;
    if uri.scheme() != "https"
        || uri.port_or_known_default() != Some(443)
        || !uri.host_str().is_some_and(|h| h.ends_with(".aliyuncs.com"))
        || !uri.username().is_empty()
        || uri.password().is_some()
        || uri.fragment().is_some()
    {
        return Err(Error::Binding);
    }
    Ok(uri)
}

/// The directory that holds `path` below `root`, and the file's name.
fn parent(root: &Directory, path: &str, create: bool) -> Result<(Directory, String)> {
    let mut parts: Vec<_> = path.split('/').collect();
    let name = parts.pop().ok_or(Error::Binding)?.to_owned();
    let mut directory = root.clone();
    for part in parts {
        directory = directory
            .child(OsStr::new(part), create)
            .map_err(crate::managed::storage_failure!())?;
        if create {
            // A tree this restore sealed before it was interrupted is its own to finish.
            directory
                .source_mode(false)
                .map_err(crate::managed::storage_failure!())?;
        }
    }
    Ok((directory, name))
}

fn open_file(root: &Directory, path: &str) -> Result<File> {
    let (directory, name) = parent(root, path, false)?;
    directory
        .open_regular_read(OsStr::new(&name))
        .map_err(crate::managed::storage_failure!())
}

/// Exactly the registered size and SHA256.
fn holds(file: &mut File, record: &FileRecord) -> Result<bool> {
    if file
        .metadata()
        .map_err(crate::managed::storage_failure!())?
        .len()
        != record.bytes
    {
        return Ok(false);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(crate::managed::storage_failure!())?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut total = 0u64;
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(crate::managed::storage_failure!())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        total += n as u64;
    }
    Ok(total == record.bytes && format!("{:x}", hash.finalize()) == record.sha256)
}

/// Waits so that `sent` bytes take at least `sent / rate` seconds since `started`.
fn pace(started: Instant, sent: u64, rate: u64) {
    if rate == 0 {
        return;
    }
    let due = Duration::from_secs_f64(sent as f64 / rate as f64);
    if let Some(wait) = due.checked_sub(started.elapsed()) {
        std::thread::sleep(wait);
    }
}

fn put(http: &reqwest::blocking::Client, uri: &reqwest::Url, bytes: &[u8]) -> Result<bool> {
    let mut count = 0;
    loop {
        count += 1;
        let status = http
            .put(uri.clone())
            .header("x-oss-forbid-overwrite", "true")
            .body(bytes.to_vec())
            .send()
            .map(|response| response.status().as_u16());
        match status {
            // An object that is already there is read back by the server like any other.
            Ok(200 | 409) => return Ok(true),
            // A ticket that expired while earlier parts were sent: ask for new ones.
            Ok(403) => return Ok(false),
            Ok(status) if !retry::may_pass(status) => {
                eprintln!("ML_EXPD_CHECKPOINT_ARCHIVE=PART_REFUSED HTTP {status}");
                return Err(Error::Uncertain);
            }
            _ if count >= retry::ATTEMPTS => return Err(Error::Uncertain),
            _ => std::thread::sleep(retry::delay(count)),
        }
    }
}

/// Send every part of one registered checkpoint, then have the server read it back.
fn upload(
    http: &reqwest::blocking::Client,
    package: &Package,
    state_root: &Directory,
    checkpoint_id: &str,
    rate: u64,
    cancelled: &AtomicBool,
) -> Result<bool> {
    let context = package
        .checkpoint_context
        .as_ref()
        .ok_or(Error::Configuration)?;
    let document = send(
        http,
        package,
        Operation::Document {
            checkpoint_id: checkpoint_id.into(),
        },
    )?;
    let (_, manifest) =
        checkpoints::validate_document(&document, Some(context), Some(checkpoint_id))
            .map_err(|_| Error::Binding)?;
    let parts = archive::parts(&manifest.files).map_err(|_| Error::Binding)?;
    let mut buffer = vec![0u8; PART_BYTES as usize];
    let mut open: Option<(usize, File)> = None;
    let started = Instant::now();
    let mut sent = 0u64;
    let mut start = 0u64;
    let mut expired = 0;
    while start < parts.len() as u64 {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let reply = send(
            http,
            package,
            Operation::Upload {
                checkpoint_id: checkpoint_id.into(),
                start,
            },
        )?;
        if reply["archived"] == true {
            return Ok(true);
        }
        let tickets = reply["tickets"].as_array().ok_or(Error::Binding)?;
        if reply["parts"] != parts.len() as u64 || tickets.is_empty() {
            return Err(Error::Binding);
        }
        for ticket in tickets {
            let part = parts.get(start as usize).ok_or(Error::Binding)?;
            let record = &manifest.files[part.file];
            if ticket["index"] != part.index
                || ticket["file"] != record.path.as_str()
                || ticket["part"] != part.part
                || ticket["offset"] != part.offset
                || ticket["bytes"] != part.bytes
                || ticket["headers"] != json!({"x-oss-forbid-overwrite":"true"})
            {
                return Err(Error::Binding);
            }
            let uri = object_url(ticket)?;
            if open.as_ref().is_none_or(|(file, _)| *file != part.file) {
                let file = open_file(state_root, &record.path)?;
                ml_exp_io::directory::require_sealed(&file)
                    .map_err(crate::managed::storage_failure!())?;
                if file
                    .metadata()
                    .map_err(crate::managed::storage_failure!())?
                    .len()
                    != record.bytes
                {
                    return Err(Error::Binding);
                }
                open = Some((part.file, file));
            }
            let (_, file) = open.as_mut().ok_or(Error::Binding)?;
            let chunk = &mut buffer[..part.bytes as usize];
            file.seek(SeekFrom::Start(part.offset))
                .map_err(crate::managed::storage_failure!())?;
            file.read_exact(chunk)
                .map_err(crate::managed::storage_failure!())?;
            if !put(http, &uri, chunk)? {
                expired += 1;
                if expired > 3 {
                    return Err(Error::Uncertain);
                }
                break;
            }
            expired = 0;
            sent += part.bytes;
            start += 1;
            pace(started, sent, rate);
        }
    }
    let done = send(
        http,
        package,
        Operation::Complete {
            checkpoint_id: checkpoint_id.into(),
        },
    )?;
    Ok(done["archived"] == true)
}

/// One pass over this Attempt's registered checkpoints, newest first. True when every
/// one has a verified copy. `given_up` keeps the ones whose files are not all here, so
/// a later pass in the same process does not read them again.
fn pass(
    package: &Package,
    cancelled: &AtomicBool,
    given_up: &mut BTreeSet<String>,
) -> Result<bool> {
    let Some(context) = &package.checkpoint_context else {
        return Ok(true);
    };
    if !package.checkpoint_archive {
        return Ok(true);
    }
    let http = client()?;
    let listed = send(&http, package, Operation::List)?;
    let rate = listed["max_bytes_per_second"].as_u64().unwrap_or(0);
    let mut pending = listed["checkpoints"]
        .as_array()
        .ok_or(Error::Binding)?
        .iter()
        .filter(|row| row["archived"] != true)
        .map(|row| {
            Ok((
                row["step"].as_u64().ok_or(Error::Binding)?,
                row["checkpoint_id"]
                    .as_str()
                    .ok_or(Error::Binding)?
                    .to_owned(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if pending.is_empty() {
        return Ok(true);
    }
    pending.sort();
    let state_root = Directory::open(Path::new(&context.state_root), false)
        .map_err(crate::managed::storage_failure!())?;
    let mut complete = true;
    for (step, id) in pending.into_iter().rev() {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(false);
        }
        if given_up.contains(&id) {
            complete = false;
            continue;
        }
        match upload(&http, package, &state_root, &id, rate, cancelled) {
            Ok(true) => eprintln!("ML_EXPD_CHECKPOINT_ARCHIVE=VERIFIED step={step} {id}"),
            Ok(false) => complete = false,
            Err(error) => {
                complete = false;
                if matches!(error, Error::Storage { .. } | Error::Binding) {
                    given_up.insert(id.clone());
                }
                eprintln!("ML_EXPD_CHECKPOINT_ARCHIVE=DEFERRED step={step} {id}: {error}");
            }
        }
    }
    Ok(complete)
}

/// Copy every registered checkpoint that has no verified copy yet. True when all have one.
pub fn archive(package: &Package, cancelled: &AtomicBool) -> Result<bool> {
    pass(package, cancelled, &mut BTreeSet::new())
}

/// The copy beside training: a pass a minute until `stop`. A pass under way when
/// training ends is finished, so the last one after training has little left to do.
pub fn beside_training(package: &Package, stop: &AtomicBool, cancelled: &AtomicBool) {
    let mut given_up = BTreeSet::new();
    while !stop.load(Ordering::Relaxed) && !cancelled.load(Ordering::Relaxed) {
        if let Err(error) = pass(package, cancelled, &mut given_up) {
            eprintln!("ML_EXPD_CHECKPOINT_ARCHIVE=DEFERRED {error}");
        }
        for _ in 0..PASS_SECONDS {
            if stop.load(Ordering::Relaxed) || cancelled.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

/// Write one file of the restored generation from its verified object-store copy.
fn download(
    http: &reqwest::blocking::Client,
    package: &Package,
    directory: &Directory,
    name: &str,
    record: &FileRecord,
    parts: &[archive::Part],
    cancelled: &AtomicBool,
) -> Result<()> {
    let partial = format!("{name}{PARTIAL_SUFFIX}");
    let mut file = directory
        .create_new_regular(OsStr::new(&partial))
        .map_err(crate::managed::storage_failure!())?;
    let mut hash = Sha256::new();
    let mut written = 0u64;
    let mut index = 0;
    while index < parts.len() {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        // No verified copy, or none that can be read now: the start stops here.
        let reply = send(
            http,
            package,
            Operation::Download {
                start: parts[index].index,
            },
        )
        .map_err(|_| restore_failed("ARCHIVE_UNAVAILABLE"))?;
        let tickets = reply["tickets"].as_array().ok_or(Error::Binding)?;
        let before = index;
        for ticket in tickets {
            let Some(part) = parts.get(index) else {
                break;
            };
            if ticket["index"] != part.index
                || ticket["file"] != record.path.as_str()
                || ticket["offset"] != part.offset
                || ticket["bytes"] != part.bytes
            {
                break;
            }
            let uri = object_url(ticket)?;
            let mut count = 0;
            let bytes = loop {
                count += 1;
                let result = http
                    .get(uri.clone())
                    .send()
                    .map_err(|_| ())
                    .and_then(|response| {
                        let status = response.status().as_u16();
                        let length = response.content_length();
                        let mut bytes = Vec::new();
                        response
                            .take(part.bytes + 1)
                            .read_to_end(&mut bytes)
                            .map_err(|_| ())?;
                        Ok((status, length, bytes))
                    });
                match result {
                    Ok((200, Some(length), bytes))
                        if length == part.bytes && bytes.len() as u64 == part.bytes =>
                    {
                        break bytes;
                    }
                    Ok((status, ..)) if status != 200 && !retry::may_pass(status) => {
                        return Err(restore_failed("OBJECT_UNAVAILABLE"));
                    }
                    _ if count >= retry::ATTEMPTS => {
                        return Err(restore_failed("OBJECT_READ_FAILED"));
                    }
                    _ => std::thread::sleep(retry::delay(count)),
                }
            };
            hash.update(&bytes);
            file.write_all(&bytes)
                .map_err(crate::managed::storage_failure!())?;
            written += part.bytes;
            index += 1;
        }
        if index == before {
            return Err(restore_failed("TICKETS_INVALID"));
        }
    }
    if written != record.bytes || format!("{:x}", hash.finalize()) != record.sha256 {
        drop(file);
        let _ = directory.remove_staged_regular(OsStr::new(&partial));
        return Err(restore_failed("OBJECT_MISMATCH"));
    }
    ml_exp_io::directory::seal_regular(&file, false).map_err(crate::managed::storage_failure!())?;
    drop(file);
    directory
        .publish_regular_as(OsStr::new(&partial), OsStr::new(name))
        .map_err(crate::managed::storage_failure!())
}

fn seal_tree(directory: &Directory, depth: usize) -> Result<()> {
    if depth > 256 {
        return Err(Error::Binding);
    }
    for name in directory
        .entry_names_limited(checkpoints::MAX_FILES)
        .map_err(crate::managed::storage_failure!())?
    {
        if let Ok(child) = directory.child(&name, false) {
            seal_tree(&child, depth + 1)?;
        }
    }
    directory
        .source_mode(true)
        .map_err(crate::managed::storage_failure!())
}

/// Give a restoring Attempt its own copy of the generation it resumes from, before
/// training. Nothing here starts training; a copy that cannot be proven stops the start.
pub fn restore(package: &Package, cancelled: &AtomicBool) -> Result<()> {
    let config = &package.config;
    let Some(document) = &config.checkpoint_restore else {
        return Ok(());
    };
    let (source, manifest) =
        checkpoints::validate_document(document, None, None).map_err(|_| Error::Binding)?;
    let plan = checkpoints::restore_plan(document, None).map_err(|_| Error::Binding)?;
    let root = archive::restore_root(&config.attempt_root, &plan.checkpoint_id)
        .map_err(|_| Error::Binding)?;
    let resume = package
        .manifest
        .environment
        .get("RESUME_DIR")
        .ok_or(Error::Binding)?;
    // An Attempt frozen to read the original generation in place has nothing to deliver.
    if archive::resume_state_root(&plan, &config.attempt_root, resume)
        .map_err(|_| Error::Binding)?
        != root.to_str().ok_or(Error::Binding)?
    {
        return Ok(());
    }
    let state = Directory::open(&config.attempt_root.join("state"), false)
        .map_err(crate::managed::storage_failure!())?;
    let restores = state
        .child(OsStr::new(archive::RESTORE_DIRECTORY), true)
        .map_err(crate::managed::storage_failure!())?;
    let target = restores
        .child(root.file_name().ok_or(Error::Binding)?, true)
        .map_err(crate::managed::storage_failure!())?;
    if ml_exp_io::checkpoints::verify_sealed(&target, &manifest).is_ok() {
        eprintln!("ML_EXPD_CHECKPOINT_RESTORE=READY {}", plan.checkpoint_id);
        return Ok(());
    }
    let original = Directory::open(Path::new(&source.state_root), false).ok();
    let parts = archive::parts(&manifest.files).map_err(|_| Error::Binding)?;
    let mut http = None;
    let (mut linked, mut downloaded) = (0, 0);
    for (index, record) in manifest.files.iter().enumerate() {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let (directory, name) = parent(&target, &record.path, true)?;
        let partial = format!("{name}{PARTIAL_SUFFIX}");
        if directory
            .entry_exists(OsStr::new(&partial))
            .map_err(crate::managed::storage_failure!())?
        {
            directory
                .remove_staged_regular(OsStr::new(&partial))
                .map_err(crate::managed::storage_failure!())?;
        }
        if directory
            .entry_exists(OsStr::new(&name))
            .map_err(crate::managed::storage_failure!())?
        {
            let mut file = directory
                .open_regular_read(OsStr::new(&name))
                .map_err(crate::managed::storage_failure!())?;
            if ml_exp_io::directory::require_sealed(&file).is_ok() && holds(&mut file, record)? {
                continue;
            }
            directory
                .remove_staged_regular(OsStr::new(&name))
                .map_err(crate::managed::storage_failure!())?;
        }
        // The original's sealed file, if it is still there with its registered bytes.
        if let Some(original) = &original {
            let found = parent(original, &record.path, false).and_then(|(from, leaf)| {
                directory
                    .link_regular_from(&from, OsStr::new(&leaf), OsStr::new(&name))
                    .map_err(crate::managed::storage_failure!())
            });
            if found.is_ok() {
                let mut file = directory
                    .open_regular_read(OsStr::new(&name))
                    .map_err(crate::managed::storage_failure!())?;
                if ml_exp_io::directory::require_sealed(&file).is_ok() && holds(&mut file, record)?
                {
                    linked += 1;
                    continue;
                }
                directory
                    .remove_staged_regular(OsStr::new(&name))
                    .map_err(crate::managed::storage_failure!())?;
            }
        }
        if !package.checkpoint_archive {
            return Err(restore_failed("ORIGINAL_UNAVAILABLE"));
        }
        if http.is_none() {
            http = Some(client()?);
        }
        let http = http.as_ref().ok_or(Error::Configuration)?;
        let own: Vec<_> = parts.iter().copied().filter(|p| p.file == index).collect();
        download(http, package, &directory, &name, record, &own, cancelled)?;
        downloaded += 1;
    }
    let generation = target
        .child(OsStr::new("checkpoints"), false)
        .and_then(|d| d.child(OsStr::new(&plan.generation), false))
        .map_err(crate::managed::storage_failure!())?;
    seal_tree(&generation, 0)?;
    eprintln!(
        "ML_EXPD_CHECKPOINT_RESTORE=DELIVERED {} linked={linked} downloaded={downloaded}",
        plan.checkpoint_id
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    fn context(root: &Path, attempt: &str) -> checkpoints::Context {
        checkpoints::Context {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: attempt.into(),
            source_id: format!("source.{}", "a".repeat(64)),
            image_id: format!("sha256:{}", "b".repeat(64)),
            storage_scope: format!("sha256:{}", "c".repeat(64)),
            state_root: root
                .join(format!("demo/train/{attempt}/state"))
                .to_str()
                .unwrap()
                .into(),
        }
    }

    #[test]
    fn a_restore_links_the_originals_sealed_file_and_then_finds_its_copy_ready() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let root = temp.path().join("runs");
        let original = root.join("demo/train/attempt-001/state/checkpoints/s1");
        std::fs::create_dir_all(&original).unwrap();
        std::fs::create_dir_all(root.join("demo/train/attempt-002/state")).unwrap();
        let bytes = b"weights".to_vec();
        std::fs::write(original.join("model.bin"), &bytes).unwrap();
        std::fs::set_permissions(
            original.join("model.bin"),
            std::fs::Permissions::from_mode(0o400),
        )
        .unwrap();
        let ready = json!({"step":1,"files":[{"path":"checkpoints/s1/model.bin","bytes":bytes.len(),
            "sha256":format!("{:x}", Sha256::digest(&bytes))}]});
        let document = checkpoints::register(
            &checkpoints::document(&context(&root, "attempt-001"), &ready).unwrap(),
            "2026-11-08T00:00:00Z",
        )
        .unwrap();
        let mut package = ml_exp_core::cloud::launch::compile(ml_exp_core::cloud::launch::Input {
            root: root.to_str().unwrap(),
            project: "demo",
            run: "train",
            attempt: "attempt-002",
            workdir: temp.path().to_str().unwrap(),
            argv: &["/bin/true".into()],
            timeout_seconds: 60,
            worker_sha256: &format!("sha256:{}", "a".repeat(64)),
            storage: ml_exp_core::cloud::launch::StorageLayout {
                mount: temp.path().into(),
                runs: root.clone(),
                datasets: temp.path().join("datasets"),
            },
            channel: ml_exp_core::cloud::exchange::Channel {
                url: "https://127.0.0.1:1/api/worker-exchange/demo/train/attempt-002".into(),
                token: "b".repeat(64),
            },
        })
        .unwrap();
        package.checkpoint_context = Some(context(&root, "attempt-002"));
        package.checkpoint_archive = true;
        package
            .attach(
                temp.path().join("datasets").to_str().unwrap(),
                Vec::new(),
                Some(document.clone()),
            )
            .unwrap();
        let hex = &document["checkpoint_id"].as_str().unwrap()["checkpoint.".len()..];
        let own = root.join(format!("demo/train/attempt-002/state/.restore/{hex}"));
        assert_eq!(
            package.manifest.environment["RESUME_DIR"],
            own.join("checkpoints/s1").to_str().unwrap()
        );
        restore(&package, &AtomicBool::new(false)).unwrap();
        let copy = own.join("checkpoints/s1/model.bin");
        assert_eq!(
            std::fs::metadata(&copy).unwrap().ino(),
            std::fs::metadata(original.join("model.bin")).unwrap().ino()
        );
        let (_, manifest) = checkpoints::validate_document(&document, None, None).unwrap();
        ml_exp_io::checkpoints::verify_sealed(&Directory::open(&own, false).unwrap(), &manifest)
            .unwrap();
        // Again, as after a restart: the sealed copy is found and nothing is fetched.
        restore(&package, &AtomicBool::new(false)).unwrap();
        // An Attempt frozen to read the original in place has nothing delivered.
        let mut in_place = package.clone();
        in_place.checkpoint_archive = false;
        in_place
            .attach(
                temp.path().join("datasets").to_str().unwrap(),
                Vec::new(),
                Some(document),
            )
            .unwrap();
        assert_eq!(
            in_place.manifest.environment["RESUME_DIR"],
            original.to_str().unwrap()
        );
        restore(&in_place, &AtomicBool::new(false)).unwrap();
    }
}
