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
    io::{Cursor, Read, Seek, SeekFrom},
    path::Path,
    sync::Arc,
    time::Duration,
};
const DIRECT_UPLOAD: &str = "native-dataset-direct-upload.v1";
/// One part's bytes, shared with the request sending them: a retry sends them again
/// without a second copy, so an upload holds about one part in memory.
#[derive(Clone)]
pub struct Part(Arc<Vec<u8>>);
impl AsRef<[u8]> for Part {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
impl Part {
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        Self(Arc::new(bytes))
    }
    pub(crate) fn reader(&self) -> Cursor<Self> {
        Cursor::new(self.clone())
    }
    pub(crate) fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn into_vec(self) -> Vec<u8> {
        Arc::unwrap_or_clone(self.0)
    }
}
/// What a direct upload does outside this process. Tests replace it, because a ticket
/// must be HTTPS and a local fake is not.
pub trait ObjectStore {
    /// One PUT. The status, or why there is none, in words that hold no ticket.
    fn put(&mut self, url: &url::Url, part: &Part) -> Result<u16, String>;
    fn wait(&mut self, delay: Duration);
}
struct Real<'a>(&'a Client);
impl ObjectStore for Real<'_> {
    fn put(&mut self, url: &url::Url, part: &Part) -> Result<u16, String> {
        self.0.put_object(url, part)
    }
    fn wait(&mut self, delay: Duration) {
        std::thread::sleep(delay);
    }
}
enum Attempt {
    Again(String),
    Stop(Error),
}
/// A call to the server: a passing failure is tried again, and the rest stop.
fn server_attempt(error: Error) -> Attempt {
    match error {
        Error::Transport => Attempt::Again("the server did not answer".into()),
        Error::Http(failure) if ml_exp_core::retry::may_pass(failure.status) => {
            Attempt::Again(format!("the server answered HTTP {}", failure.status))
        }
        other => Attempt::Stop(other),
    }
}
/// As the worker repeats a result part: bounded, with the same waits.
fn repeated<T>(
    store: &mut dyn ObjectStore,
    gave_up: &'static str,
    mut attempt: impl FnMut(&mut dyn ObjectStore) -> Result<T, Attempt>,
) -> Result<T, Error> {
    let mut count = 0;
    loop {
        count += 1;
        match attempt(store) {
            Ok(value) => return Ok(value),
            Err(Attempt::Stop(error)) => return Err(error),
            Err(Attempt::Again(failure)) => {
                if count >= ml_exp_core::retry::ATTEMPTS {
                    eprintln!("Giving up after {count} attempts: {failure}");
                    return Err(Error::Observation(gave_up));
                }
                eprintln!("Attempt {count} failed: {failure}; trying again");
                store.wait(ml_exp_core::retry::delay(count));
            }
        }
    }
}
/// The checks the worker makes on a result ticket, and that it names this very part.
fn ticket_url(ticket: &Value, part: u64, bytes: u64, tail: &str) -> Result<url::Url, Error> {
    if ticket["part"] != part
        || ticket["bytes"] != bytes
        || ticket["headers"] != serde_json::json!({"x-oss-forbid-overwrite":"true"})
    {
        return Err(Error::Response);
    }
    let uri = url::Url::parse(ticket["url"].as_str().ok_or(Error::Response)?)
        .map_err(|_| Error::Response)?;
    if uri.scheme() != "https"
        || !uri.host_str().is_some_and(|h| h.ends_with(".aliyuncs.com"))
        || !uri.username().is_empty()
        || uri.password().is_some()
        || uri.fragment().is_some()
        || !uri.path().ends_with(tail)
    {
        return Err(Error::Response);
    }
    Ok(uri)
}
const PUT_GAVE_UP: &str = "direct upload to object storage kept failing. Accepted parts are kept: \
     repeat this command to resume, or add --via-server to send the rest through the server";
const PUT_REFUSED: &str = "object storage refused a direct upload. Accepted parts are kept: \
     repeat this command to resume, or add --via-server to send the rest through the server";
const RECORD_GAVE_UP: &str = "the server did not record a part sent to object storage. Repeat \
     this command to resume: a part already stored is not sent twice";
struct Direct<'a> {
    client: &'a Client,
    base: String,
    /// The object key's end, up to the part number.
    tail: String,
}
impl Direct<'_> {
    /// `None` when this executor has no object store. Only the first part can choose
    /// that: nothing has been sent yet, so sending through the server is no retreat.
    fn send(
        &self,
        store: &mut dyn ObjectStore,
        number: u64,
        sha256: &str,
        part: &Part,
        first: bool,
    ) -> Result<Option<Value>, Error> {
        let identity = serde_json::json!({"part":number,"sha256":sha256});
        let tail = format!("{}/{number:05}", self.tail);
        let mut unsupported = false;
        repeated(store, PUT_GAVE_UP, |store| {
            // A new ticket each time: one lasts minutes, and waits can pass that.
            let ticket = match self
                .client
                .post(&format!("{}/upload-ticket", self.base), &identity)
            {
                Ok(ticket) => ticket,
                Err(Error::Http(failure))
                    if failure.error_code == "DATASET_DIRECT_UPLOAD_UNSUPPORTED" =>
                {
                    unsupported = true;
                    return Ok(());
                }
                Err(error) => return Err(server_attempt(error)),
            };
            let uri = ticket_url(&ticket, number, part.len(), &tail).map_err(Attempt::Stop)?;
            match store.put(&uri, part) {
                // 409: already stored and never replaced. Complete proves its bytes.
                Ok(200 | 409) => Ok(()),
                Ok(status) if ml_exp_core::retry::may_pass(status) => Err(Attempt::Again(format!(
                    "object storage answered HTTP {status}"
                ))),
                Ok(status) => {
                    eprintln!("Object storage answered HTTP {status}");
                    Err(Attempt::Stop(Error::Observation(PUT_REFUSED)))
                }
                Err(failure) => Err(Attempt::Again(failure)),
            }
        })?;
        if unsupported {
            return if first {
                Ok(None)
            } else {
                Err(Error::Observation(
                    "this executor stopped offering direct upload during this upload; \
                     add --via-server",
                ))
            };
        }
        repeated(store, RECORD_GAVE_UP, |_| {
            self.client
                .post(&format!("{}/part-uploaded", self.base), &identity)
                .map_err(server_attempt)
        })
        .map(Some)
    }
}
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
            "new_request_allowed",
            "diagnostic",
            "import_source",
            "runtime_id",
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
pub fn upload(
    client: &Client,
    project: &str,
    executor: &str,
    path: &Path,
    health: &Value,
    via_server: bool,
) -> Result<Value, Error> {
    upload_with(
        client,
        project,
        executor,
        path,
        health,
        via_server,
        &mut Real(client),
    )
}
/// [`upload`], with what it does to the object store supplied.
#[doc(hidden)]
pub fn upload_with(
    client: &Client,
    project: &str,
    executor: &str,
    path: &Path,
    health: &Value,
    via_server: bool,
    store: &mut dyn ObjectStore,
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
    definition.manifest_budget().map_err(|_| {
        Error::Observation(
            "dataset file list exceeds the 256 KiB a Run can mount, about 2,000 files; package many small files into fewer",
        )
    })?;
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
        let direct = Direct {
            client,
            base: format!("{base}/{asset}"),
            tail: format!("/datasets/{project}/{executor}/{}/parts", definition.sha256),
        };
        let offered = health["capabilities"]
            .as_array()
            .is_some_and(|names| names.iter().any(|name| name == DIRECT_UPLOAD));
        let mut use_direct = !via_server && offered;
        let mut sent_directly = false;
        if !via_server && !offered {
            eprintln!("This server does not offer direct upload; sending through it.");
        }
        if current.ready.is_none() {
            for number in 0..definition.part_count().map_err(|_| Error::Response)? {
                let count = definition.part_size(number).map_err(|_| Error::Response)?;
                archive
                    .seek(SeekFrom::Start(number * ml_exp_core::multipart::PART_BYTES))
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
                let part = Part::new(part);
                let mut stored = None;
                if use_direct {
                    stored = direct.send(store, number, &sha256, &part, !sent_directly)?;
                    sent_directly = stored.is_some();
                    if stored.is_none() {
                        eprintln!("This executor has no object store; sending through the server.");
                        use_direct = false;
                    }
                }
                let value = match stored {
                    Some(value) => value,
                    None => send(
                        client,
                        format!("{base}/{asset}/parts/{number}/{sha256}"),
                        reqwest::Method::PUT,
                        part.into_vec(),
                        false,
                    )?,
                };
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
    ml_exp_io::sync_directory(parent)
        .map_err(|_| Error::Observation("dataset saved; directory sync could not be confirmed"))?;
    Ok(
        serde_json::json!({"asset_id":asset,"sha256":metadata.definition.sha256,"bytes":metadata.definition.archive_bytes,"output":output,"verified":true,"acknowledged":false}),
    )
}
