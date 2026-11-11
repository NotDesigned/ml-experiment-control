//! Fetch the code source frozen in a package into its own Attempt, and prove it before
//! anything runs. Every Attempt has its own copy, so there is no shared cache to race
//! for and no lock to hold across nodes.
//!
//! The listing is published only after the tree is proven, and that is what says a tree
//! is complete. A directory rename would say it atomically, but the worker must not
//! need `renameat2` from the shared filesystem.
use crate::managed::{Error, Result};
use ml_exp_core::{
    cloud::launch::{CODE, Package},
    sources::{self, CodeFile, CodeManifest},
};
use ml_exp_io::directory::{Directory, EntryType, require_sealed, seal_regular};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::PermissionsExt,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

const MANIFEST: &str = ".code-manifest.json";
const MANIFEST_BYTES: u64 = sources::MAX_CODE_LISTING_BYTES as u64;
const NOT_STARTED: &str = "training was not started";

/// `status` and `logs` show this, so it names the file and what it means for the job.
macro_rules! refused {
    ($($why:tt)*) => {
        Error::storage(concat!(file!(), ":", line!()), &format_args!($($why)*))
    };
}
fn storage(error: std::io::Error) -> Failure {
    Failure::Final(Error::storage(concat!(file!(), ":", line!()), &error))
}
enum Failure {
    /// May pass by itself, and repeating the read has no effect on the server.
    Again(String),
    Final(Error),
}
fn repeated<T>(
    what: &str,
    cancelled: &AtomicBool,
    mut attempt: impl FnMut() -> std::result::Result<T, Failure>,
) -> Result<T> {
    let mut count = 0;
    loop {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        count += 1;
        match attempt() {
            Ok(value) => return Ok(value),
            Err(Failure::Final(error)) => return Err(error),
            Err(Failure::Again(why)) => {
                if count >= crate::retry::ATTEMPTS {
                    return Err(refused!(
                        "the code download gave up on {what}: {why}; {NOT_STARTED}"
                    ));
                }
                eprintln!("ML_EXPD_CODE=RETRY {count} of {what}: {why}");
                std::thread::sleep(crate::retry::delay(count));
            }
        }
    }
}
fn get(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
) -> std::result::Result<reqwest::blocking::Response, Failure> {
    let response = client.get(url).bearer_auth(token).send().map_err(|error| {
        Failure::Again(experiment_redact::redact_line(
            &error.without_url().to_string(),
        ))
    })?;
    let status = response.status().as_u16();
    if status == 200 {
        Ok(response)
    } else if crate::retry::may_pass(status) {
        Err(Failure::Again(format!("HTTP {status}")))
    } else {
        Err(Failure::Final(refused!(
            "the server refused the code download with HTTP {status}; {NOT_STARTED}"
        )))
    }
}
fn parse(bytes: &[u8], source: &str) -> Result<CodeManifest> {
    let manifest: Option<CodeManifest> = ml_exp_core::json::mapping(bytes)
        .ok()
        .and_then(|value| serde_json::from_value(serde_json::Value::Object(value)).ok());
    let manifest =
        manifest.ok_or_else(|| refused!("the code listing cannot be read; {NOT_STARTED}"))?;
    manifest
        .validate(source)
        .map_err(|_| refused!("the code listing is not one of {source}; {NOT_STARTED}"))?;
    Ok(manifest)
}
fn listing(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
    source: &str,
) -> std::result::Result<(Vec<u8>, CodeManifest), Failure> {
    let mut bytes = Vec::new();
    get(client, url, token)?
        .take(MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| Failure::Again(format!("the code listing: {error}")))?;
    if bytes.len() as u64 > MANIFEST_BYTES {
        return Err(Failure::Final(refused!(
            "the code listing is too large; {NOT_STARTED}"
        )));
    }
    let manifest = parse(&bytes, source).map_err(Failure::Final)?;
    Ok((bytes, manifest))
}
fn split(path: &str) -> (&str, &str) {
    path.rsplit_once('/').unwrap_or(("", path))
}
fn receive(
    response: &mut impl Read,
    file: &CodeFile,
    parent: &Directory,
) -> std::result::Result<(), Failure> {
    let path = &file.path;
    let (_, name) = split(path);
    let mut output = parent
        .create_new_regular(OsStr::new(name))
        .map_err(|error| {
            Failure::Final(refused!(
                "code file {path}: cannot be created: {error}; {NOT_STARTED}"
            ))
        })?;
    let mut remaining = file.bytes;
    let mut buffer = [0u8; 65536];
    while remaining > 0 {
        let limit = remaining.min(buffer.len() as u64) as usize;
        let count = response
            .read(&mut buffer[..limit])
            .map_err(|error| Failure::Again(format!("code file {path}: {error}")))?;
        if count == 0 {
            return Err(Failure::Again(format!(
                "code file {path}: the response ended early"
            )));
        }
        output.write_all(&buffer[..count]).map_err(|error| {
            Failure::Final(refused!(
                "code file {path}: cannot be written: {error}; {NOT_STARTED}"
            ))
        })?;
        remaining -= count as u64;
    }
    seal_regular(&output, file.executable).map_err(|error| {
        Failure::Final(refused!(
            "code file {path}: cannot be sealed: {error}; {NOT_STARTED}"
        ))
    })
}
fn fill(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
    tree: &Directory,
    manifest: &CodeManifest,
) -> std::result::Result<(), Failure> {
    let mut response = get(client, url, token)?;
    if response
        .content_length()
        .is_some_and(|bytes| bytes != manifest.total_bytes())
    {
        return Err(Failure::Again(
            "the code content is not the size of its listing".into(),
        ));
    }
    fill_from(&mut response, tree, manifest)
}
/// The files of the listing, one after another in its order, and nothing after them.
fn fill_from(
    response: &mut impl Read,
    tree: &Directory,
    manifest: &CodeManifest,
) -> std::result::Result<(), Failure> {
    let mut opened = BTreeMap::from([(String::new(), tree.clone())]);
    // Sorted, so a parent comes before everything inside it.
    for path in &manifest.directories {
        let (parent, name) = split(path);
        let parent = opened.get(parent).ok_or(Failure::Final(Error::Binding))?;
        let child = parent.create_new_child(OsStr::new(name)).map_err(storage)?;
        opened.insert(path.clone(), child);
    }
    for file in &manifest.files {
        let parent = opened
            .get(split(&file.path).0)
            .ok_or(Failure::Final(Error::Binding))?;
        receive(&mut *response, file, parent)?;
    }
    match response.read(&mut [0u8; 1]) {
        Ok(0) => {}
        Ok(_) => {
            return Err(Failure::Final(refused!(
                "the code content is longer than its listing; {NOT_STARTED}"
            )));
        }
        Err(error) => {
            return Err(Failure::Again(format!(
                "the end of the code content: {error}"
            )));
        }
    }
    ml_exp_io::datasets::seal(tree).map_err(storage)
}
fn extract(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
    attempt: &Directory,
    manifest: &CodeManifest,
) -> std::result::Result<Directory, Failure> {
    let tree = attempt
        .create_new_child(OsStr::new(CODE))
        .map_err(storage)?;
    if let Err(failure) = fill(client, url, token, &tree, manifest) {
        // Never verified and never used, so this is a staging tree and not code.
        ml_exp_io::datasets::remove_staging(attempt, &tree).map_err(storage)?;
        return Err(failure);
    }
    Ok(tree)
}
fn open(tree: &Directory, path: &str) -> std::io::Result<std::fs::File> {
    let (parents, name) = split(path);
    let mut directory = tree.clone();
    if !parents.is_empty() {
        for part in parents.split('/') {
            directory = directory.child(OsStr::new(part), false)?;
        }
    }
    directory.open_regular_read(OsStr::new(name))
}
/// Nothing more than the listing, and nothing that can be written to.
fn check_shape(
    directory: &Directory,
    prefix: &str,
    directories: &BTreeSet<&str>,
    files: &BTreeSet<&str>,
) -> Result<()> {
    if directory
        .mode()
        .map_err(crate::managed::storage_failure!())?
        & 0o222
        != 0
    {
        let shown = if prefix.is_empty() { CODE } else { prefix };
        return Err(refused!(
            "code directory {shown}: is writable; {NOT_STARTED}"
        ));
    }
    let names = directory
        .entry_names_limited(ml_exp_core::artifacts::MAX_FILES)
        .map_err(crate::managed::storage_failure!())?;
    for name in names {
        let text = name.to_string_lossy();
        let path = if prefix.is_empty() {
            text.into_owned()
        } else {
            format!("{prefix}/{text}")
        };
        let kind = directory
            .entry_type(&name)
            .map_err(crate::managed::storage_failure!())?;
        if kind == Some(EntryType::Directory) && directories.contains(path.as_str()) {
            let child = directory
                .child(&name, false)
                .map_err(crate::managed::storage_failure!())?;
            check_shape(&child, &path, directories, files)?;
        } else if !(kind == Some(EntryType::Regular) && files.contains(path.as_str())) {
            return Err(refused!(
                "code file {path}: is not part of the frozen source; {NOT_STARTED}"
            ));
        }
    }
    Ok(())
}
/// Each file against the listing, which names one that differs, and then the whole tree
/// against the source's identity, which is what proves the listing itself.
fn verify(tree: &Directory, manifest: &CodeManifest, source: &str) -> Result<()> {
    let plan = manifest
        .validate(source)
        .map_err(|_| refused!("the code listing is not one of {source}; {NOT_STARTED}"))?;
    check_shape(
        tree,
        "",
        &manifest.directories.iter().map(String::as_str).collect(),
        &manifest.files.iter().map(|f| f.path.as_str()).collect(),
    )?;
    for file in &manifest.files {
        let path = &file.path;
        let mut opened = open(tree, path)
            .map_err(|error| refused!("code file {path}: {error}; {NOT_STARTED}"))?;
        require_sealed(&opened)
            .map_err(|_| refused!("code file {path}: is writable; {NOT_STARTED}"))?;
        let mode = opened
            .metadata()
            .map_err(|error| refused!("code file {path}: {error}; {NOT_STARTED}"))?
            .permissions()
            .mode();
        if (mode & 0o111 != 0) != file.executable {
            return Err(refused!(
                "code file {path}: its executable bit differs from the frozen source; {NOT_STARTED}"
            ));
        }
        let (sha256, bytes) = ml_exp_io::artifacts::hash(&mut opened)
            .map_err(|error| refused!("code file {path}: {error}; {NOT_STARTED}"))?;
        if bytes != file.bytes || sha256 != file.sha256 {
            return Err(refused!(
                "code file {path}: its content differs from the frozen source; {NOT_STARTED}"
            ));
        }
    }
    let found = sources::tree_source_id(plan, |path, offset, len| {
        let mut file = open(tree, path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; len];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    })
    .map_err(crate::managed::storage_failure!())?;
    if found != source {
        return Err(refused!(
            "the code is {found}, not the frozen {source}; {NOT_STARTED}"
        ));
    }
    Ok(())
}
fn deliver_from(
    url: &str,
    token: &str,
    attempt_root: &Path,
    source: &str,
    cancelled: &AtomicBool,
) -> Result<()> {
    let attempt =
        Directory::open(attempt_root, false).map_err(crate::managed::storage_failure!())?;
    let name = OsStr::new(MANIFEST);
    let code = OsStr::new(CODE);
    if attempt
        .entry_exists(name)
        .map_err(crate::managed::storage_failure!())?
    {
        // A repeat of this Attempt uses what it proved before, or refuses. It never
        // fetches again over a tree that was proven once.
        let bytes = attempt
            .read_regular(name, MANIFEST_BYTES)
            .map_err(crate::managed::storage_failure!())?;
        let manifest = parse(&bytes, source)?;
        let tree = attempt.child(code, false).map_err(|error| {
            refused!("the code of this Attempt is gone: {error}; {NOT_STARTED}")
        })?;
        verify(&tree, &manifest, source)?;
        eprintln!("ML_EXPD_CODE=REUSED files={}", manifest.files.len());
        return Ok(());
    }
    if attempt
        .entry_exists(code)
        .map_err(crate::managed::storage_failure!())?
    {
        // Without a listing it was never proven, and the phase ledger says no process
        // has used it: an earlier download of this Attempt stopped part of the way.
        eprintln!("ML_EXPD_CODE=RESTART an unfinished download is removed");
        let unfinished = attempt
            .child(code, false)
            .map_err(crate::managed::storage_failure!())?;
        ml_exp_io::datasets::remove_staging(&attempt, &unfinished)
            .map_err(crate::managed::storage_failure!())?;
    }
    eprintln!("ML_EXPD_CODE=START {source}");
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(15 * 60))
        .build()
        .map_err(|_| Error::Configuration)?;
    let (bytes, manifest) = repeated("the code listing", cancelled, || {
        listing(&client, &format!("{url}/code"), token, source)
    })?;
    let tree = repeated("the code content", cancelled, || {
        extract(
            &client,
            &format!("{url}/code-content"),
            token,
            &attempt,
            &manifest,
        )
    })?;
    verify(&tree, &manifest, source)?;
    attempt
        .publish_new(name, &bytes)
        .map_err(crate::managed::storage_failure!())?;
    eprintln!(
        "ML_EXPD_CODE=READY files={} bytes={}",
        manifest.files.len(),
        manifest.total_bytes()
    );
    Ok(())
}
/// The tree the published listing describes, without reading its content again: the
/// delivery has just done that, and this only has to catch a tree changed since.
pub(crate) fn require_published(
    attempt: &Directory,
    attempt_root: &Path,
    workdir: &Path,
) -> Result<()> {
    if workdir != attempt_root.join(CODE) {
        return Err(Error::Configuration);
    }
    published_sizes(attempt).map(|_| ())
}
/// The listing's names and sizes against the tree beside it, and the source it names.
fn published_sizes(attempt: &Directory) -> Result<String> {
    let bytes = attempt
        .read_regular(OsStr::new(MANIFEST), MANIFEST_BYTES)
        .map_err(|error| refused!("the code has no published listing: {error}; {NOT_STARTED}"))?;
    let Some(source) = serde_json::from_slice::<CodeManifest>(&bytes)
        .ok()
        .map(|manifest| manifest.source_id)
    else {
        return Err(refused!("the code listing cannot be read; {NOT_STARTED}"));
    };
    let manifest = parse(&bytes, &source)?;
    let tree = attempt
        .child(OsStr::new(CODE), false)
        .map_err(|error| refused!("the code of this Attempt is gone: {error}; {NOT_STARTED}"))?;
    check_shape(
        &tree,
        "",
        &manifest.directories.iter().map(String::as_str).collect(),
        &manifest.files.iter().map(|f| f.path.as_str()).collect(),
    )?;
    for file in &manifest.files {
        let path = &file.path;
        let opened = open(&tree, path)
            .map_err(|error| refused!("code file {path}: {error}; {NOT_STARTED}"))?;
        let size = opened
            .metadata()
            .map_err(|error| refused!("code file {path}: {error}; {NOT_STARTED}"))?
            .len();
        if size != file.bytes {
            return Err(refused!(
                "code file {path}: its size differs from the frozen source; {NOT_STARTED}"
            ));
        }
    }
    Ok(source)
}
/// One code source on a Slurm cluster's shared storage: `PARENT/SOURCE` holds the sealed
/// tree `code` and, once that is proven, its listing. The gateway writes it once for any
/// number of Runs and Attempts, and every job proves it again before starting.
fn shared_directory(parent: &Path, source: &str, create: bool) -> Result<Option<Directory>> {
    sources::validate_source_id(source).map_err(|_| Error::Configuration)?;
    let parent = match Directory::open(parent, create) {
        Ok(directory) => directory,
        Err(error) if !create && error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => {
            return Err(refused!(
                "the shared code directory cannot be opened: {error}; {NOT_STARTED}"
            ));
        }
    };
    let name = OsStr::new(source);
    if !create
        && !parent
            .entry_exists(name)
            .map_err(crate::managed::storage_failure!())?
    {
        return Ok(None);
    }
    parent
        .child(name, create)
        .map(Some)
        .map_err(crate::managed::storage_failure!())
}
/// Every file against the listing and the tree against the source, as for an Attempt.
fn verify_listed(directory: &Directory, source: &str) -> Result<CodeManifest> {
    let bytes = directory
        .read_regular(OsStr::new(MANIFEST), MANIFEST_BYTES)
        .map_err(|error| refused!("the code {source} has no listing: {error}; {NOT_STARTED}"))?;
    let manifest = parse(&bytes, source)?;
    let tree = directory
        .child(OsStr::new(CODE), false)
        .map_err(|error| refused!("the code of {source} is gone: {error}; {NOT_STARTED}"))?;
    verify(&tree, &manifest, source)?;
    Ok(manifest)
}
/// Whether the source is on shared storage and proven. A tree without a listing was
/// never finished, and counts as absent: receiving it again starts over.
pub fn check_shared(parent: &Path, source: &str) -> Result<bool> {
    let Some(directory) = shared_directory(parent, source, false)? else {
        return Ok(false);
    };
    if !directory
        .entry_exists(OsStr::new(MANIFEST))
        .map_err(crate::managed::storage_failure!())?
    {
        return Ok(false);
    }
    verify_listed(&directory, source).map(|_| true)
}
/// Receive one code source: a four-byte big-endian length, its listing, then the content
/// of its files in the listing's order. True when it was there and proven already. A
/// proven tree is never written again; one that differs is refused, and stays for an
/// operator to look at.
pub fn receive_shared(parent: &Path, source: &str, input: &mut impl Read) -> Result<bool> {
    let directory = shared_directory(parent, source, true)?.ok_or(Error::Configuration)?;
    let name = OsStr::new(MANIFEST);
    let code = OsStr::new(CODE);
    if directory
        .entry_exists(name)
        .map_err(crate::managed::storage_failure!())?
    {
        verify_listed(&directory, source)?;
        // Read to the end, so that the sender's write completes.
        std::io::copy(input, &mut std::io::sink()).map_err(crate::managed::storage_failure!())?;
        return Ok(true);
    }
    if directory
        .entry_exists(code)
        .map_err(crate::managed::storage_failure!())?
    {
        // Unproven, so no job has used it: an earlier receipt stopped part of the way.
        let unfinished = directory
            .child(code, false)
            .map_err(crate::managed::storage_failure!())?;
        ml_exp_io::datasets::remove_staging(&directory, &unfinished)
            .map_err(crate::managed::storage_failure!())?;
    }
    let mut header = [0u8; 4];
    input
        .read_exact(&mut header)
        .map_err(|error| refused!("the code listing did not arrive: {error}; {NOT_STARTED}"))?;
    let size = u64::from(u32::from_be_bytes(header));
    if size == 0 || size > MANIFEST_BYTES {
        return Err(refused!("the code listing is too large; {NOT_STARTED}"));
    }
    let mut bytes = vec![0; size as usize];
    input
        .read_exact(&mut bytes)
        .map_err(|error| refused!("the code listing did not arrive: {error}; {NOT_STARTED}"))?;
    let manifest = parse(&bytes, source)?;
    let tree = directory
        .create_new_child(code)
        .map_err(crate::managed::storage_failure!())?;
    if let Err(failure) = fill_from(input, &tree, &manifest) {
        ml_exp_io::datasets::remove_staging(&directory, &tree)
            .map_err(crate::managed::storage_failure!())?;
        return Err(match failure {
            Failure::Final(error) => error,
            Failure::Again(why) => refused!("the code content: {why}; {NOT_STARTED}"),
        });
    }
    verify(&tree, &manifest, source)?;
    directory
        .publish_new(name, &bytes)
        .map_err(crate::managed::storage_failure!())?;
    Ok(false)
}
/// The delivered code a Slurm job works in, `PARENT/SOURCE/code`. With `content`, every
/// file is hashed, as a job does before it starts; without, names and sizes only, as the
/// gateway checks before it asks the scheduler for anything.
pub fn require_shared(workdir: &Path, source: &str, content: bool) -> Result<()> {
    let directory = workdir
        .parent()
        .filter(|parent| {
            workdir.file_name() == Some(OsStr::new(CODE))
                && parent.file_name() == Some(OsStr::new(source))
        })
        .ok_or(Error::Configuration)?;
    let directory = Directory::open(directory, false).map_err(|error| {
        refused!("the delivered code {source} is not on shared storage: {error}; {NOT_STARTED}")
    })?;
    if content {
        let manifest = verify_listed(&directory, source)?;
        eprintln!(
            "ML_EXPD_CODE=VERIFIED files={} bytes={}",
            manifest.files.len(),
            manifest.total_bytes()
        );
    } else if published_sizes(&directory)? != source {
        return Err(refused!(
            "the delivered code is not {source}; {NOT_STARTED}"
        ));
    }
    Ok(())
}
/// Before training, and before `managed::run` opens the working directory. The launch
/// lock is held throughout, because this may remove an unfinished tree.
pub fn deliver(package: &Package, cancelled: &AtomicBool) -> Result<()> {
    let Some(source) = &package.code_source_id else {
        return Ok(());
    };
    let _held = crate::managed::hold_unstarted(&package.config)?;
    deliver_from(
        &package.channel.url,
        &package.channel.token,
        &package.config.attempt_root,
        source,
        cancelled,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_core::cloud::{
        exchange::Channel,
        launch::{self, StorageLayout},
    };
    use std::{collections::BTreeMap, fs, net::TcpListener};

    /// A manifest with its real identity, and what the server sends for it.
    fn source(files: &[(&str, &[u8], bool)]) -> (CodeManifest, Vec<u8>) {
        let mut listed = Vec::new();
        let mut directories = BTreeSet::new();
        let mut content = Vec::new();
        let mut files = files.to_vec();
        files.sort_by_key(|file| file.0);
        for (path, body, executable) in &files {
            let mut parent = *path;
            while let Some((above, _)) = parent.rsplit_once('/') {
                directories.insert(above.to_owned());
                parent = above;
            }
            let (sha256, bytes) = ml_exp_io::artifacts::hash(&mut &body[..]).unwrap();
            listed.push(CodeFile {
                path: (*path).into(),
                bytes,
                sha256,
                executable: *executable,
            });
            content.extend_from_slice(body);
        }
        let mut manifest = CodeManifest {
            source_id: String::new(),
            directories: directories.into_iter().collect(),
            files: listed,
        };
        let by_path: BTreeMap<_, _> = files.iter().map(|f| (f.0, f.1)).collect();
        manifest.source_id =
            sources::tree_source_id(manifest.plan().unwrap(), |path, offset, len| {
                let offset = offset as usize;
                Ok(by_path[path][offset..offset + len].to_vec())
            })
            .unwrap();
        (manifest, content)
    }
    /// One real HTTP reply per request, in order, on a real socket.
    fn server(replies: Vec<(u16, Vec<u8>)>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for (status, body) in replies {
                let (mut socket, _) = listener.accept().unwrap();
                let mut head = Vec::new();
                while !head.ends_with(b"\r\n\r\n") {
                    let mut byte = [0];
                    socket.read_exact(&mut byte).unwrap();
                    head.push(byte[0]);
                }
                let head = String::from_utf8(head).unwrap();
                seen.push(head.split_whitespace().nth(1).unwrap().to_owned());
                assert!(
                    head.to_lowercase()
                        .contains("authorization: bearer fixture-token")
                );
                write!(
                    socket,
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                socket.write_all(&body).unwrap();
            }
            seen
        });
        (url, thread)
    }
    fn deliver_with(url: &str, attempt: &Path, source: &str) -> Result<()> {
        deliver_from(
            url,
            "fixture-token",
            attempt,
            source,
            &AtomicBool::new(false),
        )
    }
    fn train() -> (CodeManifest, Vec<u8>) {
        source(&[
            ("train.py", &b"print('one')\n"[..], false),
            ("src/model/net.py", &b"layers = 2\n"[..], false),
            ("run.sh", &b"#!/bin/sh\n"[..], true),
        ])
    }
    fn delivered(temp: &tempfile::TempDir) -> (std::path::PathBuf, CodeManifest, Vec<u8>) {
        let attempt = temp.path().join("attempt");
        fs::create_dir(&attempt).unwrap();
        let (manifest, content) = train();
        let (url, thread) = server(vec![
            (200, serde_json::to_vec(&manifest).unwrap()),
            (200, content.clone()),
        ]);
        deliver_with(&url, &attempt, &manifest.source_id).unwrap();
        assert_eq!(thread.join().unwrap(), ["/code", "/code-content"]);
        (attempt, manifest, content)
    }
    fn message(error: Error) -> String {
        error.to_string()
    }
    /// What the server sends the gateway for one source.
    fn frame(manifest: &CodeManifest, content: &[u8]) -> Vec<u8> {
        let listing = serde_json::to_vec(manifest).unwrap();
        let mut frame = (listing.len() as u32).to_be_bytes().to_vec();
        frame.extend_from_slice(&listing);
        frame.extend_from_slice(content);
        frame
    }
    #[test]
    fn shared_code_is_received_once_proven_and_reused_by_every_job() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("delivered/code");
        let (manifest, content) = train();
        let source = manifest.source_id.clone();
        assert!(!check_shared(&parent, &source).unwrap());
        let reused =
            receive_shared(&parent, &source, &mut &frame(&manifest, &content)[..]).unwrap();
        assert!(!reused);
        assert!(check_shared(&parent, &source).unwrap());
        let workdir = parent.join(&source).join(CODE);
        assert_eq!(
            fs::read(workdir.join("train.py")).unwrap(),
            b"print('one')\n"
        );
        let mode = |path: &str| {
            fs::metadata(workdir.join(path))
                .unwrap()
                .permissions()
                .mode()
                & 0o777
        };
        assert_eq!(mode("train.py"), 0o400);
        assert_eq!(mode("."), 0o500);
        require_shared(&workdir, &source, true).unwrap();
        require_shared(&workdir, &source, false).unwrap();
        // A second receipt proves the copy and writes nothing, whatever it is sent.
        assert!(receive_shared(&parent, &source, &mut &b"anything"[..]).unwrap());
        // A working directory that is not this source's tree is no delivered code.
        assert!(require_shared(&parent.join(&source), &source, true).is_err());
        let (other, _) = source_of_one();
        assert!(require_shared(&workdir, &other.source_id, true).is_err());
    }
    fn source_of_one() -> (CodeManifest, Vec<u8>) {
        source(&[("one.py", &b"1\n"[..], false)])
    }
    #[test]
    fn a_tampered_shared_file_is_named_and_refused_and_never_overwritten() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("code");
        let (manifest, content) = train();
        let source = manifest.source_id.clone();
        // Changed on the way: nothing is proven, and the next receipt starts over.
        let mut tampered = content.clone();
        tampered["#!/bin/sh\n".len()] ^= 1;
        let refused = message(
            receive_shared(&parent, &source, &mut &frame(&manifest, &tampered)[..]).unwrap_err(),
        );
        assert!(refused.contains("code file src/model/net.py"), "{refused}");
        assert!(!check_shared(&parent, &source).unwrap());
        receive_shared(&parent, &source, &mut &frame(&manifest, &content)[..]).unwrap();
        // Changed after it was proven, keeping its size: every check that hashes refuses
        // it by name, and receiving it again does not repair it.
        let workdir = parent.join(&source).join(CODE);
        let file = workdir.join("train.py");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&file, b"print('two')\n").unwrap();
        fs::set_permissions(&file, fs::Permissions::from_mode(0o400)).unwrap();
        let refused = message(check_shared(&parent, &source).unwrap_err());
        assert!(refused.contains("code file train.py"), "{refused}");
        let refused = message(require_shared(&workdir, &source, true).unwrap_err());
        assert!(refused.contains("code file train.py"), "{refused}");
        assert!(refused.contains("training was not started"), "{refused}");
        require_shared(&workdir, &source, false).unwrap();
        assert!(receive_shared(&parent, &source, &mut &frame(&manifest, &content)[..]).is_err());
        assert_eq!(fs::read(&file).unwrap(), b"print('two')\n");
        // A short or a long content is refused as well.
        let other = temp.path().join("other");
        let short = frame(&manifest, &content[..content.len() - 1]);
        assert!(receive_shared(&other, &source, &mut &short[..]).is_err());
        let mut long = frame(&manifest, &content);
        long.push(b'x');
        assert!(receive_shared(&other, &source, &mut &long[..]).is_err());
        assert!(!check_shared(&other, &source).unwrap());
    }
    #[test]
    fn a_delivered_tree_is_exact_sealed_and_reused_without_asking_again() {
        let temp = tempfile::tempdir().unwrap();
        let (attempt, manifest, _) = delivered(&temp);
        let code = attempt.join("code");
        assert_eq!(
            fs::read(code.join("src/model/net.py")).unwrap(),
            b"layers = 2\n"
        );
        let mode = |path: &str| fs::metadata(code.join(path)).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode("train.py"), 0o400);
        assert_eq!(mode("run.sh"), 0o500);
        assert_eq!(mode("src/model"), 0o500);
        assert_eq!(mode("."), 0o500);
        // Nothing listens here, so a repeat that needed the server would fail.
        deliver_with("http://127.0.0.1:1", &attempt, &manifest.source_id).unwrap();
    }
    #[test]
    fn a_file_changed_in_transit_is_named_and_leaves_nothing_proven() {
        let temp = tempfile::tempdir().unwrap();
        let attempt = temp.path().join("attempt");
        fs::create_dir(&attempt).unwrap();
        let (manifest, content) = train();
        let mut tampered = content.clone();
        // run.sh, then src/model/net.py, then train.py: the listing's order.
        let at = "#!/bin/sh\n".len();
        tampered[at] ^= 1;
        let (url, thread) = server(vec![
            (200, serde_json::to_vec(&manifest).unwrap()),
            (200, tampered),
        ]);
        let refused = message(deliver_with(&url, &attempt, &manifest.source_id).unwrap_err());
        assert!(refused.contains("code file src/model/net.py"), "{refused}");
        assert!(refused.contains("training was not started"), "{refused}");
        assert_eq!(thread.join().unwrap().len(), 2);
        assert!(!attempt.join(MANIFEST).exists());
        // Nothing was proven, so the next run of this Attempt may fetch it again.
        let (url, thread) = server(vec![
            (200, serde_json::to_vec(&manifest).unwrap()),
            (200, content),
        ]);
        deliver_with(&url, &attempt, &manifest.source_id).unwrap();
        thread.join().unwrap();
        assert!(attempt.join(MANIFEST).exists());
    }
    #[test]
    fn a_proven_tree_changed_later_is_refused_by_name_and_never_overwritten() {
        let temp = tempfile::tempdir().unwrap();
        let (attempt, manifest, _) = delivered(&temp);
        let file = attempt.join("code/train.py");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&file, b"print('two')\n").unwrap();
        let refused =
            message(deliver_with("http://127.0.0.1:1", &attempt, &manifest.source_id).unwrap_err());
        assert!(refused.contains("code file train.py"), "{refused}");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o400)).unwrap();
        let refused =
            message(deliver_with("http://127.0.0.1:1", &attempt, &manifest.source_id).unwrap_err());
        assert!(refused.contains("code file train.py"), "{refused}");
        assert_eq!(fs::read(&file).unwrap(), b"print('two')\n");
    }
    #[test]
    fn an_added_file_or_a_writable_directory_is_refused_by_name() {
        let temp = tempfile::tempdir().unwrap();
        let (attempt, manifest, _) = delivered(&temp);
        let code = attempt.join("code");
        let mode = |bits| fs::set_permissions(&code, fs::Permissions::from_mode(bits)).unwrap();
        mode(0o700);
        let refused =
            message(deliver_with("http://127.0.0.1:1", &attempt, &manifest.source_id).unwrap_err());
        assert!(refused.contains("code directory code"), "{refused}");
        fs::write(code.join("extra.py"), b"").unwrap();
        mode(0o500);
        let refused =
            message(deliver_with("http://127.0.0.1:1", &attempt, &manifest.source_id).unwrap_err());
        assert!(refused.contains("code file extra.py"), "{refused}");
    }
    #[test]
    fn a_listing_altered_with_its_files_still_fails_the_source_identity() {
        let temp = tempfile::tempdir().unwrap();
        let attempt = temp.path().join("attempt");
        fs::create_dir(&attempt).unwrap();
        let (frozen, _) = train();
        let (other, content) = source(&[("train.py", &b"print('forged')\n"[..], false)]);
        let mut forged = other.clone();
        forged.source_id = frozen.source_id.clone();
        let (url, thread) = server(vec![
            (200, serde_json::to_vec(&forged).unwrap()),
            (200, content),
        ]);
        let refused = message(deliver_with(&url, &attempt, &frozen.source_id).unwrap_err());
        assert!(refused.contains(&other.source_id), "{refused}");
        assert!(refused.contains(&frozen.source_id), "{refused}");
        thread.join().unwrap();
        assert!(!attempt.join(MANIFEST).exists());
    }
    #[test]
    fn a_passing_failure_is_tried_again_and_a_refusal_is_not() {
        let temp = tempfile::tempdir().unwrap();
        let attempt = temp.path().join("attempt");
        fs::create_dir(&attempt).unwrap();
        let (manifest, content) = train();
        let (url, thread) = server(vec![(401, Vec::new())]);
        let refused = message(deliver_with(&url, &attempt, &manifest.source_id).unwrap_err());
        assert!(refused.contains("HTTP 401"), "{refused}");
        assert_eq!(thread.join().unwrap(), ["/code"]);
        // A short body and a busy server each cost one more read, and nothing partial stays.
        let (url, thread) = server(vec![
            (503, Vec::new()),
            (200, serde_json::to_vec(&manifest).unwrap()),
            (200, content[..3].to_vec()),
            (200, content),
        ]);
        deliver_with(&url, &attempt, &manifest.source_id).unwrap();
        assert_eq!(
            thread.join().unwrap(),
            ["/code", "/code", "/code-content", "/code-content"]
        );
    }
    #[test]
    fn the_real_launch_starts_in_the_delivered_tree_with_its_environment() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("runs");
        fs::create_dir(&root).unwrap();
        let (manifest, content) = train();
        let workdir = root.join("demo/train/attempt-001/code");
        let argv = vec![
            "/bin/sh".into(),
            "-c".into(),
            "pwd > \"$STATE_DIR/pwd\"; printf %s \"$SOURCE_DIR\" > \"$STATE_DIR/source\"; \
             printf %s \"$PYTHONDONTWRITEBYTECODE\" > \"$STATE_DIR/bytecode\"; \
             cat train.py > \"$STATE_DIR/code\""
                .into(),
        ];
        let mut package = launch::compile(launch::Input {
            root: root.to_str().unwrap(),
            project: "demo",
            run: "train",
            attempt: "attempt-001",
            workdir: workdir.to_str().unwrap(),
            argv: &argv,
            timeout_seconds: 30,
            worker_sha256: &format!("sha256:{}", "a".repeat(64)),
            storage: StorageLayout {
                mount: temp.path().into(),
                runs: root.clone(),
                datasets: temp.path().join("datasets"),
            },
            channel: Channel {
                url: "https://127.0.0.1:1/api/worker-exchange/demo/train/attempt-001".into(),
                token: "b".repeat(64),
            },
        })
        .unwrap();
        package.code_source_id = Some(manifest.source_id.clone());
        package.manifest.environment.extend([
            (
                "SOURCE_DIR".to_owned(),
                workdir.to_str().unwrap().to_owned(),
            ),
            ("PYTHONDONTWRITEBYTECODE".to_owned(), "1".to_owned()),
        ]);
        package
            .attach(
                temp.path().join("datasets").to_str().unwrap(),
                Vec::new(),
                None,
            )
            .unwrap();
        // The working directory cannot exist yet: it is inside the Attempt root.
        crate::cloud::prepare(&package, &BTreeMap::new()).unwrap();
        assert!(!workdir.exists());
        let cancelled = AtomicBool::new(false);
        assert!(crate::managed::run(&package.config, &AtomicBool::new(true)).is_err());
        // Without its code the launch fails before it records a start, and can go on.
        assert!(crate::managed::run(&package.config, &cancelled).is_err());
        assert_eq!(
            crate::managed::inspect(&package.config).unwrap().phase,
            crate::managed::Phase::Prepared
        );
        // A second worker cannot begin to deliver while the launch lock is held.
        {
            let _first = crate::managed::hold_unstarted(&package.config).unwrap();
            assert!(matches!(deliver(&package, &cancelled), Err(Error::Busy)));
        }
        let (url, thread) = server(vec![
            (200, serde_json::to_vec(&manifest).unwrap()),
            (200, content),
        ]);
        deliver_with(&url, &package.config.attempt_root, &manifest.source_id).unwrap();
        thread.join().unwrap();
        // The start refuses a tree that no longer matches what was published, and a
        // refusal leaves the Attempt able to go on.
        let marker = package.config.attempt_root.join(MANIFEST);
        let hidden = marker.with_extension("hidden");
        fs::rename(&marker, &hidden).unwrap();
        let refused = message(crate::managed::run(&package.config, &cancelled).unwrap_err());
        assert!(refused.contains("no published listing"), "{refused}");
        fs::rename(&hidden, &marker).unwrap();
        let file = workdir.join("train.py");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&file, b"short").unwrap();
        let refused = message(crate::managed::run(&package.config, &cancelled).unwrap_err());
        assert!(refused.contains("code file train.py"), "{refused}");
        fs::write(&file, b"print('one')\n").unwrap();
        fs::set_permissions(&file, fs::Permissions::from_mode(0o400)).unwrap();
        let receipt = crate::managed::run(&package.config, &cancelled).unwrap();
        assert_eq!(
            receipt.receipt.as_ref().unwrap().training,
            ml_exp_core::managed::evidence::Training::Completed
        );
        let state = package.config.attempt_root.join("state");
        let read = |name: &str| fs::read_to_string(state.join(name)).unwrap();
        assert_eq!(read("pwd").trim_end(), workdir.to_str().unwrap());
        assert_eq!(read("source"), workdir.to_str().unwrap());
        assert_eq!(read("bytecode"), "1");
        assert_eq!(read("code"), "print('one')\n");
    }
}
