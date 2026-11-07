//! Private OSS object effects. Large archives are immutable 16MiB objects;
//! the API retains receipts only, never payload files or full archive buffers.
use base64::{Engine, engine::general_purpose::STANDARD};
use hmac::{Hmac, Mac};
use ml_exp_core::{
    artifacts::ArchiveReceipt,
    multipart::{MAX_PARTS, PART_BYTES},
};
use serde::Deserialize;
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{self, Read},
    path::Path,
    time::Duration,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub endpoint: String,
    pub bucket: String,
    pub prefix: String,
    pub credentials_file: std::path::PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Credentials {
    access_key_id: String,
    access_key_secret: String,
    security_token: Option<String>,
}
pub struct Store {
    settings: Settings,
    /// `https://bucket.endpoint`, checked at open. Only a test points it elsewhere.
    origin: String,
    credentials: Credentials,
    http: reqwest::blocking::Client,
}
fn invalid() -> io::Error {
    io::Error::other("object store scope, credentials or response cannot be verified")
}
fn safe_component(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 255
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        && !matches!(s, "." | "..")
}
pub fn parts(archive: &ArchiveReceipt) -> io::Result<u64> {
    let n = archive.bytes.div_ceil(PART_BYTES);
    if n == 0 || n > MAX_PARTS {
        return Err(invalid());
    }
    Ok(n)
}
impl Store {
    /// A real socket for tests of the signed requests. Configuration cannot reach this.
    #[doc(hidden)]
    pub fn redirect_for_tests(&mut self, origin: String) {
        self.origin = origin;
    }
    pub fn validate_import(&self, request: &ml_exp_core::datasets::oss::Import) -> io::Result<()> {
        request.bind(&self.settings.prefix).map_err(why!(invalid()))
    }
    pub fn open_import(
        &self,
        request: &ml_exp_core::datasets::oss::Import,
    ) -> io::Result<reqwest::blocking::Response> {
        self.validate_import(request)?;
        let response = self
            .http
            .get(self.signed_url(
                "GET",
                &request.object_key,
                time::OffsetDateTime::now_utc().unix_timestamp() + 300,
            )?)
            .timeout(Duration::from_secs(1200))
            .send()
            .map_err(why_chain!(invalid()))?;
        if response.status() != 200 || response.content_length() != Some(request.archive_bytes) {
            return Err(invalid());
        }
        Ok(response)
    }
    pub fn inspect_import(
        &self,
        request: &ml_exp_core::datasets::oss::Import,
    ) -> io::Result<ml_exp_core::datasets::Definition> {
        ml_exp_io::artifacts::inspect_data_stream(
            &mut self.open_import(request)?,
            &request.sha256,
            request.archive_bytes,
        )
        .map_err(why!({
            io::Error::new(
                io::ErrorKind::InvalidData,
                "dataset archive verification failed",
            )
        }))
    }
    pub fn open(settings: Settings) -> io::Result<Self> {
        let expected = format!("https://{}.{}", settings.bucket, settings.endpoint);
        let url = url::Url::parse(&expected).map_err(why!(invalid()))?;
        if !settings.endpoint.starts_with("oss-")
            || !settings.endpoint.ends_with(".aliyuncs.com")
            || !settings
                .endpoint
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"-.".contains(&b))
            || !safe_component(&settings.bucket)
            || url.host_str() != Some(format!("{}.{}", settings.bucket, settings.endpoint).as_str())
            || !settings.prefix.starts_with("users/")
            || !settings.prefix.ends_with('/')
            || settings
                .prefix
                .trim_end_matches('/')
                .split('/')
                .any(|v| !safe_component(v))
            || settings.prefix.split('/').count() < 4
        {
            return Err(invalid());
        }
        let bytes = ml_exp_io::read_private(Path::new(&settings.credentials_file), 16384, true)?;
        let credentials: Credentials = serde_json::from_value(serde_json::Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(why!(
                invalid(),
                "object store credentials are not one mapping"
            ))?,
        ))
        .map_err(why!(invalid(), "object store credentials schema invalid"))?;
        if credentials.access_key_id.is_empty()
            || credentials.access_key_secret.is_empty()
            || [&credentials.access_key_id, &credentials.access_key_secret]
                .iter()
                .any(|v| v.chars().any(char::is_control))
            || credentials
                .security_token
                .as_ref()
                .is_some_and(|s| s.is_empty() || s.chars().any(char::is_control))
        {
            return Err(invalid());
        }
        let http = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(why!(invalid()))?;
        Ok(Self {
            origin: expected,
            settings,
            credentials,
            http,
        })
    }
    fn key(&self, archive: &ArchiveReceipt, part: u64) -> io::Result<String> {
        ml_exp_core::artifacts::validate_receipt(archive).map_err(why!(invalid()))?;
        if part >= parts(archive)? {
            return Err(invalid());
        }
        Ok(format!(
            "{}results/{}/{}/{}/{}/parts/{part:05}",
            self.settings.prefix,
            archive.project,
            archive.run_id,
            archive.attempt_id,
            archive.sha256
        ))
    }
    fn url(
        &self,
        method: &str,
        archive: &ArchiveReceipt,
        part: u64,
        expires: i64,
    ) -> io::Result<String> {
        self.signed_url(method, &self.key(archive, part)?, expires)
    }
    fn signed_url(&self, method: &str, key: &str, expires: i64) -> io::Result<String> {
        let mut query = BTreeMap::new();
        if let Some(token) = &self.credentials.security_token {
            query.insert("security-token", token.clone());
        }
        let canonical_resource = format!(
            "/{}/{}{}",
            self.settings.bucket,
            key,
            query
                .get("security-token")
                .map(|t| format!("?security-token={t}"))
                .unwrap_or_default()
        );
        let headers = if method == "PUT" {
            "x-oss-forbid-overwrite:true\n"
        } else {
            ""
        };
        let canonical = format!("{method}\n\n\n{expires}\n{headers}{canonical_resource}");
        let mut mac = Hmac::<Sha1>::new_from_slice(self.credentials.access_key_secret.as_bytes())
            .map_err(why!(invalid(), "object store signing key rejected"))?;
        mac.update(canonical.as_bytes());
        query.insert("OSSAccessKeyId", self.credentials.access_key_id.clone());
        query.insert("Expires", expires.to_string());
        query.insert("Signature", STANDARD.encode(mac.finalize().into_bytes()));
        let mut url =
            url::Url::parse(&format!("{}/{key}", self.origin)).map_err(why!(invalid()))?;
        url.query_pairs_mut().extend_pairs(query);
        Ok(url.into())
    }
    pub fn ticket(
        &self,
        archive: &ArchiveReceipt,
        part: u64,
        upload: bool,
        now: i64,
    ) -> io::Result<serde_json::Value> {
        if part >= parts(archive)? {
            return Err(invalid());
        }
        let bytes =
            (archive.bytes - part.checked_mul(PART_BYTES).ok_or_else(invalid)?).min(PART_BYTES);
        Ok(
            serde_json::json!({"url":self.url(if upload{"PUT"}else{"GET"},archive,part,now.checked_add(300).ok_or_else(invalid)?)?,"part":part,"bytes":bytes,
            "headers":if upload{serde_json::json!({"x-oss-forbid-overwrite":"true"})}else{serde_json::json!({})},"expires_in_seconds":300}),
        )
    }
    pub fn verify(&self, archive: &ArchiveReceipt) -> io::Result<()> {
        let mut reader = Reader {
            store: self,
            archive,
            part: 0,
            current: None,
            remaining: 0,
        };
        ml_exp_io::artifacts::verify_stream(&mut reader, archive)?;
        Ok(())
    }
    fn data_key(
        &self,
        project: &str,
        executor: &str,
        definition: &ml_exp_core::datasets::Definition,
        part: u64,
    ) -> io::Result<String> {
        ml_exp_core::identity::validate_id(project)
            .and_then(|_| ml_exp_core::identity::validate_id(executor))
            .map_err(why!(invalid()))?;
        if part >= definition.part_count().map_err(why!(invalid()))? {
            return Err(invalid());
        }
        Ok(format!(
            "{}datasets/{project}/{executor}/{}/parts/{part:05}",
            self.settings.prefix, definition.sha256
        ))
    }
    pub fn data_ticket(
        &self,
        project: &str,
        executor: &str,
        definition: &ml_exp_core::datasets::Definition,
        part: u64,
    ) -> io::Result<serde_json::Value> {
        let key = self.data_key(project, executor, definition, part)?;
        Ok(
            serde_json::json!({"part":part,"bytes":definition.part_size(part).map_err(why!(invalid()))?,"url":self.signed_url("GET",&key,time::OffsetDateTime::now_utc().unix_timestamp()+300)?}),
        )
    }
    /// One PUT of one part of one asset, for 300 seconds, that cannot replace an object:
    /// the header is signed, so a request without it is refused. The signature does not
    /// cover the bytes, so `verify_data` decides whether they are accepted.
    pub fn data_upload_ticket(
        &self,
        project: &str,
        executor: &str,
        definition: &ml_exp_core::datasets::Definition,
        part: u64,
    ) -> io::Result<serde_json::Value> {
        let key = self.data_key(project, executor, definition, part)?;
        Ok(serde_json::json!({
            "part": part,
            "bytes": definition.part_size(part).map_err(why!(invalid()))?,
            "url": self.signed_url("PUT", &key, time::OffsetDateTime::now_utc().unix_timestamp() + 300)?,
            "headers": {"x-oss-forbid-overwrite": "true"},
            "expires_in_seconds": 300,
        }))
    }
    /// The size of a stored part, without reading it. `None` when there is no object.
    /// The header is read directly: reqwest reports the empty body of a HEAD as length 0.
    pub fn data_part_size(
        &self,
        project: &str,
        executor: &str,
        definition: &ml_exp_core::datasets::Definition,
        part: u64,
    ) -> io::Result<Option<u64>> {
        let key = self.data_key(project, executor, definition, part)?;
        let response = self
            .http
            .head(self.signed_url(
                "HEAD",
                &key,
                time::OffsetDateTime::now_utc().unix_timestamp() + 300,
            )?)
            .send()
            .map_err(|error| error.without_url())
            .map_err(why_chain!(invalid()))?;
        match response.status().as_u16() {
            404 => Ok(None),
            200 => response
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse().ok())
                .map(Some)
                .ok_or_else(invalid),
            _ => Err(invalid()),
        }
    }
    /// Remove one part of an asset the caller holds unaccepted. Missing is fine.
    pub fn delete_data(
        &self,
        project: &str,
        executor: &str,
        definition: &ml_exp_core::datasets::Definition,
        part: u64,
    ) -> io::Result<()> {
        let key = self.data_key(project, executor, definition, part)?;
        let response = self
            .http
            .delete(self.signed_url(
                "DELETE",
                &key,
                time::OffsetDateTime::now_utc().unix_timestamp() + 300,
            )?)
            .send()
            .map_err(|error| error.without_url())
            .map_err(why_chain!(invalid()))?;
        if !matches!(response.status().as_u16(), 204 | 404) {
            return Err(invalid());
        }
        Ok(())
    }
    /// The recorded parts whose object is missing or differs from its record. Only for
    /// after a refused archive. A failed read is an error and names no part.
    pub fn wrong_data_parts(
        &self,
        project: &str,
        executor: &str,
        state: &ml_exp_core::datasets::State,
    ) -> io::Result<Vec<u64>> {
        let mut wrong = Vec::new();
        for (part, record) in &state.parts {
            let key = self.data_key(project, executor, &state.definition, *part)?;
            let response = self
                .http
                .get(self.signed_url(
                    "GET",
                    &key,
                    time::OffsetDateTime::now_utc().unix_timestamp() + 300,
                )?)
                .send()
                .map_err(|error| error.without_url())
                .map_err(why_chain!(invalid()))?;
            match response.status().as_u16() {
                404 => wrong.push(*part),
                200 => {
                    let (hash, size) =
                        ml_exp_io::artifacts::hash(&mut response.take(record.bytes + 1))?;
                    if hash != record.sha256 || size != record.bytes {
                        wrong.push(*part);
                    }
                }
                _ => return Err(invalid()),
            }
        }
        Ok(wrong)
    }
    pub fn put_data(
        &self,
        project: &str,
        executor: &str,
        definition: &ml_exp_core::datasets::Definition,
        part: u64,
        sha256: &str,
        bytes: &[u8],
    ) -> io::Result<()> {
        if definition.part_size(part).map_err(why!(invalid()))? != bytes.len() as u64
            || format!("{:x}", Sha256::digest(bytes)) != sha256
        {
            return Err(invalid());
        }
        let key = self.data_key(project, executor, definition, part)?;
        let response = self
            .http
            .put(self.signed_url(
                "PUT",
                &key,
                time::OffsetDateTime::now_utc().unix_timestamp() + 300,
            )?)
            .header("x-oss-forbid-overwrite", "true")
            .body(bytes.to_vec())
            .send()
            .map_err(why_chain!(invalid()))?;
        if !matches!(response.status().as_u16(), 200 | 409) {
            return Err(invalid());
        }
        // Lost PUT responses and duplicates never overwrite a different object.
        let response = self
            .http
            .get(self.signed_url(
                "GET",
                &key,
                time::OffsetDateTime::now_utc().unix_timestamp() + 300,
            )?)
            .send()
            .map_err(why_chain!(invalid()))?;
        if response.status() != 200 || response.content_length() != Some(bytes.len() as u64) {
            return Err(invalid());
        }
        let (hash, size) = ml_exp_io::artifacts::hash(&mut response.take(bytes.len() as u64 + 1))?;
        if hash != sha256 || size != bytes.len() as u64 {
            // An unrecorded object from a direct PUT can differ. It cannot be overwritten.
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stored object differs from this part",
            ));
        }
        Ok(())
    }
    /// Read the whole archive: the only proof of the bytes of a part that went straight
    /// to the object store. True when the archive, every member and every part's record
    /// all hold. False when the bytes are refused. An error says nothing about them.
    pub fn verify_data(
        &self,
        project: &str,
        executor: &str,
        state: &ml_exp_core::datasets::State,
    ) -> io::Result<bool> {
        let mut reader = DataReader {
            store: self,
            project,
            executor,
            state,
            part: 0,
            current: None,
            remaining: 0,
            hash: Sha256::new(),
            unreachable: None,
        };
        match ml_exp_io::artifacts::verify_data_stream(&mut reader, &state.definition) {
            Ok(()) => Ok(true),
            Err(_) => reader.unreachable.map_or(Ok(false), Err),
        }
    }
    /// Delete exactly one generated immutable part. Missing means the original
    /// deletion succeeded; other errors never advance the durable cursor.
    pub fn release(&self, archive: &ArchiveReceipt, part: u64) -> io::Result<()> {
        let expires = time::OffsetDateTime::now_utc().unix_timestamp() + 300;
        let response = self
            .http
            .delete(self.url("DELETE", archive, part, expires)?)
            .send()
            .map_err(why_chain!(invalid()))?;
        if !matches!(response.status().as_u16(), 204 | 404) {
            return Err(invalid());
        }
        Ok(())
    }
}
struct Reader<'a> {
    store: &'a Store,
    archive: &'a ArchiveReceipt,
    part: u64,
    current: Option<reqwest::blocking::Response>,
    remaining: u64,
}
impl Read for Reader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            if let Some(response) = &mut self.current {
                let limit = buf.len().min(self.remaining.saturating_add(1) as usize);
                let n = response.read(&mut buf[..limit])?;
                if n as u64 > self.remaining {
                    return Err(invalid());
                }
                if n > 0 {
                    self.remaining -= n as u64;
                    return Ok(n);
                }
                if self.remaining != 0 {
                    return Err(invalid());
                }
                self.current = None;
            }
            if self.part >= parts(self.archive)? {
                return Ok(0);
            }
            let expires = time::OffsetDateTime::now_utc().unix_timestamp() + 300;
            let response = self
                .store
                .http
                .get(self.store.url("GET", self.archive, self.part, expires)?)
                .send()
                .map_err(why_chain!(invalid()))?;
            self.remaining = (self.archive.bytes - self.part * PART_BYTES).min(PART_BYTES);
            if response.status() != 200 || response.content_length() != Some(self.remaining) {
                return Err(invalid());
            }
            self.current = Some(response);
            self.part += 1;
        }
    }
}

struct DataReader<'a> {
    store: &'a Store,
    project: &'a str,
    executor: &'a str,
    state: &'a ml_exp_core::datasets::State,
    part: u64,
    current: Option<reqwest::blocking::Response>,
    remaining: u64,
    hash: Sha256,
    /// The read itself failed, so nothing is known about the bytes.
    unreachable: Option<io::Error>,
}
impl DataReader<'_> {
    fn unreachable(&mut self, error: io::Error) -> io::Error {
        let reply = io::Error::new(error.kind(), "object store read failed");
        self.unreachable = Some(error);
        reply
    }
    /// A part's own hash, against its record, as soon as its last byte is read. The
    /// archive hash alone would accept an archive whose records name other bytes.
    fn finish_part(&mut self) -> io::Result<()> {
        let hash = format!(
            "{:x}",
            std::mem::replace(&mut self.hash, Sha256::new()).finalize()
        );
        if self
            .state
            .parts
            .get(&(self.part - 1))
            .map(|r| r.sha256.as_str())
            != Some(hash.as_str())
        {
            return Err(invalid());
        }
        Ok(())
    }
}
impl Read for DataReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            if let Some(response) = &mut self.current {
                let limit = buf.len().min(self.remaining.saturating_add(1) as usize);
                let n = match response.read(&mut buf[..limit]) {
                    Ok(n) => n,
                    Err(error) => return Err(self.unreachable(error)),
                };
                if n as u64 > self.remaining {
                    return Err(invalid());
                }
                if n > 0 {
                    self.hash.update(&buf[..n]);
                    self.remaining -= n as u64;
                    if self.remaining == 0 {
                        self.finish_part()?;
                    }
                    return Ok(n);
                }
                if self.remaining != 0 {
                    let short = invalid();
                    return Err(self.unreachable(short));
                }
                self.current = None;
            }
            if self.part
                >= self
                    .state
                    .definition
                    .part_count()
                    .map_err(why!(invalid()))?
            {
                return Ok(0);
            }
            let key = self.store.data_key(
                self.project,
                self.executor,
                &self.state.definition,
                self.part,
            )?;
            let url = self.store.signed_url(
                "GET",
                &key,
                time::OffsetDateTime::now_utc().unix_timestamp() + 300,
            )?;
            let response = match self.store.http.get(url).send() {
                Ok(response) => response,
                Err(error) => {
                    crate::why::record_chain(file!(), line!(), &error.without_url());
                    let failed = invalid();
                    return Err(self.unreachable(failed));
                }
            };
            self.remaining = self
                .state
                .definition
                .part_size(self.part)
                .map_err(why!(invalid()))?;
            match response.status().as_u16() {
                // An object that is gone or the wrong size is a fact about the bytes.
                404 => return Err(invalid()),
                200 if response.content_length() != Some(self.remaining) => {
                    return Err(invalid());
                }
                200 => {}
                _ => {
                    let failed = invalid();
                    return Err(self.unreachable(failed));
                }
            }
            self.current = Some(response);
            self.part += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_core::datasets::{Definition, State};
    use ml_exp_core::{artifacts::FileReceipt, multipart::PartRecord};
    use std::{
        collections::BTreeMap as Map,
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        sync::{Arc, Mutex},
    };

    /// The object-store behaviour these requests rely on, over a real socket, so that
    /// real reqwest parses each response. Real OSS checks signatures; this does not.
    #[derive(Default)]
    struct Oss {
        objects: Mutex<Map<String, Vec<u8>>>,
        requests: Mutex<Vec<String>>,
    }
    fn serve(oss: Arc<Oss>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for socket in listener.incoming() {
                let mut socket = socket.unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let mut words = line.split_whitespace();
                let method = words.next().unwrap_or_default().to_owned();
                let target = words.next().unwrap_or_default().to_owned();
                let mut length = 0;
                let mut forbid = false;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    let lower = header.trim().to_lowercase();
                    if lower.is_empty() {
                        break;
                    }
                    if let Some(value) = lower.strip_prefix("content-length: ") {
                        length = value.parse().unwrap();
                    }
                    forbid |= lower == "x-oss-forbid-overwrite: true";
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let key = target.split('?').next().unwrap().to_owned();
                oss.requests.lock().unwrap().push(format!("{method} {key}"));
                let mut objects = oss.objects.lock().unwrap();
                let (status, reply, size): (&str, Vec<u8>, usize) = match method.as_str() {
                    "PUT" if forbid && objects.contains_key(&key) => ("409 Conflict", vec![], 0),
                    "PUT" => {
                        objects.insert(key, body);
                        ("200 OK", vec![], 0)
                    }
                    "DELETE" => {
                        objects.remove(&key);
                        ("204 No Content", vec![], 0)
                    }
                    "HEAD" | "GET" => match objects.get(&key) {
                        None => ("404 Not Found", vec![], 0),
                        Some(found) if method == "HEAD" => ("200 OK", vec![], found.len()),
                        Some(found) => ("200 OK", found.clone(), found.len()),
                    },
                    _ => ("400 Bad Request", vec![], 0),
                };
                let _ = write!(
                    socket,
                    "HTTP/1.1 {status}\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n"
                );
                let _ = socket.write_all(&reply);
            }
        });
        origin
    }
    fn store(oss: &Arc<Oss>) -> (tempfile::TempDir, Store) {
        let temp = tempfile::tempdir().unwrap();
        let credentials = temp.path().join("credentials");
        ml_exp_io::write_atomic(
            &credentials,
            br#"{"access_key_id":"id","access_key_secret":"secret"}"#,
            true,
        )
        .unwrap();
        let mut store = Store::open(Settings {
            endpoint: "oss-cn-example.aliyuncs.com".into(),
            bucket: "bucket".into(),
            prefix: "users/alice/ml-expd/".into(),
            credentials_file: credentials,
        })
        .unwrap();
        store.redirect_for_tests(serve(oss.clone()));
        (temp, store)
    }
    /// A tar of one file, with its parts recorded as the client would report them.
    fn asset() -> (Vec<u8>, State) {
        let mut tar = tar::Builder::new(Vec::new());
        let content = vec![7u8; 1024];
        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o600);
        header.set_cksum();
        tar.append_data(&mut header, "train.bin", content.as_slice())
            .unwrap();
        let archive = tar.into_inner().unwrap();
        let definition = Definition {
            sha256: format!("{:x}", Sha256::digest(&archive)),
            archive_bytes: archive.len() as u64,
            files: vec![FileReceipt {
                path: "train.bin".into(),
                sha256: format!("{:x}", Sha256::digest(&content)),
                bytes: content.len() as u64,
            }],
        };
        let target = ml_exp_core::slurm::helper::Target {
            backend_id: "pai".into(),
            profile_sha256: format!("sha256:{}", "3".repeat(64)),
            storage_scope: format!("sha256:{}", "4".repeat(64)),
        };
        let mut state = State::new("demo".into(), target, definition).unwrap();
        state.parts.insert(
            0,
            PartRecord {
                sha256: format!("{:x}", Sha256::digest(&archive)),
                bytes: archive.len() as u64,
            },
        );
        (archive, state)
    }
    fn key(state: &State) -> String {
        format!(
            "/users/alice/ml-expd/datasets/demo/pai/{}/parts/00000",
            state.definition.sha256
        )
    }
    #[test]
    fn a_ticket_names_one_part_of_one_asset_for_a_short_time_and_cannot_overwrite() {
        let oss = Arc::new(Oss::default());
        let (_temp, store) = store(&oss);
        let (_, state) = asset();
        let ticket = store
            .data_upload_ticket("demo", "pai", &state.definition, 0)
            .unwrap();
        let url = url::Url::parse(ticket["url"].as_str().unwrap()).unwrap();
        assert_eq!(url.path(), key(&state));
        assert_eq!(ticket["bytes"], state.definition.archive_bytes);
        assert_eq!(ticket["headers"]["x-oss-forbid-overwrite"], "true");
        assert_eq!(ticket["expires_in_seconds"], 300);
        let query: Map<_, _> = url.query_pairs().into_owned().collect();
        let expires: i64 = query["Expires"].parse().unwrap();
        let now = time::OffsetDateTime::now_utc().unix_timestamp();
        assert!((now..=now + 300).contains(&expires));
        // The method, the key and the overwrite header are all inside the signature.
        let signature = |method: &str, name: &str, headers: &str| {
            let canonical =
                format!("{method}\n\n\n{expires}\n{headers}/bucket/users/alice/ml-expd/{name}");
            let mut mac = Hmac::<Sha1>::new_from_slice(b"secret").unwrap();
            mac.update(canonical.as_bytes());
            STANDARD.encode(mac.finalize().into_bytes())
        };
        let name = format!("datasets/demo/pai/{}/parts/00000", state.definition.sha256);
        let forbid = "x-oss-forbid-overwrite:true\n";
        assert_eq!(query["Signature"], signature("PUT", &name, forbid));
        assert_ne!(query["Signature"], signature("PUT", &name, ""));
        assert_ne!(query["Signature"], signature("GET", &name, ""));
        let other = format!("datasets/other/pai/{}/parts/00000", state.definition.sha256);
        assert_ne!(query["Signature"], signature("PUT", &other, forbid));
        // A part this asset does not have gets no ticket.
        assert!(
            store
                .data_upload_ticket("demo", "pai", &state.definition, 1)
                .is_err()
        );
    }
    #[test]
    fn the_size_of_a_part_is_read_from_a_real_head_without_its_bytes() {
        let oss = Arc::new(Oss::default());
        let (_temp, store) = store(&oss);
        let (archive, state) = asset();
        let size = || store.data_part_size("demo", "pai", &state.definition, 0);
        assert_eq!(size().unwrap(), None);
        oss.objects
            .lock()
            .unwrap()
            .insert(key(&state), archive.clone());
        assert_eq!(size().unwrap(), Some(archive.len() as u64));
        store
            .delete_data("demo", "pai", &state.definition, 0)
            .unwrap();
        assert_eq!(size().unwrap(), None);
    }
    #[test]
    fn only_the_exact_bytes_verify_and_a_wrong_part_is_named() {
        let oss = Arc::new(Oss::default());
        let (_temp, store) = store(&oss);
        let (archive, state) = asset();
        let verify = |state| store.verify_data("demo", "pai", state);
        // Missing.
        assert!(!verify(&state).unwrap());
        assert_eq!(store.wrong_data_parts("demo", "pai", &state).unwrap(), [0]);
        // The right size with other bytes.
        let mut wrong = archive.clone();
        *wrong.last_mut().unwrap() ^= 1;
        oss.objects.lock().unwrap().insert(key(&state), wrong);
        assert!(!verify(&state).unwrap());
        assert_eq!(store.wrong_data_parts("demo", "pai", &state).unwrap(), [0]);
        // The exact bytes.
        oss.objects
            .lock()
            .unwrap()
            .insert(key(&state), archive.clone());
        assert!(verify(&state).unwrap());
        assert!(
            store
                .wrong_data_parts("demo", "pai", &state)
                .unwrap()
                .is_empty()
        );
        // The exact bytes under a record naming others: the record is what a resume trusts.
        let mut lying = state.clone();
        lying.parts.get_mut(&0).unwrap().sha256 = "9".repeat(64);
        assert!(!verify(&lying).unwrap());
        assert_eq!(store.wrong_data_parts("demo", "pai", &lying).unwrap(), [0]);
    }
    #[test]
    fn an_unreachable_store_says_nothing_about_the_bytes() {
        let oss = Arc::new(Oss::default());
        let (_temp, mut store) = store(&oss);
        let (_, state) = asset();
        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", closed.local_addr().unwrap());
        drop(closed);
        store.redirect_for_tests(origin);
        assert!(store.verify_data("demo", "pai", &state).is_err());
        assert!(store.wrong_data_parts("demo", "pai", &state).is_err());
        assert!(
            store
                .data_part_size("demo", "pai", &state.definition, 0)
                .is_err()
        );
    }
    #[test]
    fn an_unrecorded_wrong_object_is_reported_as_different_bytes_not_a_failed_store() {
        let oss = Arc::new(Oss::default());
        let (_temp, store) = store(&oss);
        let (archive, state) = asset();
        let mut wrong = archive.clone();
        wrong[0] ^= 1;
        oss.objects.lock().unwrap().insert(key(&state), wrong);
        let sha = format!("{:x}", Sha256::digest(&archive));
        let error = store
            .put_data("demo", "pai", &state.definition, 0, &sha, &archive)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}
