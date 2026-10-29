//! Private OSS object effects. Large archives are immutable 16MiB objects;
//! the API retains receipts only, never payload files or full archive buffers.
use base64::{Engine, engine::general_purpose::STANDARD};
use hmac::{Hmac, Mac};
use ml_exp_core::artifacts::ArchiveReceipt;
use serde::Deserialize;
use sha1::Sha1;
use std::{
    collections::BTreeMap,
    io::{self, Read},
    path::Path,
    time::Duration,
};
pub const PART_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_PARTS: u64 = 65536;
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
    pub fn open(settings: Settings) -> io::Result<Self> {
        let expected = format!("https://{}.{}", settings.bucket, settings.endpoint);
        let url = url::Url::parse(&expected).map_err(|_| invalid())?;
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
            ml_exp_core::json::mapping(&bytes).map_err(|_| invalid())?,
        ))
        .map_err(|_| invalid())?;
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
            .map_err(|_| invalid())?;
        Ok(Self {
            settings,
            credentials,
            http,
        })
    }
    fn key(&self, archive: &ArchiveReceipt, part: u64) -> io::Result<String> {
        ml_exp_core::artifacts::validate_receipt(archive).map_err(|_| invalid())?;
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
            .map_err(|_| invalid())?;
        mac.update(canonical.as_bytes());
        query.insert("OSSAccessKeyId", self.credentials.access_key_id.clone());
        query.insert("Expires", expires.to_string());
        query.insert("Signature", STANDARD.encode(mac.finalize().into_bytes()));
        let mut url = url::Url::parse(&format!(
            "https://{}.{}/{key}",
            self.settings.bucket, self.settings.endpoint
        ))
        .map_err(|_| invalid())?;
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
            .map_err(|_| invalid())?;
        if part >= definition.part_count().map_err(|_| invalid())? {
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
            serde_json::json!({"part":part,"bytes":definition.part_size(part).map_err(|_|invalid())?,"url":self.signed_url("GET",&key,time::OffsetDateTime::now_utc().unix_timestamp()+300)?}),
        )
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
        use sha2::{Digest, Sha256};
        if definition.part_size(part).map_err(|_| invalid())? != bytes.len() as u64
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
            .map_err(|_| invalid())?;
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
            .map_err(|_| invalid())?;
        if response.status() != 200 || response.content_length() != Some(bytes.len() as u64) {
            return Err(invalid());
        }
        let (hash, size) = ml_exp_io::artifacts::hash(&mut response.take(bytes.len() as u64 + 1))?;
        if hash != sha256 || size != bytes.len() as u64 {
            return Err(invalid());
        }
        Ok(())
    }
    pub fn verify_data(
        &self,
        project: &str,
        executor: &str,
        definition: &ml_exp_core::datasets::Definition,
    ) -> io::Result<()> {
        let mut reader = DataReader {
            store: self,
            project,
            executor,
            definition,
            part: 0,
            current: None,
            remaining: 0,
        };
        ml_exp_io::artifacts::verify_data_stream(&mut reader, definition)
    }
    /// Delete exactly one generated immutable part. Missing means the original
    /// deletion succeeded; other errors never advance the durable cursor.
    pub fn release(&self, archive: &ArchiveReceipt, part: u64) -> io::Result<()> {
        let expires = time::OffsetDateTime::now_utc().unix_timestamp() + 300;
        let response = self
            .http
            .delete(self.url("DELETE", archive, part, expires)?)
            .send()
            .map_err(|_| invalid())?;
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
                .map_err(|_| invalid())?;
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
    definition: &'a ml_exp_core::datasets::Definition,
    part: u64,
    current: Option<reqwest::blocking::Response>,
    remaining: u64,
}
impl Read for DataReader<'_> {
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
            if self.part >= self.definition.part_count().map_err(|_| invalid())? {
                return Ok(0);
            }
            let key =
                self.store
                    .data_key(self.project, self.executor, self.definition, self.part)?;
            let response = self
                .store
                .http
                .get(self.store.signed_url(
                    "GET",
                    &key,
                    time::OffsetDateTime::now_utc().unix_timestamp() + 300,
                )?)
                .send()
                .map_err(|_| invalid())?;
            self.remaining = self
                .definition
                .part_size(self.part)
                .map_err(|_| invalid())?;
            if response.status() != 200 || response.content_length() != Some(self.remaining) {
                return Err(invalid());
            }
            self.current = Some(response);
            self.part += 1;
        }
    }
}
