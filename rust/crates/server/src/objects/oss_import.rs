//! Metadata-only OSS range copies. No object body is read by the API host.
use super::*;
use ml_exp_core::datasets::oss::Import;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectIdentity {
    pub bytes: u64,
    pub etag: String,
}

fn xml_field(text: &str, name: &str) -> io::Result<String> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let value = text
        .split_once(&open)
        .and_then(|(_, rest)| rest.split_once(&close))
        .map(|(v, _)| v.trim())
        .ok_or_else(invalid)?;
    if value.is_empty() || value.len() > 256 || value.contains(['<', '>', '&']) {
        return Err(invalid());
    }
    Ok(value.into())
}

impl Store {
    /// OSS and the verifier must share a region; only verifier GET tickets use
    /// the internal endpoint. The control plane can remain outside that region.
    pub fn import_region(&self, region: &str) -> bool {
        self.settings.endpoint == format!("oss-{region}.aliyuncs.com")
            || self.settings.endpoint == format!("oss-{region}-internal.aliyuncs.com")
    }

    fn import_key(
        &self,
        project: &str,
        executor: &str,
        source: &Import,
        part: u64,
    ) -> io::Result<String> {
        self.validate_import(source)?;
        ml_exp_core::identity::validate_id(project)
            .and_then(|_| ml_exp_core::identity::validate_id(executor))
            .map_err(|_| invalid())?;
        ml_exp_core::multipart::part_size(source.archive_bytes, part).map_err(|_| invalid())?;
        Ok(format!(
            "{}datasets/{project}/{executor}/{}/parts/{part:05}",
            self.settings.prefix, source.sha256
        ))
    }

    fn import_request(
        &self,
        method: &str,
        key: &str,
        mut query: BTreeMap<String, String>,
        headers: BTreeMap<String, String>,
        body: Vec<u8>,
    ) -> io::Result<reqwest::blocking::Response> {
        if let Some(token) = &self.credentials.security_token {
            query.insert("security-token".into(), token.clone());
        }
        let resource = format!(
            "/{}/{}{}",
            self.settings.bucket,
            key,
            if query.is_empty() {
                String::new()
            } else {
                format!(
                    "?{}",
                    query
                        .iter()
                        .map(|(k, v)| if v.is_empty() {
                            k.clone()
                        } else {
                            format!("{k}={v}")
                        })
                        .collect::<Vec<_>>()
                        .join("&")
                )
            }
        );
        let expires = time::OffsetDateTime::now_utc().unix_timestamp() + 300;
        let signed_headers = headers
            .iter()
            .map(|(k, v)| format!("{k}:{v}\n"))
            .collect::<String>();
        let canonical = format!("{method}\n\n\n{expires}\n{signed_headers}{resource}");
        let mut mac = Hmac::<Sha1>::new_from_slice(self.credentials.access_key_secret.as_bytes())
            .map_err(|_| invalid())?;
        mac.update(canonical.as_bytes());
        query.insert(
            "OSSAccessKeyId".into(),
            self.credentials.access_key_id.clone(),
        );
        query.insert("Expires".into(), expires.to_string());
        query.insert(
            "Signature".into(),
            STANDARD.encode(mac.finalize().into_bytes()),
        );
        let mut url = url::Url::parse(&format!("{}/{key}", self.origin)).map_err(|_| invalid())?;
        url.query_pairs_mut().extend_pairs(query);
        let mut request = self
            .http
            .request(
                reqwest::Method::from_bytes(method.as_bytes()).map_err(|_| invalid())?,
                url,
            )
            .timeout(Duration::from_secs(20));
        for (k, v) in headers {
            request = request.header(k, v);
        }
        request.body(body).send().map_err(|_| invalid())
    }

    fn import_head(&self, key: &str) -> io::Result<Option<ObjectIdentity>> {
        let response =
            self.import_request("HEAD", key, BTreeMap::new(), BTreeMap::new(), Vec::new())?;
        match response.status().as_u16() {
            404 => Ok(None),
            200 => {
                let etag = response
                    .headers()
                    .get("etag")
                    .and_then(|v| v.to_str().ok())
                    .filter(|v| !v.is_empty() && v.len() <= 256 && !v.chars().any(char::is_control))
                    .ok_or_else(invalid)?
                    .to_owned();
                let bytes = response
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(invalid)?;
                Ok(Some(ObjectIdentity { bytes, etag }))
            }
            _ => Err(invalid()),
        }
    }

    pub fn import_source_identity(&self, source: &Import) -> io::Result<ObjectIdentity> {
        self.validate_import(source)?;
        self.import_head(&source.object_key)?
            .filter(|m| m.bytes == source.archive_bytes)
            .ok_or_else(invalid)
    }

    pub fn import_part_identity(
        &self,
        project: &str,
        executor: &str,
        source: &Import,
        part: u64,
    ) -> io::Result<Option<ObjectIdentity>> {
        self.import_head(&self.import_key(project, executor, source, part)?)
    }

    pub fn begin_import_part(
        &self,
        project: &str,
        executor: &str,
        source: &Import,
        part: u64,
    ) -> io::Result<String> {
        let key = self.import_key(project, executor, source, part)?;
        let response = self.import_request(
            "POST",
            &key,
            BTreeMap::from([("uploads".into(), String::new())]),
            BTreeMap::new(),
            Vec::new(),
        )?;
        if response.status() != 200 {
            return Err(invalid());
        }
        let mut text = String::new();
        response.take(16385).read_to_string(&mut text)?;
        if text.len() > 16384 {
            return Err(invalid());
        }
        let id = xml_field(&text, "UploadId")?;
        if !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
        {
            return Err(invalid());
        }
        Ok(id)
    }

    pub fn finish_import_part(
        &self,
        project: &str,
        executor: &str,
        source: &Import,
        part: u64,
        source_etag: &str,
        upload_id: &str,
    ) -> io::Result<ObjectIdentity> {
        let key = self.import_key(project, executor, source, part)?;
        let size =
            ml_exp_core::multipart::part_size(source.archive_bytes, part).map_err(|_| invalid())?;
        let first = part * PART_BYTES;
        let response = self.import_request(
            "PUT",
            &key,
            BTreeMap::from([
                ("partNumber".into(), "1".into()),
                ("uploadId".into(), upload_id.into()),
            ]),
            BTreeMap::from([
                (
                    "x-oss-copy-source".into(),
                    format!("/{}/{}", self.settings.bucket, source.object_key),
                ),
                ("x-oss-copy-source-if-match".into(), source_etag.into()),
                (
                    "x-oss-copy-source-range".into(),
                    format!("bytes={first}-{}", first + size - 1),
                ),
            ]),
            Vec::new(),
        )?;
        if response.status() != 200 {
            return Err(invalid());
        }
        let mut text = String::new();
        response.take(16385).read_to_string(&mut text)?;
        if text.len() > 16384 {
            return Err(invalid());
        }
        let etag = xml_field(&text, "ETag")?;
        if !etag
            .trim_matches('"')
            .bytes()
            .all(|b| b.is_ascii_hexdigit() || b == b'-')
        {
            return Err(invalid());
        }
        let body = format!(
            "<CompleteMultipartUpload><Part><PartNumber>1</PartNumber><ETag>{etag}</ETag></Part></CompleteMultipartUpload>"
        );
        let response = self.import_request(
            "POST",
            &key,
            BTreeMap::from([("uploadId".into(), upload_id.into())]),
            BTreeMap::from([("x-oss-forbid-overwrite".into(), "true".into())]),
            body.into_bytes(),
        )?;
        if !matches!(response.status().as_u16(), 200 | 409) {
            return Err(invalid());
        }
        // A 200 may contain an XML Error. Only a subsequent HEAD is evidence of
        // the committed destination. A lost completion response is recovered by
        // the same HEAD at the beginning of the next visit.
        self.import_head(&key)?
            .filter(|m| m.bytes == size)
            .ok_or_else(invalid)
    }

    pub fn import_read_ticket(
        &self,
        project: &str,
        executor: &str,
        source: &Import,
        part: u64,
        identity: &ObjectIdentity,
    ) -> io::Result<serde_json::Value> {
        let key = self.import_key(project, executor, source, part)?;
        let mut url = url::Url::parse(&self.signed_url(
            "GET",
            &key,
            time::OffsetDateTime::now_utc().unix_timestamp() + 300,
        )?)
        .map_err(|_| invalid())?;
        // The signature binds the bucket/key, not the HTTP Host. The same OSS
        // resource is reachable on its regional internal endpoint.
        let endpoint = self.settings.endpoint.trim_end_matches(".aliyuncs.com");
        let internal = if endpoint.ends_with("-internal") {
            endpoint.to_owned()
        } else {
            format!("{endpoint}-internal")
        };
        url.set_host(Some(&format!(
            "{}.{}.aliyuncs.com",
            self.settings.bucket, internal
        )))
        .map_err(|_| invalid())?;
        Ok(
            serde_json::json!({"part":part,"bytes":identity.bytes,"etag":identity.etag,"url":url.as_str(),"headers":{"If-Match":identity.etag}}),
        )
    }
}
