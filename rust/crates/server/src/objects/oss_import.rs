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
        let mut signed_headers = String::new();
        for (key, value) in &headers {
            use std::fmt::Write;
            writeln!(signed_headers, "{key}:{value}").map_err(|_| invalid())?;
        }
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

    #[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
    };
    #[test]
    fn range_copy_signs_control_requests_and_never_downloads_an_object() {
        let temp = tempfile::tempdir().unwrap();
        let credentials = temp.path().join("credentials");
        ml_exp_io::write_atomic(
            &credentials,
            br#"{"access_key_id":"id","access_key_secret":"secret"}"#,
            true,
        )
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            for expected in ["HEAD", "POST", "PUT", "POST", "HEAD", "HEAD"] {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let words: Vec<_> = line.split_whitespace().collect();
                assert_eq!(words[0], expected);
                let url = url::Url::parse(&format!("http://localhost{}", words[1])).unwrap();
                let query: BTreeMap<String, String> = url
                    .query_pairs()
                    .map(|(k, v)| (k.into_owned(), v.into_owned()))
                    .collect();
                let mut headers = BTreeMap::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    let (key, value) = line.split_once(':').unwrap();
                    headers.insert(key.to_lowercase(), value.trim().to_owned());
                }
                let size = headers
                    .get("content-length")
                    .map(|v| v.parse().unwrap())
                    .unwrap_or(0);
                let mut body = vec![0; size];
                reader.read_exact(&mut body).unwrap();
                let resource_query = query
                    .iter()
                    .filter(|(k, _)| matches!(k.as_str(), "uploads" | "uploadId" | "partNumber"))
                    .map(|(k, v)| {
                        if v.is_empty() {
                            k.clone()
                        } else {
                            format!("{k}={v}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("&");
                let mut signed = String::new();
                for (k, v) in &headers {
                    if k.starts_with("x-oss-") {
                        use std::fmt::Write;
                        writeln!(signed, "{k}:{v}").unwrap();
                    }
                }
                let canonical = format!(
                    "{expected}\n\n\n{}\n{signed}/bucket{}{}",
                    query["Expires"],
                    url.path(),
                    if resource_query.is_empty() {
                        String::new()
                    } else {
                        format!("?{resource_query}")
                    }
                );
                let mut mac = Hmac::<Sha1>::new_from_slice(b"secret").unwrap();
                mac.update(canonical.as_bytes());
                assert_eq!(
                    query["Signature"],
                    STANDARD.encode(mac.finalize().into_bytes())
                );
                let (reply, length, etag) = if expected == "HEAD" {
                    let source = url.path().ends_with("train.tar");
                    (
                        String::new(),
                        if source { PART_BYTES + 1024 } else { 1024 },
                        if source { "source" } else { "destination" },
                    )
                } else if query.contains_key("uploads") {
                    assert!(body.is_empty());
                    ("<InitiateMultipartUploadResult><UploadId>uploadABC</UploadId></InitiateMultipartUploadResult>".into(),0,"")
                } else if expected == "PUT" {
                    assert!(body.is_empty());
                    assert_eq!(
                        headers["x-oss-copy-source-range"],
                        format!("bytes={PART_BYTES}-{}", PART_BYTES + 1023)
                    );
                    assert_eq!(headers["x-oss-copy-source-if-match"], "\"source\"");
                    (
                        "<CopyPartResult><ETag>\"AB12\"</ETag></CopyPartResult>".into(),
                        0,
                        "",
                    )
                } else {
                    assert_eq!(headers["x-oss-forbid-overwrite"], "true");
                    assert!(body.len() < 512);
                    ("<CompleteMultipartUploadResult/>".into(), 0, "")
                };
                let size = if expected == "HEAD" {
                    length
                } else {
                    reply.len() as u64
                };
                // Deliberately no CRC64 header: copied objects need not have one.
                write!(socket,"HTTP/1.1 200 OK\r\nContent-Length: {size}\r\nETag: \"{etag}\"\r\nConnection: close\r\n\r\n{reply}").unwrap();
            }
        });
        let mut store = Store::open(Settings {
            endpoint: "oss-us-southeast-1.aliyuncs.com".into(),
            bucket: "bucket".into(),
            prefix: "users/alice/ml-expd/".into(),
            credentials_file: credentials,
        })
        .unwrap();
        store.redirect_for_tests(origin);
        assert!(store.import_region("us-southeast-1"));
        assert!(!store.import_region("cn-hangzhou"));
        let source = Import {
            object_key: "users/alice/datasets/train.tar".into(),
            sha256: "a".repeat(64),
            archive_bytes: PART_BYTES + 1024,
        };
        let identity = store.import_source_identity(&source).unwrap();
        let id = store.begin_import_part("demo", "pai", &source, 1).unwrap();
        let copied = store
            .finish_import_part("demo", "pai", &source, 1, &identity.etag, &id)
            .unwrap();
        assert_eq!(copied.bytes, 1024);
        assert_eq!(
            store
                .import_part_identity("demo", "pai", &source, 1)
                .unwrap(),
            Some(copied.clone())
        );
        let ticket = store
            .import_read_ticket("demo", "pai", &source, 1, &copied)
            .unwrap();
        assert_eq!(
            url::Url::parse(ticket["url"].as_str().unwrap())
                .unwrap()
                .host_str(),
            Some("bucket.oss-us-southeast-1-internal.aliyuncs.com")
        );
        assert_eq!(ticket["headers"]["If-Match"], copied.etag);
        server.join().unwrap();
    }
}
