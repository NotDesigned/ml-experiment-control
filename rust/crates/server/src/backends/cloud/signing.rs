//! Pure request signing. Inputs are explicit; no credential, time or entropy I/O.
use super::{Error, Result};
use base64::Engine;
use hmac::{Hmac, Mac};
use ml_exp_core::cloud::{Endpoint, Method, Request};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
pub(super) struct Signed {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}
pub(super) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, b| {
        write!(out, "{b:02x}").expect("String write");
        out
    })
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn mac(secret: &str, bytes: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
        .expect("HMAC accepts arbitrary key sizes");
    mac.update(bytes);
    mac.finalize().into_bytes().to_vec()
}
pub(super) fn encode(value: &str) -> String {
    let mut out = String::new();
    for c in value.bytes() {
        if c.is_ascii_alphanumeric() || b"-_.~".contains(&c) {
            out.push(c as char);
        } else {
            use std::fmt::Write;
            write!(out, "%{c:02X}").expect("String write");
        }
    }
    out
}
fn method(value: Method) -> &'static str {
    match value {
        Method::Get => "GET",
        Method::Post => "POST",
        Method::Delete => "DELETE",
        Method::Put => "PUT",
    }
}
pub(super) fn sign(
    request: &Request,
    id: &str,
    secret: &str,
    security_token: Option<&str>,
    date: &str,
    nonce: &str,
) -> Result<Signed> {
    request.validate().map_err(|_| Error::configuration())?;
    if id.is_empty()
        || id.len() > 256
        || !id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
        || secret.is_empty()
        || secret.len() > 4096
        || secret.chars().any(char::is_control)
        || date.len() > 64
        || date.chars().any(char::is_control)
        || security_token
            .is_some_and(|s| s.is_empty() || s.len() > 16384 || s.chars().any(char::is_control))
    {
        return Err(Error::configuration());
    }
    let host = request
        .endpoint
        .host()
        .map_err(|_| Error::configuration())?;
    let body = request
        .body
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()
        .map_err(|_| Error::configuration())?
        .unwrap_or_default();
    let query = request
        .query
        .iter()
        .map(|(k, v)| format!("{}={}", encode(k), encode(v)))
        .collect::<Vec<_>>()
        .join("&");
    let path = request
        .path
        .split('/')
        .map(encode)
        .collect::<Vec<_>>()
        .join("/");
    let mut headers: BTreeMap<String, String> =
        BTreeMap::from([("content-type".into(), "application/json".into())]);
    match &request.endpoint {
        Endpoint::Pai { product, .. } => {
            if !valid_acs_date(date)
                || nonce.is_empty()
                || nonce.len() > 128
                || !nonce
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
            {
                return Err(Error::configuration());
            }
            headers.extend([
                ("host".into(), host.clone()),
                ("x-acs-action".into(), request.action.clone()),
                ("x-acs-version".into(), product.version().into()),
                ("x-acs-content-sha256".into(), sha(&body)),
                ("x-acs-date".into(), date.into()),
                ("x-acs-signature-nonce".into(), nonce.into()),
            ]);
            if let Some(token) = security_token {
                headers.insert("x-acs-security-token".into(), token.into());
            }
            let signed = headers
                .keys()
                .filter(|k| k.as_str() == "host" || k.starts_with("x-acs-"))
                .cloned()
                .collect::<Vec<_>>();
            let names = signed.join(";");
            let canonical_headers = signed.iter().fold(String::new(), |mut out, key| {
                use std::fmt::Write;
                writeln!(out, "{key}:{}", headers[key].trim()).expect("String write");
                out
            });
            let canonical = format!(
                "{}\n{path}\n{query}\n{canonical_headers}\n{names}\n{}",
                method(request.method),
                sha(&body)
            );
            let to_sign = format!("ACS3-HMAC-SHA256\n{}", sha(canonical.as_bytes()));
            let signature = hex(&mac(secret, to_sign.as_bytes()));
            headers.insert(
                "authorization".into(),
                format!(
                    "ACS3-HMAC-SHA256 Credential={id},SignedHeaders={names},Signature={signature}"
                ),
            );
        }
        Endpoint::Sensecore { .. } => {
            if date.len() != 29 || !date.ends_with(" GMT") || !date.is_ascii() {
                return Err(Error::configuration());
            }
            let signature = base64::engine::general_purpose::STANDARD
                .encode(mac(secret, format!("x-date: {date}").as_bytes()));
            headers.insert("x-date".into(), date.into());
            headers.insert("authorization".into(),format!("hmac accesskey=\"{id}\", algorithm=\"hmac-sha256\", headers=\"x-date\", signature=\"{signature}\""));
        }
    }
    Ok(Signed {
        url: format!(
            "https://{host}{path}{}",
            if query.is_empty() {
                String::new()
            } else {
                format!("?{query}")
            }
        ),
        headers,
        body,
    })
}
fn valid_acs_date(value: &str) -> bool {
    value.len() == 20
        && value.ends_with('Z')
        && time::OffsetDateTime::parse(value, &time::format_description::well_known::Rfc3339)
            .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_core::cloud::{Endpoint, Product};
    #[test]
    fn exact_signatures_bind_body_query_identity_and_never_accept_header_injection() {
        let request = Request {
            endpoint: Endpoint::Pai {
                product: Product::PaiDlc,
                region: "us-southeast-1".into(),
            },
            method: Method::Get,
            action: "GetJob".into(),
            path: "/api/v1/jobs/dlc123".into(),
            query: BTreeMap::from([("Tags".into(), "a b/+".into())]),
            body: None,
            read_only: true,
        };
        let signed = sign(
            &request,
            "exampleID",
            "exampleSecret",
            None,
            "2026-10-29T00:00:00Z",
            "nonce-123",
        )
        .unwrap();
        assert!(signed.url.ends_with("?Tags=a%20b%2F%2B"));
        assert_eq!(
            signed.headers["authorization"],
            "ACS3-HMAC-SHA256 Credential=exampleID,SignedHeaders=host;x-acs-action;x-acs-content-sha256;x-acs-date;x-acs-signature-nonce;x-acs-version,Signature=480c0fdc4c5f4039c883c36bb2a03ee979232c0b89cda03a4049330e0eaf38ac"
        );
        assert!(
            sign(
                &request,
                "injected\r\nHost",
                "s",
                None,
                "2026-10-29T00:00:00Z",
                "n"
            )
            .is_err()
        );
        let mut sense = request;
        sense.endpoint = Endpoint::Sensecore {
            service: "iam".into(),
            region: None,
        };
        let signed = sign(
            &sense,
            "exampleID",
            "exampleSecret",
            None,
            "Thu, 29 Oct 2026 00:00:00 GMT",
            "",
        )
        .unwrap();
        assert_eq!(
            signed.headers["authorization"],
            "hmac accesskey=\"exampleID\", algorithm=\"hmac-sha256\", headers=\"x-date\", signature=\"r9nHTTD24XOH89hH4/jhHXEZaPgy6wEmGpVZqf7u2B8=\""
        );
    }
}
