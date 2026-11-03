//! Original S3 downloads need a short GET capability, not a cache restore or a
//! new native publication. Signing is pure; the only effect is a pinned profile read.
use crate::read_config;
use hmac::{Hmac, Mac};
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};
use time::OffsetDateTime;

const ENCODE: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'!')
    .add(b'"')
    .add(b'#')
    .add(b'$')
    .add(b'%')
    .add(b'&')
    .add(b'\'')
    .add(b'(')
    .add(b')')
    .add(b'*')
    .add(b'+')
    .add(b',')
    .add(b'/')
    .add(b':')
    .add(b';')
    .add(b'<')
    .add(b'=')
    .add(b'>')
    .add(b'?')
    .add(b'@')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectConfig {
    pub profile_file: PathBuf,
    pub profile_sha256: String,
}
impl ObjectConfig {
    pub(super) fn validate(&self) -> Result<(), read_config::Error> {
        if !self.profile_file.is_absolute()
            || self.profile_file.components().any(|c| {
                matches!(
                    c,
                    std::path::Component::ParentDir | std::path::Component::CurDir
                )
            })
            || ml_exp_core::sources::validate_archive_digest(&self.profile_sha256).is_err()
        {
            return Err(read_config::Error);
        }
        Ok(())
    }
    pub(crate) fn ticket(&self, inventory: &Value, now: i64) -> Result<Value, read_config::Error> {
        self.validate()?;
        let raw = ml_exp_io::read_private(&self.profile_file, 64 * 1024, true)
            .map_err(|_| read_config::Error)?;
        if format!("{:x}", Sha256::digest(&raw)) != self.profile_sha256 {
            return Err(read_config::Error);
        }
        let profile =
            Value::Object(ml_exp_core::json::mapping(&raw).map_err(|_| read_config::Error)?);
        sign(&profile, inventory, now)
    }
}

fn encoded(value: &str) -> String {
    utf8_percent_encode(value, ENCODE).to_string()
}
fn mac(key: &[u8], value: &str) -> Result<hmac::digest::Output<Hmac<Sha256>>, read_config::Error> {
    let mut h = Hmac::<Sha256>::new_from_slice(key).map_err(|_| read_config::Error)?;
    h.update(value.as_bytes());
    Ok(h.finalize().into_bytes())
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, read_config::Error> {
    value[key].as_str().ok_or(read_config::Error)
}
fn component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        && !matches!(value, "." | "..")
}

/// No network, clock lookup, payload access, acknowledgement or deletion.
fn sign(profile: &Value, inventory: &Value, now: i64) -> Result<Value, read_config::Error> {
    if inventory["available"] != true
        || inventory["storage_status"] != "AVAILABLE"
        || inventory["source"] != "historical-published-receipt"
        || inventory["read_only"] != true
    {
        return Err(read_config::Error);
    }
    let project = text(inventory, "project")?;
    let run = text(inventory, "run_id")?;
    let attempt = text(inventory, "attempt_id")?;
    ml_exp_core::identity::validate_id(project).map_err(|_| read_config::Error)?;
    ml_exp_core::identity::validate_observation_id(run).map_err(|_| read_config::Error)?;
    ml_exp_core::identity::validate_attempt(attempt).map_err(|_| read_config::Error)?;
    let sha = text(&inventory["archive"], "sha256")?;
    ml_exp_core::sources::validate_archive_digest(sha).map_err(|_| read_config::Error)?;
    let bytes = inventory["archive"]["bytes"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or(read_config::Error)?;
    let endpoint = text(profile, "public_endpoint")?;
    let url = url::Url::parse(endpoint).map_err(|_| read_config::Error)?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || !matches!(url.path(), "" | "/")
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(read_config::Error);
    }
    let bucket = text(profile, "bucket")?;
    let region = profile
        .get("region")
        .map(|_| text(profile, "region"))
        .transpose()?
        .unwrap_or("garage");
    let key = text(profile, "access_key")?;
    let secret = text(profile, "secret_key")?;
    if !component(bucket)
        || !component(region)
        || !component(key)
        || secret.len() < 8
        || secret.len() > 512
        || !secret.bytes().all(|b| b.is_ascii_graphic())
    {
        return Err(read_config::Error);
    }
    let expires = profile
        .get("download_url_seconds")
        .map(|v| v.as_i64().ok_or(read_config::Error))
        .transpose()?
        .unwrap_or(300);
    if !(30..=3600).contains(&expires) {
        return Err(read_config::Error);
    }
    let at = OffsetDateTime::from_unix_timestamp(now).map_err(|_| read_config::Error)?;
    let date = format!("{:04}{:02}{:02}", at.year(), u8::from(at.month()), at.day());
    let stamp = format!(
        "{date}T{:02}{:02}{:02}Z",
        at.hour(),
        at.minute(),
        at.second()
    );
    let scope = format!("{date}/{region}/s3/aws4_request");
    let mut query = BTreeMap::from([
        ("X-Amz-Algorithm", "AWS4-HMAC-SHA256".to_owned()),
        ("X-Amz-Credential", format!("{key}/{scope}")),
        ("X-Amz-Date", stamp.clone()),
        ("X-Amz-Expires", expires.to_string()),
        ("X-Amz-SignedHeaders", "host".to_owned()),
    ]);
    let query_string = query
        .iter()
        .map(|(k, v)| format!("{}={}", encoded(k), encoded(v)))
        .collect::<Vec<_>>()
        .join("&");
    let uri = format!(
        "/{}/{}/{}/{}/{}.tar",
        encoded(bucket),
        encoded(project),
        encoded(run),
        encoded(attempt),
        sha
    );
    let host = &url[url::Position::BeforeHost..url::Position::AfterPort];
    let canonical = format!("GET\n{uri}\n{query_string}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD");
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{:x}",
        Sha256::digest(canonical.as_bytes())
    );
    let signing_key = mac(
        &mac(
            &mac(&mac(format!("AWS4{secret}").as_bytes(), &date)?, region)?,
            "s3",
        )?,
        "aws4_request",
    )?;
    let signature = format!("{:x}", mac(&signing_key, &to_sign)?);
    query.insert("X-Amz-Signature", signature);
    let query = query
        .iter()
        .map(|(k, v)| format!("{}={}", encoded(k), encoded(v)))
        .collect::<Vec<_>>()
        .join("&");
    let expires_at =
        OffsetDateTime::from_unix_timestamp(now.checked_add(expires).ok_or(read_config::Error)?)
            .map_err(|_| read_config::Error)?
            .format(&time::format_description::well_known::Rfc3339)
            .map_err(|_| read_config::Error)?;
    Ok(
        json!({"transport":"s3-presigned-get","url":format!("{}{uri}?{query}", endpoint.trim_end_matches('/')),
        "sha256":sha,"bytes":bytes,"expires_at":expires_at,"files":inventory["files"],
        "checksums_complete":inventory["checksums_complete"],"read_only":true,"acknowledgement_sent":false,
        "acknowledgement_supported":false,"payload_verified":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Value, Value) {
        (
            json!({"public_endpoint":"https://example.com","region":"garage","bucket":"results","access_key":"fixture-key","secret_key":"fixture-secret"}),
            json!({"project":"demo","run_id":"train","attempt_id":"attempt-001","available":true,"storage_status":"AVAILABLE","source":"historical-published-receipt","read_only":true,
             "archive":{"sha256":"a".repeat(64),"bytes":10240},"files":[{"path":"outputs/model.pt","bytes":128,"sha256":null}],"checksums_complete":false}),
        )
    }
    #[test]
    fn exact_get_signing_preserves_unknown_hashes_and_has_no_deletion_capability() {
        let (profile, inventory) = fixture();
        let ticket = sign(&profile, &inventory, 1793232000).unwrap();
        let url = url::Url::parse(ticket["url"].as_str().unwrap()).unwrap();
        let query: BTreeMap<_, _> = url.query_pairs().collect();
        assert_eq!(
            url.path(),
            format!("/results/demo/train/attempt-001/{}.tar", "a".repeat(64))
        );
        assert_eq!(query["X-Amz-Expires"], "300");
        // Independently generated with botocore S3SigV4QueryAuth and fixed time.
        assert_eq!(
            query["X-Amz-Signature"],
            "6965bd632c1b522bc3f3dcdfa980e508ecb92541734ddad884aac619e2a1256d"
        );
        assert_eq!(ticket["files"][0]["sha256"], Value::Null);
        assert_eq!(ticket["acknowledgement_supported"], false);
        assert!(!ticket.to_string().contains("fixture-secret"));
        assert_eq!(ticket, sign(&profile, &inventory, 1793232000).unwrap());
    }
    #[test]
    fn missing_released_expired_and_untrusted_profiles_cannot_issue_a_ticket() {
        let (profile, inventory) = fixture();
        for status in ["MISSING", "RELEASED", "EXPIRED"] {
            let mut value = inventory.clone();
            value["storage_status"] = json!(status);
            assert!(sign(&profile, &value, 1793232000).is_err());
        }
        for endpoint in [
            "http://example.com",
            "https://user@example.com",
            "https://example.com/private",
            "https://example.com/?key=secret",
            "https://example.com/#fragment",
        ] {
            let mut profile = profile.clone();
            profile["public_endpoint"] = json!(endpoint);
            assert!(sign(&profile, &inventory, 1793232000).is_err());
        }
    }
}
