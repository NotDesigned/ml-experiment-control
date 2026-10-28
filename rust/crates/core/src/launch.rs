//! Sealed exact-Attempt launch contract. Validation, encoding and verification
//! are pure; capability loading, HTTPS and process execution belong to shells.
use crate::{
    canonical::python_json_bytes,
    identity::{validate_attempt, validate_id},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const CONTRACT: &str = "launcher-manifest.v1";
pub const MAX_BYTES: usize = 256 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct Error(&'static str);
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub project: String,
    pub run_id: String,
    pub attempt_id: String,
}
// Environment values can contain signed URLs. Do not expose them through Debug.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub contract: String,
    pub identity: Identity,
    pub environment: BTreeMap<String, String>,
    pub argv: Vec<String>,
    pub workdir: String,
    pub timeout_seconds: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub origin: String,
    /// Original path, before any URL-library normalization.
    pub path: String,
    pub identity: Identity,
    pub sha256: String,
}

/// A local sealed manifest has the same identity checks as a downloaded one,
/// without pretending that an HTTPS endpoint was contacted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub identity: Identity,
    pub sha256: String,
}
impl Binding {
    pub fn validate(&self) -> Result<(), Error> {
        validate_identity(&self.identity)?;
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error("invalid sealed manifest checksum"));
        }
        Ok(())
    }
}

fn exact_keys(value: &Value, expected: &[&str]) -> bool {
    value
        .as_object()
        .is_some_and(|v| v.len() == expected.len() && expected.iter().all(|k| v.contains_key(*k)))
}
fn environment_key(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
        && bytes.all(|c| c.is_ascii_alphanumeric() || c == b'_')
}
fn validate_identity(value: &Identity) -> Result<(), Error> {
    if validate_id(&value.project).is_err()
        || validate_id(&value.run_id).is_err()
        || validate_attempt(&value.attempt_id).is_err()
    {
        return Err(Error("invalid launch manifest identity"));
    }
    Ok(())
}

pub fn encode(document: &Value) -> Result<Vec<u8>, Error> {
    if !exact_keys(
        document,
        &[
            "contract",
            "identity",
            "environment",
            "argv",
            "workdir",
            "timeout_seconds",
        ],
    ) || document["contract"] != CONTRACT
    {
        return Err(Error("invalid launch manifest contract"));
    }
    if !exact_keys(&document["identity"], &["project", "run_id", "attempt_id"]) {
        return Err(Error("invalid launch manifest identity"));
    }
    let identity: Identity = serde_json::from_value(document["identity"].clone())
        .map_err(|_| Error("invalid launch manifest identity"))?;
    validate_identity(&identity)?;
    let environment = document["environment"]
        .as_object()
        .ok_or(Error("invalid launch manifest environment"))?;
    if environment.iter().any(|(key, value)| {
        !environment_key(key)
            || value.as_str().is_none_or(|v| v.contains('\0'))
            || matches!(
                key.as_str(),
                "ML_EXPD_UPLOAD_TOKEN" | "ML_EXPD_BOOTSTRAP_TOKEN"
            )
    }) || !environment
        .get("OUTPUT_DIR")
        .and_then(Value::as_str)
        .is_some_and(|v| v.starts_with('/'))
    {
        return Err(Error("invalid launch manifest environment"));
    }
    if !document["argv"].as_array().is_some_and(|args| {
        !args.is_empty()
            && args
                .first()
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
            && args
                .iter()
                .all(|arg| arg.as_str().is_some_and(|s| !s.contains('\0')))
    }) {
        return Err(Error("invalid launch manifest argv"));
    }
    if !document["workdir"]
        .as_str()
        .is_some_and(|v| v.starts_with('/') && !v.contains('\0'))
    {
        return Err(Error("invalid launch manifest workdir"));
    }
    if !document["timeout_seconds"]
        .as_number()
        .is_some_and(|n| n.is_u64() && n.as_u64().is_some_and(|n| (1..1000 * 3600).contains(&n)))
    {
        return Err(Error("invalid launch manifest timeout"));
    }
    let body =
        python_json_bytes(document).map_err(|_| Error("invalid launch manifest encoding"))?;
    if body.len() > MAX_BYTES {
        return Err(Error("launch manifest exceeds size limit"));
    }
    Ok(body)
}

impl Manifest {
    pub fn from_value(document: &Value) -> Result<Self, Error> {
        encode(document)?;
        serde_json::from_value(document.clone())
            .map_err(|_| Error("invalid launch manifest contract"))
    }
    pub fn encode(&self) -> Result<Vec<u8>, Error> {
        encode(&serde_json::to_value(self).map_err(|_| Error("invalid launch manifest encoding"))?)
    }
}

/// Refuse userinfo, query, fragment, escaped paths and URL normalization. This
/// stricter transport policy never changes a frozen manifest's hash inputs.
pub fn endpoint(value: &str) -> Result<Endpoint, Error> {
    let invalid = || Error("launch manifest requires an exact HTTPS endpoint");
    if value.chars().any(|c| c.is_control() || c.is_whitespace())
        || value.contains(['?', '#', '\\'])
    {
        return Err(invalid());
    }
    let (_, rest) = value.split_once("://").ok_or_else(invalid)?;
    let (authority, suffix) = rest.split_once('/').ok_or_else(invalid)?;
    if authority.contains('@') {
        return Err(invalid());
    }
    let target = url::Url::parse(value).map_err(|_| invalid())?;
    if target.scheme() != "https"
        || target.host_str().is_none()
        || !target.username().is_empty()
        || target.password().is_some()
    {
        return Err(invalid());
    }
    let path = format!("/{suffix}");
    let parts: Vec<_> = suffix.split('/').collect();
    if parts.len() < 6 {
        return Err(invalid());
    }
    let tail = &parts[parts.len() - 6..];
    if tail[0] != "api"
        || tail[1] != "launch-transfers"
        || parts[..parts.len() - 6]
            .iter()
            .any(|p| validate_id(p).is_err())
    {
        return Err(invalid());
    }
    let identity = Identity {
        project: tail[2].into(),
        run_id: tail[3].into(),
        attempt_id: tail[4].into(),
    };
    validate_identity(&identity).map_err(|_| invalid())?;
    if tail[5].len() != 64
        || !tail[5]
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid());
    }
    if target.path() != path {
        return Err(invalid());
    }
    Ok(Endpoint {
        origin: target.origin().ascii_serialization(),
        path,
        identity,
        sha256: tail[5].into(),
    })
}

pub fn validate_capability(token: &str) -> Result<(), Error> {
    if token.is_empty()
        || token.len() > 256
        || !token
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
    {
        Err(Error("invalid launch capability"))
    } else {
        Ok(())
    }
}

pub fn verify(body: &[u8], endpoint: &Endpoint) -> Result<Manifest, Error> {
    verify_binding(
        body,
        &Binding {
            identity: endpoint.identity.clone(),
            sha256: endpoint.sha256.clone(),
        },
    )
}

pub fn verify_binding(body: &[u8], binding: &Binding) -> Result<Manifest, Error> {
    binding.validate()?;
    if body.len() > MAX_BYTES {
        return Err(Error("launch manifest exceeds size limit"));
    }
    if format!("{:x}", Sha256::digest(body)) != binding.sha256 {
        return Err(Error("launch manifest checksum differs"));
    }
    let document: Value =
        serde_json::from_slice(body).map_err(|_| Error("invalid launch manifest encoding"))?;
    if encode(&document)? != body {
        return Err(Error("launch manifest identity or encoding differs"));
    }
    let manifest = Manifest::from_value(&document)?;
    if manifest.identity != binding.identity {
        return Err(Error("launch manifest identity or encoding differs"));
    }
    Ok(manifest)
}
