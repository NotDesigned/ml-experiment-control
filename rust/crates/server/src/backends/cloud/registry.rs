//! Exact-host private registry credentials, read only at the effect boundary.
use super::{Error, Result};
use base64::Engine;
use ml_exp_core::cloud::{Endpoint, Product, Request};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};
struct Credential {
    username: String,
    password: String,
}
pub(super) struct Auth {
    hosts: BTreeMap<String, Credential>,
}
impl Auth {
    pub(super) fn load(path: &Path) -> Result<Self> {
        let bytes =
            ml_exp_io::read_private(path, 1024 * 1024, true).map_err(|_| Error::configuration())?;
        let document =
            Value::Object(ml_exp_core::json::mapping(&bytes).map_err(|_| Error::configuration())?);
        let auths = document["auths"]
            .as_object()
            .filter(|a| !a.is_empty() && a.len() <= 64)
            .ok_or_else(Error::configuration)?;
        let mut hosts = BTreeMap::new();
        for (raw, value) in auths {
            let host = raw.strip_prefix("https://").unwrap_or(raw);
            if host.is_empty()
                || host.len() > 253
                || !host.contains('.')
                || !host
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || b".-".contains(&c))
            {
                return Err(Error::configuration());
            }
            let auth = value["auth"]
                .as_str()
                .filter(|s| s.len() <= 32768)
                .ok_or_else(Error::configuration)?;
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(auth)
                .map_err(|_| Error::configuration())?;
            let text = std::str::from_utf8(&decoded).map_err(|_| Error::configuration())?;
            let (username, password) = text.split_once(':').ok_or_else(Error::configuration)?;
            if username.is_empty()
                || password.is_empty()
                || text.chars().any(char::is_control)
                || hosts
                    .insert(
                        host.to_owned(),
                        Credential {
                            username: username.into(),
                            password: password.into(),
                        },
                    )
                    .is_some()
            {
                return Err(Error::configuration());
            }
        }
        Ok(Self { hosts })
    }
    pub(super) fn inject(&self, request: &mut Request) -> Result<()> {
        if !matches!(
            request.endpoint,
            Endpoint::Pai {
                product: Product::PaiDlc,
                ..
            }
        ) || request.action != "CreateJob"
        {
            return Ok(());
        }
        let specs = request
            .body
            .as_mut()
            .and_then(|b| b["JobSpecs"].as_array_mut())
            .ok_or_else(Error::configuration)?;
        for spec in specs {
            let host = spec["Image"]
                .as_str()
                .and_then(|s| s.split('/').next())
                .ok_or_else(Error::configuration)?;
            let credential = self.hosts.get(host).ok_or_else(|| Error {
                code: "PAI_IMAGE_AUTH_UNAVAILABLE".into(),
                http_status: None,
                uncertain: false,
            })?;
            if spec.get("ImageConfig").is_some() {
                return Err(Error::configuration());
            }
            spec["ImageConfig"] = json!({"DockerRegistry":host,"Username":credential.username,"Password":credential.password});
        }
        Ok(())
    }
}
