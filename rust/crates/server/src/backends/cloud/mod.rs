//! Native signed REST effect shell. One HTTP request, no redirects, environment
//! proxy, credential logging, implicit mutation retry or provider CLI.
pub mod action;
pub mod adapter;
mod evidence;
pub mod outbox;
mod registry;
mod signing;
pub use evidence::FailureEvidence;
use ml_exp_core::cloud::{Endpoint, Method, Request};
use serde::Deserialize;
use serde_json::Value;
use std::{
    io::Read,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
const LIMIT: u64 = 32 * 1024 * 1024;
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("cloud REST request failed ({code}); HTTP {http_status:?}; outcome uncertain: {uncertain}")]
pub struct Error {
    pub code: String,
    pub http_status: Option<u16>,
    pub uncertain: bool,
}
pub type Result<T> = std::result::Result<T, Error>;
impl Error {
    pub fn configuration() -> Self {
        Self {
            code: "CLOUD_CONFIGURATION_INVALID".into(),
            http_status: None,
            uncertain: false,
        }
    }
    pub fn evidence() -> Self {
        Self {
            code: "CLOUD_EVIDENCE_INVALID".into(),
            http_status: None,
            uncertain: false,
        }
    }
    fn transport(uncertain: bool) -> Self {
        Self {
            code: "CLOUD_RESPONSE_UNAVAILABLE".into(),
            http_status: None,
            uncertain,
        }
    }
}
// Intentionally no Debug, Serialize or public credential fields.
#[derive(Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
pub enum Credentials {
    Pai {
        access_key_id: String,
        access_key_secret: String,
        #[serde(default)]
        security_token: Option<String>,
    },
    Sensecore {
        access_key_id: String,
        access_key_secret: String,
    },
}
impl Credentials {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            ml_exp_io::read_private(path, 32768, true).map_err(|_| Error::configuration())?;
        let value: Self = serde_json::from_value(Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::configuration())?,
        ))
        .map_err(|_| Error::configuration())?;
        let (id, secret, token) = match &value {
            Self::Pai {
                access_key_id,
                access_key_secret,
                security_token,
            } => (access_key_id, access_key_secret, security_token.as_deref()),
            Self::Sensecore {
                access_key_id,
                access_key_secret,
            } => (access_key_id, access_key_secret, None),
        };
        if std::iter::once(id.as_str())
            .chain(std::iter::once(secret.as_str()))
            .chain(token)
            .any(|s| s.is_empty() || s.len() > 16384 || s.chars().any(char::is_control))
        {
            return Err(Error::configuration());
        }
        Ok(value)
    }
    pub(crate) fn matches_profile(&self, profile: &adapter::Profile) -> bool {
        matches!(
            (self, profile),
            (Self::Pai { .. }, adapter::Profile::Pai(_))
                | (Self::Sensecore { .. }, adapter::Profile::Sensecore(_))
        )
    }
}
pub struct Client {
    credentials: Credentials,
    http: reqwest::blocking::Client,
    timeout: Duration,
    registry: Option<registry::Auth>,
}
impl Client {
    pub fn new(credentials: Credentials, timeout: Duration) -> Result<Self> {
        if timeout < Duration::from_secs(1) || timeout > Duration::from_secs(120) {
            return Err(Error::configuration());
        }
        let http = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(Duration::from_secs(5))
            .timeout(timeout)
            .build()
            .map_err(|_| Error::configuration())?;
        Ok(Self {
            credentials,
            http,
            timeout,
            registry: None,
        })
    }
    /// Registry passwords are injected only into the signed request copy, never
    /// into the frozen plan, Run, private outbox or outward observation.
    pub fn with_registry_auth(mut self, path: &Path) -> Result<Self> {
        self.registry = Some(registry::Auth::load(path)?);
        Ok(self)
    }
    pub fn call(&self, request: &Request, cancelled: &AtomicBool) -> Result<Value> {
        self.call_observed(request, cancelled, &mut |_| {})
    }
    pub(crate) fn observation_message(&self, raw: &str) -> Option<String> {
        evidence::failure_evidence(
            &serde_json::json!({"Message":raw}),
            "PROVIDER_OBSERVATION",
            200,
            &self.credentials,
            None,
        )
        .message
    }
    pub(crate) fn call_observed(
        &self,
        request: &Request,
        cancelled: &AtomicBool,
        on_failure: &mut dyn FnMut(FailureEvidence),
    ) -> Result<Value> {
        request.validate().map_err(|_| Error::configuration())?;
        let mut private = request.clone();
        if let Some(registry) = &self.registry {
            registry.inject(&mut private)?;
        }
        let request = &private;
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::transport(false));
        }
        let now = time::OffsetDateTime::now_utc();
        let signed = match (&self.credentials, &request.endpoint) {
            (
                Credentials::Pai {
                    access_key_id,
                    access_key_secret,
                    security_token,
                },
                Endpoint::Pai { .. },
            ) => {
                let mut bytes = [0u8; 16];
                getrandom::fill(&mut bytes).map_err(|_| Error::configuration())?;
                let nonce = signing::hex(&bytes);
                let format = time::format_description::parse(
                    "[year]-[month]-[day]T[hour]:[minute]:[second]Z",
                )
                .map_err(|_| Error::configuration())?;
                let date = now.format(&format).map_err(|_| Error::configuration())?;
                signing::sign(
                    request,
                    access_key_id,
                    access_key_secret,
                    security_token.as_deref(),
                    &date,
                    &nonce,
                )?
            }
            (
                Credentials::Sensecore {
                    access_key_id,
                    access_key_secret,
                },
                Endpoint::Sensecore { .. },
            ) => {
                let format=time::format_description::parse("[weekday repr:short], [day] [month repr:short] [year] [hour]:[minute]:[second] GMT").map_err(|_|Error::configuration())?;
                let date = now.format(&format).map_err(|_| Error::configuration())?;
                signing::sign(request, access_key_id, access_key_secret, None, &date, "")?
            }
            _ => return Err(Error::configuration()),
        };
        let method = match request.method {
            Method::Get => reqwest::Method::GET,
            Method::Post => reqwest::Method::POST,
            Method::Delete => reqwest::Method::DELETE,
            Method::Put => reqwest::Method::PUT,
        };
        let mutation = !request.read_only;
        let mut builder = self.http.request(method, signed.url).timeout(self.timeout);
        for (name, value) in signed.headers {
            builder = builder.header(name, value);
        }
        if !signed.body.is_empty() {
            builder = builder.body(signed.body);
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::transport(false));
        }
        let response = builder.send().map_err(|_| Error::transport(mutation))?;
        let status = response.status();
        let mut bytes = Vec::new();
        response
            .take(LIMIT + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| Error::transport(mutation))?;
        if bytes.len() as u64 > LIMIT || cancelled.load(Ordering::Relaxed) {
            return Err(Error::transport(mutation));
        }
        let value = if bytes.is_empty() {
            Value::Object(Default::default())
        } else {
            Value::Object(ml_exp_core::json::mapping(&bytes).map_err(|_| Error {
                code: "CLOUD_RESPONSE_INVALID".into(),
                http_status: Some(status.as_u16()),
                uncertain: mutation,
            })?)
        };
        let provider_error = match request.endpoint {
            Endpoint::Pai { .. } => {
                !matches!(value.get("Code"), None | Some(Value::Null))
                    && !matches!(value["Code"].as_str(), Some("Success" | "200"))
                    && value["Code"] != 200
                    || value["Success"] == false
                    || value["IsSuccess"] == false
            }
            Endpoint::Sensecore { .. } => false,
        };
        if !status.is_success() || provider_error {
            let raw = match request.endpoint {
                Endpoint::Pai { .. } => value["Code"].as_str(),
                Endpoint::Sensecore { .. } => value["details"]
                    .as_array()
                    .and_then(|a| a.iter().find_map(|v| v["reason"].as_str())),
            };
            let code = raw
                .filter(|s| {
                    s.len() <= 100
                        && !s.is_empty()
                        && s.bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
                })
                .unwrap_or("CLOUD_REQUEST_FAILED");
            on_failure(evidence::failure_evidence(
                &value,
                code,
                status.as_u16(),
                &self.credentials,
                request.body.as_ref(),
            ));
            return Err(Error {
                code: code.into(),
                http_status: Some(status.as_u16()),
                uncertain: mutation,
            });
        }
        Ok(value)
    }
}
