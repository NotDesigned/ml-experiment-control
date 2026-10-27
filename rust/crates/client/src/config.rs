//! Pure configuration and connection decisions. Files and environment are inputs,
//! never read here. Credential-bearing types intentionally do not implement Debug.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use url::Url;

use crate::Error;

pub fn normalize_url(value: &str) -> Result<String, Error> {
    if value.chars().any(|c| c.is_whitespace() || c.is_control()) || value.len() > 8192 {
        return Err(Error::Url);
    }
    // Even an empty userinfo component is forbidden; URL normalization may
    // otherwise erase the bare '@' before username/password can be inspected.
    if value
        .split_once("://")
        .and_then(|(_, rest)| rest.split(['/', '\\', '?', '#']).next())
        .is_some_and(|authority| authority.contains('@'))
    {
        return Err(Error::Url);
    }
    let url = Url::parse(value).map_err(|_| Error::Url)?;
    let loopback = url.host_str().is_some_and(|h| {
        h == "localhost"
            || h.trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    });
    if url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !(url.scheme() == "https" || url.scheme() == "http" && loopback)
    {
        return Err(Error::Url);
    }
    Ok(url.to_string().trim_end_matches('/').to_owned())
}

pub(crate) fn validate_token(token: &str) -> Result<(), Error> {
    ml_exp_core::auth::validate_bearer(token, 1, 8192).map_err(|_| Error::Token)
}

pub fn default_config_path(xdg_config_home: Option<&Path>, home: Option<&Path>) -> PathBuf {
    xdg_config_home
        .map(Path::to_path_buf)
        .unwrap_or_else(|| home.unwrap_or(Path::new("")).join(".config"))
        .join("ml-expd-rust/client.json")
}

/// A selected credential, not yet loaded from disk. Never formatted or serialized.
pub enum TokenSource {
    Inline(String),
    File(PathBuf),
}

/// The result of pure precedence and URL-binding checks. Constructing this plan
/// does not establish a connection or require a referenced token file to exist.
pub struct ConnectionPlan {
    pub base_url: String,
    pub token_source: TokenSource,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_file: Option<PathBuf>,
}

impl Settings {
    /// Pure constructor. The credential I/O boundary verifies and canonicalizes
    /// token files before this reference is saved by the CLI.
    pub fn from_token_reference(url: &str, token_file: PathBuf) -> Result<Self, Error> {
        Ok(Self {
            url: Some(normalize_url(url)?),
            token: None,
            token_file: Some(token_file),
        })
    }

    fn validate(&self) -> Result<(), Error> {
        if let Some(url) = &self.url {
            normalize_url(url)?;
        }
        if self.token.is_some() && self.token_file.is_some()
            || (self.token.is_some() || self.token_file.is_some()) && self.url.is_none()
        {
            return Err(Error::Config);
        }
        if let Some(token) = &self.token {
            validate_token(token)?;
        }
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        let settings: Self = serde_json::from_slice(bytes).map_err(|_| Error::Config)?;
        settings.validate()?;
        Ok(settings)
    }

    pub fn to_private_json(&self) -> Result<Vec<u8>, Error> {
        self.validate()?;
        serde_json::to_vec_pretty(self).map_err(|_| Error::Config)
    }

    pub fn public(&self) -> serde_json::Value {
        serde_json::json!({"url":self.url, "token_configured": self.token.is_some() || self.token_file.is_some(), "token_file":self.token_file})
    }

    /// Explicit environment candidates are supplied by the shell. Saved secrets
    /// remain bound to their normalized URL; an explicit token overrides them.
    pub fn resolve(
        &self,
        selected_url: Option<&str>,
        env_token: Option<&str>,
        env_file: Option<&Path>,
    ) -> Result<ConnectionPlan, Error> {
        self.validate()?;
        let base_url = normalize_url(selected_url.or(self.url.as_deref()).ok_or(Error::Url)?)?;
        if env_token.is_some() && env_file.is_some() {
            return Err(Error::Token);
        }
        let token_source = if let Some(token) = env_token {
            validate_token(token)?;
            TokenSource::Inline(token.to_owned())
        } else if let Some(path) = env_file {
            TokenSource::File(path.to_path_buf())
        } else if self.token.is_some() || self.token_file.is_some() {
            if self
                .url
                .as_deref()
                .map(normalize_url)
                .transpose()?
                .as_deref()
                != Some(&base_url)
            {
                return Err(Error::CredentialScope);
            }
            match &self.token_file {
                Some(path) => TokenSource::File(path.clone()),
                None => TokenSource::Inline(self.token.clone().ok_or(Error::Token)?),
            }
        } else {
            return Err(Error::Token);
        };
        Ok(ConnectionPlan {
            base_url,
            token_source,
        })
    }
}

/// Preserve the origin and configured API prefix before the transport sees a URL.
pub fn api_url(base: &str, path: &str) -> Result<Url, Error> {
    if !path.starts_with("/api/")
        || path.contains(['\\', '#'])
        || path.chars().any(char::is_control)
    {
        return Err(Error::Path);
    }
    let base = normalize_url(base)?;
    let endpoint = Url::parse(&format!("{base}{path}")).map_err(|_| Error::Path)?;
    let prefix = format!(
        "{}/api/",
        Url::parse(&base)
            .map_err(|_| Error::Url)?
            .path()
            .trim_end_matches('/')
    );
    if !endpoint.path().starts_with(&prefix) {
        return Err(Error::Path);
    }
    Ok(endpoint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planning_token_files_is_pure_and_scope_is_checked_before_io() {
        let missing = PathBuf::from("/deliberately-missing/native-preview-token");
        let settings =
            Settings::from_token_reference("https://example.invalid/ml-expd/", missing.clone())
                .unwrap();
        let plan = settings.resolve(None, None, None).unwrap();
        assert_eq!(plan.base_url, "https://example.invalid/ml-expd");
        assert!(matches!(plan.token_source, TokenSource::File(path) if path == missing));
        assert!(matches!(
            settings.resolve(Some("https://other.invalid"), None, None),
            Err(Error::CredentialScope)
        ));
        let plan = settings
            .resolve(
                Some("https://other.invalid"),
                None,
                Some(Path::new("explicit-environment-token")),
            )
            .unwrap();
        assert_eq!(plan.base_url, "https://other.invalid");
        assert!(
            matches!(plan.token_source, TokenSource::File(path) if path == Path::new("explicit-environment-token"))
        );
    }

    #[test]
    fn environment_precedence_is_explicit_and_secrets_are_not_public() {
        let settings = Settings::from_json(
            br#"{"url":"https://example.invalid","token":"saved-test-secret"}"#,
        )
        .unwrap();
        assert!(!settings.public().to_string().contains("saved-test-secret"));
        let plan = settings.resolve(None, None, None).unwrap();
        assert!(
            matches!(plan.token_source, TokenSource::Inline(token) if token == "saved-test-secret")
        );
        let plan = settings
            .resolve(
                Some("https://other.invalid"),
                Some("explicit-test-secret"),
                None,
            )
            .unwrap();
        assert!(
            matches!(plan.token_source, TokenSource::Inline(token) if token == "explicit-test-secret")
        );
        assert!(matches!(
            settings.resolve(None, Some("a"), Some(Path::new("ignored"))),
            Err(Error::Token)
        ));
        assert!(matches!(
            settings.resolve(None, Some("bad\r\nheader"), None),
            Err(Error::Token)
        ));
        assert!(matches!(
            Settings::default().resolve(Some("https://example.invalid"), None, None),
            Err(Error::Token)
        ));
    }

    #[test]
    fn private_json_rejects_ambiguous_or_unbound_secrets_and_unknown_fields() {
        for bytes in [
            br#"{"token":"test-secret"}"#.as_slice(),
            br#"{"token_file":"private-token"}"#,
            br#"{"url":"https://example.invalid","token":"test-secret","token_file":"token"}"#,
            br#"{"url":"https://example.invalid","unknown":true}"#,
            br#"{"url":"https://example.invalid","token":"bad header"}"#,
            b"not JSON",
        ] {
            assert!(Settings::from_json(bytes).is_err());
        }
    }

    #[test]
    fn url_and_api_path_validation_preserve_prefix_and_loopback_only_http() {
        for value in [
            "http://example.invalid",
            "https://u:p@example.invalid",
            "https://@example.invalid",
            "https://example.invalid?token=x",
            "https://example.invalid/#x",
            "https://example.invalid/white space",
            "https://example.invalid/\n",
            "file:///private/token",
        ] {
            assert!(normalize_url(value).is_err(), "{value:?}");
        }
        for value in [
            "http://localhost:1234",
            "http://127.0.0.1:1234/ml-expd",
            "http://[::1]:1234/ml-expd",
            "https://example.invalid/ml-expd",
        ] {
            assert!(normalize_url(value).is_ok(), "{value:?}");
        }
        for path in [
            "/outside",
            "//other.invalid/api/health",
            "/api/../../outside",
            "/api/%2e%2e/outside",
            "/api/health#fragment",
            "/api/health\\outside",
            "/api/health\0",
            "/api/heal\nth",
            "/api/health?limit=1\r\n",
            "/api/health\u{7f}",
        ] {
            assert!(api_url("https://example.invalid/ml-expd", path).is_err());
        }
        assert_eq!(
            api_url("https://example.invalid/ml-expd/", "/api/metrics?limit=1")
                .unwrap()
                .as_str(),
            "https://example.invalid/ml-expd/api/metrics?limit=1"
        );
    }

    #[test]
    fn preview_default_is_separate_from_existing_python_client_configuration() {
        assert_eq!(
            default_config_path(None, Some(Path::new("/home/researcher"))),
            Path::new("/home/researcher/.config/ml-expd-rust/client.json")
        );
        assert_eq!(
            default_config_path(
                Some(Path::new("/private/config")),
                Some(Path::new("ignored"))
            ),
            Path::new("/private/config/ml-expd-rust/client.json")
        );
    }
}
