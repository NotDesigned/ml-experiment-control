//! Pure preflight decisions for a read-only session, with explicit saved-state
//! and handshake evidence. The final file-read function is the only I/O here.

use crate::{ConnectionPlan, Error, Settings, observation};
use serde_json::Value;
use std::path::Path;

/// Validate the saved server and exact selection before loading a credential
/// file or sending even a handshake. A downloaded state cannot select the
/// destination for an ambient credential.
pub fn connection(
    settings: &Settings,
    selected_url: Option<&str>,
    env_token: Option<&str>,
    env_file: Option<&Path>,
    saved: Option<&Value>,
    selection: observation::Selection<'_>,
) -> Result<ConnectionPlan, Error> {
    let hints = saved.map(observation::reference).transpose()?;
    let selected_url = selected_url.or(settings.url.as_deref());
    let saved_url = hints.as_ref().and_then(|h| h.server.as_deref());
    if let Some(saved_url) = saved_url {
        crate::normalize_url(saved_url)?;
        if let Some(url) = selected_url {
            if url.trim_end_matches('/') != saved_url.trim_end_matches('/') {
                return Err(Error::Observation(
                    "saved state belongs to a different server; no request was sent",
                ));
            }
        } else if env_token.is_some() || env_file.is_some() {
            return Err(Error::Observation(
                "saved state cannot choose a server for API credentials; configure --url first; no request was sent",
            ));
        }
    }
    let plan = settings.resolve(selected_url.or(saved_url), env_token, env_file)?;
    // A state-machine step is a value, not an HTTP request. Validate the full
    // local binding now; the observation shell executes any returned GET later.
    if saved.is_some() || selection.project.is_some() || selection.run.is_some() {
        observation::begin(&plan.base_url, saved, selection)?;
    }
    Ok(plan)
}

/// Accept protocol evidence only after a successful authenticated GET. This
/// does not claim any training/provider capability is available.
pub fn validate_health(value: &Value) -> Result<(), Error> {
    let min = value
        .get("min_client_protocol_version")
        .and_then(Value::as_u64);
    let max = value.get("api_protocol_version").and_then(Value::as_u64);
    if !matches!((min,max), (Some(min),Some(max)) if min > 0 && min <= 2 && max >= 2)
        || value.get("capabilities").is_some_and(|values| {
            !values.as_array().is_some_and(|values| {
                values.len() <= 512
                    && values.iter().all(|v| {
                        v.as_str().is_some_and(|s| {
                            !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control)
                        })
                    })
            })
        })
    {
        return Err(Error::Protocol);
    }
    Ok(())
}

/// Saved state can contain signed capabilities. Require a bounded private
/// regular file; do not echo its contents or path on failure.
pub fn load_saved(path: &Path) -> Result<Value, Error> {
    let bytes = ml_exp_io::read_private(path, 8 * 1024 * 1024, true).map_err(|_| {
        Error::Observation(
            "cannot read private observation state; use the original saved JSON file",
        )
    })?;
    let saved = serde_json::from_slice(&bytes).map_err(|_| {
        Error::Observation("cannot read observation state; use the original saved JSON file")
    })?;
    observation::reference(&saved)?;
    Ok(saved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn saved_server_and_selection_are_checked_without_io() {
        let settings = Settings::from_token_reference(
            "https://trusted.invalid",
            "/missing/private-token".into(),
        )
        .unwrap();
        let saved = json!({"server":"https://trusted.invalid","project":"p","run_id":"r","config_sha256":"frozen"});
        assert!(
            connection(
                &settings,
                None,
                None,
                None,
                Some(&saved),
                observation::Selection::default()
            )
            .is_ok()
        );
        for (url, project, run) in [
            (Some("https://other.invalid"), None, None),
            (None, Some("foreign"), None),
            (None, None, Some("foreign")),
        ] {
            assert!(
                connection(
                    &settings,
                    url,
                    None,
                    None,
                    Some(&saved),
                    observation::Selection {
                        project,
                        run,
                        attempt: None
                    }
                )
                .is_err()
            );
        }
        assert!(
            connection(
                &Settings::default(),
                None,
                Some("test-secret"),
                None,
                Some(&saved),
                observation::Selection::default()
            )
            .is_err()
        );
        assert!(
            connection(
                &Settings::default(),
                None,
                None,
                Some(Path::new("/missing/private-token")),
                Some(&saved),
                observation::Selection::default()
            )
            .is_err()
        );
        let malformed =
            json!({"server":"https://trusted.invalid","contract":"training-operation.v1"});
        assert!(
            connection(
                &settings,
                None,
                None,
                None,
                Some(&malformed),
                observation::Selection::default()
            )
            .is_err()
        );
    }

    #[test]
    fn handshake_is_explicit_and_does_not_infer_training_readiness() {
        for (value, valid) in [
            (
                json!({"min_client_protocol_version":2,"api_protocol_version":2}),
                true,
            ),
            (
                json!({"min_client_protocol_version":1,"api_protocol_version":3,"capabilities":[]}),
                true,
            ),
            (
                json!({"min_client_protocol_version":2,"api_protocol_version":2,"deployment":{"training":"INCOMPLETE"}}),
                true,
            ),
            (json!({"ok":true}), false),
            (
                json!({"min_client_protocol_version":true,"api_protocol_version":2}),
                false,
            ),
            (
                json!({"min_client_protocol_version":2,"api_protocol_version":1}),
                false,
            ),
            (
                json!({"min_client_protocol_version":3,"api_protocol_version":3}),
                false,
            ),
            (
                json!({"min_client_protocol_version":0,"api_protocol_version":2}),
                false,
            ),
            (
                json!({"min_client_protocol_version":2.0,"api_protocol_version":2}),
                false,
            ),
            (
                json!({"min_client_protocol_version":2,"api_protocol_version":2,"capabilities":"all"}),
                false,
            ),
            (
                json!({"min_client_protocol_version":2,"api_protocol_version":2,"capabilities":[false]}),
                false,
            ),
        ] {
            assert_eq!(validate_health(&value).is_ok(), valid, "{value}");
        }
    }
}
