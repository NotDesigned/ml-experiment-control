//! Pure preflight decisions for a read-only session, with explicit saved-state
//! and health evidence. The final file-read function is the only I/O here.

use crate::{ConnectionPlan, Error, Settings, observation};
use serde_json::Value;
use std::path::Path;

/// Validate the saved server and exact selection before loading a credential
/// file or sending even a health request. A downloaded state cannot select the
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

/// Accept health only after a successful authenticated GET: a bounded capability
/// list is required, so an unrelated JSON reply is not taken for a daemon. This
/// does not claim any training/provider capability is available.
pub fn validate_health(value: &Value) -> Result<(), Error> {
    let valid = value
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|values| {
            values.len() <= 512
                && values.iter().all(|v| {
                    v.as_str().is_some_and(|s| {
                        !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control)
                    })
                })
        });
    if !valid {
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
        let malformed = json!({"server":"https://trusted.invalid"});
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
    fn health_needs_a_bounded_capability_list_and_does_not_infer_training_readiness() {
        for (value, valid) in [
            (json!({"capabilities":[]}), true),
            (
                json!({"capabilities":[],"deployment":{"training":"INCOMPLETE"}}),
                true,
            ),
            (json!({"ok":true}), false),
            (json!({"capabilities":null}), false),
            (json!({"capabilities":"all"}), false),
            (json!({"capabilities":[false]}), false),
            (json!({"capabilities":[""]}), false),
            (json!({"capabilities":["a\nb"]}), false),
            (json!({"capabilities":["a".repeat(257)]}), false),
            (json!({"capabilities":vec!["a"; 513]}), false),
        ] {
            assert_eq!(validate_health(&value).is_ok(), valid, "{value}");
        }
    }
}
