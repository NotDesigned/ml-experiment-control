//! Private file and credential I/O. Precedence and URL binding are decided in
//! config; this boundary only executes explicit reads and writes.

use std::path::{Path, PathBuf};

use ml_exp_io::{create_private_dir, read_private, write_atomic};

use crate::Error;
use crate::config::{Settings, TokenSource, validate_token};

const CONFIG_LIMIT: u64 = 32 * 1024;

pub fn load_settings(path: &Path) -> Result<Settings, Error> {
    let bytes = match read_private(path, CONFIG_LIMIT, true) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Settings::default());
        }
        Err(_) => return Err(Error::Config),
    };
    Settings::from_json(&bytes)
}

pub fn save_settings(settings: &Settings, path: &Path) -> Result<(), Error> {
    let bytes = settings.to_private_json()?;
    if std::fs::symlink_metadata(path).is_ok() {
        read_private(path, CONFIG_LIMIT, true).map_err(|_| Error::Config)?;
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        create_private_dir(parent).map_err(|_| Error::Config)?;
    }
    write_atomic(path, &bytes, false).map_err(|_| Error::Config)
}

/// Verify the explicit file before saving a canonical reference to it.
pub fn settings_from_token_file(url: &str, token_file: PathBuf) -> Result<Settings, Error> {
    load_token_file(&token_file)?;
    let canonical = token_file.canonicalize().map_err(|_| Error::Config)?;
    Settings::from_token_reference(url, canonical)
}

pub fn load_token(source: TokenSource) -> Result<String, Error> {
    match source {
        TokenSource::Inline(token) => {
            validate_token(&token)?;
            Ok(token)
        }
        TokenSource::File(path) => load_token_file(&path),
    }
}

fn load_token_file(path: &Path) -> Result<String, Error> {
    let token = String::from_utf8(read_private(path, 8192, true).map_err(|_| Error::Config)?)
        .map_err(|_| Error::Token)?
        .trim()
        .to_owned();
    validate_token(&token)?;
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_settings_preserve_url_binding_and_never_show_secret() {
        let temp = tempfile::tempdir().unwrap();
        let token = temp.path().join("token");
        write_atomic(&token, b"native-test-token-value-123456789012345\n", true).unwrap();
        let config = temp.path().join("nested/client.json");
        let settings = settings_from_token_file("https://example.invalid/ml-expd", token).unwrap();
        save_settings(&settings, &config).unwrap();
        let settings = load_settings(&config).unwrap();
        assert!(
            !settings
                .public()
                .to_string()
                .contains("native-test-token-value")
        );
        assert!(matches!(
            settings.resolve(Some("https://other.invalid"), None, None),
            Err(Error::CredentialScope)
        ));
        let plan = settings.resolve(None, None, None).unwrap();
        assert_eq!(
            load_token(plan.token_source).unwrap(),
            "native-test-token-value-123456789012345"
        );
        assert!(settings.token_file.unwrap().is_absolute());
        assert!(
            !String::from_utf8(std::fs::read(&config).unwrap())
                .unwrap()
                .contains("native-test-token-value")
        );
    }

    #[test]
    fn references_are_loaded_at_the_io_boundary_not_during_planning() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("token");
        let settings =
            Settings::from_token_reference("https://example.invalid", missing.clone()).unwrap();
        let plan = settings.resolve(None, None, None).unwrap();
        assert!(matches!(load_token(plan.token_source), Err(Error::Config)));
        assert!(settings_from_token_file("https://example.invalid", missing).is_err());
        assert!(matches!(
            load_token(TokenSource::Inline("bad\nheader".into())),
            Err(Error::Token)
        ));
    }

    #[test]
    fn explicit_legacy_private_json_is_supported_without_touching_other_files() {
        let temp = tempfile::tempdir().unwrap();
        let legacy = temp.path().join("legacy-client.json");
        let original =
            br#"{"url":"https://example.invalid/ml-expd/","token":"legacy-test-secret"}"#;
        write_atomic(&legacy, original, true).unwrap();
        let settings = load_settings(&legacy).unwrap();
        let plan = settings
            .resolve(Some("https://example.invalid/ml-expd"), None, None)
            .unwrap();
        assert_eq!(load_token(plan.token_source).unwrap(), "legacy-test-secret");
        assert_eq!(std::fs::read(&legacy).unwrap(), original);
        assert_eq!(
            load_settings(&temp.path().join("absent")).unwrap().public()["token_configured"],
            false
        );
    }

    #[cfg(unix)]
    #[test]
    fn public_or_symlinked_configs_and_credentials_are_rejected() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let temp = tempfile::tempdir().unwrap();
        let token = temp.path().join("token");
        write_atomic(&token, b"test-secret", true).unwrap();
        let link = temp.path().join("token-link");
        symlink(&token, &link).unwrap();
        assert!(settings_from_token_file("https://example.invalid", link).is_err());
        std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(load_token(TokenSource::File(token)).is_err());

        let config = temp.path().join("config.json");
        let settings = Settings::default();
        save_settings(&settings, &config).unwrap();
        assert_eq!(
            std::fs::metadata(&config).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let alias = temp.path().join("config-link");
        symlink(&config, &alias).unwrap();
        assert!(load_settings(&alias).is_err());
        assert!(save_settings(&settings, &alias).is_err());
        std::fs::set_permissions(&config, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(load_settings(&config).is_err());
        assert!(save_settings(&settings, &config).is_err());
    }
}
