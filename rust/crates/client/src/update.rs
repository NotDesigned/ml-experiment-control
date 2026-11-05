//! Standalone update; API credentials never reach the release service. The old
//! executable is replaced only after byte/SHA and version-command validation.
use crate::Error;
use ml_exp_core::distribution::{self, Manifest};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    time::Duration,
};
fn fail() -> Error {
    Error::Observation(
        "native release verification or replacement failed; installed executable was not intentionally changed",
    )
}
pub fn run(requested: Option<&str>, check: bool) -> Result<serde_json::Value, Error> {
    if requested.is_some_and(|value| !distribution::version(value)) {
        return Err(fail());
    }
    let target = distribution::client_target(std::env::consts::OS, std::env::consts::ARCH)
        .map_err(Error::Observation)?;
    let http = reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            let url = attempt.url();
            if attempt.previous().len() < 3
                && url.scheme() == "https"
                && url.host_str().is_some_and(|h| {
                    h == "github.com"
                        || h == "release-assets.githubusercontent.com"
                        || h == "objects.githubusercontent.com"
                })
            {
                attempt.follow()
            } else {
                attempt.stop()
            }
        }))
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|_| fail())?;
    let version = match requested {
        Some(value) => value.to_owned(),
        None => {
            let value = discover(&http)?;
            distribution::prevent_default_downgrade(env!("CARGO_PKG_VERSION"), &value)
                .map_err(Error::Observation)?;
            value
        }
    };
    let version = version.as_str();
    let base = format!(
        "https://github.com/NotDesigned/ml-experiment-control/releases/download/native-v{version}"
    );
    let response = http
        .get(format!("{base}/native-release.json"))
        .send()
        .map_err(|_| fail())?;
    if response.status() != 200 {
        return Err(fail());
    }
    let mut bytes = Vec::new();
    response
        .take(65537)
        .read_to_end(&mut bytes)
        .map_err(|_| fail())?;
    if bytes.len() > 65536 {
        return Err(fail());
    }
    let manifest: Manifest = serde_json::from_value(serde_json::Value::Object(
        ml_exp_core::json::mapping(&bytes).map_err(|_| fail())?,
    ))
    .map_err(|_| fail())?;
    let asset = manifest
        .select(version, target, "ml-exp")
        .map_err(Error::Observation)?;
    if check {
        return Ok(
            serde_json::json!({"version":version,"target":target,"asset":asset.name,"sha256":asset.sha256,"installed":false}),
        );
    }
    let path = std::env::current_exe().map_err(|_| fail())?;
    let original = ml_exp_io::sha256_file(&path).map_err(|_| fail())?;
    let parent = path.parent().ok_or_else(fail)?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".ml-exp-update-")
        .suffix(if cfg!(windows) { ".exe" } else { "" })
        .tempfile_in(parent)
        .map_err(|_| fail())?;
    let response = http
        .get(format!("{base}/{}", asset.name))
        .send()
        .map_err(|_| fail())?;
    if response.status() != 200 || response.content_length().is_some_and(|n| n != asset.bytes) {
        return Err(fail());
    }
    let mut reader = response.take(asset.bytes + 1);
    let mut hash = Sha256::new();
    let mut total = 0u64;
    let mut buf = [0u8; 65536];
    loop {
        let n = reader.read(&mut buf).map_err(|_| fail())?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > asset.bytes {
            return Err(fail());
        }
        hash.update(&buf[..n]);
        temporary.write_all(&buf[..n]).map_err(|_| fail())?;
    }
    if total != asset.bytes || format!("{:x}", hash.finalize()) != asset.sha256 {
        return Err(fail());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        temporary
            .as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o755))
            .map_err(|_| fail())?;
    }
    temporary.as_file().sync_all().map_err(|_| fail())?;
    let temporary = temporary.into_temp_path();
    verify_executable(&temporary, version)?;
    if ml_exp_io::sha256_file(&path).map_err(|_| fail())? != original {
        return Err(fail());
    }
    #[cfg(unix)]
    temporary.persist(&path).map_err(|_| fail())?;
    #[cfg(windows)]
    self_replace::self_replace(&temporary).map_err(|_| fail())?;
    ml_exp_io::sync_directory(parent).map_err(|_| {
        Error::Observation(
            "binary replaced but directory sync failed; check installed --version before retrying",
        )
    })?;
    Ok(
        serde_json::json!({"version":version,"target":target,"sha256":asset.sha256,"installed":true}),
    )
}

/// Only executes an already hash-verified release. Capture and total time are
/// bounded on all client platforms, without shell commands or API credentials.
pub fn verify_executable(path: &std::path::Path, version: &str) -> Result<(), Error> {
    use std::{
        process::{Command, Stdio},
        sync::mpsc,
        time::Instant,
    };
    let mut command = Command::new(path);
    command
        .arg("--version")
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    if let Some(root) = std::env::var_os("SystemRoot") {
        command.env("SystemRoot", root);
    }
    let mut child = command.spawn().map_err(|_| fail())?;
    let stdout = child.stdout.take().ok_or_else(fail)?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.take(4097).read_to_end(&mut bytes).map(|_| bytes);
        let _ = tx.send(result);
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(fail());
            }
        }
    };
    let bytes = rx
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|_| fail())?
        .map_err(|_| fail())?;
    if !status.success()
        || bytes.len() > 4096
        || String::from_utf8_lossy(&bytes).trim() != format!("ml-exp {version}")
    {
        return Err(fail());
    }
    Ok(())
}

fn discover(http: &reqwest::blocking::Client) -> Result<String, Error> {
    let mut releases = Vec::new();
    for page in 1..=10 {
        let response = match http
            .get(format!("https://api.github.com/repos/NotDesigned/ml-experiment-control/releases?per_page=100&page={page}"))
            .header(reqwest::header::USER_AGENT, concat!("ml-exp/", env!("CARGO_PKG_VERSION")))
            .header(reqwest::header::ACCEPT, "application/vnd.github+json")
            .send() { Ok(r) => r, Err(_) => return discover_web(http) };
        if response.status() != 200 {
            return discover_web(http);
        }
        let mut bytes = Vec::new();
        response
            .take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| fail())?;
        if bytes.len() > 4 * 1024 * 1024 {
            return Err(fail());
        }
        let page: Vec<serde_json::Value> = serde_json::from_slice(&bytes).map_err(|_| fail())?;
        if page.len() > 100 {
            return Err(fail());
        }
        let complete = page.len() < 100;
        releases.extend(page);
        if complete {
            return distribution::latest_native(&releases).map_err(Error::Observation);
        }
    }
    Err(Error::Observation(
        "native release listing exceeds discovery limit; select --version explicitly",
    ))
}

// Public website discovery remains usable when the unauthenticated REST quota
// is exhausted. Selection still requires a stable native tag and the normal
// manifest/SHA/platform checks; API credentials never go to GitHub.
fn discover_web(http: &reqwest::blocking::Client) -> Result<String, Error> {
    let response = http.get("https://github.com/NotDesigned/ml-experiment-control/releases/latest")
        .send().map_err(|_| Error::Observation("latest-release discovery failed on both GitHub REST and website; use update --version VERSION"))?;
    if response.status() != 200 {
        return Err(Error::Observation(
            "latest-release website is unavailable; use update --version VERSION",
        ));
    }
    web_version(response.url().as_str())
}
fn web_version(url: &str) -> Result<String, Error> {
    let value = url
        .strip_prefix("https://github.com/NotDesigned/ml-experiment-control/releases/tag/native-v")
        .filter(|v| distribution::version(v))
        .ok_or(Error::Observation(
            "latest public release is not a stable native tag; use update --version VERSION",
        ))?;
    Ok(value.into())
}
#[cfg(test)]
mod discovery_tests {
    use super::*;
    #[test]
    fn website_fallback_requires_exact_repository_and_stable_native_tag() {
        assert_eq!(
            web_version(
                "https://github.com/NotDesigned/ml-experiment-control/releases/tag/native-v0.1.24"
            )
            .unwrap(),
            "0.1.24"
        );
        for url in [
            "https://github.com/NotDesigned/ml-experiment-control/releases/tag/native-v0.1.24-rc1",
            "https://github.com/NotDesigned/ml-experiment-control/releases/tag/v1.5.1",
            "https://github.com/other/repository/releases/tag/native-v0.1.24",
            "https://github.com/NotDesigned/ml-experiment-control/releases/tag/native-v0.1.24?unexpected=1",
        ] {
            assert!(web_version(url).is_err());
        }
    }
}
