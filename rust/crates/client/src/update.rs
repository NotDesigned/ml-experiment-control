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
    let target = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        _ => return Err(Error::Observation("no native updater for this platform")),
    };
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
        .select(version, target, "ml-exp-rs")
        .map_err(Error::Observation)?;
    if check {
        return Ok(
            serde_json::json!({"version":version,"target":target,"asset":asset.name,"sha256":asset.sha256,"installed":false}),
        );
    }
    let path = std::env::current_exe().map_err(|_| fail())?;
    let original = ml_exp_io::sha256_file(&path).map_err(|_| fail())?;
    let parent = path.parent().ok_or_else(fail)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|_| fail())?;
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
    let argv = vec![
        temporary.to_str().ok_or_else(fail)?.to_owned(),
        "--version".into(),
    ];
    let environment = std::collections::BTreeMap::new();
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    // Execute only the authenticated, hash-verified release with bounded I/O.
    let output = ml_exp_io::process::execute(ml_exp_io::process::Request {
        argv: &argv,
        cwd: parent,
        environment: ml_exp_io::process::Environment::Replace(&environment),
        stdin: ml_exp_io::process::Input::Null,
        stdout: ml_exp_io::process::Capture::Memory { limit: 4096 },
        stderr: ml_exp_io::process::Capture::Memory { limit: 4096 },
        deadline: std::time::Instant::now() + Duration::from_secs(5),
        termination_grace: Duration::from_secs(1),
        drain_grace: Duration::from_secs(1),
        cancelled: &cancelled,
    })
    .map_err(|_| fail())?;
    if output.exit_code != Some(0)
        || output.timed_out
        || output.failure.is_some()
        || !output.stdout.complete
        || output.stdout.truncated
        || String::from_utf8_lossy(&output.stdout.bytes).trim() != format!("ml-exp-rs {version}")
    {
        return Err(fail());
    }
    if ml_exp_io::sha256_file(&path).map_err(|_| fail())? != original {
        return Err(fail());
    }
    temporary.persist(&path).map_err(|_| fail())?;
    std::fs::File::open(parent).and_then(|f|f.sync_all()).map_err(|_|Error::Observation("binary replaced but directory sync failed; check installed --version before retrying"))?;
    Ok(
        serde_json::json!({"version":version,"target":target,"sha256":asset.sha256,"installed":true}),
    )
}

fn discover(http: &reqwest::blocking::Client) -> Result<String, Error> {
    let mut releases = Vec::new();
    for page in 1..=10 {
        let response = http
            .get(format!("https://api.github.com/repos/NotDesigned/ml-experiment-control/releases?per_page=100&page={page}"))
            .header(reqwest::header::USER_AGENT, concat!("ml-exp-rs/", env!("CARGO_PKG_VERSION")))
            .header(reqwest::header::ACCEPT, "application/vnd.github+json")
            .send().map_err(|_| fail())?;
        if response.status() != 200 {
            return Err(Error::Observation(
                "cannot discover latest stable native release; no version was selected",
            ));
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
