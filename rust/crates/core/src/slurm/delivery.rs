//! The worker and code a Slurm job reads from shared storage instead of its image.
//!
//! The server places both there over its SSH channel before a submission. Each is
//! named by its content: a worker by its SHA256 under `ROOT/bin/workers`, and a code
//! source by its identity under `ROOT/code`. Neither is ever replaced or removed, so
//! any number of Runs and Attempts reuse one copy. This module holds only what a Run
//! freezes, where that lives and the fixed commands that check or place a worker.
use super::{Error, absolute_path, helper, path_within, shell_quote};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const FORMAT: &str = "slurm-delivery.v1";
/// Where a Run keeps what it froze. A Run without it works from its image, as every
/// Run did before delivery existed.
pub const KEY: &str = "native_slurm_delivery";
/// Below a code source's directory: the sealed tree, and the listing that says it is
/// complete. The same names an Attempt uses for delivered code on a cloud backend.
pub const CODE: &str = crate::cloud::launch::CODE;
pub const LISTING: &str = ".code-manifest.json";
/// The worker subcommand that checks or receives one code source on the gateway.
pub const RECEIVE: &str = "code-receive";

/// Paths here are joined into shell text on the gateway, so only plain characters.
fn plain(value: &str) -> bool {
    absolute_path(value)
        && value.len() > 1
        && !value.ends_with('/')
        && !value.contains("//")
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"/._-".contains(&c))
}
fn invalid() -> Error {
    Error::Configuration("delivery")
}

/// The operator's root on shared storage for what the server delivers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub root: String,
}
impl Settings {
    /// Inside the root that every job binds, and apart from Runs and datasets, whose
    /// directories are named by Project and could otherwise be called `bin` or `code`.
    pub fn validate(&self, mount_root: &str, run_root: &str, data_root: &str) -> Result<(), Error> {
        let bin = format!("{}/bin", self.root);
        let code = format!("{}/code", self.root);
        if !plain(&self.root)
            || !path_within(&self.root, mount_root)
            || [run_root, data_root].iter().any(|other| {
                path_within(&self.root, other)
                    || path_within(other, &bin)
                    || path_within(other, &code)
            })
        {
            return Err(invalid());
        }
        Ok(())
    }
    /// What a new Run freezes: this root, the worker its jobs will run, and its code.
    pub fn freeze(
        &self,
        worker_sha256: &str,
        code_source_id: Option<&str>,
    ) -> Result<Frozen, Error> {
        let frozen = Frozen {
            format: FORMAT.into(),
            root: self.root.clone(),
            worker_sha256: worker_sha256.into(),
            code_source_id: code_source_id.map(str::to_owned),
        };
        frozen.validate()?;
        Ok(frozen)
    }
}

/// What a Run froze. Its Attempts, retries and resumes run this worker, and this code
/// when it has some, wherever the operator's settings move later.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frozen {
    pub format: String,
    pub root: String,
    pub worker_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_source_id: Option<String>,
}
impl Frozen {
    pub fn validate(&self) -> Result<(), Error> {
        if self.format != FORMAT
            || !plain(&self.root)
            || !helper::valid_digest(&self.worker_sha256)
            || self
                .code_source_id
                .as_deref()
                .is_some_and(|code| crate::sources::validate_source_id(code).is_err())
        {
            return Err(invalid());
        }
        Ok(())
    }
    /// The delivery a Run or Attempt document froze, if it froze one.
    pub fn from_document(document: &Value) -> Result<Option<Self>, Error> {
        let Some(value) = document.get(KEY) else {
            return Ok(None);
        };
        let frozen: Self = serde_json::from_value(value.clone()).map_err(|_| invalid())?;
        frozen.validate()?;
        Ok(Some(frozen))
    }
    pub fn workers(&self) -> String {
        format!("{}/bin/workers", self.root)
    }
    pub fn worker_path(&self) -> String {
        format!("{}/{}", self.workers(), worker_name(&self.worker_sha256))
    }
    /// The directory every code source sits in, each named by its identity.
    pub fn code_parent(&self) -> String {
        format!("{}/code", self.root)
    }
    /// The working directory of a job whose code was delivered.
    pub fn workdir(&self) -> Option<String> {
        self.code_source_id
            .as_ref()
            .map(|code| format!("{}/{code}/{CODE}", self.code_parent()))
    }
}
fn worker_name(sha256: &str) -> &str {
    sha256.strip_prefix("sha256:").unwrap_or(sha256)
}

/// The fixed answers of the two worker commands, one per line.
pub const PRESENT: &str = "ML_EXPD_WORKER=PRESENT";
pub const ABSENT: &str = "ML_EXPD_WORKER=ABSENT";
pub const PLACED: &str = "ML_EXPD_WORKER=PLACED";
/// A file with the worker's name that is not that worker. It is never overwritten.
pub const CORRUPT: &str = "ML_EXPD_WORKER=CORRUPT";
/// The bytes sent did not have the hash they were sent for. Nothing was placed.
pub const MISMATCH: &str = "ML_EXPD_WORKER=MISMATCH";

/// These run under the gateway account's login shell before any ML-Expd binary, so they
/// work whatever worker is installed there, or none. One line, no `!` and no single
/// quote inside, so that a csh login shell passes it to `sh` unchanged.
fn remote(script: String) -> Result<Vec<String>, Error> {
    if !script.is_ascii() || script.contains(['\n', '!', '\'']) {
        return Err(invalid());
    }
    Ok(vec!["sh".into(), "-c".into(), shell_quote(&script)?])
}
fn check(path: &str, hex: &str) -> String {
    format!("echo \"{hex}  {path}\" | sha256sum --check --status")
}
/// Whether the worker is there with its hash, without changing it. With `helper`, the
/// gateway's fixed helper path is also made that worker when it differs, through a
/// verified copy beside it and one rename.
pub fn check_worker(frozen: &Frozen, helper: Option<&str>) -> Result<Vec<String>, Error> {
    frozen.validate()?;
    let hex = worker_name(&frozen.worker_sha256);
    let mut script = format!(
        "set -eu; f={path}; if test -h \"$f\"; then echo {CORRUPT}; exit 0; fi; \
         if test -f \"$f\"; then :; else if test -e \"$f\"; then echo {CORRUPT}; else echo {ABSENT}; fi; exit 0; fi; \
         if {verified}; then :; else echo {CORRUPT}; exit 0; fi; ",
        path = frozen.worker_path(),
        verified = check("$f", hex),
    );
    if let Some(helper) = helper {
        if !plain(helper) {
            return Err(invalid());
        }
        script.push_str(&format!(
            "h={helper}; if test -f \"$h\" && {current}; then :; else t=\"$h.ml-expd-incoming\"; rm -f \"$t\"; \
             cp \"$f\" \"$t\"; chmod 0755 \"$t\"; if {copied}; then mv -f \"$t\" \"$h\"; echo ML_EXPD_HELPER=REFRESHED; \
             else rm -f \"$t\"; echo {CORRUPT}; exit 0; fi; fi; ",
            current = check("$h", hex),
            copied = check("$t", hex),
        ));
    }
    script.push_str(&format!("echo {PRESENT}"));
    remote(script)
}
/// Receive the worker on standard input and place it under its name, only once its hash
/// is proven. An existing file of that name is left as it is.
pub fn place_worker(frozen: &Frozen) -> Result<Vec<String>, Error> {
    frozen.validate()?;
    let hex = worker_name(&frozen.worker_sha256);
    remote(format!(
        "set -eu; umask 022; d={workers}; f=\"$d/{hex}\"; mkdir -p \"$d\"; t=\"$d/.incoming-{hex}-$$\"; rm -f \"$t\"; \
         cat > \"$t\"; if {verified}; then :; else rm -f \"$t\"; echo {MISMATCH}; exit 0; fi; chmod 0555 \"$t\"; \
         if test -e \"$f\" || test -h \"$f\"; then rm -f \"$t\"; else mv \"$t\" \"$f\"; fi; echo {PLACED}",
        workers = frozen.workers(),
        verified = check("$t", hex),
    ))
}
/// The one answer line of either command, if there is exactly one.
pub fn answer(stdout: &str) -> Option<&str> {
    let mut lines = stdout
        .lines()
        .filter(|line| line.starts_with("ML_EXPD_WORKER="));
    let line = lines.next()?;
    lines.next().is_none().then_some(line)
}
/// The worker command that checks, or with content on standard input receives, one code
/// source. It runs as the Run's own frozen worker, which the same submission placed.
pub fn receive_code(frozen: &Frozen, check_only: bool) -> Result<Vec<String>, Error> {
    frozen.validate()?;
    let code = frozen.code_source_id.as_ref().ok_or_else(invalid)?;
    let mut argv = vec![
        frozen.worker_path(),
        RECEIVE.into(),
        "--root".into(),
        frozen.code_parent(),
        "--source".into(),
        code.clone(),
    ];
    if check_only {
        argv.push("--check".into());
    }
    Ok(argv)
}
pub const CODE_READY: &str = "ML_EXPD_CODE=READY";
pub const CODE_REUSED: &str = "ML_EXPD_CODE=REUSED";
pub const CODE_ABSENT: &str = "ML_EXPD_CODE=ABSENT";
/// The worker ran and refused: the copy there differs, or cannot be read or written.
pub const CODE_REFUSED: &str = "ML_EXPD_CODE=REFUSED";

#[cfg(test)]
mod tests {
    use super::*;
    fn settings() -> Settings {
        Settings {
            root: "/shared/ml-exp".into(),
        }
    }
    #[test]
    fn the_root_is_bound_into_every_job_and_apart_from_runs_and_datasets() {
        let ok = |root: &str, runs: &str, data: &str| {
            Settings { root: root.into() }
                .validate("/shared", runs, data)
                .is_ok()
        };
        assert!(ok(
            "/shared/ml-exp",
            "/shared/ml-exp/runs",
            "/shared/ml-exp/data"
        ));
        assert!(ok("/shared/delivered", "/shared/runs", "/shared/data"));
        assert!(!ok("/elsewhere", "/shared/runs", "/shared/data"));
        assert!(!ok(
            "/shared/ml-exp/runs",
            "/shared/ml-exp/runs",
            "/shared/data"
        ));
        assert!(!ok("/shared/ml-exp", "/shared/ml-exp/code", "/shared/data"));
        assert!(!ok(
            "/shared/ml-exp",
            "/shared/runs",
            "/shared/ml-exp/bin/data"
        ));
        assert!(!ok("/shared/ml exp", "/shared/runs", "/shared/data"));
        assert!(!ok("/shared/ml-exp/", "/shared/runs", "/shared/data"));
    }
    #[test]
    fn a_frozen_delivery_names_its_worker_and_code_by_content() {
        let worker = format!("sha256:{}", "a".repeat(64));
        let code = format!("source.{}", "b".repeat(64));
        let frozen = settings().freeze(&worker, Some(&code)).unwrap();
        assert_eq!(
            frozen.worker_path(),
            format!("/shared/ml-exp/bin/workers/{}", "a".repeat(64))
        );
        assert_eq!(
            frozen.workdir().unwrap(),
            format!("/shared/ml-exp/code/{code}/code")
        );
        let document = serde_json::json!({ KEY: frozen });
        assert_eq!(
            Frozen::from_document(&document).unwrap(),
            Some(frozen.clone())
        );
        assert_eq!(Frozen::from_document(&serde_json::json!({})).unwrap(), None);
        assert!(settings().freeze("sha256:short", None).is_err());
        assert!(settings().freeze(&worker, Some("source.short")).is_err());
        let plain = settings().freeze(&worker, None).unwrap();
        assert!(plain.workdir().is_none());
        assert!(
            serde_json::to_value(&plain)
                .unwrap()
                .get("code_source_id")
                .is_none()
        );
        assert!(receive_code(&plain, true).is_err());
        assert_eq!(
            receive_code(&frozen, true).unwrap()[1..],
            [
                RECEIVE.to_owned(),
                "--root".into(),
                "/shared/ml-exp/code".into(),
                "--source".into(),
                code,
                "--check".into()
            ]
        );
    }
    #[test]
    fn worker_commands_are_one_shell_word_a_login_shell_passes_on() {
        let frozen = settings()
            .freeze(&format!("sha256:{}", "c".repeat(64)), None)
            .unwrap();
        for argv in [
            check_worker(&frozen, None).unwrap(),
            check_worker(&frozen, Some("/shared/ml-exp/bin/ml-exp-worker-rs")).unwrap(),
            place_worker(&frozen).unwrap(),
        ] {
            assert_eq!(argv[..2], ["sh".to_owned(), "-c".into()]);
            assert!(argv[2].starts_with('\'') && argv[2].ends_with('\''));
            assert_eq!(argv[2].matches('\'').count(), 2);
        }
        assert!(check_worker(&frozen, Some("/bin/x;y")).is_err());
        assert_eq!(answer("x\nML_EXPD_WORKER=PRESENT\n"), Some(PRESENT));
        assert_eq!(
            answer("ML_EXPD_WORKER=ABSENT\nML_EXPD_WORKER=PRESENT\n"),
            None
        );
        assert_eq!(answer(""), None);
    }
}
