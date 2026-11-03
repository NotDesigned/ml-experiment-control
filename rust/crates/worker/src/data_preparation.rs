//! Explicit script effects. Persistent input caches and diagnostics stay outside
//! training output; incomplete or failed executions are never silently replayed.
use crate::managed::{Error, Result};
use ml_exp_core::{artifacts::FileReceipt, data_preparation::Plan};
use ml_exp_io::{artifacts, datasets::inventory, directory::Directory, process};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    path::Path,
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    plan: Plan,
    script_sha256: String,
    content_sha256: String,
    files: Vec<FileReceipt>,
}
fn failure(code: &'static str, error: std::io::Error) -> Error {
    Error::DataPreparation {
        code,
        exit_code: None,
        errno: error.raw_os_error(),
    }
}

pub fn prepare(plan: &Plan, allow_download: bool, cancelled: &AtomicBool) -> Result<String> {
    plan.validate().map_err(|_| Error::Configuration)?;
    let workdir = Directory::open(Path::new(&plan.workdir), false)
        .map_err(|e| failure("SCRIPT_WORKDIR", e))?;
    let mut parent = workdir.clone();
    let parts = plan.request.script.split('/').collect::<Vec<_>>();
    for part in &parts[..parts.len() - 1] {
        parent = parent
            .child(OsStr::new(part), false)
            .map_err(|e| failure("SCRIPT_UNAVAILABLE", e))?;
    }
    let mut script = parent
        .open_regular_read(OsStr::new(parts.last().ok_or(Error::Configuration)?))
        .map_err(|e| failure("SCRIPT_UNAVAILABLE", e))?;
    let (script_sha256, _) = artifacts::hash(&mut script).map_err(|e| failure("SCRIPT_HASH", e))?;
    let mut root = Directory::open(Path::new(&plan.mounted_root), false)
        .map_err(|e| failure("CACHE_MOUNT", e))?;
    for component in Path::new(&plan.cache_root)
        .strip_prefix(&plan.mounted_root)
        .map_err(|_| Error::Configuration)?
        .components()
    {
        let std::path::Component::Normal(name) = component else {
            return Err(Error::Configuration);
        };
        root = root
            .child(name, true)
            .map_err(|e| failure("CACHE_ROOT", e))?;
        root.verify_private_parent()
            .map_err(|e| failure("CACHE_ROOT", e))?;
    }
    root.verify_private_parent()
        .map_err(|e| failure("CACHE_ROOT", e))?;
    let cache = root
        .child(OsStr::new("script-preparations"), true)
        .map_err(|e| failure("CACHE_ROOT", e))?;
    let id = plan.id().map_err(|_| Error::Configuration)?;
    let lock = cache
        .open_lock(OsStr::new(&format!("{id}.lock")))
        .map_err(|e| failure("CACHE_LOCK", e))?;
    fs2::FileExt::try_lock_exclusive(&lock).map_err(|_| Error::Busy)?;
    let existing = cache
        .entry_exists(OsStr::new(&id))
        .map_err(|e| failure("CACHE_LOOKUP", e))?;
    if existing {
        let slot = cache
            .child(OsStr::new(&id), false)
            .map_err(|e| failure("CACHE_SLOT", e))?;
        let saved: Receipt = serde_json::from_slice(
            &slot
                .read_regular(OsStr::new("receipt.json"), 8 * 1024 * 1024)
                .map_err(|e| failure("CACHE_INCOMPLETE", e))?,
        )
        .map_err(|_| Error::Binding)?;
        if saved.plan.id().map_err(|_| Error::Binding)? != id
            || saved.script_sha256 != script_sha256
        {
            return Err(Error::Binding);
        }
        let tree = slot
            .child(OsStr::new("tree"), false)
            .map_err(|e| failure("CACHE_TREE", e))?;
        let files = inventory(&tree).map_err(|e| failure("CACHE_VERIFY", e))?;
        let digest =
            ml_exp_core::data_preparation::content_hash(&files).map_err(|_| Error::Binding)?;
        if files != saved.files || digest != saved.content_sha256 {
            return Err(Error::Binding);
        }
        ml_exp_io::datasets::verify_inventory(&tree, &files, true)
            .map_err(|e| failure("CACHE_VERIFY", e))?;
        println!("ML_EXPD_DATA_PREPARATION=READY {id} reused=true");
        return plan.tree().map_err(|_| Error::Configuration);
    }
    if plan.require_cached && !allow_download {
        return Err(Error::DataPreparation {
            code: "CACHE_NOT_READY",
            exit_code: None,
            errno: None,
        });
    }
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    let slot = cache
        .create_new_child(OsStr::new(&id))
        .map_err(|e| failure("CACHE_RESERVATION", e))?;
    slot.publish_new(
        OsStr::new("intent.json"),
        &serde_json::to_vec(plan).map_err(|_| Error::Configuration)?,
    )
    .map_err(|e| failure("CACHE_INTENT", e))?;
    let tree = slot
        .create_new_child(OsStr::new("tree"))
        .map_err(|e| failure("CACHE_TREE", e))?;
    let mut stdout = slot
        .create_new_regular(OsStr::new("stdout.log"))
        .map_err(|e| failure("SCRIPT_LOG", e))?;
    let mut stderr = slot
        .create_new_regular(OsStr::new("stderr.log"))
        .map_err(|e| failure("SCRIPT_LOG", e))?;
    let mut env = BTreeMap::from([
        (
            "PATH".into(),
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".into(),
        ),
        (
            "DATA_DIR".into(),
            plan.tree().map_err(|_| Error::Configuration)?,
        ),
        ("SOURCE_ID".into(), plan.source_id.clone()),
    ]);
    env.extend(plan.env.clone());
    let mut argv = vec![
        plan.request.interpreter.clone(),
        format!("{}/{}", plan.workdir, plan.request.script),
    ];
    argv.extend(plan.request.arguments.clone());
    println!("ML_EXPD_DATA_PREPARATION=RUNNING {id}");
    let result = process::execute(process::Request {
        argv: &argv,
        cwd: Path::new(&plan.workdir),
        environment: process::Environment::Replace(&env),
        stdin: process::Input::Null,
        stdout: process::Capture::File {
            file: &mut stdout,
            limit: 8 * 1024 * 1024,
        },
        stderr: process::Capture::File {
            file: &mut stderr,
            limit: 8 * 1024 * 1024,
        },
        deadline: Instant::now() + Duration::from_secs(plan.request.timeout_seconds),
        termination_grace: Duration::from_secs(3),
        drain_grace: Duration::from_secs(3),
        cancelled,
    })
    .map_err(|e| failure("SCRIPT_PROCESS", e))?;
    stdout.sync_all().map_err(|e| failure("SCRIPT_LOG", e))?;
    stderr.sync_all().map_err(|e| failure("SCRIPT_LOG", e))?;
    let code = if result.cancelled {
        "SCRIPT_CANCELLED"
    } else if result.timed_out {
        "SCRIPT_TIMEOUT"
    } else if result.exit_code != Some(0) || result.failure.is_some() {
        "SCRIPT_FAILED"
    } else {
        "SCRIPT_COMPLETED"
    };
    slot.publish_new(OsStr::new("execution.json"),&serde_json::to_vec(&serde_json::json!({"code":code,"exit_code":result.exit_code,"timed_out":result.timed_out,"cancelled":result.cancelled,"stdout_bytes":result.stdout.received_bytes,"stderr_bytes":result.stderr.received_bytes})).map_err(|_|Error::Configuration)?).map_err(|e|failure("SCRIPT_RECEIPT",e))?;
    if code != "SCRIPT_COMPLETED" {
        return Err(Error::DataPreparation {
            code,
            exit_code: result.exit_code,
            errno: None,
        });
    }
    let files = inventory(&tree).map_err(|e| failure("DATA_VERIFY", e))?;
    let content_sha256 =
        ml_exp_core::data_preparation::content_hash(&files).map_err(|_| Error::Binding)?;
    if plan
        .request
        .expected_content_sha256
        .as_ref()
        .is_some_and(|hash| hash != &content_sha256)
    {
        return Err(Error::DataPreparation {
            code: "CONTENT_HASH_MISMATCH",
            exit_code: Some(0),
            errno: None,
        });
    }
    ml_exp_io::datasets::seal_generated(&tree).map_err(|e| failure("DATA_SEAL", e))?;
    ml_exp_io::datasets::verify_inventory(&tree, &files, true)
        .map_err(|e| failure("DATA_VERIFY", e))?;
    let receipt = Receipt {
        plan: plan.clone(),
        script_sha256,
        content_sha256,
        files,
    };
    let bytes = serde_json::to_vec(&receipt).map_err(|_| Error::Configuration)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err(Error::Configuration);
    }
    slot.publish_new(OsStr::new("receipt.json"), &bytes)
        .map_err(|e| failure("DATA_RECEIPT", e))?;
    cache
        .validate_regular_entry(OsStr::new(&format!("{id}.lock")), &lock)
        .map_err(|e| failure("CACHE_LOCK", e))?;
    println!("ML_EXPD_DATA_PREPARATION=READY {id} reused=false");
    plan.tree().map_err(|_| Error::Configuration)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan(root: &Path, body: &str) -> Plan {
        let work = root.join("code");
        std::fs::create_dir(&work).unwrap();
        std::fs::write(work.join("fetch.sh"), body).unwrap();
        Plan {contract:ml_exp_core::data_preparation::CONTRACT.into(),request:serde_json::from_value(serde_json::json!({"script":"fetch.sh","interpreter":"/bin/sh","timeout_seconds":5})).unwrap(),source_id:format!("source.{}","a".repeat(64)),image:format!("registry/image@sha256:{}","b".repeat(64)),mounted_root:root.to_str().unwrap().into(),cache_root:root.join("data").to_str().unwrap().into(),workdir:work.to_str().unwrap().into(),env:BTreeMap::new(),require_cached:true}
    }
    #[test]
    fn verified_cache_is_reused_but_incomplete_and_changed_data_are_refused() {
        let temp = tempfile::tempdir().unwrap();
        let plan = plan(
            temp.path(),
            "printf downloaded > \"$DATA_DIR/sample\"\nprintf x >> calls\n",
        );
        let cancelled = AtomicBool::new(false);
        assert!(prepare(&plan, false, &cancelled).is_err());
        let tree = prepare(&plan, true, &cancelled).unwrap();
        assert_eq!(prepare(&plan, false, &cancelled).unwrap(), tree);
        assert_eq!(
            std::fs::read(Path::new(&plan.workdir).join("calls")).unwrap(),
            b"x"
        );
        std::fs::write(Path::new(&plan.workdir).join("fetch.sh"), "exit 0").unwrap();
        assert!(prepare(&plan, true, &cancelled).is_err());
    }
    #[test]
    fn failed_download_keeps_diagnostics_and_cannot_restart_implicitly() {
        let temp = tempfile::tempdir().unwrap();
        let plan = plan(
            temp.path(),
            "printf failed >&2\nprintf x >> calls\nexit 17\n",
        );
        let cancelled = AtomicBool::new(false);
        assert!(matches!(
            prepare(&plan, true, &cancelled),
            Err(Error::DataPreparation {
                code: "SCRIPT_FAILED",
                exit_code: Some(17),
                ..
            })
        ));
        assert!(prepare(&plan, true, &cancelled).is_err());
        assert_eq!(
            std::fs::read(Path::new(&plan.workdir).join("calls")).unwrap(),
            b"x"
        );
        let slot = Path::new(&plan.cache_root)
            .join("script-preparations")
            .join(plan.id().unwrap());
        assert!(slot.join("execution.json").is_file());
        assert!(!slot.join("receipt.json").exists());
        assert_eq!(std::fs::read(slot.join("stderr.log")).unwrap(), b"failed");
    }
}
