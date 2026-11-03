//! Descriptor-based effects for the core shared-Attempt plan. No network call
//! or scheduler effect occurs here. Partial publications are never adopted.
use super::*;
use ml_exp_core::slurm::staging as domain;
const MARKER: &str = ".native-staging.json";
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StorageBinding {
    path: PathBuf,
    identity: (u64, u64),
}
pub(super) fn open_root(config: &HelperConfig) -> Result<Option<Directory>> {
    config
        .shared_run_root
        .as_ref()
        .map(|path| {
            let root = Directory::open(path, false)?;
            root.verify_private_parent()?;
            Ok(root)
        })
        .transpose()
}
pub(super) fn binding(directory: Option<&Directory>) -> Result<Option<StorageBinding>> {
    directory
        .map(|directory| {
            directory.verify_private_parent()?;
            Ok(StorageBinding {
                path: directory.path(),
                identity: directory.identity()?,
            })
        })
        .transpose()
}
#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Sealed {
    plan: domain::Plan,
    directory_identities: Vec<(u64, u64)>,
}
fn identities(directories: &[Directory]) -> Result<Vec<(u64, u64)>> {
    directories
        .iter()
        .map(|directory| {
            directory.verify_private_parent()?;
            Ok(directory.identity()?)
        })
        .collect()
}
impl Helper {
    fn shared_plan(&self, intent: &FrozenIntent, manifest: &[u8]) -> Result<Option<domain::Plan>> {
        let plan = domain::plan(intent, manifest)?;
        if let Some(plan) = &plan {
            let root = self.shared.as_ref().ok_or(Error::Configuration)?;
            root.verify_private_parent()?;
            if root.path().to_str() != Some(plan.runs_root.as_str()) {
                return Err(Error::Configuration);
            }
        }
        Ok(plan)
    }
    fn shared_directories(&self, plan: &domain::Plan, create: bool) -> Result<Vec<Directory>> {
        let root = self.shared.as_ref().ok_or(Error::Configuration)?;
        let project = root.child(OsStr::new(&plan.scope.project), create)?;
        let run = project.child(OsStr::new(&plan.scope.run_id), create)?;
        let attempts = run.child(OsStr::new("attempts"), create)?;
        // A missing marker/manifest is never permission to adopt an old path.
        let attempt = if create {
            attempts.create_new_child(OsStr::new(&plan.scope.attempt_id))?
        } else {
            attempts.child(OsStr::new(&plan.scope.attempt_id), false)?
        };
        let inputs = attempt.child(OsStr::new("inputs"), create)?;
        let outputs = attempt.child(OsStr::new("outputs"), create)?;
        let mut named = Vec::new();
        if let Some(package) = &plan.managed {
            for dataset in &package.config.datasets {
                named.push(inputs.child(OsStr::new(&dataset.name), create)?);
            }
        }
        let mut directories = vec![project, run, attempts, attempt, inputs, outputs];
        directories.extend(named);
        Ok(directories)
    }
    pub(super) fn prepare_shared(
        &self,
        intent: &FrozenIntent,
        manifest: &[u8],
        diagnostics: &Directory,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let Some(plan) = self.shared_plan(intent, manifest)? else {
            return Ok(());
        };
        if let Some(package) = &plan.managed {
            if let Some(preparation) = &package.config.data_preparation {
                let program = self
                    .config
                    .data_preparer
                    .as_ref()
                    .ok_or(Error::Configuration)?;
                let document: serde_json::Value = serde_json::from_slice(manifest)?;
                let backend: &serde_json::Value = &document["backend"];
                let mut argv = vec![
                    program.to_str().ok_or(Error::Configuration)?.into(),
                    "exec".into(),
                    "--cleanenv".into(),
                ];
                let mount = backend["mount_root"].as_str().ok_or(Error::Configuration)?;
                let cache = Directory::open(std::path::Path::new(&preparation.cache_root), true)?;
                cache.verify_private_parent()?;
                argv.extend([
                    "--bind".into(),
                    format!("{mount}:{mount}"),
                    "--bind".into(),
                    format!("{}:{}", preparation.cache_root, preparation.cache_root),
                ]);
                if backend["oci_image"].is_null() {
                    argv.extend([
                        "--bind".into(),
                        format!(
                            "{}:{}:ro",
                            backend["source_dir"].as_str().ok_or(Error::Configuration)?,
                            document["execution"]["source_mount"]
                                .as_str()
                                .ok_or(Error::Configuration)?
                        ),
                    ]);
                }
                let worker = intent
                    .staging
                    .as_ref()
                    .and_then(|s| s.worker.as_ref())
                    .ok_or(Error::Configuration)?;
                argv.extend([
                    "--pwd".into(),
                    preparation.workdir.clone(),
                    backend["sif_path"]
                        .as_str()
                        .ok_or(Error::Configuration)?
                        .into(),
                    worker.program.clone(),
                    "script-prepare".into(),
                    "--plan-json".into(),
                    serde_json::to_string(preparation)?,
                ]);
                let mut stdout =
                    diagnostics.create_new_regular(OsStr::new("data-preparation.stdout"))?;
                let mut stderr =
                    diagnostics.create_new_regular(OsStr::new("data-preparation.stderr"))?;
                let result = process::execute(process::Request {
                    argv: &argv,
                    cwd: &self.config.root,
                    environment: process::Environment::Replace(&self.config.environment),
                    stdin: process::Input::Null,
                    stdout: process::Capture::File {
                        file: &mut stdout,
                        limit: 1024 * 1024,
                    },
                    stderr: process::Capture::File {
                        file: &mut stderr,
                        limit: 1024 * 1024,
                    },
                    deadline: Instant::now()
                        + Duration::from_secs(preparation.request.timeout_seconds + 30),
                    termination_grace: Duration::from_secs(3),
                    drain_grace: Duration::from_secs(3),
                    cancelled,
                })?;
                stdout.sync_all()?;
                stderr.sync_all()?;
                diagnostics.publish_new(OsStr::new("data-preparation.json"),&serde_json::to_vec(&serde_json::json!({"phase":"DATA_PREPARATION","exit_code":result.exit_code,"timed_out":result.timed_out,"cancelled":result.cancelled,"scheduler_submitted":false}))?)?;
                if result.exit_code != Some(0)
                    || result.failure.is_some()
                    || result.timed_out
                    || result.cancelled
                {
                    return Err(Error::Bootstrap);
                }
            }
            for dataset in &package.config.datasets {
                let root = Directory::open(std::path::Path::new(&dataset.receipt.root), false)?;
                ml_exp_io::datasets::verify(&root, &dataset.receipt.definition, true)?;
            }
        }
        let directories = self.shared_directories(&plan, true)?;
        let sealed = Sealed {
            plan,
            directory_identities: identities(&directories)?,
        };
        let attempt = &directories[3];
        attempt.publish_new(OsStr::new("attempt.json"), manifest)?;
        if let Some(package) = &sealed.plan.managed {
            attempt.publish_new(
                OsStr::new(ml_exp_core::slurm::managed::CONFIG_FILE),
                &package.config_bytes()?,
            )?;
            attempt.publish_new(
                OsStr::new(ml_exp_core::slurm::managed::MANIFEST_FILE),
                &package.manifest_bytes()?,
            )?;
        }
        attempt.publish_new(OsStr::new(MARKER), &json(&sealed)?)?;
        if identities(&directories)? != sealed.directory_identities {
            return Err(Error::Bootstrap);
        }
        Ok(())
    }
    pub(super) fn verify_shared(&self, intent: &FrozenIntent, ledger: &Directory) -> Result<()> {
        self.observe_shared(intent, ledger, true).map(|_| ())
    }
    pub(super) fn observe_shared(
        &self,
        intent: &FrozenIntent,
        ledger: &Directory,
        fresh: bool,
    ) -> Result<Option<(domain::Plan, Vec<Directory>)>> {
        if intent.staging.is_none() {
            return Ok(None);
        }
        let manifest =
            ledger.read_regular(OsStr::new(MANIFEST), helper::MAX_METADATA_BYTES as u64)?;
        let plan = self
            .shared_plan(intent, &manifest)?
            .ok_or(Error::Configuration)?;
        let directories = self.shared_directories(&plan, false)?;
        let attempt = &directories[3];
        let sealed: Sealed = serde_json::from_slice(
            &attempt.read_regular(OsStr::new(MARKER), helper::MAX_METADATA_BYTES as u64)?,
        )?;
        if sealed
            != (Sealed {
                plan: plan.clone(),
                directory_identities: identities(&directories)?,
            })
            || attempt.read_regular(
                OsStr::new("attempt.json"),
                helper::MAX_METADATA_BYTES as u64,
            )? != manifest
        {
            return Err(Error::Bootstrap);
        }
        if let Some(package) = &sealed.plan.managed {
            if fresh {
                for dataset in &package.config.datasets {
                    let root = Directory::open(std::path::Path::new(&dataset.receipt.root), false)?;
                    ml_exp_io::datasets::verify(&root, &dataset.receipt.definition, true)?;
                }
            }
            if attempt.read_regular(
                OsStr::new(ml_exp_core::slurm::managed::CONFIG_FILE),
                1024 * 1024,
            )? != package.config_bytes()?
                || attempt.read_regular(
                    OsStr::new(ml_exp_core::slurm::managed::MANIFEST_FILE),
                    1024 * 1024,
                )? != package.manifest_bytes()?
                || fresh
                    && (attempt.entry_exists(OsStr::new("worker"))?
                        || directories[5].entry_exists(OsStr::new("payload"))?)
            {
                return Err(Error::Bootstrap);
            }
        }
        Ok(Some((plan, directories)))
    }
}
