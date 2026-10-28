//! Small effect shell: approved roots, source/profile observations, a single
//! immutable directory publication. Authorship decisions remain in pure core.
use super::*;
use ml_exp_core::first_submission::native::creation::{self, Request};

impl Service {
    pub fn can_create_runs(&self) -> bool {
        !self.creation_roots.is_empty()
    }
    /// Explicit operator opt-in. Existing read locators do not imply write
    /// permission; only a single canonical discoverable root is accepted.
    pub fn with_run_creation(mut self, enabled: bool) -> Result<Self> {
        if enabled {
            for project in &self.settings.projects {
                if project.discovery_enabled() && project.run_roots.len() == 1 {
                    let directory = Directory::open(&project.run_roots[0], false)
                        .map_err(|_| Error::Configuration)?;
                    directory
                        .verify_private_parent()
                        .map_err(|_| Error::Configuration)?;
                    self.creation_roots
                        .insert(project.project.clone(), directory);
                }
            }
            if self.creation_roots.is_empty() {
                return Err(Error::Configuration);
            }
        } else {
            self.creation_roots.clear();
        }
        Ok(self)
    }
    pub fn create_run(&self, project: &str, request: Request, now: &str) -> Result<Value> {
        request.validate(project)?;
        let root = self
            .creation_roots
            .get(project)
            .ok_or(Error::RunCreationDisabled)?;
        let _guard = self
            .fence
            .existing_observation_guard(project, &request.run_id)
            .map_err(|_| Error::Authority)?;
        let _lock = self.lock()?;
        self.validate_workspace(project, &request.run_id)?;
        root.verify_private_parent().map_err(|_| Error::Authority)?;
        let project_document = self
            .evidence
            .project(project, None)
            .map_err(|_| Error::Authority)?;
        let backend = self
            .backends
            .get(&request.executor)
            .ok_or(Error::Authority)?;
        let runtime = backend
            .runtimes
            .get(&request.runtime_id)
            .ok_or(Error::Authority)?;
        let source = self
            .sources
            .read(project, &runtime.source_id)
            .map_err(|_| Error::Authority)?;
        let source_sha = canonical::protocol_identity(&source).map_err(|_| Error::Authority)?;
        let backend_sha = backend.fingerprint()?;
        let documents = creation::plan(
            project,
            &request,
            &backend.profile,
            runtime,
            creation::Authority {
                project_sha256: &project_document.raw_sha256,
                source_receipt_sha256: &source_sha,
                backend_config_sha256: &backend_sha,
            },
            now,
        )?;
        let discovery = self
            .evidence
            .discover(project, &experiments::DiscoveryLimits::default())
            .map_err(|_| Error::Authority)?;
        let existing = discovery
            .runs
            .iter()
            .find(|run| run.reference.run_id == request.run_id);
        if let Some(existing) = existing {
            if existing.reference.root_index != 0
                || existing.reference.relative_components != [request.run_id.clone()]
            {
                return Err(Error::Conflict);
            }
            let published = root
                .child(OsStr::new(&request.run_id), false)
                .map_err(|_| Error::Authority)?;
            return creation_receipt(project, &request, &published, &documents, true);
        }
        if root
            .entry_exists(OsStr::new(&request.run_id))
            .map_err(|_| Error::Authority)?
        {
            return Err(Error::Conflict);
        }
        // The staging parent is outside discovery. The no-replace directory
        // rename publishes all manifests and states together or none of them.
        let mut random = [0u8; 16];
        getrandom::fill(&mut random).map_err(|_| Error::Configuration)?;
        let stage = self
            .slots
            .create_new_child(OsStr::new(&format!(
                "run-{:032x}",
                u128::from_be_bytes(random)
            )))
            .map_err(|_| Error::Uncertain)?;
        let result = (|| {
            publish_json(&stage, "manifest.json", &documents.run)?;
            publish_json(&stage, "request.json", &documents.request)?;
            publish_json(&stage, "status.json", &documents.status)?;
            let attempts = stage
                .create_new_child(OsStr::new("attempts"))
                .map_err(|_| Error::Uncertain)?;
            let attempt = attempts
                .create_new_child(OsStr::new("attempt-001"))
                .map_err(|_| Error::Uncertain)?;
            publish_json(&attempt, "attempt.json", &documents.attempt)?;
            publish_json(&attempt, "status.json", &documents.status)?;
            self.validate_workspace(project, &request.run_id)?;
            self.evidence
                .project(project, Some(&project_document.raw_sha256))
                .map_err(|_| Error::Authority)?;
            if backend.fingerprint()? != backend_sha
                || self
                    .sources
                    .read(project, &runtime.source_id)
                    .map_err(|_| Error::Authority)?
                    != source
            {
                return Err(Error::Authority);
            }
            let published = stage
                .publish_directory(root, OsStr::new(&request.run_id))
                .map_err(|_| Error::Uncertain)?;
            self.validate_workspace(project, &request.run_id)
                .map_err(|_| Error::Uncertain)?;
            let reference = RunRef {
                project: project.into(),
                run_id: request.run_id.clone(),
                root_index: 0,
                relative_components: vec![request.run_id.clone()],
            };
            let snapshot = self
                .evidence
                .snapshot_metadata(&reference)
                .map_err(|_| Error::Uncertain)?;
            if snapshot.run.raw_sha256
                != helper::sha256_bytes(
                    &serde_json::to_vec(&documents.run).map_err(|_| Error::Configuration)?,
                )
                || snapshot.attempts.len() != 1
                || !snapshot.attempts[0].provenance.compared_fields_complete
            {
                return Err(Error::Uncertain);
            }
            creation_receipt(project, &request, &published, &documents, false)
        })();
        // A renamed stage fails its old attachment check, so cleanup can never
        // follow the published destination after a post-rename error.
        if stage.verify_attached().is_ok() && cleanup_stage(&self.slots, &stage).is_err() {
            return Err(Error::Uncertain);
        }
        result
    }
}
fn publish_json(directory: &Directory, name: &str, value: &Value) -> Result<()> {
    directory
        .publish_new(
            OsStr::new(name),
            &serde_json::to_vec(value).map_err(|_| Error::Configuration)?,
        )
        .map_err(|_| Error::Uncertain)
}
fn read_json(directory: &Directory, name: &str) -> Result<Value> {
    let bytes = directory
        .read_regular(OsStr::new(name), creation::MAX_REQUEST_BYTES as u64 * 4)
        .map_err(|_| Error::Authority)?;
    let value = experiments::parse_mapping(&bytes, experiments::Format::Json)
        .map(Value::Object)
        .map_err(|_| Error::Authority)?;
    // These files are our immutable publications. A semantic JSON match alone
    // cannot establish the original byte hash reported by the creation receipt.
    if serde_json::to_vec(&value).map_err(|_| Error::Authority)? != bytes {
        return Err(Error::Authority);
    }
    Ok(value)
}
fn creation_receipt(
    project: &str,
    request: &Request,
    directory: &Directory,
    expected: &creation::Documents,
    reused: bool,
) -> Result<Value> {
    let run = read_json(directory, "manifest.json")?;
    let attempts = directory
        .child(OsStr::new("attempts"), false)
        .map_err(|_| Error::Authority)?;
    let attempt_dir = attempts
        .child(OsStr::new("attempt-001"), false)
        .map_err(|_| Error::Authority)?;
    let attempt = read_json(&attempt_dir, "attempt.json")?;
    let original = read_json(directory, "request.json")?;
    if !creation::same_creation(expected, &run, &attempt, &original) {
        return Err(Error::Conflict);
    }
    creation::receipt(project, request, &run, &attempt, reused).map_err(Error::from)
}
fn cleanup_stage(parent: &Directory, stage: &Directory) -> Result<()> {
    if stage
        .entry_exists(OsStr::new("attempts"))
        .map_err(|_| Error::Authority)?
    {
        let attempts = stage
            .child(OsStr::new("attempts"), false)
            .map_err(|_| Error::Authority)?;
        if attempts
            .entry_exists(OsStr::new("attempt-001"))
            .map_err(|_| Error::Authority)?
        {
            let child = attempts
                .child(OsStr::new("attempt-001"), false)
                .map_err(|_| Error::Authority)?;
            clean_files(&child, &["attempt.json", "status.json"])?;
            attempts
                .remove_staged_empty_child(&child)
                .map_err(|_| Error::Authority)?;
        }
        stage
            .remove_staged_empty_child(&attempts)
            .map_err(|_| Error::Authority)?;
    }
    clean_files(stage, &["manifest.json", "request.json", "status.json"])?;
    parent
        .remove_staged_empty_child(stage)
        .map_err(|_| Error::Authority)
}
fn clean_files(directory: &Directory, names: &[&str]) -> Result<()> {
    for name in names {
        if directory
            .entry_exists(OsStr::new(name))
            .map_err(|_| Error::Authority)?
        {
            let file = directory
                .open_regular_read(OsStr::new(name))
                .map_err(|_| Error::Authority)?;
            directory
                .remove_staged_regular(OsStr::new(name), &file)
                .map_err(|_| Error::Authority)?;
        }
    }
    Ok(())
}
