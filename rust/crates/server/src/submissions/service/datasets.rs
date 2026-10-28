//! Metadata-only API storage and bounded pinned-SSH gateway delivery.
use super::*;
use ml_exp_core::datasets::{self, Command, Request, State};
use ml_exp_io::process::{self, Capture, Environment, Input};
use std::time::{Duration, Instant};

impl Service {
    pub fn has_datasets(&self) -> bool {
        self.backends
            .values()
            .any(|backend| backend.data_helper_config.is_some())
    }
    fn dataset_backend(&self, project: &str, executor: &str) -> Result<&Backend> {
        ml_exp_core::identity::validate_id(project).map_err(|_| Error::Authority)?;
        ml_exp_core::identity::validate_id(executor).map_err(|_| Error::Authority)?;
        self.validate_workspace(project, "dataset-transfer")?;
        self.evidence
            .project(project, None)
            .map_err(|_| Error::Authority)?;
        self.backends
            .get(executor)
            .filter(|b| b.data_helper_config.is_some())
            .ok_or(Error::DatasetUnavailable)
    }
    fn data_name(project: &str, executor: &str, asset: &str) -> Result<String> {
        let digest = canonical::protocol_identity(&json!([project, executor, asset]))
            .map_err(|_| Error::Configuration)?;
        Ok(format!("dataset-{}.json", &digest[7..]))
    }
    fn data_read(&self, project: &str, backend: &Backend, asset: &str) -> Result<Option<State>> {
        if !datasets::asset_id(asset) {
            return Err(Error::DatasetUnavailable);
        }
        let name = Self::data_name(project, &backend.profile.target.backend_id, asset)?;
        if !self
            .slots
            .entry_exists(OsStr::new(&name))
            .map_err(|_| Error::DatasetUnavailable)?
        {
            return Ok(None);
        }
        let bytes = self
            .slots
            .read_regular(OsStr::new(&name), datasets::MAX_METADATA as u64)
            .map_err(|_| Error::DatasetUnavailable)?;
        let mapping = ml_exp_core::json::mapping(&bytes).map_err(|_| Error::DatasetUnavailable)?;
        let state: State = serde_json::from_value(Value::Object(mapping))
            .map_err(|_| Error::DatasetUnavailable)?;
        state.validate().map_err(|_| Error::DatasetUnavailable)?;
        if state.project != project
            || state.target != backend.profile.target
            || state.asset_id != asset
        {
            return Err(Error::DatasetUnavailable);
        }
        if let Some(receipt) = &state.ready {
            receipt
                .bind(project, &backend.profile.target, &backend.profile.data_root)
                .map_err(|_| Error::DatasetUnavailable)?;
        }
        Ok(Some(state))
    }
    pub fn dataset_state(&self, project: &str, executor: &str, asset: &str) -> Result<State> {
        let backend = self.dataset_backend(project, executor)?;
        self.data_read(project, backend, asset)?
            .ok_or(Error::DatasetMissing)
    }
    pub fn dataset_transfer(
        &self,
        project: &str,
        executor: &str,
        command: Command,
        payload: &[u8],
        cancelled: &AtomicBool,
    ) -> Result<State> {
        let backend = self.dataset_backend(project, executor)?;
        let fingerprint = backend.fingerprint()?;
        let request = Request {
            version: 1,
            project: project.into(),
            target: backend.profile.target.clone(),
            command,
        };
        request.validate().map_err(|_| Error::DatasetUnavailable)?;
        let mut frame =
            datasets::encode_request(&request).map_err(|_| Error::DatasetUnavailable)?;
        match &request.command {
            Command::Part { bytes, .. } if *bytes == payload.len() as u64 => {}
            Command::Part { .. } => return Err(Error::DatasetUnavailable),
            _ if !payload.is_empty() => return Err(Error::DatasetUnavailable),
            _ => {}
        }
        frame.extend_from_slice(payload);
        let mut ssh = backend.ssh.clone();
        ssh.helper_config_path = backend
            .data_helper_config
            .clone()
            .ok_or(Error::DatasetUnavailable)?;
        ssh.budget.timeout =
            Duration::from_secs(if matches!(request.command, Command::Complete { .. }) {
                1200
            } else {
                90
            });
        ssh.validate().map_err(|_| Error::Configuration)?;
        let mut argv = ssh.argv();
        let index = argv
            .iter()
            .position(|arg| arg == "slurm-helper")
            .ok_or(Error::Configuration)?;
        argv[index] = "data-helper".into();
        let environment = ssh.environment();
        let outcome = process::execute(process::Request {
            argv: &argv,
            cwd: &ssh.local_cwd,
            environment: Environment::Replace(&environment),
            stdin: Input::Bytes(&frame),
            stdout: Capture::Memory {
                limit: datasets::MAX_METADATA + 4,
            },
            stderr: Capture::Memory { limit: 65536 },
            deadline: Instant::now() + ssh.budget.timeout,
            termination_grace: ssh.budget.termination_grace,
            drain_grace: ssh.budget.drain_grace,
            cancelled,
        })
        .map_err(|_| Error::DatasetUnavailable)?;
        if outcome.exit_code != Some(0)
            || outcome.timed_out
            || outcome.cancelled
            || outcome.failure.is_some()
            || !outcome.stdin_complete
            || !outcome.stdout.complete
            || outcome.stdout.truncated
        {
            return Err(Error::DatasetUnavailable);
        }
        let reply = datasets::decode_reply(&outcome.stdout.bytes, &request)
            .map_err(|_| Error::DatasetUnavailable)?;
        if let Some(receipt) = &reply.state.ready {
            receipt
                .bind(project, &backend.profile.target, &backend.profile.data_root)
                .map_err(|_| Error::DatasetUnavailable)?;
        }
        self.dataset_backend(project, executor)?;
        if backend.fingerprint()? != fingerprint {
            return Err(Error::DatasetUnavailable);
        }
        // Remote effects happen outside this short metadata critical section.
        // A late response can reuse a newer state but cannot roll it back.
        let _lock = self.lock()?;
        let state = if let Some(old) = self.data_read(project, backend, &request.asset_id())? {
            if old.validate_successor(&reply.state).is_ok() {
                reply.state
            } else if reply.state.validate_successor(&old).is_ok() {
                old
            } else {
                return Err(Error::DatasetUnavailable);
            }
        } else {
            reply.state
        };
        let name = Self::data_name(project, executor, &state.asset_id)?;
        let bytes = serde_json::to_vec(&state).map_err(|_| Error::DatasetUnavailable)?;
        self.slots
            .replace_regular(OsStr::new(&name), &bytes)
            .map_err(|_| Error::DatasetUnavailable)?;
        Ok(state)
    }
}
