//! Uses the configured SSH identity and pinned known-hosts policy. The only
//! remote command is the pure scheduler observation plan, with bounded output.
use super::*;
use ml_exp_core::capacity::{self, Observation};
impl Service {
    pub fn catalogue(&self) -> Vec<Value> {
        self.backends.iter().map(|(id,b)|json!({"executor_id":id,"kind":"slurm","enabled":self.enabled,"run_creation":self.can_create_runs(),"limits":{"gpus":b.profile.max_gpus,"cpus":b.profile.max_cpus,"memory_gb":b.profile.max_memory_gb,"walltime_seconds":b.profile.max_walltime_seconds},"partition":b.profile.partition,
            "capabilities":{"oci_to_sif":true,"persistent_checkpoints":true,"dataset_upload":b.data_helper_config.is_some(),"data_preparation":if b.script_preparation{serde_json::json!("before_job")}else{serde_json::Value::Null},"metrics":true,"backend_result_stream":true,"capacity_query":true},"live_availability_checked":false})).collect()
    }
    pub fn accepts(&self, executor: &str) -> bool {
        self.backends.contains_key(executor)
    }
    pub fn capacity(&self, executor: &str, refresh: bool, now: &str) -> Result<Value> {
        let backend = self.backends.get(executor).ok_or(Error::Configuration)?;
        let name = format!("capacity-{executor}.json");
        let _effect = if refresh {
            let file = self
                .slots
                .open_lock(OsStr::new(&format!("capacity-{executor}.lock")))
                .map_err(|_| Error::Authority)?;
            fs2::FileExt::try_lock_exclusive(&file).map_err(|_| Error::Authority)?;
            Some(file)
        } else {
            None
        };
        let old = if self
            .slots
            .entry_exists(OsStr::new(&name))
            .map_err(|_| Error::Authority)?
        {
            serde_json::from_slice::<Observation>(
                &self
                    .slots
                    .read_regular(OsStr::new(&name), 4 * 1024 * 1024)
                    .map_err(|_| Error::Authority)?,
            )
            .map_err(|_| Error::Authority)?
        } else {
            capacity::unknown(executor).map_err(|_| Error::Configuration)?
        };
        if old.executor_id != executor {
            return Err(Error::Authority);
        }
        if !refresh {
            return Ok(json!(
                capacity::view(old, now).map_err(|_| Error::Authority)?
            ));
        }
        let data = (|| {
            let plan = ml_exp_core::slurm::allocation_plan(
                &ml_exp_core::slurm::SshConnection {
                    ssh_binary: backend.ssh.ssh_binary.to_string_lossy().into_owned(),
                    rsync_binary: "rsync".into(),
                    alias: backend.ssh.target_alias.clone(),
                    control_path: None,
                },
                &backend.profile.partition,
            )
            .map_err(|_| Error::Configuration)?;
            let mut argv = backend.ssh.argv();
            let at = argv
                .iter()
                .position(|s| s == &backend.ssh.helper_path)
                .ok_or(Error::Configuration)?;
            argv.truncate(at);
            argv.push(plan.argv.last().ok_or(Error::Configuration)?.clone());
            let environment = backend.ssh.environment();
            let result = ml_exp_io::process::execute(ml_exp_io::process::Request {
                argv: &argv,
                cwd: &backend.ssh.local_cwd,
                environment: ml_exp_io::process::Environment::Replace(&environment),
                stdin: ml_exp_io::process::Input::Null,
                stdout: ml_exp_io::process::Capture::Memory { limit: 1024 * 1024 },
                stderr: ml_exp_io::process::Capture::Memory { limit: 65536 },
                deadline: std::time::Instant::now() + std::time::Duration::from_secs(12),
                termination_grace: std::time::Duration::from_secs(2),
                drain_grace: std::time::Duration::from_secs(2),
                cancelled: &std::sync::atomic::AtomicBool::new(false),
            })
            .map_err(|_| Error::Authority)?;
            if result.exit_code != Some(0)
                || result.timed_out
                || result.cancelled
                || result.failure.is_some()
                || result.stdout.truncated
                || !result.stdout.complete
            {
                return Err(Error::Authority);
            }
            let output = std::str::from_utf8(&result.stdout.bytes).map_err(|_| Error::Authority)?;
            let snapshot = ml_exp_core::slurm::parse_allocation(output, &backend.profile.partition)
                .map_err(|_| Error::Authority)?;
            let mut value = json!(snapshot);
            value["source"] = json!("slurm");
            value["scope"] = json!("partition");
            value["partition"] = json!(backend.profile.partition);
            Ok(value)
        })();
        let next = capacity::refresh(old, data.map_err(|_: Error| "unavailable"), now)
            .map_err(|_| Error::Authority)?;
        self.slots
            .replace_regular(
                OsStr::new(&name),
                &serde_json::to_vec(&next).map_err(|_| Error::Authority)?,
            )
            .map_err(|_| Error::Authority)?;
        Ok(json!(next))
    }
}
