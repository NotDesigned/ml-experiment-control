use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::identity::{submission_marker, validate_attempt, validate_id};

use super::{Error, JobId, Resources, SlurmConfig, Storage, absolute_path, component, printable};

/// POSIX shell quoting: a word of safe characters stays bare, anything else is
/// single-quoted.
/// NUL is not representable in a process argument and is always refused.
pub fn shell_quote(value: &str) -> Result<String, Error> {
    if value.contains('\0') {
        return Err(Error::Configuration("command argument"));
    }
    if !value.is_empty()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&c))
    {
        Ok(value.to_owned())
    } else {
        Ok(format!("'{}'", value.replace('\'', "'\"'\"'")))
    }
}
pub fn shell_join(args: &[String]) -> Result<String, Error> {
    if args.is_empty() {
        return Err(Error::Configuration("command"));
    }
    args.iter()
        .map(|value| shell_quote(value))
        .collect::<Result<Vec<_>, _>>()
        .map(|values| values.join(" "))
}

pub fn scheduler_job_name(run_id: &str, attempt_id: &str) -> Result<String, Error> {
    validate_id(run_id).map_err(|_| Error::Identity("run_id"))?;
    validate_attempt(attempt_id).map_err(|_| Error::Identity("attempt_id"))?;
    if attempt_id.len() > 128 {
        return Err(Error::Identity("attempt_id"));
    }
    let raw = format!("{run_id}--{attempt_id}");
    if raw.len() <= 128 {
        Ok(raw)
    } else {
        Ok(format!("{}--{:x}", &raw[..113], Sha256::digest(raw.as_bytes()))[..127].to_owned())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfig {
    pub source_mount: String,
    pub workdir: String,
    #[serde(default)]
    pub managed_io: bool,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub env: std::collections::BTreeMap<String, String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobSpec {
    pub run_id: String,
    pub attempt_id: String,
    pub image_id: String,
    pub backend: SlurmConfig,
    pub storage: Storage,
    pub resources: Resources,
    pub execution: ExecutionConfig,
    pub command: Vec<String>,
}
impl JobSpec {
    pub fn validate(&self) -> Result<(), Error> {
        crate::environment::validate(&self.execution.env)
            .map_err(|_| Error::Configuration("execution environment"))?;
        scheduler_job_name(&self.run_id, &self.attempt_id)?;
        self.backend
            .validate(&self.resources, &self.storage, &self.image_id)?;
        for value in [&self.execution.source_mount, &self.execution.workdir] {
            if !absolute_path(value) {
                return Err(Error::Configuration("execution path"));
            }
        }
        if self.command.is_empty()
            || self.command[0].is_empty()
            || self.command.iter().any(|v| v.contains('\0'))
        {
            return Err(Error::Configuration("command"));
        }
        Ok(())
    }
}

pub fn render_job_with_datasets(
    spec: &JobSpec,
    token: Option<&str>,
    datasets: &[crate::datasets::Mount],
) -> Result<String, Error> {
    spec.validate()?;
    crate::datasets::mounts(datasets).map_err(|_| Error::Configuration("dataset bindings"))?;
    let mut dataset_binds = String::new();
    let mut roots = std::collections::BTreeSet::new();
    for mount in datasets {
        if !spec.execution.managed_io {
            return Err(Error::Configuration("dataset managed I/O"));
        }
        let source = &mount.receipt.root;
        let inputs = format!(
            "{}/attempts/{}/inputs/{}",
            spec.storage.run_dir, spec.attempt_id, mount.name
        );
        let mut destinations = vec![inputs, format!("/inputs/{}", mount.name)];
        if roots.insert(source) {
            destinations.push(source.clone());
        }
        for destination in destinations {
            dataset_binds.push_str(&format!(
                "  --bind {} \\\n",
                shell_quote(&format!("{source}:{destination}:ro"))?
            ));
        }
    }
    let backend = &spec.backend;
    let run_dir = &spec.storage.run_dir;
    let cache = backend
        .apptainer_cache_dir
        .clone()
        .unwrap_or_else(|| format!("{}/apptainer/cache", spec.storage.project_data_root));
    let temp = backend
        .apptainer_tmp_dir
        .clone()
        .unwrap_or_else(|| format!("{}/apptainer/tmp", spec.storage.project_data_root));
    let command = if spec.execution.env.is_empty() {
        shell_join(&spec.command)?
    } else {
        let mut argv = vec!["env".to_owned()];
        argv.extend(spec.execution.env.iter().map(|(k, v)| format!("{k}={v}")));
        argv.extend(spec.command.clone());
        shell_join(&argv)?
    };
    let source_check = if backend.oci_image.is_some() {
        String::new()
    } else {
        format!("test -d {}\n", shell_quote(&backend.source_dir)?)
    };
    let source_bind = if backend.oci_image.is_some() {
        String::new()
    } else {
        format!(
            "  --bind {}:{} \\\n",
            shell_quote(&backend.source_dir)?,
            shell_quote(&spec.execution.source_mount)?
        )
    };
    let unsquash = if backend.apptainer_unsquash {
        " --unsquash"
    } else {
        ""
    };
    let (managed_dirs, managed_binds) = if spec.execution.managed_io {
        let root = format!("{run_dir}/attempts/{}", spec.attempt_id);
        let inputs = shell_quote(&format!("{root}/inputs"))?;
        let outputs = shell_quote(&format!("{root}/outputs"))?;
        (
            format!("mkdir -p {inputs} {outputs}\n"),
            format!("  --bind {inputs}:/inputs \\\n  --bind {outputs}:/outputs \\\n"),
        )
    } else {
        (String::new(), String::new())
    };
    let marker = token
        .map(submission_marker)
        .transpose()
        .map_err(|_| Error::Identity("submission token"))?
        .unwrap_or_else(|| "ml-exp-dry-run".to_owned());
    let name = scheduler_job_name(&spec.run_id, &spec.attempt_id)?;
    let memory = spec
        .resources
        .memory_gb
        .map(|value| format!("#SBATCH --mem={value}G\n"))
        .unwrap_or_default();
    let out = shell_quote(&format!(
        "{run_dir}/attempts/{}/slurm-%j.out",
        spec.attempt_id
    ))?;
    let err = shell_quote(&format!(
        "{run_dir}/attempts/{}/slurm-%j.err",
        spec.attempt_id
    ))?;
    Ok(format!(
        "#!/usr/bin/env bash\n#SBATCH --partition={}\n#SBATCH --account={}\n#SBATCH --qos={}\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task={}\n{memory}#SBATCH --gres={}\n#SBATCH --time={}\n#SBATCH --job-name={name}\n#SBATCH --comment={}\n#SBATCH --output={out}\n#SBATCH --error={err}\n\nset -euo pipefail\nexport APPTAINER_CACHEDIR={}\nexport APPTAINER_TMPDIR={}\nexport BACKEND_JOB_ID=\"$SLURM_JOB_ID\"\nmkdir -p {} {} {}\nattempt_log_dir={}\nmkdir -p \"$attempt_log_dir\"\n{managed_dirs}{source_check}test -s {}\nsrun apptainer exec --nv{unsquash} \\\n  --bind {}:{} \\\n{source_bind}{managed_binds}{dataset_binds}  --pwd {} \\\n  {} \\\n  {command}\n",
        backend.partition,
        backend.account,
        backend.qos,
        spec.resources.cpus,
        backend.gres,
        backend.time,
        shell_quote(&marker)?,
        shell_quote(&cache)?,
        shell_quote(&temp)?,
        shell_quote(run_dir)?,
        shell_quote(&cache)?,
        shell_quote(&temp)?,
        shell_quote(&format!("{run_dir}/attempts/{}", spec.attempt_id))?,
        shell_quote(&backend.sif_path)?,
        shell_quote(&backend.mount_root)?,
        shell_quote(&backend.mount_root)?,
        shell_quote(&spec.execution.workdir)?,
        shell_quote(&backend.sif_path)?
    ))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandPlan {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub timeout_seconds: u32,
    pub max_stdout_bytes: u32,
}

/// All host-specific executable names, SSH aliases and multiplexing identity are
/// explicit inputs. No EXPERIMENTCTL_* or uid/pid lookup occurs inside a planner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SshConnection {
    pub ssh_binary: String,
    pub alias: String,
    pub control_path: Option<String>,
}
impl SshConnection {
    pub fn validate(&self) -> Result<(), Error> {
        if !component(&self.alias) {
            return Err(Error::Configuration("ssh_alias"));
        }
        if !printable(&self.ssh_binary, 4096) || self.ssh_binary.starts_with('-') {
            return Err(Error::Configuration("transport binary"));
        }
        if self
            .control_path
            .as_deref()
            .is_some_and(|path| !absolute_path(path))
        {
            return Err(Error::Configuration("control_path"));
        }
        Ok(())
    }
    pub fn remote_shell(&self, script: String) -> Result<CommandPlan, Error> {
        self.validate()?;
        if script.contains('\0') {
            return Err(Error::Configuration("remote command"));
        }
        let mut argv = vec![self.ssh_binary.clone(), "-o".into(), "BatchMode=yes".into()];
        if let Some(path) = &self.control_path {
            argv.extend([
                "-o".into(),
                "ControlMaster=auto".into(),
                "-o".into(),
                "ControlPersist=900".into(),
                "-o".into(),
                format!("ControlPath={path}"),
            ]);
        }
        argv.extend([self.alias.clone(), script]);
        Ok(CommandPlan {
            argv,
            env: BTreeMap::new(),
            timeout_seconds: 30,
            max_stdout_bytes: 1024 * 1024,
        })
    }
}

/// sbatch may add a federation cluster after ';'. Multiple rows, control bytes,
/// empty cluster identifiers and invalid IDs are not accepted as submission proof.
pub fn parse_submit_response(output: &str) -> Result<JobId, Error> {
    let value = output.trim_end_matches(['\r', '\n']);
    let parts: Vec<_> = value.split(';').collect();
    if parts.len() > 2 || parts.get(1).is_some_and(|v| !component(v)) {
        return Err(Error::Evidence("sbatch response"));
    }
    let id = JobId::parse(parts[0])?;
    if id.as_str().contains('_') {
        return Err(Error::Evidence("sbatch response"));
    }
    Ok(id)
}

pub fn allocation_plan(ssh: &SshConnection, partition: &str) -> Result<CommandPlan, Error> {
    if !component(partition) {
        return Err(Error::Configuration("partition"));
    }
    let script = format!(
        "scontrol show nodes --oneliner && printf '\\nML_EXPD_QUEUE\\n' && squeue -h -p {} -t PENDING -o '%i'",
        shell_quote(partition)?
    );
    let command = format!(
        "bash -o pipefail -c {}",
        shell_quote(&format!("({script}) 2>/dev/null | head -c 1048577"))?
    );
    let mut plan = ssh.remote_shell(command)?;
    plan.timeout_seconds = 12;
    plan.max_stdout_bytes = 1024 * 1024;
    Ok(plan)
}
