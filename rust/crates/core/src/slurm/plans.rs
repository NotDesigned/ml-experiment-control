use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::identity::{submission_marker, validate_attempt, validate_id};
use crate::state::RunState;

use super::{
    Error, JobId, JobStatus, Ownership, Resources, SlurmConfig, Storage, absolute_path, component,
    printable,
};

/// POSIX shell quoting compatible with Python shlex.quote for UTF-8 strings.
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

/// The generated script preserves the deployed Python renderer byte-for-byte
/// for accepted equivalent inputs. This constructs data; it never writes a file.
pub fn render_job(spec: &JobSpec, token: Option<&str>) -> Result<String, Error> {
    render_job_with_datasets(spec, token, &[])
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Observe,
    ClaimSubmission,
    CopyManifest,
    CopyScript,
    Submit,
    Cancel,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandPlan {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub effect: Effect,
    pub timeout_seconds: u32,
    pub max_stdout_bytes: u32,
}

/// All host-specific executable names, SSH aliases and multiplexing identity are
/// explicit inputs. No EXPERIMENTCTL_* or uid/pid lookup occurs inside a planner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SshConnection {
    pub ssh_binary: String,
    pub rsync_binary: String,
    pub alias: String,
    pub control_path: Option<String>,
}
impl SshConnection {
    pub fn validate(&self) -> Result<(), Error> {
        if !component(&self.alias) {
            return Err(Error::Configuration("ssh_alias"));
        }
        for binary in [&self.ssh_binary, &self.rsync_binary] {
            if !printable(binary, 4096) || binary.starts_with('-') {
                return Err(Error::Configuration("transport binary"));
            }
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
    pub fn remote_shell(&self, script: String, effect: Effect) -> Result<CommandPlan, Error> {
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
            effect,
            timeout_seconds: 30,
            max_stdout_bytes: 1024 * 1024,
        })
    }
    pub fn remote(&self, args: &[String], effect: Effect) -> Result<CommandPlan, Error> {
        self.remote_shell(shell_join(args)?, effect)
    }
    fn transport(&self) -> Result<String, Error> {
        let mut args = vec![self.ssh_binary.clone()];
        if let Some(path) = &self.control_path {
            args.extend([
                "-o".into(),
                "ControlMaster=auto".into(),
                "-o".into(),
                "ControlPersist=900".into(),
                "-o".into(),
                format!("ControlPath={path}"),
            ]);
        }
        shell_join(&args)
    }
    fn copy(&self, local: &str, remote: &str, effect: Effect) -> Result<CommandPlan, Error> {
        self.validate()?;
        if !absolute_path(local) || !absolute_path(remote) {
            return Err(Error::Configuration("copy path"));
        }
        Ok(CommandPlan {
            argv: vec![
                self.rsync_binary.clone(),
                "-a".into(),
                "--protect-args".into(),
                "-e".into(),
                self.transport()?,
                local.into(),
                format!("{}:{remote}", self.alias),
            ],
            env: BTreeMap::new(),
            effect,
            timeout_seconds: 300,
            max_stdout_bytes: 16384,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmissionRequest {
    pub scheduler_name: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmissionIntent {
    pub submission_token: String,
    pub request: SubmissionRequest,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitPlan {
    pub scheduler_name: String,
    pub marker: String,
    pub remote_script: String,
    pub claim_directory: String,
    pub local_script: String,
    pub script_mode: u32,
    pub script: String,
    /// Execute in order. Each step needs its own durable effect marker/evidence.
    pub steps: Vec<CommandPlan>,
    pub requires_authorization: bool,
    pub requires_prior_reconciliation: bool,
    pub requires_live_preflight: bool,
    pub automatic_replay_allowed: bool,
}
pub fn submission_request(run_id: &str, attempt_id: &str) -> Result<SubmissionRequest, Error> {
    Ok(SubmissionRequest {
        scheduler_name: scheduler_job_name(run_id, attempt_id)?,
    })
}
pub fn plan_submit(
    spec: &JobSpec,
    intent: &SubmissionIntent,
    ssh: &SshConnection,
    local_manifest: &str,
    local_script: &str,
) -> Result<SubmitPlan, Error> {
    spec.validate()?;
    if ssh.alias != spec.backend.ssh_alias {
        return Err(Error::Configuration("submission SSH scope"));
    }
    let request = submission_request(&spec.run_id, &spec.attempt_id)?;
    if intent.request != request {
        return Err(Error::Identity("scheduler_name"));
    }
    let marker = submission_marker(&intent.submission_token)
        .map_err(|_| Error::Identity("submission token"))?;
    let remote = format!(
        "{}/controller-{}.sbatch",
        spec.storage.run_dir, spec.attempt_id
    );
    let claim = format!("{}/.submission-{}", spec.storage.run_dir, spec.attempt_id);
    let claim_command = format!(
        "{} && {}",
        shell_join(&[
            "mkdir".into(),
            "-p".into(),
            spec.storage.run_dir.clone(),
            format!("{}/attempts/{}", spec.storage.run_dir, spec.attempt_id)
        ])?,
        shell_join(&["mkdir".into(), claim.clone()])?
    );
    let steps = vec![
        ssh.remote_shell(claim_command, Effect::ClaimSubmission)?,
        ssh.copy(
            local_manifest,
            &format!("{}/manifest.yaml", spec.storage.run_dir),
            Effect::CopyManifest,
        )?,
        ssh.copy(local_script, &remote, Effect::CopyScript)?,
        ssh.remote(
            &["sbatch".into(), "--parsable".into(), remote.clone()],
            Effect::Submit,
        )?,
    ];
    Ok(SubmitPlan {
        scheduler_name: request.scheduler_name,
        marker,
        remote_script: remote,
        claim_directory: claim,
        local_script: local_script.into(),
        script_mode: 0o600,
        script: render_job(spec, Some(&intent.submission_token))?,
        steps,
        requires_authorization: true,
        requires_prior_reconciliation: true,
        requires_live_preflight: true,
        automatic_replay_allowed: false,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimResult {
    Fresh,
    ExistingReconcileRequired,
    UnknownDoNotRetry,
}
pub fn claim_result(exit_code: Option<i32>) -> ClaimResult {
    match exit_code {
        Some(0) => ClaimResult::Fresh,
        Some(1) => ClaimResult::ExistingReconcileRequired,
        _ => ClaimResult::UnknownDoNotRetry,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CancelPlan {
    AlreadyTerminal,
    Execute(CommandPlan),
}
/// A cancellation outbox and current authorization remain mandatory in the
/// server. This plan refuses foreign IDs even when a caller has a status row.
pub fn plan_cancel(
    ssh: &SshConnection,
    ownership: &Ownership,
    status: &JobStatus,
) -> Result<CancelPlan, Error> {
    let Some(job) = ownership.exact_job() else {
        return Err(Error::Ownership);
    };
    if ssh.alias != ownership.ssh_alias
        || ssh.alias != status.ssh_alias
        || job != &status.job_id
        || ownership.scheduler_name != status.scheduler_name
    {
        return Err(Error::Ownership);
    }
    if status.state.is_terminal() {
        return Ok(CancelPlan::AlreadyTerminal);
    }
    Ok(CancelPlan::Execute(ssh.remote(
        &["scancel".into(), job.as_str().into()],
        Effect::Cancel,
    )?))
}

pub fn accounting_plan(ssh: &SshConnection, job: &JobId) -> Result<CommandPlan, Error> {
    ssh.remote(
        &[
            "sacct".into(),
            "-j".into(),
            job.as_str().into(),
            "-X".into(),
            "-n".into(),
            "-P".into(),
            "-o".into(),
            "JobID,JobName,Partition,State,Elapsed,ExitCode".into(),
        ],
        Effect::Observe,
    )
}
pub fn queue_status_plan(ssh: &SshConnection, job: &JobId) -> Result<CommandPlan, Error> {
    ssh.remote(
        &[
            "squeue".into(),
            "-j".into(),
            job.as_str().into(),
            "-h".into(),
            "-o".into(),
            "%i|%j|%P|%T|%M|0:0|%R".into(),
        ],
        Effect::Observe,
    )
}
pub fn identity_plans(
    ssh: &SshConnection,
    user: &str,
    accounting_since: &str,
) -> Result<Vec<CommandPlan>, Error> {
    if !component(user) {
        return Err(Error::Configuration("scheduler user"));
    }
    // The shell supplies an absolute start date. Never read the clock here or
    // hide a relative 'now-30days' window in a supposedly exact lookup plan.
    if !valid_since(accounting_since) {
        return Err(Error::Configuration("accounting_since"));
    }
    Ok(vec![
        ssh.remote(
            &[
                "squeue".into(),
                "-u".into(),
                user.into(),
                "-h".into(),
                "-o".into(),
                "%i|%j|%k".into(),
            ],
            Effect::Observe,
        )?,
        ssh.remote(
            &[
                "sacct".into(),
                "-S".into(),
                accounting_since.into(),
                "-u".into(),
                user.into(),
                "-X".into(),
                "-n".into(),
                "-P".into(),
                "-o".into(),
                "JobIDRaw,JobName,Comment".into(),
            ],
            Effect::Observe,
        )?,
    ])
}
pub fn live_plans(
    ssh: &SshConnection,
    partition: &str,
    user: &str,
) -> Result<Vec<CommandPlan>, Error> {
    if !component(partition) || !component(user) {
        return Err(Error::Configuration("live query scope"));
    }
    Ok(vec![
        ssh.remote(
            &[
                "sinfo".into(),
                "-h".into(),
                "-p".into(),
                partition.into(),
                "-o".into(),
                "%P|%a|%l|%G".into(),
            ],
            Effect::Observe,
        )?,
        ssh.remote(
            &[
                "sacctmgr".into(),
                "-n".into(),
                "-P".into(),
                "show".into(),
                "assoc".into(),
                "where".into(),
                format!("user={user}"),
                "format=User,Account,Partition,QOS,DefaultQOS".into(),
            ],
            Effect::Observe,
        )?,
    ])
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
    let mut plan = ssh.remote_shell(command, Effect::Observe)?;
    plan.timeout_seconds = 12;
    plan.max_stdout_bytes = 1024 * 1024;
    Ok(plan)
}

pub fn log_probe_plan(
    ssh: &SshConnection,
    paths: &[String],
    tail: u32,
) -> Result<CommandPlan, Error> {
    if paths.is_empty()
        || paths.len() > 8
        || !(1..=10000).contains(&tail)
        || paths.iter().any(|path| !absolute_path(path))
    {
        return Err(Error::Configuration("log probe"));
    }
    let candidates = paths
        .iter()
        .map(|value| shell_quote(value))
        .collect::<Result<Vec<_>, _>>()?
        .join(" ");
    ssh.remote_shell(format!("for path in {candidates}; do if test -f \"$path\"; then printf '%s\\n' \"$path\"; tail -n {tail} -- \"$path\"; exit 0; fi; done; exit 1"), Effect::Observe)
}

pub fn predicate_result(exit_code: Option<i32>) -> Result<bool, Error> {
    match exit_code {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(Error::Unavailable),
    }
}

/// Only scheduler-terminal states permit offline result recovery. This does not
/// prove that outputs exist or that training itself completed successfully.
pub fn is_scheduler_terminal(state: RunState) -> bool {
    state.is_terminal()
}

fn valid_since(value: &str) -> bool {
    if value.len() != 10
        || !value.bytes().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                c == b'-'
            } else {
                c.is_ascii_digit()
            }
        })
    {
        return false;
    }
    let Ok(year) = value[..4].parse::<i32>() else {
        return false;
    };
    let Ok(month) = value[5..7]
        .parse::<u8>()
        .map_err(|_| ())
        .and_then(|m| time::Month::try_from(m).map_err(|_| ()))
    else {
        return false;
    };
    let Ok(day) = value[8..].parse::<u8>() else {
        return false;
    };
    time::Date::from_calendar_date(year, month, day).is_ok()
}
