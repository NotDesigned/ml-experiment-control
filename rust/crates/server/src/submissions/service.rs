use crate::{
    backends::slurm::{
        SshConfig,
        action::{Execute, Runner},
    },
    deletion::RunFence,
    experiments::{self, EvidenceDocument, RunRef},
    operations::{self, ClaimContext},
    projects, read_config, sources,
};
use fs2::FileExt;
use ml_exp_core::{
    canonical,
    first_submission::native::{self, Approval, Binding, Prepared, Profile, Runtime},
    operations::{FrozenFiles, Policy},
    slurm::helper,
};
use ml_exp_io::directory::{Directory, effective_uid};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fs::File,
    sync::{Arc, atomic::AtomicBool},
};

const NAMESPACE: &str = "submissions";
const LOCK: &str = ".submissions.lock";
const MAX_SLOT_BYTES: u64 = 24 * 1024 * 1024;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("native submission configuration or retained workspace is invalid")]
    Configuration,
    #[error("native submission preparation is busy; query the existing Run before retry")]
    Busy,
    #[error("no native preparation is recorded for this Run")]
    Missing,
    #[error("remote worker evidence is temporarily unavailable; retained evidence is unchanged")]
    ObservationUnavailable,
    #[error(
        "remote worker history conflicts with its accepted sample; previous evidence is retained"
    )]
    ObservationConflict,
    #[error("approved Run, first Attempt, source or runtime evidence is unavailable or changed")]
    Authority,
    #[error(
        "the first-submission identity is already bound to a different request; no new effect is permitted"
    )]
    Conflict,
    #[error("native submission execution is disabled")]
    Disabled,
    #[error("native Run creation is disabled for this Project")]
    RunCreationDisabled,
    #[error("native preparation may have been saved; query the same Run before retry")]
    Uncertain,
    #[error(transparent)]
    Admission(#[from] native::Error),
    #[error(transparent)]
    Action(#[from] operations::Error),
    #[error(transparent)]
    Runner(#[from] crate::backends::slurm::action::Error),
}
type Result<T> = std::result::Result<T, Error>;

/// Constructed only by the trusted operator/builder shell. A runtime entry is
/// an already-published code image and SIF materialization, not client evidence.
pub struct Backend {
    pub profile: Profile,
    pub ssh: SshConfig,
    pub runtimes: BTreeMap<String, Runtime>,
}
#[derive(Clone, Copy)]
pub struct Context<'a> {
    pub now: &'a str,
    pub expires: &'a str,
    pub transition_id: &'a str,
}
pub struct Service {
    settings: read_config::Settings,
    workspace: String,
    directory: Directory,
    slots: Directory,
    lock_anchor: Arc<File>,
    evidence: experiments::Store,
    fence: RunFence,
    actions: operations::Store,
    sources: Arc<sources::Service>,
    lifecycle: Option<Arc<projects::Service>>,
    backends: BTreeMap<String, Backend>,
    enabled: bool,
    creation_roots: BTreeMap<String, Directory>,
}
struct Facts {
    reference: RunRef,
    run: Value,
    attempt: Value,
    binding: Binding,
    scheduler_state: String,
    submitted: bool,
    backend: String,
    runtime: String,
}
impl Backend {
    pub(super) fn validate(&self) -> Result<()> {
        self.profile.validate()?;
        self.ssh.validate().map_err(|_| Error::Configuration)?;
        if self.profile.target != self.ssh.target
            || self.profile.ssh_alias != self.ssh.target_alias
            || self.runtimes.is_empty()
            || self.runtimes.len() > 4096
            || self
                .runtimes
                .iter()
                .any(|(id, runtime)| id != &runtime.runtime_id)
        {
            return Err(Error::Configuration);
        }
        Ok(())
    }
    fn fingerprint(&self) -> Result<String> {
        let ssh = &self.ssh;
        let configuration = ml_exp_io::read_private(&ssh.config_file, 1024 * 1024, true)
            .map_err(|_| Error::Authority)?;
        let hosts = ml_exp_io::read_private(&ssh.known_hosts_file, 1024 * 1024, true)
            .map_err(|_| Error::Authority)?;
        canonical::protocol_identity(&json!({"target":ssh.target,"binary":ssh.ssh_binary,"config_path":ssh.config_file,
            "config_sha256":helper::sha256_bytes(&configuration),"known_hosts_path":ssh.known_hosts_file,"known_hosts_sha256":helper::sha256_bytes(&hosts),
            "alias":ssh.target_alias,"helper":ssh.helper_path,"helper_config":ssh.helper_config_path,"cwd":ssh.local_cwd,"agent":ssh.agent_socket,
            "timeout_nanos":ssh.budget.timeout.as_nanos().to_string(),"connect_timeout":ssh.budget.connect_timeout_seconds,
            "termination_grace_nanos":ssh.budget.termination_grace.as_nanos().to_string(),"drain_grace_nanos":ssh.budget.drain_grace.as_nanos().to_string()}))
            .map_err(|_|Error::Configuration)
    }
}
mod authority;
mod creation;
mod observation;
mod preparation;

impl Service {
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn matches_configuration(
        &self,
        registry: &std::path::Path,
        files: &BTreeMap<String, std::path::PathBuf>,
        workspace: &str,
        sources: Option<&Arc<sources::Service>>,
        lifecycle: Option<&Arc<projects::Service>>,
        read_scope_sha256: &str,
    ) -> bool {
        self.settings.registry_root == registry
            && self
                .settings
                .identity()
                .is_ok_and(|scope| scope == read_scope_sha256)
            && self.workspace == workspace
            && self
                .settings
                .projects
                .iter()
                .map(|p| (p.project.clone(), p.project_file.clone()))
                .collect::<BTreeMap<_, _>>()
                == *files
            && sources.is_some_and(|source| Arc::ptr_eq(source, &self.sources))
            && match (&self.lifecycle, lifecycle) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
    /// Resolve only the original immutable slot; a lost response is not a new
    /// preparation request and this lookup never allocates a replacement ID.
    pub fn for_run(&self, project: &str, run: &str, now: &str) -> Result<Value> {
        self.validate_workspace(project, run)?;
        let prepared = self.saved(project, run)?.ok_or(Error::Missing)?;
        self.get(
            prepared.plan["action_id"]
                .as_str()
                .ok_or(Error::Authority)?,
            now,
        )
    }
    /// Explicit offline bootstrap only. Existing namespaces are never repaired
    /// or reset. Ordinary open/GET has no initialization side effects.
    pub fn initialize(settings: &read_config::Settings) -> Result<()> {
        settings.validate().map_err(|_| Error::Configuration)?;
        let (root, _) =
            read_config::workspace(&settings.registry_root).map_err(|_| Error::Configuration)?;
        let namespace = root
            .create_new_child(OsStr::new(NAMESPACE))
            .map_err(|_| Error::Configuration)?;
        namespace
            .create_new_child(OsStr::new("slots"))
            .map_err(|_| Error::Configuration)?;
        let lock = namespace
            .create_new_regular(OsStr::new(LOCK))
            .map_err(|_| Error::Configuration)?;
        lock.sync_all().map_err(|_| Error::Configuration)?;
        operations::Store::open(settings.registry_root.join("actions"))?;
        namespace
            .verify_private_parent()
            .map_err(|_| Error::Configuration)
    }
    pub fn open(
        settings: read_config::Settings,
        sources: Arc<sources::Service>,
        lifecycle: Option<Arc<projects::Service>>,
        backends: Vec<Backend>,
        enabled: bool,
    ) -> Result<Self> {
        settings.validate().map_err(|_| Error::Configuration)?;
        let (root, workspace) =
            read_config::workspace(&settings.registry_root).map_err(|_| Error::Configuration)?;
        let files = settings
            .projects
            .iter()
            .map(|p| (p.project.clone(), p.project_file.clone()))
            .collect();
        if !sources.matches_configuration(
            &settings.registry_root,
            &files,
            &workspace,
            lifecycle.as_ref(),
        ) || lifecycle
            .as_ref()
            .is_some_and(|l| !l.matches_configuration(&settings.registry_root, &files, &workspace))
            || backends.is_empty()
            || backends.len() > 64
        {
            return Err(Error::Configuration);
        }
        let mut bindings = BTreeMap::new();
        let mut aliases = std::collections::BTreeSet::new();
        for backend in backends {
            backend.validate()?;
            if !aliases.insert((
                backend.profile.ssh_alias.clone(),
                backend.profile.partition.clone(),
            )) || bindings
                .insert(backend.profile.target.backend_id.clone(), backend)
                .is_some()
            {
                return Err(Error::Configuration);
            }
        }
        let directory = root
            .child(OsStr::new(NAMESPACE), false)
            .map_err(|_| Error::Configuration)?;
        directory
            .verify_private_parent()
            .map_err(|_| Error::Configuration)?;
        let slots = directory
            .child(OsStr::new("slots"), false)
            .map_err(|_| Error::Configuration)?;
        let lock_anchor = Arc::new(
            directory
                .open_existing_lock(OsStr::new(LOCK))
                .map_err(|_| Error::Configuration)?,
        );
        let actions = operations::Store::open_existing(settings.registry_root.join("actions"))?;
        let fence = RunFence::new(&settings.registry_root);
        let evidence = experiments::Store::open(settings.bindings(), fence.clone())
            .map_err(|_| Error::Configuration)?;
        Ok(Self {
            settings,
            workspace,
            directory,
            slots,
            lock_anchor,
            evidence,
            fence,
            actions,
            sources,
            lifecycle,
            backends: bindings,
            enabled,
            creation_roots: BTreeMap::new(),
        })
    }
    fn public(&self, snapshot: Value, now: &str) -> Result<Value> {
        Ok(ml_exp_core::operations::submission_view(&snapshot, now)
            .map_err(operations::Error::from)?)
    }
    pub fn get(&self, action: &str, now: &str) -> Result<Value> {
        let prepared = self.load(action)?;
        let snapshot = self.actions.observe(action)?;
        self.check(&prepared)?;
        self.public(snapshot, now)
    }
    pub fn authorize(
        &self,
        action: &str,
        actor: &str,
        note: &str,
        context: Context<'_>,
    ) -> Result<Value> {
        let prepared = self.load(action)?;
        let _guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        self.check(&prepared)?;
        let snapshot =
            self.actions
                .authorize(action, actor, note, context.now, context.transition_id)?;
        self.check(&prepared).map_err(|_| Error::Uncertain)?;
        self.public(snapshot, context.now)
    }
    pub fn execute(
        &self,
        action: &str,
        confirmation: &str,
        context: Context<'_>,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        if !self.enabled {
            return Err(Error::Disabled);
        }
        // Held through the provider call and durable finish. A later explicit
        // reconcile can distinguish a dead claimant from an executor in flight.
        let _execution_lock = self.lock()?;
        let prepared = self.load(action)?;
        let guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        if self.actions.observe(action)?["execution"]["status"] == "AUTHORIZED" {
            let facts = self.facts(&prepared.binding.project, &prepared.binding.run_id)?;
            if facts.scheduler_state != "NOT_SUBMITTED" || facts.submitted {
                return Err(Error::Authority);
            }
        }
        let backend = self
            .backends
            .values()
            .find(|b| b.profile.target == prepared.request.intent().target)
            .ok_or(Error::Authority)?;
        let policy = Policy {
            allow_scheduler_mutations: true,
            ..Default::default()
        };
        let files = FrozenFiles {
            execution_campaign_sha256: Some(prepared.binding.run_sha256.clone()),
            execution_manifest_sha256: Some(prepared.request.intent().manifest_sha256.clone()),
            ..Default::default()
        };
        let check = || {
            guard
                .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
                .map_err(|_| crate::backends::slurm::action::Error::Guard)?;
            self.check(&prepared)
                .map_err(|_| crate::backends::slurm::action::Error::Guard)
        };
        let snapshot = Runner {
            actions: &self.actions,
            ssh: &backend.ssh,
        }
        .execute(
            &prepared.request,
            Execute {
                action_id: action,
                confirmation,
                policy: &policy,
                files: &files,
                claim: ClaimContext {
                    now: context.now,
                    transition_id: context.transition_id,
                    pid: std::process::id(),
                },
            },
            &check,
            cancelled,
        )?;
        self.public(snapshot, context.now)
    }
    /// Read original ledger/owned scheduler evidence only. This never submits.
    pub fn reconcile(&self, action: &str, now: &str, cancelled: &AtomicBool) -> Result<Value> {
        let _execution_lock = self.lock()?;
        let prepared = self.load(action)?;
        if self.actions.observe(action)?["execution"]["status"] == "EXECUTING" {
            let mut identity = [0u8; 16];
            getrandom::fill(&mut identity).map_err(|_| Error::Configuration)?;
            self.recover_interrupted(
                action,
                now,
                &format!("{:032x}", u128::from_be_bytes(identity)),
            )?;
        }
        let guard = self
            .fence
            .existing_observation_guard(&prepared.binding.project, &prepared.binding.run_id)
            .map_err(|_| Error::Authority)?;
        let backend = self
            .backends
            .values()
            .find(|b| b.profile.target == prepared.request.intent().target)
            .ok_or(Error::Authority)?;
        let check = || {
            guard
                .require_observed_live(&prepared.binding.project, &prepared.binding.run_id)
                .map_err(|_| crate::backends::slurm::action::Error::Guard)?;
            self.check(&prepared)
                .map_err(|_| crate::backends::slurm::action::Error::Guard)
        };
        let snapshot = Runner {
            actions: &self.actions,
            ssh: &backend.ssh,
        }
        .reconcile(action, &check, cancelled)?;
        self.public(snapshot, now)
    }
    /// Explicit interrupted-claim recovery. The caller must be offline or hold
    /// the exclusive service execution lock, as reconcile does. GET/open never
    /// calls this. It records uncertainty and issues no provider request.
    pub fn recover_interrupted(
        &self,
        action: &str,
        now: &str,
        transition_id: &str,
    ) -> Result<Value> {
        let prepared = self.load(action)?;
        let snapshot = self.actions.observe(action)?;
        if let Some(transition) = ml_exp_core::operations::interrupted(&snapshot["execution"], now)
            .map_err(operations::Error::from)?
        {
            self.actions
                .commit(action, &transition, transition_id, now)?;
        }
        self.check(&prepared)?;
        self.public(self.actions.observe(action)?, now)
    }
}
