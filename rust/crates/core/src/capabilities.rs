//! Pure executor-capability declarations and deterministic matching.
//!
//! Provider probes and profile loading remain in the imperative server shell.
//! The shell converts its observations into these values; this module decides
//! whether a request is compatible and which already-compatible candidate is
//! preferred. A match never claims live capacity or submits a job.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    Slurm,
    Sensecore,
    PaiDlc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Materialization {
    Oci,
    OciToSif,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparationMode {
    BeforeJob,
    InsideJob,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resources {
    pub gpus: u32,
    pub cpus: u32,
    pub memory_gb: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub backend: Backend,
    pub platforms: Vec<String>,
    pub materialization: Materialization,
    pub api_relay: bool,
    pub persistent_storage: bool,
    pub storage_id: Option<String>,
    pub assets: bool,
    pub preparation_mode: Option<PreparationMode>,
    pub preparation_timeout_seconds: u64,
    pub queue_reason: bool,
    pub exit_code: bool,
    pub preemption_reason: bool,
    pub scheduler_walltime: bool,
    pub offline_output_export: bool,
    pub allocation_policies: Vec<String>,
    pub priority: bool,
    pub capacity: Option<Resources>,
    pub fixed_gpu_count: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirements {
    pub platform: String,
    pub data_assets: bool,
    pub data_preparation: bool,
    pub data_preparation_before_job: bool,
    pub persistent_checkpoints: bool,
    pub queue_reason: bool,
    pub exit_code: bool,
    pub preemption_reason: bool,
    pub scheduler_walltime: bool,
    pub offline_output_export: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub resources: Resources,
    pub has_inputs: bool,
    pub has_data_preparation: bool,
    pub data_preparation_timeout_seconds: Option<u64>,
    pub has_checkpoint_persistence: bool,
    pub has_resume_from: bool,
    pub has_checkpoint_upload: bool,
    pub allocation_policy: Option<String>,
    pub priority: Option<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selector {
    pub backend: Option<Backend>,
    pub candidates: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub name: String,
    pub capability: Capability,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Match {
    pub executor: String,
    pub capability: Capability,
    pub compatible_executors: Vec<String>,
    pub rejected: BTreeMap<String, Vec<String>>,
    pub live_availability_checked: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("an executor or an explicit selector is required")]
    MissingSelector,
    #[error("executor and selector cannot both be supplied")]
    ConflictingSelector,
    #[error("no executor matches the request")]
    NoMatch,
    #[error("worker is missing required capabilities")]
    MissingWorkerCapabilities(Vec<String>),
}
pub type Result<T> = std::result::Result<T, Error>;

fn resource_mismatch(capability: &Capability, request: &Request) -> Vec<String> {
    let mut problems = Vec::new();
    if let Some(capacity) = &capability.capacity {
        if request.resources.gpus > capacity.gpus {
            problems.push("resources.gpus".into());
        }
        if request.resources.cpus > capacity.cpus {
            problems.push("resources.cpus".into());
        }
        if request.resources.memory_gb > capacity.memory_gb {
            problems.push("resources.memory_gb".into());
        }
    }
    if capability
        .fixed_gpu_count
        .is_some_and(|count| count != request.resources.gpus)
    {
        problems.push("resources.fixed_gpu_count".into());
    }
    problems
}

/// Return stable field paths explaining why a declaration cannot satisfy a
/// request. The order is part of the diagnostic contract.
pub fn mismatches(
    capability: &Capability,
    request: &Request,
    requirements: &Requirements,
) -> Vec<String> {
    let mut problems = Vec::new();
    if !capability
        .platforms
        .iter()
        .any(|value| value == &requirements.platform)
    {
        problems.push("runtime.platforms".into());
    }
    if !matches!(
        capability.materialization,
        Materialization::Oci | Materialization::OciToSif
    ) {
        problems.push("runtime.materialization".into());
    }
    for (needed, supported, path) in [
        (
            requirements.queue_reason,
            capability.queue_reason,
            "observability.queue_reason",
        ),
        (
            requirements.exit_code,
            capability.exit_code,
            "observability.exit_code",
        ),
        (
            requirements.preemption_reason,
            capability.preemption_reason,
            "observability.preemption_reason",
        ),
    ] {
        if needed && !supported {
            problems.push(path.into());
        }
    }
    if requirements.scheduler_walltime && !capability.scheduler_walltime {
        problems.push("jobs.scheduler_walltime".into());
    }
    if requirements.offline_output_export && !capability.offline_output_export {
        problems.push("outputs.offline_export".into());
    }
    if (request.has_inputs || requirements.data_assets) && !capability.assets {
        problems.push("data.assets".into());
    }
    if (request.has_data_preparation || requirements.data_preparation)
        && capability.preparation_mode.is_none()
    {
        problems.push("data.download_script".into());
    }
    if requirements.data_preparation_before_job
        && capability.preparation_mode != Some(PreparationMode::BeforeJob)
    {
        problems.push("data.download_script.before_job".into());
    }
    if let Some(timeout) = request.data_preparation_timeout_seconds {
        if timeout > capability.preparation_timeout_seconds {
            problems.push("data.download_script.timeout_seconds".into());
        }
    }
    if (request.has_checkpoint_persistence
        || request.has_resume_from
        || requirements.persistent_checkpoints)
        && (!capability.persistent_storage || capability.storage_id.is_none())
    {
        problems.push("storage.checkpoint_registration".into());
    }
    if request.has_checkpoint_upload
        && (!capability.assets || matches!(capability.backend, Backend::PaiDlc))
    {
        problems.push("checkpoint_upload".into());
    }
    if let Some(policy) = &request.allocation_policy {
        if !capability
            .allocation_policies
            .iter()
            .any(|value| value == policy)
        {
            problems.push("resources.allocation_policy".into());
        }
    }
    if request.priority.is_some() && !capability.priority {
        problems.push("resources.priority".into());
    }
    problems.extend(resource_mismatch(capability, request));
    problems
}

/// Match configured candidates without probing live provider capacity.
pub fn select(
    candidates: &[Candidate],
    explicit_executor: Option<&str>,
    selector: Option<&Selector>,
    request: &Request,
    requirements: &Requirements,
) -> Result<Match> {
    if explicit_executor.is_some() && selector.is_some() {
        return Err(Error::ConflictingSelector);
    }
    if explicit_executor.is_none()
        && selector.is_none_or(|value| value.backend.is_none() && value.candidates.is_empty())
    {
        return Err(Error::MissingSelector);
    }
    let names: Vec<&str> = if let Some(name) = explicit_executor {
        vec![name]
    } else if let Some(selector) = selector {
        if selector.candidates.is_empty() {
            candidates.iter().map(|value| value.name.as_str()).collect()
        } else {
            selector.candidates.iter().map(String::as_str).collect()
        }
    } else {
        unreachable!("selector was checked above")
    };
    let by_name: BTreeMap<&str, &Candidate> = candidates
        .iter()
        .map(|value| (value.name.as_str(), value))
        .collect();
    let backend = selector.and_then(|value| value.backend);
    let mut rejected = BTreeMap::new();
    let mut accepted = Vec::new();
    for name in names {
        let Some(candidate) = by_name.get(name) else {
            rejected.insert(name.to_owned(), vec!["unknown_executor".into()]);
            continue;
        };
        if backend.is_some_and(|expected| candidate.capability.backend != expected) {
            rejected.insert(name.to_owned(), vec!["backend".into()]);
            continue;
        }
        let reasons = mismatches(&candidate.capability, request, requirements);
        if reasons.is_empty() {
            accepted.push(*candidate);
        } else {
            rejected.insert(name.to_owned(), reasons);
        }
    }
    if accepted.is_empty() {
        return Err(Error::NoMatch);
    }
    // An explicit candidate list is preference order. Automatic matching uses
    // the smallest configured fitting capacity, then the stable executor ID.
    if selector.is_none_or(|value| value.candidates.is_empty()) && explicit_executor.is_none() {
        accepted.sort_by_key(|candidate| {
            let capacity = candidate.capability.capacity.clone().unwrap_or_default();
            (
                capacity.gpus,
                capacity.cpus,
                capacity.memory_gb,
                candidate.name.clone(),
            )
        });
    }
    let chosen = accepted[0];
    Ok(Match {
        executor: chosen.name.clone(),
        capability: chosen.capability.clone(),
        compatible_executors: accepted
            .into_iter()
            .map(|value| value.name.clone())
            .collect(),
        rejected,
        live_availability_checked: false,
    })
}

/// Check the worker capability list required by an already-selected request.
pub fn missing_worker_capabilities(
    request: &Request,
    capability: &Capability,
    worker_capabilities: &[String],
) -> Vec<String> {
    let mut required = Vec::new();
    if request.has_inputs || request.has_checkpoint_upload {
        required.push("data-assets.v1");
    }
    if request.has_data_preparation {
        required.push("data-preparation.v1");
    }
    if request.has_checkpoint_persistence || request.has_resume_from {
        required.push("persistent-checkpoints.v1");
    }
    if capability.api_relay {
        required.push("api-relay.v1");
    }
    let mut missing: Vec<_> = required
        .into_iter()
        .filter(|value| !worker_capabilities.iter().any(|item| item == value))
        .map(str::to_owned)
        .collect();
    missing.sort();
    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capability(backend: Backend) -> Capability {
        Capability {
            backend,
            platforms: vec!["linux/amd64".into()],
            materialization: Materialization::Oci,
            api_relay: false,
            persistent_storage: true,
            storage_id: Some("nas/demo".into()),
            assets: true,
            preparation_mode: Some(PreparationMode::BeforeJob),
            preparation_timeout_seconds: 3600,
            queue_reason: true,
            exit_code: true,
            preemption_reason: true,
            scheduler_walltime: true,
            offline_output_export: true,
            allocation_policies: vec!["guaranteed".into(), "allow_idle".into()],
            priority: matches!(backend, Backend::PaiDlc),
            capacity: Some(Resources {
                gpus: 8,
                cpus: 200,
                memory_gb: 1600,
            }),
            fixed_gpu_count: None,
        }
    }

    fn request() -> Request {
        Request {
            resources: Resources {
                gpus: 4,
                cpus: 64,
                memory_gb: 512,
            },
            has_inputs: true,
            has_data_preparation: true,
            data_preparation_timeout_seconds: Some(1800),
            has_checkpoint_persistence: true,
            has_resume_from: false,
            has_checkpoint_upload: false,
            allocation_policy: None,
            priority: None,
        }
    }

    #[test]
    fn mismatches_are_stable_and_explain_missing_capabilities() {
        let mut value = capability(Backend::Slurm);
        value.assets = false;
        value.queue_reason = false;
        let reasons = mismatches(
            &value,
            &request(),
            &Requirements {
                platform: "linux/amd64".into(),
                data_assets: true,
                queue_reason: true,
                ..Requirements::default()
            },
        );
        assert_eq!(reasons, vec!["observability.queue_reason", "data.assets"]);
    }

    #[test]
    fn selector_preserves_explicit_preference_and_rejects_unknown() {
        let values = vec![
            Candidate {
                name: "h100".into(),
                capability: capability(Backend::Slurm),
            },
            Candidate {
                name: "pai".into(),
                capability: capability(Backend::PaiDlc),
            },
        ];
        let result = select(
            &values,
            None,
            Some(&Selector {
                backend: None,
                candidates: vec!["pai".into(), "h100".into()],
            }),
            &request(),
            &Requirements {
                platform: "linux/amd64".into(),
                ..Requirements::default()
            },
        )
        .unwrap();
        assert_eq!(result.executor, "pai");
        let error = select(
            &values,
            None,
            Some(&Selector {
                backend: None,
                candidates: vec!["missing".into()],
            }),
            &request(),
            &Requirements {
                platform: "linux/amd64".into(),
                ..Requirements::default()
            },
        )
        .unwrap_err();
        assert_eq!(error, Error::NoMatch);
    }

    #[test]
    fn automatic_selection_prefers_smallest_fitting_capacity() {
        let mut small = capability(Backend::Slurm);
        small.capacity = Some(Resources {
            gpus: 4,
            cpus: 64,
            memory_gb: 512,
        });
        let mut large = capability(Backend::Slurm);
        large.capacity = Some(Resources {
            gpus: 8,
            cpus: 128,
            memory_gb: 1024,
        });
        let result = select(
            &[
                Candidate {
                    name: "large".into(),
                    capability: large,
                },
                Candidate {
                    name: "small".into(),
                    capability: small,
                },
            ],
            None,
            Some(&Selector {
                backend: Some(Backend::Slurm),
                candidates: Vec::new(),
            }),
            &request(),
            &Requirements {
                platform: "linux/amd64".into(),
                ..Requirements::default()
            },
        )
        .unwrap();
        assert_eq!(result.executor, "small");
    }

    #[test]
    fn worker_requirements_are_explicit() {
        let mut value = request();
        value.has_resume_from = true;
        let mut selected = capability(Backend::Slurm);
        selected.api_relay = true;
        let missing = missing_worker_capabilities(&value, &selected, &[]);
        assert_eq!(
            missing,
            vec![
                "api-relay.v1",
                "data-assets.v1",
                "data-preparation.v1",
                "persistent-checkpoints.v1"
            ]
        );
    }
}
