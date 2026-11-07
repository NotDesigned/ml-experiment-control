//! Bound PAI adapters. Every observation validates the original
//! request, workspace, owner and provider identity before returning evidence.
use super::{Client, Error, Result};
use ml_exp_core::cloud::{self, Job, JobObservation, Page, Request, pai};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    sync::{Arc, atomic::AtomicBool},
};

pub trait Transport: Send + Sync {
    fn request(&self, request: &Request, cancelled: &AtomicBool) -> Result<Value>;
}
/// What a configured backend's adapters hold: the real `Client`, or a counting
/// fake when a service-level no-replay test opens the service.
pub type Shared = Arc<dyn Transport>;
impl Transport for Client {
    fn request(&self, request: &Request, cancelled: &AtomicBool) -> Result<Value> {
        self.call(request, cancelled)
    }
}
impl<T: Transport> Transport for &T {
    fn request(&self, request: &Request, cancelled: &AtomicBool) -> Result<Value> {
        (**self).request(request, cancelled)
    }
}
impl Transport for Shared {
    fn request(&self, request: &Request, cancelled: &AtomicBool) -> Result<Value> {
        (**self).request(request, cancelled)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
pub enum Profile {
    Pai(pai::Profile),
}
impl Profile {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Pai(p) => p.validate(),
        }
        .map_err(why!(Error::configuration()))
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        ml_exp_core::canonical::protocol_identity(
            &serde_json::to_value(self).map_err(why!(Error::configuration()))?,
        )
        .map_err(why!(Error::configuration()))
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Handle {
    pub profile_sha256: String,
    pub request_sha256: String,
    pub job_id: String,
    pub provider_uid: Option<String>,
}
pub fn job_digest(job: &Job) -> Result<String> {
    job.validate().map_err(why!(Error::evidence()))?;
    ml_exp_core::canonical::protocol_identity(
        &serde_json::to_value(job).map_err(why!(Error::evidence()))?,
    )
    .map_err(why!(Error::evidence()))
}
impl Handle {
    pub fn validate(&self, profile: &Profile, job: &Job) -> Result<()> {
        if self.profile_sha256 != profile.digest()?
            || self.request_sha256 != job_digest(job)?
            || cloud::component(&self.job_id).is_err()
        {
            return Err(Error::evidence());
        }
        match profile {
            Profile::Pai(_) => {
                if self.provider_uid.is_some() {
                    return Err(Error::evidence());
                }
            }
        }
        Ok(())
    }
}
pub struct Adapter<T = Client> {
    pub profile: Profile,
    transport: T,
}
impl<T: Transport> Adapter<T> {
    pub fn with_profile(&self, profile: Profile) -> Result<Adapter<&T>> {
        Adapter::new(profile, &self.transport)
    }
    pub fn new(profile: Profile, transport: T) -> Result<Self> {
        profile.validate()?;
        Ok(Self { profile, transport })
    }
    fn call(&self, request: cloud::Result<Request>, cancelled: &AtomicBool) -> Result<Value> {
        self.transport
            .request(&request.map_err(why!(Error::configuration()))?, cancelled)
    }
    fn handle(&self, job: &Job, row: &Value) -> Result<Handle> {
        let (observation, uid) = match &self.profile {
            Profile::Pai(p) => (
                p.validate_job(row, job, None)
                    .map_err(why!(Error::evidence()))?,
                None,
            ),
        };
        let handle = Handle {
            profile_sha256: self.profile.digest()?,
            request_sha256: job_digest(job)?,
            job_id: observation.job_id,
            provider_uid: uid,
        };
        handle.validate(&self.profile, job)?;
        Ok(handle)
    }
    fn raw_job(&self, id: &str, cancelled: &AtomicBool) -> Result<Value> {
        match &self.profile {
            Profile::Pai(p) => self.call(p.get(id), cancelled),
        }
    }
    /// The caller must durably reserve its submit effect before this method.
    /// The native outbox provides that boundary; no HTTP route calls it directly.
    pub(super) fn create(
        &self,
        job: &Job,
        allocation: pai::Allocation,
        cancelled: &AtomicBool,
    ) -> Result<Handle> {
        let response = match &self.profile {
            Profile::Pai(p) => self.call(p.create(job, allocation), cancelled)?,
        };
        let id = match &self.profile {
            Profile::Pai(_) => response["JobId"].as_str(),
        };
        let id = id
            .filter(|id| cloud::component(id).is_ok())
            .ok_or_else(|| Error {
                code: "CLOUD_CREATE_UNCONFIRMED".into(),
                http_status: None,
                uncertain: true,
            })?;
        // Create acceptance is not an ownership proof. Read the exact returned
        // ID and compare immutable scope/tags before committing an owned handle.
        let row = self.raw_job(id, cancelled).map_err(|mut e| {
            e.uncertain = true;
            e
        })?;
        let handle = self.handle(job, &row).map_err(|mut e| {
            e.uncertain = true;
            e
        })?;
        if handle.job_id != id {
            return Err(Error {
                code: "CLOUD_CREATE_IDENTITY_CONFLICT".into(),
                http_status: None,
                uncertain: true,
            });
        }
        Ok(handle)
    }
    pub fn observe(
        &self,
        job: &Job,
        handle: &Handle,
        cancelled: &AtomicBool,
    ) -> Result<JobObservation> {
        handle.validate(&self.profile, job)?;
        let row = self.raw_job(&handle.job_id, cancelled)?;
        match &self.profile {
            Profile::Pai(p) => {
                let mut observation = p
                    .validate_job(&row, job, Some(&handle.job_id))
                    .map_err(why!(Error::evidence()))?;
                if observation.state == ml_exp_core::state::RunState::Queued {
                    // A separate diagnostic failure cannot erase a known job.
                    if let Ok(queue) = self.call(p.queue(&handle.job_id), cancelled) {
                        if let Ok(diagnostic) = p.validate_queue(&queue, &handle.job_id) {
                            observation.queue = diagnostic;
                        }
                    }
                }
                Ok(observation)
            }
        }
    }
    /// Positive absence requires complete, stable and duplicate-free pages.
    /// A failure remains an error, never `None` or permission to submit again.
    pub fn find(&self, job: &Job, cancelled: &AtomicBool) -> Result<Option<Handle>> {
        let mut token = "1".to_owned();
        let mut tokens = BTreeSet::new();
        let mut identities = BTreeSet::new();
        let mut result = None;
        let mut total = None;
        let mut received = 0u64;
        for _ in 0..100 {
            if !tokens.insert(token.clone()) {
                return Err(Error::evidence());
            }
            let page = match &self.profile {
                Profile::Pai(p) => {
                    let page = token.parse().map_err(why!(Error::evidence()))?;
                    let value = self.call(p.list(job, page), cancelled)?;
                    Page::pai(&value, "Jobs", page, 100)
                }
            }
            .map_err(why!(Error::evidence()))?;
            if total.is_some_and(|previous| page.total != Some(previous)) {
                return Err(Error::evidence());
            }
            if total.is_none() {
                total = page.total;
            }
            received += page.rows.len() as u64;
            for row in page.rows {
                let handle = self.handle(job, &row)?;
                if !identities.insert(handle.job_id.clone()) || result.is_some() {
                    return Err(Error::evidence());
                }
                result = Some(handle);
            }
            if let Some(next) = page.next {
                token = next;
                continue;
            }
            if let Some(total) = total {
                if received != total {
                    return Err(Error::evidence());
                }
            }
            return Ok(result);
        }
        Err(Error::evidence())
    }
    pub(super) fn stop(
        &self,
        job: &Job,
        handle: &Handle,
        cancelled: &AtomicBool,
    ) -> Result<JobObservation> {
        let observation = self.observe(job, handle, cancelled)?;
        if observation.state.is_terminal() {
            return Ok(observation);
        }
        match &self.profile {
            Profile::Pai(p) => self.call(p.stop(&handle.job_id), cancelled)?,
        };
        // A successful POST is not proof of termination.
        self.observe(job, handle, cancelled).map_err(|mut e| {
            e.uncertain = true;
            e
        })
    }
    pub fn pai_quota(&self, cancelled: &AtomicBool) -> Result<Value> {
        let Profile::Pai(p) = &self.profile;
        let value = self.call(p.quota(), cancelled)?;
        let quota = value.get("Quota").unwrap_or(&value);
        if quota["QuotaId"] != p.quota_id {
            return Err(Error::evidence());
        }
        Ok(quota.clone())
    }
    pub fn pai_diagnostic_logs(
        &self,
        job: &Job,
        handle: &Handle,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        handle.validate(&self.profile, job)?;
        let Profile::Pai(p) = &self.profile;
        let value = self.raw_job(&handle.job_id, cancelled)?;
        p.validate_job(&value, job, Some(&handle.job_id))
            .map_err(why!(Error::evidence()))?;
        let pods = value["Pods"].as_array().ok_or_else(Error::evidence)?;
        let mut logs = Vec::new();
        for pod in pods {
            let id = pod["PodId"].as_str().ok_or_else(Error::evidence)?;
            logs.push(self.call(p.pod_logs(&handle.job_id, id, 100), cancelled)?);
        }
        Ok(Value::Array(logs))
    }
    /// Read-only provider account of one job. A failed logs or events read
    /// stays beside the job's state instead of hiding it.
    pub fn pai_provider_diagnostics(
        &self,
        job: &Job,
        handle: &Handle,
        lines: u32,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        handle.validate(&self.profile, job)?;
        let Profile::Pai(p) = &self.profile;
        let row = self.raw_job(&handle.job_id, cancelled)?;
        p.validate_job(&row, job, Some(&handle.job_id))
            .map_err(why!(Error::evidence()))?;
        let failure = |error: Error| json!({"code":error.code,"http_status":error.http_status});
        let all = row["Pods"].as_array().cloned().unwrap_or_default();
        let mut pods = Vec::new();
        for pod in all.iter().take(4) {
            let mut view = pai::diagnostic_pod(pod);
            let id = pod["PodId"].as_str().ok_or_else(Error::evidence)?;
            match self.call(p.pod_logs(&handle.job_id, id, lines), cancelled) {
                Ok(value) => {
                    view["logs"] = json!(pai::diagnostic_tail(&value["Logs"], lines as usize))
                }
                Err(error) => view["logs_error"] = failure(error),
            }
            pods.push(view);
        }
        let mut result = pai::diagnostic_job(&row);
        result["job_id"] = json!(handle.job_id);
        result["pods"] = json!(pods);
        result["pods_total"] = json!(all.len());
        match self.call(p.job_events(&handle.job_id), cancelled) {
            Ok(value) => result["events"] = json!(pai::diagnostic_tail(&value["Events"], 100)),
            Err(error) => result["events_error"] = failure(error),
        }
        Ok(experiment_redact::redact_json(result))
    }
}
