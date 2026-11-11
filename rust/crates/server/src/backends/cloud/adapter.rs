//! Bound PAI adapters. Every observation validates the original
//! request, workspace, owner and provider identity before returning evidence.
use super::{Client, Error, Result};
use ml_exp_core::cloud::{self, Job, JobObservation, Page, Request, pai};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
/// A backend's provider profile. PAI DLC is the only provider, but the enum
/// stays: its derived form `{"provider":"pai",...}` is what configurations hold
/// and what `digest` hashes, so every recorded `profile_sha256` depends on it.
/// Code reads the PAI profile through `pai()` instead of matching.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
pub enum Profile {
    Pai(pai::Profile),
}
impl Profile {
    pub fn pai(&self) -> &pai::Profile {
        let Self::Pai(p) = self;
        p
    }
    pub fn validate(&self) -> Result<()> {
        self.pai().validate().map_err(why!(Error::configuration()))
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
    /// Always null for PAI, and written so. Retained SenseCore records carry a
    /// string here; they still load, and no PAI profile validates them.
    pub(super) provider_uid: Option<String>,
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
            || self.provider_uid.is_some()
        {
            return Err(Error::evidence());
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
        let observation = self
            .profile
            .pai()
            .validate_job(row, job, None)
            .map_err(why!(Error::evidence()))?;
        let handle = Handle {
            profile_sha256: self.profile.digest()?,
            request_sha256: job_digest(job)?,
            job_id: observation.job_id,
            provider_uid: None,
        };
        handle.validate(&self.profile, job)?;
        Ok(handle)
    }
    fn raw_job(&self, id: &str, cancelled: &AtomicBool) -> Result<Value> {
        self.call(self.profile.pai().get(id), cancelled)
    }
    /// The caller must durably reserve its submit effect before this method.
    /// The native outbox provides that boundary; no HTTP route calls it directly.
    pub(super) fn create(
        &self,
        job: &Job,
        allocation: pai::Allocation,
        cancelled: &AtomicBool,
    ) -> Result<Handle> {
        let response = self.call(
            self.profile.pai().create_request(job, allocation),
            cancelled,
        )?;
        let id = response["JobId"]
            .as_str()
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
        let p = self.profile.pai();
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
            let number = token.parse().map_err(why!(Error::evidence()))?;
            let value = self.call(self.profile.pai().list(job, number), cancelled)?;
            let page = Page::pai(&value, "Jobs", number, 100).map_err(why!(Error::evidence()))?;
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
        self.call(self.profile.pai().stop(&handle.job_id), cancelled)?;
        // A successful POST is not proof of termination.
        self.observe(job, handle, cancelled).map_err(|mut e| {
            e.uncertain = true;
            e
        })
    }
}
mod diagnostics;
