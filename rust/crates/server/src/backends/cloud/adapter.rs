//! Bound PAI/SenseCore adapters. Every observation validates the original
//! request, workspace, owner and provider identity before returning evidence.
use super::{Client, Error, Result};
use ml_exp_core::cloud::{self, Job, JobObservation, Page, Request, pai, sensecore};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::AtomicBool,
};

pub trait Transport: Send + Sync {
    fn request(&self, request: &Request, cancelled: &AtomicBool) -> Result<Value>;
}
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
pub enum Profile {
    Pai(pai::Profile),
    Sensecore(sensecore::Profile),
}
impl Profile {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Pai(p) => p.validate(),
            Self::Sensecore(p) => p.validate(),
        }
        .map_err(|_| Error::configuration())
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        ml_exp_core::canonical::protocol_identity(
            &serde_json::to_value(self).map_err(|_| Error::configuration())?,
        )
        .map_err(|_| Error::configuration())
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
    job.validate().map_err(|_| Error::evidence())?;
    ml_exp_core::canonical::protocol_identity(
        &serde_json::to_value(job).map_err(|_| Error::evidence())?,
    )
    .map_err(|_| Error::evidence())
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
            Profile::Sensecore(_) => {
                if self.job_id != job.name().map_err(|_| Error::evidence())?
                    || self
                        .provider_uid
                        .as_deref()
                        .is_none_or(|s| cloud::component(s).is_err())
                {
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
            .request(&request.map_err(|_| Error::configuration())?, cancelled)
    }
    fn handle(&self, job: &Job, row: &Value) -> Result<Handle> {
        let (observation, uid) = match &self.profile {
            Profile::Pai(p) => (
                p.validate_job(row, job, None)
                    .map_err(|_| Error::evidence())?,
                None,
            ),
            Profile::Sensecore(p) => (
                p.validate_job(row, job, None)
                    .map_err(|_| Error::evidence())?,
                Some(row["uid"].as_str().ok_or_else(Error::evidence)?.into()),
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
            Profile::Sensecore(p) => self.call(p.get(id), cancelled),
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
            Profile::Sensecore(p) => {
                if allocation != pai::Allocation::Guaranteed {
                    return Err(Error::configuration());
                }
                self.call(p.create(job), cancelled)?
            }
        };
        let id = match &self.profile {
            Profile::Pai(_) => response["JobId"].as_str(),
            Profile::Sensecore(_) => response["name"].as_str(),
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
                    .map_err(|_| Error::evidence())?;
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
            Profile::Sensecore(p) => p
                .validate_job(&row, job, handle.provider_uid.as_deref())
                .map_err(|_| Error::evidence()),
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
                    let page = token.parse().map_err(|_| Error::evidence())?;
                    let value = self.call(p.list(job, page), cancelled)?;
                    Page::pai(&value, "Jobs", page, 100)
                }
                Profile::Sensecore(p) => {
                    let value = self.call(p.list(job, &token), cancelled)?;
                    Page::sensecore(&value, "training_jobs", &token)
                }
            }
            .map_err(|_| Error::evidence())?;
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
            Profile::Sensecore(p) => self.call(p.stop(&handle.job_id), cancelled)?,
        };
        // A successful POST is not proof of termination.
        self.observe(job, handle, cancelled).map_err(|mut e| {
            e.uncertain = true;
            e
        })
    }
    pub fn pai_quota(&self, cancelled: &AtomicBool) -> Result<Value> {
        let Profile::Pai(p) = &self.profile else {
            return Err(Error::configuration());
        };
        let value = self.call(p.quota(), cancelled)?;
        let quota = value.get("Quota").unwrap_or(&value);
        if quota["QuotaId"] != p.quota_id {
            return Err(Error::evidence());
        }
        Ok(quota.clone())
    }
    pub fn pai_logs(
        &self,
        job: &Job,
        handle: &Handle,
        pod: &str,
        tail: u32,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        handle.validate(&self.profile, job)?;
        let Profile::Pai(p) = &self.profile else {
            return Err(Error::configuration());
        };
        let row = self.raw_job(&handle.job_id, cancelled)?;
        p.validate_job(&row, job, Some(&handle.job_id))
            .map_err(|_| Error::evidence())?;
        let pods = row["Pods"].as_array().ok_or_else(Error::evidence)?;
        if pods.iter().filter(|p| p["PodId"] == pod).count() != 1 {
            return Err(Error::evidence());
        }
        let value = self.call(p.pod_logs(&handle.job_id, pod, tail), cancelled)?;
        // Logs are scientific/user text: outward rendering uses the common
        // redactor; neither an error response nor another pod is substituted.
        Ok(experiment_redact::redact_json(value))
    }
    fn sense_pages(
        &self,
        field: &str,
        make: impl Fn(&str) -> cloud::Result<Request>,
        cancelled: &AtomicBool,
    ) -> Result<Vec<Value>> {
        let mut token = "1".to_owned();
        let mut seen = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut total = None;
        let mut result = Vec::new();
        for _ in 0..100 {
            if !seen.insert(token.clone()) {
                return Err(Error::evidence());
            }
            let value = self.call(make(&token), cancelled)?;
            let page = Page::sensecore(&value, field, &token).map_err(|_| Error::evidence())?;
            if total.is_some_and(|n| page.total != Some(n)) {
                return Err(Error::evidence());
            }
            if total.is_none() {
                total = page.total;
            }
            let empty = page.rows.is_empty();
            for row in page.rows {
                let name = row["name"]
                    .as_str()
                    .filter(|n| cloud::component(n).is_ok())
                    .ok_or_else(Error::evidence)?;
                if !names.insert(name.to_owned()) {
                    return Err(Error::evidence());
                }
                result.push(row);
            }
            if result.len() > 10000 {
                return Err(Error::evidence());
            }
            if let Some(next) = page.next {
                token = next;
                continue;
            }
            if total.is_some_and(|n| n > result.len() as u64) {
                if empty {
                    return Err(Error::evidence());
                }
                token = (token
                    .parse::<u32>()
                    .map_err(|_| Error::evidence())?
                    .checked_add(1)
                    .ok_or_else(Error::evidence)?)
                .to_string();
                continue;
            }
            if total.is_some_and(|n| n != result.len() as u64) {
                return Err(Error::evidence());
            }
            return Ok(result);
        }
        Err(Error::evidence())
    }
    pub fn sensecore_clusters(&self, cancelled: &AtomicBool) -> Result<Vec<sensecore::Cluster>> {
        let Profile::Sensecore(p) = &self.profile else {
            return Err(Error::configuration());
        };
        let pools = self.sense_pages("aec2s", |token| p.pools(token), cancelled)?;
        let mut result = Vec::new();
        for row in pools {
            if row["state"] != "ACTIVE" {
                continue;
            }
            let name = row["name"].as_str().ok_or_else(Error::evidence)?;
            // This profile only authorizes its zone; other-zone pools are not
            // queried or considered as implicit cross-zone storage fallback.
            let expected = p.scope.path("aec2s", name).map_err(|_| Error::evidence())?;
            if row["id"] != expected {
                continue;
            }
            let specs =
                self.sense_pages("resource_specs", |token| p.specs(name, token), cancelled)?;
            result.push(sensecore::cluster(p, &row, &specs).map_err(|_| Error::evidence())?);
        }
        Ok(result)
    }
    pub fn highest_spot(
        &self,
        allowed: Option<&BTreeSet<String>>,
        cancelled: &AtomicBool,
    ) -> Result<sensecore::Cluster> {
        let Profile::Sensecore(p) = &self.profile else {
            return Err(Error::configuration());
        };
        sensecore::highest_spot(p, &self.sensecore_clusters(cancelled)?, allowed)
            .map_err(|_| Error::evidence())
    }
    pub fn sensecore_workers(
        &self,
        job: &Job,
        handle: &Handle,
        cancelled: &AtomicBool,
    ) -> Result<Vec<Value>> {
        self.observe(job, handle, cancelled)?;
        let Profile::Sensecore(p) = &self.profile else {
            return Err(Error::configuration());
        };
        self.sense_pages(
            "workers",
            |token| p.workers(&handle.job_id, token),
            cancelled,
        )
    }
    pub fn sensecore_offline_logs(
        &self,
        job: &Job,
        handle: &Handle,
        tail: u32,
        end_unix_seconds: i64,
        cancelled: &AtomicBool,
    ) -> Result<sensecore::OfflineLogs> {
        handle.validate(&self.profile, job)?;
        let Profile::Sensecore(p) = &self.profile else {
            return Err(Error::configuration());
        };
        let row = self.raw_job(&handle.job_id, cancelled)?;
        p.validate_job(&row, job, handle.provider_uid.as_deref())
            .map_err(|_| Error::evidence())?;
        let start = row["create_time"]
            .as_str()
            .and_then(|s| {
                time::OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).ok()
            })
            .and_then(|t| t.unix_timestamp().checked_sub(60))
            .filter(|n| *n >= 0)
            .ok_or_else(Error::evidence)?;
        let stations =
            self.sense_pages("resources", |token| p.telemetry_stations(token), cancelled)?;
        let expected_name = format!("ts-user-{}", p.scope.user_id);
        let stations = stations
            .iter()
            .filter(|s| {
                s["name"] == expected_name
                    && s["type"] == "monitor.ts.v1.telemetryStation"
                    && s["deleted"] != true
                    && s["zone"] == p.scope.zone
                    && s["subscription_name"] == p.scope.subscription
                    && s["resource_group_name"] == p.scope.resource_group
            })
            .collect::<Vec<_>>();
        if stations.len() != 1
            || stations[0]["id"]
                != p.scope
                    .path("telemetryStations", &expected_name)
                    .map_err(|_| Error::evidence())?
        {
            return Err(Error::evidence());
        }
        let workers = self.sensecore_workers(job, handle, cancelled)?;
        let worker = workers.first().ok_or_else(Error::evidence)?;
        let name = worker["name"].as_str().ok_or_else(Error::evidence)?;
        let container = match worker.get("containers") {
            None | Some(Value::Null) => "container-worker-0",
            Some(Value::Array(containers)) if containers.is_empty() => "container-worker-0",
            Some(Value::Array(containers)) => {
                containers[0]["name"].as_str().ok_or_else(Error::evidence)?
            }
            _ => return Err(Error::evidence()),
        };
        let value = self.call(
            p.offline_logs(
                &expected_name,
                name,
                container,
                start,
                end_unix_seconds,
                tail,
            ),
            cancelled,
        )?;
        sensecore::validate_offline_logs(p, &value, name, container, start, end_unix_seconds, tail)
            .map_err(|_| Error::evidence())
    }
}

/// Exact normalized observations can be cached independently of frozen Actions.
pub fn observation_document(observation: &JobObservation) -> Result<BTreeMap<String, Value>> {
    serde_json::from_value(serde_json::to_value(observation).map_err(|_| Error::evidence())?)
        .map_err(|_| Error::evidence())
}
