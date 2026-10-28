//! Persistent one-effect boundary for native cloud adapters. This is a private
//! adapter journal, not another public action API. Existing Actions supply the
//! Run/source/deletion guard before entering this boundary.
use super::{
    Error, Result,
    adapter::{Adapter, Handle, Profile, Transport, job_digest},
};
use fs2::FileExt;
use ml_exp_core::{
    canonical,
    cloud::{Job, JobObservation, pai::Allocation},
};
use ml_exp_io::directory::Directory;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    ffi::OsStr,
    fs::File,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
const LOCK: &str = ".cloud-effects.lock";
const MARKER: &str = ".cloud-effects.json";
const FORMAT: &str = "native-cloud-effects.v1";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    format: String,
    profile_sha256: String,
    request_sha256: String,
    job: Job,
    allocation: Allocation,
    started: bool,
    handle: Option<Handle>,
    cancellation_started: bool,
    termination_verified: bool,
    observation: Option<JobObservation>,
    diagnostic: Option<ErrorDocument>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ErrorDocument {
    code: String,
    http_status: Option<u16>,
    uncertain: bool,
}
impl From<&Error> for ErrorDocument {
    fn from(e: &Error) -> Self {
        Self {
            code: e.code.clone(),
            http_status: e.http_status,
            uncertain: e.uncertain,
        }
    }
}
pub struct Outbox {
    directory: Directory,
    anchor: File,
}
struct Guard<'a> {
    outbox: &'a Outbox,
    file: File,
}
impl Guard<'_> {
    fn validate(&self) -> Result<()> {
        self.outbox
            .directory
            .verify_private_parent()
            .map_err(|_| Error::evidence())?;
        for file in [&self.file, &self.outbox.anchor] {
            self.outbox
                .directory
                .validate_regular_entry(OsStr::new(LOCK), file)
                .map_err(|_| Error::evidence())?;
        }
        Ok(())
    }
}
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}
impl Outbox {
    pub fn initialize(path: &Path) -> Result<()> {
        let directory = Directory::open(path, false).map_err(|_| Error::configuration())?;
        directory
            .verify_private_parent()
            .map_err(|_| Error::configuration())?;
        if !directory
            .entry_names()
            .map_err(|_| Error::configuration())?
            .is_empty()
        {
            return Err(Error::configuration());
        }
        let lock = directory
            .open_lock(OsStr::new(LOCK))
            .map_err(|_| Error::configuration())?;
        lock.lock_exclusive().map_err(|_| Error::configuration())?;
        directory
            .validate_regular_entry(OsStr::new(LOCK), &lock)
            .map_err(|_| Error::configuration())?;
        directory
            .publish_new(
                OsStr::new(MARKER),
                &serde_json::to_vec(&json!({"format":FORMAT}))
                    .map_err(|_| Error::configuration())?,
            )
            .map_err(|_| Error::configuration())?;
        Ok(())
    }
    pub fn open(path: &Path) -> Result<Self> {
        let directory = Directory::open(path, false).map_err(|_| Error::configuration())?;
        directory
            .verify_private_parent()
            .map_err(|_| Error::configuration())?;
        let marker = directory
            .read_regular(OsStr::new(MARKER), 1024)
            .map_err(|_| Error::configuration())?;
        if Value::Object(ml_exp_core::json::mapping(&marker).map_err(|_| Error::configuration())?)
            != json!({"format":FORMAT})
        {
            return Err(Error::configuration());
        }
        let anchor = directory
            .open_existing_lock(OsStr::new(LOCK))
            .map_err(|_| Error::configuration())?;
        directory
            .validate_regular_entry(OsStr::new(LOCK), &anchor)
            .map_err(|_| Error::configuration())?;
        Ok(Self { directory, anchor })
    }
    fn lock(&self) -> Result<Guard<'_>> {
        self.directory
            .validate_regular_entry(OsStr::new(LOCK), &self.anchor)
            .map_err(|_| Error::evidence())?;
        let file = self
            .directory
            .open_existing_lock(OsStr::new(LOCK))
            .map_err(|_| Error::evidence())?;
        file.try_lock_exclusive().map_err(|_| Error {
            code: "CLOUD_EFFECT_BUSY".into(),
            http_status: None,
            uncertain: false,
        })?;
        let guard = Guard { outbox: self, file };
        guard.validate()?;
        Ok(guard)
    }
    pub fn slot(job: &Job) -> Result<String> {
        job.validate().map_err(|_| Error::configuration())?;
        let digest = canonical::protocol_identity(
            &json!({"project":job.project,"run_id":job.run_id,"attempt_id":job.attempt_id}),
        )
        .map_err(|_| Error::configuration())?;
        Ok(format!(
            "{}.json",
            digest
                .strip_prefix("sha256:")
                .ok_or_else(Error::configuration)?
        ))
    }
    fn load(&self, slot: &str) -> Result<Entry> {
        let bytes = self
            .directory
            .read_regular(OsStr::new(slot), 2 * 1024 * 1024)
            .map_err(|_| Error::evidence())?;
        let value =
            Value::Object(ml_exp_core::json::mapping(&bytes).map_err(|_| Error::evidence())?);
        let entry: Entry = serde_json::from_value(value).map_err(|_| Error::evidence())?;
        if entry.format != FORMAT
            || Self::slot(&entry.job)? != slot
            || entry.request_sha256 != job_digest(&entry.job)?
            || entry.handle.is_some() && !entry.started
            || entry.cancellation_started && entry.handle.is_none()
            || entry.termination_verified && !entry.cancellation_started
        {
            return Err(Error::evidence());
        }
        Ok(entry)
    }
    fn check<T: Transport>(
        &self,
        entry: &Entry,
        adapter: &Adapter<T>,
        job: &Job,
        allocation: Allocation,
    ) -> Result<()> {
        if entry.profile_sha256 != adapter.profile.digest()?
            || entry.job != *job
            || entry.request_sha256 != job_digest(job)?
            || entry.allocation != allocation
        {
            return Err(Error::evidence());
        }
        if let Some(handle) = &entry.handle {
            handle.validate(&adapter.profile, job)?;
        }
        Ok(())
    }
    fn save(&self, entry: &Entry, lock: &Guard<'_>) -> Result<()> {
        lock.validate()?;
        self.directory
            .replace_regular(
                OsStr::new(&Self::slot(&entry.job)?),
                &serde_json::to_vec(entry).map_err(|_| Error::evidence())?,
            )
            .map_err(|_| Error::transport(entry.started))
    }
    pub fn prepare(
        &self,
        profile: &Profile,
        job: &Job,
        allocation: Allocation,
        guard: &dyn Fn() -> Result<()>,
    ) -> Result<Value> {
        guard()?;
        let lock = self.lock()?;
        profile.validate()?;
        job.validate().map_err(|_| Error::configuration())?;
        match profile {
            Profile::Pai(p) => {
                p.create(job, allocation)
                    .map_err(|_| Error::configuration())?;
            }
            Profile::Sensecore(p) => {
                if allocation != Allocation::Guaranteed {
                    return Err(Error::configuration());
                }
                p.create(job).map_err(|_| Error::configuration())?;
            }
        }
        let slot = Self::slot(job)?;
        let entry = if self
            .directory
            .entry_exists(OsStr::new(&slot))
            .map_err(|_| Error::evidence())?
        {
            let entry = self.load(&slot)?;
            if entry.profile_sha256 != profile.digest()?
                || entry.job != *job
                || entry.allocation != allocation
            {
                return Err(Error::evidence());
            }
            entry
        } else {
            let entry = Entry {
                format: FORMAT.into(),
                profile_sha256: profile.digest()?,
                request_sha256: job_digest(job)?,
                job: job.clone(),
                allocation,
                started: false,
                handle: None,
                cancellation_started: false,
                termination_verified: false,
                observation: None,
                diagnostic: None,
            };
            guard()?;
            lock.validate()?;
            self.directory
                .publish_new(
                    OsStr::new(&slot),
                    &serde_json::to_vec(&entry).map_err(|_| Error::evidence())?,
                )
                .map_err(|_| Error::transport(false))?;
            entry
        };
        guard()?;
        lock.validate()?;
        self.view(&entry)
    }
    pub fn status(&self, job: &Job) -> Result<Value> {
        let lock = self.lock()?;
        let entry = self.load(&Self::slot(job)?)?;
        if entry.job != *job {
            return Err(Error::evidence());
        }
        lock.validate()?;
        self.view(&entry)
    }
    fn view(&self, entry: &Entry) -> Result<Value> {
        let slot = Self::slot(&entry.job)?;
        Ok(
            json!({"slot":slot,"project":entry.job.project,"run_id":entry.job.run_id,"attempt_id":entry.job.attempt_id,
            "phase":if entry.handle.is_some(){"OWNED"}else if entry.started{"RECONCILE_REQUIRED"}else{"PREPARED"},
            "handle":entry.handle,"observation":entry.observation,"diagnostic":entry.diagnostic,
            "confirmation":format!("SUBMIT {slot}"),"cancellation_started":entry.cancellation_started,"termination_verified":entry.termination_verified,
            "retry_safe":!entry.started}),
        )
    }
    pub fn execute<T: Transport>(
        &self,
        adapter: &Adapter<T>,
        job: &Job,
        allocation: Allocation,
        confirmation: &str,
        guard: &dyn Fn() -> Result<()>,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        guard()?;
        let lock = self.lock()?;
        let slot = Self::slot(job)?;
        let mut entry = self.load(&slot)?;
        self.check(&entry, adapter, job, allocation)?;
        if confirmation != format!("SUBMIT {slot}") {
            return Err(Error::configuration());
        }
        if entry.started {
            return self.view(&entry);
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::transport(false));
        }
        guard()?;
        entry.started = true;
        self.save(&entry, &lock)?;
        // From this durable point, failure or interruption is reconciliation,
        // even if the HTTP request may not actually have reached the provider.
        match guard().and_then(|_| {
            lock.validate()?;
            adapter.create(job, allocation, cancelled)
        }) {
            Ok(handle) => {
                entry.handle = Some(handle);
                entry.diagnostic = None;
            }
            Err(error) => entry.diagnostic = Some((&error).into()),
        }
        // Persist evidence even when the caller's Run guard changed after the
        // effect. It must never erase the durable effect marker or allow replay.
        self.save(&entry, &lock)?;
        self.view(&entry)
    }
    pub fn reconcile<T: Transport>(
        &self,
        adapter: &Adapter<T>,
        job: &Job,
        allocation: Allocation,
        guard: &dyn Fn() -> Result<()>,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        guard()?;
        let lock = self.lock()?;
        let mut entry = self.load(&Self::slot(job)?)?;
        self.check(&entry, adapter, job, allocation)?;
        if !entry.started {
            return self.view(&entry);
        }
        let outcome = if let Some(handle) = &entry.handle {
            adapter
                .observe(job, handle, cancelled)
                .map(|o| (Some(handle.clone()), Some(o)))
        } else {
            adapter.find(job, cancelled).and_then(|handle| {
                let observation = handle
                    .as_ref()
                    .map(|h| adapter.observe(job, h, cancelled))
                    .transpose()?;
                Ok((handle, observation))
            })
        };
        match outcome {
            Ok((handle, observation)) => {
                guard()?;
                entry.handle = handle;
                if let Some(observation) = observation {
                    if entry
                        .observation
                        .as_ref()
                        .is_some_and(|previous| !previous.state.allows(observation.state))
                    {
                        return Err(Error::evidence());
                    }
                    entry.termination_verified =
                        entry.cancellation_started && observation.state.is_terminal();
                    entry.observation = Some(observation);
                }
                entry.diagnostic = None;
            }
            Err(error) => entry.diagnostic = Some((&error).into()),
        }
        // Positive list absence cannot fence an in-flight create request. The
        // original started marker stays set, so absence never permits a replay.
        self.save(&entry, &lock)?;
        self.view(&entry)
    }
    pub fn stop<T: Transport>(
        &self,
        adapter: &Adapter<T>,
        job: &Job,
        allocation: Allocation,
        confirmation: &str,
        guard: &dyn Fn() -> Result<()>,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        guard()?;
        let lock = self.lock()?;
        let mut entry = self.load(&Self::slot(job)?)?;
        self.check(&entry, adapter, job, allocation)?;
        if confirmation != format!("STOP {}", Self::slot(job)?) {
            return Err(Error::configuration());
        }
        let handle = entry.handle.clone().ok_or_else(Error::evidence)?;
        if entry.cancellation_started {
            return self.view(&entry);
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::transport(false));
        }
        guard()?;
        entry.cancellation_started = true;
        self.save(&entry, &lock)?;
        match guard().and_then(|_| {
            lock.validate()?;
            adapter.stop(job, &handle, cancelled)
        }) {
            Ok(observation) => {
                entry.termination_verified = observation.state.is_terminal();
                entry.observation = Some(observation);
                entry.diagnostic = None;
            }
            Err(error) => entry.diagnostic = Some((&error).into()),
        }
        self.save(&entry, &lock)?;
        self.view(&entry)
    }
}
#[cfg(test)]
mod tests;
