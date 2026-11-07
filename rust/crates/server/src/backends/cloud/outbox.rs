//! Persistent one-effect boundary for native cloud adapters. This is a private
//! adapter journal, not another public action API. Existing Actions supply the
//! Run/source/deletion guard before entering this boundary.
use super::{
    Error, Result,
    adapter::{Adapter, Handle, Profile, Transport, job_digest},
    lock_diagnostics,
};
use ml_exp_core::{
    canonical,
    cloud::{Job, JobObservation, pai::Allocation},
};
use ml_exp_io::directory::{Directory, LockOptions, Wait};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    ffi::OsStr,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
const LOCK: &str = ".cloud-effects.lock";
/// How long a caller waits for `.cloud-effects.lock` before CLOUD_EFFECT_BUSY.
/// Reads and observation commits hold it for file I/O only. Create, stop and a
/// handle-less find still hold it across their provider calls, so this covers an
/// ordinary provider round trip (seconds) with room to spare; a call that runs
/// toward the configured HTTP timeout still yields the same busy error. It stays
/// well below the 30 s and longer HTTP budgets of the clients and the server.
const LOCK_WAIT: Duration = Duration::from_secs(10);
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
    holder: lock_diagnostics::Holder,
}
/// Proof that the caller holds `.cloud-effects.lock`.
type Guard = lock_diagnostics::Guard;
impl Outbox {
    pub fn initialize(path: &Path) -> Result<()> {
        let directory = Directory::open(path, false).map_err(why!(Error::configuration()))?;
        if !directory
            .entry_names()
            .map_err(why!(Error::configuration()))?
            .is_empty()
        {
            return Err(Error::configuration());
        }
        let _lock = directory
            .lock(
                OsStr::new(LOCK),
                LockOptions {
                    create: true,
                    wait: Wait::Block,
                    ..Default::default()
                },
            )
            .map_err(why!(Error::configuration()))?;
        directory
            .publish_new(
                OsStr::new(MARKER),
                &serde_json::to_vec(&json!({"format":FORMAT}))
                    .map_err(why!(Error::configuration()))?,
            )
            .map_err(why!(Error::configuration()))?;
        Ok(())
    }
    pub fn open(path: &Path) -> Result<Self> {
        let directory = Directory::open(path, false).map_err(why!(Error::configuration()))?;
        let marker = directory
            .read_regular(OsStr::new(MARKER), 1024)
            .map_err(why!(Error::configuration()))?;
        if Value::Object(ml_exp_core::json::mapping(&marker).map_err(why!(Error::configuration()))?)
            != json!({"format":FORMAT})
        {
            return Err(Error::configuration());
        }
        directory
            .open_regular_read(OsStr::new(LOCK))
            .map_err(why!(Error::configuration()))?;
        Ok(Self {
            directory,
            holder: lock_diagnostics::Holder::new("cloud-effects-lock"),
        })
    }
    /// `label` names the caller in the lock diagnostics; it is a static string.
    fn lock(&self, label: &'static str) -> Result<Guard> {
        let locked = self.directory.lock(
            OsStr::new(LOCK),
            LockOptions {
                wait: Wait::Until {
                    deadline: Instant::now() + LOCK_WAIT,
                    cancelled: None,
                },
                ..Default::default()
            },
        );
        match locked {
            Ok(lock) => Ok(self.holder.acquired(lock, label)),
            Err(cause) => {
                if cause.kind() == std::io::ErrorKind::WouldBlock {
                    self.holder.busy(label);
                }
                Err((why_lock!(
                    Error {
                        code: "CLOUD_EFFECT_BUSY".into(),
                        http_status: Some(503),
                        uncertain: false,
                    },
                    Error::evidence()
                ))(cause))
            }
        }
    }
    pub fn slot(job: &Job) -> Result<String> {
        job.validate().map_err(why!(Error::configuration()))?;
        let digest = canonical::protocol_identity(
            &json!({"project":job.project,"run_id":job.run_id,"attempt_id":job.attempt_id}),
        )
        .map_err(why!(Error::configuration()))?;
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
            .map_err(why!(Error::evidence()))?;
        let value =
            Value::Object(ml_exp_core::json::mapping(&bytes).map_err(why!(Error::evidence()))?);
        let entry: Entry = serde_json::from_value(value).map_err(why!(Error::evidence()))?;
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
    fn save(&self, entry: &Entry, _lock: &Guard) -> Result<()> {
        self.directory
            .replace_regular(
                OsStr::new(&Self::slot(&entry.job)?),
                &serde_json::to_vec(entry).map_err(why!(Error::evidence()))?,
            )
            .map_err(why!(Error::transport(entry.started)))
    }
    pub fn prepare(
        &self,
        profile: &Profile,
        job: &Job,
        allocation: Allocation,
        guard: &dyn Fn() -> Result<()>,
    ) -> Result<Value> {
        guard()?;
        let _lock = self.lock("outbox.prepare")?;
        profile.validate()?;
        job.validate().map_err(why!(Error::configuration()))?;
        match profile {
            Profile::Pai(p) => {
                p.create(job, allocation)
                    .map_err(why!(Error::configuration()))?;
            }
        }
        let slot = Self::slot(job)?;
        let entry = if self
            .directory
            .entry_exists(OsStr::new(&slot))
            .map_err(why!(Error::evidence()))?
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
            self.directory
                .publish_new(
                    OsStr::new(&slot),
                    &serde_json::to_vec(&entry).map_err(why!(Error::evidence()))?,
                )
                .map_err(why!(Error::transport(false)))?;
            entry
        };
        guard()?;
        self.view(&entry)
    }
    pub fn status(&self, job: &Job) -> Result<Value> {
        let _lock = self.lock("outbox.status")?;
        let entry = self.load(&Self::slot(job)?)?;
        if entry.job != *job {
            return Err(Error::evidence());
        }
        self.view(&entry)
    }
    pub fn optional_status(&self, job: &Job) -> Result<Option<Value>> {
        let _lock = self.lock("outbox.optional_status")?;
        let name = Self::slot(job)?;
        if !self
            .directory
            .entry_exists(OsStr::new(&name))
            .map_err(why!(Error::evidence()))?
        {
            return Ok(None);
        }
        let entry = self.load(&name)?;
        if entry.job != *job {
            return Err(Error::evidence());
        }
        self.view(&entry).map(Some)
    }
    /// The job as first recorded in this job's slot. Read-only, and never a reason
    /// to create: `prepare` still refuses any job that differs from this record.
    pub fn recorded_job(&self, job: &Job) -> Result<Option<Job>> {
        let _lock = self.lock("outbox.recorded_job")?;
        let name = Self::slot(job)?;
        if !self
            .directory
            .entry_exists(OsStr::new(&name))
            .map_err(why!(Error::evidence()))?
        {
            return Ok(None);
        }
        Ok(Some(self.load(&name)?.job))
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
        let lock = self.lock("outbox.execute")?;
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
        match guard().and_then(|_| adapter.create(job, allocation, cancelled)) {
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
        let slot = Self::slot(job)?;
        let lock = self.lock("outbox.reconcile")?;
        let mut entry = self.load(&slot)?;
        self.check(&entry, adapter, job, allocation)?;
        if !entry.started {
            return self.view(&entry);
        }
        let (lock, outcome) = if let Some(handle) = entry.handle.clone() {
            // A recorded handle is only read: no find and no create can follow
            // here. The lock is released for the provider GET, and the reply is
            // committed only if the entry is still exactly the record read, so a
            // concurrent stop, find or observation is never overwritten.
            let read = serde_json::to_value(&entry).map_err(why!(Error::evidence()))?;
            drop(lock);
            let observed = adapter.observe(job, &handle, cancelled);
            let lock = self.lock("outbox.reconcile.commit")?;
            entry = self.load(&slot)?;
            self.check(&entry, adapter, job, allocation)?;
            if serde_json::to_value(&entry).map_err(why!(Error::evidence()))? != read {
                return self.view(&entry);
            }
            (lock, observed.map(|o| (Some(handle), Some(o))))
        } else {
            // Finding a handle stays under the lock, exactly as every effect does.
            let outcome = adapter.find(job, cancelled).and_then(|handle| {
                let observation = handle
                    .as_ref()
                    .map(|h| adapter.observe(job, h, cancelled))
                    .transpose()?;
                Ok((handle, observation))
            });
            (lock, outcome)
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
        let lock = self.lock("outbox.stop")?;
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
        match guard().and_then(|_| adapter.stop(job, &handle, cancelled)) {
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
