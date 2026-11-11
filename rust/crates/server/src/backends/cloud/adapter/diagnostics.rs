//! Read-only provider reads beside the effect path: quota and job diagnostics.
//! Nothing here creates or stops a job, so it lies outside the no-replay pin.
use super::{Adapter, Error, Handle, Result, Transport};
use ml_exp_core::cloud::{Job, pai};
use serde_json::{Value, json};
use std::sync::atomic::AtomicBool;

impl<T: Transport> Adapter<T> {
    pub fn pai_quota(&self, cancelled: &AtomicBool) -> Result<Value> {
        let p = self.profile.pai();
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
        let p = self.profile.pai();
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
        let p = self.profile.pai();
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
