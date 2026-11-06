//! Replacing a dataset copy with a new request. The old job must be proven to have
//! ended without success. Its frozen job stays in a per-request file, so that request
//! id can still be read but can never be driven again.
use super::{sync_jobs::proven_terminal, *};
use ml_exp_core::{
    cloud::{Job, pai::Allocation},
    state::RunState,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Superseded {
    project: String,
    executor: String,
    asset_id: String,
    request_id: String,
    job: Job,
}
fn superseded_name(
    project: &str,
    executor: &str,
    asset: &str,
    request: &str,
) -> cloud::Result<String> {
    Ok(format!(
        "superseded-copy-{}.json",
        canonical::digest(&json!([project, executor, asset, request]))
            .map_err(why!(cloud::Error::evidence()))?
    ))
}
fn identity_conflict() -> cloud::Error {
    cloud::Error {
        code: "DATA_COPY_IDENTITY_CONFLICT".into(),
        http_status: Some(409),
        uncertain: false,
    }
}
/// `data_copy_cli` exits non-zero unless READY was acknowledged, so SUCCEEDED without
/// READY is contradictory evidence and never permits a second copy.
pub(super) fn allows_new_request(status: &Value) -> bool {
    matches!(
        proven_terminal(status),
        Ok(Some(observation)) if matches!(
            observation.state,
            RunState::Failed | RunState::Cancelled | RunState::Preempted
        )
    )
}
impl Service {
    /// Refuse `request` unless it is new and a fresh read of the provider proves the
    /// current job ended without success. No outbox entry means nothing is proven.
    pub(super) fn prove_copy_replaceable(
        &self,
        adapter: &Adapter<Shared>,
        current: &Job,
        project: &str,
        executor: &str,
        asset: &str,
        request: &str,
    ) -> cloud::Result<()> {
        let name = superseded_name(project, executor, asset, request)?;
        if self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Err(cloud::Error {
                code: "DATA_COPY_REQUEST_SUPERSEDED".into(),
                http_status: Some(409),
                uncertain: false,
            });
        }
        if self.sync_outbox.optional_status(current)?.is_none() {
            return Err(identity_conflict());
        }
        let status = self.sync_outbox.reconcile(
            adapter,
            current,
            Allocation::Guaranteed,
            &|| self.project_hash(project).map(|_| ()),
            &AtomicBool::new(false),
        )?;
        if !allows_new_request(&status) {
            return Err(identity_conflict());
        }
        Ok(())
    }
    /// Keep the proven job before `saved` stops naming it. The caller holds the service
    /// lock and has re-read `saved`. A crash after this leaves `saved` unchanged: that
    /// id is still current and a later request repeats this step.
    pub(super) fn retain_superseded_copy(
        &self,
        saved: &datasets::Saved,
        project: &str,
        executor: &str,
        asset: &str,
        previous: &str,
        job: &Job,
    ) -> cloud::Result<()> {
        if saved.copy_request_id.as_deref() != Some(previous) {
            return Err(identity_conflict());
        }
        let name = superseded_name(project, executor, asset, previous)?;
        if self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            let retained: Superseded = serde_json::from_value(read(&self.slots, &name)?)
                .map_err(why!(cloud::Error::evidence()))?;
            return if retained.job == *job {
                Ok(())
            } else {
                Err(cloud::Error::evidence())
            };
        }
        write(
            &self.slots,
            &name,
            &Superseded {
                project: project.into(),
                executor: executor.into(),
                asset_id: asset.into(),
                request_id: previous.into(),
                job: job.clone(),
            },
        )
    }
    /// The frozen job of `request`, whether current or superseded, for reading only.
    pub(super) fn copy_job_of_request(
        &self,
        saved: datasets::Saved,
        project: &str,
        executor: &str,
        asset: &str,
        request: &str,
    ) -> cloud::Result<Job> {
        if saved.copy_request_id.as_deref() == Some(request) {
            return saved.copy_intent.ok_or_else(cloud::Error::evidence);
        }
        let name = superseded_name(project, executor, asset, request)?;
        if !self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            return Err(identity_conflict());
        }
        let retained: Superseded = serde_json::from_value(read(&self.slots, &name)?)
            .map_err(why!(cloud::Error::evidence()))?;
        if retained.project != project
            || retained.executor != executor
            || retained.asset_id != asset
            || retained.request_id != request
        {
            return Err(cloud::Error::evidence());
        }
        Ok(retained.job)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn status(state: RunState) -> Value {
        json!({"diagnostic":null,"handle":{"job_id":"job"},
            "observation":{"job_id":"job","provider_state":"text","state":state,"queue":null}})
    }
    #[test]
    fn only_a_proven_end_without_success_allows_a_new_request() {
        for state in [RunState::Failed, RunState::Cancelled, RunState::Preempted] {
            assert!(allows_new_request(&status(state)));
        }
        for state in [
            RunState::Succeeded,
            RunState::Running,
            RunState::Queued,
            RunState::Unknown,
        ] {
            assert!(!allows_new_request(&status(state)));
        }
    }
    #[test]
    fn a_raw_state_that_may_still_run_allows_no_new_request() {
        let raw = |text: &str| {
            let mut value = status(ml_exp_core::cloud::sensecore::normalize(text));
            value["observation"]["provider_state"] = json!(text);
            value
        };
        for text in ["SUSPENDED", "DELETING", "deleting"] {
            assert!(!allows_new_request(&raw(text)), "{text}");
        }
        for text in ["STOPPED", "DELETED", "CANCELLED", "CANCELED", "FAILED"] {
            assert!(allows_new_request(&raw(text)), "{text}");
        }
    }
    #[test]
    fn stale_or_unbound_evidence_allows_no_new_request() {
        let mut failed_read = status(RunState::Failed);
        failed_read["diagnostic"] = json!({"code":"PROVIDER_DOWN"});
        assert!(!allows_new_request(&failed_read));
        let mut other_job = status(RunState::Failed);
        other_job["handle"]["job_id"] = json!("another");
        assert!(!allows_new_request(&other_job));
        let mut no_handle = status(RunState::Failed);
        no_handle["handle"] = Value::Null;
        assert!(!allows_new_request(&no_handle));
        let mut no_observation = status(RunState::Failed);
        no_observation["observation"] = Value::Null;
        assert!(!allows_new_request(&no_observation));
    }
}
