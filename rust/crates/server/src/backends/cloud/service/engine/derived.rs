//! Derived metrics computed after an Attempt's Run ended. One file is one batch,
//! kept apart from the training journal and the result archive, and published
//! by the W&B publisher to the same W&B Run.
use super::*;
use ml_exp_core::records::{DerivedRefusal, Scope};

fn refused(code: &str, status: u16) -> cloud::Error {
    cloud::Error {
        code: code.into(),
        http_status: Some(status),
        uncertain: false,
    }
}

impl Service {
    /// `sha256` is the digest of `bytes`, checked by the HTTP boundary. The same
    /// file again returns the same document and adds nothing.
    pub fn append_derived(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        sha256: &str,
        bytes: &[u8],
        now: &str,
    ) -> cloud::Result<Value> {
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(missing)?;
        self.require_live(project, run)?;
        if prepared.execution.job.attempt_id != attempt {
            return Err(missing());
        }
        let records = self.exchange_records(&prepared)?;
        let guard = records
            .guard(project, run)
            .map_err(why!(cloud::Error::evidence()))?;
        let binding = records
            .binding(&guard)
            .map_err(why!(cloud::Error::evidence()))?
            .ok_or_else(cloud::Error::evidence)?;
        let observations = ml_exp_core::records::derived_observations(bytes, &binding["config"])
            .map_err(|refusal| match refusal {
                DerivedRefusal::TooLarge => refused("DERIVED_BATCH_TOO_LARGE", 413),
                DerivedRefusal::Invalid => refused("DERIVED_METRICS_INVALID", 422),
            })?;
        let scope = Scope::new(project, run, attempt)
            .encoded()
            .map_err(why!(cloud::Error::evidence()))?;
        let batch = records
            .append_derived(&guard, &scope, sha256, now, &observations)
            .map_err(|error| match error {
                // No journal scope means the Attempt never reported, so it has not ended.
                crate::records::Error::NotEnded | crate::records::Error::Scope => {
                    refused("DERIVED_RUN_NOT_ENDED", 409)
                }
                crate::records::Error::DerivedNameConflict => {
                    refused("DERIVED_METRIC_NAME_CONFLICT", 409)
                }
                error => {
                    crate::why::record(file!(), line!(), &error);
                    cloud::Error::evidence()
                }
            })?;
        let status = records
            .scope_status(&guard, &scope)
            .map_err(why!(cloud::Error::evidence()))?
            .ok_or_else(cloud::Error::evidence)?;
        Ok(
            json!({"project":project,"run_id":run,"attempt_id":attempt,"batch":batch.document(),
            "wandb":{"enabled":binding["route"]["enabled"],"wandb_id":status["wandb_id"]},
            "training_records_changed":false,"result_archive_changed":false}),
        )
    }
}
