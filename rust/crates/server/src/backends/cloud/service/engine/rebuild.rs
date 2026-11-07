//! A W&B rebuild of one finished Attempt: a new W&B Run published from the
//! platform's own records without the names of one exclusion list. The old W&B
//! Run is never touched or deleted; its id stays in the rebuild's audit.
use super::*;
use ml_exp_core::records::{RebuildRefusal, Scope};

fn refused(code: &str, status: u16) -> cloud::Error {
    cloud::Error {
        code: code.into(),
        http_status: Some(status),
        uncertain: false,
    }
}

impl Service {
    /// `sha256` is the digest of `names`, checked by the HTTP boundary. The list
    /// of the latest rebuild again returns that rebuild and changes nothing.
    pub fn rebuild_wandb(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        sha256: &str,
        names: &str,
        now: &str,
    ) -> cloud::Result<Value> {
        let excluded = ml_exp_core::records::rebuild_names(names.as_bytes()).map_err(
            |refusal| match refusal {
                RebuildRefusal::TooLarge => refused("WANDB_REBUILD_LIST_TOO_LARGE", 413),
                RebuildRefusal::Invalid => refused("WANDB_REBUILD_NAMES_INVALID", 422),
            },
        )?;
        let prepared = self
            .saved_attempt(project, run, attempt)?
            .ok_or_else(missing)?;
        self.require_live(project, run)?;
        if prepared.execution.job.attempt_id != attempt {
            return Err(missing());
        }
        // A Run the provider stopped never writes its own end into the journal.
        let provider_ended =
            crate::tracking::Source::provider_terminal(self, project, run, attempt)
                .ok()
                .flatten()
                .is_some();
        let records = self.exchange_records(&prepared)?;
        let guard = records
            .guard(project, run)
            .map_err(why!(cloud::Error::evidence()))?;
        let binding = records
            .binding(&guard)
            .map_err(why!(cloud::Error::evidence()))?
            .ok_or_else(cloud::Error::evidence)?;
        let scope = Scope::new(project, run, attempt)
            .encoded()
            .map_err(why!(cloud::Error::evidence()))?;
        let (rebuild, created) = records
            .rebuild_wandb(&guard, &scope, sha256, names, now, provider_ended)
            .map_err(|error| match error {
                crate::records::Error::RebuildNotFinished | crate::records::Error::Scope => {
                    refused("WANDB_REBUILD_PUBLICATION_NOT_FINISHED", 409)
                }
                crate::records::Error::NotEnded => refused("WANDB_REBUILD_RUN_NOT_ENDED", 409),
                error => {
                    crate::why::record(file!(), line!(), &error);
                    cloud::Error::evidence()
                }
            })?;
        let route = &binding["route"];
        let url = |id: &str| {
            format!(
                "https://wandb.ai/{}/{}/runs/{id}",
                route["entity"].as_str().unwrap_or_default(),
                route["project"].as_str().unwrap_or_default()
            )
        };
        Ok(json!({"project":project,"run_id":run,"attempt_id":attempt,
            "status":if created {"STARTED"} else {"ALREADY_REQUESTED"},
            "names":excluded.len(),"rebuild":rebuild.document(),
            "wandb":{"wandb_id":rebuild.wandb_id,"url":url(&rebuild.wandb_id),
                "previous_wandb_id":rebuild.previous_wandb_id,"previous_url":url(&rebuild.previous_wandb_id),
                "display_name":format!("{run}/{attempt}"),"previous_wandb_run_deleted":false},
            "training_records_changed":false,"result_archive_changed":false}))
    }
}
