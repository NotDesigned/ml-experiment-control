//! Explicit backend storage maintenance. A reviewed plan fixes the exact native
//! Attempt configuration and selected paths; datasets and source trees are out
//! of scope. Scheduler terminal evidence is an outer-shell admission rule.
use crate::{canonical, checkpoints, managed, slurm::helper};
use serde::{Deserialize, Serialize};
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Selection {
    Outputs,
    State,
    Checkpoint { checkpoint: Box<serde_json::Value> },
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub contract: String,
    pub config: managed::Config,
    pub selection: Selection,
    pub request_id: String,
}
impl Plan {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.config.validate().map_err(|_| "maintenance config")?;
        crate::submission::validate_submission_token(&self.request_id)
            .map_err(|_| "maintenance request identity")?;
        if self.contract != "native-storage-maintenance.v1" {
            return Err("maintenance contract");
        }
        let binding = self.config.binding().map_err(|_| "maintenance binding")?;
        let identity = &binding.identity;
        let suffix = if self.config.output_directory.is_some() {
            format!(
                "/{}/{}/attempts/{}",
                identity.project, identity.run_id, identity.attempt_id
            )
        } else {
            format!(
                "/{}/{}/{}",
                identity.project, identity.run_id, identity.attempt_id
            )
        };
        if !self
            .storage_root()
            .to_str()
            .is_some_and(|p| p.ends_with(&suffix))
            || self.config.output_directory.is_some()
                && (self.config.attempt_root != self.storage_root().join("worker")
                    || self.config.output_path() != self.storage_root().join("outputs/payload"))
        {
            return Err("maintenance storage layout");
        }
        if let Selection::Checkpoint { checkpoint } = &self.selection {
            let (context, _) = checkpoints::validate_document(checkpoint, None, None)
                .map_err(|_| "maintenance checkpoint")?;
            let binding = self.config.binding().map_err(|_| "maintenance binding")?;
            if context.project != binding.identity.project
                || context.run_id != binding.identity.run_id
                || context.attempt_id != binding.identity.attempt_id
                || std::path::Path::new(&context.state_root) != self.state_root()
            {
                return Err("maintenance scope");
            }
        }
        Ok(())
    }
    pub fn id(&self) -> Result<String, &'static str> {
        self.validate()?;
        canonical::digest(&serde_json::json!(self)).map_err(|_| "maintenance identity")
    }
    pub fn storage_root(&self) -> std::path::PathBuf {
        if self.config.output_directory.is_some() {
            self.config
                .attempt_root
                .parent()
                .unwrap_or(&self.config.attempt_root)
                .into()
        } else {
            self.config.attempt_root.clone()
        }
    }
    pub fn state_root(&self) -> std::path::PathBuf {
        if self.config.output_directory.is_some() {
            self.config.output_path()
        } else {
            self.config.attempt_root.join("state")
        }
    }
    pub fn paths(&self) -> Result<Vec<String>, &'static str> {
        self.validate()?;
        Ok(match &self.selection {
            Selection::Outputs => vec![
                self.config
                    .output_path()
                    .strip_prefix(self.storage_root())
                    .map_err(|_| "output scope")?
                    .to_str()
                    .ok_or("output encoding")?
                    .into(),
            ],
            Selection::State => vec![if self.config.output_directory.is_some() {
                "outputs/payload/checkpoints".into()
            } else {
                "state".into()
            }],
            Selection::Checkpoint { checkpoint } => {
                let plan = checkpoints::restore_plan(checkpoint, None)
                    .map_err(|_| "maintenance checkpoint")?;
                vec![format!(
                    "{}/checkpoints/{}",
                    self.state_root()
                        .strip_prefix(self.storage_root())
                        .map_err(|_| "state scope")?
                        .to_str()
                        .ok_or("state encoding")?,
                    plan.generation
                )]
            }
        })
    }
    pub fn receipt(&self) -> Result<serde_json::Value, &'static str> {
        Ok(
            serde_json::json!({"contract":self.contract,"plan_sha256":helper::sha256_bytes(&serde_json::to_vec(self).map_err(|_|"maintenance receipt")?),"plan_id":self.id()?,"identity":self.config.binding().map_err(|_|"maintenance binding")?.identity,"paths":self.paths()?,"status":"REMOVED"}),
        )
    }
}
