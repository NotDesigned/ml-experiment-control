//! Bounded worker-to-service exchange. Acknowledgement advances a stream only
//! after its events and checkpoint receipts have been committed durably.
use super::*;
use crate::managed::{
    Config,
    history::{Cursor, Page},
    remote::Evidence,
};

pub const CONTRACT: &str = "native-cloud-exchange.v1";
pub const MAX_BYTES: usize = 2 * 1024 * 1024;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRequest {
    pub package_sha256: String,
    pub operation: ArtifactOperation,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArtifactOperation {
    Prepare {
        bundle: Box<crate::artifacts::publication::Bundle>,
    },
    Ticket {
        part: u64,
    },
    Complete,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sync {
    pub request_id: String,
    pub confirmation: String,
}
impl Sync {
    pub fn validate(&self, action: &str) -> Result<()> {
        crate::operations::validate_submission_id(action).map_err(|_| Error("sync submission"))?;
        crate::submission::validate_submission_token(&self.request_id)
            .map_err(|_| Error("sync request ID"))?;
        if self.confirmation != format!("SYNC {action}") {
            return Err(Error("sync confirmation"));
        }
        Ok(())
    }
    pub fn job(&self, parent: &Job) -> Result<Job> {
        parent.validate()?;
        let mut job = parent.clone();
        job.run_id = format!("sync-{}", self.request_id);
        job.submission_token = canonical::digest(&json!([
            "native-cloud-sync.v1",
            parent.submission_token,
            self.request_id
        ]))
        .map_err(|_| Error("sync identity"))?[..32]
            .into();
        job.gpus = 0;
        job.cpus = 2;
        job.memory_gb = 4;
        job.max_time_seconds = 600;
        job.command.push("--sync".into());
        job.validate()?;
        Ok(job)
    }
}

// No Debug: this is a scope-limited credential, never the daemon credential.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Channel {
    pub url: String,
    pub token: String,
}
impl Channel {
    pub fn validate(&self, identity: &crate::launch::Identity) -> Result<()> {
        crate::identity::validate_id(&identity.project)
            .and_then(|_| crate::identity::validate_id(&identity.run_id))
            .and_then(|_| crate::identity::validate_attempt(&identity.attempt_id))
            .map_err(|_| Error("exchange identity"))?;
        let uri = url::Url::parse(&self.url).map_err(|_| Error("exchange URL"))?;
        let path = format!(
            "/api/worker-exchange/{}/{}/{}",
            identity.project, identity.run_id, identity.attempt_id
        );
        if uri.scheme() != "https"
            || uri.host_str().is_none()
            || !uri.username().is_empty()
            || uri.password().is_some()
            || uri.query().is_some()
            || uri.fragment().is_some()
            || !uri.path().ends_with(&path)
            || !crate::slurm::helper::valid_digest(&format!("sha256:{}", self.token))
        {
            return Err(Error("exchange channel"));
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Packet {
    pub contract: String,
    pub sequence: u64,
    pub previous_sha256: Option<String>,
    pub package_sha256: String,
    pub evidence: Evidence,
    pub history: Option<Page>,
    pub checkpoint: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<crate::input_progress::Record>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_error: Option<String>,
}
impl Packet {
    pub fn validate(&self, config: &Config, package_sha256: &str) -> Result<()> {
        if self.contract != CONTRACT
            || self.sequence == 0
            || self.package_sha256 != package_sha256
            || !helper::valid_digest(package_sha256)
            || (self.sequence == 1) != self.previous_sha256.is_none()
            || self
                .previous_sha256
                .as_ref()
                .is_some_and(|s| !helper::valid_digest(s))
            || serde_json::to_vec(self)
                .map_err(|_| Error("exchange encoding"))?
                .len()
                > MAX_BYTES
        {
            return Err(Error("exchange packet"));
        }
        self.evidence
            .validate_config(config)
            .map_err(|_| Error("exchange evidence"))?;
        let binding = config.binding().map_err(|_| Error("exchange binding"))?;
        self.evidence
            .validate_scope(
                &helper::Scope {
                    project: binding.identity.project,
                    run_id: binding.identity.run_id,
                    attempt_id: binding.identity.attempt_id,
                },
                &self.evidence.job_id,
            )
            .map_err(|_| Error("exchange scope"))?;
        if let Some(page) = &self.history {
            page.validate(page.start.as_ref())
                .map_err(|_| Error("exchange page"))?;
            if page.config_sha256 != self.evidence.config_sha256
                || page.process_finished
                    != (self.evidence.worker.phase == crate::managed::evidence::Phase::Finished)
            {
                return Err(Error("exchange history binding"));
            }
        }
        if self.inputs.len() > config.datasets.len()
            || self
                .checkpoint_error
                .as_ref()
                .is_some_and(|v| v != "CHECKPOINT_VERIFICATION_FAILED")
        {
            return Err(Error("exchange diagnostics"));
        }
        let mut names = std::collections::BTreeSet::new();
        for input in &self.inputs {
            let line = format!(
                "{}{}",
                crate::input_progress::MARKER,
                serde_json::to_string(input).map_err(|_| Error("input progress"))?
            );
            if crate::input_progress::parse_line(&line).record.as_ref() != Some(input)
                || !config
                    .datasets
                    .iter()
                    .any(|m| m.receipt.asset_id == input.asset_id)
                || !names.insert(&input.asset_id)
            {
                return Err(Error("input progress binding"));
            }
        }
        Ok(())
    }
    pub fn sha256(&self) -> Result<String> {
        Ok(helper::sha256_bytes(
            &serde_json::to_vec(self).map_err(|_| Error("exchange encoding"))?,
        ))
    }
    pub fn follows(
        &self,
        sequence: u64,
        previous: Option<&str>,
        cursor: Option<&Cursor>,
        complete: bool,
    ) -> Result<()> {
        if self.sequence != sequence.checked_add(1).ok_or(Error("exchange sequence"))?
            || self.previous_sha256.as_deref() != previous
        {
            return Err(Error("exchange order"));
        }
        if let Some(page) = &self.history {
            if complete {
                return Err(Error("history already drained"));
            }
            page.validate(cursor)
                .map_err(|_| Error("exchange cursor"))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ack {
    pub contract: String,
    pub sequence: u64,
    pub packet_sha256: String,
}
impl Ack {
    pub fn for_packet(packet: &Packet) -> Result<Self> {
        Ok(Self {
            contract: CONTRACT.into(),
            sequence: packet.sequence,
            packet_sha256: packet.sha256()?,
        })
    }
    pub fn validate(&self, packet: &Packet) -> Result<()> {
        if *self != Self::for_packet(packet)? {
            return Err(Error("exchange acknowledgement"));
        }
        Ok(())
    }
}
