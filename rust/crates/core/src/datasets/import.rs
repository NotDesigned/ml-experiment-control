//! Direct backend dataset materialization. Transport locations are temporary
//! private inputs; only the immutable archive/file identity enters a receipt.
use super::*;
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Source {
    DatasetParts {
        channel: DataChannel,
    },
    ExchangeParts {
        channel: crate::cloud::exchange::Channel,
    },
    File {
        path: String,
    },
    Https {
        url: String,
        allowed_hosts: BTreeSet<String>,
    },
}
/// A data-only capability. It cannot register metrics, checkpoints or results,
/// and is never an API credential or a credential sent to object storage.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataChannel {
    pub url: String,
    pub token: String,
}
impl DataChannel {
    pub fn validate(&self, project: &str, executor: &str, asset: &str) -> Result<()> {
        let uri = url::Url::parse(&self.url).map_err(|_| Error)?;
        if uri.scheme() != "https"
            || uri.host_str().is_none()
            || !uri.username().is_empty()
            || uri.password().is_some()
            || uri.query().is_some()
            || uri.fragment().is_some()
            || !uri.path().ends_with(&format!(
                "/api/dataset-exchange/{project}/{executor}/{asset}"
            ))
            || !crate::slurm::helper::valid_digest(&format!("sha256:{}", self.token))
        {
            return Err(Error);
        }
        Ok(())
    }
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub root: String,
    pub target: Target,
    pub project: String,
    pub definition: Definition,
    pub source: Source,
    pub timeout_seconds: u64,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        self.target.validate().map_err(|_| Error)?;
        identity::validate_id(&self.project).map_err(|_| Error)?;
        self.definition.validate()?;
        if !root_path(&self.root) || self.timeout_seconds == 0 || self.timeout_seconds > 14400 {
            return Err(Error);
        }
        match &self.source {
            Source::DatasetParts { channel } => channel.validate(
                &self.project,
                &self.target.backend_id,
                &self.definition.asset_id(),
            )?,
            Source::ExchangeParts { channel } => {
                let url = url::Url::parse(&channel.url).map_err(|_| Error)?;
                let parts = url.path().rsplitn(4, '/').collect::<Vec<_>>();
                if parts.len() != 4 || parts[2] != self.project {
                    return Err(Error);
                }
                channel
                    .validate(&crate::launch::Identity {
                        project: self.project.clone(),
                        run_id: parts[1].into(),
                        attempt_id: parts[0].into(),
                    })
                    .map_err(|_| Error)?;
            }
            Source::File { path } => {
                if !crate::slurm::absolute_path(path)
                    || path.ends_with('/')
                    || path == &self.root
                    || crate::slurm::path_within(path, &self.root)
                {
                    return Err(Error);
                }
            }
            Source::Https { url, allowed_hosts } => {
                let uri = url::Url::parse(url).map_err(|_| Error)?;
                if allowed_hosts.is_empty()
                    || allowed_hosts.len() > 16
                    || url.len() > 16384
                    || uri.scheme() != "https"
                    || !uri.username().is_empty()
                    || uri.password().is_some()
                    || uri.fragment().is_some()
                    || uri.port_or_known_default() != Some(443)
                    || !matches!(uri.host(),Some(url::Host::Domain(host)) if host.contains('.')&&!host.ends_with(".localhost")&&!host.ends_with(".local")&&allowed_hosts.contains(host))
                {
                    return Err(Error);
                }
                for host in allowed_hosts {
                    if host.len() > 253
                        || !host.contains('.')
                        || !host.bytes().all(|b| {
                            b.is_ascii_lowercase() || b.is_ascii_digit() || b".-".contains(&b)
                        })
                    {
                        return Err(Error);
                    }
                }
            }
        }
        Ok(())
    }
    pub fn request(&self, command: Command) -> Result<Request> {
        self.validate()?;
        let request = Request {
            version: 1,
            project: self.project.clone(),
            target: self.target.clone(),
            command,
        };
        request.validate()?;
        Ok(request)
    }
}
pub use crate::input_progress::Phase as Stage;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    pub stage: Stage,
    pub asset_id: String,
    pub received_bytes: u64,
    pub total_bytes: u64,
    pub completed_parts: u64,
    pub total_parts: u64,
    pub resumed_parts: u64,
    pub error_code: Option<String>,
}
impl Config {
    pub fn progress(
        &self,
        state: Option<&State>,
        stage: Stage,
        resumed_parts: u64,
        error_code: Option<&str>,
    ) -> Result<Progress> {
        self.validate()?;
        if let Some(state) = state {
            state.validate()?;
            if state.definition != self.definition
                || state.project != self.project
                || state.target != self.target
            {
                return Err(Error);
            }
        }
        let completed = state.map_or(0, |s| s.parts.len() as u64);
        let received = state
            .map(|s| {
                s.parts
                    .values()
                    .try_fold(0u64, |n, p| n.checked_add(p.bytes).ok_or(Error))
            })
            .transpose()?
            .unwrap_or(0);
        if resumed_parts > completed {
            return Err(Error);
        }
        Ok(Progress {
            stage,
            asset_id: self.definition.asset_id(),
            received_bytes: received,
            total_bytes: self.definition.archive_bytes,
            completed_parts: completed,
            total_parts: self.definition.part_count()?,
            resumed_parts,
            error_code: error_code.map(str::to_owned),
        })
    }
}
