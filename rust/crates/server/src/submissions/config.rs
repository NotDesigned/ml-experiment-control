//! Trusted operator configuration; clients never supply SSH commands or image
//! materialization receipts. Parsing is pure and opening files is explicit.
use super::{Backend, Error};
use crate::backends::slurm::{Budget, SshConfig};
use ml_exp_core::first_submission::native::{Profile, Runtime};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub schema_version: u32,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub allow_run_creation: bool,
    #[serde(default)]
    pub polling: ml_exp_core::polling::Policy,
    pub backends: Vec<Definition>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub profile: Profile,
    pub ssh: Ssh,
    pub runtimes: BTreeMap<String, Runtime>,
    #[serde(default)]
    pub data_helper_config: Option<String>,
    #[serde(default)]
    pub script_preparation: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ssh {
    pub binary: PathBuf,
    pub config_file: PathBuf,
    pub known_hosts_file: PathBuf,
    pub helper_path: String,
    pub helper_config_path: String,
    pub local_cwd: PathBuf,
    pub agent_socket: Option<PathBuf>,
    pub timeout_seconds: u32,
    pub connect_timeout_seconds: u32,
}
impl Settings {
    pub fn from_slice(bytes: &[u8]) -> Result<Self, Error> {
        // The shared parser rejects duplicate keys before typed deserialization.
        let value = crate::experiments::parse_mapping(bytes, crate::experiments::Format::Json)
            .map_err(|_| Error::Configuration)?;
        let settings: Self = serde_json::from_value(serde_json::Value::Object(value))
            .map_err(|_| Error::Configuration)?;
        if settings.schema_version != 1
            || !settings.polling.validate()
            || settings.backends.is_empty()
            || settings.backends.len() > 64
        {
            return Err(Error::Configuration);
        }
        Ok(settings)
    }
    pub fn into_backends(self) -> Result<Loaded, Error> {
        let enabled = self.enabled;
        let creation = self.allow_run_creation;
        let backends = self
            .backends
            .into_iter()
            .map(|definition| {
                let ssh = definition.ssh;
                if !(1..=90).contains(&ssh.timeout_seconds)
                    || ssh.connect_timeout_seconds > ssh.timeout_seconds
                {
                    return Err(Error::Configuration);
                }
                let backend = Backend {
                    ssh: SshConfig {
                        target: definition.profile.target.clone(),
                        ssh_binary: ssh.binary,
                        config_file: ssh.config_file,
                        known_hosts_file: ssh.known_hosts_file,
                        target_alias: definition.profile.ssh_alias.clone(),
                        helper_path: ssh.helper_path,
                        helper_config_path: ssh.helper_config_path,
                        local_cwd: ssh.local_cwd,
                        agent_socket: ssh.agent_socket,
                        budget: Budget {
                            timeout: Duration::from_secs(ssh.timeout_seconds.into()),
                            connect_timeout_seconds: ssh.connect_timeout_seconds,
                            termination_grace: Duration::from_secs(2),
                            drain_grace: Duration::from_secs(2),
                        },
                    },
                    profile: definition.profile,
                    runtimes: definition.runtimes,
                    data_helper_config: definition.data_helper_config,
                    script_preparation: definition.script_preparation,
                };
                backend.validate()?;
                Ok(backend)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Loaded {
            enabled,
            creation,
            backends,
            polling: self.polling,
        })
    }
}
pub struct Loaded {
    pub enabled: bool,
    pub creation: bool,
    pub backends: Vec<Backend>,
    pub polling: ml_exp_core::polling::Policy,
}
pub fn load(path: &Path) -> Result<Loaded, Error> {
    let bytes =
        ml_exp_io::read_private(path, 4 * 1024 * 1024, true).map_err(|_| Error::Configuration)?;
    Settings::from_slice(&bytes)?.into_backends()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_operator_configuration_refuses_unknown_duplicate_and_invalid_bounds() {
        for bytes in [
            br#"{"schema_version":1,"schema_version":1,"backends":[]}"#.as_slice(),
            br#"{"schema_version":1,"backends":[],"shell":"unsafe"}"#,
            br#"{"schema_version":2,"backends":[]}"#,
            br#"{"schema_version":1,"backends":[]}"#,
        ] {
            assert!(Settings::from_slice(bytes).is_err());
        }
    }
}
