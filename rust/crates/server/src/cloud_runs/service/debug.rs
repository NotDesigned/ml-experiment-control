//! Durable debug lifecycle. Every non-idempotent effect has a saved claim;
//! refresh observes that claim after response loss instead of creating again.
use super::*;
use ml_exp_core::cloud::debug as domain;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub profile: domain::Profile,
    pub credentials_file: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    #[serde(default)]
    pub enabled: bool,
    pub max_active: usize,
    pub profiles: BTreeMap<String, Definition>,
}
pub struct Runtime {
    pub enabled: bool,
    max_active: usize,
    profiles: BTreeMap<String, (domain::Profile, Client)>,
}
impl Runtime {
    pub fn open(settings: Settings) -> cloud::Result<Self> {
        if settings.max_active == 0
            || settings.max_active > 16
            || settings.profiles.is_empty()
            || settings.profiles.len() > 64
        {
            return Err(cloud::Error::configuration());
        }
        let profiles = settings
            .profiles
            .into_iter()
            .map(|(name, d)| {
                ml_exp_core::identity::validate_id(&name)
                    .map_err(|_| cloud::Error::configuration())?;
                d.profile
                    .validate()
                    .map_err(|_| cloud::Error::configuration())?;
                let credentials = Credentials::load(&d.credentials_file)?;
                let provider_matches = matches!(
                    (&credentials, &d.profile),
                    (Credentials::Pai { .. }, domain::Profile::PaiDsw { .. })
                        | (
                            Credentials::Sensecore { .. },
                            domain::Profile::SensecoreCci { .. }
                        )
                );
                if !provider_matches {
                    return Err(cloud::Error::configuration());
                }
                Ok((
                    name,
                    (
                        d.profile,
                        Client::new(credentials, Duration::from_secs(30))?,
                    ),
                ))
            })
            .collect::<cloud::Result<_>>()?;
        Ok(Self {
            enabled: settings.enabled,
            max_active: settings.max_active,
            profiles,
        })
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    recipe: domain::Recipe,
    provider_id: Option<String>,
    claims: BTreeSet<String>,
    stop_requested: bool,
    delete_requested: bool,
    terminal: bool,
    #[serde(default)]
    network_clean: bool,
    deleted: bool,
    connection: Option<Value>,
    diagnostic: Option<String>,
    observed_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    create_rejection: Option<CreateRejection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    create_error: Option<cloud::FailureEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_error: Option<OperationError>,
}
impl Saved {
    fn needs_refresh(&self, timestamp: i64) -> bool {
        !self.deleted
            && (self.delete_requested || !(self.terminal && self.network_clean))
            && (self.recipe.expires_at <= timestamp
                || self.stop_requested
                || self.delete_requested
                || self.terminal)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationError {
    operation: String,
    response: cloud::FailureEvidence,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateRejection {
    code: String,
    http_status: u16,
}
impl CreateRejection {
    fn from_error(error: &cloud::Error) -> Option<Self> {
        // This is a positive DSW validation rejection, not a transport failure
        // or an absent lookup after a possibly accepted create.
        (error.http_status == Some(400) && error.code == "ValidationError").then(|| Self {
            code: error.code.clone(),
            http_status: 400,
        })
    }
}
fn name(id: &str) -> cloud::Result<String> {
    if id
        .strip_prefix("debug.")
        .is_none_or(|s| ml_exp_core::sources::validate_archive_digest(s).is_err())
    {
        return Err(cloud::Error::configuration());
    }
    Ok(format!("{id}.debug.json"))
}
impl Service {
    pub fn has_debug(&self) -> bool {
        self.debug.as_ref().is_some_and(|d| d.enabled)
    }
    fn debug_save(&self, record: &Saved) -> cloud::Result<()> {
        self.slots
            .replace_regular(
                OsStr::new(&name(&record.recipe.identity)?),
                &encode(record)?,
            )
            .map_err(io)
    }
    fn debug_load(&self, id: &str) -> cloud::Result<Saved> {
        let saved: Saved = serde_json::from_value(read(&self.slots, &name(id)?)?)
            .map_err(|_| cloud::Error::evidence())?;
        let runtime = self
            .debug
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let (profile, _) = runtime
            .profiles
            .get(&saved.recipe.request.profile)
            .ok_or_else(cloud::Error::configuration)?;
        if saved.recipe.identity != id || profile.check_recipe(&saved.recipe).is_err() {
            return Err(cloud::Error::evidence());
        }
        if let Some(rejection) = &saved.create_rejection {
            if rejection.http_status != 400
                || rejection.code != "ValidationError"
                || saved.provider_id.is_some()
                || !saved.claims.contains("create")
                || !saved.terminal
                || !saved.network_clean
                || !matches!(profile, domain::Profile::PaiDsw { .. })
            {
                return Err(cloud::Error::evidence());
            }
        }
        Ok(saved)
    }
    fn debug_records(&self) -> cloud::Result<Vec<Saved>> {
        let mut records = Vec::new();
        for n in self.slots.entry_names_limited(32768).map_err(io)? {
            if let Some(id) = n.to_str().and_then(|s| s.strip_suffix(".debug.json")) {
                records.push(self.debug_load(id)?);
            }
            if records.len() > 4096 {
                return Err(cloud::Error::evidence());
            }
        }
        Ok(records)
    }
    fn debug_view(record: &Saved) -> Value {
        let provider_status = domain::status(record.provider_state.as_deref(), record.terminal);
        let state = if record.deleted {
            "DELETED"
        } else if record.delete_requested {
            "DELETING"
        } else if record.create_rejection.is_some() {
            "FAILED"
        } else if record.terminal && record.network_clean {
            provider_status
        } else if record.stop_requested || record.terminal {
            "STOPPING"
        } else if record.connection.is_some() && provider_status == "RUNNING" {
            "RUNNING"
        } else if record.provider_id.is_some() {
            if provider_status == "QUEUED" {
                "QUEUED"
            } else {
                "STARTING"
            }
        } else {
            "RECONCILE_REQUIRED"
        };
        json!({"debug_id":record.recipe.identity,"profile":record.recipe.request.profile,"name":record.recipe.name,"provider_id":record.provider_id,
            "state":state,
            "resources":{"gpus":record.recipe.request.gpus,"cpus":record.recipe.request.cpus,"memory_gb":record.recipe.request.memory_gb},"expires_at":record.recipe.expires_at,
            "connection":record.connection.as_ref().filter(|_|state=="RUNNING"),"diagnostic":record.diagnostic,"provider_state":record.provider_state,"provider_message":record.provider_message,"create_rejection":record.create_rejection,"create_error":record.create_error,"last_error":record.last_error,"observed_at":record.observed_at,"confirmation":format!("STOP {}",record.recipe.identity),"retry_safe":false})
    }
    pub fn debug_list(&self) -> cloud::Result<Value> {
        let runtime = self
            .debug
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let profiles=runtime.profiles.iter().map(|(name,(profile,_))|json!({"name":name,"provider":if matches!(profile,domain::Profile::PaiDsw{..}){"pai_dsw"}else{"sensecore_cci"},"delete_supported":matches!(profile,domain::Profile::PaiDsw{..})})).collect::<Vec<_>>();
        Ok(
            json!({"enabled":runtime.enabled,"profiles":profiles,"machines":self.debug_records()?.iter().map(Self::debug_view).collect::<Vec<_>>()}),
        )
    }
    pub fn debug_status(&self, id: &str) -> cloud::Result<Value> {
        Ok(Self::debug_view(&self.debug_load(id)?))
    }
    pub fn debug_create(&self, request: domain::Create, now: &str) -> cloud::Result<Value> {
        let lock = self.lock()?;
        let runtime = self
            .debug
            .as_ref()
            .filter(|d| d.enabled)
            .ok_or_else(cloud::Error::configuration)?;
        let (profile, _) = runtime
            .profiles
            .get(&request.profile)
            .ok_or_else(cloud::Error::configuration)?;
        let recipe = profile
            .recipe(request.clone(), now)
            .map_err(|_| cloud::Error::configuration())?;
        let records = self.debug_records()?;
        if let Some(existing) = records
            .iter()
            .find(|r| r.recipe.request.request_id == request.request_id)
        {
            if existing.recipe.request != request
                || existing.recipe.profile_sha256 != recipe.profile_sha256
            {
                return Err(cloud::Error::evidence());
            }
            return Ok(Self::debug_view(existing));
        }
        if records
            .iter()
            .filter(|r| !(r.terminal && r.network_clean || r.deleted))
            .count()
            >= runtime.max_active
            || matches!(profile, domain::Profile::SensecoreCci { .. })
                && records.iter().any(|r| {
                    r.recipe.request.profile == request.profile
                        && !(r.terminal && r.network_clean || r.deleted)
                })
        {
            return Err(cloud::Error {
                code: "DEBUG_MACHINE_LIMIT".into(),
                http_status: Some(409),
                uncertain: false,
            });
        }
        let record = Saved {
            recipe,
            provider_id: None,
            claims: BTreeSet::new(),
            stop_requested: false,
            delete_requested: false,
            terminal: false,
            network_clean: matches!(profile, domain::Profile::PaiDsw { .. }),
            deleted: false,
            connection: None,
            diagnostic: None,
            observed_at: None,
            create_rejection: None,
            create_error: None,
            provider_state: None,
            provider_message: None,
            last_error: None,
        };
        lock.validate()?;
        self.debug_save(&record)?;
        drop(lock);
        self.debug_refresh(&record.recipe.identity, now)
    }
    fn debug_lookup(
        profile: &domain::Profile,
        client: &Client,
        saved: &Saved,
    ) -> cloud::Result<Option<Value>> {
        let request = profile
            .get(&saved.recipe, saved.provider_id.as_deref())
            .map_err(|_| cloud::Error::configuration())?;
        let result = client.call(&request, &AtomicBool::new(false));
        let value = match result {
            Ok(v) => v,
            Err(e) if e.http_status == Some(404) || e.code == "InstanceNotExistError" => {
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        if matches!(profile, domain::Profile::PaiDsw { .. }) && saved.provider_id.is_none() {
            let rows = value["Instances"]
                .as_array()
                .ok_or_else(cloud::Error::evidence)?;
            let total = value["TotalCount"]
                .as_u64()
                .ok_or_else(cloud::Error::evidence)?;
            if total != rows.len() as u64 || total > 100 {
                return Err(cloud::Error::evidence());
            }
            let matching = rows
                .iter()
                .filter(|r| r["InstanceName"] == saved.recipe.name)
                .collect::<Vec<_>>();
            if matching.len() > 1 {
                return Err(cloud::Error::evidence());
            }
            let Some(row) = matching.first() else {
                return Ok(None);
            };
            let id = row["InstanceId"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?;
            let value = client.call(
                &profile
                    .get(&saved.recipe, Some(id))
                    .map_err(|_| cloud::Error::configuration())?,
                &AtomicBool::new(false),
            )?;
            profile
                .checked(&saved.recipe, None, &value)
                .map_err(|_| cloud::Error::evidence())?;
            return Ok(Some(value));
        }
        profile
            .checked(&saved.recipe, saved.provider_id.as_deref(), &value)
            .map_err(|_| cloud::Error::evidence())?;
        Ok(Some(value))
    }
    fn debug_call(
        &self,
        saved: &mut Saved,
        client: &Client,
        request: &ml_exp_core::cloud::Request,
    ) -> cloud::Result<Value> {
        let mut evidence = None;
        let result = client.call_observed(request, &AtomicBool::new(false), &mut |v| {
            evidence = Some(v);
        });
        if let Some(evidence) = evidence {
            if request.action == "CreateInstance" {
                saved.create_error = Some(evidence.clone());
            }
            saved.last_error = Some(OperationError {
                operation: request.action.clone(),
                response: evidence,
            });
            self.debug_save(saved)?;
        }
        result
    }
    fn debug_effect(
        &self,
        saved: &mut Saved,
        label: &str,
        client: &Client,
        request: ml_exp_core::cloud::Request,
    ) -> cloud::Result<Option<Value>> {
        if saved.claims.contains(label) {
            return Ok(None);
        }
        saved.claims.insert(label.into());
        self.debug_save(saved)?;
        let result = self.debug_call(saved, client, &request);
        if label == "create"
            && matches!(
                request.endpoint,
                ml_exp_core::cloud::Endpoint::Pai {
                    product: ml_exp_core::cloud::Product::PaiDsw,
                    ..
                }
            )
        {
            if let Err(error) = &result {
                if let Some(rejection) = CreateRejection::from_error(error) {
                    saved.create_rejection = Some(rejection);
                    saved.terminal = true;
                    saved.network_clean = true;
                    self.debug_save(saved)?;
                }
            }
        }
        result.map(Some)
    }
    pub fn debug_refresh(&self, id: &str, now: &str) -> cloud::Result<Value> {
        let lock = self.lock()?;
        let mut saved = self.debug_load(id)?;
        if saved.deleted {
            return Ok(Self::debug_view(&saved));
        }
        if saved.create_rejection.is_some() {
            if saved.delete_requested {
                saved.deleted = true;
                self.debug_save(&saved)?;
            }
            return Ok(Self::debug_view(&saved));
        }
        let runtime = self
            .debug
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        let (profile, client) = &runtime.profiles[&saved.recipe.request.profile];
        let timestamp =
            time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
                .map_err(|_| cloud::Error::configuration())?
                .unix_timestamp();
        if timestamp >= saved.recipe.expires_at {
            saved.stop_requested = true;
            self.debug_save(&saved)?;
        }
        let result = (|| {
            let mut value = Self::debug_lookup(profile, client, &saved)?;
            if value.is_none() {
                if saved.delete_requested
                    && saved.claims.contains("delete")
                    && saved.provider_id.is_some()
                {
                    saved.deleted = true;
                    saved.network_clean = true;
                    saved.connection = None;
                    return Ok(());
                }
                if saved.provider_id.is_some() || saved.claims.contains("create") {
                    return Err(cloud::Error {
                        code: "DEBUG_CREATE_UNCERTAIN".into(),
                        http_status: None,
                        uncertain: true,
                    });
                }
                if saved.stop_requested {
                    saved.terminal = true;
                    saved.network_clean = true;
                    return Ok(());
                }
                let create = profile
                    .create(&saved.recipe)
                    .map_err(|_| cloud::Error::configuration())?;
                let response = self.debug_effect(&mut saved, "create", client, create)?;
                if matches!(profile, domain::Profile::PaiDsw { .. }) {
                    if let Some(id) = response.as_ref().and_then(|v| v["InstanceId"].as_str()) {
                        ml_exp_core::identity::validate_id(id)
                            .map_err(|_| cloud::Error::evidence())?;
                        saved.provider_id = Some(id.into());
                        self.debug_save(&saved)?;
                    }
                }
                value = Self::debug_lookup(profile, client, &saved)?;
            }
            let Some(value) = value else {
                return Err(cloud::Error {
                    code: "DEBUG_CREATE_UNCERTAIN".into(),
                    http_status: None,
                    uncertain: true,
                });
            };
            saved.provider_id = Some(
                profile
                    .checked(&saved.recipe, saved.provider_id.as_deref(), &value)
                    .map_err(|_| cloud::Error::evidence())?,
            );
            self.debug_save(&saved)?;
            let provider_id = saved
                .provider_id
                .clone()
                .ok_or_else(cloud::Error::evidence)?;
            saved.terminal = profile.terminal(&value);
            saved.provider_state = Some(
                profile
                    .provider_state(&value)
                    .map_err(|_| cloud::Error::evidence())?
                    .into(),
            );
            if saved.terminal
                && matches!(profile, domain::Profile::PaiDsw { .. })
                && saved.provider_message.is_none()
            {
                // Diagnostic reads are advisory. Failure to read them must not
                // turn a known provider outcome into an uncertain submission
                // or prevent stopping/deleting this exact owned instance.
                saved.provider_message = profile
                    .lifecycle(&saved.recipe, &provider_id)
                    .ok()
                    .and_then(|r| client.call(&r, &AtomicBool::new(false)).ok())
                    .and_then(|v| {
                        domain::lifecycle_reason(&v, saved.provider_state.as_deref()?)
                            .ok()
                            .flatten()
                            .and_then(|s| client.observation_message(s))
                    });
            }
            if saved.stop_requested || saved.terminal {
                saved.connection = None;
                if !saved.terminal {
                    // Stop is idempotent and always rechecks this exact instance.
                    let request = profile
                        .stop(&saved.recipe, &provider_id, false)
                        .map_err(|_| cloud::Error::configuration())?;
                    self.debug_call(&mut saved, client, &request)?;
                } else if matches!(profile, domain::Profile::SensecoreCci { .. }) {
                    saved.network_clean =
                        self.debug_network_stop(&saved, profile, client, &provider_id)?;
                }
                if saved.delete_requested
                    && saved.terminal
                    && matches!(profile, domain::Profile::PaiDsw { .. })
                {
                    saved.claims.insert("delete".into());
                    self.debug_save(&saved)?;
                    let request = profile
                        .stop(&saved.recipe, &provider_id, true)
                        .map_err(|_| cloud::Error::configuration())?;
                    self.debug_call(&mut saved, client, &request)?;
                    if Self::debug_lookup(profile, client, &saved)?.is_none() {
                        saved.deleted = true;
                    }
                }
            } else {
                match profile {
                    domain::Profile::PaiDsw { .. } => {
                        client.call(
                            &profile
                                .timer(&saved.recipe, &provider_id, timestamp)
                                .map_err(|_| cloud::Error::configuration())?,
                            &AtomicBool::new(false),
                        )?;
                        let mut timer = profile
                            .get(&saved.recipe, Some(&provider_id))
                            .map_err(|_| cloud::Error::configuration())?;
                        timer.path.push_str("/shutdowntimer");
                        timer.action = "GetInstanceShutdownTimer".into();
                        let response = client.call(&timer, &AtomicBool::new(false))?;
                        if response["InstanceId"] != provider_id
                            || response["Success"] != true
                            || !response["RemainingTimeInMs"].as_i64().is_some_and(|n| {
                                n > 0 && n <= (saved.recipe.expires_at - timestamp).max(1) * 1000
                            })
                        {
                            return Err(cloud::Error::evidence());
                        }
                        saved.connection = profile.connection(&value);
                    }
                    domain::Profile::SensecoreCci { network, .. } => {
                        let request = profile
                            .service(&saved.recipe, false)
                            .map_err(|_| cloud::Error::configuration())?;
                        let value = match client.call(&request, &AtomicBool::new(false)) {
                            Ok(v) => Some(v),
                            Err(e) if e.http_status == Some(404) => None,
                            Err(e) => return Err(e),
                        };
                        if let Some(service) = value {
                            if service["name"] != saved.recipe.name
                                || service["selector"] != json!({"app":saved.recipe.name})
                                || !domain::subset(
                                    &json!([{"port":22,"target_port":22}]),
                                    &service["ports"],
                                )
                            {
                                return Err(cloud::Error::evidence());
                            }
                            let request = profile
                                .network(&saved.recipe, "get", &provider_id)
                                .map_err(|_| cloud::Error::configuration())?;
                            let row = match client.call(&request, &AtomicBool::new(false)) {
                                Ok(v) => Some(v),
                                Err(e) if e.http_status == Some(404) => None,
                                Err(e) => return Err(e),
                            };
                            if let Some(row) = row {
                                if profile
                                    .check_network(&saved.recipe, &provider_id, &row)
                                    .map_err(|_| cloud::Error::evidence())?
                                {
                                    saved.connection = Some(
                                        json!({"host":network.external_ip,"port":network.external_port,"user":"root","access":"internet"}),
                                    );
                                } else if row["state"] == "CREATED"
                                    && row["properties"]["internal_instance_name"]
                                        .as_str()
                                        .unwrap_or("")
                                        .is_empty()
                                {
                                    let mut request = profile
                                        .network(&saved.recipe, "bind", &provider_id)
                                        .map_err(|_| cloud::Error::configuration())?;
                                    let mut body = row.clone();
                                    body["properties"]["internal_instance_name"] =
                                        json!(provider_id);
                                    body["properties"]["internal_instance_type"] =
                                        json!("CCI_DEPLOYMENT_SERVICE");
                                    request.body = Some(body);
                                    self.debug_effect(&mut saved, "dnat-bind", client, request)?;
                                }
                            } else {
                                let request = profile
                                    .network(&saved.recipe, "create", &provider_id)
                                    .map_err(|_| cloud::Error::configuration())?;
                                self.debug_effect(&mut saved, "dnat-create", client, request)?;
                            }
                        } else {
                            let request = profile
                                .service(&saved.recipe, true)
                                .map_err(|_| cloud::Error::configuration())?;
                            self.debug_effect(&mut saved, "service", client, request)?;
                        }
                    }
                }
            }
            Ok(())
        })();
        saved.observed_at = Some(now.into());
        saved.diagnostic = result.err().map(|e| e.code);
        lock.validate()?;
        self.debug_save(&saved)?;
        Ok(Self::debug_view(&saved))
    }
    fn debug_network_stop(
        &self,
        saved: &Saved,
        profile: &domain::Profile,
        client: &Client,
        id: &str,
    ) -> cloud::Result<bool> {
        if !saved.claims.contains("dnat-create") {
            return Ok(true);
        }
        let request = profile
            .network(&saved.recipe, "get", id)
            .map_err(|_| cloud::Error::configuration())?;
        let row = match client.call(&request, &AtomicBool::new(false)) {
            Ok(v) => v,
            Err(e) if e.http_status == Some(404) => return Ok(true),
            Err(e) => return Err(e),
        };
        profile
            .check_network(&saved.recipe, id, &row)
            .map_err(|_| cloud::Error::evidence())?;
        let operation = if row["properties"]["internal_instance_name"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
        {
            "unbind"
        } else {
            "delete"
        };
        client.call(
            &profile
                .network(&saved.recipe, operation, id)
                .map_err(|_| cloud::Error::configuration())?,
            &AtomicBool::new(false),
        )?;
        Ok(false)
    }
    pub fn debug_stop(
        &self,
        id: &str,
        confirmation: &str,
        delete: bool,
        now: &str,
    ) -> cloud::Result<Value> {
        if confirmation != format!("{} {id}", if delete { "DELETE" } else { "STOP" }) {
            return Err(cloud::Error::configuration());
        }
        let lock = self.lock()?;
        let mut saved = self.debug_load(id)?;
        if delete
            && matches!(
                self.debug
                    .as_ref()
                    .ok_or_else(cloud::Error::configuration)?
                    .profiles[&saved.recipe.request.profile]
                    .0,
                domain::Profile::SensecoreCci { .. }
            )
        {
            return Err(cloud::Error::configuration());
        }
        saved.stop_requested = true;
        saved.delete_requested |= delete;
        lock.validate()?;
        self.debug_save(&saved)?;
        drop(lock);
        self.debug_refresh(id, now)
    }
    pub fn debug_expire(&self, now: &str) -> cloud::Result<()> {
        if !self.has_debug() {
            return Ok(());
        }
        let timestamp =
            time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
                .map_err(|_| cloud::Error::configuration())?
                .unix_timestamp();
        for saved in self
            .debug_records()?
            .into_iter()
            .filter(|r| r.needs_refresh(timestamp))
            .take(16)
        {
            self.debug_refresh(&saved.recipe.identity, now)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn stopped_receipt() -> Saved {
        serde_json::from_value(json!({
            "recipe":{"contract":"native-debug-machine.v4","identity":format!("debug.{}","0".repeat(64)),"name":"owned","profile_sha256":"0".repeat(64),
                "request":{"profile":"cpu-debug","request_id":"0".repeat(32),"ssh_public_key":"unused public fixture","lease_seconds":60,"gpus":0,"cpus":2,"memory_gb":4},"created_at":"2026-10-30T00:00:00Z","expires_at":1793318460},
            "provider_id":"dsw-owned","claims":["create"],"stop_requested":false,"delete_requested":false,"terminal":true,"network_clean":true,"deleted":false,
            "connection":null,"diagnostic":null,"observed_at":null
        })).unwrap()
    }
    #[test]
    fn provider_failure_is_not_a_normal_stop_and_queued_connections_are_not_advertised() {
        let mut saved = stopped_receipt();
        // Old receipts without a native-state observation cannot prove a normal stop.
        assert_eq!(Service::debug_view(&saved)["state"], "TERMINAL");
        saved.provider_state = Some("Failed".into());
        saved.provider_message = Some("mountPath must be unique".into());
        let view = Service::debug_view(&saved);
        assert_eq!(view["state"], "FAILED");
        assert_eq!(view["provider_message"], "mountPath must be unique");
        saved.provider_state = Some("Stopped".into());
        assert_eq!(Service::debug_view(&saved)["state"], "STOPPED");
        saved.terminal = false;
        saved.connection = Some(json!({"host":"example.test","port":22}));
        saved.provider_state = Some("Queuing".into());
        let view = Service::debug_view(&saved);
        assert_eq!(view["state"], "QUEUED");
        assert!(view["connection"].is_null());
        saved.provider_state = Some("Running".into());
        assert_eq!(Service::debug_view(&saved)["state"], "RUNNING");
    }
    #[test]
    fn asynchronous_delete_stays_visible_and_observed_until_absence_is_confirmed() {
        let mut saved = stopped_receipt();
        let now = saved.recipe.expires_at - 1;
        saved.provider_state = Some("Stopped".into());
        assert_eq!(Service::debug_view(&saved)["state"], "STOPPED");
        assert!(!saved.needs_refresh(now));
        saved.stop_requested = true;
        saved.delete_requested = true;
        saved.claims.insert("delete".into());
        // A successful DeleteInstance reply may precede actual disappearance.
        // Read-only show must report the pending work while the observer polls.
        assert_eq!(Service::debug_view(&saved)["state"], "DELETING");
        assert!(saved.needs_refresh(now));
        saved.diagnostic = Some("CLOUD_EVIDENCE_INVALID".into());
        assert_eq!(Service::debug_view(&saved)["state"], "DELETING");
        assert!(saved.needs_refresh(now));
        saved.deleted = true;
        assert_eq!(Service::debug_view(&saved)["state"], "DELETED");
        assert!(!saved.needs_refresh(now));
    }
    #[test]
    fn only_positive_dsw_validation_rejection_is_terminal_without_a_provider_handle() {
        for (status, code, expected) in [
            (Some(400), "ValidationError", true),
            (None, "ValidationError", false),
            (Some(503), "ValidationError", false),
            (Some(400), "CLOUD_RESPONSE_INVALID", false),
            (Some(409), "AlreadyExists", false),
            (Some(429), "Throttling", false),
        ] {
            let error = cloud::Error {
                http_status: status,
                code: code.into(),
                uncertain: true,
            };
            assert_eq!(CreateRejection::from_error(&error).is_some(), expected);
        }
    }
}
