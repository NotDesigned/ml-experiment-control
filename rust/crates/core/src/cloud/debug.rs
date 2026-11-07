//! Debug-machine recipes and exact ownership checks. No clocks, credentials or
//! provider effects; a machine is independent of training Runs and Attempts.
use super::*;
use base64::Engine;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
fn ssh_field<'a>(bytes: &mut &'a [u8]) -> Option<&'a [u8]> {
    let n = u32::from_be_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
    let value = bytes.get(4..4usize.checked_add(n)?)?;
    *bytes = bytes.get(4 + n..)?;
    Some(value)
}
fn public_key(kind: &str, encoded: &str) -> bool {
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(encoded) else {
        return false;
    };
    let mut wire = bytes.as_slice();
    if ssh_field(&mut wire) != Some(kind.as_bytes()) {
        return false;
    }
    let valid = match kind {
        "ssh-ed25519" => ssh_field(&mut wire).is_some_and(|v| v.len() == 32),
        "ssh-rsa" => {
            let exponent = ssh_field(&mut wire);
            let modulus = ssh_field(&mut wire);
            exponent.is_some_and(|v| {
                !v.is_empty() && v.len() <= 8 && v[0] & 0x80 == 0 && v.iter().any(|b| *b != 0)
            }) && modulus.is_some_and(|v| {
                (256..=1025).contains(&v.len()) && v[0] & 0x80 == 0 && v.iter().any(|b| *b != 0)
            })
        }
        "ecdsa-sha2-nistp256" | "ecdsa-sha2-nistp384" | "ecdsa-sha2-nistp521" => {
            let curve = kind.strip_prefix("ecdsa-sha2-").unwrap_or("");
            let size = match curve {
                "nistp256" => 65,
                "nistp384" => 97,
                _ => 133,
            };
            ssh_field(&mut wire) == Some(curve.as_bytes())
                && ssh_field(&mut wire).is_some_and(|v| v.len() == size && v[0] == 4)
        }
        _ => false,
    };
    valid && wire.is_empty()
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
pub enum Profile {
    PaiDsw {
        profile: pai::Profile,
        image_id: String,
        access: Access,
    },
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Access {
    Intranet,
    Internet {
        nat_gateway_id: String,
        eip_id: String,
    },
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Create {
    pub profile: String,
    pub request_id: String,
    pub ssh_public_key: String,
    pub lease_seconds: u64,
    pub gpus: u32,
    pub cpus: u32,
    pub memory_gb: u32,
}
impl Create {
    pub fn validate(&self) -> Result<()> {
        identity::validate_id(&self.profile).map_err(|_| Error("debug profile"))?;
        submission::validate_submission_token(&self.request_id)
            .map_err(|_| Error("debug request identity"))?;
        let words = self.ssh_public_key.split_whitespace().collect::<Vec<_>>();
        if self.ssh_public_key.len() > 8192
            || self.ssh_public_key.contains(['\r', '\n', '\0'])
            || words.len() < 2
            || words.len() > 3
            || !matches!(
                words[0],
                "ssh-ed25519"
                    | "ssh-rsa"
                    | "ecdsa-sha2-nistp256"
                    | "ecdsa-sha2-nistp384"
                    | "ecdsa-sha2-nistp521"
            )
            || !public_key(words[0], words[1])
            || self.lease_seconds == 0
            || self.lease_seconds > i64::MAX as u64
            || self.cpus == 0
            || self.memory_gb == 0
        {
            return Err(Error("debug machine request"));
        }
        Ok(())
    }
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub contract: String,
    pub identity: String,
    pub name: String,
    pub profile_sha256: String,
    pub request: Create,
    pub created_at: String,
    pub expires_at: i64,
}
impl Profile {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::PaiDsw {
                profile,
                image_id,
                access,
            } => {
                profile.validate()?;
                component(image_id)?;
                if profile.storage.is_none() || profile.vpc.is_none() {
                    return Err(Error("debug personal CPFS and VPC"));
                }
                if let Access::Internet {
                    nat_gateway_id,
                    eip_id,
                } = access
                {
                    if !nat_gateway_id.starts_with("ngw-") || !eip_id.starts_with("eip-") {
                        return Err(Error("existing debug NAT/EIP"));
                    }
                    component(nat_gateway_id)?;
                    component(eip_id)?;
                }
            }
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        digest(&json!(self))
    }
    /// Each provider issues exactly one contract. It is part of the identity
    /// digest, and the provider also decides the machine's name format.
    pub fn recipe(&self, request: Create, now: &str) -> Result<Recipe> {
        let contract = match self {
            Self::PaiDsw { .. } => "native-debug-machine.v4",
        };
        self.validate()?;
        request.validate()?;
        let profile_sha256 = self.digest()?;
        let time =
            OffsetDateTime::parse(now, &Rfc3339).map_err(|_| Error("debug creation time"))?;
        match self {
            Self::PaiDsw { profile, .. }
                if request.gpus <= profile.max_gpus
                    && request.cpus <= profile.max_cpus
                    && request.memory_gb <= profile.max_memory_gb
                    && request.lease_seconds <= 14400 => {}
            _ => return Err(Error("debug resource policy")),
        }
        let identity = canonical::digest(&json!([contract, profile_sha256, request]))
            .map_err(|_| Error("debug identity"))?;
        Ok(Recipe {
            contract: contract.into(),
            // PAI's InstanceName is at most 27 ASCII letters/digits/underscores.
            // Ownership additionally binds the complete digest label and quota.
            name: match self {
                Self::PaiDsw { .. } => format!("ml_exp_{}", &identity[..20]),
            },
            identity: format!("debug.{identity}"),
            profile_sha256,
            expires_at: time
                .unix_timestamp()
                .checked_add(request.lease_seconds as i64)
                .ok_or(Error("debug expiry"))?,
            created_at: now.into(),
            request,
        })
    }
    /// A saved recipe must be exactly what this profile issues for its request
    /// and creation time, so observing an uncertain create searches a name this
    /// profile owns. Any changed name, contract or identity is refused.
    pub fn check_recipe(&self, recipe: &Recipe) -> Result<()> {
        if self.recipe(recipe.request.clone(), &recipe.created_at)? != *recipe {
            return Err(Error("debug frozen recipe"));
        }
        Ok(())
    }
    fn req(
        &self,
        method: Method,
        action: &str,
        path: String,
        body: Option<Value>,
    ) -> Result<Request> {
        let endpoint = match self {
            Self::PaiDsw { profile, .. } => Endpoint::Pai {
                product: Product::PaiDsw,
                region: profile.region.clone(),
            },
        };
        request(
            endpoint,
            method,
            action,
            path,
            BTreeMap::new(),
            body,
            method == Method::Get,
        )
    }
    pub fn path(&self, _recipe: &Recipe, id: Option<&str>) -> Result<String> {
        match self {
            Self::PaiDsw { .. } => Ok(format!(
                "/api/v2/instances{}",
                id.map(|s| component(s).map(|s| format!("/{s}")))
                    .transpose()?
                    .unwrap_or_default()
            )),
        }
    }
    pub fn get(&self, recipe: &Recipe, id: Option<&str>) -> Result<Request> {
        match (self, id) {
            (Self::PaiDsw { profile, .. }, None) => {
                let mut r =
                    self.req(Method::Get, "ListInstances", self.path(recipe, None)?, None)?;
                r.query = BTreeMap::from([
                    ("WorkspaceId".into(), profile.workspace_id.clone()),
                    ("InstanceName".into(), recipe.name.clone()),
                    ("PageNumber".into(), "1".into()),
                    ("PageSize".into(), "100".into()),
                ]);
                Ok(r)
            }
            _ => self.req(
                Method::Get,
                "GetInstance",
                self.path(recipe, id.or(Some(&recipe.name)))?,
                None,
            ),
        }
    }
    pub fn create(&self, recipe: &Recipe) -> Result<Request> {
        self.check_recipe(recipe)?;
        let r = &recipe.request;
        let body = match self {
            Self::PaiDsw {
                profile,
                image_id,
                access,
            } => {
                let storage = profile.storage.as_ref().ok_or(Error("debug CPFS"))?;
                let mut forward = json!({"Enable":true,"ContainerName":"dsw-notebook","ForwardPort":"22","Name":"ssh","SSHPublicKey":r.ssh_public_key,"AccessType":[if *access==Access::Intranet{"intranet"}else{"internet"}]});
                if let Access::Internet {
                    nat_gateway_id,
                    eip_id,
                } = access
                {
                    forward["NatGatewayId"] = json!(nat_gateway_id);
                    forward["EipAllocationId"] = json!(eip_id);
                }
                let mut vpc = json!(profile.vpc);
                vpc["DefaultRoute"] = json!("eth0");
                vpc["ForwardInfos"] = json!([forward]);
                let account = storage
                    .role_arn
                    .strip_prefix("acs:ram::")
                    .and_then(|s| s.split(':').next())
                    .ok_or(Error("debug role"))?;
                let mut mounts = storage.mounts();
                // DSW also mounts WorkspaceSource at /mnt/workspace. Its explicit
                // dataset mount must have a different name; DLC keeps the
                // ordinary /mnt/workspace mount convention.
                mounts[0]["MountPath"] = json!("/mnt/ml-expd-personal");
                json!({"WorkspaceId":profile.workspace_id,"ResourceId":profile.quota_id,"InstanceName":recipe.name,"ImageId":image_id,"Accessibility":"PRIVATE",
                    "RequestedResource":{"CPU":r.cpus.to_string(),"Memory":format!("{}Gi",r.memory_gb),"GPU":r.gpus.to_string()},"OversoldType":"ForbiddenQuotaOverSold",
                    "Labels":[{"Key":"ml-expd.debug","Value":recipe.identity}],"Datasets":mounts,"WorkspaceSource":"/mnt/ml-expd-personal","UserVpc":vpc,
                    "CredentialConfig":{"Enable":true,"AliyunEnvRoleKey":"0","Configs":[{"Key":"0","Type":"Role","Roles":[{"RoleArn":storage.role_arn,"RoleType":"service","AssumeRoleFor":account}]}]}})
            }
        };
        self.req(
            Method::Post,
            "CreateInstance",
            self.path(recipe, None)?,
            Some(body),
        )
    }
    pub fn checked(&self, recipe: &Recipe, id: Option<&str>, row: &Value) -> Result<String> {
        self.check_recipe(recipe)?;
        let identifier = match self {
            Self::PaiDsw { profile, .. } => {
                if row["InstanceName"] != recipe.name
                    || pai::text(&row["WorkspaceId"])? != profile.workspace_id
                    || row["ResourceId"] != profile.quota_id
                    || !row["Labels"].as_array().is_some_and(|a| {
                        a.iter()
                            .any(|v| *v == json!({"Key":"ml-expd.debug","Value":recipe.identity}))
                    })
                {
                    return Err(Error("debug DSW ownership"));
                }
                row["InstanceId"]
                    .as_str()
                    .ok_or(Error("debug DSW identity"))?
            }
        };
        component(identifier)?;
        if id.is_some_and(|id| id != identifier) {
            return Err(Error("debug instance changed"));
        }
        Ok(identifier.into())
    }
    pub fn stop(&self, recipe: &Recipe, id: &str, delete: bool) -> Result<Request> {
        let path = self.path(recipe, Some(id))?;
        if delete {
            self.req(Method::Delete, "DeleteInstance", path, None)
        } else {
            match self {
                Self::PaiDsw { .. } => self.req(
                    Method::Put,
                    "StopInstance",
                    format!("{path}/stop"),
                    Some(json!({})),
                ),
            }
        }
    }
    pub fn terminal(&self, row: &Value) -> bool {
        match self {
            Self::PaiDsw { .. } => matches!(row["Status"].as_str(), Some("Stopped" | "Failed")),
        }
    }
    pub fn provider_state<'a>(&self, row: &'a Value) -> Result<&'a str> {
        row[if matches!(self, Self::PaiDsw { .. }) {
            "Status"
        } else {
            "state"
        }]
        .as_str()
        .filter(|s| {
            !s.is_empty()
                && s.len() <= 100
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
        })
        .ok_or(Error("debug provider state"))
    }
    pub fn lifecycle(&self, recipe: &Recipe, id: &str) -> Result<Request> {
        if !matches!(self, Self::PaiDsw { .. }) {
            return Err(Error("DSW lifecycle"));
        }
        let mut request = self.get(recipe, Some(id))?;
        request.path.push_str("/lifecycle");
        request.action = "GetLifecycle".into();
        request.query.insert("Limit".into(), "1".into());
        Ok(request)
    }
    pub fn connection(&self, row: &Value) -> Option<Value> {
        match self {
            Self::PaiDsw{access,..}=>row["UserVpc"]["ForwardInfos"].as_array()?.iter().find_map(|f|{
                let key=if *access==Access::Intranet{"Intranet"}else{"Internet"};let route=&f["ConnectInfo"][key];
                let port=route["Port"].as_u64().or_else(||route["Port"].as_str()?.parse().ok())?;
                let host=route["Endpoint"].as_str()?;
                (f["Name"]=="ssh"&&f["ForwardPort"]=="22"&&f["ConnectInfo"]["Phase"]=="Ready"&&port>0&&port<=65535&&host.len()<=253&&host.bytes().all(|b|b.is_ascii_alphanumeric()||b".-:".contains(&b))).then(||json!({"host":host,"port":port,"user":"root","access":key.to_lowercase()}))
            }),
        }
    }
    pub fn timer(&self, recipe: &Recipe, id: &str, now: i64) -> Result<Request> {
        if !matches!(self, Self::PaiDsw { .. }) {
            return Err(Error("DSW timer"));
        }
        let remaining = recipe
            .expires_at
            .checked_sub(now)
            .ok_or(Error("debug expiry"))?
            .max(1)
            .checked_mul(1000)
            .ok_or(Error("debug expiry"))?;
        self.req(
            Method::Post,
            "CreateInstanceShutdownTimer",
            format!("{}/shutdowntimer", self.path(recipe, Some(id))?),
            Some(json!({"RemainingTimeInMs":remaining})),
        )
    }
}

/// A provider's terminal failure and a normal stop must remain distinct even
/// after the instance has released its resources. An absent old observation
/// cannot establish that its termination was successful.
pub fn status(native: Option<&str>, terminal: bool) -> &'static str {
    match native {
        Some("Failed" | "FAILED") => "FAILED",
        Some("Stopped" | "SUSPENDED") => "STOPPED",
        _ if terminal => "TERMINAL",
        Some("Queuing" | "QUEUED" | "PENDING") => "QUEUED",
        Some("Running" | "RUNNING") => "RUNNING",
        _ => "STARTING",
    }
}

/// Select the most recent reason for the currently observed state, without
/// mixing an earlier lifecycle state into the current diagnostic.
pub fn lifecycle_reason<'a>(value: &'a Value, native: &str) -> Result<Option<&'a str>> {
    let sessions = value["Lifecycle"]
        .as_array()
        .filter(|s| s.len() <= 100)
        .ok_or(Error("DSW lifecycle sessions"))?;
    for session in sessions.iter().rev() {
        let rows = session
            .as_array()
            .filter(|r| r.len() <= 1000)
            .ok_or(Error("DSW lifecycle entries"))?;
        for row in rows.iter().rev() {
            if row["Status"] == native {
                return Ok(row["ReasonMessage"]
                    .as_str()
                    .filter(|s| !s.is_empty() && s.len() <= 16384));
            }
        }
    }
    Ok(None)
}
