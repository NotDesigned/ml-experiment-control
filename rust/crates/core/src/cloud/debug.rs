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
    SensecoreCci {
        profile: sensecore::Profile,
        image: String,
        network: Network,
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
pub struct Network {
    pub eip_name: String,
    pub eip_id: String,
    pub nat_gateway_id: String,
    pub external_ip: String,
    pub external_port: u16,
    pub tenant_id: String,
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
            || !(60..=14400).contains(&self.lease_seconds)
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
            Self::SensecoreCci {
                profile,
                image,
                network,
            } => {
                profile.validate()?;
                // A configured CPU debug pool and the existing personal NAS.
                if profile.purpose == sensecore::Purpose::Training
                    || !profile.storage_subdir.starts_with('/')
                    || profile.storage_subdir[1..].contains('/')
                    || profile.mount_path != "/data"
                {
                    return Err(Error("debug NAS scope"));
                }
                Job {
                    project: "check".into(),
                    run_id: "check".into(),
                    attempt_id: "attempt-001".into(),
                    submission_token: "0".repeat(32),
                    source_id: format!("source.{}", "0".repeat(64)),
                    image: image.clone(),
                    command: vec!["true".into()],
                    gpus: 0,
                    cpus: 1,
                    memory_gb: 1,
                    max_time_seconds: 60,
                    priority: 5,
                }
                .validate()?;
                for s in [
                    &network.eip_name,
                    &network.nat_gateway_id,
                    &network.tenant_id,
                ] {
                    component(s)?;
                }
                let expected = profile.scope.path("eips", &network.eip_name)?;
                if network.eip_id != expected
                    || network.external_port < 1024
                    || network.external_ip.parse::<std::net::Ipv4Addr>().is_err()
                {
                    return Err(Error("existing debug network"));
                }
            }
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        digest(&json!(self))
    }
    pub fn recipe(&self, request: Create, now: &str) -> Result<Recipe> {
        let contract = if matches!(self, Self::PaiDsw { .. }) {
            "native-debug-machine.v2"
        } else {
            "native-debug-machine.v1"
        };
        self.recipe_version(request, now, contract)
    }
    /// Old issued recipes retain their original name and identity. In
    /// particular, observing an uncertain v1 create must search its old name.
    pub fn check_recipe(&self, recipe: &Recipe) -> Result<()> {
        if self.recipe_version(recipe.request.clone(), &recipe.created_at, &recipe.contract)?
            != *recipe
        {
            return Err(Error("debug frozen recipe"));
        }
        Ok(())
    }
    fn recipe_version(&self, request: Create, now: &str, contract: &str) -> Result<Recipe> {
        if contract != "native-debug-machine.v1"
            && !(contract == "native-debug-machine.v2" && matches!(self, Self::PaiDsw { .. }))
        {
            return Err(Error("debug recipe contract"));
        }
        self.validate()?;
        request.validate()?;
        let profile_sha256 = self.digest()?;
        let time =
            OffsetDateTime::parse(now, &Rfc3339).map_err(|_| Error("debug creation time"))?;
        match self {
            Self::PaiDsw { profile, .. }
                if request.gpus <= profile.max_gpus
                    && request.cpus <= profile.max_cpus
                    && request.memory_gb <= profile.max_memory_gb => {}
            Self::SensecoreCci { profile, .. }
                if (request.gpus, request.cpus, request.memory_gb)
                    == (
                        profile.worker_gpus,
                        profile.worker_cpus,
                        profile.worker_memory_gb,
                    )
                    && request.lease_seconds <= 7200 => {}
            _ => return Err(Error("debug resource policy")),
        }
        let identity = canonical::digest(&json!([contract, profile_sha256, request]))
            .map_err(|_| Error("debug identity"))?;
        Ok(Recipe {
            contract: contract.into(),
            // PAI's InstanceName is at most 27 ASCII letters/digits/underscores.
            // Ownership additionally binds the complete digest label and quota.
            name: if contract == "native-debug-machine.v2" {
                format!("ml_exp_{}", &identity[..20])
            } else {
                format!("ml-expd-debug-{}", &identity[..32])
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
            Self::SensecoreCci { profile, .. } => Endpoint::Sensecore {
                service: "cci".into(),
                region: Some(profile.scope.region.clone()),
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
    pub fn path(&self, recipe: &Recipe, id: Option<&str>) -> Result<String> {
        match self {
            Self::PaiDsw { .. } => Ok(format!(
                "/api/v2/instances{}",
                id.map(|s| component(s).map(|s| format!("/{s}")))
                    .transpose()?
                    .unwrap_or_default()
            )),
            Self::SensecoreCci { profile, .. } => Ok(format!(
                "/compute/cci/data/v2{}/apps{}",
                profile.scope.workspace_path()?,
                if id.is_some() {
                    format!("/{}", recipe.name)
                } else {
                    String::new()
                }
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
                json!({"WorkspaceId":profile.workspace_id,"ResourceId":profile.quota_id,"InstanceName":recipe.name,"ImageId":image_id,"Accessibility":"PRIVATE",
                    "RequestedResource":{"CPU":r.cpus.to_string(),"Memory":format!("{}Gi",r.memory_gb),"GPU":r.gpus.to_string()},"OversoldType":"ForbiddenQuotaOverSold",
                    "Labels":[{"Key":"ml-expd.debug","Value":recipe.identity}],"Datasets":storage.mounts(),"WorkspaceSource":"/mnt/ml-expd-personal","UserVpc":vpc,
                    "CredentialConfig":{"Enable":true,"AliyunEnvRoleKey":"0","Configs":[{"Key":"0","Type":"Role","Roles":[{"RoleArn":storage.role_arn,"RoleType":"service","AssumeRoleFor":account}]}]}})
            }
            Self::SensecoreCci { profile, image, .. } => {
                json!({"name":recipe.name,"display_name":recipe.identity,
                "resource_pool":{"name":profile.cluster,"available_zone":profile.scope.zone,"vpc_id":profile.vpc_id},"replicas":1,
                "template":{"resource_spec":{"name":profile.worker_spec},"containers":[{"name":"main","image_path":image,"resource_request":{"cpu":r.cpus.to_string(),"memory":format!("{}GiB",r.memory_gb)},
                    "command":["/bin/sh","-c",startup(recipe)?],"env":[],"volume_mounts":[{"type":"PV_AFS","mount_path":profile.mount_path,"id":profile.storage_volume,"subdir":profile.storage_subdir}]}]},
                "scheduling":{"priority":"NORMAL","quota_type":"RESERVED"},"termination_grace_period_seconds":15})
            }
        };
        let mut request = self.req(
            Method::Post,
            "CreateInstance",
            self.path(recipe, None)?,
            Some(body),
        )?;
        if matches!(self, Self::SensecoreCci { .. }) {
            request.query = BTreeMap::from([
                ("app_name".into(), recipe.name.clone()),
                ("request_id".into(), recipe.request.request_id.clone()),
            ]);
        }
        Ok(request)
    }
    pub fn checked(&self, recipe: &Recipe, id: Option<&str>, row: &Value) -> Result<String> {
        self.check_recipe(recipe)?;
        let identifier = match self {
            Self::PaiDsw { profile, .. } => {
                if row["InstanceName"] != recipe.name
                    || row["WorkspaceId"].as_str() != Some(&profile.workspace_id)
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
            Self::SensecoreCci { profile, .. } => {
                let expected = self.create(recipe)?.body.ok_or(Error("debug document"))?;
                if row["name"] != recipe.name
                    || row["ownership"]["user_id"] != profile.scope.user_id
                    || !["display_name", "resource_pool", "template"]
                        .iter()
                        .all(|k| subset(&expected[k], &row[k]))
                {
                    return Err(Error("debug CCI ownership"));
                }
                row["uid"].as_str().ok_or(Error("debug CCI identity"))?
            }
        };
        component(identifier)?;
        if id.is_some_and(|id| id != identifier) {
            return Err(Error("debug instance changed"));
        }
        Ok(identifier.into())
    }
    pub fn stop(&self, recipe: &Recipe, id: &str, delete: bool) -> Result<Request> {
        if delete && matches!(self, Self::SensecoreCci { .. }) {
            return Err(Error("CCI stop only"));
        }
        let path = self.path(recipe, Some(id))?;
        if delete {
            self.req(Method::Delete, "DeleteInstance", path, None)
        } else {
            match self {
                Self::PaiDsw { .. } => {
                    self.req(Method::Put, "StopInstance", format!("{path}/stop"), None)
                }
                Self::SensecoreCci { .. } => self.req(
                    Method::Post,
                    "StopInstance",
                    format!("{path}:stop"),
                    Some(json!({})),
                ),
            }
        }
    }
    pub fn terminal(&self, row: &Value) -> bool {
        match self {
            Self::PaiDsw { .. } => matches!(row["Status"].as_str(), Some("Stopped" | "Failed")),
            Self::SensecoreCci { .. } => {
                matches!(row["state"].as_str(), Some("SUSPENDED" | "FAILED"))
            }
        }
    }
    pub fn connection(&self, row: &Value) -> Option<Value> {
        match self {
            Self::PaiDsw{access,..}=>row["UserVpc"]["ForwardInfos"].as_array()?.iter().find_map(|f|{
                let key=if *access==Access::Intranet{"Intranet"}else{"Internet"};let route=&f["ConnectInfo"][key];
                let port=route["Port"].as_u64().or_else(||route["Port"].as_str()?.parse().ok())?;
                let host=route["Endpoint"].as_str()?;
                (f["Name"]=="ssh"&&f["ForwardPort"]=="22"&&f["ConnectInfo"]["Phase"]=="Ready"&&port>0&&port<=65535&&host.len()<=253&&host.bytes().all(|b|b.is_ascii_alphanumeric()||b".-:".contains(&b))).then(||json!({"host":host,"port":port,"user":"root","access":key.to_lowercase()}))
            }),Self::SensecoreCci{..}=>None,
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
    pub fn service(&self, recipe: &Recipe, create: bool) -> Result<Request> {
        let Self::SensecoreCci { profile, .. } = self else {
            return Err(Error("CCI service"));
        };
        let base = format!(
            "/compute/service/data/v2{}/services",
            profile.scope.workspace_path()?
        );
        let mut r=self.req(if create{Method::Post}else{Method::Get},if create{"CreateService"}else{"GetService"},if create{base}else{format!("{base}/{}",recipe.name)},create.then(||json!({"name":recipe.name,"selector":{"app":recipe.name},"ports":[{"port":22,"target_port":22}]})))?;
        if create {
            r.query = BTreeMap::from([
                ("name".into(), recipe.name.clone()),
                ("request_id".into(), recipe.request.request_id.clone()),
            ]);
        }
        Ok(r)
    }
    pub fn network(&self, recipe: &Recipe, operation: &str, id: &str) -> Result<Request> {
        let Self::SensecoreCci {
            profile, network, ..
        } = self
        else {
            return Err(Error("CCI network"));
        };
        let hex = recipe
            .identity
            .strip_prefix("debug.")
            .ok_or(Error("debug identity"))?;
        let uid = format!(
            "{}-{}-{}-{}-{}",
            &hex[..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..32]
        );
        let path = format!(
            "/network/eip/data/v1{}/dnatRules/{}",
            network.eip_id, recipe.name
        );
        let mut body = json!({"name":recipe.name,"display_name":recipe.name,"id":format!("{}/dnatRules/{}",network.eip_id,recipe.name),"uid":uid,"resource_type":"network.eip.v1.dnatRule","zone":profile.scope.zone,
            "creator_id":profile.scope.user_id,"owner_id":profile.scope.user_id,"tenant_id":network.tenant_id,"properties":{"external_ip":network.external_ip,"external_port":network.external_port.to_string(),"internal_port":"22","protocol":"tcp","eip_id":network.eip_id,"nat_gateway_id":network.nat_gateway_id,"internal_ip":"","internal_instance_name":"","internal_instance_type":"UNSPECIFIED"}});
        if operation == "bind" {
            body["properties"]["internal_instance_name"] = json!(id);
            body["properties"]["internal_instance_type"] = json!("CCI_DEPLOYMENT_SERVICE");
        }
        let (method, suffix, payload) = match operation {
            "get" => (Method::Get, "", None),
            "create" => (Method::Post, "", Some(body)),
            "bind" => (Method::Post, "/bind", Some(body)),
            "unbind" => (Method::Post, "/unbind", Some(json!({}))),
            "delete" => (Method::Delete, "", None),
            _ => return Err(Error("debug network action")),
        };
        request(
            Endpoint::Sensecore {
                service: "network".into(),
                region: Some(profile.scope.region.clone()),
            },
            method,
            "DebugNetwork",
            format!("{path}{suffix}"),
            BTreeMap::new(),
            payload,
            method == Method::Get,
        )
    }
    pub fn check_network(&self, recipe: &Recipe, id: &str, row: &Value) -> Result<bool> {
        let expected = self
            .network(recipe, "create", id)?
            .body
            .ok_or(Error("debug network identity"))?;
        if !["name", "uid", "owner_id", "creator_id", "tenant_id"]
            .iter()
            .all(|k| expected[k] == row[k])
            || ![
                "external_ip",
                "external_port",
                "protocol",
                "eip_id",
                "nat_gateway_id",
            ]
            .iter()
            .all(|k| expected["properties"][k] == row["properties"][k])
        {
            return Err(Error("debug network ownership"));
        }
        let target = row["properties"]["internal_instance_name"]
            .as_str()
            .unwrap_or("");
        if !target.is_empty() && target != id {
            return Err(Error("debug foreign network target"));
        }
        Ok(row["state"] == "ACTIVE"
            && target == id
            && row["properties"]["internal_port"] == "22"
            && row["properties"]["internal_instance_type"] == "CCI_DEPLOYMENT_SERVICE")
    }
}
pub fn subset(expected: &Value, actual: &Value) -> bool {
    match expected {
        Value::Object(m) => actual.as_object().is_some_and(|a| {
            m.iter()
                .all(|(k, v)| a.get(k).is_some_and(|a| subset(v, a)))
        }),
        Value::Array(m) => actual
            .as_array()
            .is_some_and(|a| m.len() == a.len() && m.iter().zip(a).all(|(e, a)| subset(e, a))),
        _ => expected == actual,
    }
}
fn startup(recipe: &Recipe) -> Result<String> {
    let key = shell_join(&[recipe.request.ssh_public_key.clone()])?;
    Ok(format!(
        "set -eu\numask 077\ntest -x /usr/sbin/sshd\nmkdir -p /run/sshd /run/ml-expd-ssh\nprintf '%s\\n' {key} > /run/ml-expd-ssh/authorized_keys\nssh-keygen -q -t ed25519 -N '' -f /run/ml-expd-ssh/host_ed25519\ncat > /run/ml-expd-ssh/sshd_config <<'EOF'\nPort 22\nHostKey /run/ml-expd-ssh/host_ed25519\nAuthorizedKeysFile /run/ml-expd-ssh/authorized_keys\nPermitRootLogin prohibit-password\nPubkeyAuthentication yes\nAuthenticationMethods publickey\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nAllowUsers root\nSubsystem sftp internal-sftp\nEOF\nremaining=$(({} - $(date +%s)))\ntest \"$remaining\" -gt 0\nexec timeout \"$remaining\" /usr/sbin/sshd -D -e -f /run/ml-expd-ssh/sshd_config\n",
        recipe.expires_at
    ))
}
