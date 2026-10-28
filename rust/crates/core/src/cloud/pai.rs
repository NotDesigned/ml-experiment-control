//! PAI DLC training compiler. DSW remains a separate debug/build service.
use super::*;
use crate::state::RunState;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Allocation {
    #[default]
    Guaranteed,
    AllowIdle,
    IdleOnly,
}
impl Allocation {
    pub fn oversold(self) -> &'static str {
        match self {
            Self::Guaranteed => "ForbiddenQuotaOverSold",
            Self::AllowIdle => "AcceptQuotaOverSold",
            Self::IdleOnly => "ForceQuotaOverSold",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vpc {
    #[serde(rename = "VpcId")]
    pub vpc_id: String,
    #[serde(rename = "VSwitchId")]
    pub switch_id: String,
    #[serde(rename = "SecurityGroupId")]
    pub security_group_id: String,
}
impl Vpc {
    pub fn validate(&self) -> Result<()> {
        for (s, p) in [
            (&self.vpc_id, "vpc-"),
            (&self.switch_id, "vsw-"),
            (&self.security_group_id, "sg-"),
        ] {
            if !s.strip_prefix(p).is_some_and(|v| {
                !v.is_empty()
                    && v.bytes()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
            }) {
                return Err(Error("VPC"));
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Storage {
    pub cpfs_uri: String,
    pub workspace_user: String,
    pub role_arn: String,
}
impl Storage {
    pub fn validate(&self, region: &str, vpc: Option<&Vpc>) -> Result<()> {
        component(&self.workspace_user)?;
        if self.workspace_user.len() > 64
            || !self.cpfs_uri.starts_with("bmcpfs://cpfs-")
            || !self
                .cpfs_uri
                .ends_with(&format!(".{region}.cpfs.aliyuncs.com/"))
            || !self.cpfs_uri[9..self.cpfs_uri.len() - 1]
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || b".-".contains(&c))
            || self.cpfs_uri.contains("-vpc-") && vpc.is_none()
        {
            return Err(Error("approved CPFS"));
        }
        let account = self
            .role_arn
            .strip_prefix("acs:ram::")
            .and_then(|s| s.strip_suffix(&format!(":role/pai-data-{}", self.workspace_user)))
            .ok_or(Error("personal data role"))?;
        if account.is_empty() || account.len() > 32 || !account.bytes().all(|c| c.is_ascii_digit())
        {
            return Err(Error("role account"));
        }
        Ok(())
    }
    pub fn mounts(&self) -> Vec<Value> {
        [
        (format!("users/{}/",self.workspace_user),"/mnt/workspace","RW"),
        ("public/models/".into(),"/mnt/models","RO"),("public/datasets/".into(),"/mnt/datasets","RO"),
    ].into_iter().map(|(path,mount,access)|json!({"Uri":format!("{}{path}",self.cpfs_uri),"MountPath":mount,"MountAccess":access})).collect()
    }
    pub fn credential_config(&self) -> Result<Value> {
        let account = self
            .role_arn
            .strip_prefix("acs:ram::")
            .and_then(|s| s.split(':').next())
            .ok_or(Error("role account"))?;
        Ok(
            json!({"EnableCredentialInject":true,"AliyunEnvRoleKey":"0","CredentialConfigItems":[{"Key":"0","Type":"Role","Roles":[{"RoleArn":self.role_arn,"RoleType":"service","AssumeRoleFor":account}]}]}),
        )
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub region: String,
    pub workspace_id: String,
    pub quota_id: String,
    pub max_gpus: u32,
    pub max_cpus: u32,
    pub max_memory_gb: u32,
    #[serde(default)]
    pub gpu_type: Option<String>,
    #[serde(default)]
    pub storage: Option<Storage>,
    #[serde(default)]
    pub vpc: Option<Vpc>,
}
impl Profile {
    pub fn validate(&self) -> Result<()> {
        region_id(&self.region)?;
        component(&self.workspace_id)?;
        component(&self.quota_id)?;
        if self.max_gpus > 4096
            || self.max_cpus == 0
            || self.max_cpus > 65536
            || self.max_memory_gb == 0
            || self.max_memory_gb > 1048576
        {
            return Err(Error("capacity"));
        }
        if let Some(gpu) = &self.gpu_type {
            component(gpu)?;
        }
        if let Some(vpc) = &self.vpc {
            vpc.validate()?;
        }
        if let Some(storage) = &self.storage {
            storage.validate(&self.region, self.vpc.as_ref())?;
        }
        Ok(())
    }
    pub fn scope(&self) -> Result<String> {
        self.validate()?;
        digest(&serde_json::to_value(self).map_err(|_| Error("profile"))?)
    }
    fn request(
        &self,
        product: Product,
        method: Method,
        action: &str,
        path: String,
        query: BTreeMap<String, String>,
        body: Option<Value>,
    ) -> Result<Request> {
        self.validate()?;
        request(
            Endpoint::Pai {
                product,
                region: self.region.clone(),
            },
            method,
            action,
            path,
            query,
            body,
            method == Method::Get,
        )
    }
    pub fn create(&self, job: &Job, allocation: Allocation) -> Result<Request> {
        self.validate()?;
        job.validate()?;
        if job.max_time_seconds > 86400
            || job.gpus > self.max_gpus
            || job.cpus > self.max_cpus
            || job.memory_gb > self.max_memory_gb
        {
            return Err(Error("resource limit"));
        }
        let mut document = json!({"WorkspaceId":self.workspace_id,"ResourceId":self.quota_id,"DisplayName":job.name()?,"JobType":"PyTorchJob","Accessibility":"PRIVATE",
            "UserCommand":shell_join(&job.command)?,"JobMaxRunningTimeMinutes":job.max_time_seconds.div_ceil(60),"Priority":job.priority,
            "DataSources":self.storage.as_ref().map(Storage::mounts).unwrap_or_default(),"UserVpc":self.vpc.as_ref().map(|v|serde_json::to_value(v).expect("VPC serialization")).unwrap_or(json!({})),
            "Settings":{"Tags":job.tags(&self.scope()?)?,"JobReservedMinutes":0,"EnableErrorMonitoringInAIMaster":false,"EnableRDMA":false,
                "EnableTideResource":allocation!=Allocation::Guaranteed,"OversoldType":allocation.oversold()},
            "JobSpecs":[{"Type":"Worker","PodCount":1,"UseSpotInstance":false,"RestartPolicy":"Never","Image":job.image,
                "OversoldType":allocation.oversold(),"ResourceConfig":{"CPU":job.cpus.to_string(),"Memory":format!("{}Gi",job.memory_gb),"GPU":job.gpus.to_string()}}]});
        if let Some(gpu) = &self.gpu_type {
            document["JobSpecs"][0]["ResourceConfig"]["GPUType"] = json!(gpu);
        }
        if let Some(storage) = &self.storage {
            document["CredentialConfig"] = storage.credential_config()?;
        }
        self.request(
            Product::PaiDlc,
            Method::Post,
            "CreateJob",
            "/api/v1/jobs".into(),
            BTreeMap::new(),
            Some(document),
        )
    }
    pub fn get(&self, job_id: &str) -> Result<Request> {
        component(job_id)?;
        self.request(
            Product::PaiDlc,
            Method::Get,
            "GetJob",
            format!("/api/v1/jobs/{job_id}"),
            BTreeMap::new(),
            None,
        )
    }
    pub fn list(&self, job: &Job, page: u32) -> Result<Request> {
        job.validate()?;
        if !(1..=100).contains(&page) {
            return Err(Error("page"));
        }
        self.request(
            Product::PaiDlc,
            Method::Get,
            "ListJobs",
            "/api/v1/jobs".into(),
            BTreeMap::from([
                ("WorkspaceId".into(), self.workspace_id.clone()),
                ("ResourceId".into(), self.quota_id.clone()),
                ("ShowOwn".into(), "true".into()),
                ("PageNumber".into(), page.to_string()),
                ("PageSize".into(), "100".into()),
                ("StartTime".into(), "2020-01-01T00:00:00Z".into()),
                ("DisplayName".into(), job.name()?),
                (
                    "Tags".into(),
                    serde_json::to_string(&job.tags(&self.scope()?)?).map_err(|_| Error("tags"))?,
                ),
            ]),
            None,
        )
    }
    pub fn stop(&self, job_id: &str) -> Result<Request> {
        component(job_id)?;
        self.request(
            Product::PaiDlc,
            Method::Post,
            "StopJob",
            format!("/api/v1/jobs/{job_id}/stop"),
            BTreeMap::new(),
            None,
        )
    }
    pub fn quota(&self) -> Result<Request> {
        self.request(
            Product::PaiStudio,
            Method::Get,
            "GetQuota",
            format!("/api/v1/quotas/{}", self.quota_id),
            BTreeMap::from([("Verbose".into(), "true".into())]),
            None,
        )
    }
    pub fn queue(&self, job_id: &str) -> Result<Request> {
        component(job_id)?;
        self.request(
            Product::PaiStudio,
            Method::Get,
            "ListQuotaWorkloads",
            format!("/api/v1/quotas/{}/workloads", self.quota_id),
            BTreeMap::from([
                ("WorkspaceIds".into(), self.workspace_id.clone()),
                ("WorkloadType".into(), "dlc".into()),
                ("WorkloadIds".into(), job_id.into()),
                ("PageNumber".into(), "1".into()),
                ("PageSize".into(), "2".into()),
            ]),
            None,
        )
    }
    pub fn pod_logs(&self, job_id: &str, pod: &str, tail: u32) -> Result<Request> {
        component(job_id)?;
        component(pod)?;
        if !(1..=10000).contains(&tail) {
            return Err(Error("log tail"));
        }
        self.request(
            Product::PaiDlc,
            Method::Get,
            "GetPodLogs",
            format!("/api/v1/jobs/{job_id}/pods/{pod}/logs"),
            BTreeMap::from([("MaxLines".into(), tail.to_string())]),
            None,
        )
    }
    pub fn validate_job(
        &self,
        row: &Value,
        job: &Job,
        expected_id: Option<&str>,
    ) -> Result<JobObservation> {
        self.validate()?;
        job.validate()?;
        let id = row["JobId"].as_str().ok_or(Error("job ID"))?;
        component(id)?;
        if expected_id.is_some_and(|expected| expected != id)
            || text(&row["WorkspaceId"])? != self.workspace_id
            || row["ResourceId"] != self.quota_id
            || row["DisplayName"] != job.name()?
            || row["Settings"]["Tags"] != job.tags(&self.scope()?)?
        {
            return Err(Error("job ownership"));
        }
        let raw = row["Status"].as_str().ok_or(Error("job state"))?;
        if raw.len() > 128 || raw.chars().any(char::is_control) {
            return Err(Error("job state"));
        }
        Ok(JobObservation {
            job_id: id.into(),
            provider_state: raw.into(),
            state: normalize(raw),
            queue: None,
        })
    }
    pub fn validate_queue(&self, row: &Value, job_id: &str) -> Result<Option<QueueDiagnostic>> {
        self.validate()?;
        component(job_id)?;
        let rows = row["Workloads"].as_array().ok_or(Error("queue rows"))?;
        if row["TotalCount"].as_u64() != Some(rows.len() as u64) || rows.len() > 1 {
            return Err(Error("ambiguous queue"));
        }
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        if row["WorkloadId"] != job_id
            || text(&row["WorkspaceId"])? != self.workspace_id
            || row["QuotaId"] != self.quota_id
            || row["WorkloadType"] != "dlc"
        {
            return Err(Error("queue scope"));
        }
        QueueDiagnostic::pai(row).map(Some)
    }
}
fn text(v: &Value) -> Result<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) if n.is_u64() => Ok(n.to_string()),
        _ => Err(Error("resource identity")),
    }
}
pub fn normalize(value: &str) -> RunState {
    match value {
        "Creating" | "Queuing" | "Bidding" => RunState::Queued,
        "Dequeued" | "WaitingConfig" | "EnvPreparing" | "SanityChecking" => RunState::Starting,
        "Running" | "Restarting" => RunState::Running,
        "SucceededReserving" | "Succeeded" => RunState::Succeeded,
        "FailedReserving" | "Failed" => RunState::Failed,
        "Stopped" => RunState::Cancelled,
        _ => RunState::Unknown,
    }
}
