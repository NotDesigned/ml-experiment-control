//! SenseCore ACP REST compiler and exact scoped pool/job evidence.
use super::*;
use crate::state::RunState;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    pub subscription: String,
    pub resource_group: String,
    pub region: String,
    pub zone: String,
    pub workspace: String,
    pub user_id: String,
}
impl Scope {
    pub fn validate(&self) -> Result<()> {
        for s in [
            &self.subscription,
            &self.resource_group,
            &self.zone,
            &self.workspace,
            &self.user_id,
        ] {
            component(s)?;
        }
        sense_region(&self.region)?;
        if self.zone.len() != self.region.len() + 1
            || !self.zone.starts_with(&self.region)
            || !self
                .zone
                .as_bytes()
                .last()
                .is_some_and(u8::is_ascii_lowercase)
        {
            return Err(Error("zone"));
        }
        if !regex::Regex::new(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
            .expect("UUID regex")
            .is_match(&self.user_id)
        {
            return Err(Error("owner UUID"));
        }
        Ok(())
    }
    pub fn path(&self, collection: &str, name: &str) -> Result<String> {
        self.validate()?;
        component(collection)?;
        component(name)?;
        Ok(format!(
            "/subscriptions/{}/resourceGroups/{}/zones/{}/{collection}/{name}",
            self.subscription, self.resource_group, self.zone
        ))
    }
    pub fn workspace_path(&self) -> Result<String> {
        self.path("workspaces", &self.workspace)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    Training,
    DataCopy,
    ResultCollection,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Quota {
    #[default]
    Spot,
    Reserved,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub scope: Scope,
    pub cluster: String,
    pub vpc_id: String,
    pub worker_spec: String,
    pub worker_gpus: u32,
    pub worker_cpus: u32,
    pub worker_memory_gb: u32,
    pub worker_nodes: u32,
    pub storage_volume: String,
    pub storage_subdir: String,
    pub mount_path: String,
    pub purpose: Purpose,
    #[serde(default)]
    pub quota: Quota,
}
impl Profile {
    pub fn validate(&self) -> Result<()> {
        self.scope.validate()?;
        for s in [
            &self.cluster,
            &self.vpc_id,
            &self.worker_spec,
            &self.storage_volume,
        ] {
            component(s)?;
        }
        absolute_path(&self.storage_subdir)?;
        absolute_path(&self.mount_path)?;
        if self.worker_nodes == 0
            || self.worker_nodes > 512
            || self.worker_cpus == 0
            || self.worker_memory_gb == 0
            || self.worker_gpus > 4096
            || self.worker_cpus > 65536
            || self.worker_memory_gb > 1048576
        {
            return Err(Error("worker resources"));
        }
        match self.purpose {
            Purpose::Training => {
                if self.worker_gpus == 0 || self.cluster.to_ascii_lowercase().contains("debug") {
                    return Err(Error("GPU cluster"));
                }
            }
            Purpose::DataCopy | Purpose::ResultCollection => {
                if (
                    self.worker_gpus,
                    self.worker_cpus,
                    self.worker_memory_gb,
                    self.worker_nodes,
                ) != (0, 2, 4, 1)
                    || !self.worker_spec.ends_with(".2c4g")
                    || self.quota != Quota::Reserved
                    || !self.cluster.to_ascii_lowercase().contains("debug")
                {
                    return Err(Error("CPU copy profile"));
                }
            }
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        digest(&serde_json::to_value(self).map_err(|_| Error("profile"))?)
    }
    pub fn jobs_path(&self) -> Result<String> {
        Ok(format!(
            "/compute/acp/data/v2{}/trainingJobs",
            self.scope.workspace_path()?
        ))
    }
    fn request(
        &self,
        method: Method,
        action: &str,
        path: String,
        query: BTreeMap<String, String>,
        body: Option<Value>,
        read_only: bool,
    ) -> Result<Request> {
        self.validate()?;
        request(
            Endpoint::Sensecore {
                service: "aec2".into(),
                region: Some(self.scope.region.clone()),
            },
            method,
            action,
            path,
            query,
            body,
            read_only,
        )
    }
    pub fn create(&self, job: &Job) -> Result<Request> {
        self.validate()?;
        job.validate()?;
        if u64::from(job.gpus) != u64::from(self.worker_gpus) * u64::from(self.worker_nodes)
            || u64::from(job.cpus) != u64::from(self.worker_cpus) * u64::from(self.worker_nodes)
            || u64::from(job.memory_gb)
                != u64::from(self.worker_memory_gb) * u64::from(self.worker_nodes)
        {
            return Err(Error("fixed worker shape"));
        }
        let name = job.name()?;
        let mut command = vec![
            "timeout".into(),
            "--signal=TERM".into(),
            "--kill-after=30s".into(),
            format!("{}s", job.max_time_seconds),
        ];
        command.extend(job.command.clone());
        let body = json!({"name":name,"display_name":name,"framework":"PYTORCH",
            "roles":[{"name":"Worker","resource_spec":[{"name":self.worker_spec}],"total_replicas":self.worker_nodes,"startup_script":format!("exec {}",shell_join(&command)?),"image_path":job.image}],
            "resource_pool":{"name":self.cluster},"mount":[{"type":"PV_AFS","id":self.storage_volume,"mount_path":self.mount_path,"subdir":self.storage_subdir,"zone":self.scope.zone}],
            "scheduling":{"priority":"NORMAL","quota_type":self.quota},"fault_tolerance":{"backoff_limit":0}});
        self.request(
            Method::Post,
            "CreateTrainingJob",
            self.jobs_path()?,
            BTreeMap::from([("training_job_name".into(), name)]),
            Some(body),
            false,
        )
    }
    pub fn get(&self, name: &str) -> Result<Request> {
        component(name)?;
        self.request(
            Method::Get,
            "GetTrainingJob",
            format!("{}/{name}", self.jobs_path()?),
            BTreeMap::new(),
            None,
            true,
        )
    }
    pub fn list(&self, job: &Job, token: &str) -> Result<Request> {
        component(token)?;
        let name = job.name()?;
        self.request(
            Method::Get,
            "ListTrainingJobs",
            self.jobs_path()?,
            BTreeMap::from([
                ("filter".into(), format!("name='{name}'")),
                ("name".into(), name),
                ("page_size".into(), "100".into()),
                ("page_token".into(), token.into()),
            ]),
            None,
            true,
        )
    }
    pub fn stop(&self, name: &str) -> Result<Request> {
        component(name)?;
        self.request(Method::Post,"StopTrainingJob",format!("{}:batchStop",self.jobs_path()?),BTreeMap::new(),Some(json!({"subscription_name":self.scope.subscription,"resource_group_name":self.scope.resource_group,"zone":self.scope.zone,"workspace_name":self.scope.workspace,"training_job_names":[name]})),false)
    }
    pub fn workers(&self, name: &str, token: &str) -> Result<Request> {
        component(name)?;
        component(token)?;
        self.request(
            Method::Get,
            "ListWorkers",
            format!("{}/{name}/workers", self.jobs_path()?),
            BTreeMap::from([
                ("page_size".into(), "100".into()),
                ("page_token".into(), token.into()),
            ]),
            None,
            true,
        )
    }
    pub fn telemetry_stations(&self, token: &str) -> Result<Request> {
        self.validate()?;
        component(token)?;
        request(
            Endpoint::Sensecore {
                service: "management".into(),
                region: None,
            },
            Method::Get,
            "ListTelemetryStations",
            "/rmh/v1/resources".into(),
            BTreeMap::from([
                (
                    "filter".into(),
                    "resource_type='monitor.ts.v1.telemetryStation'".into(),
                ),
                ("page_size".into(), "100".into()),
                ("page_token".into(), token.into()),
            ]),
            None,
            true,
        )
    }
    pub fn offline_logs(
        &self,
        station: &str,
        worker: &str,
        container: &str,
        start: i64,
        end: i64,
        tail: u32,
    ) -> Result<Request> {
        self.validate()?;
        component(station)?;
        component(worker)?;
        component(container)?;
        if station != format!("ts-user-{}", self.scope.user_id)
            || start < 0
            || end < start
            || !(1..=10000).contains(&tail)
        {
            return Err(Error("offline log scope"));
        }
        let body = json!({"start":start.to_string(),"end":end.to_string(),"resource_id":[self.scope.workspace_path()?],"page_size":tail.min(200),"offset":"0",
            "custom_filter":[{"key":"Attributes.k8s.pod.name","value":worker},{"key":"Attributes.k8s.container.name","value":container}]});
        request(
            Endpoint::Sensecore {
                service: "monitor".into(),
                region: None,
            },
            Method::Post,
            "SearchOfflineLogs",
            format!(
                "/monitor/ts/data/v1{}/logStream/products/product.lepton-acp-new/logs",
                self.scope.path("telemetryStations", station)?
            ),
            BTreeMap::new(),
            Some(body),
            true,
        )
    }
    pub fn pools(&self, token: &str) -> Result<Request> {
        component(token)?;
        self.request(
            Method::Get,
            "ListClusters",
            format!(
                "/compute/workspace/data/v1{}/workspaceAEC2Bindings",
                self.scope.workspace_path()?
            ),
            BTreeMap::from([
                ("page_size".into(), "100".into()),
                ("page_token".into(), token.into()),
            ]),
            None,
            true,
        )
    }
    pub fn specs(&self, cluster: &str, token: &str) -> Result<Request> {
        component(cluster)?;
        component(token)?;
        self.request(
            Method::Get,
            "ListResourceSpecs",
            format!(
                "/compute/aec2/data/v1{}/resourceSpecs",
                self.scope.path("aec2s", cluster)?
            ),
            BTreeMap::from([
                ("page_size".into(), "100".into()),
                ("page_token".into(), token.into()),
            ]),
            None,
            true,
        )
    }
    pub fn validate_job(
        &self,
        row: &Value,
        job: &Job,
        uid: Option<&str>,
    ) -> Result<JobObservation> {
        self.validate()?;
        let name = job.name()?;
        let observed_uid = row["uid"].as_str().ok_or(Error("job UID"))?;
        component(observed_uid)?;
        if row["name"] != name
            || row["ownership"]["user_id"] != self.scope.user_id
            || uid.is_some_and(|id| id != observed_uid)
        {
            return Err(Error("job ownership"));
        }
        // Optional provider resource ID, when present, must agree with scope.
        if let Some(id) = row.get("id") {
            let expected =
                self.scope.path("workspaces", &self.scope.workspace)? + "/trainingJobs/" + &name;
            if id != &json!(expected) {
                return Err(Error("job scope"));
            }
        }
        let raw = row["state"].as_str().ok_or(Error("job state"))?;
        if raw.len() > 128 || raw.chars().any(char::is_control) {
            return Err(Error("job state"));
        }
        Ok(JobObservation {
            job_id: name,
            provider_state: raw.into(),
            state: normalize(raw),
            queue: None,
        })
    }
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct OfflineLogs {
    pub text: String,
    pub available: bool,
    pub truncated: bool,
    pub last_log_at: Option<String>,
    pub source: String,
    pub worker: String,
    pub unavailable_reason: Option<String>,
}
pub fn validate_offline_logs(
    profile: &Profile,
    value: &Value,
    worker: &str,
    container: &str,
    start: i64,
    end: i64,
    tail: u32,
) -> Result<OfflineLogs> {
    profile.validate()?;
    component(worker)?;
    component(container)?;
    if start < 0 || end < start || !(1..=10000).contains(&tail) {
        return Err(Error("offline interval"));
    }
    let hits = value["hits"].as_array().ok_or(Error("offline hits"))?;
    let total = match &value["total"] {
        Value::Number(n) => n.as_u64(),
        Value::String(s) if s.bytes().all(|c| c.is_ascii_digit()) => s.parse().ok(),
        _ => None,
    }
    .ok_or(Error("offline count"))?;
    if hits.len() > tail.min(200) as usize || total < hits.len() as u64 {
        return Err(Error("offline count"));
    }
    let mut rows = Vec::new();
    for hit in hits {
        if hit["attributes"]["k8s.pod.name"] != worker
            || hit["attributes"]["k8s.container.name"] != container
            || hit["resource"]["resource_id"] != profile.scope.workspace_path()?
        {
            return Err(Error("offline worker identity"));
        }
        let timestamp = hit["log_time"].as_str().ok_or(Error("offline timestamp"))?;
        let instant =
            time::OffsetDateTime::parse(timestamp, &time::format_description::well_known::Rfc3339)
                .map_err(|_| Error("offline timestamp"))?;
        if instant.unix_timestamp() < start || instant.unix_timestamp() > end.saturating_add(1) {
            return Err(Error("offline timestamp"));
        }
        let text = hit["body"]
            .as_str()
            .filter(|s| s.len() <= 1024 * 1024)
            .ok_or(Error("offline line"))?;
        rows.push((
            instant,
            timestamp.to_owned(),
            experiment_redact::redact_line(text),
        ));
    }
    rows.sort();
    Ok(OfflineLogs {
        available: !rows.is_empty(),
        truncated: total > rows.len() as u64,
        last_log_at: rows.last().map(|r| r.1.clone()),
        text: rows
            .iter()
            .map(|r| r.2.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        source: "offline".into(),
        worker: worker.into(),
        unavailable_reason: rows.is_empty().then(|| "OFFLINE_LOGS_EMPTY".into()),
    })
}
pub fn normalize(value: &str) -> RunState {
    match value.to_ascii_uppercase().as_str() {
        "WAITING" | "INIT" | "QUEUEING" | "PENDING" | "CREATING" => RunState::Queued,
        "STARTING" | "RECOVERING" => RunState::Starting,
        "RUNNING" | "RESTARTING" => RunState::Running,
        "SUCCEEDED" | "COMPLETED" => RunState::Succeeded,
        "FAILED" | "ERROR" => RunState::Failed,
        "SUSPENDED" | "STOPPED" | "DELETING" | "DELETED" | "CANCELLED" | "CANCELED" => {
            RunState::Cancelled
        }
        _ => RunState::Unknown,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cluster {
    pub name: String,
    pub zone: String,
    pub vpc_id: String,
    pub spot_devices_nano: u128,
    pub compatible: bool,
}
// Exact fixed decimal arithmetic; no float rounding in quota ordering.
fn decimal_nano(value: &Value) -> Result<u128> {
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => return Err(Error("spot quota")),
    };
    let (whole, fraction) = raw.split_once('.').unwrap_or((&raw, ""));
    if whole.is_empty()
        || !whole.bytes().all(|b| b.is_ascii_digit())
        || fraction.len() > 9
        || !fraction.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(Error("spot quota"));
    }
    let whole = whole.parse::<u128>().map_err(|_| Error("spot quota"))?;
    if whole > 2147483647 {
        return Err(Error("spot quota"));
    }
    let fractional = if fraction.is_empty() {
        0
    } else {
        fraction.parse::<u128>().map_err(|_| Error("spot quota"))?
            * 10u128.pow((9 - fraction.len()) as u32)
    };
    Ok(whole * 1_000_000_000 + fractional)
}
pub fn cluster(profile: &Profile, row: &Value, specs: &[Value]) -> Result<Cluster> {
    profile.validate()?;
    let name = row["name"].as_str().ok_or(Error("cluster name"))?;
    component(name)?;
    let expected = profile.scope.path("aec2s", name)?;
    if row["state"] != "ACTIVE" || row["id"] != expected {
        return Err(Error("cluster scope"));
    }
    let shares = row["spot_status"]
        .as_array()
        .filter(|s| !s.is_empty() && s.len() <= 1024)
        .ok_or(Error("spot quota"))?;
    let mut quota = 0u128;
    for share in shares {
        quota = quota
            .checked_add(decimal_nano(&share["spot_quota"]["device"])?)
            .ok_or(Error("spot quota"))?;
    }
    let mut names = BTreeSet::new();
    let mut compatible = false;
    for spec in specs {
        let spec_name = spec["name"].as_str().ok_or(Error("resource spec"))?;
        component(spec_name)?;
        if !names.insert(spec_name) {
            return Err(Error("duplicate spec"));
        }
        let cpus = spec["cpu"]["vcpu_allocatable"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or(Error("resource CPU"))?;
        let memory = spec["memory"]["allocatable"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or(Error("resource memory"))?;
        let gpus = spec["device"]["number"]
            .as_u64()
            .ok_or(Error("resource GPU"))?;
        let zones = spec["zones"].as_array().ok_or(Error("resource zones"))?;
        if !zones.contains(&json!(profile.scope.zone)) {
            return Err(Error("resource zone"));
        }
        if spec_name == profile.worker_spec
            && gpus == u64::from(profile.worker_gpus)
            && cpus >= u64::from(profile.worker_cpus)
            && memory >= u64::from(profile.worker_memory_gb) * 1024 * 1024 * 1024
        {
            compatible = true;
        }
    }
    let vpc = row["vpc_id"].as_str().ok_or(Error("cluster VPC"))?;
    component(vpc)?;
    Ok(Cluster {
        name: name.into(),
        zone: profile.scope.zone.clone(),
        vpc_id: vpc.into(),
        spot_devices_nano: quota,
        compatible,
    })
}
/// Quota is an allowance, not the number of currently idle GPUs or an ETA.
pub fn highest_spot(
    profile: &Profile,
    clusters: &[Cluster],
    allowed: Option<&BTreeSet<String>>,
) -> Result<Cluster> {
    profile.validate()?;
    if profile.purpose != Purpose::Training
        || profile.quota != Quota::Spot
        || clusters.len() > 10000
        || allowed.is_some_and(BTreeSet::is_empty)
    {
        return Err(Error("selection policy"));
    }
    let mut names = BTreeSet::new();
    for cluster in clusters {
        component(&cluster.name)?;
        if !names.insert(&cluster.name) {
            return Err(Error("duplicate cluster"));
        }
    }
    clusters
        .iter()
        .filter(|c| {
            c.compatible
                && !c.name.to_ascii_lowercase().contains("debug")
                && c.zone == profile.scope.zone
                && c.vpc_id == profile.vpc_id
                && allowed.is_none_or(|a| a.contains(&c.name))
        })
        .min_by(|a, b| {
            b.spot_devices_nano
                .cmp(&a.spot_devices_nano)
                .then(a.name.cmp(&b.name))
        })
        .cloned()
        .ok_or(Error("no compatible spot cluster"))
}
