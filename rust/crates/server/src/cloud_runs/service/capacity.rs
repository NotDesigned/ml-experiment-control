use super::*;
use ml_exp_core::capacity::{self, Observation};
impl Service {
    pub fn catalogue(&self) -> cloud::Result<Vec<Value>> {
        self.backends.iter().map(|(id,backend)|{
            let (kind,limits,allocation)=match &backend.adapter.profile {
                cloud::adapter::Profile::Pai(p)=>("pai-dlc",json!({"gpus":p.max_gpus,"cpus":p.max_cpus,"memory_gb":p.max_memory_gb,"walltime_seconds":86400}),json!(["guaranteed","allow_idle","idle_only"])),
                cloud::adapter::Profile::Sensecore(p)=>("sensecore",json!({"gpus":p.worker_gpus,"cpus":p.worker_cpus,"memory_gb":p.worker_memory_gb,"fixed_resources":true,"cluster":p.cluster}),json!(["guaranteed"])),
            };
            Ok(json!({"executor_id":id,"kind":kind,"enabled":self.enabled,"run_creation":self.creation,"limits":limits,"allocation_policies":allocation,"default_priority":5,
                "capabilities":{"oci_images":true,"persistent_checkpoints":true,"dataset_upload":self.has_datasets(),"data_image_copy":backend.data_copier.is_some(),"data_preparation":if kind=="pai-dlc"{json!("inside_job")}else if backend.data_copier.is_some(){json!("before_job")}else{Value::Null},"metrics":self.has_exchange(),"result_objects":self.has_artifacts(),"cpu_collection":backend.collector.is_some(),"capacity_query":true},"live_availability_checked":false}))
        }).collect()
    }
    pub fn capacity(&self, executor: &str, refresh: bool, now: &str) -> cloud::Result<Value> {
        let backend = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        let name = format!("capacity-{executor}.json");
        let _effect = if refresh {
            let file = self
                .slots
                .open_lock(OsStr::new(&format!("capacity-{executor}.lock")))
                .map_err(io)?;
            fs2::FileExt::try_lock_exclusive(&file).map_err(io)?;
            Some(file)
        } else {
            None
        };
        let old = if self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            serde_json::from_value::<Observation>(read(&self.slots, &name)?)
                .map_err(|_| cloud::Error::evidence())?
        } else {
            capacity::unknown(executor).map_err(|_| cloud::Error::configuration())?
        };
        if old.executor_id != executor {
            return Err(cloud::Error::evidence());
        }
        if !refresh {
            return Ok(json!(
                capacity::view(old, now).map_err(|_| cloud::Error::evidence())?
            ));
        }
        let data=match &backend.adapter.profile {
            cloud::adapter::Profile::Pai(profile)=>backend.adapter.pai_quota(&AtomicBool::new(false)).and_then(|row|ml_exp_core::cloud::pai::quota_view(profile,&row).map_err(|_|cloud::Error::evidence())),
            cloud::adapter::Profile::Sensecore(_)=>backend.adapter.sensecore_clusters(&AtomicBool::new(false)).map(|clusters|json!({"source":"sensecore-workspace","scope":"zone","clusters":clusters,"quota_is_free_capacity":false})),
        };
        let next = capacity::refresh(old, data.map_err(|_| "unavailable"), now)
            .map_err(|_| cloud::Error::evidence())?;
        self.slots
            .replace_regular(OsStr::new(&name), &encode(&next)?)
            .map_err(io)?;
        Ok(json!(next))
    }
}
