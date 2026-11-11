use super::*;
use ml_exp_core::capacity::{self, Observation};
impl Service {
    /// What a client needs to know a Runtime's identity before anything is uploaded.
    /// Absent where code cannot be delivered, so that client keeps code in the image.
    fn code_delivery(&self, executor: &str) -> Option<Value> {
        if !self.has_exchange() {
            return None;
        }
        let builds = self.builds.as_ref()?;
        // An earlier client reads only the worker. It asks after an ID that is not there
        // and says a build is to come, which the server then finds it does not need.
        Some(match self.bootstrap_identity(executor) {
            Some(identity) => {
                json!({"worker_sha256":builds.worker_sha256,"runtime_identity":identity})
            }
            None => json!({"worker_sha256":builds.worker_sha256}),
        })
    }
    pub fn catalogue(&self) -> cloud::Result<Vec<Value>> {
        self.backends.iter().map(|(id,backend)|{
            let p=backend.adapter.profile.pai();
            let (kind,limits,allocation)=("pai-dlc",json!({"gpus":p.max_gpus,"cpus":p.max_cpus,"memory_gb":p.max_memory_gb,"walltime_seconds":86400}),json!(["guaranteed","allow_idle","idle_only"]));
            Ok(json!({"executor_id":id,"kind":kind,"enabled":self.enabled,"run_creation":self.creation,"limits":limits,"allocation_policies":allocation,"default_priority":5,
                "capabilities":{"code_delivery":self.code_delivery(id),"oci_images":true,"persistent_checkpoints":true,"dataset_upload":self.has_datasets(),"data_image_copy":false,"data_preparation":"inside_job","metrics":self.has_exchange(),"result_objects":self.has_artifacts(),"cpu_collection":backend.collector.is_some(),"capacity_query":true},"live_availability_checked":false}))
        }).collect()
    }
    pub fn capacity(&self, executor: &str, refresh: bool, now: &str) -> cloud::Result<Value> {
        let backend = self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        let name = format!("capacity-{executor}.json");
        let _effect = if refresh {
            Some(
                self.slots
                    .lock(
                        OsStr::new(&format!("capacity-{executor}.lock")),
                        LockOptions {
                            create: true,
                            ..Default::default()
                        },
                    )
                    .map_err(io)?,
            )
        } else {
            None
        };
        let old = if self.slots.entry_exists(OsStr::new(&name)).map_err(io)? {
            serde_json::from_value::<Observation>(read(&self.slots, &name)?)
                .map_err(why!(cloud::Error::evidence()))?
        } else {
            capacity::unknown(executor).map_err(why!(cloud::Error::configuration()))?
        };
        if old.executor_id != executor {
            return Err(cloud::Error::evidence());
        }
        if !refresh {
            return Ok(json!(
                capacity::view(old, now).map_err(why!(cloud::Error::evidence()))?
            ));
        }
        let data = backend
            .adapter
            .pai_quota(&AtomicBool::new(false))
            .and_then(|row| {
                ml_exp_core::cloud::pai::quota_view(backend.adapter.profile.pai(), &row)
                    .map_err(why!(cloud::Error::evidence()))
            });
        let next = capacity::refresh(old, data.map_err(why!("unavailable")), now)
            .map_err(why!(cloud::Error::evidence()))?;
        self.slots
            .replace_regular(OsStr::new(&name), &encode(&next)?)
            .map_err(io)?;
        Ok(json!(next))
    }
}
