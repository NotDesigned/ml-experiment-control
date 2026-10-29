//! A single experiment configuration and stable recovery identity. These
//! decisions neither build an image nor authorize/dispatch a training job.
use crate::{
    canonical,
    cloud::build,
    first_submission::native::{Approval, creation},
    submission,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub request_id: String,
    pub run: Map<String, Value>,
    pub runtime: Option<build::Definition>,
    pub approval: Approval,
}
impl Request {
    pub fn validate(&self, project: &str) -> Result<(), &'static str> {
        submission::validate_submission_token(&self.request_id)
            .map_err(|_| "preparation request identity")?;
        let run = self.resolve(project, &format!("sha256:{}", "0".repeat(64)))?;
        let hours = run.resources.gpus as f64 * run.walltime_seconds as f64 / 3600.0;
        if !self.approval.max_gpu_hours.is_finite()
            || self.approval.max_gpu_hours <= 0.0
            || hours > self.approval.max_gpu_hours
            || self.approval.reason.len() > 4000
        {
            return Err("preparation budget");
        }
        Ok(())
    }
    pub fn resolve(&self, project: &str, worker: &str) -> Result<creation::Request, &'static str> {
        let mut run = self.run.clone();
        if let Some(definition) = &self.runtime {
            let id = definition
                .id(project, worker)
                .map_err(|_| "preparation environment")?;
            if run.get("executor").and_then(Value::as_str) != Some(&definition.executor)
                || run.contains_key("runtime_id")
            {
                return Err("preparation environment scope");
            }
            run.insert("runtime_id".into(), json!(id));
        }
        let run: creation::Request = serde_json::from_value(Value::Object(run))
            .map_err(|_| "preparation Run configuration")?;
        run.validate(project)
            .map_err(|_| "preparation Run validation")?;
        Ok(run)
    }
    pub fn id(&self, project: &str) -> Result<String, &'static str> {
        self.validate(project)?;
        Ok(format!(
            "preparation-{}",
            &canonical::digest(&json!([
                "native-experiment-preparation.v1",
                project,
                self.request_id
            ]))
            .map_err(|_| "preparation identity")?[..32]
        ))
    }
}

/// A bounded Cartesian product. Expanding a batch never submits work and does
/// not choose a winning metric or merge different evaluation protocols.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    pub request_id: String,
    pub experiment: Request,
    #[serde(default)]
    pub parameters: std::collections::BTreeMap<String, Vec<Value>>,
    #[serde(default)]
    pub seeds: Vec<i64>,
}
pub const MAX_BATCH: usize = 256;
/// Unknown submissions consume capacity. Only exact terminal scheduler
/// evidence releases a slot; process completion and result publication do not.
pub fn occupied(row: &Value) -> bool {
    if row["status"] == "UNKNOWN" {
        return true;
    }
    let submission = &row["submission"];
    if submission.is_null() {
        return false;
    }
    if submission["status"]
        .as_str()
        .is_some_and(|s| matches!(s, "PREPARED" | "AUTHORIZED"))
    {
        return false;
    }
    let state = submission
        .pointer("/cloud_observation/observation/state")
        .or_else(|| submission.pointer("/execution/result/observation/status/state"));
    !state.and_then(Value::as_str).is_some_and(|s| {
        matches!(
            s,
            "SUCCEEDED" | "FAILED" | "CANCELLED" | "TIMEOUT" | "PREEMPTED"
        )
    })
}
impl Batch {
    pub fn expand(&self, project: &str) -> Result<Vec<Request>, &'static str> {
        submission::validate_submission_token(&self.request_id)
            .map_err(|_| "batch request identity")?;
        self.experiment.validate(project)?;
        let base = self
            .experiment
            .run
            .get("run_id")
            .and_then(Value::as_str)
            .ok_or("batch run identity")?;
        if base.len() > 100 || self.parameters.len() > 32 || self.seeds.len() > MAX_BATCH {
            return Err("batch bounds");
        }
        let mut choices = vec![Map::new()];
        for (key, values) in &self.parameters {
            if key.is_empty()
                || key.len() > 128
                || values.is_empty()
                || choices
                    .len()
                    .checked_mul(values.len())
                    .is_none_or(|n| n > MAX_BATCH)
            {
                return Err("batch parameter bounds");
            }
            let mut unique = std::collections::BTreeSet::new();
            for value in values {
                if !unique.insert(canonical::digest(value).map_err(|_| "batch parameter value")?) {
                    return Err("duplicate batch parameter");
                }
            }
            choices = choices
                .into_iter()
                .flat_map(|row| {
                    values.iter().map(move |value| {
                        let mut next = row.clone();
                        next.insert(key.clone(), value.clone());
                        next
                    })
                })
                .collect();
        }
        let seeds = if self.seeds.is_empty() {
            vec![
                self.experiment
                    .run
                    .get("seed")
                    .and_then(Value::as_i64)
                    .ok_or("batch seed")?,
            ]
        } else {
            self.seeds.clone()
        };
        if seeds
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != seeds.len()
            || choices
                .len()
                .checked_mul(seeds.len())
                .is_none_or(|n| n > MAX_BATCH)
        {
            return Err("batch seed or size bounds");
        }
        let mut requests = Vec::new();
        for row in choices {
            for seed in &seeds {
                let digest = canonical::digest(&json!([
                    "native-experiment-batch.v1",
                    project,
                    self.request_id,
                    self.experiment.request_id,
                    &row,
                    seed
                ]))
                .map_err(|_| "batch child identity")?;
                let mut request = self.experiment.clone();
                request.request_id = digest[..32].into();
                request
                    .run
                    .insert("run_id".into(), json!(format!("{base}-{}", &digest[..16])));
                request.run.insert("seed".into(), json!(seed));
                let parameters = request
                    .run
                    .entry("parameters")
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .ok_or("batch parameters object")?;
                parameters.extend(row.clone());
                request.validate(project)?;
                requests.push(request);
            }
        }
        Ok(requests)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Request {
        serde_json::from_value(json!({"request_id":"a".repeat(32),"run":{"run_id":"trial","runtime_id":format!("runtime.{}","1".repeat(64)),"executor":"dlc","command":["train"],"resources":{"gpus":1,"cpus":2,"memory_gb":4},"walltime_seconds":300,"seed":42},"approval":{"max_gpu_hours":1.0,"reason":"test"}})).unwrap()
    }
    #[test]
    fn prepare_refuses_budget_or_runtime_ambiguity_before_build_effects() {
        let mut a = request();
        assert!(a.validate("demo").is_ok());
        a.approval.max_gpu_hours = 0.01;
        assert!(a.validate("demo").is_err());
        a = request();
        let id = a.id("demo").unwrap();
        a.run.insert("seed".into(), json!(43));
        assert_eq!(id, a.id("demo").unwrap()); // same recovery slot, shell detects payload conflict
        assert_ne!(id, a.id("other").unwrap());
        a.runtime = Some(crate::cloud::build::Definition {
            source_id: format!("source.{}", "2".repeat(64)),
            executor: "dlc".into(),
            base_image: format!("registry.example/base@sha256:{}", "3".repeat(64)),
            dockerfile: "Dockerfile".into(),
        });
        assert!(a.validate("demo").is_err());
        a.run.remove("runtime_id");
        assert!(a.validate("demo").is_ok());
    }
    #[test]
    fn matrix_has_stable_independent_children_and_rejects_unbounded_or_duplicate_choices() {
        let mut b = Batch {
            request_id: "b".repeat(32),
            experiment: request(),
            parameters: std::collections::BTreeMap::from([(
                "lr".into(),
                vec![json!(0.01), json!(0.02)],
            )]),
            seeds: vec![42, 43],
        };
        let children = b.expand("demo").unwrap();
        assert_eq!(children.len(), 4);
        assert_eq!(json!(children), json!(b.expand("demo").unwrap()));
        assert_eq!(
            children
                .iter()
                .map(|c| c.id("demo").unwrap())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );
        assert_eq!(children[0].run["parameters"]["lr"], 0.01);
        assert_eq!(children[1].run["seed"], 43);
        b.seeds = vec![42, 42];
        assert!(b.expand("demo").is_err());
        b.seeds = vec![42];
        b.parameters.insert("lr".into(), vec![json!(1), json!(1)]);
        assert!(b.expand("demo").is_err());
        b.parameters
            .insert("lr".into(), (0..257).map(|n| json!(n)).collect());
        assert!(b.expand("demo").is_err());
    }
    #[test]
    fn scheduling_slots_follow_scheduler_evidence_and_reserve_unknown_effects() {
        assert!(!occupied(&json!({"status":"PREPARING","submission":null})));
        assert!(occupied(&json!({"status":"UNKNOWN"})));
        for status in ["PREPARED", "AUTHORIZED"] {
            assert!(!occupied(&json!({"submission":{"status":status}})));
        }
        assert!(occupied(
            &json!({"submission":{"status":"RECONCILE_REQUIRED"}})
        ));
        assert!(occupied(
            &json!({"submission":{"status":"VERIFIED","training":"COMPLETED","execution":{"result":{"observation":{"status":{"state":"RUNNING"}}}}}})
        ));
        for terminal in ["SUCCEEDED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED"] {
            assert!(!occupied(
                &json!({"submission":{"status":"VERIFIED","cloud_observation":{"observation":{"state":terminal}}}})
            ));
            assert!(!occupied(
                &json!({"submission":{"status":"VERIFIED","execution":{"result":{"observation":{"status":{"state":terminal}}}}}})
            ));
        }
    }
}
