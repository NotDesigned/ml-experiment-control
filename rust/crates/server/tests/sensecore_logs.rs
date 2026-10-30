//! Recorded provider shapes: RMH UUIDs and a workspace zone distinct from compute.
use ml_exp_core::cloud::{Job, Request, sensecore};
use ml_exp_server::backends::cloud::{
    Result,
    adapter::{Adapter, Handle, Profile, Transport, job_digest},
};
use serde_json::{Value, json};
use std::sync::{Mutex, atomic::AtomicBool};

const WORKSPACE_ID: &str = "01234567-89ab-cdef-0123-456789abcdef";

struct LogsTransport {
    profile: sensecore::Profile,
    row: Value,
    foreign_workspace: bool,
    searched: Mutex<bool>,
}
impl Transport for LogsTransport {
    fn request(&self, request: &Request, _: &AtomicBool) -> Result<Value> {
        assert!(request.read_only, "log lookup must not mutate the provider");
        request.validate().unwrap();
        let scope = &self.profile.scope;
        Ok(match request.action.as_str() {
            "GetTrainingJob" => self.row.clone(),
            "ListTelemetryStations" => json!({"resources":[{
                "name":format!("ts-user-{}",scope.user_id),
                "type":"monitor.ts.v1.telemetryStation",
                "zone":"cn-east-1z", "subscription_name":scope.subscription,
                "resource_group_name":scope.resource_group,
                "id":"11111111-1111-1111-1111-111111111111"
            }],"next_page_token":""}),
            "ListWorkspaces" => json!({"resources":[{
                "name":scope.workspace,"type":"compute.workspace.v1.instance",
                "zone":"cn-east-1z", "subscription_name":scope.subscription,
                "resource_group_name":if self.foreign_workspace {"foreign"} else {&scope.resource_group},
                "id":WORKSPACE_ID
            }],"next_page_token":""}),
            "ListWorkers" => {
                json!({"workers":[{"name":"owned-pod", "containers":[{"name":"pytorch"}]}],"next_page_token":""})
            }
            "SearchOfflineLogs" => {
                *self.searched.lock().unwrap() = true;
                assert!(
                    request
                        .path
                        .contains("/zones/cn-east-1z/telemetryStations/")
                );
                let body = request.body.as_ref().unwrap();
                assert_eq!(body["resource_id"], json!([WORKSPACE_ID]));
                assert_eq!(body["custom_filter"][0]["value"], "owned-pod");
                assert_eq!(body["custom_filter"][1]["value"], "pytorch");
                json!({"total":"1","hits":[{
                    "resource":{"resource_id":WORKSPACE_ID},
                    "attributes":{"k8s.pod.name":"owned-pod","k8s.container.name":"pytorch"},
                    "log_time":"2026-10-29T00:00:00Z", "body":"fixture failure evidence"
                }]})
            }
            _ => panic!("unexpected log operation"),
        })
    }
}

#[test]
fn exact_owned_logs_resolve_rmh_uuid_without_rewriting_the_frozen_profile() {
    let p = sensecore::Profile {
        scope: sensecore::Scope {
            subscription: "sub".into(),
            resource_group: "group".into(),
            region: "cn-east-1".into(),
            zone: "cn-east-1a".into(),
            workspace_zone: Some("cn-east-1z".into()),
            workspace: "work".into(),
            user_id: "12345678-1234-1234-1234-123456789abc".into(),
        },
        cluster: "training".into(),
        vpc_id: "vpc1".into(),
        worker_spec: "gpu.8c64g".into(),
        worker_gpus: 1,
        worker_cpus: 8,
        worker_memory_gb: 64,
        worker_nodes: 1,
        storage_volume: "nas".into(),
        storage_subdir: "/personal/researcher".into(),
        mount_path: "/data".into(),
        purpose: sensecore::Purpose::Training,
        quota: sensecore::Quota::Spot,
    };
    let job = Job {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        submission_token: "a".repeat(32),
        source_id: format!("source.{}", "b".repeat(64)),
        image: format!("registry.example/team/image@sha256:{}", "c".repeat(64)),
        command: vec!["train".into()],
        gpus: 1,
        cpus: 8,
        memory_gb: 64,
        max_time_seconds: 300,
        priority: 5,
    };
    let row = json!({"name":job.name().unwrap(),"uid":"owned-job-uid",
        "id":format!("{}/trainingJobs/{}",p.scope.workspace_path().unwrap(),job.name().unwrap()),
        "ownership":{"user_id":p.scope.user_id},"state":"FAILED","create_time":"2026-10-29T00:00:00Z"});
    let profile = Profile::Sensecore(p.clone());
    let digest = profile.digest().unwrap();
    let handle = Handle {
        profile_sha256: digest.clone(),
        request_sha256: job_digest(&job).unwrap(),
        job_id: job.name().unwrap(),
        provider_uid: Some("owned-job-uid".into()),
    };
    for foreign_workspace in [false, true] {
        let transport = LogsTransport {
            profile: p.clone(),
            row: row.clone(),
            foreign_workspace,
            searched: Mutex::new(false),
        };
        let adapter = Adapter::new(profile.clone(), &transport).unwrap();
        let result =
            adapter.sensecore_offline_logs(&job, &handle, 50, 1793232001, &AtomicBool::new(false));
        if foreign_workspace {
            assert!(result.is_err());
            assert!(!*transport.searched.lock().unwrap());
        } else {
            let logs = result.unwrap();
            assert!(logs.available);
            assert_eq!(logs.text, "fixture failure evidence");
            assert!(*transport.searched.lock().unwrap());
        }
        assert_eq!(adapter.profile.digest().unwrap(), digest);
    }
}
