use super::*;
fn job() -> Job {
    Job {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        submission_token: "a".repeat(32),
        source_id: format!("source.{}", "b".repeat(64)),
        image: format!("registry.example/team/image@sha256:{}", "c".repeat(64)),
        command: vec![
            "python".into(),
            "train.py".into(),
            "a'b\n$(do-not-run)".into(),
        ],
        gpus: 1,
        cpus: 8,
        memory_gb: 64,
        max_time_seconds: 86400,
        priority: 5,
    }
}
fn pai() -> pai::Profile {
    pai::Profile {
        region: "us-southeast-1".into(),
        workspace_id: "253".into(),
        quota_id: "quota1".into(),
        max_gpus: 8,
        max_cpus: 200,
        max_memory_gb: 1600,
        gpu_type: None,
        storage: Some(pai::Storage {
            cpfs_uri: "bmcpfs://cpfs-example.us-southeast-1.cpfs.aliyuncs.com/".into(),
            workspace_user: "researcher".into(),
            role_arn: "acs:ram::12345:role/pai-data-researcher".into(),
        }),
        vpc: None,
    }
}
fn sense() -> sensecore::Profile {
    sensecore::Profile {
        scope: sensecore::Scope {
            subscription: "sub".into(),
            resource_group: "group".into(),
            region: "cn-east-1".into(),
            zone: "cn-east-1a".into(),
            workspace_zone: None,
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
    }
}
#[test]
fn cloud_run_authorship_launch_and_approval_preserve_science_and_exact_attempt() {
    let runtime = run::Runtime {
        runtime_id: format!("runtime.{}", "1".repeat(64)),
        source_id: job().source_id,
        image: job().image,
        source_dir: "/workspace".into(),
        worker_path: "/opt/ml-expd/worker".into(),
        worker_sha256: format!("sha256:{}", "d".repeat(64)),
        worker_features: Default::default(),
    };
    let request:run::Create=serde_json::from_value(json!({"run_id":"train","runtime_id":runtime.runtime_id,"executor":"pai-h200","command":["python","train.py","'$(not-a-shell)'","中文"],"resources":{"gpus":1,"cpus":8,"memory_gb":64},"walltime_seconds":3600,"seed":42})).unwrap();
    let p = format!("sha256:{}", "a".repeat(64));
    let s = format!("sha256:{}", "b".repeat(64));
    let profile = pai().scope().unwrap();
    let docs = run::documents(
        "demo",
        &request,
        &runtime,
        run::Authority {
            project_sha256: &p,
            source_receipt_sha256: &s,
            profile_sha256: &profile,
            tracking_route: &json!({"enabled":false}),
            target: &helper::Target {
                backend_id: "pai-h200".into(),
                profile_sha256: profile.clone(),
                storage_scope: format!(
                    "sha256:{}",
                    canonical::digest(
                        &json!({"profile":profile,"run_root":"/mnt/workspace/ml-expd/runs"})
                    )
                    .unwrap()
                ),
            },
            run_root: "/mnt/workspace/ml-expd/runs",
            data_root: "/mnt/workspace/ml-expd/datasets",
            datasets: &[],
            checkpoint: None,
        },
        "2026-10-29T00:00:00Z",
    )
    .unwrap();
    let binding = crate::first_submission::native::Binding {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        project_sha256: p,
        run_sha256: helper::sha256_bytes(&serde_json::to_vec(&docs.run).unwrap()),
        attempt_sha256: helper::sha256_bytes(&serde_json::to_vec(&docs.attempt).unwrap()),
        source_receipt_sha256: s,
        backend_config_sha256: profile,
    };
    let execution = run::execution(
        &docs.run,
        &docs.attempt,
        binding.clone(),
        &runtime,
        "/mnt/workspace/ml-expd/runs",
        &"0".repeat(32),
        pai::Allocation::AllowIdle,
    )
    .unwrap();
    assert_eq!(execution.job.command[1], "cloud-launch");
    let package: launch::Package = serde_json::from_str(&execution.job.command[3]).unwrap();
    assert_eq!(package.manifest.argv, request.command);
    assert_eq!(package.manifest.identity.attempt_id, "attempt-001");
    assert!(
        package
            .config
            .attempt_root
            .ends_with("demo/train/attempt-001")
    );
    let approval = crate::first_submission::native::Approval {
        max_gpu_hours: 1.,
        reason: String::new(),
    };
    let plan = action::prepare(
        &execution,
        &approval,
        "2026-10-29T00:00:00Z",
        "2026-10-29T01:00:00Z",
    )
    .unwrap();
    assert!(!plan.to_string().contains(&"0".repeat(32)));
    action::validate(&plan, &execution).unwrap();
    let mut mixed = docs.attempt.clone();
    mixed["seed"] = json!(43);
    assert!(
        run::execution(
            &docs.run,
            &mixed,
            binding,
            &runtime,
            "/mnt/workspace/ml-expd/runs",
            &"0".repeat(32),
            pai::Allocation::AllowIdle
        )
        .is_err()
    );
}
#[test]
fn dlc_resources_idle_permission_personal_mounts_and_frozen_ownership_are_explicit() {
    let p = pai();
    let job = job();
    for allocation in [
        pai::Allocation::Guaranteed,
        pai::Allocation::AllowIdle,
        pai::Allocation::IdleOnly,
    ] {
        let request = p.create(&job, allocation).unwrap();
        request.validate().unwrap();
        let body = request.body.unwrap();
        assert_eq!(body["JobMaxRunningTimeMinutes"], 1440);
        assert_eq!(body["Priority"], 5);
        assert_eq!(
            body["Settings"]["EnableTideResource"],
            allocation != pai::Allocation::Guaranteed
        );
        assert_eq!(body["Settings"]["OversoldType"], allocation.oversold());
        assert_eq!(body["JobSpecs"][0]["OversoldType"], allocation.oversold());
        assert_eq!(body["JobSpecs"][0]["RestartPolicy"], "Never");
        assert_eq!(body["JobSpecs"][0]["UseSpotInstance"], false);
        assert!(
            body["JobSpecs"][0]["ResourceConfig"]
                .get("GPUType")
                .is_none()
        );
        assert_eq!(body["DataSources"].as_array().unwrap().len(), 3);
        assert_eq!(body["DataSources"][0]["MountPath"], "/mnt/workspace");
        assert_eq!(body["DataSources"][1]["MountAccess"], "RO");
    }
    let row = json!({"JobId":"dlc1","WorkspaceId":253,"ResourceId":p.quota_id,"DisplayName":job.name().unwrap(),"Settings":{"Tags":job.tags(&p.scope().unwrap()).unwrap()},"Status":"Queuing"});
    p.validate_job(&row, &job, Some("dlc1")).unwrap();
    let mut bad = row.clone();
    bad["Settings"]["Tags"]["ml-expd.attempt"] = json!("attempt-002");
    assert!(p.validate_job(&bad, &job, Some("dlc1")).is_err());
    let mut bad = p.clone();
    bad.storage.as_mut().unwrap().role_arn = "acs:ram::12345:role/pai-data-someone-else".into();
    assert!(bad.create(&job, pai::Allocation::Guaranteed).is_err());
    let mut bad = job.clone();
    bad.max_time_seconds = 86401;
    assert!(p.create(&bad, pai::Allocation::Guaranteed).is_err());
    let queue = json!({"TotalCount":1,"Workloads":[{"WorkloadId":"dlc1","WorkspaceId":"253","QuotaId":"quota1","WorkloadType":"dlc","Position":3,"Reason":"{\"CodeType\":\"ConfigRule\",\"Code\":\"roleMaximumResource\",\"Reason\":\"GPU: 8 limited by role\"}"}]});
    let detail = p.validate_queue(&queue, "dlc1").unwrap().unwrap();
    assert_eq!(detail.code.as_deref(), Some("roleMaximumResource"));
    assert_eq!(detail.position, Some(3));
    assert!(p.validate_queue(&queue, "dlc-foreign").is_err());
}
#[test]
fn sensecore_training_copy_and_quota_selection_preserve_boundaries() {
    let p = sense();
    let job = job();
    let body = p.create(&job).unwrap().body.unwrap();
    assert_eq!(body["fault_tolerance"]["backoff_limit"], 0);
    assert_eq!(body["scheduling"]["quota_type"], "SPOT");
    let mut debug = p.clone();
    debug.cluster = "debug-cluster".into();
    assert!(debug.create(&job).is_err());
    debug.purpose = sensecore::Purpose::DataCopy;
    debug.worker_spec = "cpu.2c4g".into();
    debug.worker_gpus = 0;
    debug.worker_cpus = 2;
    debug.worker_memory_gb = 4;
    debug.quota = sensecore::Quota::Reserved;
    let mut cpu = job.clone();
    cpu.gpus = 0;
    cpu.cpus = 2;
    cpu.memory_gb = 4;
    assert!(debug.create(&cpu).is_ok());
    let spec = json!({"name":p.worker_spec,"cpu":{"vcpu_allocatable":8},"memory":{"allocatable":64},"device":{"number":1},"zones":[p.scope.zone]});
    let make = |name: &str, quota: &str| {
        let row = json!({"name":name,"id":p.scope.path("aec2s",name).unwrap(),"state":"ACTIVE","vpc_id":p.vpc_id,"spot_status":[{"spot_quota":{"device":quota}}]});
        sensecore::cluster(&p, &row, std::slice::from_ref(&spec)).unwrap()
    };
    let clusters = vec![
        make("c", "12.125"),
        make("b", "12.125000001"),
        make("debug-big", "999"),
        make("a", "12.125000001"),
    ];
    assert_eq!(
        sensecore::highest_spot(&p, &clusters, None).unwrap().name,
        "a"
    );
    let mut incompatible = clusters.clone();
    incompatible[3].vpc_id = "other".into();
    assert_eq!(
        sensecore::highest_spot(&p, &incompatible, None)
            .unwrap()
            .name,
        "b"
    );
    let mut duplicate = clusters.clone();
    duplicate.push(clusters[0].clone());
    assert!(sensecore::highest_spot(&p, &duplicate, None).is_err());
    let mut reserved = json!({"name":"debug","id":p.scope.path("aec2s","debug").unwrap(),"state":"ACTIVE","vpc_id":p.vpc_id,"spot_status":[]});
    let zero = sensecore::cluster(&p, &reserved, std::slice::from_ref(&spec)).unwrap();
    assert_eq!(zero.spot_devices_nano, 0);
    assert!(sensecore::highest_spot(&p, &[zero], None).is_err());
    reserved["spot_status"] = Value::Null;
    assert!(sensecore::cluster(&p, &reserved, std::slice::from_ref(&spec)).is_err());
    let row = json!({"name":"training","id":p.scope.path("aec2s","training").unwrap(),"state":"ACTIVE","vpc_id":p.vpc_id,"spot_status":[{"spot_quota":{"device":"1"}}]});
    let mut too_small = spec.clone();
    too_small["memory"]["allocatable"] = json!(63);
    assert!(
        !sensecore::cluster(&p, &row, &[too_small])
            .unwrap()
            .compatible
    );
    let row = json!({"name":job.name().unwrap(),"uid":"uid1","ownership":{"user_id":p.scope.user_id},"state":"RUNNING"});
    p.validate_job(&row, &job, Some("uid1")).unwrap();
    assert!(p.validate_job(&row, &job, Some("uid2")).is_err());
}
#[test]
fn sensecore_workspace_zone_does_not_change_pool_or_mount_zone_and_legacy_hashes_remain() {
    let mut p = sense();
    let legacy = serde_json::to_value(&p).unwrap();
    assert!(legacy["scope"].get("workspace_zone").is_none());
    let restored: sensecore::Profile = serde_json::from_value(legacy).unwrap();
    assert_eq!(restored.digest().unwrap(), p.digest().unwrap());
    p.scope.workspace_zone = Some("cn-east-1z".into());
    assert!(
        p.jobs_path()
            .unwrap()
            .contains("/zones/cn-east-1z/workspaces/work/")
    );
    assert!(
        p.pools("1")
            .unwrap()
            .path
            .contains("/zones/cn-east-1z/workspaces/work/")
    );
    assert!(
        p.specs("training", "1")
            .unwrap()
            .path
            .contains("/zones/cn-east-1a/aec2s/training/")
    );
    assert_eq!(
        p.create(&job()).unwrap().body.unwrap()["mount"][0]["zone"],
        "cn-east-1a"
    );
    let job = job();
    let mut row = json!({"name":job.name().unwrap(),"uid":"uid1","ownership":{"user_id":p.scope.user_id},"state":"STARTING","id":format!("{}/trainingJobs/{}",p.scope.workspace_path().unwrap(),job.name().unwrap())});
    assert!(p.validate_job(&row, &job, Some("uid1")).is_ok());
    row["id"] = json!(format!(
        "{}/trainingJobs/{}",
        p.scope.path("workspaces", &p.scope.workspace).unwrap(),
        job.name().unwrap()
    ));
    assert!(p.validate_job(&row, &job, Some("uid1")).is_err());
    assert_eq!(
        p.stop(&job.name().unwrap()).unwrap().body.unwrap()["zone"],
        "cn-east-1z"
    );
    assert_ne!(restored.digest().unwrap(), p.digest().unwrap());
    p.scope.workspace_zone = Some("cn-other-1z".into());
    assert!(p.validate().is_err());
}
#[test]
fn offline_logs_and_pagination_refuse_foreign_hits_truncation_and_ambiguous_absence() {
    let p = sense();
    let workspace_id = "01234567-89ab-cdef-0123-456789abcdef";
    let hit = json!({"attributes":{"k8s.pod.name":"pod1","k8s.container.name":"container1"},"resource":{"resource_id":workspace_id},"log_time":"2026-10-29T00:00:00Z","body":"checkpoint saved"});
    let value = json!({"hits":[hit],"total":"1"});
    let end = 1793232000;
    let logs = sensecore::validate_offline_logs(
        workspace_id,
        &value,
        "pod1",
        "container1",
        end - 1,
        end + 1,
        100,
    )
    .unwrap();
    assert!(logs.available);
    assert_eq!(logs.text, "checkpoint saved");
    assert!(
        sensecore::validate_offline_logs(
            workspace_id,
            &value,
            "foreign",
            "container1",
            end - 1,
            end + 1,
            100
        )
        .is_err()
    );
    assert!(
        p.offline_logs(
            "another-user-station",
            workspace_id,
            "pod1",
            "container1",
            1..=2,
            10
        )
        .is_err()
    );
    let mut foreign = value.clone();
    foreign["hits"][0]["resource"]["resource_id"] = json!(p.scope.workspace_path().unwrap());
    assert!(
        sensecore::validate_offline_logs(
            workspace_id,
            &foreign,
            "pod1",
            "container1",
            end - 1,
            end + 1,
            100
        )
        .is_err()
    );
    assert!(Page::pai(&json!({"Jobs":[],"TotalCount":2}), "Jobs", 1, 100).is_err());
    assert!(
        Page::sensecore(
            &json!({"training_jobs":[],"next_page_token":"1"}),
            "training_jobs",
            "1"
        )
        .is_err()
    );
    assert!(
        Endpoint::Pai {
            product: Product::PaiDlc,
            region: "example.com/escape".into()
        }
        .host()
        .is_err()
    );
}

#[test]
fn offline_requests_use_exact_workspace_uuid_and_workspace_telemetry_zone() {
    let mut p = sense();
    p.scope.workspace_zone = Some("cn-east-1z".into());
    let row = json!({"name":p.scope.workspace,"type":"compute.workspace.v1.instance","zone":"cn-east-1z","subscription_name":p.scope.subscription,"resource_group_name":p.scope.resource_group,"id":"01234567-89ab-cdef-0123-456789abcdef"});
    let id = sensecore::log_workspace_id(&p, &[row.clone()]).unwrap();
    let station = format!("ts-user-{}", p.scope.user_id);
    let request = p
        .offline_logs(&station, &id, "pod1", "pytorch", 1..=2, 20)
        .unwrap();
    assert!(request.read_only);
    assert!(
        request
            .path
            .contains("/zones/cn-east-1z/telemetryStations/")
    );
    assert_eq!(request.body.unwrap()["resource_id"], json!([id]));
    assert_eq!(
        p.workspaces("1").unwrap().query["filter"],
        "resource_type='compute.workspace.v1.instance'"
    );
    assert!(sensecore::log_workspace_id(&p, &[]).is_err());
    assert!(sensecore::log_workspace_id(&p, &[row.clone(), row.clone()]).is_err());
    for (key, value) in [
        ("id", p.scope.workspace_path().unwrap()),
        ("name", "foreign".into()),
        ("zone", p.scope.zone.clone()),
        ("subscription_name", "foreign".into()),
        ("resource_group_name", "foreign".into()),
        ("type", "other.resource".into()),
    ] {
        let mut invalid = row.clone();
        invalid[key] = json!(value);
        assert!(
            sensecore::log_workspace_id(&p, &[invalid]).is_err(),
            "{key}"
        );
    }
    let mut deleted = row;
    deleted["deleted"] = json!(true);
    assert!(sensecore::log_workspace_id(&p, &[deleted]).is_err());
    assert!(
        p.offline_logs(&station, "not-a-uuid", "pod1", "pytorch", 1..=2, 20)
            .is_err()
    );
}
