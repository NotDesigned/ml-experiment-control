//! Public cloud authorship is exercised with real native stores and HTTP
//! routing. Credentials and images are synthetic; no provider call is made.
use ml_exp_core::{
    cloud::{pai, run::Runtime},
    first_submission::native::{Approval, creation::Request},
    sources::Policy,
};
use ml_exp_server::{
    backends::cloud::{
        adapter::{Adapter, Profile, Transport},
        outbox::Outbox,
    },
    cloud_runs::{Service, Settings},
    read_config, sources,
};
use serde_json::{Value, json};
use std::{fs, io::Cursor, os::unix::fs::PermissionsExt, sync::Arc};
fn private(path: &std::path::Path, value: &Value) {
    ml_exp_io::write_atomic(path, &serde_json::to_vec(value).unwrap(), true).unwrap();
}
fn archive() -> Vec<u8> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    let mut tar = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    let body = b"print('cloud fixture')\n";
    header.set_size(body.len() as u64);
    header.set_mode(0o500);
    header.set_cksum();
    tar.append_data(&mut header, "train.py", &body[..]).unwrap();
    tar.into_inner().unwrap().finish().unwrap()
}
#[test]
fn create_reuse_authorize_restart_and_frozen_guard_without_provider_effects() {
    let temp = tempfile::tempdir().unwrap();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let registry = temp.path().join("registry");
    read_config::initialize(&registry).unwrap();
    let project = temp.path().join("project.json");
    let runs = temp.path().join("runs");
    fs::create_dir(&runs).unwrap();
    private(
        &project,
        &json!({"schema_version":1,"project":"demo","title":"Native cloud fixture","run_roots":["runs"]}),
    );
    let settings = read_config::Settings {
        schema_version: 1,
        registry_root: registry.clone(),
        projects: vec![read_config::Project {
            project: "demo".into(),
            project_file: project,
            run_roots: vec![runs.clone()],
            runs: Vec::new(),
            discover_runs: None,
        }],
        source_policy: Some(Policy::default()),
    };
    sources::Service::initialize(&settings).unwrap();
    let sources = Arc::new(sources::Service::open(&settings, None).unwrap());
    let bytes = archive();
    let source = sources
        .publish(
            "demo",
            &mut Cursor::new(&bytes),
            &ml_exp_core::slurm::helper::sha256_bytes(&bytes)[7..],
            "2026-10-29T00:00:00Z",
        )
        .unwrap();
    let runtime = Runtime {
        runtime_id: format!("runtime.{}", "1".repeat(64)),
        source_id: source["source_id"].as_str().unwrap().into(),
        image: format!("registry.example/team/code@sha256:{}", "2".repeat(64)),
        source_dir: "/workspace".into(),
        worker_path: "/opt/ml-expd/worker".into(),
        worker_sha256: format!("sha256:{}", "3".repeat(64)),
        worker_features: ["storage-layout.v1".into(), "package-download.v1".into()].into(),
    };
    let credentials = temp.path().join("credentials.json");
    private(
        &credentials,
        &json!({"provider":"pai","access_key_id":"synthetic","access_key_secret":"never-sent"}),
    );
    let profile = pai::Profile {
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
    };
    let signing_key = temp.path().join("worker-signing-key");
    ml_exp_io::write_atomic(
        &signing_key,
        b"synthetic-exchange-key-0000000000000000000000",
        true,
    )
    .unwrap();
    let config = json!({"schema_version":1,"enabled":false,"allow_run_creation":true,"exchange":{"public_url":"https://api.example/ml-expd","signing_key_file":signing_key},"backends":[{"executor":"pai-h200","profile":{"provider":"pai","region":profile.region,"workspace_id":profile.workspace_id,"quota_id":profile.quota_id,"max_gpus":8,"max_cpus":200,"max_memory_gb":1600,"storage":profile.storage},"run_root":"/mnt/workspace/ml-expd/runs","credentials_file":credentials,"registry_auth_file":null,"runtimes":{runtime.runtime_id.clone():runtime}}]});
    Service::initialize(&settings).unwrap();
    assert!(Service::initialize(&settings).is_err());
    let open = || {
        Service::open(
            settings.clone(),
            Settings::from_slice(&serde_json::to_vec(&config).unwrap()).unwrap(),
            sources.clone(),
            None,
            Default::default(),
        )
        .unwrap()
    };
    let service = open();
    let request:Request=serde_json::from_value(json!({"run_id":"train","runtime_id":runtime.runtime_id,"executor":"pai-h200","command":["python","train.py"],"resources":{"gpus":1,"cpus":8,"memory_gb":64},"walltime_seconds":300,"seed":42})).unwrap();
    // One recovery slot owns the exact request. Preparing or reading it cannot
    // allocate a provider job, silently accept a changed payload or create a Run.
    let setup: ml_exp_core::preparation::Request = serde_json::from_value(json!({
        "request_id":"a".repeat(32), "run": {"run_id":"setup-only", "runtime_id":runtime.runtime_id,
        "executor":"pai-h200", "command":["python","train.py"], "resources":{"gpus":1,"cpus":8,"memory_gb":64},
        "walltime_seconds":300,"seed":42}, "approval":{"max_gpu_hours":1.0}
    })).unwrap();
    let setup_view = service
        .experiment_prepare("demo", setup.clone(), None)
        .unwrap();
    let setup_id = setup_view["preparation_id"].as_str().unwrap();
    assert_eq!(setup_view["status"], "PREPARED");
    assert_eq!(setup_view["scheduler_submitted"], false);
    assert_eq!(
        service
            .experiment_prepare("demo", setup.clone(), None)
            .unwrap(),
        setup_view
    );
    assert_eq!(
        service.experiment_status("demo", setup_id).unwrap(),
        setup_view
    );
    assert_eq!(
        service.experiment_for_run("demo", "setup-only").unwrap(),
        Some(setup_view.clone())
    );
    assert_eq!(
        service.experiment_for_run("demo", "not-prepared").unwrap(),
        None
    );
    assert_eq!(
        service
            .experiment_status("demo", &format!("preparation-{}", "9".repeat(32)))
            .unwrap_err()
            .http_status,
        Some(404)
    );
    let mut discover_setup = setup.clone();
    discover_setup.request_id = "d".repeat(32);
    discover_setup
        .run
        .insert("run_id".into(), json!("discover-only"));
    let discover_view = service
        .experiment_prepare("demo", discover_setup, None)
        .unwrap();
    let mut changed_setup = setup.clone();
    changed_setup.run.insert("seed".into(), json!(43));
    assert!(
        service
            .experiment_prepare("demo", changed_setup, None)
            .is_err()
    );
    let mut oversized = setup.clone();
    oversized.request_id = "b".repeat(32);
    oversized.run.insert(
        "resources".into(),
        json!({"gpus":9,"cpus":8,"memory_gb":64}),
    );
    let error = service
        .experiment_prepare("demo", oversized.clone(), None)
        .unwrap_err();
    assert_eq!(error.http_status, Some(422));
    assert!(
        service
            .experiment_status("demo", &oversized.id("demo").unwrap())
            .is_err()
    );
    assert!(!runs.join("setup-only").exists());
    // Research parameters need not fit an invented half-package estimate.
    // Preparing persists the same identity but creates no Run or provider job.
    let mut metadata = setup.clone();
    metadata.request_id = "c".repeat(32);
    metadata.run.insert(
        "parameters".into(),
        json!({"description":"x".repeat(40 * 1024)}),
    );
    let value = service
        .experiment_prepare("demo", metadata.clone(), None)
        .unwrap();
    assert_eq!(value["status"], "PREPARED");
    assert_eq!(value["scheduler_submitted"], false);
    assert_eq!(
        service.experiment_prepare("demo", metadata, None).unwrap(),
        value
    );
    assert_eq!(
        service
            .experiment_status("demo", value["preparation_id"].as_str().unwrap())
            .unwrap(),
        value
    );
    assert!(!runs.join("setup-only").exists());
    // CPU Runs use the same durable preparation path. Storage provisioning is
    // frozen only for new authorship; legacy absence remains absent on replay.
    let mut cpu = request.clone();
    cpu.run_id = "cpu-only".into();
    cpu.resources.gpus = 0;
    cpu.env
        .insert("EXPERIMENT_LABEL".into(), "frozen literal $(text)".into());
    service
        .create_run("demo", cpu.clone(), "2026-10-29T00:00:00Z")
        .unwrap();
    let cpu_manifest = runs.join("cpu-only/manifest.json");
    let mut frozen: Value = serde_json::from_slice(&fs::read(&cpu_manifest).unwrap()).unwrap();
    assert_eq!(frozen["env"]["EXPERIMENT_LABEL"], "frozen literal $(text)");
    assert_eq!(
        frozen["evaluation"]["protocol"]["env"],
        serde_json::json!(cpu.env)
    );
    assert_eq!(frozen["native_cloud"]["storage"]["mount"], "/mnt/workspace");
    frozen["native_cloud"]
        .as_object_mut()
        .unwrap()
        .remove("storage");
    fs::write(&cpu_manifest, serde_json::to_vec(&frozen).unwrap()).unwrap();
    let mut old_attempt = frozen.clone();
    old_attempt["attempt_id"] = json!("attempt-001");
    fs::write(
        runs.join("cpu-only/attempts/attempt-001/attempt.json"),
        serde_json::to_vec(&old_attempt).unwrap(),
    )
    .unwrap();
    assert_eq!(
        service
            .create_run("demo", cpu.clone(), "2026-10-29T00:01:00Z")
            .unwrap()["reused"],
        true
    );
    let cpu_plan = service
        .prepare(
            "demo",
            "cpu-only",
            Approval {
                max_gpu_hours: 1.,
                reason: "CPU test".into(),
            },
            "2026-10-29T00:01:00Z",
            "2026-10-29T01:01:00Z",
            &"9".repeat(32),
        )
        .unwrap();
    assert_eq!(cpu_plan["status"], "PREPARED");
    let action = cpu_plan["submission_id"].as_str().unwrap();
    let error = service.artifacts(action).unwrap_err();
    assert_eq!(error.code, "ARTIFACT_NOT_PUBLISHED");
    assert_eq!(error.http_status, Some(409));
    assert_eq!(
        service
            .artifacts("action-0000000000000000")
            .unwrap_err()
            .http_status,
        Some(404)
    );
    assert_eq!(
        service.dataset_list("demo", "pai-h200").unwrap()["datasets"],
        json!([])
    );
    let dataset = ml_exp_core::datasets::State::new(
        "demo".into(),
        serde_json::from_value(frozen["native_cloud"]["target"].clone()).unwrap(),
        ml_exp_core::datasets::Definition {
            sha256: "a".repeat(64),
            archive_bytes: 4096,
            files: vec![ml_exp_core::artifacts::FileReceipt {
                path: "train.bin".into(),
                bytes: 5,
                sha256: "b".repeat(64),
            }],
        },
    )
    .unwrap();
    let name = format!(
        "dataset-{}.json",
        ml_exp_core::canonical::digest(&json!(["demo", "pai-h200", dataset.asset_id])).unwrap()
    );
    let dataset_file = settings
        .registry_root
        .join("cloud-submissions/slots")
        .join(name);
    private(
        &dataset_file,
        &json!({"state":dataset,"archive_verified":false}),
    );
    let dataset_bytes = fs::read(&dataset_file).unwrap();
    let catalog = service.dataset_list("demo", "pai-h200").unwrap();
    assert_eq!(catalog["datasets"][0]["status"], "UPLOADING");
    assert_eq!(catalog["datasets"][0]["received_parts"], 0);
    assert_eq!(fs::read(&dataset_file).unwrap(), dataset_bytes);
    // A real store compiles the full worker package, including all 304 hashes.
    // These are synthetic receipt fixtures, not claims about a provider tree.
    let definition = ml_exp_core::datasets::Definition {
        sha256: "e".repeat(64),
        archive_bytes: 1024 * 1024,
        files: (0..304)
            .map(|i| ml_exp_core::artifacts::FileReceipt {
                path: format!("dataset/full-inventory/part-{i:04}.bin"),
                bytes: 128,
                sha256: "f".repeat(64),
            })
            .collect(),
    };
    let mut ready = ml_exp_core::datasets::State::new(
        "demo".into(),
        dataset.target.clone(),
        definition.clone(),
    )
    .unwrap();
    ready.parts.insert(
        0,
        ml_exp_core::multipart::PartRecord {
            bytes: definition.archive_bytes,
            sha256: "d".repeat(64),
        },
    );
    let receipt = ml_exp_core::datasets::Receipt {
        contract: ml_exp_core::datasets::CONTRACT.into(),
        project: "demo".into(),
        target: ready.target.clone(),
        asset_id: ready.asset_id.clone(),
        definition: definition.clone(),
        root: format!(
            "/mnt/workspace/ml-expd/datasets/demo/{}/tree",
            ready.asset_id
        ),
    };
    ready = ready.publish(receipt).unwrap();
    let ready_file = registry.join("cloud-submissions/slots").join(format!(
        "dataset-{}.json",
        ml_exp_core::canonical::digest(&json!(["demo", "pai-h200", ready.asset_id])).unwrap()
    ));
    private(&ready_file, &json!({"state":ready,"archive_verified":true}));
    let mut large = request.clone();
    large.run_id = "full-inventory".into();
    large.data = vec![ml_exp_core::datasets::Input {
        asset_id: ready.asset_id.clone(),
        name: "data".into(),
    }];
    service
        .create_run("demo", large, "2026-10-29T00:00:00Z")
        .unwrap();
    let large_plan = service
        .prepare(
            "demo",
            "full-inventory",
            Approval {
                max_gpu_hours: 1.,
                reason: String::new(),
            },
            "2026-10-29T00:01:00Z",
            "2026-10-29T01:01:00Z",
            &"8".repeat(32),
        )
        .unwrap();
    let large_slot = registry.join("cloud-submissions/slots").join(format!(
        "{}.json",
        ml_exp_core::first_submission::native::attempt_slot(
            "demo",
            "full-inventory",
            "attempt-001"
        )
        .unwrap()
    ));
    let stored: Value = serde_json::from_slice(&fs::read(large_slot).unwrap()).unwrap();
    let execution: ml_exp_core::cloud::action::Execution =
        serde_json::from_value(stored["execution"].clone()).unwrap();
    let package = execution.package().unwrap();
    assert_eq!(package.config.datasets[0].receipt.definition, definition);
    assert!(package.imports.is_empty());
    assert!(serde_json::to_vec(&package).unwrap().len() <= 65536);
    let ready_bytes = fs::read(&ready_file).unwrap();
    // A verified upload with no backend READY receipt needs the full import
    // inventory too. This was the real 304-file >64KiB failure.
    let mut staging = ready.clone();
    staging.ready = None;
    fs::write(
        &ready_file,
        serde_json::to_vec(&json!({"state":staging,"archive_verified":true})).unwrap(),
    )
    .unwrap();
    let mut download_run = request.clone();
    download_run.run_id = "download-inventory".into();
    download_run.data = vec![ml_exp_core::datasets::Input {
        asset_id: ready.asset_id.clone(),
        name: "data".into(),
    }];
    service
        .create_run("demo", download_run.clone(), "2026-10-29T00:00:00Z")
        .unwrap();
    let prepare_download = || {
        service
            .prepare(
                "demo",
                "download-inventory",
                Approval {
                    max_gpu_hours: 1.,
                    reason: String::new(),
                },
                "2026-10-29T00:01:00Z",
                "2026-10-29T01:01:00Z",
                &"6".repeat(32),
            )
            .unwrap()
    };
    let download_plan = prepare_download();
    assert_eq!(
        prepare_download()["submission_id"],
        download_plan["submission_id"]
    );
    let download_slot = registry.join("cloud-submissions/slots").join(format!(
        "{}.json",
        ml_exp_core::first_submission::native::attempt_slot(
            "demo",
            "download-inventory",
            "attempt-001"
        )
        .unwrap()
    ));
    let download_saved: Value = serde_json::from_slice(&fs::read(&download_slot).unwrap()).unwrap();
    let download_execution: ml_exp_core::cloud::action::Execution =
        serde_json::from_value(download_saved["execution"].clone()).unwrap();
    let download_package = download_execution.package().unwrap();
    let download_bytes =
        serde_json::to_vec(&serde_json::to_value(&download_package).unwrap()).unwrap();
    assert!(download_bytes.len() > 65536);
    assert_eq!(download_package.imports.len(), 1);
    assert_eq!(
        download_package.config.datasets[0]
            .receipt
            .definition
            .files
            .len(),
        304
    );
    assert_eq!(download_execution.job.command[2], "--package-url");
    assert!(download_execution.job.command.join(" ").len() < 1024);
    assert_eq!(
        download_execution.job.command[7],
        ml_exp_core::slurm::helper::sha256_bytes(&download_bytes)
    );
    assert!(download_package.command(&runtime.worker_path).is_err());
    let mut legacy_config = config.clone();
    legacy_config["backends"][0]["runtimes"][&runtime.runtime_id]["worker_features"] =
        json!(["storage-layout.v1"]);
    let legacy = Service::open(
        settings.clone(),
        Settings::from_slice(&serde_json::to_vec(&legacy_config).unwrap()).unwrap(),
        sources.clone(),
        None,
        Default::default(),
    )
    .unwrap();
    download_run.run_id = "legacy-inventory".into();
    legacy
        .create_run("demo", download_run, "2026-10-29T00:00:00Z")
        .unwrap();
    let error = legacy
        .prepare(
            "demo",
            "legacy-inventory",
            Approval {
                max_gpu_hours: 1.,
                reason: String::new(),
            },
            "2026-10-29T00:01:00Z",
            "2026-10-29T01:01:00Z",
            &"5".repeat(32),
        )
        .unwrap_err();
    assert_eq!(error.code, "LAUNCH_METADATA_TOO_LARGE");
    assert_eq!(error.http_status, Some(422));
    fs::write(&ready_file, &ready_bytes).unwrap();
    // Simulate a stop after the complete copy intent was saved but before its
    // outbox entry. A later API URL or launcher must not rebuild this command.
    let storage = Settings::from_slice(&serde_json::to_vec(&config).unwrap())
        .unwrap()
        .backends[0]
        .storage_layout()
        .unwrap();
    let channel = ml_exp_core::datasets::import::DataChannel {
        url: format!(
            "https://previous.example/api/dataset-exchange/demo/pai-h200/{}",
            ready.asset_id
        ),
        token: "9".repeat(64),
    };
    let import = ml_exp_core::datasets::import::Config {
        root: storage.datasets.to_string_lossy().into(),
        target: ready.target.clone(),
        project: "demo".into(),
        definition: definition.clone(),
        source: ml_exp_core::datasets::import::Source::DatasetParts {
            channel: channel.clone(),
        },
        timeout_seconds: 3600,
    };
    let mut copy = ml_exp_core::cloud::data::materialization_job(
        &runtime,
        &import,
        &channel,
        &storage,
        &"7".repeat(32),
    )
    .unwrap();
    copy.command[3] = "# original frozen launcher\n".to_owned() + &copy.command[3];
    let frozen_copy = json!({"state":ready,"archive_verified":true,"copy_runtime":runtime,"copy_storage":storage,"copy_request_id":"7".repeat(32),"copy_intent":copy});
    fs::write(&ready_file, serde_json::to_vec(&frozen_copy).unwrap()).unwrap();
    let intent_bytes = fs::read(&ready_file).unwrap();
    let restarted = open();
    assert_eq!(
        restarted
            .dataset_state("demo", "pai-h200", &ready.asset_id)
            .unwrap()["runtime_id"],
        runtime.runtime_id
    );
    assert_eq!(fs::read(&ready_file).unwrap(), intent_bytes);
    let outbox = ml_exp_server::backends::cloud::outbox::Outbox::open(
        &registry.join("cloud-submissions/sync-effects"),
    )
    .unwrap();
    assert!(outbox.optional_status(&copy).unwrap().is_none());
    outbox
        .prepare(
            &ml_exp_server::backends::cloud::adapter::Profile::Pai(profile.clone()),
            &copy,
            pai::Allocation::Guaranteed,
            &|| Ok(()),
        )
        .unwrap();
    assert_eq!(outbox.status(&copy).unwrap()["retry_safe"], true);
    // Preparation after the simulated crash retains exactly the original Job.
    assert_eq!(
        restarted
            .dataset_state("demo", "pai-h200", &ready.asset_id)
            .unwrap()["materialization"],
        "READY"
    );
    let mut missing_intent = frozen_copy.clone();
    missing_intent
        .as_object_mut()
        .unwrap()
        .remove("copy_intent");
    fs::write(&ready_file, serde_json::to_vec(&missing_intent).unwrap()).unwrap();
    assert!(
        restarted
            .dataset_state("demo", "pai-h200", &ready.asset_id)
            .is_err()
    );
    fs::write(&ready_file, &ready_bytes).unwrap();
    let mut absent = ready.clone();
    absent.ready = None;
    fs::write(
        &ready_file,
        serde_json::to_vec(&json!({"state":absent,"archive_verified":true})).unwrap(),
    )
    .unwrap();
    // A download-capable plan no longer fails on the inline size of the
    // recomputed inventory; reading returns the same frozen Submission.
    assert_eq!(
        service
            .for_run("demo", "full-inventory", "2026-10-29T00:02:00Z")
            .unwrap()["submission_id"],
        large_plan["submission_id"]
    );
    let mut partial = ready.clone();
    partial.parts.clear();
    fs::write(
        &ready_file,
        serde_json::to_vec(&json!({"state":partial,"archive_verified":true})).unwrap(),
    )
    .unwrap();
    // Reading stays with the frozen Submission; only a first spend looks at the
    // data delivery state as it is now.
    assert_eq!(
        service
            .for_run("demo", "full-inventory", "2026-10-29T00:02:00Z")
            .unwrap()["submission_id"],
        large_plan["submission_id"]
    );
    assert!(
        service
            .prepare(
                "demo",
                "full-inventory",
                Approval {
                    max_gpu_hours: 1.,
                    reason: String::new(),
                },
                "2026-10-29T00:02:00Z",
                "2026-10-29T01:01:00Z",
                &"a".repeat(32),
            )
            .is_err()
    );
    fs::write(&ready_file, ready_bytes).unwrap();
    assert_eq!(
        service
            .for_run("demo", "full-inventory", "2026-10-29T00:02:00Z")
            .unwrap()["submission_id"],
        large_plan["submission_id"]
    );
    let action = large_plan["submission_id"].as_str().unwrap();
    let progress = service.submission_progress(action).unwrap().unwrap();
    assert_eq!(progress["submission_id"], action);
    assert_eq!(progress["run_id"], "full-inventory");
    assert_eq!(progress["attempt_id"], "attempt-001");
    assert_eq!(progress["availability"], "NOT_REPORTED");
    assert!(progress.get("input_delivery").is_none());
    let listed = service.run_submissions("demo", "full-inventory").unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["submission_id"], action);
    assert!(
        serde_json::from_slice::<Value>(&fs::read(&cpu_manifest).unwrap()).unwrap()["native_cloud"]
            .get("storage")
            .is_none()
    );
    let first = service
        .create_run("demo", request.clone(), "2026-10-29T00:00:00Z")
        .unwrap();
    assert_eq!(first["scheduler_mutation_performed"], false);
    assert_eq!(
        service
            .create_run("demo", request.clone(), "2026-10-29T00:01:00Z")
            .unwrap()["reused"],
        true
    );
    let prepared = service
        .prepare(
            "demo",
            "train",
            Approval {
                max_gpu_hours: 1.,
                reason: String::new(),
            },
            "2026-10-29T00:01:00Z",
            "2026-10-29T01:01:00Z",
            &"1".repeat(32),
        )
        .unwrap();
    let action = prepared["submission_id"].as_str().unwrap();
    assert_eq!(prepared["status"], "PREPARED");
    assert!(service.handles_action(action).unwrap());
    let original_run = fs::read(runs.join("train/manifest.json")).unwrap();
    let mut relocated = config.clone();
    relocated["exchange"]["public_url"] = json!("https://new-api.example");
    let relocated_service = Service::open(
        settings.clone(),
        Settings::from_slice(&serde_json::to_vec(&relocated).unwrap()).unwrap(),
        sources.clone(),
        None,
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        relocated_service
            .for_run("demo", "train", "2026-10-29T00:02:00Z")
            .unwrap()["submission_id"],
        action
    );
    assert_eq!(
        fs::read(runs.join("train/manifest.json")).unwrap(),
        original_run
    );
    drop(relocated_service);
    let authorized = service
        .authorize(
            action,
            "owned fixture",
            "2026-10-29T00:02:00Z",
            &"2".repeat(32),
        )
        .unwrap();
    assert_eq!(authorized["status"], "AUTHORIZED");
    assert!(
        service
            .execute(
                action,
                &format!("EXECUTE {action}"),
                "2026-10-29T00:03:00Z",
                &"3".repeat(32),
                &std::sync::atomic::AtomicBool::new(false)
            )
            .is_err()
    );
    drop(service);
    let service = open();
    assert_eq!(
        service
            .for_run("demo", "train", "2026-10-29T00:04:00Z")
            .unwrap()["submission_id"],
        action
    );
    // Start the actual server executable with cloud-only configuration and use
    // the official client. This also checks blocking clients stay off Tokio.
    let read_file = temp.path().join("reads.json");
    private(&read_file, &serde_json::to_value(&settings).unwrap());
    let cloud_file = temp.path().join("cloud.json");
    private(&cloud_file, &config);
    let token = "native-cloud-http-synthetic-token-000000000000";
    let token_file = temp.path().join("token");
    ml_exp_io::write_atomic(&token_file, token.as_bytes(), true).unwrap();
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_ml-expd-rs"))
        .args(["--bind", "127.0.0.1:0", "--token-file"])
        .arg(&token_file)
        .arg("--read-config")
        .arg(&read_file)
        .arg("--source-archives")
        .arg("--cloud-config")
        .arg(&cloud_file)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    struct Server(std::process::Child);
    impl Drop for Server {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let mut server = Server(child);
    let stdout = server.0.stdout.take().unwrap();
    let (sent, received) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let line = std::io::BufReader::new(stdout)
            .lines()
            .next()
            .unwrap()
            .unwrap();
        let _ = sent.send(line);
    });
    let address = received
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("native cloud startup");
    let client = ml_exp_client::Client::new(&format!("http://{address}"), token.into()).unwrap();
    let health = client.negotiate().unwrap();
    let policy = client.get("/api/actions/policy").unwrap();
    assert_eq!(policy["allow_source_imports"], true);
    assert_eq!(policy["allow_scheduler_mutations"], false);
    assert_eq!(policy["allow_resource_deletions"], false);
    assert_eq!(
        client.get("/api/readiness").unwrap()["training"],
        "INCOMPLETE"
    );
    let slots = registry.join("cloud-submissions/slots");
    let listing = || {
        let mut names = fs::read_dir(&slots)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect::<Vec<_>>();
        names.sort();
        names
    };
    let before = listing();
    let datasets = client.get("/api/projects/demo/datasets/pai-h200").unwrap();
    assert_eq!(
        client
            .get("/api/runs/demo/discover-only/preparation")
            .unwrap(),
        discover_view
    );
    assert!(matches!(
        client.get(&format!(
            "/api/projects/demo/experiment-preparations/preparation-{}",
            "9".repeat(32)
        )),
        Err(ml_exp_client::Error::Http(error)) if error.status == 404
    ));
    assert_eq!(datasets["datasets"][0]["asset_id"], dataset.asset_id);
    assert_eq!(listing(), before);
    let capacity = client.get("/api/backends/pai-h200/capacity").unwrap();
    assert_eq!(capacity["status"], "UNKNOWN");
    assert_eq!(listing(), before); // no lock, cache or provider work on GET
    assert_eq!(
        client.get("/api/executors").unwrap()["executors"][0]["executor_id"],
        "pai-h200"
    );
    assert!(
        health["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "native-cloud-submissions.v1")
    );
    assert!(
        health["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "checkpoint-record-read.v1")
    );
    let plan = ml_exp_client::submissions::plan(ml_exp_client::submissions::Command::Create {
        project: "demo",
        request: &request,
    })
    .unwrap();
    assert_eq!(
        ml_exp_client::submissions::execute(&client, &plan, &health).unwrap()["reused"],
        true
    );
    assert_eq!(
        client.get(&format!("/api/submissions/{action}")).unwrap()["status"],
        "AUTHORIZED"
    );
    assert_eq!(
        client.get("/api/runs/demo/train").unwrap()["native_submission"]["status"],
        "AUTHORIZED"
    );
    // A provider observation can hold the outbox lock across a slow GET. A
    // concurrent cached read must report temporary contention, not corrupt
    // frozen evidence, and cannot dispatch or change the original Action.
    let effects_lock =
        fs::File::open(registry.join("cloud-submissions/effects/.cloud-effects.lock")).unwrap();
    fs2::FileExt::lock_exclusive(&effects_lock).unwrap();
    let http = reqwest::blocking::Client::new();
    for path in [
        format!("/api/submissions/{action}"),
        "/api/runs/demo/train".into(),
    ] {
        let response = http
            .get(format!("http://{address}{path}"))
            .bearer_auth(token)
            .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2")
            .send()
            .unwrap();
        assert_eq!(response.status().as_u16(), 503);
        let error: Value = serde_json::from_slice(&response.bytes().unwrap()).unwrap();
        assert_eq!(error["code"], "CLOUD_RUN_BUSY", "{error}");
    }
    fs2::FileExt::unlock(&effects_lock).unwrap();
    assert_eq!(
        client.get(&format!("/api/submissions/{action}")).unwrap()["status"],
        "AUTHORIZED"
    );
    assert_eq!(
        fs::read(runs.join("train/manifest.json")).unwrap(),
        original_run
    );
    let initial_status = ml_exp_client::status::observe(
        &client,
        &health,
        None,
        ml_exp_client::observation::Selection {
            project: Some("demo"),
            run: Some("train"),
            attempt: Some("attempt-001"),
        },
        true,
    )
    .unwrap();
    assert!(initial_status["worker"]["state"].is_null());
    assert!(initial_status["training"]["process_state"].is_null());
    assert_eq!(initial_status["training"]["status"], "UNKNOWN");
    assert_eq!(
        initial_status["publication"]["diagnostic"],
        "ARTIFACT_PUBLICATION_NOT_OBSERVED"
    );
    // A synthetic Action receipt exercises ingress without calling any cloud
    // provider. This is a state-machine/HTTP fixture, not GPU acceptance.
    let actions =
        ml_exp_server::operations::Store::open_existing(registry.join("actions")).unwrap();
    let execution = actions.cloud_request(action).unwrap();
    let package = execution.package().unwrap();
    let binding = package.config.binding().unwrap();
    let sha = ml_exp_core::slurm::helper::sha256_bytes;
    let bytes=b"{\"name\":\"loss\",\"unit\":\"nats/token\",\"step\":10,\"value\":1.2,\"variant_id\":\"one\"}\n{\"name\":\"loss\",\"unit\":\"nats/token\",\"step\":10,\"value\":1.7,\"variant_id\":\"two\"}\n".to_vec();
    let config_sha256 = sha(&serde_json::to_vec(&package.config).unwrap());
    let packet:ml_exp_core::cloud::exchange::Packet=serde_json::from_value(json!({"contract":ml_exp_core::cloud::exchange::CONTRACT,"sequence":1,"previous_sha256":null,"package_sha256":sha(&serde_json::to_vec(&package).unwrap()),
        "evidence":{"contract":ml_exp_core::managed::remote::OBSERVATION_CONTRACT,"config_sha256":config_sha256,"job_id":binding.sha256,
            "worker":{"contract":ml_exp_core::managed::evidence::CONTRACT,"identity":binding.identity,"manifest_sha256":binding.sha256,"phase":"STARTED","safe_to_retry":false,"receipt":null,"runtime_environment_sha256":"a".repeat(64)},
            "stdout":{"text":"native worker fixture\n","offset_bytes":0,"total_bytes":22,"problem":null},
            "stderr":{"text":"","offset_bytes":0,"total_bytes":0,"problem":null},
            "metrics":{"text":String::from_utf8(bytes.clone()).unwrap(),"offset_bytes":0,"total_bytes":bytes.len(),"problem":null}},
        "history":{"contract":ml_exp_core::managed::history::CONTRACT,"config_sha256":config_sha256,"start":null,"next":{"device":1,"inode":2,"offset_bytes":bytes.len(),"prefix_sha256":sha(&bytes)},"bytes":bytes,"total_bytes":bytes.len(),"process_finished":false},"checkpoint":null})).unwrap();
    let payload = serde_json::to_vec(&packet).unwrap();
    let http = reqwest::blocking::Client::new();
    let channel = package.channel.as_ref().unwrap();
    let download_channel = download_package.channel.as_ref().unwrap();
    let download_path = "/api/worker-exchange/demo/download-inventory/attempt-001/package";
    for _ in 0..2 {
        let response = http
            .get(format!("http://{address}{download_path}"))
            .bearer_auth(&download_channel.token)
            .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2")
            .send()
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let body = response.bytes().unwrap();
        assert_eq!(body.as_ref(), download_bytes.as_slice());
        let fetched: ml_exp_core::cloud::launch::Package = serde_json::from_slice(&body).unwrap();
        fetched.validate().unwrap();
        assert_eq!(
            fetched.config.datasets[0].receipt.definition.files.len(),
            304
        );
    }
    for (credential, path, protocol, status) in [
        ("wrong", download_path, "2", 401),
        (
            download_channel.token.as_str(),
            "/api/worker-exchange/demo/download-inventory/attempt-002/package",
            "2",
            401,
        ),
        (download_channel.token.as_str(), "/api/health", "2", 401),
        (download_channel.token.as_str(), download_path, "1", 426),
    ] {
        assert_eq!(
            http.get(format!("http://{address}{path}"))
                .bearer_auth(credential)
                .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, protocol)
                .send()
                .unwrap()
                .status()
                .as_u16(),
            status
        );
    }
    assert_eq!(
        fs::read(&download_slot).unwrap(),
        serde_json::to_vec(&download_saved).unwrap()
    );
    let post = |token: &str, path: &str, body: Vec<u8>| {
        http.post(format!("http://{address}{path}"))
            .bearer_auth(token)
            .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2")
            .header("content-type", "application/json")
            .body(body)
            .send()
            .unwrap()
    };
    let path = "/api/worker-exchange/demo/train/attempt-001";
    assert_eq!(
        post("wrong-worker-token", path, payload.clone())
            .status()
            .as_u16(),
        401
    );
    assert!(!registry.join("tracking").exists());
    assert_eq!(
        post(&channel.token, path, payload.clone())
            .status()
            .as_u16(),
        409
    );
    let mut current = actions.observe(action).unwrap()["execution"].clone();
    current["status"] = json!("VERIFIED");
    actions
        .commit(
            action,
            &ml_exp_core::operations::Transition {
                execution: current,
                event: json!({"event":"SYNTHETIC_TEST_RECEIPT"}),
                expected_status: Some("AUTHORIZED".into()),
            },
            &"4".repeat(32),
            "2026-10-29T00:04:01Z",
        )
        .unwrap();
    for _ in 0..2 {
        let response = post(&channel.token, path, payload.clone());
        let status = response.status().as_u16();
        let ack: Value = serde_json::from_slice(&response.bytes().unwrap()).unwrap();
        assert_eq!(status, 200, "{ack}");
        assert_eq!(ack["packet_sha256"], packet.sha256().unwrap());
    }
    // The diagnostic capability reads only its exact Attempt, without a global
    // API token or a mutation packet. It grants no ordinary API access.
    let status_path = format!("{path}/status");
    let get_status = |token: &str, path: &str, protocol: &str| {
        http.get(format!("http://{address}{path}"))
            .bearer_auth(token)
            .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, protocol)
            .send()
            .unwrap()
    };
    for _ in 0..2 {
        let response = get_status(&channel.token, &status_path, "2");
        assert_eq!(response.status().as_u16(), 200);
        let value: Value = serde_json::from_slice(&response.bytes().unwrap()).unwrap();
        assert_eq!(value["contract"], "sync-diagnostic-status.v1");
        assert_eq!(value["exchange"]["sequence"], 1);
        assert_eq!(value["artifacts"]["prepared"], false);
        assert!(value.get("package").is_none());
        assert!(!value.to_string().contains(&channel.token));
    }
    assert_eq!(
        get_status("wrong-worker-token", &status_path, "2")
            .status()
            .as_u16(),
        401
    );
    assert_eq!(
        get_status(&channel.token, &status_path, "1")
            .status()
            .as_u16(),
        426
    );
    assert_eq!(
        get_status(
            &channel.token,
            "/api/worker-exchange/demo/train/attempt-002/status",
            "2"
        )
        .status()
        .as_u16(),
        401
    );
    assert_eq!(
        get_status(&channel.token, "/api/health", "2")
            .status()
            .as_u16(),
        401
    );
    assert_eq!(
        get_status(&channel.token, &format!("{status_path}?refresh=true"), "2")
            .status()
            .as_u16(),
        422
    );
    assert!(
        ml_exp_client::Client::new(&format!("http://{address}"), channel.token.clone())
            .unwrap()
            .negotiate()
            .is_err()
    );
    let records = client
        .get("/api/attempts/demo/train::attempt-001/metrics?format=records")
        .unwrap();
    assert_eq!(records["records"].as_array().unwrap().len(), 2, "{records}");
    let logs = client
        .get("/api/attempts/demo/train::attempt-001/logs")
        .unwrap();
    assert!(logs.to_string().contains("native worker fixture"));
    let mut skipped = packet.clone();
    skipped.sequence = 3;
    skipped.previous_sha256 = Some(packet.sha256().unwrap());
    assert_eq!(
        post(&channel.token, path, serde_json::to_vec(&skipped).unwrap())
            .status()
            .as_u16(),
        409
    );
    assert_eq!(
        post(
            &channel.token,
            "/api/worker-exchange/demo/train/attempt-002",
            payload
        )
        .status()
        .as_u16(),
        401
    );
    assert_eq!(
        client.get("/api/runs/demo/train/tracking").unwrap()["scopes"][0]["total"],
        3
    );
    drop(server);
    let mut renew_run = request.clone();
    renew_run.run_id = "renew-only".into();
    service
        .create_run("demo", renew_run, "2026-10-29T00:00:00Z")
        .unwrap();
    let prior = service
        .prepare(
            "demo",
            "renew-only",
            Approval {
                max_gpu_hours: 1.,
                reason: String::new(),
            },
            "2026-10-29T00:00:00Z",
            "2026-10-29T00:30:00Z",
            &"7".repeat(32),
        )
        .unwrap();
    let old = prior["submission_id"].as_str().unwrap();
    service
        .authorize(
            old,
            "owned fixture",
            "2026-10-29T00:01:00Z",
            &"8".repeat(32),
        )
        .unwrap();
    let renewal = ml_exp_core::renewal::Request {
        request_id: "c".repeat(32),
        confirmation: format!("RENEW {old}"),
    };
    let context = || ml_exp_server::submissions::Context {
        now: "2026-10-29T01:00:00Z",
        expires: "2026-10-29T01:30:00Z",
        transition_id: "99999999999999999999999999999999",
    };
    let next = service.renew(old, renewal.clone(), context()).unwrap();
    assert_ne!(next["submission_id"], old);
    assert_eq!(next["status"], "PREPARED");
    assert_eq!(next["attempt_id"], "attempt-001");
    assert_eq!(
        service.renew(old, renewal, context()).unwrap()["submission_id"],
        next["submission_id"]
    );
    assert_eq!(
        service.get(old, context().now).unwrap()["status"],
        "AUTHORIZED"
    );
    assert_eq!(
        service
            .for_run("demo", "renew-only", context().now)
            .unwrap()["submission_id"],
        next["submission_id"]
    );
    assert!(
        service
            .authorize(old, "expired", context().now, &"0".repeat(32))
            .is_err()
    );
    check_builder_dispatch_evidence(&settings, &config, sources.clone(), &runtime, temp.path());
    let mut enabled_config = config.clone();
    enabled_config["enabled"] = json!(true);
    let enabled = Service::open(
        settings.clone(),
        Settings::from_slice(&serde_json::to_vec(&enabled_config).unwrap()).unwrap(),
        sources.clone(),
        None,
        Default::default(),
    )
    .unwrap();
    // Two Attempts that have not started, stopped at different first-spend gates.
    let unspent = |run: &str, transition: &str| {
        let mut unspent_run = request.clone();
        unspent_run.run_id = run.into();
        service
            .create_run("demo", unspent_run, "2026-10-29T01:00:00Z")
            .unwrap();
        service
            .prepare(
                "demo",
                run,
                Approval {
                    max_gpu_hours: 1.,
                    reason: String::new(),
                },
                "2026-10-29T01:00:00Z",
                "2026-10-29T01:30:00Z",
                transition,
            )
            .unwrap()["submission_id"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let reused = unspent("unspent-prepared", &"a".repeat(32));
    let authorized = unspent("unspent-authorized", &"b".repeat(32));
    service
        .authorize(
            &authorized,
            "owned fixture",
            "2026-10-29T01:01:00Z",
            &"c".repeat(32),
        )
        .unwrap();
    // Editing the project file is ordinary work while an Attempt is running.
    ml_exp_io::write_atomic(
        &temp.path().join("project.json"),
        &serde_json::to_vec(
            &json!({"schema_version":1,"project":"demo","title":"Edited after submission","run_roots":["runs"]}),
        )
        .unwrap(),
        false,
    )
    .unwrap();
    let now = "2026-10-29T01:02:00Z";
    let refused = "CLOUD_EVIDENCE_INVALID";
    assert_eq!(
        service
            .prepare(
                "demo",
                "unspent-prepared",
                Approval {
                    max_gpu_hours: 1.,
                    reason: String::new(),
                },
                now,
                "2026-10-29T01:30:00Z",
                &"d".repeat(32),
            )
            .unwrap_err()
            .code,
        refused
    );
    assert_eq!(
        service
            .authorize(&reused, "owned fixture", now, &"e".repeat(32))
            .unwrap_err()
            .code,
        refused
    );
    // Cancellation is set, so a missing gate would fail differently and in no
    // case reach a provider.
    assert_eq!(
        enabled
            .execute(
                &authorized,
                &format!("EXECUTE {authorized}"),
                now,
                &"f".repeat(32),
                &std::sync::atomic::AtomicBool::new(true),
            )
            .unwrap_err()
            .code,
        refused
    );
    let unstarted = service.get(&authorized, now).unwrap();
    assert_eq!(unstarted["status"], "AUTHORIZED");
    assert!(unstarted.get("cloud_observation").is_none());
    check_running_attempt_after_project_edit(
        &service,
        &enabled,
        &registry,
        &config,
        &profile,
        action,
        &execution,
        &channel.token,
        &packet,
    );
    let mut changed = request;
    changed.seed = 43;
    assert!(
        service
            .create_run("demo", changed, "2026-10-29T00:05:00Z")
            .is_err()
    );
    // Reads keep serving the frozen Attempt; only a first spend looks at the Run.
    let path = runs.join("train/manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["seed"] = json!(123);
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert_eq!(service.get(action, now).unwrap()["status"], "VERIFIED");
    // Deleting the Run is the one change that stops a running Attempt's service.
    ml_exp_server::deletion::RunFence::new(&registry)
        .mark("demo", "train", &serde_json::Map::new(), now)
        .unwrap();
    assert_eq!(service.get(action, now).unwrap_err().code, refused);
    assert_eq!(
        service
            .worker_package("demo", "train", "attempt-001", &channel.token)
            .unwrap_err()
            .code,
        refused
    );
    assert_eq!(
        service
            .receive("demo", "train", "attempt-001", &channel.token, &packet, now)
            .unwrap_err()
            .code,
        refused
    );
    assert_eq!(
        service
            .refresh(action, now, &std::sync::atomic::AtomicBool::new(true))
            .unwrap_err()
            .code,
        refused
    );
}

/// Stands in for the provider while the outbox records one owned job.
struct OwnedProvider {
    row: Value,
    stopped: std::sync::atomic::AtomicBool,
}
impl Transport for OwnedProvider {
    fn request(
        &self,
        request: &ml_exp_core::cloud::Request,
        _: &std::sync::atomic::AtomicBool,
    ) -> ml_exp_server::backends::cloud::Result<Value> {
        let mut row = self.row.clone();
        if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
            row["Status"] = json!("Stopped");
        }
        Ok(match request.action.as_str() {
            "CreateJob" => json!({"JobId":row["JobId"]}),
            "GetJob" => row,
            "StopJob" => {
                self.stopped
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                json!({})
            }
            other => panic!("unexpected provider operation {other}"),
        })
    }
}

/// `service` and `enabled` hold real provider clients, so each call passes a set
/// cancellation flag: the client refuses before opening a connection, and an error
/// or diagnostic from that point proves the call got past every local gate.
#[allow(clippy::too_many_arguments)]
fn check_running_attempt_after_project_edit(
    service: &Service,
    enabled: &Service,
    registry: &std::path::Path,
    config: &Value,
    profile: &pai::Profile,
    action: &str,
    execution: &ml_exp_core::cloud::action::Execution,
    token: &str,
    packet: &ml_exp_core::cloud::exchange::Packet,
) {
    use std::sync::atomic::AtomicBool;
    let now = "2026-10-29T01:03:00Z";
    let job = &execution.job;
    assert_eq!(service.get(action, now).unwrap()["status"], "VERIFIED");
    assert_eq!(
        service.for_run("demo", "train", now).unwrap()["submission_id"],
        action
    );

    // Worker exchange keeps its own trust edge, the channel token.
    assert!(
        service
            .worker_package("demo", "train", "attempt-001", token)
            .is_ok()
    );
    assert_eq!(
        service
            .worker_package("demo", "train", "attempt-001", "wrong-worker-token")
            .unwrap_err()
            .http_status,
        Some(401)
    );
    let mut next = packet.clone();
    next.sequence = 2;
    next.previous_sha256 = Some(packet.sha256().unwrap());
    next.history = None;
    assert_eq!(
        service
            .receive(
                "demo",
                "train",
                "attempt-001",
                "wrong-worker-token",
                &next,
                now
            )
            .unwrap_err()
            .http_status,
        Some(401)
    );
    let ack = service
        .receive("demo", "train", "attempt-001", token, &next, now)
        .unwrap();
    assert_eq!(ack.sequence, 2);

    let provider_profile: Profile =
        serde_json::from_value(config["backends"][0]["profile"].clone()).unwrap();
    assert_eq!(provider_profile.digest().unwrap(), execution.profile_sha256);
    let provider = Adapter::new(
        provider_profile.clone(),
        OwnedProvider {
            row: json!({"JobId":"dlc-owned","WorkspaceId":profile.workspace_id,"ResourceId":profile.quota_id,
                "DisplayName":job.name().unwrap(),"Settings":{"Tags":job.tags(&profile.scope().unwrap()).unwrap()},"Status":"Running"}),
            stopped: AtomicBool::new(false),
        },
    )
    .unwrap();
    let outbox = Outbox::open(&registry.join("cloud-submissions/effects")).unwrap();
    let slot = Outbox::slot(job).unwrap();
    let unguarded = || -> ml_exp_server::backends::cloud::Result<()> { Ok(()) };
    outbox
        .prepare(&provider_profile, job, execution.allocation, &unguarded)
        .unwrap();
    outbox
        .execute(
            &provider,
            job,
            execution.allocation,
            &format!("SUBMIT {slot}"),
            &unguarded,
            &AtomicBool::new(false),
        )
        .unwrap();

    // Refresh reaches the provider client and retains its failure as a diagnostic.
    let refreshed = service
        .refresh(action, now, &AtomicBool::new(true))
        .unwrap();
    assert_eq!(
        refreshed["cloud_observation"]["handle"]["job_id"],
        "dlc-owned"
    );
    assert!(!refreshed["cloud_observation"]["diagnostic"].is_null());

    // Cancel reaches the stop step and takes no effect while cancellation is set.
    let confirmation = format!("CANCEL {action}");
    assert_eq!(
        enabled
            .cancel(action, &confirmation, now, &AtomicBool::new(true))
            .unwrap_err()
            .code,
        "CLOUD_RESPONSE_UNAVAILABLE"
    );
    assert_eq!(outbox.status(job).unwrap()["cancellation_started"], false);
    // A cancel already recorded continues through reconcile, again without a replay.
    outbox
        .stop(
            &provider,
            job,
            execution.allocation,
            &format!("STOP {slot}"),
            &unguarded,
            &AtomicBool::new(false),
        )
        .unwrap();
    let cancelling = enabled
        .cancel(action, &confirmation, now, &AtomicBool::new(true))
        .unwrap();
    assert_eq!(
        cancelling["cloud_observation"]["cancellation_started"],
        true
    );
    assert_eq!(
        cancelling["cloud_observation"]["termination_verified"],
        true
    );
}

fn check_builder_dispatch_evidence(
    settings: &read_config::Settings,
    base: &Value,
    sources: Arc<sources::Service>,
    original: &Runtime,
    root: &std::path::Path,
) {
    use std::{
        io::{BufRead, Read, Write},
        os::unix::net::UnixListener,
        time::{Duration, Instant},
    };
    let socket = root.join("owned-builder.sock");
    let mut config = base.clone();
    config["enabled"] = json!(true);
    config["builds"] = json!({"socket":socket,"peer_uid":ml_exp_io::directory::effective_uid(),
        "worker_sha256":original.worker_sha256});
    let service = Arc::new(
        Service::open(
            settings.clone(),
            Settings::from_slice(&serde_json::to_vec(&config).unwrap()).unwrap(),
            sources.clone(),
            None,
            Default::default(),
        )
        .unwrap(),
    );
    let definition = ml_exp_core::cloud::build::Definition {
        source_id: original.source_id.clone(),
        executor: "pai-h200".into(),
        base_image: original.image.clone(),
        dockerfile: "Dockerfile".into(),
    };
    let prepared = service.prepare_runtime("demo", definition.clone()).unwrap();
    let id = prepared["runtime_id"].as_str().unwrap();
    let wait = |id: &str, expected: &str| {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let value = service.runtime_status("demo", id).unwrap();
            if value["status"] == expected {
                break value;
            }
            assert!(
                Instant::now() < deadline,
                "builder state did not settle: {value}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    service
        .execute_runtime("demo", id, &format!("BUILD {id}"))
        .unwrap();
    let failed = wait(id, "PREPARED");
    assert_eq!(failed["retry_safe"], true);
    assert_eq!(failed["build_error"]["code"], "NATIVE_BUILD_NOT_DISPATCHED");
    // A stale socket has the same proven pre-send boundary as a missing one.
    drop(UnixListener::bind(&socket).unwrap());
    service
        .execute_runtime("demo", id, &format!("BUILD {id}"))
        .unwrap();
    assert_eq!(
        wait(id, "PREPARED")["build_error"]["code"],
        "NATIVE_BUILD_NOT_DISPATCHED"
    );
    fs::remove_file(&socket).unwrap();
    let listener = UnixListener::bind(&socket).unwrap();
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (finish_tx, finish_rx) = std::sync::mpsc::channel();
    let builder = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut reader = std::io::BufReader::new(stream);
        let mut length = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            if let Some(value) = line.strip_prefix("Content-Length: ") {
                length = value.trim().parse::<usize>().unwrap();
            }
        }
        assert!(length > 0 && length < 65536);
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        let request: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(request["operation"], "native-build");
        entered_tx.send(()).unwrap();
        finish_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let reply = json!({"project":request["project"],"runtime_id":request["runtime_id"],
            "source_id":request["source_id"],"recipe":ml_exp_core::cloud::build::CONTRACT,
            "runtime":{"runtime_id":request["runtime_id"],"source_id":request["source_id"],
                "image":request["base_image"],"source_dir":"/workspace","worker_path":"/worker",
                "worker_sha256":request["worker_sha256"]}})
        .to_string();
        write!(
            reader.get_mut(),
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
            reply.len(),
            reply
        )
        .unwrap();
    });
    service
        .execute_runtime("demo", id, &format!("BUILD {id}"))
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(
        service.runtime_status("demo", id).unwrap()["status"],
        "EXECUTING"
    );
    assert_eq!(
        service.reconcile_runtime("demo", id).unwrap()["status"],
        "EXECUTING"
    );
    let reopened = Service::open(
        settings.clone(),
        Settings::from_slice(&serde_json::to_vec(&config).unwrap()).unwrap(),
        sources.clone(),
        None,
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        reopened.runtime_status("demo", id).unwrap()["status"],
        "RECONCILE_REQUIRED"
    );
    finish_tx.send(()).unwrap();
    assert_eq!(wait(id, "READY")["retry_safe"], false);
    builder.join().unwrap();
    check_materializer_diagnostics(settings, &config, sources.clone(), id, original, root);
    fs::remove_file(&socket).unwrap();
    let listener = UnixListener::bind(&socket).unwrap();
    let ambiguous = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut bytes = [0; 1];
        stream.read_exact(&mut bytes).unwrap(); // A partial send is already uncertain.
    });
    let mut lost = definition;
    lost.dockerfile = "Lost.Dockerfile".into();
    let lost = service.prepare_runtime("demo", lost).unwrap();
    let lost_id = lost["runtime_id"].as_str().unwrap();
    service
        .execute_runtime("demo", lost_id, &format!("BUILD {lost_id}"))
        .unwrap();
    ambiguous.join().unwrap();
    // The in-flight view is EXECUTING; a lost reply must settle as uncertain.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let value = service.runtime_status("demo", lost_id).unwrap();
        if value["build_error"]["code"] == "NATIVE_BUILD_RPC_UNCERTAIN" {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(5));
    }
    let value = service
        .execute_runtime("demo", lost_id, &format!("BUILD {lost_id}"))
        .unwrap();
    assert_eq!(value["status"], "RECONCILE_REQUIRED");
    assert_eq!(value["retry_safe"], false);
    fs::remove_file(&socket).unwrap();
    let listener = UnixListener::bind(&socket).unwrap();
    let proof = json!({"source":"pai-dsw-reconcile.v1","bundle_id":lost_id.strip_prefix("runtime.").unwrap(),
        "observed_at":time::OffsetDateTime::now_utc().format(&time::format_description::well_known::Rfc3339).unwrap(),
        "registry_manifest_status":404,"active_builders":0,"leases_clear":true});
    let replies = vec![
        json!({"code":"PAI_DSW_BUILD_UNCERTAIN","details":{"phase":"BUILDING_ON_DSW","reason":"DSW_OBSERVATION_UNAVAILABLE","log_offset":40330,"password":"never-exposed"}}),
        json!({"code":"BUILD_RECEIPT_MISSING"}),
        json!({"code":"BUILD_RECEIPT_MISSING","retry_proof":{}}),
        json!({"code":"BUILD_RECEIPT_MISSING","retry_proof":proof}),
    ];
    let recovery = std::thread::spawn(move || {
        for reply in replies {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = std::io::BufReader::new(stream);
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.strip_prefix("Content-Length: ") {
                    length = value.trim().parse::<usize>().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let request: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(request["operation"], "native-reconcile");
            let reply = reply.to_string();
            write!(
                reader.get_mut(),
                "HTTP/1.1 409 Conflict\r\nContent-Length: {}\r\n\r\n{}",
                reply.len(),
                reply
            )
            .unwrap();
        }
    });
    let failed = service.reconcile_runtime("demo", lost_id).unwrap();
    assert_eq!(failed["build_error"]["details"]["log_offset"], 40330);
    assert!(failed["build_error"]["details"].get("password").is_none());
    for _ in 0..2 {
        let missing = service.reconcile_runtime("demo", lost_id).unwrap();
        assert_eq!(missing["retry_safe"], false);
        assert_eq!(
            missing["build_error"]["build_failure"]["code"],
            "PAI_DSW_BUILD_UNCERTAIN"
        );
    }
    // Upgrading the configured worker must not block reading an earlier build.
    let mut upgraded_config = config.clone();
    upgraded_config["builds"]["worker_sha256"] = json!(format!("sha256:{}", "9".repeat(64)));
    let upgraded = Arc::new(
        Service::open(
            settings.clone(),
            Settings::from_slice(&serde_json::to_vec(&upgraded_config).unwrap()).unwrap(),
            sources.clone(),
            None,
            Default::default(),
        )
        .unwrap(),
    );
    let ready = upgraded.reconcile_runtime("demo", lost_id).unwrap();
    assert_eq!(ready["runtime_id"], lost_id);
    assert_eq!(ready["status"], "PREPARED");
    assert_eq!(ready["retry_safe"], true);
    assert_eq!(ready["scheduler_submitted"], false);
    assert_eq!(
        ready["build_error"]["build_failure"]["code"],
        "PAI_DSW_BUILD_UNCERTAIN"
    );
    recovery.join().unwrap();
    // A new build with the old worker was never sent, so it stays retry-safe
    // instead of being recorded as an uncertain effect.
    upgraded
        .execute_runtime("demo", lost_id, &format!("BUILD {lost_id}"))
        .unwrap();
    let stale = wait(lost_id, "PREPARED");
    assert_eq!(stale["retry_safe"], true);
    assert_eq!(stale["build_error"]["code"], "NATIVE_BUILD_NOT_DISPATCHED");
}

fn check_materializer_diagnostics(
    settings: &read_config::Settings,
    base: &Value,
    sources: Arc<sources::Service>,
    id: &str,
    original: &Runtime,
    root: &std::path::Path,
) {
    let binary = root.join("materializer-ssh");
    fs::write(&binary, b"#!/bin/sh\ncat >/dev/null\nprintf 'access denied password=materializer-canary' >&2\nexit 255\n").unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
    let mut config = base.clone();
    config["builds"]["slurm"] = json!({"pai-h200":{
        "target":{"backend_id":"pai-h200","profile_sha256":ml_exp_core::slurm::helper::sha256_bytes(b"profile"),"storage_scope":ml_exp_core::slurm::helper::sha256_bytes(b"scope")},
        "ssh_alias":"fixture-gateway",
        "ssh":{"binary":binary,"config_file":root.join("ssh-config"),"known_hosts_file":root.join("known-hosts"),
            "helper_path":"/bin/worker","helper_config_path":"/config/materializer.json","local_cwd":root,
            "agent_socket":null,"timeout_seconds":30,"connect_timeout_seconds":5}
    }});
    let open = || {
        Service::open(
            settings.clone(),
            Settings::from_slice(&serde_json::to_vec(&config).unwrap()).unwrap(),
            sources.clone(),
            None,
            Default::default(),
        )
        .unwrap()
    };
    let service = open();
    // The build thread that published READY may still hold the metadata lock.
    let materialize = || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let result = service.materialize_runtime("demo", id, true);
            match &result {
                Err(error)
                    if error.code == "CLOUD_RUN_BUSY" && std::time::Instant::now() < deadline =>
                {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                _ => return result,
            }
        }
    };
    assert_eq!(materialize().unwrap_err().code, "OCI_TO_SIF_UNCONFIRMED");
    let failed = open().runtime_status("demo", id).unwrap();
    assert_eq!(failed["status"], "RECONCILE_REQUIRED");
    assert_eq!(failed["scheduler_submitted"], false);
    assert_eq!(failed["build_error"]["phase"], "OCI_TO_SIF");
    assert_eq!(failed["build_error"]["process"]["exit_code"], 255);
    assert!(
        failed["build_error"]["process"]["stderr"]["text"]
            .as_str()
            .unwrap()
            .contains("access denied")
    );
    assert!(!failed.to_string().contains("materializer-canary"));

    let reply = json!({"runtime":{"runtime_id":id,"source_id":original.source_id,
        "image_id":original.image.rsplit_once('@').unwrap().1,"sif_path":"/shared/test.sif",
        "sif_sha256":ml_exp_core::slurm::helper::sha256_bytes(b"SIF"),"oci_image":original.image,
        "source_dir":"/workspace","worker":{"program":"/worker","base_environment":{},"runtime_environment":[]}}});
    private(&root.join("materializer-reply.json"), &reply);
    fs::write(
        &binary,
        b"#!/bin/sh\ncat >/dev/null\ncat materializer-reply.json\n",
    )
    .unwrap();
    let ready = materialize().unwrap();
    assert_eq!(ready["status"], "READY");
    assert!(ready["build_error"].is_null());
}
