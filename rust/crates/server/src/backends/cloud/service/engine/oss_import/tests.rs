use super::*;
use crate::{backends::cloud::adapter::Transport, sources};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct Provider {
    creates: AtomicUsize,
    actions: Mutex<Vec<String>>,
}
impl Transport for Provider {
    fn request(
        &self,
        request: &ml_exp_core::cloud::Request,
        _: &AtomicBool,
    ) -> cloud::Result<Value> {
        self.actions.lock().unwrap().push(request.action.clone());
        if request.action == "CreateJob" {
            self.creates.fetch_add(1, Ordering::SeqCst);
            return Err(cloud::Error {
                code: "REPLY_LOST".into(),
                http_status: None,
                uncertain: true,
            });
        }
        Ok(json!({"Jobs":[],"TotalCount":0}))
    }
}
struct Fixture {
    temp: tempfile::TempDir,
    settings: read_config::Settings,
    sources: Arc<sources::Service>,
    config: Vec<u8>,
    provider: Arc<Provider>,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        read_config::initialize(&registry).unwrap();
        let project = temp.path().join("project.json");
        let runs = temp.path().join("runs");
        std::fs::create_dir(&runs).unwrap();
        ml_exp_io::write_atomic(&project,&encode(&json!({"schema_version":1,"project":"demo","title":"Import fixture","run_roots":["runs"]})).unwrap(),true).unwrap();
        let settings = read_config::Settings {
            schema_version: 1,
            registry_root: registry,
            projects: vec![read_config::Project {
                project: "demo".into(),
                project_file: project,
                run_roots: vec![runs],
                runs: vec![],
                discover_runs: None,
            }],
            source_policy: Some(Default::default()),
        };
        sources::Service::initialize(&settings).unwrap();
        let sources = Arc::new(sources::Service::open(&settings, None).unwrap());
        let key = temp.path().join("key");
        let credentials = temp.path().join("objects");
        ml_exp_io::write_atomic(&key, b"synthetic-exchange-key-0000000000000000000000", true)
            .unwrap();
        ml_exp_io::write_atomic(
            &credentials,
            br#"{"access_key_id":"synthetic","access_key_secret":"private"}"#,
            true,
        )
        .unwrap();
        let profile = json!({"provider":"pai","region":"us-southeast-1","workspace_id":"253","quota_id":"quota1","max_gpus":0,"max_cpus":4,"max_memory_gb":8,"storage":{"cpfs_uri":"bmcpfs://cpfs-example.us-southeast-1.cpfs.aliyuncs.com/","workspace_user":"alice","role_arn":"acs:ram::12345:role/pai-data-alice"}});
        let config=encode(&json!({"schema_version":1,"enabled":true,"exchange":{"public_url":"https://api.example","signing_key_file":key},"artifacts":{"endpoint":"oss-us-southeast-1.aliyuncs.com","bucket":"bucket","prefix":"users/alice/ml-expd/","credentials_file":credentials},"backends":[{"executor":"pai","profile":profile,"collector":profile,"run_root":"/mnt/workspace/ml-expd/runs","credentials_file":temp.path().join("not-read"),"runtimes":{},"oss_import_image":format!("docker.io/library/python@sha256:{}","1".repeat(64))}]})).unwrap();
        Service::initialize(&settings).unwrap();
        Self {
            temp,
            settings,
            sources,
            config,
            provider: Arc::new(Provider::default()),
        }
    }
    fn open(&self) -> Service {
        Service::open_with_transport(
            self.settings.clone(),
            Settings::from_slice(&self.config).unwrap(),
            self.sources.clone(),
            None,
            Default::default(),
            self.provider.clone(),
        )
        .unwrap()
    }
    fn source(&self) -> Import {
        Import {
            object_key: "users/alice/datasets/train.tar".into(),
            sha256: "a".repeat(64),
            archive_bytes: 2048,
        }
    }
}
#[test]
fn lost_submit_response_repeats_and_restart_never_create_a_second_cpu_job() {
    let fixture = Fixture::new();
    let source = fixture.source();
    for _ in 0..3 {
        let service = fixture.open();
        let view = service
            .import_oss_dataset("demo", "pai", source.clone())
            .unwrap();
        assert_eq!(view["contract"], "oss-import.v2");
        assert_eq!(view["archive_verified"], false);
        let saved = service
            .oss_saved("demo", "pai", &source.asset_id())
            .unwrap()
            .unwrap();
        assert_eq!(
            (saved.job.gpus, saved.job.cpus, saved.job.memory_gb),
            (0, 2, 4)
        );
        assert!(saved.job.read_only_storage);
        assert!(!encode(&view).unwrap().windows(7).any(|b| b == b"private"));
        assert!(!encode(&view).unwrap().windows(5).any(|b| b == b"token"));
        assert!(
            service
                .dataset_download("demo", "pai", &source.asset_id(), 0)
                .is_err()
        );
    }
    assert_eq!(fixture.provider.creates.load(Ordering::SeqCst), 1);
}
#[test]
fn failed_or_wrong_scope_callbacks_cannot_publish_a_dataset() {
    let fixture = Fixture::new();
    let service = fixture.open();
    let source = fixture.source();
    service
        .import_oss_dataset("demo", "pai", source.clone())
        .unwrap();
    let asset = source.asset_id();
    assert!(
        service
            .oss_import_exchange("demo", "pai", &asset, "wrong", "oss-failed", json!({}))
            .is_err()
    );
    let channel = service.oss_channel("demo", "pai", &asset).unwrap();
    let saved = service.oss_saved("demo", "pai", &asset).unwrap().unwrap();
    let failed = service
        .oss_import_exchange(
            "demo",
            "pai",
            &asset,
            &channel.token,
            "oss-failed",
            json!({"request_id":saved.request_id,"code":"ARCHIVE_SHA256_MISMATCH"}),
        )
        .unwrap();
    assert_eq!(failed["status"], "FAILED");
    assert_eq!(
        service.dataset_state("demo", "pai", &asset).unwrap()["status"],
        "FAILED"
    );
    service.import_oss_dataset("demo", "pai", source).unwrap();
    assert_eq!(fixture.provider.creates.load(Ordering::SeqCst), 1);
}
#[test]
fn a_bound_report_publishes_once_without_reading_any_object_body() {
    let fixture = Fixture::new();
    let service = fixture.open();
    let source = fixture.source();
    service
        .import_oss_dataset("demo", "pai", source.clone())
        .unwrap();
    let asset = source.asset_id();
    let mut saved = service.oss_saved("demo", "pai", &asset).unwrap().unwrap();
    saved.status = "COPYING_VERIFYING".into();
    saved.copied.insert(
        0,
        ObjectIdentity {
            bytes: 2048,
            etag: "copied-object-etag".into(),
        },
    );
    service.oss_write(&saved).unwrap();
    let channel = service.oss_channel("demo", "pai", &asset).unwrap();
    let body = json!({"request_id":saved.request_id,"definition":{"sha256":source.sha256,"archive_bytes":2048,"files":[{"path":"train.bin","bytes":10,"sha256":"b".repeat(64)}]},"parts":{"0":{"bytes":2048,"sha256":"c".repeat(64)}}});
    let mut wrong = body.clone();
    wrong["definition"]["sha256"] = json!("f".repeat(64));
    assert!(
        service
            .oss_import_exchange("demo", "pai", &asset, &channel.token, "oss-complete", wrong)
            .is_err()
    );
    for _ in 0..2 {
        assert_eq!(
            service
                .oss_import_exchange(
                    "demo",
                    "pai",
                    &asset,
                    &channel.token,
                    "oss-complete",
                    body.clone()
                )
                .unwrap()["archive_verified"],
            true
        );
    }
    assert_eq!(
        fixture.open().dataset_state("demo", "pai", &asset).unwrap()["archive_verified"],
        true
    );
    // Existing verified bytes are reused, and no provider is contacted.
    let before = fixture.provider.actions.lock().unwrap().len();
    assert_eq!(
        fixture
            .open()
            .import_oss_dataset("demo", "pai", source)
            .unwrap()["archive_verified"],
        true
    );
    assert_eq!(fixture.provider.actions.lock().unwrap().len(), before);
    // A crash after the dataset commit but before the progress commit cannot
    // hide an accepted dataset or let a delayed failure demote it.
    let mut stale = service.oss_saved("demo", "pai", &asset).unwrap().unwrap();
    stale.status = "COPYING_VERIFYING".into();
    service.oss_write(&stale).unwrap();
    assert_eq!(
        fixture.open().dataset_state("demo", "pai", &asset).unwrap()["archive_verified"],
        true
    );
    let reconciled = service
        .oss_import_exchange(
            "demo",
            "pai",
            &asset,
            &channel.token,
            "oss-failed",
            json!({"request_id":stale.request_id,"code":"IMPORT_TRANSPORT_FAILED"}),
        )
        .unwrap();
    assert_eq!(reconciled["status"], "ARCHIVED");
    assert!(fixture.temp.path().exists());
}
#[test]
fn a_changed_source_is_refused_before_any_new_effect() {
    let fixture = Fixture::new();
    let service = fixture.open();
    let source = fixture.source();
    service
        .import_oss_dataset("demo", "pai", source.clone())
        .unwrap();
    let mut changed = source.clone();
    changed.object_key = "users/alice/datasets/other.tar".into();
    assert!(service.import_oss_dataset("demo", "pai", changed).is_err());
    assert_eq!(fixture.provider.creates.load(Ordering::SeqCst), 1);
}

#[test]
fn import_only_configuration_requires_a_pinned_image_and_a_cpu_collector() {
    let fixture = Fixture::new();
    assert!(Settings::from_slice(&fixture.config).is_ok());
    for (key, value) in [
        ("oss_import_image", json!("docker.io/library/python:latest")),
        ("collector", Value::Null),
    ] {
        let mut config: Value = serde_json::from_slice(&fixture.config).unwrap();
        config["backends"][0][key] = value;
        assert!(Settings::from_slice(&encode(&config).unwrap()).is_err());
    }
    let mut config: Value = serde_json::from_slice(&fixture.config).unwrap();
    config["artifacts"]["endpoint"] = json!("oss-cn-hangzhou.aliyuncs.com");
    let service = Service::open_with_transport(
        fixture.settings.clone(),
        Settings::from_slice(&encode(&config).unwrap()).unwrap(),
        fixture.sources.clone(),
        None,
        Default::default(),
        fixture.provider.clone(),
    )
    .unwrap();
    assert!(
        service
            .import_oss_dataset("demo", "pai", fixture.source())
            .is_err()
    );
    assert_eq!(fixture.provider.creates.load(Ordering::SeqCst), 0);
}
