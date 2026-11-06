//! Worker and server agreement through the real entry points on each side. One test
//! binary links both. Inputs carry non-ASCII text and floats, because ASCII float-free
//! values cannot detect a mismatch. Literals are pinned at a fixed root so that a
//! shared regression cannot pass by agreeing with itself; the filesystem cases run the
//! same builders at a temporary root.
#![cfg(unix)]
use ml_exp_core::{
    artifacts::FileReceipt,
    canonical,
    checkpoints::{self, Context},
    cloud::{
        exchange::{self, Ack, Packet},
        launch as cloud_launch,
    },
    data_preparation, launch,
    slurm::{self, helper, managed as slurm_managed, staging},
};
use ml_exp_server::checkpoints::Store;
use ml_exp_worker::{cloud_exchange, managed};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    sync::{Mutex, atomic::AtomicBool},
};

const FIXED_ROOT: &str = "/srv/ml-expd-agreement";
const LABEL: &str = "基础\u{2028}模型😀";
const ARGUMENT: &str = "--名称=模型😀";
const CLOUD_SCOPE: &str = "存储/范围";

fn repeated(prefix: &str, digit: &str) -> String {
    format!("{prefix}{}", digit.repeat(64))
}
fn image_id() -> String {
    repeated("sha256:", "b")
}
fn source_id() -> String {
    repeated("source.", "a")
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn ready() -> Value {
    json!({"step":42,"files":[
        {"path":"checkpoints/step-42/模型.pt","bytes":7,"sha256":sha(b"weights")}
    ]})
}
fn context(run_id: &str, storage_scope: &str, state_root: String) -> Context {
    Context {
        project: "demo".into(),
        run_id: run_id.into(),
        attempt_id: "attempt-001".into(),
        source_id: source_id(),
        image_id: image_id(),
        storage_scope: storage_scope.into(),
        state_root,
    }
}
fn scope() -> helper::Scope {
    helper::Scope {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
    }
}
fn runtime() -> slurm_managed::Runtime {
    slurm_managed::Runtime {
        program: "/opt/ml-expd/worker".into(),
        base_environment: BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
        runtime_environment: vec![],
    }
}
/// The frozen Slurm Attempt as the server authors it.
fn attempt(root: &str) -> Value {
    let code = format!("{root}/代码");
    json!({"schema_version":1,"project":"demo","run_id":"train","attempt_id":"attempt-001",
        "image_id":image_id(),
        "backend":{"kind":"slurm","ssh_alias":"test","partition":"h100","account":"science",
            "qos":"normal","gres":"gpu:h100:1","time":"00:05:00","source_dir":format!("{root}/source"),
            "sif_path":format!("{root}/image.sif"),"mount_root":root},
        "storage":{"run_dir":format!("{root}/runs/demo/train"),"project_data_root":format!("{root}/data/demo")},
        "resources":{"gpus":1,"cpus":2,"memory_gb":4},
        "execution":{"source_mount":code,"workdir":code,"managed_io":true,"env":{"LABEL":LABEL}},
        "command":["python","训练.py",ARGUMENT],
        "parameters":{"lr":1e-7,"scale":1e16,"zero":-0.0,"exact":1.5,"count":3}})
}
/// Resuming a registered checkpoint brings a float into the sealed configuration.
fn resuming_attempt(root: &str) -> Value {
    let storage_scope = repeated("sha256:", "d");
    let state_root = format!("{root}/runs/demo/earlier/attempts/attempt-001/state");
    let mut registered = checkpoints::register(
        &checkpoints::document(&context("earlier", &storage_scope, state_root), &ready()).unwrap(),
        "2026-01-01T00:00:00Z",
    )
    .unwrap();
    registered["validation_loss"] = serde_json::from_str("1E-7").unwrap();
    let mut document = attempt(root);
    document["resume_from"] = json!({"run_id":"earlier","attempt_id":"attempt-001",
        "checkpoint_id":registered["checkpoint_id"]});
    document["native_checkpoint"] = registered;
    document
}
fn preparing_attempt(root: &str) -> Value {
    let mut document = attempt(root);
    document["backend"]["oci_image"] = json!(format!(
        "registry.example/team/image@sha256:{}",
        "b".repeat(64)
    ));
    document["source_id"] = json!(source_id());
    document["data_preparation"] = json!({"script":"获取.sh","interpreter":"/bin/sh",
        "arguments":[ARGUMENT],"timeout_seconds":30});
    document
}
/// The path native submission authors, from the document before it is encoded.
fn authored(document: &Value) -> slurm_managed::Package {
    let spec: slurm::JobSpec = serde_json::from_value(json!({
        "run_id":document["run_id"],"attempt_id":document["attempt_id"],
        "image_id":document["image_id"],"backend":document["backend"],
        "storage":document["storage"],"resources":document["resources"],
        "execution":document["execution"],"command":document["command"]}))
    .unwrap();
    slurm_managed::plan_for_document(&scope(), &spec, &runtime(), document, &[]).unwrap()
}
/// The path the worker's helper takes, from the stored Attempt bytes alone.
fn staged(root: &str, attempt_bytes: &[u8]) -> slurm_managed::Package {
    let intent = helper::FrozenIntent {
        target: helper::Target {
            backend_id: "slurm-a".into(),
            profile_sha256: repeated("sha256:", "c"),
            storage_scope: repeated("sha256:", "d"),
        },
        scope: scope(),
        staging: staging::Requirement {
            runs_root: format!("{root}/runs"),
            worker: runtime(),
        },
        submission_token: "0123456789abcdef0123456789abcdef".into(),
        scheduler_name: slurm::scheduler_job_name("train", "attempt-001").unwrap(),
        manifest_sha256: helper::sha256_bytes(attempt_bytes),
        script_sha256: helper::sha256_bytes(b"#!/bin/sh\n"),
    };
    staging::plan(&intent, attempt_bytes)
        .unwrap()
        .unwrap()
        .managed
        .unwrap()
}
fn private_directory(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn private_file(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

#[test]
fn the_helper_re_derives_exactly_what_the_server_authored() {
    let document = resuming_attempt(FIXED_ROOT);
    let bytes = canonical::bytes(&document).unwrap();
    assert!(
        std::str::from_utf8(&bytes)
            .unwrap()
            .contains(r#""parameters":{"count":3,"exact":1.5,"lr":1e-7,"scale":1e16,"zero":0.0}"#)
    );
    assert_eq!(
        sha(&bytes),
        "3566264a8f5cce87320c7b3146f69d6acd7b675ac799c1757642b1a80b0eb9bf"
    );
    let server = authored(&document);
    let worker = staged(FIXED_ROOT, &bytes);
    assert!(server == worker);
    assert_eq!(
        server.config_bytes().unwrap(),
        worker.config_bytes().unwrap()
    );
    let manifest = server.manifest_bytes().unwrap();
    assert_eq!(manifest, worker.manifest_bytes().unwrap());
    assert_eq!(
        sha(&manifest),
        "340921a785935a1ab6ddc2cad45154ac4006a61aecad4afb66bf1ff3dd009ee0"
    );
    assert_eq!(
        server.config.digest().unwrap(),
        "489ebbbd7b6574b812d7eb3068c6c1e66a1eb801540a4847c6603773586ee98b"
    );
    assert_eq!(server.config.digest(), worker.config.digest());
    let accepted = launch::verify_binding(&manifest, &server.config.binding().unwrap()).unwrap();
    assert_eq!(accepted.environment["LABEL"], LABEL);
    assert_eq!(accepted.argv[2], ARGUMENT);
}

#[test]
fn the_worker_accepts_the_server_manifest_and_reports_its_config_digest() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().to_str().unwrap();
    let package = authored(&resuming_attempt(root));
    let attempt_directory = package.config.manifest_path.parent().unwrap();
    private_directory(attempt_directory);
    private_directory(&package.config.workdir);
    let manifest = package.manifest_bytes().unwrap();
    private_file(&package.config.manifest_path, &manifest);
    let config_bytes = package.config_bytes().unwrap();
    let config_path = attempt_directory.join(slurm_managed::CONFIG_FILE);
    private_file(&config_path, &config_bytes);

    let loaded = managed::load_config(&config_path, Some(sha(&config_bytes).as_str())).unwrap();
    assert_eq!(loaded.digest(), package.config.digest());
    let prepared = managed::initialize(&loaded).unwrap();
    assert_eq!(prepared.manifest_sha256, sha(&manifest));
    let inspected = managed::inspect(&loaded).unwrap();
    assert_eq!(inspected.manifest_sha256, sha(&manifest));
    assert_eq!(
        inspected.identity,
        package.config.binding().unwrap().identity
    );
    // The diagnostic route compares this ledger binding with the server's own digest.
    let ledger = fs::read(package.config.attempt_root.join(".managed-launch.lock")).unwrap();
    let record: Value =
        serde_json::from_slice(ledger.split(|c| *c == b'\n').next().unwrap()).unwrap();
    assert_eq!(
        record["binding_sha256"],
        package.config.digest().unwrap().as_str()
    );
}

fn cloud_package(root: &str, checkpoint: Option<Context>) -> cloud_launch::Package {
    let command = [
        "python".to_owned(),
        "训练.py".to_owned(),
        ARGUMENT.to_owned(),
    ];
    let mut package = cloud_launch::compile(cloud_launch::Input {
        root: &format!("{root}/runs"),
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        workdir: &format!("{root}/代码"),
        argv: &command,
        timeout_seconds: 60,
        worker_sha256: &repeated("sha256:", "e"),
        storage: cloud_launch::StorageLayout {
            mount: root.into(),
            runs: format!("{root}/runs").into(),
            datasets: format!("{root}/datasets").into(),
        },
        channel: exchange::Channel {
            url: "https://api.example/ml-expd/api/worker-exchange/demo/train/attempt-001".into(),
            token: "b".repeat(64),
        },
    })
    .unwrap();
    package
        .manifest
        .environment
        .insert("LABEL".into(), LABEL.into());
    package
        .attach(&format!("{root}/datasets"), Vec::new(), None)
        .unwrap();
    package.checkpoint_context = checkpoint;
    package.validate().unwrap();
    package
}
fn cloud_context(root: &str) -> Context {
    context(
        "train",
        CLOUD_SCOPE,
        format!("{root}/runs/demo/train/attempt-001/state"),
    )
}
#[derive(Default)]
struct Capture(Mutex<Option<Value>>);
impl cloud_exchange::Transport for Capture {
    fn send(&self, _: &exchange::Channel, packet: &Packet) -> managed::Result<Ack> {
        *self.0.lock().unwrap() = packet.checkpoint.clone();
        Ack::for_packet(packet).map_err(|_| managed::Error::Binding)
    }
}

#[test]
fn a_cloud_manifest_and_config_stay_pinned() {
    let package = cloud_package(FIXED_ROOT, None);
    let manifest = package.manifest.encode().unwrap();
    assert_eq!(
        sha(&manifest),
        "f5d0655dd9e2a4d667bdb5769b694cf57841e01c4b0833cb3962dac33bea5546"
    );
    assert_eq!(
        package.config.digest().unwrap(),
        "70b85c091c50eb15df5cab9084f8384a40f34e110a28fdf1758a818735d4c6f3"
    );
    launch::verify_binding(&manifest, &package.config.binding().unwrap()).unwrap();
}

#[test]
fn a_worker_checkpoint_document_registers_on_the_server_with_the_same_id() {
    assert_eq!(
        checkpoints::document(&cloud_context(FIXED_ROOT), &ready()).unwrap()["checkpoint_id"],
        "checkpoint.181e9b979697eda62968bdfa3ff4e4e5a884ebb3d48eb6979f21d4071b9f3001"
    );
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().to_str().unwrap();
    fs::create_dir(format!("{root}/runs")).unwrap();
    fs::create_dir(format!("{root}/代码")).unwrap();
    let context = cloud_context(root);
    let package = cloud_package(root, Some(context.clone()));
    ml_exp_worker::cloud::prepare(&package, &BTreeMap::new()).unwrap();
    let state = Path::new(&context.state_root);
    fs::create_dir_all(state.join("checkpoints/step-42")).unwrap();
    fs::write(state.join("checkpoints/step-42/模型.pt"), b"weights").unwrap();
    fs::write(
        state.join("checkpoint.ready.json"),
        serde_json::to_vec(&ready()).unwrap(),
    )
    .unwrap();

    let sender = Capture::default();
    cloud_exchange::once(&package, &sender).unwrap();
    let sent = sender.0.lock().unwrap().clone().unwrap();
    assert_eq!(sent, checkpoints::document(&context, &ready()).unwrap());

    // The exchange route validates the packet's document, then registers its files.
    let (_, manifest) = checkpoints::validate_document(&sent, Some(&context), None).unwrap();
    let registry = temp.path().join("registry");
    private_directory(&registry);
    let store = Store::open(&registry).unwrap();
    let registered = store
        .register(
            &context,
            &json!({"step":manifest.step,"files":sent["files"]}),
            "2026-01-01T00:00:00Z",
        )
        .unwrap();
    assert_eq!(registered["checkpoint_id"], sent["checkpoint_id"]);
    let id = sent["checkpoint_id"].as_str().unwrap();
    assert_eq!(store.read(&context, id).unwrap().unwrap(), registered);
    assert_eq!(store.list(&context).unwrap(), vec![registered]);
}

#[test]
fn a_preparation_plan_id_and_content_hash_agree() {
    let pinned = authored(&preparing_attempt(FIXED_ROOT));
    assert_eq!(
        pinned
            .config
            .data_preparation
            .as_ref()
            .unwrap()
            .id()
            .unwrap(),
        "preparation.a49f10ac81a02d5ba802814fc0b80bf948d51192d549abdeddb91ba7e6bf2820"
    );
    let expected_files = [FileReceipt {
        path: "样本.txt".into(),
        bytes: 5,
        sha256: sha(b"hello"),
    }];
    let content = data_preparation::content_hash(&expected_files).unwrap();
    assert_eq!(
        content,
        "e1d957914802bd7aefa1f645864dbea5e802ee7ea64419c8674beb337ad0735f"
    );

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().to_str().unwrap();
    let document = preparing_attempt(root);
    let server = authored(&document);
    let worker = staged(root, &canonical::bytes(&document).unwrap());
    let plan = server.config.data_preparation.clone().unwrap();
    assert_eq!(
        worker.config.data_preparation.as_ref().unwrap().id(),
        plan.id()
    );
    private_directory(Path::new(&plan.workdir));
    fs::write(
        Path::new(&plan.workdir).join("获取.sh"),
        "printf hello > \"$DATA_DIR/样本.txt\"\n",
    )
    .unwrap();
    let cancelled = AtomicBool::new(false);
    let tree = ml_exp_worker::data_preparation::prepare(&plan, true, &cancelled).unwrap();
    assert_eq!(tree, plan.tree().unwrap());
    assert_eq!(fs::read(format!("{tree}/样本.txt")).unwrap(), b"hello");
    let receipt: Value = serde_json::from_slice(
        &fs::read(Path::new(&tree).parent().unwrap().join("receipt.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["content_sha256"], content.as_str());
    assert_eq!(
        receipt["files"],
        serde_json::to_value(&expected_files).unwrap()
    );
    let saved: data_preparation::Plan = serde_json::from_value(receipt["plan"].clone()).unwrap();
    assert_eq!(saved.id(), plan.id());
    // A reuse recomputes both on the worker side and refuses any difference.
    assert_eq!(
        ml_exp_worker::data_preparation::prepare(&plan, false, &cancelled).unwrap(),
        tree
    );
}
