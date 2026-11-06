use super::*;
use ml_exp_core::{
    cloud::launch::{self, Input},
    datasets::{
        self, Mount, Receipt,
        import::{Config, Source},
    },
    slurm::helper::Target,
};
/// What docs/data.md warns of: DATA_DIR made before the store exists leaves a root with no
/// marker, and the store refuses it. Made inside a store that exists, it does no harm.
#[test]
fn a_data_dir_made_before_the_store_blocks_it_and_one_made_after_does_not() {
    let temp = tempfile::tempdir().unwrap();
    let tree = temp.path().join("source");
    std::fs::create_dir(&tree).unwrap();
    std::fs::write(tree.join("data.bin"), b"delivered").unwrap();
    let archive = temp.path().join("data.tar");
    let mut file = std::fs::File::create(&archive).unwrap();
    let (sha256, archive_bytes, files) =
        ml_exp_io::artifacts::pack_data(&Directory::open(&tree, false).unwrap(), &mut file)
            .unwrap();
    let root = temp.path().join("datasets");
    let config = Config {
        root: root.to_str().unwrap().into(),
        target: Target {
            backend_id: "pai-h200".into(),
            profile_sha256: format!("sha256:{}", "a".repeat(64)),
            storage_scope: format!("sha256:{}", "b".repeat(64)),
        },
        project: "demo".into(),
        definition: datasets::Definition {
            sha256,
            archive_bytes,
            files,
        },
        source: Source::File {
            path: archive.to_str().unwrap().into(),
        },
        timeout_seconds: 30,
    };
    let data_dir = root.join("demo");

    std::fs::create_dir_all(&data_dir).unwrap();
    assert!(cache(&config).is_err());
    assert!(!root.join(".native-data.json").exists());

    std::fs::remove_dir_all(&root).unwrap();
    cache(&config).unwrap();
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(data_dir.join("mine.bin"), b"not delivered").unwrap();
    cache(&config).unwrap();
    let receipt =
        data_import::materialize(&config, &AtomicBool::new(false), &mut |_| Ok(())).unwrap();
    assert_eq!(
        std::fs::read(std::path::Path::new(&receipt.root).join("data.bin")).unwrap(),
        b"delivered"
    );
    assert_eq!(
        std::fs::read(data_dir.join("mine.bin")).unwrap(),
        b"not delivered"
    );
}

#[test]
fn a_cloud_input_is_imported_once_into_shared_storage_and_reused_without_the_source_archive() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("native/runs");
    let work = temp.path().join("code");
    let tree = temp.path().join("source");
    for path in [&work, &tree] {
        std::fs::create_dir(path).unwrap();
    }
    std::fs::write(tree.join("data.bin"), b"immutable cloud input").unwrap();
    let archive = temp.path().join("data.tar");
    let mut file = std::fs::File::create(&archive).unwrap();
    let (sha256, archive_bytes, files) =
        ml_exp_io::artifacts::pack_data(&Directory::open(&tree, false).unwrap(), &mut file)
            .unwrap();
    let definition = datasets::Definition {
        sha256,
        archive_bytes,
        files,
    };
    let data_root = temp.path().join("native/inputs/datasets");
    let target = Target {
        backend_id: "pai-h200".into(),
        profile_sha256: format!("sha256:{}", "a".repeat(64)),
        storage_scope: format!("sha256:{}", "b".repeat(64)),
    };
    let receipt = Receipt {
        contract: datasets::CONTRACT.into(),
        project: "demo".into(),
        target: target.clone(),
        asset_id: definition.asset_id(),
        definition: definition.clone(),
        root: format!(
            "{}/demo/{}/tree",
            data_root.display(),
            definition.asset_id()
        ),
    };
    let command = vec![
        "/bin/sh".into(),
        "-c".into(),
        "test -d \"$DATA_DIR\"; test -n \"$DATASETS_JSON\"; printf done > \"$OUTPUT_DIR/result\""
            .into(),
    ];
    let mut package = launch::compile(Input {
        root: root.to_str().unwrap(),
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        workdir: work.to_str().unwrap(),
        argv: &command,
        timeout_seconds: 3,
        worker_sha256: &format!("sha256:{}", "c".repeat(64)),
        storage: launch::StorageLayout {
            mount: temp.path().into(),
            runs: root.clone(),
            datasets: data_root.clone(),
        },
        // Nothing listens here, so delivery defers without leaving this host.
        channel: ml_exp_core::cloud::exchange::Channel {
            url: "https://127.0.0.1:1/api/worker-exchange/demo/train/attempt-001".into(),
            token: "d".repeat(64),
        },
    })
    .unwrap();
    package
        .attach(
            data_root.to_str().unwrap(),
            vec![Mount {
                name: "training".into(),
                receipt: receipt.clone(),
            }],
            None,
        )
        .unwrap();
    package.imports = vec![Config {
        root: data_root.to_str().unwrap().into(),
        target,
        project: "demo".into(),
        definition,
        source: Source::File {
            path: archive.to_str().unwrap().into(),
        },
        timeout_seconds: 30,
    }];
    let mut bad_storage = package.storage.clone();
    bad_storage.datasets = temp.path().join("wrong/cache");
    assert!(scoped_cache(&package.imports[0], &bad_storage).is_err());
    assert!(!temp.path().join("wrong").exists());
    scoped_cache(&package.imports[0], &package.storage).unwrap();
    assert!(!package.root.exists()); // A copy never initializes a training Run.
    crate::cloud::prepare(&package, &BTreeMap::new()).unwrap();
    materialize(&package, &AtomicBool::new(false)).unwrap();
    assert_eq!(read(&package).unwrap()[0].phase, Phase::Ready);
    assert_eq!(
        std::fs::read(std::path::Path::new(&receipt.root).join("data.bin")).unwrap(),
        b"immutable cloud input"
    );
    std::fs::remove_file(archive).unwrap();
    // The next compute node mounts this same cache under another st_dev.
    let marker = data_root.join(".native-data.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
    value["identity"][0] = serde_json::json!(value["identity"][0].as_u64().unwrap() ^ 1);
    std::fs::write(&marker, serde_json::to_vec(&value).unwrap()).unwrap();
    materialize(&package, &AtomicBool::new(false)).unwrap();
    // A cache rejection must leave a durable failure without starting training.
    value["identity"][1] = serde_json::json!(value["identity"][1].as_u64().unwrap() ^ 1);
    std::fs::write(&marker, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(crate::cloud::run(&package, &BTreeMap::new(), &AtomicBool::new(false)).is_err());
    assert_eq!(
        crate::managed::inspect(&package.config).unwrap().phase,
        ml_exp_core::managed::evidence::Phase::Prepared
    );
    let failed = read(&package).unwrap();
    assert_eq!(failed[0].phase, Phase::Failed);
    assert_eq!(failed[0].code.as_deref(), Some("DATA_CACHE_UNAVAILABLE"));
    assert!(!package.config.output_path().join("result").exists());
    value["identity"][1] = serde_json::json!(value["identity"][1].as_u64().unwrap() ^ 1);
    std::fs::write(&marker, serde_json::to_vec(&value).unwrap()).unwrap();
    materialize(&package, &AtomicBool::new(false)).unwrap();
    assert!(
        crate::managed::run(&package.config, &AtomicBool::new(false))
            .unwrap()
            .receipt
            .unwrap()
            .succeeded()
    );
    assert!(package.config.output_path().join("result").is_file());
    assert!(!package.config.output_path().join("data.bin").exists());
}
