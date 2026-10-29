use super::*;
use ml_exp_core::{
    cloud::launch::{self, Input},
    datasets::{
        self, Mount, Receipt,
        import::{Config, Source},
    },
    slurm::helper::Target,
};
#[test]
fn a_cloud_input_is_imported_once_into_shared_storage_and_reused_without_the_source_archive() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("runs");
    let work = temp.path().join("code");
    let tree = temp.path().join("source");
    for path in [&root, &work, &tree] {
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
    let data_root = temp.path().join("datasets");
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
    crate::cloud::prepare(&package, &BTreeMap::new()).unwrap();
    materialize(&package, &AtomicBool::new(false)).unwrap();
    assert_eq!(read(&package).unwrap()[0].phase, Phase::Ready);
    assert_eq!(
        std::fs::read(std::path::Path::new(&receipt.root).join("data.bin")).unwrap(),
        b"immutable cloud input"
    );
    std::fs::remove_file(archive).unwrap();
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
