use super::*;
use ml_exp_core::slurm::helper::Target;
use std::{fs, os::unix::fs::PermissionsExt};
#[test]
fn image_dataset_resumes_durable_parts_verifies_tree_and_reuses_ready_without_original_payload() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source");
    let cache = temp.path().join("cache");
    for path in [&source, &cache] {
        fs::create_dir(path).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    fs::write(source.join("data.bin"), vec![42; 17 * 1024 * 1024]).unwrap();
    let archive_path = temp.path().join("archive.tar");
    let mut archive = fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&archive_path)
        .unwrap();
    let (sha256, archive_bytes, files) =
        ml_exp_io::artifacts::pack_data(&Directory::open(&source, false).unwrap(), &mut archive)
            .unwrap();
    let config = Config {
        root: cache.to_str().unwrap().into(),
        target: Target {
            backend_id: "cloud".into(),
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
            path: archive_path.to_str().unwrap().into(),
        },
        timeout_seconds: 60,
    };
    initialize(&config).unwrap();
    assert!(initialize(&config).is_err());
    let flag = AtomicBool::new(false);
    let error = materialize(&config, &flag, &mut |p| {
        if p.completed_parts == 1 && p.stage != Stage::Failed {
            Err(io::Error::other("simulated observer loss"))
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert_eq!(error.code, "PROGRESS_UNAVAILABLE");
    let mut progress = Vec::new();
    let receipt = materialize(&config, &flag, &mut |p| {
        progress.push(p.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(progress[0].resumed_parts, 1);
    assert_eq!(progress.last().unwrap().stage, Stage::Ready);
    assert_eq!(progress.last().unwrap().received_bytes, archive_bytes);
    assert_eq!(
        fs::read(Path::new(&receipt.root).join("data.bin"))
            .unwrap()
            .len(),
        17 * 1024 * 1024
    );
    fs::remove_file(&archive_path).unwrap();
    let mut phases = Vec::new();
    assert_eq!(
        materialize(&config, &flag, &mut |p| {
            phases.push(p.stage);
            Ok(())
        })
        .unwrap(),
        receipt
    );
    assert!(!phases.contains(&Stage::Downloading));
    let asset = cache.join("demo").join(&receipt.asset_id);
    assert!(fs::read_dir(asset.join("parts")).unwrap().next().is_none());
    assert!(!asset.join("archive.tar").exists());
    let mut invalid = config.clone();
    invalid.source = Source::Https {
        url: "https://127.0.0.1/data".into(),
        allowed_hosts: std::collections::BTreeSet::from(["127.0.0.1".into()]),
    };
    assert!(invalid.validate().is_err());
}
