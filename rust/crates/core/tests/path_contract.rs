//! The paths training code and maintenance see, compared across Slurm and cloud.
use ml_exp_core::{
    artifacts::FileReceipt,
    checkpoints,
    cloud::{exchange, launch as cloud_launch},
    datasets::{self, Definition, Mount, Receipt},
    maintenance::{Plan, Selection},
    managed::{self, AdmissionError, Config},
    slurm::{self, helper, managed as slurm_managed},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

fn scope(spec: &slurm::JobSpec) -> helper::Scope {
    helper::Scope {
        project: "demo".into(),
        run_id: spec.run_id.clone(),
        attempt_id: spec.attempt_id.clone(),
    }
}
fn spec() -> slurm::JobSpec {
    let cases: Value = serde_json::from_str(include_str!("fixtures/slurm/cases.json")).unwrap();
    let mut spec: slurm::JobSpec =
        serde_json::from_value(cases["render_cases"][0]["spec"].clone()).unwrap();
    spec.storage.run_dir = format!("/shared/project/runs/demo/{}", spec.run_id);
    spec
}
fn runtime() -> slurm_managed::Runtime {
    slurm_managed::Runtime {
        program: "/opt/ml-expd/worker".into(),
        base_environment: BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
        runtime_environment: vec![],
    }
}
fn slurm_package(inputs: &[Mount]) -> slurm_managed::Package {
    let spec = spec();
    slurm_managed::plan_with_inputs(&scope(&spec), &spec, &runtime(), None, inputs).unwrap()
}
fn cloud_package(inputs: Vec<Mount>) -> cloud_launch::Package {
    let mut package = cloud_launch::compile(cloud_launch::Input {
        root: "/mnt/workspace/runs",
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        workdir: "/app",
        argv: &["python".to_owned()],
        timeout_seconds: 60,
        worker_sha256: &format!("sha256:{}", "e".repeat(64)),
        storage: cloud_launch::StorageLayout {
            mount: "/mnt/workspace".into(),
            runs: "/mnt/workspace/runs".into(),
            datasets: "/mnt/workspace/datasets".into(),
        },
        channel: exchange::Channel {
            url: "https://api.example/ml-expd/api/worker-exchange/demo/train/attempt-001".into(),
            token: "b".repeat(64),
        },
    })
    .unwrap();
    package
        .attach("/mnt/workspace/datasets", inputs, None)
        .unwrap();
    package
}
fn mount(name: &str, data_root: &str, digit: &str) -> Mount {
    let definition = Definition {
        sha256: digit.repeat(64),
        archive_bytes: 1024,
        files: vec![FileReceipt {
            path: "train.bin".into(),
            bytes: 16,
            sha256: "f".repeat(64),
        }],
    };
    let asset_id = definition.asset_id();
    Mount {
        name: name.into(),
        receipt: Receipt {
            contract: datasets::CONTRACT.into(),
            project: "demo".into(),
            target: helper::Target {
                backend_id: "backend-a".into(),
                profile_sha256: format!("sha256:{}", "c".repeat(64)),
                storage_scope: format!("sha256:{}", "d".repeat(64)),
            },
            root: format!("{data_root}/demo/{asset_id}/tree"),
            asset_id,
            definition,
        },
    }
}

#[test]
fn slurm_and_cloud_give_training_the_same_path_variables_and_layout() {
    let slurm = slurm_package(&[]).manifest.environment;
    let cloud = cloud_package(Vec::new()).manifest.environment;
    for environment in [&slurm, &cloud] {
        let names: Vec<_> = environment.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            ["DATA_DIR", "METRICS_PATH", "OUTPUT_DIR", "STATE_DIR"]
        );
        let output = Path::new(&environment["OUTPUT_DIR"]);
        assert_eq!(output.file_name().unwrap(), "outputs");
        let attempt = output.parent().unwrap();
        assert_eq!(Path::new(&environment["STATE_DIR"]), attempt.join("state"));
        assert_eq!(
            Path::new(&environment["METRICS_PATH"]),
            output.join("metrics.jsonl")
        );
    }
    assert_eq!(slurm["DATA_DIR"], spec().storage.project_data_root);
    assert_eq!(cloud["DATA_DIR"], "/mnt/workspace/datasets/demo");
}

#[test]
fn datasets_json_names_the_receipt_root_on_every_backend() {
    let first = mount("training", "/mnt/workspace/datasets", "a");
    let cloud = cloud_package(vec![first.clone()]);
    let expected = json!({"training": first.receipt.root});
    assert_eq!(
        serde_json::from_str::<Value>(&cloud.manifest.environment["DATASETS_JSON"]).unwrap(),
        expected
    );
    let first = mount("training", "/shared/data", "a");
    let slurm = slurm_package(std::slice::from_ref(&first));
    let expected = json!({"training": first.receipt.root});
    assert_eq!(
        serde_json::from_str::<Value>(&slurm.manifest.environment["DATASETS_JSON"]).unwrap(),
        expected
    );
}

#[test]
fn slurm_binds_each_source_once_read_only_at_its_own_path() {
    let first = mount("training", "/shared/data", "a");
    let same_asset = mount("training-again", "/shared/data", "a");
    let other = mount("validation", "/shared/data", "b");
    let script =
        slurm::render_job_with_datasets(&spec(), None, &[first.clone(), same_asset, other.clone()])
            .unwrap();
    let binds: Vec<_> = script.lines().filter(|l| l.contains(":ro")).collect();
    let expected: Vec<_> = [&first, &other]
        .iter()
        .map(|m| format!("  --bind {0}:{0}:ro \\", m.receipt.root))
        .collect();
    assert_eq!(binds, expected);
    for removed in [
        "/inputs",
        "/outputs",
        "BACKEND_JOB_ID",
        "INPUT_DIR",
        "OUTPUT_DIR",
    ] {
        assert!(!script.contains(removed), "{removed}");
    }
}

#[test]
fn admission_refuses_a_datasets_json_that_differs_from_the_receipts() {
    fn reseal(package: &mut slurm_managed::Package) {
        package.config.local_binding.as_mut().unwrap().sha256 =
            helper::sha256_bytes(&package.manifest.encode().unwrap())[7..].into();
    }
    let input = mount("training", "/shared/data", "a");
    let mut package = slurm_package(&[input]);
    managed::execution_environment(&package.config, &package.manifest).unwrap();
    package.manifest.environment.insert(
        "DATASETS_JSON".into(),
        json!({"training": "/shared/elsewhere"}).to_string(),
    );
    reseal(&mut package);
    assert_eq!(
        managed::execution_environment(&package.config, &package.manifest),
        Err(AdmissionError::Binding)
    );
    package.manifest.environment.remove("DATASETS_JSON");
    reseal(&mut package);
    assert_eq!(
        managed::execution_environment(&package.config, &package.manifest),
        Err(AdmissionError::Binding)
    );
}

#[test]
fn a_slurm_restore_reads_the_earlier_attempts_state_directory() {
    let state_root = "/shared/project/runs/demo/earlier/attempts/attempt-001/state";
    let context = checkpoints::Context {
        project: "demo".into(),
        run_id: "earlier".into(),
        attempt_id: "attempt-001".into(),
        source_id: format!("source.{}", "a".repeat(64)),
        image_id: format!("sha256:{}", "b".repeat(64)),
        storage_scope: format!("sha256:{}", "d".repeat(64)),
        state_root: state_root.into(),
    };
    let ready = json!({"step":4,"files":[
        {"path":"checkpoints/step-4/state.bin","bytes":7,"sha256":"a".repeat(64)}
    ]});
    let document = checkpoints::document(&context, &ready).unwrap();
    let spec = spec();
    let package =
        slurm_managed::plan_restoring(&scope(&spec), &spec, &runtime(), Some(&document)).unwrap();
    let environment = &package.manifest.environment;
    assert_eq!(
        environment["RESUME_DIR"],
        format!("{state_root}/checkpoints/step-4")
    );
    assert_eq!(environment["RESUME_STEP"], "4");
    assert_eq!(
        environment["RESUME_CHECKPOINT_ID"],
        document["checkpoint_id"]
    );
}

#[test]
fn maintenance_selects_the_same_relative_paths_on_every_backend() {
    let plan = |config: &Config, selection| Plan {
        contract: "native-storage-maintenance.v1".into(),
        config: config.clone(),
        selection,
        request_id: "a".repeat(32),
    };
    for config in [slurm_package(&[]).config, cloud_package(Vec::new()).config] {
        assert_eq!(
            plan(&config, Selection::Outputs).paths().unwrap(),
            ["outputs"]
        );
        let state = plan(&config, Selection::State);
        assert_eq!(state.paths().unwrap(), ["state"]);
        assert_eq!(
            state.state_root(),
            config.output_path().parent().unwrap().join("state")
        );
    }
}
