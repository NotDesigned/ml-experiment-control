#![cfg(unix)]
use ml_exp_client::{Client, Error, datasets};
use serde_json::json;
use std::fs;

/// Nothing listens on this port, so anything that reaches the network comes back as
/// an unconfirmed transfer rather than the local refusal.
fn upload(files: usize) -> Error {
    let directory = tempfile::tempdir().unwrap();
    for n in 0..files {
        let name = format!("dataset-tasks-sample-{n:05}.bin");
        fs::write(directory.path().join(name), b"x").unwrap();
    }
    let client = Client::new("http://127.0.0.1:9", "source-test-secret".into()).unwrap();
    let health = json!({"capabilities":["native-datasets.v1"]});
    datasets::upload(&client, "demo", "slurm", directory.path(), &health).unwrap_err()
}

#[test]
fn a_dataset_no_run_could_mount_is_refused_before_any_request() {
    assert!(matches!(
        upload(2500),
        Error::Observation(message) if message.contains("256 KiB")
    ));
    assert!(matches!(upload(3), Error::DatasetOutcomeUnknown { .. }));
}
