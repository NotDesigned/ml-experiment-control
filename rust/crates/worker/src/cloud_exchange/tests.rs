use super::*;
use ml_exp_core::{
    cloud::launch::{self, Input},
    managed::evidence::Phase,
};
use std::{collections::BTreeMap, sync::Mutex};
fn storage(mount: &Path, runs: &Path) -> launch::StorageLayout {
    launch::StorageLayout {
        mount: mount.into(),
        runs: runs.into(),
        datasets: mount.join("datasets"),
    }
}
fn channel() -> exchange::Channel {
    exchange::Channel {
        url: "https://api.example/ml-expd/api/worker-exchange/demo/train/attempt-001".into(),
        token: "b".repeat(64),
    }
}
struct Sender {
    calls: Mutex<Vec<(u64, String)>>,
    lose_first: bool,
    checkpoint_error: Mutex<Option<String>>,
}
#[test]
fn one_pending_prepared_packet_cannot_stand_in_for_a_fresh_process_observation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("runs");
    let work = temp.path().join("code");
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(&work).unwrap();
    let package = launch::compile(Input {
        root: root.to_str().unwrap(),
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        workdir: work.to_str().unwrap(),
        argv: &["/bin/true".into()],
        timeout_seconds: 3,
        worker_sha256: &format!("sha256:{}", "a".repeat(64)),
        storage: storage(temp.path(), &root),
        channel: channel(),
    })
    .unwrap();
    crate::cloud::prepare(&package, &BTreeMap::new()).unwrap();
    let sender = Sender {
        calls: Mutex::new(Vec::new()),
        lose_first: true,
        checkpoint_error: Mutex::new(None),
    };
    assert!(once(&package, &sender).is_err());
    // The lost packet said PREPARED, but training then actually executed.
    managed::run(&package.config, &AtomicBool::new(false)).unwrap();
    assert!(!once(&package, &sender).unwrap()); // only acknowledges the old packet
    let mut state: State =
        serde_json::from_slice(&std::fs::read(package.config.attempt_root.join(STATE)).unwrap())
            .unwrap();
    let fresh = packet(&package, &mut state).unwrap();
    assert_eq!(fresh.sequence, 2);
    assert_eq!(fresh.evidence.worker.phase, Phase::Finished);
    assert!(once(&package, &sender).unwrap());
    assert_eq!(
        sender
            .calls
            .lock()
            .unwrap()
            .iter()
            .map(|v| v.0)
            .collect::<Vec<_>>(),
        vec![1, 1, 2]
    );
}
#[test]
fn a_log_that_dropped_output_reports_its_end_and_no_hash_mismatch() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("runs");
    let work = temp.path().join("code");
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(&work).unwrap();
    let package = launch::compile(Input {
        root: root.to_str().unwrap(),
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        workdir: work.to_str().unwrap(),
        argv: &[
            "/bin/sh".into(),
            "-c".into(),
            "PATH=/usr/bin:/bin; head -c 9437184 /dev/zero | tr '\\0' x; printf last-line".into(),
        ],
        timeout_seconds: 3,
        worker_sha256: &format!("sha256:{}", "a".repeat(64)),
        storage: storage(temp.path(), &root),
        channel: channel(),
    })
    .unwrap();
    crate::cloud::prepare(&package, &BTreeMap::new()).unwrap();
    let finished = managed::run(&package.config, &AtomicBool::new(false)).unwrap();
    assert!(finished.succeeded());
    let process = &finished.receipt.as_ref().unwrap().process;
    assert!(process.stdout_dropped_bytes > 0);
    let log = package.config.output_path().join("stdout.log");
    assert_eq!(
        process.stdout_sha256.as_deref(),
        Some(ml_exp_io::sha256_file(&log).unwrap().as_str())
    );
    let mut state = State {
        package_sha256: crate::cloud::package_sha(&package).unwrap(),
        sequence: 0,
        packet_sha256: None,
        cursor: None,
        complete: false,
        pending: None,
        verified_ready_sha256: None,
    };
    let sent = packet(&package, &mut state).unwrap();
    assert_eq!(sent.evidence.stdout.problem, Some(Problem::Truncated));
    assert!(sent.evidence.stdout.text.ends_with("last-line"));
    assert_eq!(
        sent.evidence.stdout.total_bytes,
        std::fs::metadata(&log).unwrap().len()
    );
    assert_eq!(sent.evidence.stderr.problem, None);
}
impl Transport for Sender {
    fn send(&self, _: &exchange::Channel, packet: &Packet) -> Result<Ack> {
        let mut calls = self.calls.lock().unwrap();
        calls.push((packet.sequence, packet.sha256().unwrap()));
        *self.checkpoint_error.lock().unwrap() = packet.checkpoint_error.clone();
        if self.lose_first && calls.len() == 1 {
            return Err(Error::Uncertain);
        }
        Ack::for_packet(packet).map_err(|_| Error::Binding)
    }
}
#[test]
fn lost_ack_replays_only_the_exact_packet_and_a_corrupt_checkpoint_does_not_block_metrics() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("runs");
    let work = temp.path().join("code");
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(&work).unwrap();
    let command=vec!["/bin/sh".into(),"-c".into(),"printf x >> \"$STATE_DIR/executions\"; printf '{\"name\":\"loss\",\"unit\":\"nats/token\",\"value\":1.25,\"step\":1}\\n' > \"$METRICS_PATH\"; printf broken > \"$STATE_DIR/checkpoint.ready.json\"".into()];
    let mut package = launch::compile(Input {
        root: root.to_str().unwrap(),
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        workdir: work.to_str().unwrap(),
        argv: &command,
        timeout_seconds: 3,
        worker_sha256: &format!("sha256:{}", "a".repeat(64)),
        storage: storage(temp.path(), &root),
        channel: channel(),
    })
    .unwrap();
    package.checkpoint_context = Some(ml_exp_core::checkpoints::Context {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        source_id: format!("source.{}", "c".repeat(64)),
        image_id: format!("sha256:{}", "d".repeat(64)),
        storage_scope: format!("sha256:{}", "e".repeat(64)),
        state_root: package
            .config
            .attempt_root
            .join("state")
            .to_str()
            .unwrap()
            .into(),
    });
    crate::cloud::prepare(&package, &BTreeMap::new()).unwrap();
    assert_eq!(
        managed::run(&package.config, &AtomicBool::new(false))
            .unwrap()
            .phase,
        Phase::Finished
    );
    let sender = Sender {
        calls: Mutex::new(Vec::new()),
        lose_first: true,
        checkpoint_error: Mutex::new(None),
    };
    assert!(matches!(once(&package, &sender), Err(Error::Uncertain)));
    let pending = std::fs::read(package.config.attempt_root.join(STATE)).unwrap();
    assert!(once(&package, &sender).unwrap());
    assert!(once(&package, &sender).unwrap());
    let calls = sender.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0], calls[1]);
    assert_eq!(
        sender.checkpoint_error.lock().unwrap().as_deref(),
        Some("CHECKPOINT_VERIFICATION_FAILED")
    );
    assert_eq!(
        std::fs::read(package.config.attempt_root.join("state/executions")).unwrap(),
        b"x"
    );
    let mut replaced: State = serde_json::from_slice(&pending).unwrap();
    replaced.package_sha256 = format!("sha256:{}", "f".repeat(64));
    std::fs::write(
        package.config.attempt_root.join(STATE),
        serde_json::to_vec(&replaced).unwrap(),
    )
    .unwrap();
    assert!(matches!(once(&package, &sender), Err(Error::Binding)));
}
#[test]
fn one_ready_marker_is_verified_once_and_a_new_marker_is_verified_again() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("runs");
    let work = temp.path().join("code");
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(&work).unwrap();
    let mut package = launch::compile(Input {
        root: root.to_str().unwrap(),
        project: "demo",
        run: "train",
        attempt: "attempt-001",
        workdir: work.to_str().unwrap(),
        argv: &["/bin/true".into()],
        timeout_seconds: 3,
        worker_sha256: &format!("sha256:{}", "a".repeat(64)),
        storage: storage(temp.path(), &root),
        channel: channel(),
    })
    .unwrap();
    let state_root = package.config.attempt_root.join("state");
    package.checkpoint_context = Some(ml_exp_core::checkpoints::Context {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        source_id: format!("source.{}", "c".repeat(64)),
        image_id: format!("sha256:{}", "d".repeat(64)),
        storage_scope: format!("sha256:{}", "e".repeat(64)),
        state_root: state_root.to_str().unwrap().into(),
    });
    crate::cloud::prepare(&package, &BTreeMap::new()).unwrap();
    let generation = state_root.join("checkpoints/step-4");
    std::fs::create_dir_all(&generation).unwrap();
    let file = generation.join("state.bin");
    std::fs::write(&file, b"weights").unwrap();
    let marker = |step: u64| {
        let bytes = serde_json::to_vec(&serde_json::json!({"step":step,"files":[{
            "path":"checkpoints/step-4/state.bin","bytes":7,
            "sha256":"9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"}]}))
        .unwrap();
        std::fs::write(state_root.join("checkpoint.ready.json"), &bytes).unwrap();
        ml_exp_core::slurm::helper::sha256_bytes(&bytes)
    };
    let rewrite = |bytes: &[u8]| {
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::write(&file, bytes).unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o400)).unwrap();
    };
    let saved = || -> Option<String> {
        let bytes = std::fs::read(package.config.attempt_root.join(STATE)).unwrap();
        serde_json::from_slice::<State>(&bytes)
            .unwrap()
            .verified_ready_sha256
    };
    let sender = Sender {
        calls: Mutex::new(Vec::new()),
        lose_first: false,
        checkpoint_error: Mutex::new(None),
    };
    let first = marker(4);
    assert!(!once(&package, &sender).unwrap());
    assert_eq!(*sender.checkpoint_error.lock().unwrap(), None);
    assert_eq!(saved().as_ref(), Some(&first));
    assert_eq!(
        std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
        0o400
    );

    // Changed bytes under the same marker prove the next packet did not hash.
    rewrite(b"changed");
    assert!(!once(&package, &sender).unwrap());
    assert_eq!(*sender.checkpoint_error.lock().unwrap(), None);

    // A new marker is verified, and its failure is not remembered as success.
    marker(5);
    assert!(!once(&package, &sender).unwrap());
    assert_eq!(
        sender.checkpoint_error.lock().unwrap().as_deref(),
        Some("CHECKPOINT_VERIFICATION_FAILED")
    );
    assert_eq!(saved().as_ref(), Some(&first));

    // So the same marker is tried again once its files are right.
    rewrite(b"weights");
    let second = marker(5);
    assert!(!once(&package, &sender).unwrap());
    assert_eq!(*sender.checkpoint_error.lock().unwrap(), None);
    assert_eq!(saved().as_ref(), Some(&second));
}
