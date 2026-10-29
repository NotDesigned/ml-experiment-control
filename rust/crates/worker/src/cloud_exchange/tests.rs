use super::*;
use ml_exp_core::{
    cloud::launch::{self, Input},
    managed::evidence::Phase,
};
use std::{collections::BTreeMap, sync::Mutex};
struct Sender {
    calls: Mutex<Vec<(u64, String)>>,
    lose_first: bool,
    checkpoint_error: Mutex<Option<String>>,
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
    })
    .unwrap();
    package.channel = Some(exchange::Channel {
        url: "https://api.example/ml-expd/api/worker-exchange/demo/train/attempt-001".into(),
        token: "b".repeat(64),
    });
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
