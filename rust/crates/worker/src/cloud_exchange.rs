//! Metadata transport is independent from training. Persist one exact pending
//! packet before sending; acknowledge it before advancing the history cursor.
use crate::managed::{self, Error, Result};
use fs2::FileExt;
use ml_exp_core::{
    cloud::{
        exchange::{self, Ack, Packet},
        launch::Package,
    },
    managed::{
        evidence::Phase,
        history::Cursor,
        remote::{Evidence, OBSERVATION_CONTRACT, Problem},
    },
};
use ml_exp_io::{directory::Directory, observations};
use serde::{Deserialize, Serialize};
use std::{
    ffi::OsStr,
    io::Read,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
const STATE: &str = ".cloud-exchange.json";
#[cfg(test)]
mod tests;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    package_sha256: String,
    sequence: u64,
    packet_sha256: Option<String>,
    cursor: Option<Cursor>,
    complete: bool,
    pending: Option<Packet>,
}
fn packet(package: &Package, state: &State) -> Result<Packet> {
    let config = &package.config;
    let worker = managed::sample(config)?;
    let binding = config.binding()?;
    let config_sha256 = ml_exp_core::slurm::helper::sha256_bytes(
        &serde_json::to_vec(config).map_err(|_| Error::Configuration)?,
    );
    let outputs = Directory::open(&config.output_path(), false).map_err(|_| Error::Storage);
    let (mut stdout, mut stderr, metrics, history) = match outputs {
        Ok(outputs) => {
            let value = (
                observations::window(&outputs, "stdout.log"),
                observations::window(&outputs, "stderr.log"),
                observations::window(&outputs, "metrics.jsonl"),
                if state.complete {
                    None
                } else {
                    observations::history(
                        &outputs,
                        state.cursor.as_ref(),
                        worker.phase == Phase::Finished,
                        &config_sha256,
                    )
                    .map_err(|_| Error::Storage)?
                },
            );
            outputs
                .verify_private_parent()
                .map_err(|_| Error::Storage)?;
            value
        }
        Err(_) if worker.phase == Phase::Prepared && state.cursor.is_none() => (
            ml_exp_core::managed::remote::Window::unavailable(Problem::Missing),
            ml_exp_core::managed::remote::Window::unavailable(Problem::Missing),
            ml_exp_core::managed::remote::Window::unavailable(Problem::Missing),
            None,
        ),
        Err(error) => return Err(error),
    };
    if let Some(receipt) = &worker.receipt {
        for (window, expected) in [
            (&mut stdout, &receipt.process.stdout_sha256),
            (&mut stderr, &receipt.process.stderr_sha256),
        ] {
            if window.problem.is_none()
                && expected.as_ref().is_some_and(|sha| {
                    ml_exp_core::slurm::helper::sha256_bytes(window.text.as_bytes())[7..] != *sha
                })
            {
                window.problem = Some(Problem::HashMismatch);
            }
        }
    }
    let checkpoint_result = (|| -> Result<Option<serde_json::Value>> {
        Ok(match &package.checkpoint_context {
            None => None,
            Some(context) => {
                let state_root = Directory::open(Path::new(&context.state_root), false)
                    .map_err(|_| Error::Storage)?;
                if state_root
                    .entry_exists(OsStr::new("checkpoint.ready.json"))
                    .map_err(|_| Error::Storage)?
                {
                    let before = state_root
                        .read_regular(
                            OsStr::new("checkpoint.ready.json"),
                            ml_exp_core::checkpoints::MANIFEST_LIMIT as u64,
                        )
                        .map_err(|_| Error::Storage)?;
                    let ready = serde_json::Value::Object(
                        ml_exp_core::json::mapping(&before).map_err(|_| Error::Binding)?,
                    );
                    let manifest =
                        ml_exp_core::checkpoints::manifest(&ready).map_err(|_| Error::Binding)?;
                    ml_exp_io::checkpoints::verify(&state_root, &manifest, true)
                        .map_err(|_| Error::Binding)?;
                    if before
                        != state_root
                            .read_regular(
                                OsStr::new("checkpoint.ready.json"),
                                ml_exp_core::checkpoints::MANIFEST_LIMIT as u64,
                            )
                            .map_err(|_| Error::Storage)?
                    {
                        return Err(Error::Uncertain);
                    }
                    Some(
                        ml_exp_core::checkpoints::document(context, &ready)
                            .map_err(|_| Error::Binding)?,
                    )
                } else {
                    None
                }
            }
        })
    })();
    let (checkpoint, checkpoint_error) = match checkpoint_result {
        Ok(value) => (value, None),
        Err(_) => (None, Some("CHECKPOINT_VERIFICATION_FAILED".into())),
    };
    if managed::sample(config)? != worker {
        return Err(Error::Uncertain);
    }
    let packet = Packet {
        contract: exchange::CONTRACT.into(),
        sequence: state.sequence.checked_add(1).ok_or(Error::Binding)?,
        previous_sha256: state.packet_sha256.clone(),
        package_sha256: state.package_sha256.clone(),
        evidence: Evidence {
            contract: OBSERVATION_CONTRACT.into(),
            config_sha256,
            job_id: binding.sha256,
            worker,
            stdout,
            stderr,
            metrics,
        },
        history,
        checkpoint,
        checkpoint_error,
        inputs: crate::cloud_data::read(package)?,
    };
    packet
        .validate(config, &state.package_sha256)
        .map_err(|_| Error::Binding)?;
    Ok(packet)
}
fn save(root: &Directory, state: &State) -> Result<()> {
    let bytes = serde_json::to_vec(state).map_err(|_| Error::Configuration)?;
    if bytes.len() > exchange::MAX_BYTES + 1024 * 1024 {
        return Err(Error::Storage);
    }
    root.replace_regular(OsStr::new(STATE), &bytes)
        .map_err(|_| Error::Storage)
}
pub trait Transport {
    fn send(&self, channel: &exchange::Channel, packet: &Packet) -> Result<Ack>;
}
pub struct Https {
    client: reqwest::blocking::Client,
}
impl Https {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never())
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(15))
                .build()
                .map_err(|_| Error::Configuration)?,
        })
    }
}
impl Transport for Https {
    fn send(&self, channel: &exchange::Channel, packet: &Packet) -> Result<Ack> {
        let bytes = serde_json::to_vec(packet).map_err(|_| Error::Configuration)?;
        if bytes.len() > exchange::MAX_BYTES {
            return Err(Error::Binding);
        }
        let response = self
            .client
            .post(&channel.url)
            .bearer_auth(&channel.token)
            .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2")
            .header("content-type", "application/json")
            .body(bytes)
            .send()
            .map_err(|_| Error::Uncertain)?;
        if response.status().as_u16() != 200 || response.content_length().is_some_and(|n| n > 16384)
        {
            return Err(Error::Uncertain);
        }
        let mut bytes = Vec::new();
        response
            .take(16385)
            .read_to_end(&mut bytes)
            .map_err(|_| Error::Uncertain)?;
        if bytes.len() > 16384 {
            return Err(Error::Binding);
        }
        let ack: Ack = serde_json::from_value(serde_json::Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Binding)?,
        ))
        .map_err(|_| Error::Binding)?;
        ack.validate(packet).map_err(|_| Error::Binding)?;
        Ok(ack)
    }
}
/// Returns whether all history has been durably received after training ends.
/// This can run alone on a CPU collector without invoking a training process.
pub fn once(package: &Package, transport: &impl Transport) -> Result<bool> {
    package.validate().map_err(|_| Error::Configuration)?;
    let channel = package.channel.as_ref().ok_or(Error::Configuration)?;
    let root = Directory::open(&package.config.attempt_root, false).map_err(|_| Error::Storage)?;
    root.verify_private_parent().map_err(|_| Error::Storage)?;
    let lock = root
        .open_lock(OsStr::new(".cloud-exchange.lock"))
        .map_err(|_| Error::Storage)?;
    lock.try_lock_exclusive().map_err(|_| Error::Busy)?;
    let sha = crate::cloud::package_sha(package)?;
    let mut state: State = if root
        .entry_exists(OsStr::new(STATE))
        .map_err(|_| Error::Storage)?
    {
        let bytes = root
            .read_regular(
                OsStr::new(STATE),
                (exchange::MAX_BYTES + 1024 * 1024) as u64,
            )
            .map_err(|_| Error::Storage)?;
        serde_json::from_value(serde_json::Value::Object(
            ml_exp_core::json::mapping(&bytes).map_err(|_| Error::Binding)?,
        ))
        .map_err(|_| Error::Binding)?
    } else {
        State {
            package_sha256: sha.clone(),
            sequence: 0,
            packet_sha256: None,
            cursor: None,
            complete: false,
            pending: None,
        }
    };
    if state.package_sha256 != sha || (state.sequence == 0) != state.packet_sha256.is_none() {
        return Err(Error::Binding);
    }
    if state.complete && state.pending.is_none() {
        return Ok(true);
    }
    if state.pending.is_none() {
        state.pending = Some(packet(package, &state)?);
        save(&root, &state)?;
    }
    let pending = state.pending.as_ref().ok_or(Error::Binding)?;
    pending
        .validate(&package.config, &sha)
        .map_err(|_| Error::Binding)?;
    pending
        .follows(
            state.sequence,
            state.packet_sha256.as_deref(),
            state.cursor.as_ref(),
            state.complete,
        )
        .map_err(|_| Error::Binding)?;
    let ack = transport.send(channel, pending)?;
    ack.validate(pending).map_err(|_| Error::Binding)?;
    if let Some(history) = &pending.history {
        state.cursor = Some(history.next.clone());
        state.complete = history.complete();
    }
    state.sequence = ack.sequence;
    state.packet_sha256 = Some(ack.packet_sha256);
    state.pending = None;
    root.validate_regular_entry(OsStr::new(".cloud-exchange.lock"), &lock)
        .map_err(|_| Error::Storage)?;
    root.verify_private_parent().map_err(|_| Error::Storage)?;
    save(&root, &state)?;
    Ok(state.complete)
}
pub fn drain(package: &Package, cancelled: &AtomicBool) -> Result<bool> {
    let transport = Https::new()?;
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline && !cancelled.load(Ordering::Relaxed) {
        match once(package, &transport) {
            Ok(true) => return Ok(true),
            Ok(false) => {}
            Err(Error::Binding | Error::Configuration) => return Err(Error::Binding),
            Err(_) => std::thread::sleep(Duration::from_millis(500)),
        }
    }
    Ok(false)
}
