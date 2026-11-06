//! A zero-GPU copy job only unpacks its image's fixed archive to the approved
//! persistent data store. The callback contains a receipt, never dataset bytes.
use clap::Parser;
use ml_exp_core::datasets::{
    Receipt,
    import::{Config, DataChannel, Source},
};
use std::{
    ffi::OsString,
    io::Read,
    path::PathBuf,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};
const MAX_CONFIG_BYTES: usize = 8 * 1024 * 1024;
#[derive(Parser)]
struct Args {
    /// A private file holding the config, written by the launcher for this process.
    #[arg(
        long,
        conflicts_with = "config_json",
        required_unless_present = "config_json"
    )]
    config: Option<PathBuf>,
    /// The config text, repeated in parts that join back to the whole. A part can
    /// begin with a hyphen, because a split can land inside a file name.
    #[arg(long, allow_hyphen_values = true)]
    config_json: Vec<String>,
    #[arg(long)]
    channel_json: String,
    /// Frozen backend mount and cache path.
    #[arg(long)]
    storage_json: String,
}
fn config_bytes(args: &Args) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if let Some(path) = &args.config {
        return Ok(ml_exp_io::read_private(
            path,
            MAX_CONFIG_BYTES as u64,
            true,
        )?);
    }
    if args.config_json.iter().map(String::len).sum::<usize>() > MAX_CONFIG_BYTES {
        return Err("data-copy request too large".into());
    }
    Ok(args.config_json.concat().into_bytes())
}
fn termination_flag() -> std::io::Result<Arc<AtomicBool>> {
    let flag = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(signal, Arc::clone(&flag))?;
    }
    Ok(flag)
}
/// The copy is complete by now, so a busy server must not fail it. Sending the same
/// receipt again is harmless: the server keeps the first and accepts an identical one.
const READY_ATTEMPTS: u32 = 10;
fn ready_delay(attempt: u32) -> Duration {
    Duration::from_secs((1u64 << attempt.min(6)).min(60))
}
fn ready_status_may_pass(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504)
}
enum ReadyFailure {
    Again(String),
    Final(String),
}
fn post_ready(
    http: &reqwest::blocking::Client,
    channel: &DataChannel,
    receipt: &Receipt,
) -> Result<(), ReadyFailure> {
    let body =
        serde_json::to_vec(receipt).map_err(|error| ReadyFailure::Final(error.to_string()))?;
    let response = http
        .post(format!("{}/ready", channel.url))
        .bearer_auth(&channel.token)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .map_err(|error| ReadyFailure::Again(error.without_url().to_string()))?;
    let status = response.status().as_u16();
    if status != 200 {
        let text = format!("DATA_READY_CALLBACK_HTTP_{status}");
        return Err(if ready_status_may_pass(status) {
            ReadyFailure::Again(text)
        } else {
            ReadyFailure::Final(text)
        });
    }
    let mut bytes = Vec::new();
    response
        .take(1025)
        .read_to_end(&mut bytes)
        .map_err(|error| ReadyFailure::Again(error.to_string()))?;
    let ack: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| ReadyFailure::Again(error.to_string()))?;
    if ack["asset_id"] != receipt.asset_id || ack["status"] != "READY" {
        return Err(ReadyFailure::Final("data ready callback differs".into()));
    }
    Ok(())
}
fn confirm_ready(
    http: &reqwest::blocking::Client,
    channel: &DataChannel,
    receipt: &Receipt,
) -> Result<(), String> {
    let mut attempt = 0;
    loop {
        attempt += 1;
        match post_ready(http, channel, receipt) {
            Ok(()) => return Ok(()),
            Err(ReadyFailure::Final(text)) => return Err(text),
            Err(ReadyFailure::Again(text)) if attempt >= READY_ATTEMPTS => return Err(text),
            Err(ReadyFailure::Again(text)) => {
                eprintln!(
                    "data ready callback attempt {attempt} failed: {}; trying again",
                    experiment_redact::redact_line(&text)
                );
                std::thread::sleep(ready_delay(attempt));
            }
        }
    }
}
pub fn main_from(args: impl IntoIterator<Item = OsString>) -> ExitCode {
    let args = match Args::try_parse_from(args) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("invalid native data-copy arguments");
            return ExitCode::from(2);
        }
    };
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        if args.channel_json.len() > 16384 || args.storage_json.len() > 16384 {
            return Err("data-copy request too large".into());
        }
        let config: Config = serde_json::from_value(serde_json::Value::Object(
            ml_exp_core::json::mapping(&config_bytes(&args)?)?,
        ))?;
        let channel: DataChannel = serde_json::from_value(serde_json::Value::Object(
            ml_exp_core::json::mapping(args.channel_json.as_bytes())?,
        ))?;
        config.validate()?;
        channel.validate(
            &config.project,
            &config.target.backend_id,
            &config.definition.asset_id(),
        )?;
        if !matches!(&config.source,Source::File{path} if path=="/ml-expd/data/archive.tar") {
            return Err("data copy requires the fixed image archive".into());
        }
        let storage: ml_exp_core::cloud::launch::StorageLayout = serde_json::from_value(
            serde_json::Value::Object(ml_exp_core::json::mapping(args.storage_json.as_bytes())?),
        )?;
        ml_exp_worker::cloud_data::scoped_cache(&config, &storage)?;
        let cancelled = termination_flag()?;
        let receipt =
            ml_exp_worker::data_import::materialize(&config, &cancelled, &mut |progress| {
                println!(
                    "{}{}",
                    ml_exp_core::input_progress::MARKER,
                    serde_json::to_string(progress)?
                );
                Ok(())
            })?;
        println!("ML_EXPD_DATA_READY={}", serde_json::to_string(&receipt)?);
        let http = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
        confirm_ready(&http, &channel, &receipt)?;
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!(
                "native data copy failed: {}",
                experiment_redact::redact_line(&error.to_string())
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    const REST: [&str; 4] = ["--channel-json", "{}", "--storage-json", "{}"];
    fn parse(config: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(["data-copy"].iter().chain(config).chain(&REST))
    }
    #[test]
    fn config_parts_join_back_to_the_whole_text() {
        let args = parse(&["--config-json", "{\"a\"", "--config-json", ":1}"]).unwrap();
        assert_eq!(config_bytes(&args).unwrap(), b"{\"a\":1}");
    }
    #[test]
    fn a_config_part_may_begin_with_a_hyphen() {
        let args = parse(&["--config-json", "{\"a\":\"b", "--config-json", "-c\"}"]).unwrap();
        assert_eq!(config_bytes(&args).unwrap(), b"{\"a\":\"b-c\"}");
    }
    #[test]
    fn a_config_file_is_read_only_when_private_and_within_the_cap() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("copy.json");
        std::fs::write(&path, b"{}").unwrap();
        let set_mode =
            |mode| std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        let args = parse(&["--config", path.to_str().unwrap()]).unwrap();
        set_mode(0o644);
        assert!(config_bytes(&args).is_err());
        set_mode(0o600);
        assert_eq!(config_bytes(&args).unwrap(), b"{}");
        std::fs::write(&path, vec![b' '; MAX_CONFIG_BYTES + 1]).unwrap();
        assert!(config_bytes(&args).is_err());
    }
    #[test]
    fn config_parts_above_the_cap_are_refused() {
        let part = " ".repeat(MAX_CONFIG_BYTES / 2 + 1);
        let args = parse(&["--config-json", &part, "--config-json", &part]).unwrap();
        assert!(config_bytes(&args).is_err());
    }
    #[test]
    fn a_term_signal_cancels_the_copy_instead_of_killing_it() {
        let flag = termination_flag().unwrap();
        signal_hook::low_level::raise(signal_hook::consts::SIGTERM).unwrap();
        assert!(flag.load(std::sync::atomic::Ordering::Relaxed));
    }
    #[test]
    fn exactly_one_config_source_is_required() {
        assert!(parse(&[]).is_err());
        assert!(parse(&["--config", "/x", "--config-json", "{}"]).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_a_passing_failure_is_tried_again() {
        for status in [408, 429, 500, 502, 503, 504] {
            assert!(ready_status_may_pass(status), "{status}");
        }
        for status in [400, 401, 403, 404, 409, 410, 422] {
            assert!(!ready_status_may_pass(status), "{status}");
        }
    }
    #[test]
    fn the_wait_grows_and_stays_bounded() {
        let waits: Vec<u64> = (1..READY_ATTEMPTS)
            .map(|n| ready_delay(n).as_secs())
            .collect();
        assert_eq!(waits[0], 2);
        assert!(waits.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(waits.iter().all(|wait| *wait <= 60));
        assert!(waits.iter().sum::<u64>() >= 300, "covers a long-held lock");
    }
}
