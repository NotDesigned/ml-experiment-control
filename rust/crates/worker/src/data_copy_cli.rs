//! A zero-GPU copy job only unpacks its image's fixed archive to the approved
//! persistent data store. The callback contains a receipt, never dataset bytes.
use clap::Parser;
use ml_exp_core::datasets::import::{Config, DataChannel, Source};
use std::{ffi::OsString, io::Read, process::ExitCode, sync::atomic::AtomicBool};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config_json: String,
    #[arg(long)]
    channel_json: String,
    /// Frozen backend mount and cache path; absent for older copy jobs.
    #[arg(long)]
    storage_json: Option<String>,
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
        if args.config_json.len() > 8 * 1024 * 1024 || args.channel_json.len() > 16384 {
            return Err("data-copy request too large".into());
        }
        let config: Config = serde_json::from_value(serde_json::Value::Object(
            ml_exp_core::json::mapping(args.config_json.as_bytes())?,
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
        let storage: Option<ml_exp_core::cloud::launch::StorageLayout> = args
            .storage_json
            .as_deref()
            .map(
                |raw| -> Result<
                    ml_exp_core::cloud::launch::StorageLayout,
                    Box<dyn std::error::Error>,
                > {
                    if raw.len() > 16384 {
                        return Err("data storage plan too large".into());
                    }
                    let value = ml_exp_core::json::mapping(raw.as_bytes())?;
                    Ok(serde_json::from_value(serde_json::Value::Object(value))?)
                },
            )
            .transpose()?;
        ml_exp_worker::cloud_data::scoped_cache(&config, storage.as_ref())?;
        let receipt = ml_exp_worker::data_import::materialize(
            &config,
            &AtomicBool::new(false),
            &mut |progress| {
                println!(
                    "{}{}",
                    ml_exp_core::input_progress::MARKER,
                    serde_json::to_string(progress)?
                );
                Ok(())
            },
        )?;
        println!("ML_EXPD_DATA_READY={}", serde_json::to_string(&receipt)?);
        let http = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
        let response = http
            .post(format!("{}/ready", channel.url))
            .bearer_auth(&channel.token)
            .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2")
            .header("Content-Type", "application/json")
            .body(serde_json::to_vec(&receipt)?)
            .send()?;
        if response.status() != 200 {
            return Err(format!("DATA_READY_CALLBACK_HTTP_{}", response.status().as_u16()).into());
        }
        let mut bytes = Vec::new();
        response.take(1025).read_to_end(&mut bytes)?;
        let ack: serde_json::Value = serde_json::from_slice(&bytes)?;
        if ack["asset_id"] != receipt.asset_id || ack["status"] != "READY" {
            return Err("data ready callback differs".into());
        }
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
