use clap::Parser;
use std::{ffi::OsString, io::Read, path::PathBuf, process::ExitCode, time::Duration};
#[derive(Parser)]
struct Args {
    #[arg(
        long,
        required_unless_present = "plan_json",
        conflicts_with = "plan_json"
    )]
    config: Option<PathBuf>,
    #[arg(long)]
    plan_json: Option<String>,
    #[arg(long, requires = "plan_json")]
    channel_json: Option<String>,
}
pub fn main_from(args: impl IntoIterator<Item = OsString>) -> ExitCode {
    let args = match Args::try_parse_from(args) {
        Ok(a) => a,
        Err(e) => {
            let _ = e.print();
            return ExitCode::from(2);
        }
    };
    let result = (|| -> Result<_, Box<dyn std::error::Error>> {
        let mut bytes = Vec::new();
        if let Some(value) = args.plan_json {
            bytes = value.into_bytes();
        } else {
            std::io::stdin().take(262145).read_to_end(&mut bytes)?;
        }
        if bytes.len() > 262144 {
            return Err("maintenance request too large".into());
        }
        let plan: ml_exp_core::maintenance::Plan = serde_json::from_value(
            serde_json::Value::Object(ml_exp_core::json::mapping(&bytes)?),
        )?;
        if let Some(path) = args.config {
            let config = serde_json::Value::Object(ml_exp_core::json::mapping(
                &ml_exp_io::read_private(&path, 131072, true)?,
            )?);
            let root = config["shared_run_root"]
                .as_str()
                .ok_or("maintenance requires an approved shared run root")?;
            let binding = plan.config.binding()?;
            if plan.storage_root()
                != PathBuf::from(root)
                    .join(&binding.identity.project)
                    .join(&binding.identity.run_id)
                    .join("attempts")
                    .join(&binding.identity.attempt_id)
            {
                return Err("maintenance root differs".into());
            }
        }
        let receipt = ml_exp_worker::maintenance::run(&plan)?;
        if let Some(value) = args.channel_json {
            let channel: ml_exp_core::cloud::exchange::Channel = serde_json::from_str(&value)?;
            channel.validate(&plan.config.binding()?.identity)?;
            let http = reqwest::blocking::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never())
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(30))
                .build()?;
            let response = http
                .post(format!("{}/maintenance", channel.url))
                .bearer_auth(channel.token)
                .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2")
                .body(serde_json::to_vec(&receipt)?)
                .send()?;
            if response.status() != 200 {
                return Err(
                    "maintenance receipt is unconfirmed; original plan remains queryable".into(),
                );
            }
        }
        Ok(receipt)
    })();
    match result {
        Ok(value) => {
            println!("{value}");
            ExitCode::SUCCESS
        }
        Err(_) => {
            eprintln!("native storage maintenance is unconfirmed; reconcile the original plan");
            ExitCode::FAILURE
        }
    }
}
