//! Published historical results are evidence, never submission or deletion
//! authority. Project only explicitly allowed fields from private transfers.
use super::{Error, MAX_FILES, Result, bounded, receipt_path, sha256};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoricalFile {
    path: String,
    bytes: u64,
    #[serde(default)]
    sha256: Option<String>,
}

struct HistoricalArchive {
    sha256: String,
    bytes: u64,
    files: Vec<HistoricalFile>,
}

fn published_receipt(
    value: &Value,
    project: &str,
    run: &str,
    attempt: &str,
) -> Result<HistoricalArchive> {
    bounded(value)?;
    let archive = HistoricalArchive {
        sha256: value["sha256"].as_str().ok_or(Error::Receipt)?.into(),
        bytes: value["bytes"].as_u64().ok_or(Error::Receipt)?,
        files: serde_json::from_value(value["files"].clone()).map_err(|_| Error::Receipt)?,
    };
    if archive.bytes == 0
        || !sha256(&archive.sha256)
        || archive.files.is_empty()
        || archive.files.len() > MAX_FILES
        || value["object_key"] != format!("{project}/{run}/{attempt}/{}.tar", archive.sha256)
    {
        return Err(Error::Receipt);
    }
    let mut paths = BTreeSet::new();
    for file in &archive.files {
        if !receipt_path(&file.path)
            || file.sha256.as_deref().is_some_and(|hash| !sha256(hash))
            || !paths.insert(file.path.as_str())
        {
            return Err(Error::Receipt);
        }
    }
    Ok(archive)
}

pub fn inventory(
    transfer: Option<&Value>,
    project: &str,
    run: &str,
    attempt: &str,
    source: &str,
) -> Result<Value> {
    crate::identity::validate_id(project).map_err(|_| Error::Identity)?;
    crate::identity::validate_observation_id(run).map_err(|_| Error::Identity)?;
    crate::identity::validate_attempt(attempt).map_err(|_| Error::Identity)?;
    let mut archive = None;
    let mut retention = "MISSING";
    let mut exit_code = None;
    if let Some(value) = transfer {
        if !value.is_object()
            || value["project"] != project
            || value["run_id"] != run
            || value["attempt_id"] != attempt
        {
            return Err(Error::Identity);
        }
        if let Some(launch) = value.get("launch_manifest") {
            if launch["identity"] != json!({"project":project,"run_id":run,"attempt_id":attempt})
                || launch["environment"]
                    .get("SOURCE_ID")
                    .is_some_and(|id| id != source)
            {
                return Err(Error::Identity);
            }
        }
        if let Some(result) = value.get("results_ready").filter(|v| !v.is_null()) {
            if result["project"] != project
                || result["run_id"] != run
                || result["attempt_id"] != attempt
            {
                return Err(Error::Identity);
            }
            exit_code = Some(
                result["exit_code"]
                    .as_i64()
                    .filter(|code| (-255..=4096).contains(code))
                    .ok_or(Error::Receipt)?,
            );
        }
        if let Some(receipt) = value.get("receipt").filter(|v| !v.is_null()) {
            let receipt = published_receipt(receipt, project, run, attempt)?;
            retention = "AVAILABLE";
            if let Some(kept) = value.get("retention").filter(|v| !v.is_null()) {
                if kept["sha256"] != receipt.sha256 {
                    return Err(Error::Receipt);
                }
                retention = match kept["status"].as_str() {
                    Some("KEEP" | "SCHEDULED") => "AVAILABLE",
                    Some("RELEASING" | "RELEASED") => "RELEASED",
                    Some("EXPIRING" | "EXPIRED") => "EXPIRED",
                    _ => return Err(Error::Receipt),
                };
            }
            archive = Some(receipt);
        } else if value.get("retention").is_some_and(|v| !v.is_null()) {
            return Err(Error::Receipt);
        }
    }
    let mut files = archive
        .as_ref()
        .map(|a| {
            a.files
                .iter()
                .map(|f| json!({"path":format!("outputs/{}",f.path),"bytes":f.bytes,"sha256":f.sha256}))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    files.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    let checksums_complete = archive
        .as_ref()
        .is_some_and(|a| a.files.iter().all(|file| file.sha256.is_some()));
    Ok(json!({"project":project,"run_id":run,"attempt_id":attempt,
        "files":files,"truncated":false,"available":retention=="AVAILABLE",
        "collection_required":archive.is_none(),"storage_status":retention,
        "archive":archive.map(|a| json!({"sha256":a.sha256,"bytes":a.bytes})),
        "training":{"status":match exit_code {Some(0)=>"COMPLETED",Some(_)=>"FAILED",None=>"UNKNOWN"},
            "exit_code":exit_code,"evidence":exit_code.map(|_|"worker-result-manifest")},
        "source":"historical-published-receipt","read_only":true,
        "inventory_scope":"published_archive","local_files_included":false,
        "checksums_source":"published-receipt","payload_verified":false,
        "checksums_complete":checksums_complete,
        "acknowledgement_sent":false}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transfer() -> Value {
        let sha = "a".repeat(64);
        json!({"project":"demo","run_id":"train","attempt_id":"attempt-001",
            "token":"private-transfer-token","run_dir":"/private/server/path",
            "launch_manifest":{"identity":{"project":"demo","run_id":"train","attempt_id":"attempt-001"},
                "environment":{"SOURCE_ID":"source.fixture","PRIVATE_KEY":"never-expose"}},
            "results_ready":{"project":"demo","run_id":"train","attempt_id":"attempt-001","exit_code":0},
            "receipt":{"bytes":10240,"sha256":sha,"object_key":format!("demo/train/attempt-001/{sha}.tar"),
                "files":[{"path":"final.pt","bytes":128,"sha256":"b".repeat(64)}]}})
    }

    #[test]
    fn projects_receipts_without_private_capabilities_or_invented_verification() {
        let value = inventory(
            Some(&transfer()),
            "demo",
            "train",
            "attempt-001",
            "source.fixture",
        )
        .unwrap();
        assert_eq!(value["files"][0]["path"], "outputs/final.pt");
        assert_eq!(value["training"]["exit_code"], 0);
        assert_eq!(value["available"], true);
        assert_eq!(value["payload_verified"], false);
        assert_eq!(value["checksums_complete"], true);
        let text = value.to_string();
        for secret in [
            "private-transfer-token",
            "never-expose",
            "/private/server/path",
            "object_key",
        ] {
            assert!(!text.contains(secret));
        }
    }

    #[test]
    fn historical_missing_file_hash_is_unknown_without_weakening_native_receipts() {
        let mut value = transfer();
        value["receipt"]["files"][0]
            .as_object_mut()
            .unwrap()
            .remove("sha256");
        let result = inventory(
            Some(&value),
            "demo",
            "train",
            "attempt-001",
            "source.fixture",
        )
        .unwrap();
        assert_eq!(result["files"][0]["sha256"], Value::Null);
        assert_eq!(result["checksums_complete"], false);
        assert_eq!(result["payload_verified"], false);
        assert_eq!(result["archive"]["sha256"], "a".repeat(64));
        assert!(
            serde_json::from_value::<super::super::FileReceipt>(
                value["receipt"]["files"][0].clone()
            )
            .is_err()
        );
        value["receipt"]["files"][0]["sha256"] = json!("invalid");
        assert!(
            inventory(
                Some(&value),
                "demo",
                "train",
                "attempt-001",
                "source.fixture"
            )
            .is_err()
        );
    }

    #[test]
    fn release_expiry_and_unknown_are_distinct_from_training_completion() {
        for status in ["RELEASING", "RELEASED", "EXPIRING", "EXPIRED"] {
            let mut value = transfer();
            value["retention"] = json!({"status":status,"sha256":"a".repeat(64)});
            let result = inventory(
                Some(&value),
                "demo",
                "train",
                "attempt-001",
                "source.fixture",
            )
            .unwrap();
            assert_eq!(result["available"], false);
            assert_eq!(result["collection_required"], false);
            assert_eq!(result["files"].as_array().unwrap().len(), 1);
            assert_eq!(result["training"]["status"], "COMPLETED");
        }
        let empty = inventory(None, "demo", "train", "attempt-001", "source.fixture").unwrap();
        assert_eq!(empty["training"]["exit_code"], Value::Null);
        assert_eq!(empty["training"]["status"], "UNKNOWN");
        assert_eq!(empty["collection_required"], true);
    }

    #[test]
    fn rejects_foreign_or_malformed_receipts_and_unsafe_paths() {
        for (pointer, replacement) in [
            ("/project", json!("foreign")),
            ("/attempt_id", json!("attempt-002")),
            (
                "/launch_manifest/environment/SOURCE_ID",
                json!("source.other"),
            ),
            ("/launch_manifest/identity/attempt_id", json!("attempt-002")),
            ("/receipt/object_key", json!("foreign/object")),
            ("/receipt/sha256", json!("invalid")),
            ("/receipt/files/0/path", json!("../private")),
            ("/receipt/files/0/sha256", json!("bad")),
            ("/receipt/bytes", json!(-1)),
            ("/results_ready/exit_code", json!("0")),
        ] {
            let mut value = transfer();
            *value.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                inventory(
                    Some(&value),
                    "demo",
                    "train",
                    "attempt-001",
                    "source.fixture"
                )
                .is_err(),
                "{pointer}"
            );
        }
        let mut value = transfer();
        value["retention"] = json!({"status":"RELEASED","sha256":"c".repeat(64)});
        assert!(
            inventory(
                Some(&value),
                "demo",
                "train",
                "attempt-001",
                "source.fixture"
            )
            .is_err()
        );
    }
}
