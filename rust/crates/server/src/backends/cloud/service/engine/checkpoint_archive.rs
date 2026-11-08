//! Object-store copies of registered checkpoints. The worker sends bytes straight to
//! the object store with short-lived tickets; this side issues them and reads every
//! object back before it records a receipt. Nothing here starts or replays a job.
use super::*;
use ml_exp_core::checkpoints::archive::{self, Operation, Request};

fn unavailable() -> cloud::Error {
    cloud::Error {
        code: "CHECKPOINT_ARCHIVE_UNAVAILABLE".into(),
        http_status: Some(409),
        uncertain: false,
    }
}
fn mismatch() -> cloud::Error {
    cloud::Error {
        code: "CHECKPOINT_ARCHIVE_MISMATCH".into(),
        http_status: Some(409),
        uncertain: false,
    }
}
fn unix(now: &str) -> cloud::Result<i64> {
    Ok(
        time::OffsetDateTime::parse(now, &time::format_description::well_known::Rfc3339)
            .map_err(why!(cloud::Error::configuration()))?
            .unix_timestamp(),
    )
}

impl Service {
    fn checkpoint_registry(&self) -> cloud::Result<crate::checkpoints::Store> {
        crate::checkpoints::Store::open_existing(&self.settings.registry_root)
            .map_err(why!(cloud::Error::evidence()))
    }
    /// Every registered checkpoint of this Attempt, each with its verified copy.
    pub(super) fn archive_rows(
        &self,
        prepared: &Prepared,
    ) -> cloud::Result<Vec<(Value, Option<archive::Receipt>)>> {
        let context = self.checkpoint_context(prepared)?;
        self.checkpoint_registry()?
            .list_archived(&context)
            .map_err(why!(cloud::Error::evidence()))
    }
    pub fn receive_checkpoint_archive(
        &self,
        project: &str,
        run: &str,
        attempt: &str,
        token: &str,
        request: Request,
        now: &str,
    ) -> cloud::Result<Value> {
        let objects = self.objects.as_ref().ok_or_else(unavailable)?;
        let timestamp = unix(now)?;
        // Restoring reads only this Attempt's own frozen restore choice, before training
        // starts and before its execution can be verified.
        if let Operation::Download { start } = request.operation {
            let package = self.worker_authorized(project, run, attempt, token)?;
            if request.package_sha256 != helper::sha256_bytes(&encode(&package)?) {
                return Err(cloud::Error::evidence());
            }
            if !package.checkpoint_archive {
                return Err(unavailable());
            }
            let frozen = package
                .config
                .checkpoint_restore
                .as_ref()
                .ok_or_else(missing)?;
            let (source, _) = ml_exp_core::checkpoints::validate_document(frozen, None, None)
                .map_err(why!(cloud::Error::evidence()))?;
            let id = frozen["checkpoint_id"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?;
            let (document, receipt) = self
                .checkpoint_registry()?
                .archived(&source, id)
                .map_err(why!(cloud::Error::evidence()))?
                .ok_or_else(missing)?;
            let receipt = receipt.ok_or_else(unavailable)?;
            return objects
                .checkpoint_tickets(
                    &document,
                    start,
                    archive::TICKETS_PER_REQUEST,
                    false,
                    Some(&receipt.part_crc64),
                    timestamp,
                )
                .map_err(io);
        }
        let prepared = self.worker_authority(project, run, attempt, token)?;
        let package = prepared
            .execution
            .package()
            .map_err(why!(cloud::Error::evidence()))?;
        if request.package_sha256 != helper::sha256_bytes(&encode(&package)?) {
            return Err(cloud::Error::evidence());
        }
        if !package.checkpoint_archive {
            return Err(unavailable());
        }
        let context = self.checkpoint_context(&prepared)?;
        let registry = self.checkpoint_registry()?;
        let one = |id: &str| -> cloud::Result<(Value, Option<archive::Receipt>)> {
            ml_exp_core::checkpoints::validate_checkpoint_id(id)
                .map_err(why!(cloud::Error::evidence()))?;
            registry
                .archived(&context, id)
                .map_err(why!(cloud::Error::evidence()))?
                .ok_or_else(missing)
        };
        match &request.operation {
            Operation::List => {
                let rows = self.archive_rows(&prepared)?;
                Ok(json!({
                    "checkpoints": rows
                        .iter()
                        .map(|(document, receipt)| json!({
                            "checkpoint_id": document["checkpoint_id"],
                            "step": document["step"],
                            "archived": receipt.is_some(),
                        }))
                        .collect::<Vec<_>>(),
                    "max_bytes_per_second": self.archive.upload_bytes_per_second,
                }))
            }
            Operation::Document { checkpoint_id } => Ok(one(checkpoint_id)?.0),
            Operation::Upload {
                checkpoint_id,
                start,
            } => {
                let (document, receipt) = one(checkpoint_id)?;
                if receipt.is_some() {
                    return Ok(json!({"archived": true, "tickets": []}));
                }
                objects
                    .checkpoint_tickets(
                        &document,
                        *start,
                        archive::TICKETS_PER_REQUEST,
                        true,
                        None,
                        timestamp,
                    )
                    .map_err(io)
            }
            Operation::Complete {
                checkpoint_id,
                part_crc64,
                file_sha256,
            } => {
                let (document, receipt) = one(checkpoint_id)?;
                if let Some(receipt) = receipt {
                    return Ok(json!({"archived": true, "receipt": receipt}));
                }
                let (_, manifest) =
                    ml_exp_core::checkpoints::validate_document(&document, None, None)
                        .map_err(why!(cloud::Error::evidence()))?;
                // The worker's SHA256 of the bytes it sent must be the registered one.
                if file_sha256.len() != manifest.files.len()
                    || manifest
                        .files
                        .iter()
                        .zip(file_sha256)
                        .any(|(record, sha)| record.sha256 != *sha)
                {
                    return Err(mismatch());
                }
                // No byte moves through this host: the object store's own size and
                // CRC-64 of each part must be those of the bytes the worker hashed. An
                // unreadable store is an error that proves nothing; a missing or
                // different object is a fact. Neither records a receipt.
                if !objects
                    .verify_checkpoint_parts(&document, part_crc64)
                    .map_err(io)?
                {
                    return Err(mismatch());
                }
                let receipt = archive::Receipt::new(&document, now, part_crc64.clone())
                    .map_err(why!(cloud::Error::evidence()))?;
                let kept = registry
                    .record_archive(&context, &receipt)
                    .map_err(why!(cloud::Error::evidence()))?;
                Ok(json!({"archived": true, "receipt": kept}))
            }
            Operation::Download { .. } => Err(cloud::Error::evidence()),
        }
    }
}
