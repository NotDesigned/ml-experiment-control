//! Read a fixed metrics filename through a retained shared-storage capability.
//! This operation cannot invoke a scheduler, launch a worker or select a path.
use super::*;
use ml_exp_core::managed::{
    evidence::Phase,
    history::{self, Cursor, Page},
};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::os::unix::fs::MetadataExt;

impl Helper {
    pub(super) fn metric_history(
        &self,
        request: &Request,
        snapshot: &Snapshot,
        lock: &Lock,
    ) -> Result<ReplyOutcome> {
        let Some(receipt) = &snapshot.receipt else {
            return Ok(ReplyOutcome::Unknown {
                reason: UnknownReason::SubmitStarted,
            });
        };
        if request.requested_job_id() != Some(receipt.job_id.as_str()) {
            return Ok(ReplyOutcome::Rejected {
                code: Rejection::IdentityConflict,
            });
        }
        self.verify_issued(request.intent(), &snapshot.directory)?;
        lock.validate()?;
        let read = || -> Result<Page> {
            let (plan, directories) = self
                .observe_shared(request.intent(), &snapshot.directory, false)?
                .ok_or(Error::Configuration)?;
            let package = plan.managed.as_ref().ok_or(Error::Configuration)?;
            let worker = crate::managed::sample(&package.config).map_err(|_| Error::Metadata)?;
            let outputs = directories[5].child(OsStr::new("payload"), false)?;
            let mut file = outputs.open_regular_read(OsStr::new("metrics.jsonl"))?;
            let before = file.metadata()?;
            let start = request.history_cursor();
            let offset = start.map(|v| v.offset_bytes).unwrap_or(0);
            if before.len() > history::MAX_FILE_BYTES
                || offset > before.len()
                || start.is_some_and(|c| c.device != before.dev() || c.inode != before.ino())
            {
                return Err(Error::Metadata);
            }
            let mut hash = Sha256::new();
            let mut remaining = offset;
            let mut buffer = [0u8; 16384];
            while remaining > 0 {
                let n = remaining.min(buffer.len() as u64) as usize;
                file.read_exact(&mut buffer[..n])?;
                hash.update(&buffer[..n]);
                remaining -= n as u64;
            }
            if start
                .is_some_and(|c| format!("sha256:{:x}", hash.clone().finalize()) != c.prefix_sha256)
            {
                return Err(Error::Metadata);
            }
            let count = (before.len() - offset).min(history::PAGE_BYTES as u64) as usize;
            let mut bytes = vec![0; count];
            file.read_exact(&mut bytes)?;
            hash.update(&bytes);
            outputs.validate_regular_entry(OsStr::new("metrics.jsonl"), &file)?;
            let after = file.metadata()?;
            if before.len() != after.len()
                || before.mtime() != after.mtime()
                || before.mtime_nsec() != after.mtime_nsec()
                || before.ctime() != after.ctime()
                || before.ctime_nsec() != after.ctime_nsec()
            {
                return Err(Error::Metadata);
            }
            self.observe_shared(request.intent(), &snapshot.directory, false)?;
            let after_worker =
                crate::managed::sample(&package.config).map_err(|_| Error::Metadata)?;
            if worker != after_worker {
                return Err(Error::Metadata);
            }
            let page = Page {
                contract: history::CONTRACT.into(),
                config_sha256: helper::sha256_bytes(&package.config_bytes()?),
                start: start.cloned(),
                next: Cursor {
                    device: before.dev(),
                    inode: before.ino(),
                    offset_bytes: offset + count as u64,
                    prefix_sha256: format!("sha256:{:x}", hash.finalize()),
                },
                bytes,
                total_bytes: before.len(),
                process_finished: worker.phase == Phase::Finished,
            };
            page.validate(start)?;
            Ok(page)
        };
        let result = match read() {
            Ok(page) => ReplyOutcome::MetricHistory {
                page: Box::new(page),
            },
            Err(_) => ReplyOutcome::Unknown {
                reason: UnknownReason::StorageUnavailable,
            },
        };
        self.verify_issued(request.intent(), &snapshot.directory)?;
        lock.validate()?;
        Ok(result)
    }
}
