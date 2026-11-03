//! Fixed-name bounded shared-storage observations. This operation cannot invoke
//! a scheduler tool, initialize a worker, select a checkpoint or start a process.
use super::*;
use ml_exp_core::managed::remote::{Evidence, OBSERVATION_CONTRACT, Problem, WINDOW_BYTES, Window};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;

fn window(directory: &Directory, name: &str) -> Window {
    let read = || -> Result<Window> {
        if !directory.entry_exists(OsStr::new(name))? {
            return Ok(Window::unavailable(Problem::Missing));
        }
        let mut file = directory.open_regular_read(OsStr::new(name))?;
        let before = file.metadata()?;
        let mut offset = before.len().saturating_sub(WINDOW_BYTES as u64);
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = Vec::new();
        (&mut file)
            .take((WINDOW_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        directory.validate_regular_entry(OsStr::new(name), &file)?;
        let after = file.metadata()?;
        if bytes.len() as u64 != before.len() - offset
            || before.len() != after.len()
            || before.mtime() != after.mtime()
            || before.mtime_nsec() != after.mtime_nsec()
            || before.ctime() != after.ctime()
            || before.ctime_nsec() != after.ctime_nsec()
        {
            return Ok(Window::unavailable(Problem::Changed));
        }
        // A bounded tail may begin inside one otherwise valid UTF-8 character.
        // Skip only its leading continuation bytes; interior corruption remains
        // an explicit diagnostic, never lossy scientific JSON.
        if offset > 0 {
            let mut trim = 0;
            while trim < bytes.len().min(3) && bytes[trim] & 0xc0 == 0x80 {
                trim += 1;
            }
            bytes.drain(..trim);
            offset += trim as u64;
        }
        let text = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => return Ok(Window::unavailable(Problem::InvalidUtf8)),
        };
        Ok(Window {
            text,
            offset_bytes: offset,
            total_bytes: before.len(),
            problem: if offset > 0 {
                Some(Problem::Truncated)
            } else {
                None
            },
        })
    };
    read().unwrap_or_else(|_| Window::unavailable(Problem::Unavailable))
}
impl Helper {
    pub(super) fn worker_observation(
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
        let read = || -> Result<Evidence> {
            let (plan, directories) = self
                .observe_shared(request.intent(), &snapshot.directory, false)?
                .ok_or(Error::Configuration)?;
            let package = plan.managed.as_ref().ok_or(Error::Configuration)?;
            let worker = crate::managed::sample(&package.config).map_err(|_| Error::Metadata)?;
            let outputs = directories[5].child(OsStr::new("payload"), false);
            let (mut stdout, mut stderr, metrics) = match outputs {
                Ok(outputs) => {
                    let result = (
                        window(&outputs, "stdout.log"),
                        window(&outputs, "stderr.log"),
                        window(&outputs, "metrics.jsonl"),
                    );
                    outputs.verify_private_parent()?;
                    result
                }
                Err(_) => (
                    Window::unavailable(Problem::Missing),
                    Window::unavailable(Problem::Missing),
                    Window::unavailable(Problem::Missing),
                ),
            };
            // A second sealed-directory and worker-ledger read fences replacement
            // during sampling. A natural phase transition is retried by refresh.
            self.observe_shared(request.intent(), &snapshot.directory, false)?;
            let after = crate::managed::sample(&package.config).map_err(|_| Error::Metadata)?;
            if worker != after {
                return Err(Error::Metadata);
            }
            if let Some(receipt) = &worker.receipt {
                for (window, expected) in [
                    (&mut stdout, &receipt.process.stdout_sha256),
                    (&mut stderr, &receipt.process.stderr_sha256),
                ] {
                    if window.problem.is_none()
                        && expected.as_ref().is_some_and(|digest| {
                            helper::sha256_bytes(window.text.as_bytes())[7..] != *digest
                        })
                    {
                        window.problem = Some(Problem::HashMismatch);
                    }
                }
            }
            let value = Evidence {
                contract: OBSERVATION_CONTRACT.into(),
                config_sha256: helper::sha256_bytes(&package.config_bytes()?),
                job_id: receipt.job_id.clone(),
                worker,
                stdout,
                stderr,
                metrics,
            };
            value.validate_config(&package.config)?;
            value.validate_scope(&request.intent().scope, &receipt.job_id)?;
            Ok(value)
        };
        let outcome = match read() {
            Ok(evidence) => ReplyOutcome::WorkerObserved {
                evidence: Box::new(evidence),
            },
            Err(_) => ReplyOutcome::Unknown {
                reason: UnknownReason::StorageUnavailable,
            },
        };
        self.verify_issued(request.intent(), &snapshot.directory)?;
        lock.validate()?;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_unicode_tail_keeps_valid_text_and_diagnoses_interior_corruption() {
        let root = tempfile::tempdir().unwrap();
        let directory = Directory::open(root.path(), false).unwrap();
        let text = "训练进度\n".repeat(WINDOW_BYTES / 4);
        std::fs::write(root.path().join("stdout.log"), text.as_bytes()).unwrap();
        let tail = window(&directory, "stdout.log");
        assert_eq!(tail.problem, Some(Problem::Truncated));
        assert!(tail.offset_bytes > 0);
        assert!(!tail.text.is_empty() && tail.text.len() <= WINDOW_BYTES);
        tail.validate().unwrap();
        std::fs::write(root.path().join("stdout.log"), b"bad\xffutf8").unwrap();
        assert_eq!(
            window(&directory, "stdout.log").problem,
            Some(Problem::InvalidUtf8)
        );
    }
}
