//! Fixed-tool effects around pure ownership and cancellation decisions.
use super::*;
use ml_exp_core::{
    slurm::{Diagnostic, JobId, JobScope, Query, StatusObservation, observe_owned_status},
    submission::{
        CancelDecision, CancelIdentity, CancelStatus, ExactTarget, TargetIdentity, plan_cancel,
    },
};
use serde_json::Value;

const CANCEL_START: &str = "cancel.started.json";
const CANCEL_STATE: &str = "cancel-intent.json";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CancelRecord {
    version: u16,
    intent_sha256: String,
    intent: Value,
}

impl Helper {
    pub(super) fn job_operation(
        &self,
        request: &Request,
        snapshot: Snapshot,
        lock: &Lock,
        cancelled: &AtomicBool,
    ) -> Result<ReplyOutcome> {
        let Some(receipt) = &snapshot.receipt else {
            return Ok(if snapshot.start_exists {
                ReplyOutcome::Unknown {
                    reason: UnknownReason::SubmitStarted,
                }
            } else {
                ReplyOutcome::Rejected {
                    code: Rejection::Unprepared,
                }
            });
        };
        if request.requested_job_id() != Some(receipt.job_id.as_str()) {
            return Ok(ReplyOutcome::Rejected {
                code: Rejection::IdentityConflict,
            });
        }
        let Some(tools) = &self.config.scheduler else {
            return Ok(ReplyOutcome::Rejected {
                code: Rejection::UnsupportedOperation,
            });
        };
        self.check_job_authority(request, &snapshot, lock)?;
        let deadline = Instant::now() + self.config.execution_timeout;
        if request.operation() == OperationTag::ObserveJob {
            return self.observe_job(request, &snapshot, lock, tools, deadline, cancelled);
        }
        let identity = CancelIdentity {
            target: TargetIdentity {
                project: request.intent().scope.project.clone(),
                run_id: request.intent().scope.run_id.clone(),
                attempt_id: request.intent().scope.attempt_id.clone(),
            },
            backend: Some(request.intent().target.backend_id.clone()),
            backend_job_id: receipt.job_id.clone(),
        };
        let has_start = snapshot.directory.entry_exists(OsStr::new(CANCEL_START))?;
        let has_state = snapshot.directory.entry_exists(OsStr::new(CANCEL_STATE))?;
        if has_start != has_state {
            return Err(Error::Metadata);
        }
        let existing = if has_start {
            let start: Started = read_json(&snapshot.directory, CANCEL_START)?;
            let record: CancelRecord = read_json(&snapshot.directory, CANCEL_STATE)?;
            if start.version != helper::PROTOCOL_VERSION
                || record.version != helper::PROTOCOL_VERSION
                || start.intent_sha256 != request.intent().digest()?
                || record.intent_sha256 != start.intent_sha256
            {
                return Err(Error::Metadata);
            }
            Some(record.intent)
        } else {
            None
        };
        let now = utc_now()?;
        // Validate persisted intent before any process, including scheduler reads.
        let initial =
            plan_cancel(&identity, existing.as_ref(), None, &now).map_err(|_| Error::Metadata)?;
        if let CancelDecision::AlreadyVerified { result } = initial {
            return checked_terminal(request, result);
        }
        let before = self.observe_job(request, &snapshot, lock, tools, deadline, cancelled)?;
        let ReplyOutcome::JobObserved { observation } = before else {
            return Ok(if has_start { pending(receipt) } else { before });
        };
        if !has_start && observation.status.state.is_terminal() {
            return Ok(ReplyOutcome::JobObserved { observation });
        }
        let status = CancelStatus::Exact {
            target: ExactTarget {
                identity: identity.target.clone(),
            },
            job_id: receipt.job_id.clone(),
            state: serde_json::to_value(observation.status.state)?
                .as_str()
                .ok_or(Error::Metadata)?
                .into(),
            result: serde_json::to_value(&observation)?,
        };
        let decision = plan_cancel(&identity, existing.as_ref(), Some(&status), &now)
            .map_err(|_| Error::Metadata)?;
        match decision {
            CancelDecision::VerifyTerminal { intent, result } => {
                let outcome = checked_terminal(request, result)?;
                self.save_cancel(request, &snapshot, lock, intent, false)?;
                Ok(outcome)
            }
            CancelDecision::CreateRequested {
                intent,
                issue_job_id,
            } => {
                self.save_cancel(request, &snapshot, lock, intent.clone(), true)?;
                // REQUESTED is durable first. A lost response, failure or timeout
                // never authorizes a second scancel, even if it never spawned.
                let argv = vec![tools.scancel_program.display().to_string(), issue_job_id];
                let _ =
                    self.scheduler_command(request, &snapshot, lock, &argv, deadline, cancelled)?;
                let after =
                    self.observe_job(request, &snapshot, lock, tools, deadline, cancelled)?;
                if let ReplyOutcome::JobObserved { observation } = after {
                    let result = serde_json::to_value(&observation)?;
                    let status = CancelStatus::Exact {
                        target: ExactTarget {
                            identity: identity.target.clone(),
                        },
                        job_id: receipt.job_id.clone(),
                        state: serde_json::to_value(observation.status.state)?
                            .as_str()
                            .ok_or(Error::Metadata)?
                            .into(),
                        result,
                    };
                    if let CancelDecision::VerifyTerminal { intent, result } =
                        plan_cancel(&identity, Some(&intent), Some(&status), &utc_now()?)
                            .map_err(|_| Error::Metadata)?
                    {
                        let outcome = checked_terminal(request, result)?;
                        self.save_cancel(request, &snapshot, lock, intent, false)?;
                        return Ok(outcome);
                    }
                }
                Ok(pending(receipt))
            }
            CancelDecision::Wait { .. } => Ok(pending(receipt)),
            CancelDecision::AlreadyVerified { result } => checked_terminal(request, result),
        }
    }

    fn check_job_authority(
        &self,
        request: &Request,
        snapshot: &Snapshot,
        lock: &Lock,
    ) -> Result<()> {
        self.verify_issued(request.intent(), &snapshot.directory)?;
        lock.validate()?;
        match self.read_ledger(request.intent())? {
            ReadLedger::Found(current)
                if current
                    .receipt
                    .as_ref()
                    .is_some_and(|r| Some(r.job_id.as_str()) == request.requested_job_id()) =>
            {
                Ok(())
            }
            _ => Err(Error::Metadata),
        }
    }
    fn save_cancel(
        &self,
        request: &Request,
        snapshot: &Snapshot,
        lock: &Lock,
        intent: Value,
        new: bool,
    ) -> Result<()> {
        self.check_job_authority(request, snapshot, lock)?;
        let record = json(&CancelRecord {
            version: helper::PROTOCOL_VERSION,
            intent_sha256: request.intent().digest()?,
            intent,
        })?;
        if new {
            snapshot.directory.publish_new(
                OsStr::new(CANCEL_START),
                &json(&Started {
                    version: helper::PROTOCOL_VERSION,
                    intent_sha256: request.intent().digest()?,
                })?,
            )?;
            snapshot
                .directory
                .publish_new(OsStr::new(CANCEL_STATE), &record)?;
        } else {
            snapshot
                .directory
                .replace_regular(OsStr::new(CANCEL_STATE), &record)?;
        }
        self.check_job_authority(request, snapshot, lock)
    }
    fn scheduler_command(
        &self,
        request: &Request,
        snapshot: &Snapshot,
        lock: &Lock,
        argv: &[String],
        deadline: Instant,
        cancelled: &AtomicBool,
    ) -> Result<Option<String>> {
        self.check_job_authority(request, snapshot, lock)?;
        if Instant::now() >= deadline || cancelled.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let local = process::execute(process::Request {
            argv,
            cwd: &self.config.root,
            environment: process::Environment::Replace(&self.config.environment),
            stdin: process::Input::Null,
            stdout: process::Capture::Memory { limit: 1024 * 1024 },
            stderr: process::Capture::Memory { limit: 0 },
            deadline,
            termination_grace: Duration::from_millis(250),
            drain_grace: Duration::from_millis(100),
            cancelled,
        })?;
        self.check_job_authority(request, snapshot, lock)?;
        if local.exit_code != Some(0)
            || local.timed_out
            || local.cancelled
            || local.failure.is_some()
            || !local.stdout.complete
            || local.stdout.truncated
        {
            return Ok(None);
        }
        Ok(String::from_utf8(local.stdout.bytes).ok())
    }
    fn observe_job(
        &self,
        request: &Request,
        snapshot: &Snapshot,
        lock: &Lock,
        tools: &SchedulerTools,
        deadline: Instant,
        cancelled: &AtomicBool,
    ) -> Result<ReplyOutcome> {
        let scope = JobScope {
            ssh_alias: request.intent().target.backend_id.clone(),
            job_id: JobId::parse(request.requested_job_id().ok_or(Error::Configuration)?)
                .map_err(|_| Error::Configuration)?,
            scheduler_name: request.intent().scheduler_name.clone(),
            partition: tools.partition.clone(),
        };
        let acct = self.scheduler_command(
            request,
            snapshot,
            lock,
            &[
                tools.sacct_program.display().to_string(),
                "-j".into(),
                scope.job_id.as_str().into(),
                "-X".into(),
                "-n".into(),
                "-P".into(),
                "-o".into(),
                "JobIDRaw,JobName%128,Comment%256,Partition%128,State%64,Elapsed,ExitCode".into(),
            ],
            deadline,
            cancelled,
        )?;
        let at = utc_now()?;
        let first = observe_owned_status(
            &scope,
            &request.intent().submission_token,
            None,
            query(&acct),
            Query::NotRequested,
            Some(&at),
        )
        .map_err(|_| Error::Metadata)?;
        let queue = if matches!(first.diagnostic, Some(Diagnostic::NoEvidence))
            || first.status.state == ml_exp_core::state::RunState::Queued
        {
            Some(self.scheduler_command(
                request,
                snapshot,
                lock,
                &[
                    tools.squeue_program.display().to_string(),
                    "-j".into(),
                    scope.job_id.as_str().into(),
                    "-h".into(),
                    "-o".into(),
                    "%i|%j|%k|%P|%T|%M|0:0|%R".into(),
                ],
                deadline,
                cancelled,
            )?)
        } else {
            None
        };
        let at = utc_now()?;
        let observation = observe_owned_status(
            &scope,
            &request.intent().submission_token,
            None,
            query(&acct),
            queue.as_ref().map_or(Query::NotRequested, query),
            Some(&at),
        )
        .map_err(|_| Error::Metadata)?;
        if observation.status.observation_source.is_none() {
            return Ok(ReplyOutcome::Unknown {
                reason: match observation.diagnostic {
                    Some(Diagnostic::MalformedOrForeignEvidence) => {
                        UnknownReason::SchedulerEvidenceInvalid
                    }
                    Some(Diagnostic::QueryUnavailable) => UnknownReason::SchedulerUnavailable,
                    _ => UnknownReason::SchedulerNotObserved,
                },
            });
        }
        let outcome = ReplyOutcome::JobObserved {
            observation: Box::new(observation),
        };
        // Validate binding and known-token non-echo before an observation can
        // be persisted or become part of a Debug-printable reply.
        Reply::for_request(request, outcome.clone())?;
        Ok(outcome)
    }
}
fn utc_now() -> Result<String> {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| Error::Metadata)
}
fn query(value: &Option<String>) -> Query<'_> {
    value.as_deref().map_or(Query::Failed, Query::Success)
}
fn pending(receipt: &Receipt) -> ReplyOutcome {
    ReplyOutcome::CancelPending {
        job_id: receipt.job_id.clone(),
    }
}
fn checked_terminal(request: &Request, result: Value) -> Result<ReplyOutcome> {
    let observation: StatusObservation = serde_json::from_value(result)?;
    if !observation.status.state.is_terminal() {
        return Err(Error::Metadata);
    }
    let outcome = ReplyOutcome::JobObserved {
        observation: Box::new(observation),
    };
    Reply::for_request(request, outcome.clone())?;
    Ok(outcome)
}
