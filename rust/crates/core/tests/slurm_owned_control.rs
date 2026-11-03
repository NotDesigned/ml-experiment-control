//! Pure parsing and protocol binding; no scheduler or filesystem effects.
use ml_exp_core::{
    slurm::{self, JobId, JobScope, Query, StatusSource, helper::*},
    state::RunState,
};
fn scope() -> JobScope {
    JobScope {
        ssh_alias: "slurm-test".into(),
        job_id: JobId::parse("12345").unwrap(),
        scheduler_name: "train--attempt-001".into(),
        partition: "h100".into(),
    }
}
fn intent() -> FrozenIntent {
    FrozenIntent {
        staging: None,
        target: Target {
            backend_id: "slurm-test".into(),
            profile_sha256: sha256_bytes(b"profile"),
            storage_scope: sha256_bytes(b"storage"),
        },
        scope: Scope {
            project: "demo".into(),
            run_id: "train".into(),
            attempt_id: "attempt-001".into(),
        },
        submission_token: "a".repeat(32),
        scheduler_name: "train--attempt-001".into(),
        manifest_sha256: sha256_bytes(b"manifest"),
        script_sha256: sha256_bytes(b"script"),
    }
}
fn row(state: &str) -> String {
    format!(
        "12345|train--attempt-001|ml-exp-{}|h100|{state}|0:01|0:0",
        "a".repeat(32)
    )
}
#[test]
fn ownership_is_part_of_the_exact_single_status_row() {
    let row = row("RUNNING");
    let status = slurm::parse_owned_job_status(
        &row,
        &scope(),
        &"a".repeat(32),
        StatusSource::Sacct,
        Some("2026-10-28T00:00:00Z"),
    )
    .unwrap()
    .unwrap();
    status.validate().unwrap();
    assert_eq!(status.state, RunState::Running);
    for bad in [
        row.replace("12345|", "12345.batch|"),
        row.replace("12345|", "67890|"),
        row.replace("train--attempt-001", "foreign"),
        row.replace("|h100|", "|h200|"),
        row.replace(&"a".repeat(32), &"b".repeat(32)),
        format!("{row}\n{row}"),
        row.replacen("|0:0", "", 1),
    ] {
        assert!(
            slurm::parse_owned_job_status(
                &bad,
                &scope(),
                &"a".repeat(32),
                StatusSource::Sacct,
                None
            )
            .is_err()
        );
    }
}
#[test]
fn queue_failure_or_foreign_reply_retains_exact_accounting_without_inventing_exit() {
    let acct = row("PENDING");
    let queue = format!("{acct}|Resources");
    let valid = slurm::observe_owned_status(
        &scope(),
        &"a".repeat(32),
        None,
        Query::Success(&acct),
        Query::Success(&queue),
        None,
    )
    .unwrap();
    assert_eq!(valid.status.reason.as_deref(), Some("Resources"));
    valid.status.validate().unwrap();
    let from_queue = slurm::observe_owned_status(
        &scope(),
        &"a".repeat(32),
        None,
        Query::Success(""),
        Query::Success(&queue),
        None,
    )
    .unwrap();
    assert!(from_queue.status.exit_code.is_none());
    from_queue.status.validate().unwrap();
    for query in [Query::Failed, Query::Success("foreign"), Query::Success("")] {
        let actual = slurm::observe_owned_status(
            &scope(),
            &"a".repeat(32),
            None,
            Query::Success(&acct),
            query,
            None,
        )
        .unwrap();
        assert_eq!(actual.status.state, RunState::Queued);
        assert!(actual.diagnostic.is_some());
        actual.status.validate().unwrap();
    }
}
#[test]
fn protocol_binds_job_operation_and_refuses_known_token_echoes() {
    let request = Request::observe_job(intent(), "12345".into()).unwrap();
    let observation = slurm::observe_owned_status(
        &scope(),
        &"a".repeat(32),
        None,
        Query::Success(&row("RUNNING")),
        Query::NotRequested,
        None,
    )
    .unwrap();
    let outcome = ReplyOutcome::JobObserved {
        observation: Box::new(observation.clone()),
    };
    let reply = Reply::for_request(&request, outcome.clone()).unwrap();
    assert_eq!(
        decode_reply(&encode_reply(&reply).unwrap(), &request).unwrap(),
        reply
    );
    let wrong = Request::observe_job(intent(), "67890".into()).unwrap();
    assert!(Reply::for_request(&wrong, outcome.clone()).is_err());
    assert!(decode_reply(&encode_reply(&reply).unwrap(), &wrong).is_err());
    assert!(Reply::for_request(&Request::submit(intent()).unwrap(), outcome).is_err());
    assert!(Request::cancel_job(intent(), "12345_1".into()).is_err());
    assert!(Request::cancel_job(intent(), "--help".into()).is_err());
    let mut bad = observation;
    bad.status.state = RunState::Succeeded;
    assert!(
        Reply::for_request(
            &request,
            ReplyOutcome::JobObserved {
                observation: Box::new(bad)
            }
        )
        .is_err()
    );
    let mut pending = row("PENDING");
    pending.push('|');
    pending.push_str(&"a".repeat(32));
    let echo = slurm::observe_owned_status(
        &scope(),
        &"a".repeat(32),
        None,
        Query::Success(""),
        Query::Success(&pending),
        None,
    )
    .unwrap();
    assert!(
        Reply::for_request(
            &request,
            ReplyOutcome::JobObserved {
                observation: Box::new(echo)
            }
        )
        .is_err()
    );
}

#[test]
fn accounting_without_comment_needs_an_independently_owned_queue_row() {
    let acct = row("PENDING").replace(&format!("ml-exp-{}", "a".repeat(32)), "");
    let queue = format!("{}|Priority", row("PENDING"));
    let observed = slurm::observe_owned_status(
        &scope(),
        &"a".repeat(32),
        None,
        Query::Success(&acct),
        Query::Success(&queue),
        None,
    )
    .unwrap();
    observed.status.validate().unwrap();
    assert_eq!(
        observed.status.observation_source,
        Some(StatusSource::Squeue)
    );
    assert_eq!(observed.status.reason.as_deref(), Some("Priority"));
    assert!(observed.status.exit_code.is_none());
    assert_eq!(observed.diagnostic, None);
    let missing = slurm::observe_owned_status(
        &scope(),
        &"a".repeat(32),
        None,
        Query::Success(&acct),
        Query::NotRequested,
        None,
    )
    .unwrap();
    assert_eq!(
        missing.diagnostic,
        Some(slurm::Diagnostic::OwnershipNotRecorded)
    );
    assert_eq!(missing.status.observation_source, None);
    for bad in [
        acct.replace("12345|", "98765|"),
        format!("{acct}\n{acct}"),
        acct.replace("|h100|", "|h200|"),
        row("PENDING").replace(&"a".repeat(32), &"b".repeat(32)),
    ] {
        let refused = slurm::observe_owned_status(
            &scope(),
            &"a".repeat(32),
            None,
            Query::Success(&bad),
            Query::Success(&queue),
            None,
        )
        .unwrap();
        assert_eq!(
            refused.diagnostic,
            Some(slurm::Diagnostic::MalformedOrForeignEvidence)
        );
        assert_eq!(refused.status.observation_source, None);
    }
    let bad_queue = queue.replace(&"a".repeat(32), &"b".repeat(32));
    let refused = slurm::observe_owned_status(
        &scope(),
        &"a".repeat(32),
        Some(&observed.status),
        Query::Success(&acct),
        Query::Success(&bad_queue),
        None,
    )
    .unwrap();
    assert!(refused.retained_previous);
    assert_eq!(refused.status, observed.status);
}

#[test]
fn controller_status_keeps_identity_state_and_exit_in_one_exact_row() {
    let text = format!(
        "JobId=12345 JobName=train--attempt-001 UserId=test(1000) Comment=ml-exp-{}\n",
        "a".repeat(32)
    );
    let text = text.trim_end().to_string()
        + " Partition=h100 JobState=CANCELLED RunTime=00:00:03 ExitCode=0:15 Reason=None";
    let status = slurm::parse_owned_control_status(
        &text,
        &scope(),
        &"a".repeat(32),
        Some("2026-10-30T00:00:00Z"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(status.state, RunState::Cancelled);
    assert_eq!(status.observation_source, Some(StatusSource::Scontrol));
    assert_eq!(status.exit_code.as_deref(), Some("0:15"));
    status.validate().unwrap();
    for bad in [
        text.replace("12345", "67890"),
        text.replace("Partition=h100", "Partition=h200"),
        text.replace(&"a".repeat(32), &"b".repeat(32)),
        format!("{text} JobId=12345"),
        format!("{text}\n{text}"),
        text.replace("ExitCode=0:15", ""),
        text.replace("Reason=None", "Reason=a|b"),
        text.replace("JobName=train--attempt-001", "JobName=foreign"),
    ] {
        assert!(slurm::parse_owned_control_status(&bad, &scope(), &"a".repeat(32), None).is_err());
    }
    assert!(
        slurm::parse_owned_control_status("", &scope(), &"a".repeat(32), None)
            .unwrap()
            .is_none()
    );
}
