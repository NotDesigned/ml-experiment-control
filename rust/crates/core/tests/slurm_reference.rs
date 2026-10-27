//! Native differential tests against evaluated unchanged Python pure functions.
//! No scheduler/SSH/process or Python runtime is used when these tests execute.
use ml_exp_core::slurm::*;
use ml_exp_core::state::RunState;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/slurm/python-reference.json")).unwrap()
}
fn sample() -> JobSpec {
    serde_json::from_value(fixture()["render_cases"][0]["spec"].clone()).unwrap()
}
fn ssh() -> SshConnection {
    SshConnection {
        ssh_binary: "ssh".into(),
        rsync_binary: "rsync".into(),
        alias: "cluster-login".into(),
        control_path: Some("/tmp/native-reference-%C".into()),
    }
}
fn scope() -> JobScope {
    JobScope {
        ssh_alias: "cluster-login".into(),
        job_id: JobId::parse("1234").unwrap(),
        scheduler_name: "native-reference--attempt-001".into(),
        partition: "accelerator".into(),
    }
}
fn acct(raw: &str, code: &str) -> String {
    format!("1234|native-reference--attempt-001|accelerator|{raw}|00:01:00|{code}\n")
}
fn queued(raw: &str, reason: &str) -> String {
    format!("1234|native-reference--attempt-001|accelerator|{raw}|00:00|0:0|{reason}\n")
}
fn owned() -> Ownership {
    resolve_ownership(
        "cluster-login",
        "native-reference",
        "attempt-001",
        &"a".repeat(32),
        "native-reference--attempt-001",
        Query::Success(
            "1234|native-reference--attempt-001|ml-exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        ),
        Query::Success(""),
    )
    .unwrap()
}

#[test]
fn fixture_records_exact_baseline_sources() {
    let f = fixture();
    assert_eq!(f["schema_version"], 1);
    assert_eq!(
        f["baseline_commit"],
        "e725faec689fb8395e692e27d00a8896fa7fe5a5"
    );
    let sources = [
        include_bytes!("../../../../src/experiment_control/backends/slurm.py").as_slice(),
        include_bytes!("../../../../server/src/ml_exp_server/runs/allocation_status.py").as_slice(),
        include_bytes!("../../../../src/experiment_control/submission.py").as_slice(),
    ];
    assert_eq!(f["sources"].as_array().unwrap().len(), sources.len());
    for (source, expected) in sources.iter().zip(f["sources"].as_array().unwrap()) {
        assert_eq!(
            format!("{:x}", Sha256::digest(source)),
            expected["sha256"].as_str().unwrap()
        );
    }
}
#[test]
fn normalized_scheduler_states_preserve_python_semantics() {
    let f = fixture();
    assert_eq!(f["normalization_cases"].as_array().unwrap().len(), 100);
    for c in f["normalization_cases"].as_array().unwrap() {
        assert_eq!(
            serde_json::to_value(normalize_state(
                c["raw"].as_str().unwrap(),
                c["exit_code"].as_str()
            ))
            .unwrap(),
            c["expected"],
            "{c:?}"
        );
    }
    assert_eq!(normalize_state("", None), RunState::Unknown);
    assert_eq!(
        normalize_state("unrecognized-state", None),
        RunState::Unknown
    );
}
#[test]
fn shell_quotes_and_long_job_names_match_python() {
    let f = fixture();
    assert_eq!(f["quote_cases"].as_array().unwrap().len(), 25);
    for c in f["quote_cases"].as_array().unwrap() {
        assert_eq!(
            shell_quote(c["value"].as_str().unwrap()).unwrap(),
            c["expected"]
        );
    }
    for c in f["job_name_cases"].as_array().unwrap() {
        assert_eq!(
            scheduler_job_name(
                c["run_id"].as_str().unwrap(),
                c["attempt_id"].as_str().unwrap()
            )
            .unwrap(),
            c["expected"]
        );
    }
    assert!(shell_quote("contains\0nul").is_err());
    assert!(scheduler_job_name("../escape", "attempt-001").is_err());
    assert!(scheduler_job_name("run", "latest").is_err());
}
#[test]
fn rendering_is_byte_exact_for_source_and_managed_oci_inputs() {
    let f = fixture();
    assert_eq!(f["render_cases"].as_array().unwrap().len(), 16);
    for c in f["render_cases"].as_array().unwrap() {
        let spec: JobSpec = serde_json::from_value(c["spec"].clone()).unwrap();
        assert_eq!(
            render_job(&spec, c["token"].as_str()).unwrap(),
            c["expected"],
            "{}",
            c["id"]
        );
    }
    let mut spec = sample();
    spec.command = vec!["/bin/sh".into(), "-c".into(), "echo nothing".into()];
    assert!(render_job(&spec, Some("invalid")).is_err());
    spec.backend.partition = "safe\n#SBATCH --account=foreign".into();
    assert!(render_job(&spec, None).is_err());
}
#[test]
fn typed_configuration_preserves_reference_and_labels_hardening() {
    let f = fixture();
    assert_eq!(f["config_cases"].as_array().unwrap().len(), 24);
    for c in f["config_cases"].as_array().unwrap() {
        let valid = serde_json::from_value::<JobSpec>(c["spec"].clone())
            .map(|s| s.validate().is_ok())
            .unwrap_or(false);
        assert_eq!(valid, c["native_valid"].as_bool().unwrap(), "{}", c["id"]);
        if c["native_valid"] != c["python_valid"] {
            assert!(c["difference"].as_str().is_some());
        }
    }
    let mut s = sample();
    s.backend.kind = "wyd".into();
    assert!(s.validate().is_err());
    assert!(compatible_storage_kind("wyd"));
    assert!(compatible_storage_kind("slurm"));
    assert!(!compatible_storage_kind("sensecore"));
    assert!(delivery_requested(
        &["wyd:asset.reference".into()],
        "asset.reference"
    ));
    assert!(delivery_requested(
        &["slurm:asset.reference".into()],
        "asset.reference"
    ));
    assert!(!delivery_requested(
        &["wyd:asset.foreign".into()],
        "asset.reference"
    ));
}
#[test]
fn lexical_mount_image_resource_and_time_validation_is_explicit() {
    for (value, expected) in [
        ("00:00:01", 1),
        ("24:00:00", 86400),
        ("3-02:01:01", 266461),
        ("999:59:59", 3599999),
    ] {
        assert_eq!(walltime_seconds(value).unwrap(), expected);
    }
    for value in [
        "",
        "00:00:00",
        "1:60:00",
        "1:00:60",
        "1-24:00:00",
        "-1:00:00",
        "1:0:00",
        "1:00:00\n",
        "99999999999999999999:00:00",
    ] {
        assert!(walltime_seconds(value).is_err(), "{value}");
    }
    for path in [
        "relative",
        "/shared/../private",
        "/shared/./run",
        "/shared/a\n",
        "/shared/a:b",
        "/shared/a\\b",
    ] {
        assert!(!absolute_path(path));
    }
    assert!(path_within("/shared/a b", "/shared"));
    assert!(!path_within("/shared-other/a", "/shared"));
    for change in ["gpu", "cpu", "nodes", "mem", "digest", "oci", "path"] {
        let mut s = sample();
        match change {
            "gpu" => s.resources.gpus = 0,
            "cpu" => s.resources.cpus = 0,
            "nodes" => s.resources.nodes = 2,
            "mem" => s.resources.memory_gb = Some(0),
            "digest" => s.image_id = "latest".into(),
            "oci" => {
                s.backend.oci_image =
                    Some(format!("registry.example/image@sha256:{}", "b".repeat(64)))
            }
            _ => s.backend.sif_path = "/foreign/image.sif".into(),
        };
        assert!(s.validate().is_err(), "{change}");
    }
}
#[test]
fn accounting_fixtures_preserve_state_and_real_fields_without_queue_exit_fabrication() {
    let f = fixture();
    assert_eq!(f["accounting_cases"].as_array().unwrap().len(), 9);
    for c in f["accounting_cases"].as_array().unwrap() {
        let source = if c["source"] == "sacct" {
            StatusSource::Sacct
        } else {
            StatusSource::Squeue
        };
        let s = parse_job_status(c["output"].as_str().unwrap(), &scope(), source, None)
            .unwrap()
            .unwrap();
        let e = &c["expected"];
        assert_eq!(serde_json::to_value(s.state).unwrap(), e["state"]);
        assert_eq!(s.raw_state, e["raw_state"]);
        assert_eq!(serde_json::to_value(s.elapsed).unwrap(), e["elapsed"]);
        assert_eq!(serde_json::to_value(s.reason).unwrap(), e["reason"]);
        assert_eq!(
            serde_json::to_value(s.exit_code).unwrap(),
            c["native_exit_code"]
        );
        assert_eq!(s.observed_at, None);
    }
}
#[test]
fn exact_status_parser_rejects_foreign_duplicate_truncated_and_fake_exit_rows() {
    let valid = acct("RUNNING", "0:0");
    for text in [
        valid.replace("1234|", "1235|"),
        valid.replace("native-reference--attempt-001", "foreign"),
        valid.replace("accelerator", "another"),
        format!("{valid}{valid}"),
        valid.replace("|0:0", "|bad"),
        "1234|too|few".into(),
        "1234.batch|native-reference--attempt-001|accelerator|RUNNING|00:01|0:0".into(),
    ] {
        assert!(
            parse_job_status(&text, &scope(), StatusSource::Sacct, None).is_err(),
            "{text}"
        );
    }
    assert!(
        parse_job_status("", &scope(), StatusSource::Sacct, None)
            .unwrap()
            .is_none()
    );
    let status = parse_job_status(
        &queued("RUNNING", "node-private"),
        &scope(),
        StatusSource::Squeue,
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(status.reason, None);
    assert_eq!(status.exit_code, None);
}
#[test]
fn queued_supplement_never_overwrites_accounting_on_foreign_unavailable_or_racing_reply() {
    let text = acct("PENDING", "0:0");
    let output = observe_status(
        &scope(),
        None,
        Query::Success(&text),
        Query::Success(&queued("PENDING", "Priority")),
        Some("2026-10-27T00:00:00Z"),
    )
    .unwrap();
    assert_eq!(output.status.state, RunState::Queued);
    assert_eq!(output.status.reason.as_deref(), Some("Priority"));
    assert_eq!(output.status.observation_source, Some(StatusSource::Sacct));
    assert_eq!(
        output.status.queue.unwrap().availability,
        Availability::Observed
    );
    for query in [
        Query::Failed,
        Query::NotRequested,
        Query::Success("foreign"),
        Query::Success("1235|foreign|accelerator|PENDING|00:00|0:0|Priority\n"),
    ] {
        let result =
            observe_status(&scope(), None, Query::Success(&text), query, Some("now")).unwrap();
        assert_eq!(result.status.state, RunState::Queued);
        assert_eq!(result.status.reason, None);
        assert!(result.diagnostic.is_some());
    }
    let racing = observe_status(
        &scope(),
        None,
        Query::Success(&text),
        Query::Success(&queued("RUNNING", "node-list")),
        None,
    )
    .unwrap();
    assert_eq!(racing.status.state, RunState::Queued);
    assert_eq!(racing.status.reason, None);
    assert_eq!(
        racing.status.queue.unwrap().availability,
        Availability::NotObserved
    );
}
#[test]
fn failed_queries_preserve_exact_previous_state_and_original_observation_time() {
    let previous = parse_job_status(
        &acct("RUNNING", "0:0"),
        &scope(),
        StatusSource::Sacct,
        Some("old"),
    )
    .unwrap()
    .unwrap();
    for bad in [
        Query::Failed,
        Query::Success("1235|foreign|accelerator|FAILED|00:01|1:0\n"),
    ] {
        let result = observe_status(
            &scope(),
            Some(&previous),
            bad,
            Query::Success(&queued("PENDING", "Priority")),
            Some("new"),
        )
        .unwrap();
        assert!(result.retained_previous);
        assert_eq!(result.status, previous);
        assert_eq!(result.checked_at.as_deref(), Some("new"));
    }
    let unknown =
        observe_status(&scope(), None, Query::Failed, Query::Failed, Some("new")).unwrap();
    assert_eq!(unknown.status.state, RunState::Unknown);
    assert_eq!(unknown.status.observed_at, None);
    assert_eq!(unknown.status.exit_code, None);
    let fallback = observe_status(
        &scope(),
        None,
        Query::Success(""),
        Query::Success(&queued("PENDING", "Resources")),
        None,
    )
    .unwrap();
    assert_eq!(fallback.status.state, RunState::Queued);
    assert_eq!(fallback.status.exit_code, None);
    let mut foreign = previous.clone();
    foreign.ssh_alias = "other".into();
    assert!(observe_status(&scope(), Some(&foreign), Query::Failed, Query::Failed, None).is_err());
}
#[test]
fn identity_requires_complete_successful_queries_and_exact_owned_token() {
    let name = "native-reference--attempt-001";
    let marker = format!("ml-exp-{}", "a".repeat(32));
    let row = format!("1234|{name}|{marker}\n");
    fn resolve(q: Query<'_>, a: Query<'_>) -> Result<Ownership, Error> {
        resolve_ownership(
            "cluster-login",
            "native-reference",
            "attempt-001",
            &"a".repeat(32),
            "native-reference--attempt-001",
            q,
            a,
        )
    }
    let absent = resolve(Query::Success(""), Query::Success("")).unwrap();
    assert_eq!(absent.state, OwnershipState::Absent);
    let exact = resolve(Query::Success(&row), Query::Success(&row)).unwrap();
    assert_eq!(exact.exact_job().unwrap().as_str(), "1234");
    let duplicate = format!("{row}{}", row.replace("1234", "1235"));
    assert_eq!(
        resolve(Query::Success(&duplicate), Query::Success(""))
            .unwrap()
            .state,
        OwnershipState::Ambiguous
    );
    for other in [
        row.replace(&marker, "ml-exp-foreign"),
        row.replace(name, "foreign-name"),
    ] {
        assert_eq!(
            resolve(Query::Success(&other), Query::Success(""))
                .unwrap()
                .state,
            OwnershipState::ForeignCollision
        );
    }
    assert!(resolve(Query::Success(&format!("{row}{row}")), Query::Success("")).is_err());
    assert!(
        resolve(
            Query::Success(&row),
            Query::Success(&row.replace(&marker, "foreign"))
        )
        .is_err()
    );
    for query in [Query::Failed, Query::NotRequested] {
        assert!(resolve(Query::Success(&row), query).is_err());
    }
    assert!(
        resolve_ownership(
            "cluster-login",
            "native-reference",
            "attempt-001",
            &"a".repeat(32),
            "wrong-name",
            Query::Success(""),
            Query::Success("")
        )
        .is_err()
    );
    assert!(parse_identity_rows("1234|missing-comment").is_err());
}
#[test]
fn submit_plan_contains_exact_ordered_effects_and_never_executes_or_replays() {
    let spec = sample();
    let intent = SubmissionIntent {
        submission_token: "a".repeat(32),
        request: submission_request(&spec.run_id, &spec.attempt_id).unwrap(),
    };
    let p = plan_submit(
        &spec,
        &intent,
        &ssh(),
        "/reference/manifest.yaml",
        "/reference/attempt-001/job.sbatch",
    )
    .unwrap();
    assert_eq!(
        p.steps.iter().map(|c| c.effect).collect::<Vec<_>>(),
        [
            Effect::ClaimSubmission,
            Effect::CopyManifest,
            Effect::CopyScript,
            Effect::Submit
        ]
    );
    assert!(
        p.steps[0]
            .argv
            .last()
            .unwrap()
            .contains("mkdir /shared/project/runs/native-reference/.submission-attempt-001")
    );
    assert_eq!(
        p.steps[1].argv.last().unwrap(),
        "cluster-login:/shared/project/runs/native-reference/manifest.yaml"
    );
    assert!(
        p.steps[2]
            .argv
            .last()
            .unwrap()
            .ends_with("controller-attempt-001.sbatch")
    );
    assert_eq!(
        p.steps[3].argv.last().unwrap(),
        "sbatch --parsable /shared/project/runs/native-reference/controller-attempt-001.sbatch"
    );
    assert_eq!(p.script_mode, 0o600);
    assert!(
        p.requires_authorization && p.requires_prior_reconciliation && !p.automatic_replay_allowed
    );
    assert!(p.steps.iter().all(|p| p.env.is_empty()));
    let mut conflict = intent.clone();
    conflict.request.scheduler_name = "foreign".into();
    assert!(plan_submit(&spec, &conflict, &ssh(), "/a", "/b").is_err());
    let mut foreign = ssh();
    foreign.alias = "other-cluster".into();
    assert!(plan_submit(&spec, &intent, &foreign, "/a", "/b").is_err());
    assert_eq!(claim_result(Some(0)), ClaimResult::Fresh);
    assert_eq!(
        claim_result(Some(1)),
        ClaimResult::ExistingReconcileRequired
    );
    assert_eq!(claim_result(None), ClaimResult::UnknownDoNotRetry);
    assert_eq!(claim_result(Some(255)), ClaimResult::UnknownDoNotRetry);
}
#[test]
fn cancellation_is_exact_scoped_and_terminal_is_no_op() {
    let own = owned();
    let status = parse_job_status(&acct("RUNNING", "0:0"), &scope(), StatusSource::Sacct, None)
        .unwrap()
        .unwrap();
    let CancelPlan::Execute(p) = plan_cancel(&ssh(), &own, &status).unwrap() else {
        panic!("missing cancel plan")
    };
    assert_eq!(p.effect, Effect::Cancel);
    assert_eq!(p.argv.last().unwrap(), "scancel 1234");
    let terminal = parse_job_status(
        &acct("COMPLETED", "0:0"),
        &scope(),
        StatusSource::Sacct,
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        plan_cancel(&ssh(), &own, &terminal).unwrap(),
        CancelPlan::AlreadyTerminal
    );
    let mut wrong = ssh();
    wrong.alias = "other".into();
    assert!(plan_cancel(&wrong, &own, &status).is_err());
    let mut foreign = status.clone();
    foreign.job_id = JobId::parse("1235").unwrap();
    assert!(plan_cancel(&ssh(), &own, &foreign).is_err());
    let mut ambiguous = own.clone();
    ambiguous.state = OwnershipState::Ambiguous;
    assert!(plan_cancel(&ssh(), &ambiguous, &status).is_err());
}
#[test]
fn parsable_submit_response_and_read_only_query_plans_are_bounded() {
    assert_eq!(
        parse_submit_response("1234;cluster\n").unwrap().as_str(),
        "1234"
    );
    for value in [
        "",
        "1234\n1235",
        "-123",
        "123_4",
        "1234;",
        "1234;cluster;another",
        "1234;bad\nname",
    ] {
        assert!(parse_submit_response(value).is_err(), "{value}");
    }
    let s = ssh();
    let id = JobId::parse("1234").unwrap();
    for plan in [
        accounting_plan(&s, &id).unwrap(),
        queue_status_plan(&s, &id).unwrap(),
        allocation_plan(&s, "accelerator").unwrap(),
        log_probe_plan(&s, &["/shared/a b.log".into()], 20).unwrap(),
    ] {
        assert_eq!(plan.effect, Effect::Observe);
        assert!(!plan.argv.last().unwrap().contains("sbatch "));
        assert!(!plan.argv.last().unwrap().contains("scancel "));
        assert!(plan.max_stdout_bytes <= 1024 * 1024);
    }
    assert_eq!(identity_plans(&s, "user", "2026-10-01").unwrap().len(), 2);
    assert!(identity_plans(&s, "-option", "2026-10-01").is_err());
    assert!(identity_plans(&s, "user", "now-30days").is_err());
    assert!(allocation_plan(&s, "gpu;touch").is_err());
    assert!(log_probe_plan(&s, &[], 10).is_err());
    assert!(log_probe_plan(&s, &["/a".into()], 0).is_err());
    assert!(!predicate_result(Some(1)).unwrap());
    assert!(predicate_result(Some(255)).is_err());
}
#[test]
fn partition_and_association_evidence_checks_exact_user_account_partition_qos() {
    let spec = sample();
    let info = "accelerator*|up|3-00:00:00|gpu:accelerator:8(S:0)\n";
    let assoc = "user|lab||normal,high|normal\n";
    let rows = parse_sinfo(info).unwrap();
    assert_eq!(rows[0].partition, "accelerator");
    assert!(validate_live_evidence(&spec.backend, "user", info, assoc).is_ok());
    for bad in [
        info.replace("up", "down"),
        info.replace("gpu:accelerator", "gpu:another"),
        info.replace("accelerator*", "foreign*"),
    ] {
        assert!(validate_live_evidence(&spec.backend, "user", &bad, assoc).is_err());
    }
    for bad in [
        assoc.replace("user|", "foreign|"),
        assoc.replace("lab|", "foreign|"),
        "user|lab|other|normal|normal\n".into(),
        "user|lab||normal-extra|normal\n".into(),
    ] {
        assert!(validate_live_evidence(&spec.backend, "user", info, &bad).is_err());
    }
    assert!(parse_sinfo("malformed").is_err());
    assert!(parse_associations("user|too-few").is_err());
}
#[test]
fn allocation_fixtures_preserve_units_filtering_down_nodes_and_unknown_failures() {
    let f = fixture();
    assert_eq!(f["allocation_cases"].as_array().unwrap().len(), 14);
    for c in f["allocation_cases"].as_array().unwrap() {
        let result = parse_allocation(
            c["output"].as_str().unwrap(),
            c["partition"].as_str().unwrap(),
        );
        assert_eq!(result.is_ok(), c["valid"].as_bool().unwrap(), "{}", c["id"]);
        if let Ok(result) = result {
            assert_eq!(
                serde_json::to_value(result).unwrap(),
                c["expected"],
                "{}",
                c["id"]
            );
        }
    }
    let valid = f["allocation_cases"][0]["output"].as_str().unwrap();
    let snapshot = parse_allocation(valid, "gpu").unwrap();
    let previous = allocation_observation(None, Ok(snapshot.clone()), Some("old"));
    let stale = allocation_observation(Some(&previous), Err(Error::Unavailable), Some("new"));
    assert_eq!(stale.status, SnapshotState::Stale);
    assert_eq!(stale.observed_at.as_deref(), Some("old"));
    assert_eq!(stale.data, Some(snapshot));
    assert!(!stale.admission_guaranteed);
    let unknown = allocation_observation(None, Err(Error::Unavailable), None);
    assert_eq!(unknown.status, SnapshotState::Unknown);
    assert_eq!(unknown.data, None);
    assert_eq!(unknown.observed_at, None);
    let overflow = valid.replace("RealMemory=1024", "RealMemory=18446744073709551615");
    assert!(parse_allocation(&overflow, "gpu").is_err());
    assert!(parse_allocation(&valid.replace("CPUAlloc=4", "CPUAlloc=17"), "gpu").is_err());
    assert!(parse_allocation(&format!("{valid}private username\n"), "gpu").is_err());
    assert_eq!(
        serde_json::to_value(previous).unwrap()["admission_guaranteed"],
        json!(false)
    );
}

#[test]
fn terminal_accounting_cannot_be_regressed_by_a_later_stale_queued_reply() {
    let completed = parse_job_status(
        &acct("COMPLETED", "0:0"),
        &scope(),
        StatusSource::Sacct,
        Some("completed-time"),
    )
    .unwrap()
    .unwrap();
    let queued_text = acct("PENDING", "0:0");
    for accounting in [Query::Success(queued_text.as_str()), Query::Success("")] {
        let observed = observe_status(
            &scope(),
            Some(&completed),
            accounting,
            Query::Success(&queued("PENDING", "Priority")),
            Some("checked-time"),
        )
        .unwrap();
        assert_eq!(observed.status, completed);
        assert_eq!(observed.diagnostic, Some(Diagnostic::RegressiveState));
        assert!(observed.retained_previous);
    }
    assert!(identity_plans(&ssh(), "user", "2026-02-30").is_err());
    assert!(identity_plans(&ssh(), "user", "2026-13-01").is_err());
}
