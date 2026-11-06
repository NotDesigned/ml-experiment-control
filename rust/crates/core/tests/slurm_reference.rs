//! Slurm pure-protocol tests. No scheduler, SSH or process is used here.
use ml_exp_core::slurm::*;
use ml_exp_core::state::RunState;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/slurm/cases.json")).unwrap()
}
fn sample() -> JobSpec {
    serde_json::from_value(fixture()["render_cases"][0]["spec"].clone()).unwrap()
}
#[test]
fn user_environment_is_literal_and_bound_to_the_managed_launch() {
    let mut spec = sample();
    spec.execution.managed_io = true;
    spec.execution
        .env
        .insert("LABEL".into(), "two words; $(literal)".into());
    let scope = ml_exp_core::slurm::helper::Scope {
        project: "demo".into(),
        run_id: spec.run_id.clone(),
        attempt_id: spec.attempt_id.clone(),
    };
    let runtime = ml_exp_core::slurm::managed::Runtime {
        program: "/opt/ml-expd/worker".into(),
        base_environment: std::collections::BTreeMap::from([(
            "PATH".into(),
            "/usr/bin:/bin".into(),
        )]),
        runtime_environment: vec![],
    };
    let package = ml_exp_core::slurm::managed::plan(&scope, &spec, &runtime).unwrap();
    assert_eq!(
        package.manifest.environment["LABEL"],
        "two words; $(literal)"
    );
    assert!(
        render_job_with_datasets(&spec, None, &[])
            .unwrap()
            .contains("'LABEL=two words; $(literal)'")
    );
    let binding = package.config.binding().unwrap();
    spec.execution
        .env
        .insert("LABEL".into(), "different".into());
    let changed = ml_exp_core::slurm::managed::plan(&scope, &spec, &runtime).unwrap();
    assert_ne!(binding.sha256, changed.config.binding().unwrap().sha256);
    spec.execution
        .env
        .insert("STATE_DIR".into(), "/other".into());
    assert!(render_job_with_datasets(&spec, None, &[]).is_err());
}
fn ssh() -> SshConnection {
    SshConnection {
        ssh_binary: "ssh".into(),
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
fn owned_acct(raw: &str, code: &str) -> String {
    let marker = format!("ml-exp-{}", "a".repeat(32));
    format!("1234|native-reference--attempt-001|{marker}|accelerator|{raw}|00:01:00|{code}\n")
}
fn owned_queued(raw: &str, reason: &str) -> String {
    let marker = format!("ml-exp-{}", "a".repeat(32));
    format!("1234|native-reference--attempt-001|{marker}|accelerator|{raw}|00:00|0:0|{reason}\n")
}
fn observe(
    previous: Option<&JobStatus>,
    accounting: Query<'_>,
    queue: Query<'_>,
    at: Option<&str>,
) -> Result<StatusObservation, Error> {
    observe_owned_status(&scope(), &"a".repeat(32), previous, accounting, queue, at)
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
            render_job_with_datasets(&spec, c["token"].as_str(), &[]).unwrap(),
            c["expected"],
            "{}",
            c["id"]
        );
    }
    let mut spec = sample();
    spec.command = vec!["/bin/sh".into(), "-c".into(), "echo nothing".into()];
    assert!(render_job_with_datasets(&spec, Some("invalid"), &[]).is_err());
    spec.backend.partition = "safe\n#SBATCH --account=foreign".into();
    assert!(render_job_with_datasets(&spec, None, &[]).is_err());
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
    let text = owned_acct("PENDING", "0:0");
    let output = observe(
        None,
        Query::Success(&text),
        Query::Success(&owned_queued("PENDING", "Priority")),
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
        let result = observe(None, Query::Success(&text), query, Some("now")).unwrap();
        assert_eq!(result.status.state, RunState::Queued);
        assert_eq!(result.status.reason, None);
        assert!(result.diagnostic.is_some());
    }
    let racing = observe(
        None,
        Query::Success(&text),
        Query::Success(&owned_queued("RUNNING", "node-list")),
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
        let result = observe(
            Some(&previous),
            bad,
            Query::Success(&owned_queued("PENDING", "Priority")),
            Some("new"),
        )
        .unwrap();
        assert!(result.retained_previous);
        assert_eq!(result.status, previous);
        assert_eq!(result.checked_at.as_deref(), Some("new"));
    }
    let unknown = observe(None, Query::Failed, Query::Failed, Some("new")).unwrap();
    assert_eq!(unknown.status.state, RunState::Unknown);
    assert_eq!(unknown.status.observed_at, None);
    assert_eq!(unknown.status.exit_code, None);
    let fallback = observe(
        None,
        Query::Success(""),
        Query::Success(&owned_queued("PENDING", "Resources")),
        None,
    )
    .unwrap();
    assert_eq!(fallback.status.state, RunState::Queued);
    assert_eq!(fallback.status.exit_code, None);
    let mut foreign = previous.clone();
    foreign.ssh_alias = "other".into();
    assert!(observe(Some(&foreign), Query::Failed, Query::Failed, None).is_err());
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
    let plan = allocation_plan(&s, "accelerator").unwrap();
    assert!(!plan.argv.last().unwrap().contains("sbatch "));
    assert!(!plan.argv.last().unwrap().contains("scancel "));
    assert!(plan.max_stdout_bytes <= 1024 * 1024);
    assert_eq!(plan.timeout_seconds, 12);
    assert!(allocation_plan(&s, "gpu;touch").is_err());
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
    let overflow = valid.replace("RealMemory=1024", "RealMemory=18446744073709551615");
    assert!(parse_allocation(&overflow, "gpu").is_err());
    assert!(parse_allocation(&valid.replace("CPUAlloc=4", "CPUAlloc=17"), "gpu").is_err());
    assert!(parse_allocation(&format!("{valid}private username\n"), "gpu").is_err());
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
    let queued_text = owned_acct("PENDING", "0:0");
    for accounting in [Query::Success(queued_text.as_str()), Query::Success("")] {
        let observed = observe(
            Some(&completed),
            accounting,
            Query::Success(&owned_queued("PENDING", "Priority")),
            Some("checked-time"),
        )
        .unwrap();
        assert_eq!(observed.status, completed);
        assert_eq!(observed.diagnostic, Some(Diagnostic::RegressiveState));
        assert!(observed.retained_previous);
    }
}
