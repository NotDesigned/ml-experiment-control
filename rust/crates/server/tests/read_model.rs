use ml_exp_server::{
    experiments::{Diagnostic, EvidenceDocument, EvidenceProblem, LogTail, ManifestSource},
    read_model::{self, LiveMetrics, LogQuery, LogStream, MetricFormat, MetricQuery},
};
use serde_json::{Value, json};
#[path = "common/read_model.rs"]
mod support;
use support::*;
fn metric(name: &str, value: Value, step: u64) -> Value {
    json!({"name":name,"unit":"nats/token","value":value,"step":step})
}
fn live(records: &[Value]) -> Vec<Value> {
    records
        .iter()
        .flat_map(|record| {
            ml_exp_core::metrics::observations(record, &json!({}), None, Some("journal")).unwrap()
        })
        .collect()
}

#[test]
fn exact_current_and_live_receive_order_never_choose_the_largest_scientific_step() {
    let mut snapshot = snapshot();
    let mut other = attempt("attempt-002");
    other.status = evidence("attempt-002", "status", json!({"state":"SUCCEEDED"}));
    other.metrics.records = vec![metric("loss", json!(999), 99999)];
    snapshot.attempts.push(other);
    snapshot.attempts[0].submission = evidence(
        "attempt-001",
        "submission",
        json!({"backend":"slurm","backend_job_id":"123"}),
    );
    let records = live(&[metric("loss", json!(4), 1000), metric("loss", json!(0), 0)]);
    let result = read_model::run_detail(
        &snapshot,
        2_000_000.0,
        Some(LiveMetrics {
            attempt_id: "attempt-001",
            records: &records,
        }),
    )
    .unwrap();
    assert_eq!(result["evidence"]["model"]["detail"]["value"], 0);
    assert_eq!(result["evidence"]["model"]["detail"]["step"], 0);
    assert_eq!(result["evidence"]["scheduler"]["state"], "QUEUED");
    assert_eq!(
        result["evidence"]["scheduler"]["detail"]["queue"]["reason"],
        "Resources"
    );
    assert_eq!(
        result["evidence"]["model"]["source"],
        "canonical_record_journal"
    );
    assert_eq!(result["evidence"]["model"]["not_a_heartbeat"], true);
    let listing = read_model::run_attempts(&snapshot).unwrap();
    assert_eq!(listing["attempts"][0]["current"], true);
    assert_eq!(listing["attempts"][0]["has_submission"], true);
    assert_eq!(listing["attempts"][1]["has_submission"], false);
    assert!(matches!(
        read_model::run_detail(
            &snapshot,
            0.0,
            Some(LiveMetrics {
                attempt_id: "attempt-002",
                records: &records
            })
        ),
        Err(read_model::Error::Identity)
    ));
    assert!(matches!(
        read_model::attempt_metrics(
            &snapshot,
            "attempt-001",
            &MetricQuery::default(),
            Some(LiveMetrics {
                attempt_id: "attempt-002",
                records: &records
            })
        ),
        Err(read_model::Error::Identity)
    ));
    snapshot.current_attempt_id = None;
    let no_current = read_model::run_detail(&snapshot, 0.0, None).unwrap();
    assert!(no_current["evidence"]["scheduler"]["state"].is_null());
    assert!(
        no_current["evidence"]["model"]["detail"]
            .as_object()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn unknown_producer_timestamps_never_borrow_snapshot_time_or_file_mtime() {
    let mut snapshot = snapshot();
    snapshot.current_status = evidence("attempt-001", "status", json!({"state":"RUNNING"}));
    snapshot.attempts[0].collection = evidence(
        "attempt-001",
        "collection",
        json!({"process_state":"RUNNING","worker_state":"RUNNING"}),
    );
    snapshot.attempts[0].metrics.records = vec![metric("loss", json!(3), 0)];
    let result = read_model::run_detail(&snapshot, 1_000_000.0, None).unwrap();
    for name in ["scheduler", "worker", "process", "model"] {
        assert!(result["evidence"][name]["as_of"].is_null(), "{name}");
        assert!(result["evidence"][name]["stale"].is_null(), "{name}");
        assert_eq!(result["evidence"][name]["freshness"], "UNKNOWN");
    }
    snapshot.attempts[0].collection = evidence(
        "attempt-001",
        "collection",
        json!({"process_state":"RUNNING","process_observed_at":0,"observed_at":99999}),
    );
    let result = read_model::run_detail(&snapshot, 100_000.0, None).unwrap();
    assert_eq!(result["evidence"]["process"]["as_of"], 0.0);
    assert_eq!(result["evidence"]["process"]["stale"], true);
}

#[test]
fn training_process_scheduler_and_unmigrated_publication_remain_independent() {
    let mut snapshot = snapshot();
    snapshot.current_status = evidence("attempt-001", "status", json!({"state":"FAILED"}));
    snapshot.attempts[0].collection = evidence(
        "attempt-001",
        "collection",
        json!({"process_state":"SUCCEEDED","worker_state":"LOST","process_evidence":{"exit_code":0},
        "status":"AVAILABLE","verification":"SHA256_VERIFIED","download_available":true}),
    );
    let result = read_model::attempt_collection(&snapshot, "attempt-001", 0.0).unwrap();
    assert_eq!(result["training"]["status"], "COMPLETED");
    assert_eq!(result["training"]["exit_code"], 0);
    assert_eq!(result["evidence"]["scheduler"]["state"], "FAILED");
    assert_eq!(result["evidence"]["worker"]["state"], "LOST");
    assert_eq!(result["status"], "NOT_OBSERVED");
    assert_eq!(result["verification"], "PENDING");
    assert_eq!(result["download_available"], false);
    for (state, code) in [("FAILED", 0), ("SUCCEEDED", 1), ("RUNNING", 0)] {
        snapshot.attempts[0].collection = evidence(
            "attempt-001",
            "collection",
            json!({"process_state":state,"process_evidence":{"exit_code":code}}),
        );
        let conflicting = read_model::attempt_collection(&snapshot, "attempt-001", 0.0).unwrap();
        assert_eq!(conflicting["training"]["status"], "UNKNOWN");
        assert_eq!(
            conflicting["training"]["diagnostic"],
            "PROCESS_STATE_EXIT_CODE_CONFLICT"
        );
        assert_eq!(conflicting["training"]["exit_code"], code);
        assert_eq!(conflicting["training"]["process_state"], state);
    }
    snapshot.current_status = evidence("attempt-001", "status", json!({"state":"SUCCEEDED"}));
    snapshot.attempts[0].collection = EvidenceDocument::Missing;
    assert_eq!(
        read_model::attempt_collection(&snapshot, "attempt-001", 0.0).unwrap()["training"]["status"],
        "UNKNOWN"
    );
    snapshot.attempts[0].manifest.source = ManifestSource::AttemptJson;
    snapshot.attempts[0]
        .manifest
        .metadata
        .insert("state".into(), json!("SUCCEEDED"));
    snapshot.attempts[0]
        .manifest
        .metadata
        .insert("return_code".into(), json!(0));
    assert_eq!(
        read_model::attempt_collection(&snapshot, "attempt-001", 0.0).unwrap()["training"]["status"],
        "COMPLETED"
    );
}

#[test]
fn raw_metadata_uses_shared_redaction_and_only_daemon_assessments_may_publish_classes() {
    let mut snapshot = snapshot();
    snapshot.run.metadata.insert("environment".into(),json!({"API_KEY":"private-value","proxy":"https://user:pass@host","normal":"Bearer abcdef"}));
    snapshot
        .run
        .metadata
        .insert("failure_class".into(), json!("producer-claim"));
    snapshot.attempts[0].decision = evidence(
        "attempt-001",
        "decision",
        json!({"action":"REVIEW_RETRY","failure_class":"producer-only",
        "resume_checkpoint":{"failure_class":"nested-producer-claim","password":"hidden"}}),
    );
    let result = read_model::attempt_show(&snapshot, "attempt-001", 0.0).unwrap();
    assert!(result["attempt"]["decision"]["failure_class"].is_null());
    assert!(result["attempt"]["decision"]["resume_checkpoint"]["failure_class"].is_null());
    assert_eq!(
        result["attempt"]["decision"]["resume_checkpoint"]["password"],
        "<redacted>"
    );
    let mut detail = read_model::run_detail(&snapshot, 0.0, None).unwrap();
    assert!(detail["failure_assessment"]["failure_summary"].is_null());
    detail
        .as_object_mut()
        .unwrap()
        .shift_remove("failure_assessment");
    let text = serde_json::to_string(&detail).unwrap();
    for secret in [
        "private-value",
        "user:pass",
        "abcdef",
        "producer-claim",
        "producer-only",
        "nested-producer-claim",
    ] {
        assert!(!text.contains(secret), "{secret}");
    }
    snapshot.attempts[0].status = evidence(
        "attempt-001",
        "status",
        json!({"state":"FAILED","started_at":100,"updated_at":200}),
    );
    snapshot.attempts[0].collection = evidence(
        "attempt-001",
        "collection",
        json!({"process_state":"FAILED","observed_at":200,"failure_class":"model"}),
    );
    let show = read_model::attempt_show(&snapshot, "attempt-001", 300.0).unwrap();
    assert!(show["collection"]["failure_class"].is_null());
    assert_eq!(
        show["failure_assessment"]["failure_summary"]["failure_class"],
        "model"
    );
}

#[test]
fn unavailable_and_foreign_evidence_cannot_turn_into_a_successful_snapshot() {
    let mut snapshot = snapshot();
    snapshot.current_status = evidence("attempt-001", "status", json!({"state":"SUCCEEDED"}));
    snapshot.attempts[0].status = EvidenceDocument::Unavailable {
        source: "attempts/attempt-001/status.json".into(),
        code: EvidenceProblem::InvalidJson,
    };
    let result = read_model::run_detail(&snapshot, 0.0, None).unwrap();
    assert!(result["scheduler_state"].is_null());
    assert_eq!(
        result["evidence"]["scheduler"]["evidence_availability"]["availability"],
        "UNAVAILABLE"
    );
    snapshot.attempts[0].submission = EvidenceDocument::Unavailable {
        source: "attempts/attempt-001/submission.json".into(),
        code: EvidenceProblem::Io,
    };
    assert!(
        read_model::run_attempts(&snapshot).unwrap()["attempts"][0]["has_submission"].is_null()
    );
    snapshot.attempts[0].collection = evidence(
        "attempt-002",
        "collection",
        json!({"process_state":"SUCCEEDED"}),
    );
    assert!(matches!(
        read_model::run_detail(&snapshot, 0.0, None),
        Err(read_model::Error::Identity)
    ));
    assert!(matches!(
        read_model::attempt_show(&snapshot, "attempt-001", 0.0),
        Err(read_model::Error::Identity)
    ));
}

#[test]
fn scientific_metric_names_units_and_contexts_survive_metadata_redaction() {
    let mut snapshot = snapshot();
    let schema = json!({"definitions":{"token":{"unit":"nats/token"},"password":{"unit":"bits/raw byte"},"api_key":{"unit":"fraction"}}});
    let contract = json!({"metrics_schema":schema,"protocol_id":"sha256:protocol","context":{"token":"literal scientific protocol parameter"}});
    snapshot
        .run
        .metadata
        .insert("evaluation".into(), contract.clone());
    snapshot
        .run
        .metadata
        .insert("environment".into(), json!({"TOKEN":"secret-env"}));
    snapshot.attempts[0].metrics.records = vec![
        json!({"step":0,"token":0,"password":1,"api_key":2}),
        json!({"name":"password","unit":"bits/raw byte","value":3,"variant_id":"α","dataset_id":"d","checkpoint_id":"c","protocol_id":"p","summary":true}),
    ];
    let query = MetricQuery {
        format: MetricFormat::Records,
        ..MetricQuery::default()
    };
    let result = read_model::attempt_metrics(&snapshot, "attempt-001", &query, None).unwrap();
    assert_eq!(result["records"][0]["name"], "token");
    assert_eq!(result["records"][0]["value"], 0);
    assert_eq!(result["records"][1]["name"], "password");
    assert_eq!(result["records"][1]["value"], 1);
    assert_eq!(result["records"][3]["variant_id"], "α");
    assert_eq!(result["records"][3]["summary"], true);
    let legacy =
        read_model::attempt_metrics(&snapshot, "attempt-001", &MetricQuery::default(), None)
            .unwrap();
    assert_eq!(legacy["points"][0]["token"], 0);
    assert_eq!(legacy["points"][0]["password"], 1);
    assert_eq!(legacy["evaluation"], contract);
    assert!(
        !serde_json::to_string(&legacy)
            .unwrap()
            .contains("secret-env")
    );
    let detail = read_model::run_detail(&snapshot, 0.0, None).unwrap();
    assert!(
        !serde_json::to_string(&detail)
            .unwrap()
            .contains("secret-env")
    );
}

#[test]
fn scientific_assignment_spellings_are_preserved_in_all_four_read_views() {
    let mut snapshot = snapshot();
    snapshot.attempts[0].metrics.records = vec![
        json!({"name":"token=entropy","unit":"secret=joule","value":2,
        "step":7,"variant_id":"token=variant","protocol_id":"secret=protocol","checkpoint_id":"token=checkpoint","summary":false}),
    ];
    for output in [
        read_model::run_detail(&snapshot, 0.0, None).unwrap(),
        read_model::attempt_show(&snapshot, "attempt-001", 0.0).unwrap(),
        read_model::attempt_collection(&snapshot, "attempt-001", 0.0).unwrap(),
    ] {
        let detail = &output["evidence"]["model"]["detail"];
        assert_eq!(detail["name"], "token=entropy");
        assert_eq!(detail["unit"], "secret=joule");
        assert_eq!(detail["variant_id"], "token=variant");
        assert_eq!(detail["protocol_id"], "secret=protocol");
        assert_eq!(detail["checkpoint_id"], "token=checkpoint");
        assert_eq!(detail["summary"], false);
    }
    let metrics = read_model::attempt_metrics(
        &snapshot,
        "attempt-001",
        &MetricQuery {
            format: MetricFormat::Records,
            ..MetricQuery::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(metrics["records"][0]["name"], "token=entropy");
    assert_eq!(metrics["records"][0]["unit"], "secret=joule");
    snapshot.attempts[0].metrics.records =
        vec![json!({"step":7,"token":0,"password":1,"api_key":2})];
    for output in [
        read_model::run_detail(&snapshot, 0.0, None).unwrap(),
        read_model::attempt_show(&snapshot, "attempt-001", 0.0).unwrap(),
        read_model::attempt_collection(&snapshot, "attempt-001", 0.0).unwrap(),
    ] {
        let detail = &output["evidence"]["model"]["detail"];
        assert_eq!(detail["token"], 0);
        assert_eq!(detail["password"], 1);
        assert_eq!(detail["api_key"], 2);
    }
    snapshot.attempts[0].metrics.records = vec![
        json!({"name":"loss","unit":{"API_KEY":"hidden-unit"},"value":{"password":"hidden-value"}}),
    ];
    let output = read_model::run_detail(&snapshot, 0.0, None).unwrap();
    let encoded = serde_json::to_string(&output).unwrap();
    assert!(!encoded.contains("hidden-unit") && !encoded.contains("hidden-value"));
    assert_eq!(output["evidence"]["model"]["detail"]["value"], Value::Null);
    assert_eq!(output["evidence"]["model"]["detail"]["status"], "FAILED");
}

#[test]
fn journal_claimed_valid_observations_are_checked_before_all_read_projections() {
    let base = live(&[metric("loss", json!(0), 0)])[0].clone();
    let bad_fields = [
        ("name", json!("")),
        ("unit", json!("")),
        ("value", Value::Null),
        ("value", json!(true)),
        ("protocol_id", json!(17)),
        ("variant_id", json!({"foreign":"identity"})),
        ("step", json!("1")),
        ("epoch", json!(false)),
        ("dataset_id", json!([])),
        ("checkpoint_id", json!({})),
        ("summary", json!(1)),
        ("numerator", json!("5")),
        ("denominator", json!("0")),
        ("status", json!("unknown")),
        ("environment", json!({"API_KEY":"hidden"})),
    ];
    for (key, value) in bad_fields {
        let mut malformed = base.clone();
        malformed[key] = value;
        let rows = [malformed];
        assert!(
            read_model::validate_journal_observations(&rows, &json!({})).is_err(),
            "{key}"
        );
        assert!(
            read_model::merge_metrics(&[], &rows, &json!({}), None).is_err(),
            "{key}"
        );
        assert!(
            read_model::metric_payload(
                &rows,
                &MetricQuery::default(),
                &json!({}),
                None,
                None,
                true
            )
            .is_err(),
            "{key}"
        );
        let snapshot = snapshot();
        assert!(
            read_model::run_detail(
                &snapshot,
                0.0,
                Some(LiveMetrics {
                    attempt_id: "attempt-001",
                    records: &rows
                })
            )
            .is_err(),
            "{key}"
        );
    }
    let mut zero = base;
    zero["source"] = Value::Null;
    read_model::validate_journal_observations(&[zero], &json!({})).unwrap();
    let ratio = json!({"metrics_schema":{"definitions":{"loss":{"unit":"nats/token","aggregation":"ratio_of_sums"}}}});
    let mut missing_counts = live(&[metric("loss", json!(2), 1)])[0].clone();
    assert!(read_model::validate_journal_observations(&[missing_counts.clone()], &ratio).is_err());
    missing_counts["numerator"] = json!(2);
    missing_counts["denominator"] = json!(0);
    assert!(read_model::validate_journal_observations(&[missing_counts.clone()], &ratio).is_err());
    missing_counts["denominator"] = json!(1);
    read_model::validate_journal_observations(&[missing_counts], &ratio).unwrap();
}

#[test]
fn canonical_failure_undeclared_and_conflict_records_remain_diagnostics_without_becoming_valid() {
    let raw = [
        json!({"name":"loss","unit":"nats/token","value":"nan"}),
        json!({"step":1,"loss":2}),
        json!({"name":null,"unit":"u","value":3}),
        json!({"name":"loss","unit":"nats/token","value":null,"status":"MISSING"}),
        json!({"name":"loss","unit":"nats/token","value":2,"status":"PARTIAL"}),
    ];
    for record in raw {
        let canonical =
            ml_exp_core::metrics::observations(&record, &json!({}), None, Some("journal")).unwrap();
        let before = canonical.clone();
        read_model::validate_journal_observations(&canonical, &json!({})).unwrap();
        assert_eq!(canonical, before);
        let projected = read_model::metric_payload(
            &canonical,
            &MetricQuery {
                format: MetricFormat::Records,
                ..MetricQuery::default()
            },
            &json!({}),
            None,
            None,
            true,
        )
        .unwrap();
        assert_ne!(projected["validation"]["state"], "COMPLETE");
        assert_eq!(projected["records"][0]["status"], canonical[0]["status"]);
    }
    let conflict = ml_exp_core::metrics::metric_result(
        &[metric("loss", json!(2), 1), metric("loss", json!(3), 1)],
        None,
        None,
        None,
        false,
    )
    .unwrap();
    let records = conflict["records"].as_array().unwrap();
    read_model::validate_journal_observations(records, &json!({})).unwrap();
    let projected = read_model::metric_payload(
        records,
        &MetricQuery {
            format: MetricFormat::Records,
            ..MetricQuery::default()
        },
        &json!({}),
        None,
        None,
        true,
    )
    .unwrap();
    assert_eq!(projected["records"][0]["status"], "CONFLICTING");
    assert_eq!(projected["validation"]["state"], "FAILED");
    let protocol_rows = live(&[
        json!({"name":"loss","unit":"nats/token","value":2,"protocol_id":"protocol-a"}),
        json!({"name":"loss","unit":"nats/token","value":2,"protocol_id":"protocol-b"}),
    ]);
    read_model::validate_journal_observations(&protocol_rows, &json!({})).unwrap();
    assert!(
        read_model::validate_journal_observations(
            &protocol_rows,
            &json!({"protocol_id":"protocol-a"})
        )
        .is_err()
    );
    let failed_protocol = ml_exp_core::metrics::observations(
        &json!({"name":"loss","unit":"nats/token","value":2,"protocol_id":17}),
        &json!({}),
        Some("protocol-a"),
        Some("journal"),
    )
    .unwrap();
    read_model::validate_journal_observations(
        &failed_protocol,
        &json!({"protocol_id":"protocol-a"}),
    )
    .unwrap();
    assert_eq!(failed_protocol[0]["status"], "FAILED");
}

#[test]
fn bounded_tails_never_claim_complete_history_even_with_a_complete_live_journal() {
    let mut snapshot = snapshot();
    snapshot.attempts[0].metrics.source = Some("attempts/attempt-001/train_metrics.jsonl".into());
    snapshot.attempts[0].metrics.records = vec![metric("loss", json!(3), 1)];
    snapshot.attempts[0].metrics.truncated = true;
    let records = live(&[metric("loss", json!(2), 2)]);
    let query = MetricQuery {
        format: MetricFormat::Records,
        ..MetricQuery::default()
    };
    let result = read_model::attempt_metrics(
        &snapshot,
        "attempt-001",
        &query,
        Some(LiveMetrics {
            attempt_id: "attempt-001",
            records: &records,
        }),
    )
    .unwrap();
    assert_eq!(result["validation"]["coverage"], "bounded_partial_history");
    assert_eq!(result["validation"]["complete_history"], false);
    assert_eq!(result["validation"]["state"], "PARTIAL");
    snapshot.attempts[0].metrics.diagnostics = vec![Diagnostic {
        source: "metrics.jsonl".into(),
        code: EvidenceProblem::NotLoaded,
        count: 1,
    }];
    assert!(matches!(
        read_model::attempt_metrics(&snapshot, "attempt-001", &query, None),
        Err(read_model::Error::Evidence)
    ));
}

#[test]
fn logs_are_exact_bounded_redacted_and_never_contact_a_provider() {
    let mut snapshot = snapshot();
    snapshot.attempts[0].stdout = LogTail {
        source: Some("attempts/attempt-001/stdout.log".into()),
        lines: vec![
            "first".into(),
            "训练 ☃".into(),
            "Authorization: Bearer hidden-log-token".into(),
        ],
        available_lines: 3,
        truncated: false,
        sampled_bytes: 64,
        ..empty_log()
    };
    snapshot.attempts[0].collection = evidence(
        "attempt-001",
        "collection",
        json!({"process_evidence":{"stderr_tail":[0,false,null,{"x":1},"last"],"sources":{"stderr":"remote/stderr.log"}}}),
    );
    let result = read_model::attempt_logs(
        &snapshot,
        "attempt-001",
        &LogQuery {
            stream: LogStream::Both,
            lines: 2,
        },
    )
    .unwrap();
    assert_eq!(
        result["streams"]["stdout"]["lines"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(result["streams"]["stdout"]["lines"][0], "训练 ☃");
    assert!(
        !serde_json::to_string(&result)
            .unwrap()
            .contains("hidden-log-token")
    );
    assert_eq!(
        result["streams"]["stderr"]["lines"],
        json!(["{'x': 1}", "last"])
    );
    assert_eq!(result["streams"]["stderr"]["mode"], "collected_tail");
    assert_eq!(result["follow_supported"], false);
    assert!(matches!(
        read_model::attempt_logs(&snapshot, "attempt-999", &LogQuery::default()),
        Err(read_model::Error::UnknownAttempt)
    ));
    assert!(matches!(
        read_model::attempt_logs(
            &snapshot,
            "attempt-001",
            &LogQuery {
                lines: 0,
                ..LogQuery::default()
            }
        ),
        Err(read_model::Error::Query(_))
    ));
}

#[test]
fn oversize_outward_metadata_fails_explicitly_instead_of_silently_truncating_science() {
    let mut snapshot = snapshot();
    snapshot.run.metadata.insert(
        "notes".into(),
        json!("x".repeat(read_model::MAX_RESPONSE_BYTES)),
    );
    assert!(matches!(
        read_model::run_detail(&snapshot, 0.0, None),
        Err(read_model::Error::OutputLimit)
    ));
    assert!(matches!(
        read_model::run_detail(&snapshot, f64::NAN, None),
        Err(read_model::Error::Query(_))
    ));
    let wide = Value::Object(
        (0..=read_model::MAX_METRIC_OBSERVATIONS)
            .map(|i| (format!("metric-{i}"), json!(1)))
            .collect(),
    );
    assert!(matches!(
        read_model::metric_payload(
            &[wide],
            &MetricQuery::default(),
            &json!({}),
            None,
            None,
            false
        ),
        Err(read_model::Error::OutputLimit)
    ));
}

#[test]
fn preempted_scheduler_evidence_is_terminal_without_claiming_process_success() {
    let mut snapshot = snapshot();
    snapshot.current_status = evidence("attempt-001", "status", json!({"state":"PREEMPTED"}));
    snapshot.attempts[0].collection = EvidenceDocument::Missing;
    let result = read_model::run_detail(&snapshot, 0.0, None).unwrap();
    assert_eq!(result["scheduler_state"], "PREEMPTED");
    assert_eq!(result["is_terminal"], true);
    assert_eq!(
        read_model::attempt_collection(&snapshot, "attempt-001", 0.0).unwrap()["training"]["status"],
        "UNKNOWN"
    );
}
