use ml_exp_core::managed::remote::{Problem, WINDOW_BYTES, Window};

#[test]
fn metrics_preserve_arbitrary_context_and_expose_partial_or_corrupt_history() {
    let line = "{\"step\":7,\"name\":\"loss\",\"unit\":\"nats/token\",\"value\":1.5,\"variant_id\":\"a\"}\n";
    let window = Window {
        text: line.into(),
        offset_bytes: 0,
        total_bytes: line.len() as u64,
        problem: None,
    };
    window.validate().unwrap();
    let (records, partial, invalid) = window.metrics();
    assert_eq!(records[0]["variant_id"], "a");
    assert!(!partial);
    assert_eq!(invalid, 0);
    for text in [
        format!("{line}{{\"step\":8"),
        format!("{line}bad json\n"),
        format!("{line}{{\"step\":1,\"step\":2}}\n"),
    ] {
        let w = Window {
            total_bytes: text.len() as u64,
            text,
            ..window.clone()
        };
        let (records, partial, _) = w.metrics();
        assert_eq!(records.len(), 1);
        assert!(partial);
    }
    let text = format!("cut prefix\n{line}");
    let tail = Window {
        total_bytes: text.len() as u64 + 100,
        offset_bytes: 100,
        text,
        problem: Some(Problem::Truncated),
    };
    assert_eq!(tail.metrics().0.len(), 1);
    assert!(tail.metrics().1);
    let missing = Window::unavailable(Problem::Missing);
    assert!(missing.metrics().1);
    assert!(missing.metrics().0.is_empty());
}
#[test]
fn bounded_windows_refuse_inconsistent_offsets_and_oversized_payloads() {
    for window in [
        Window {
            text: "abc".into(),
            offset_bytes: 0,
            total_bytes: 2,
            problem: None,
        },
        Window {
            text: "".into(),
            offset_bytes: 4,
            total_bytes: 2,
            problem: Some(Problem::Truncated),
        },
        Window {
            text: "abc".into(),
            offset_bytes: 1,
            total_bytes: 4,
            problem: None,
        },
        Window {
            text: "x".repeat(WINDOW_BYTES + 1),
            offset_bytes: 0,
            total_bytes: WINDOW_BYTES as u64 + 1,
            problem: None,
        },
    ] {
        assert!(window.validate().is_err());
    }
}

#[test]
fn evidence_requires_exact_sealed_identity_and_immutable_terminal_receipt() {
    use ml_exp_core::{
        launch,
        managed::{Config, evidence::*, remote::*},
        slurm::helper::{self, Scope},
    };
    use std::collections::BTreeMap;
    let identity = launch::Identity {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
    };
    let config = Config {
        manifest_path: "/shared/manifest.json".into(),
        endpoint: String::new(),
        attempt_root: "/shared/worker".into(),
        workdir: "/workspace".into(),
        maximum_timeout_seconds: 60,
        base_environment: BTreeMap::new(),
        local_binding: Some(launch::Binding {
            identity: identity.clone(),
            sha256: "1".repeat(64),
        }),
        output_directory: None,
        runtime_environment: Vec::new(),
    };
    let scope = Scope {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
    };
    let empty = Window {
        text: String::new(),
        offset_bytes: 0,
        total_bytes: 0,
        problem: None,
    };
    let prepared = Evidence {
        contract: OBSERVATION_CONTRACT.into(),
        config_sha256: helper::sha256_bytes(&serde_json::to_vec(&config).unwrap()),
        job_id: "12345".into(),
        worker: Observation {
            contract: CONTRACT.into(),
            identity: identity.clone(),
            manifest_sha256: "1".repeat(64),
            phase: Phase::Prepared,
            safe_to_retry: false,
            receipt: None,
            runtime_environment_sha256: None,
        },
        stdout: empty.clone(),
        stderr: empty.clone(),
        metrics: empty,
    };
    prepared.validate_scope(&scope, "12345").unwrap();
    prepared.validate_config(&config).unwrap();
    let mut completed = prepared.clone();
    completed.worker.phase = Phase::Finished;
    completed.worker.receipt = Some(Receipt {
        contract: CONTRACT.into(),
        identity,
        manifest_sha256: "1".repeat(64),
        process: ProcessEvidence {
            exit_code: Some(0),
            termination_signal: None,
            timed_out: false,
            cancelled: false,
            error: None,
            stdout_sha256: Some("2".repeat(64)),
            stderr_sha256: Some("2".repeat(64)),
        },
        training: Training::Completed,
        publication: Publication::NotImplemented,
        download_available: false,
        runtime_environment_sha256: None,
    });
    completed.validate_scope(&scope, "12345").unwrap();
    completed.validate_config(&config).unwrap();
    completed.follows(&prepared).unwrap();
    let mut with_logs = completed.clone();
    with_logs.stdout = Window {
        text: "original\n".into(),
        offset_bytes: 0,
        total_bytes: 9,
        problem: None,
    };
    let mut unavailable = completed.clone();
    unavailable.stdout = Window::unavailable(Problem::Changed);
    unavailable.retain_windows(&with_logs);
    assert_eq!(unavailable.stdout.text, "original\n");
    assert_eq!(unavailable.stdout.problem, Some(Problem::Changed));
    unavailable.stdout.validate().unwrap();
    let line = "{\"step\":1,\"value\":2}\n";
    let mut first = prepared.clone();
    first.metrics = Window {
        text: line.into(),
        offset_bytes: 0,
        total_bytes: line.len() as u64,
        problem: None,
    };
    let mut appended = first.clone();
    appended.metrics.text.push_str("{\"step\":2");
    appended.metrics.total_bytes = appended.metrics.text.len() as u64;
    appended.follows(&first).unwrap();
    let mut rewritten = first.clone();
    rewritten.metrics.text = rewritten.metrics.text.replace("2", "3");
    assert!(rewritten.follows(&first).is_err());
    let mut retained = first.clone();
    retained.metrics.problem = Some(Problem::Changed);
    assert!(rewritten.follows(&retained).is_err());

    let mut erased = first.clone();
    erased.metrics = Window {
        text: String::new(),
        offset_bytes: 0,
        total_bytes: 0,
        problem: None,
    };
    assert!(erased.follows(&first).is_err());
    let mut invalid_offset = first.clone();
    invalid_offset.metrics.offset_bytes = u64::MAX;
    assert!(invalid_offset.follows(&first).is_err());

    let mut mixed = prepared.clone();
    mixed.metrics.text = "{\"step\":1,\"attempt_id\":\"attempt-002\"}\n{\"step\":2}\n".into();
    mixed.metrics.total_bytes = mixed.metrics.text.len() as u64;
    let (records, partial, invalid, foreign) = mixed.metric_records();
    assert_eq!(records.len(), 1);
    assert!(partial);
    assert_eq!(invalid, 0);
    assert_eq!(foreign, 1);
    assert!(prepared.follows(&completed).is_err());
    assert!(prepared.validate_scope(&scope, "12346").is_err());
    for mutation in [
        "identity",
        "config",
        "training",
        "download",
        "missing-receipt",
        "runtime",
    ] {
        let mut changed = completed.clone();
        match mutation {
            "identity" => changed.worker.identity.attempt_id = "attempt-002".into(),
            "config" => changed.config_sha256 = format!("sha256:{}", "3".repeat(64)),
            "training" => changed.worker.receipt.as_mut().unwrap().training = Training::Failed,
            "download" => changed.worker.receipt.as_mut().unwrap().download_available = true,
            "missing-receipt" => changed.worker.receipt = None,
            "runtime" => changed.worker.runtime_environment_sha256 = Some("4".repeat(64)),
            _ => unreachable!(),
        }
        assert!(
            changed
                .validate_scope(&scope, "12345")
                .and_then(|_| changed.validate_config(&config))
                .is_err(),
            "{mutation}"
        );
        assert!(changed.follows(&completed).is_err(), "{mutation}");
    }
}
