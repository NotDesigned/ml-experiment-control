//! Human status summaries over synthetic evidence; no clock, credentials or network.
use ml_exp_client::presentation;
use serde_json::{Value, json};

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

#[test]
fn status_summary_lines_describe_each_evidence_layer_without_changing_input() {
    for (input, expected) in [
        (
            literal(r#"{}"#),
            "Run: UNKNOWN/UNKNOWN\nStage: UNKNOWN\nScheduler: UNKNOWN\nBackend: not observed; job not observed; native not observed\nTraining: UNKNOWN; process not observed\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: NOT_OBSERVED; download unavailable\nVerification: NOT_OBSERVED\nNext: inspect existing evidence; no action specified",
        ),
        (
            literal(
                r#"{"project":"demo","run_id":"trial","attempt_id":"attempt-001","scheduler":{"state":"FAILED"},"training":{"status":"UNKNOWN"},"publication":{"status":"RELEASED","download_available":true},"verification":{"status":"PENDING"}}"#,
            ),
            "Run: demo/trial\nStage: FAILED\nScheduler: FAILED\nBackend: not observed; job not observed; native not observed\nTraining: UNKNOWN; process not observed\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: RELEASED; download unavailable\nVerification: PENDING\nNext: inspect existing evidence; no action specified",
        ),
        (
            literal(
                r#"{"project":"demo","run_id":"trial","attempt_id":"attempt-001","scheduler":{"state":"QUEUED"},"training":{"status":"COMPLETED","exit_code":0},"publication":{"status":"RELEASED","download_available":false},"verification":{"status":"SHA256_VERIFIED"},"queue":{"reason":"Current Limit is CPU: 200 limited by Role PAI.AlgoDeveloper.GPU: 8 limited by Role PAI.AlgoDeveloper.","provider_code":"CheckSelfQuotaFailed","position":1,"blocker":{"CodeType":"ConfigRule","Code":"roleMaximumResource","Reason":"Current Limit is CPU: 200 limited by Role PAI.AlgoDeveloper.GPU: 8 limited by Role PAI.AlgoDeveloper."},"as_of":"2026-10-24T00:00:00Z","stale":true,"estimated_start_at":null},"progress":{"phase":"QUEUED"}}"#,
            ),
            "Run: demo/trial\nStage: QUEUED\nScheduler: QUEUED\nBackend: not observed; job not observed; native not observed\nWaiting: Current Limit is CPU: 200 limited by Role PAI.AlgoDeveloper.GPU: 8 limited by Role PAI.AlgoDeveloper. (roleMaximumResource; CheckSelfQuotaFailed; position 1; observed 2026-10-24T00:00:00Z; stale evidence)\nTraining: COMPLETED; process not observed; exit 0\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: RELEASED; download unavailable\nVerification: SHA256_VERIFIED\nNext: inspect existing evidence; no action specified",
        ),
        (
            literal(
                r#"{"project":"demo","run_id":"trial","attempt_id":"attempt-001","scheduler":{"state":"SUCCEEDED","observed_at":1792051748.25,"stale":false},"training":{"status":"UNKNOWN"},"publication":{"status":"RELEASED","download_available":false},"verification":{"status":"SHA256_VERIFIED"},"worker_phase":{"phase":"TRAINING"},"progress":{"phase":"TRAINING"}}"#,
            ),
            "Run: demo/trial\nStage: TRAINING (last reported worker phase; not a heartbeat)\nScheduler: SUCCEEDED; observed 2026-10-15T08:09:08+00:00\nBackend: not observed; job not observed; native not observed\nTraining: UNKNOWN; process not observed\nWorker phase: TRAINING; last reported; not a heartbeat\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: RELEASED; download unavailable\nVerification: SHA256_VERIFIED\nNext: inspect existing evidence; no action specified",
        ),
        (
            literal(
                r#"{"scheduler":{},"model":{},"training":{"diagnostic":false},"progress":{},"publication":{},"verification":{},"queue":{"position":"中文😀"},"input_delivery":{},"worker_phase":{"diagnostic":[],"step":1,"position":0,"message":"中文😀"},"project":"p","run_id":"r","attempt_id":"attempt-001"}"#,
            ),
            "Run: p/r\nStage: UNKNOWN\nScheduler: UNKNOWN\nBackend: not observed; job not observed; native not observed\nTraining: UNKNOWN; process not observed\nWorker phase: UNKNOWN; last reported; not a heartbeat\nModel: NOT_OBSERVED\nProgress: recent progress not observed\nResults: NOT_OBSERVED; download unavailable\nVerification: NOT_OBSERVED\nNext: inspect existing evidence; no action specified",
        ),
    ] {
        let before = input.clone();
        assert_eq!(presentation::status_summary(&input).unwrap(), expected);
        assert_eq!(input, before);
    }
}

#[test]
fn the_attempt_is_shown_only_once_a_run_has_more_than_one_or_was_chosen() {
    let first = json!({"project":"demo","run_id":"trial","attempt_id":"attempt-001"});
    let retried = json!({"project":"demo","run_id":"trial","attempt_id":"attempt-002"});
    let heading = |value: &Value, named: bool| {
        presentation::status_summary_for(value, named)
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .to_owned()
    };
    assert_eq!(heading(&first, false), "Run: demo/trial");
    assert_eq!(heading(&first, true), "Run: demo/trial · attempt-001");
    assert_eq!(heading(&retried, false), "Run: demo/trial · attempt-002");
    assert_eq!(
        presentation::status_summary(&first).unwrap(),
        presentation::status_summary_for(&first, false).unwrap()
    );
    for (attempt, shown) in [
        (json!("attempt-001"), None),
        (json!("attempt-002"), Some("attempt-002")),
        (json!("attempt-1"), None),
        (json!(null), None),
    ] {
        assert_eq!(presentation::retried_attempt(&attempt), shown);
    }
}

#[test]
fn checkpoints_tracking_and_results_lead_with_the_run() {
    let id = format!("checkpoint.{}", "a".repeat(64));
    let document = json!({"project":"demo","run_id":"trial","attempt_id":"attempt-001","checkpoint_id":id,
        "step":4,"files":[{"path":"a"}],"registered_at":"2026-11-01T00:00:00Z"});
    let list = json!({"project":"demo","run_id":"trial","attempt_id":"attempt-001",
        "checkpoints":[document],"archives":{id.clone():{"verified":true}}});
    assert_eq!(
        presentation::checkpoints(&list, false),
        format!(
            "Run: demo/trial\nCheckpoints: 1\n  {id} · step 4 · 1 file · registered 2026-11-01T00:00:00Z · object-store copy verified\nResume from one: ml-exp retry --checkpoint ID (the Run continues as a new Attempt)"
        )
    );
    assert_eq!(
        presentation::checkpoints(&document, false),
        format!(
            "Run: demo/trial\nCheckpoint: {id} · step 4 · 1 file · registered 2026-11-01T00:00:00Z"
        )
    );
    let mut later = list.clone();
    later["attempt_id"] = json!("attempt-002");
    later["checkpoints"] = json!([]);
    assert_eq!(
        presentation::checkpoints(&later, false),
        "Run: demo/trial · attempt-002\nCheckpoints: none registered"
    );
    let scope = |attempt: &str| {
        json!({"attempt":attempt,"wandb_id":"w1","url":"https://wandb.ai/e/p/runs/w1",
            "confirmed":10,"total":12,"status":"PENDING","error":null})
    };
    let tracking = json!({"project":"demo","run_id":"trial",
        "wandb":{"enabled":true,"entity":"e","project":"p"},"scopes":[scope("attempt-001")]});
    assert_eq!(
        presentation::tracking(&tracking),
        "Run: demo/trial\nW&B: e/p\nhttps://wandb.ai/e/p/runs/w1 · 10 of 12 records confirmed · PENDING"
    );
    let mut two = tracking.clone();
    two["scopes"] = json!([scope("attempt-001"), scope("attempt-002")]);
    let text = presentation::tracking(&two);
    assert!(text.contains("\nattempt-001: https://"), "{text}");
    assert!(text.contains("\nattempt-002: https://"), "{text}");
    let downloaded = json!({"submission_id":"action-0123456789abcdef","local_archive":"/tmp/trial-results.tar",
        "local_archive_verified":true,"reused_existing_archive":false,"skipped":[],"release_requested":true});
    assert_eq!(
        presentation::results("demo", "trial", &downloaded),
        "Run: demo/trial\nResults: /tmp/trial-results.tar; every file verified\nServer copy: released"
    );
    assert!(!presentation::results("demo", "trial", &downloaded).contains("attempt"));
}

#[test]
fn status_summary_refuses_a_document_that_is_not_an_object() {
    for input in [
        json!(null),
        json!([]),
        json!(0),
        json!(false),
        json!("text"),
    ] {
        assert!(presentation::status_summary(&input).is_err(), "{input}");
    }
}

#[test]
fn unavailable_publication_does_not_change_completed_training_or_invent_metric_time() {
    let value = json!({"project":"p","run_id":"r","scheduler":{"state":"SUCCEEDED"},"training":{"status":"COMPLETED","exit_code":0},"publication":{"status":"FAILED","download_available":true},"verification":{"status":"PENDING"},"model":{"last_metric":{"name":"validation_loss","value":0,"unit":"nats/token","status":"VALID"}}});
    let text = presentation::status_summary(&value).unwrap();
    assert!(text.contains("Training: COMPLETED"));
    assert!(text.contains("Results: FAILED; download unavailable"));
    assert!(text.contains("Last metric: validation_loss=0 nats/token; VALID"));
    assert!(text.contains("freshness is unknown"));
    assert!(!text.contains("Training: FAILED"));
}
