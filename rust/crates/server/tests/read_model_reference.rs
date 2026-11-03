use ml_exp_server::experiments::{EvidenceDocument, LogTail};
use ml_exp_server::read_model::{self, MetricQuery};
use serde_json::{Value, json};
#[path = "common/read_model.rs"]
mod support;
use support::*;

#[test]
fn metric_read_projection_matches_actual_python_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../compatibility/read-model.json")).unwrap();
    assert_eq!(fixture["baseline_commit"].as_str().unwrap().len(), 40);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let input = &case["input"];
        let result = match case["operation"].as_str().unwrap() {
            "metric_payload" => {
                let query = MetricQuery {
                    format: serde_json::from_value(input["format"].clone()).unwrap(),
                    max_points: input["max_points"].as_u64().unwrap() as usize,
                    keys: input["keys"].as_str().map(str::to_owned),
                    names: input
                        .get("names")
                        .map(|v| serde_json::from_value(v.clone()).unwrap()),
                };
                read_model::metric_payload(
                    input["records"].as_array().unwrap(),
                    &query,
                    &input["contract"],
                    input["source"].as_str(),
                    input["source_attempt_id"].as_str(),
                    input["normalized"].as_bool().unwrap_or(false),
                )
            }
            "merge_metrics" => read_model::merge_metrics(
                input["records"].as_array().unwrap(),
                input["live"].as_array().unwrap(),
                &input["contract"],
                input["source"].as_str(),
            )
            .map(Value::Array),
            "outward" => read_model::outward_metadata(input),
            "run_attempts" => {
                let mut snapshot = snapshot();
                snapshot.attempts.clear();
                snapshot.current_attempt_id =
                    input["current_attempt_id"].as_str().map(str::to_owned);
                for row in input["attempts"].as_array().unwrap() {
                    let id = row["attempt_id"].as_str().unwrap();
                    let mut attempt = attempt(id);
                    attempt.status = evidence(id, "status", json!({"state":row["state"]}));
                    attempt.backend = evidence(
                        id,
                        "backend",
                        json!({"backend":row["backend"],"backend_job_id":row["backend_job_id"]}),
                    );
                    attempt.decision = evidence(id, "decision", row["decision"].clone());
                    if row["has_submission"] == true {
                        attempt.submission = evidence(id, "submission", json!({}));
                    }
                    snapshot.attempts.push(attempt);
                }
                if let Some(id) = snapshot.current_attempt_id.as_deref() {
                    snapshot.current_status = evidence(id, "status", json!({}));
                    if let EvidenceDocument::Present { source, .. } = &mut snapshot.current_status {
                        *source = "status.json".into();
                    }
                } else {
                    snapshot.current_status = EvidenceDocument::Missing;
                }
                read_model::run_attempts(&snapshot).map(|mut result| {
                    result
                        .as_object_mut()
                        .unwrap()
                        .shift_remove("current_selection_source");
                    result.as_object_mut().unwrap().shift_remove("diagnostics");
                    result
                })
            }
            "attempt_logs" => {
                let mut snapshot = snapshot();
                let attempt = &mut snapshot.attempts[0];
                attempt.collection =
                    evidence("attempt-001", "collection", input["collection"].clone());
                for (name, lines) in input["local_lines"].as_object().unwrap() {
                    let log = LogTail {
                        source: Some(format!("attempts/attempt-001/{name}.log")),
                        lines: serde_json::from_value(lines.clone()).unwrap(),
                        available_lines: lines.as_array().unwrap().len(),
                        ..empty_log()
                    };
                    if name == "stdout" {
                        attempt.stdout = log;
                    } else {
                        attempt.stderr = log;
                    }
                }
                let query = read_model::LogQuery {
                    stream: serde_json::from_value(input["stream"].clone()).unwrap(),
                    lines: input["lines"].as_u64().unwrap() as usize,
                };
                read_model::attempt_logs(&snapshot, "attempt-001", &query).map(|mut result| {
                    assert_eq!(result["follow_supported"], false);
                    for key in ["follow_supported", "read_mode", "diagnostics"] {
                        result.as_object_mut().unwrap().shift_remove(key);
                    }
                    for stream in result["streams"].as_object_mut().unwrap().values_mut() {
                        for key in [
                            "truncated",
                            "offset_bytes",
                            "sampled_bytes",
                            "diagnostics",
                            "file_modified_at_unix",
                        ] {
                            stream.as_object_mut().unwrap().shift_remove(key);
                        }
                    }
                    result
                })
            }
            _ => panic!("unknown fixture operation"),
        };
        match case.get("expected") {
            Some(expected) => assert_eq!(
                &result.unwrap_or_else(|e| panic!("case {index}: {e}")),
                expected,
                "case {index}"
            ),
            None => {
                let error = result.expect_err(&format!("case {index}"));
                assert!(
                    error.to_string().contains(case["error"].as_str().unwrap()),
                    "case {index}: {error}"
                );
            }
        }
    }
}
