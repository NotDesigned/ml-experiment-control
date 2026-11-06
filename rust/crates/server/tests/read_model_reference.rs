use ml_exp_server::experiments::{EvidenceDocument, LogTail};
use ml_exp_server::read_model::{self, Error, MetricQuery};
use serde_json::{Value, json};
#[path = "common/read_model.rs"]
mod support;
use support::*;

fn literal(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}
fn metric_payload(input: &Value) -> Result<Value, Error> {
    let query = MetricQuery {
        max_points: input["max_points"].as_u64().unwrap() as usize,
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

#[test]
fn metric_payload_reports_records_with_coverage() {
    for case in [literal(
        r#"{"operation":"metric_payload","input":{"records":[{"step":0,"timestamp":null,"loss":0,"accuracy":false,"tag":"α"}],"max_points":2,"source":null,"source_attempt_id":null,"contract":{},"names":[]},"expected":{"records":[{"step":0,"name":"loss","value":0,"unit":null,"invalid_value":null,"variant_id":null,"protocol_id":null,"status":"UNDECLARED","error":"UNIT_NOT_DECLARED","errors":["UNIT_NOT_DECLARED"]}],"validation":{"coverage":"complete_history","total_records":1,"failed_records":0,"conflicting_records":0,"incomplete_contexts":1,"state":"PARTIAL","missing_metrics":[]},"total_records":1,"downsampled":false,"missing_names":[],"protocol_id":null,"source_attempt_id":null}}"#,
    )] {
        assert_eq!(metric_payload(&case["input"]).unwrap(), case["expected"]);
    }
}

#[test]
fn downsampling_keeps_both_ends_and_reports_that_it_sampled() {
    let records: Vec<Value> = (0..5)
        .map(|step| json!({"step":step,"loss":step}))
        .collect();
    for (max_points, steps, downsampled) in [
        (1, json!([4]), true),
        (2, json!([0, 4]), true),
        (3, json!([0, 2, 4]), true),
        (5, json!([0, 1, 2, 3, 4]), false),
        (9, json!([0, 1, 2, 3, 4]), false),
    ] {
        let payload =
            metric_payload(&json!({"records":records,"max_points":max_points,"contract":{}}))
                .unwrap();
        let actual: Vec<Value> = payload["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["step"].clone())
            .collect();
        assert_eq!(Value::Array(actual), steps);
        assert_eq!(payload["downsampled"], downsampled);
        assert_eq!(payload["total_records"], 5);
        assert_eq!(payload["validation"]["total_records"], 5);
    }
}

#[test]
fn filters_report_the_names_they_did_not_find() {
    let records = json!([{"step":0,"loss":1,"accuracy":2},{"step":1,"loss":3,"accuracy":4}]);
    let payload = metric_payload(&json!({"records":records,"max_points":10,
        "contract":{},"names":["loss","absent"]}))
    .unwrap();
    assert_eq!(payload["missing_names"], json!(["absent"]));
    assert_eq!(payload["total_records"], 2);
    assert_eq!(payload["validation"]["total_records"], 4);
    let rows = payload["records"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|row| row["name"] == "loss"));
}

#[test]
fn metric_payload_refuses_invalid_queries_with_a_stable_code() {
    for case in [
        literal(
            r#"{"operation":"metric_payload","input":{"records":[],"max_points":0,"source":null,"source_attempt_id":null,"contract":{}},"error":"INVALID_MAX_POINTS"}"#,
        ),
        literal(
            r#"{"operation":"metric_payload","input":{"records":[],"max_points":1,"source":null,"source_attempt_id":null,"contract":{},"names":[""]},"error":"INVALID_METRIC_NAMES"}"#,
        ),
    ] {
        let error = metric_payload(&case["input"]).unwrap_err();
        assert!(
            error.to_string().contains(case["error"].as_str().unwrap()),
            "{error}"
        );
    }
}

#[test]
fn merged_metrics_keep_live_observations_alongside_undeclared_records() {
    for case in [
        literal(
            r#"{"operation":"merge_metrics","input":{"records":[{"step":0,"timestamp":null,"loss":0,"accuracy":false,"tag":"α"}],"live":[{"step":9,"name":"train_loss","value":0,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":null,"status":"VALID","error":null,"source":"tracking.db","errors":[]}],"contract":{},"source":null},"expected":[{"step":0,"name":"loss","value":0,"unit":null,"invalid_value":null,"variant_id":null,"protocol_id":null,"status":"UNDECLARED","error":"UNIT_NOT_DECLARED","source":null,"errors":["UNIT_NOT_DECLARED"]},{"step":9,"name":"train_loss","value":0,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":null,"status":"VALID","error":null,"source":"tracking.db","errors":[]}]}"#,
        ),
        literal(
            r#"{"operation":"merge_metrics","input":{"records":[],"live":[{"step":9,"name":"train_loss","value":0,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":"sha256:protocol-1","status":"VALID","error":null,"source":"tracking.db","errors":[]}],"contract":{"metrics_schema":{"definitions":{"train_loss":{"unit":"nats/token"},"val,loss":{"unit":"nats/token","required":true,"required_when":{"dataset_id":"val","variant_id":"基础"}},"bits/raw byte":{"unit":"bits/raw byte"}}},"protocol_id":"sha256:protocol-1"},"source":null},"expected":[{"step":9,"name":"train_loss","value":0,"unit":"nats/token","invalid_value":null,"variant_id":null,"protocol_id":"sha256:protocol-1","status":"VALID","error":null,"source":"tracking.db","errors":[]}]}"#,
        ),
    ] {
        let input = &case["input"];
        let merged = read_model::merge_metrics(
            input["records"].as_array().unwrap(),
            input["live"].as_array().unwrap(),
            &input["contract"],
            input["source"].as_str(),
        )
        .unwrap();
        assert_eq!(Value::Array(merged), case["expected"]);
    }
}

#[test]
fn run_attempts_mark_the_current_attempt_and_keep_backend_identity() {
    for case in [
        literal(
            r#"{"operation":"run_attempts","input":{"attempts":[],"current_attempt_id":null},"expected":{"project":"demo","run_id":"trial","current_attempt_id":null,"count":0,"attempts":[]}}"#,
        ),
        literal(
            r#"{"operation":"run_attempts","input":{"attempts":[{"attempt_id":"attempt-001","state":"RUNNING","backend":"slurm","backend_job_id":null,"decision":{},"has_submission":false},{"attempt_id":"attempt-002","state":"SUCCEEDED","backend":"sensecore","backend_job_id":"1","decision":{},"has_submission":true}],"current_attempt_id":"attempt-001"},"expected":{"project":"demo","run_id":"trial","current_attempt_id":"attempt-001","count":2,"attempts":[{"identity":"trial::attempt-001","attempt_id":"attempt-001","current":true,"backend":"slurm","backend_job_id":null,"state":"RUNNING","decision":null,"has_submission":false},{"identity":"trial::attempt-002","attempt_id":"attempt-002","current":false,"backend":"sensecore","backend_job_id":"1","state":"SUCCEEDED","decision":null,"has_submission":true}]}}"#,
        ),
    ] {
        let input = &case["input"];
        let mut snapshot = snapshot();
        snapshot.attempts.clear();
        snapshot.current_attempt_id = input["current_attempt_id"].as_str().map(str::to_owned);
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
        let mut result = read_model::run_attempts(&snapshot).unwrap();
        result
            .as_object_mut()
            .unwrap()
            .shift_remove("current_selection_source");
        result.as_object_mut().unwrap().shift_remove("diagnostics");
        assert_eq!(result, case["expected"]);
    }
}

#[test]
fn attempt_logs_prefer_local_files_and_fall_back_to_the_collected_tail() {
    for case in [
        literal(
            r#"{"operation":"attempt_logs","input":{"local":{},"collection":{"process_evidence":{"stdout_tail":[0,false,null,"alpha\nbeta"],"stderr_tail":["历史",{"x":1},"last"],"sources":{"stdout":"remote/stdout","stderr":"remote/stderr"}}},"stream":"stderr","lines":1,"local_lines":{}},"expected":{"project":"demo","run_id":"trial","attempt_id":"attempt-001","streams":{"stderr":{"source":"remote/stderr","mode":"collected_tail","lines":["last"],"available_lines":3}}}}"#,
        ),
        literal(
            r#"{"operation":"attempt_logs","input":{"local":{"stdout":"first\n训练 ☃\nlast\n"},"collection":{"process_evidence":{"stdout_tail":[0,false,null,"alpha\nbeta"],"stderr_tail":["历史",{"x":1},"last"],"sources":{"stdout":"remote/stdout","stderr":"remote/stderr"}}},"stream":"stdout","lines":2,"local_lines":{"stdout":["first","训练 ☃","last"]}},"expected":{"project":"demo","run_id":"trial","attempt_id":"attempt-001","streams":{"stdout":{"source":"attempts/attempt-001/stdout.log","mode":"local_file","lines":["训练 ☃","last"],"available_lines":3}}}}"#,
        ),
    ] {
        let input = &case["input"];
        let mut snapshot = snapshot();
        let attempt = &mut snapshot.attempts[0];
        attempt.collection = evidence("attempt-001", "collection", input["collection"].clone());
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
        let mut result = read_model::attempt_logs(&snapshot, "attempt-001", &query).unwrap();
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
        assert_eq!(result, case["expected"]);
    }
}
