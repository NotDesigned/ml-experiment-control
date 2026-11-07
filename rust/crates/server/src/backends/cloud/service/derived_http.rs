//! `POST /api/runs/P/R/attempts/A/derived-metrics`: one complete post-hoc JSONL
//! file for an Attempt whose Run has ended. Identity, confirmation and the file's
//! own SHA256 are checked here, before any state is read.
use crate::{
    AppState, api,
    error::Error,
    http_effect::{self, OnTimeout},
};
use axum::{
    Router,
    body::Bytes,
    extract::{
        DefaultBodyLimit, Path, RawQuery, State,
        rejection::{BytesRejection, PathRejection},
    },
    http::StatusCode,
    response::Response,
    routing::post,
};
use ml_exp_core::records::DERIVED_MAX_BYTES;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Room for the largest batch once its text is JSON-escaped.
const BODY: usize = 8 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    sha256: String,
    records: String,
    confirmation: String,
}

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/runs/{project}/{run}/attempts/{attempt}/derived-metrics",
            post(append),
        )
        .layer(DefaultBodyLimit::max(BODY))
}

fn invalid() -> Error {
    Error {
        status: 422,
        code: "INVALID_DERIVED_METRICS_REQUEST",
        detail: "send the exact Attempt, its confirmation, the file's SHA256 and its JSONL text",
    }
}

fn too_large() -> Error {
    Error {
        status: 413,
        code: "DERIVED_BATCH_TOO_LARGE",
        detail: "a derived batch holds at most 5000 records and 2 MiB; split the file",
    }
}

/// The confirmation a client repeats to append to this exact Attempt.
pub(crate) fn confirmation(project: &str, run: &str, attempt: &str) -> String {
    format!("APPEND {project}/{run}/{attempt}")
}

fn validate(project: &str, run: &str, attempt: &str, body: &[u8]) -> Result<Request, Error> {
    if ml_exp_core::identity::validate_id(project).is_err()
        || ml_exp_core::identity::validate_id(run).is_err()
        || ml_exp_core::identity::validate_attempt(attempt).is_err()
    {
        return Err(invalid());
    }
    let request: Request = ml_exp_core::json::mapping(body)
        .ok()
        .and_then(|mapping| serde_json::from_value(serde_json::Value::Object(mapping)).ok())
        .ok_or_else(invalid)?;
    if request.confirmation != confirmation(project, run, attempt) {
        return Err(invalid());
    }
    if request.records.len() > DERIVED_MAX_BYTES {
        return Err(too_large());
    }
    if format!("{:x}", Sha256::digest(request.records.as_bytes())) != request.sha256 {
        return Err(Error {
            status: 422,
            code: "DERIVED_METRICS_DIGEST_MISMATCH",
            detail: "the records differ from the SHA256 the client computed; nothing was stored",
        });
    }
    Ok(request)
}

fn clock(now: f64) -> Result<String, Error> {
    time::OffsetDateTime::from_unix_timestamp(now as i64)
        .map_err(|_| invalid())?
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| invalid())
}

async fn append(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let Ok(Path((project, run, attempt))) = path else {
        return http_effect::failure(invalid());
    };
    if api::no_parameters(raw.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    let body = match body {
        Ok(body) => body,
        Err(error) if error.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            return http_effect::failure(too_large());
        }
        Err(_) => return http_effect::failure(invalid()),
    };
    let request = match validate(&project, &run, &attempt, &body) {
        Ok(request) => request,
        Err(error) => return http_effect::failure(error),
    };
    // Repeating the same file is safe: an accepted batch is returned, not added.
    http_effect::execute(state, OnTimeout::Effect, move |api, now| {
        api.cloud_only()?
            .append_derived(
                &project,
                &run,
                &attempt,
                &request.sha256,
                request.records.as_bytes(),
                &clock(now)?,
            )
            .map_err(Into::into)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(records: &str, sha256: &str, confirmation: &str) -> Vec<u8> {
        serde_json::to_vec(
            &serde_json::json!({"sha256":sha256,"records":records,"confirmation":confirmation}),
        )
        .unwrap()
    }

    #[test]
    fn a_request_names_its_attempt_and_the_digest_of_its_own_text() {
        let records = "{\"name\":\"eval/fid\",\"value\":3.5,\"unit\":\"fid\",\"step\":10}\n";
        let sha256 = format!("{:x}", Sha256::digest(records.as_bytes()));
        let exact = confirmation("demo", "train", "attempt-001");
        assert_eq!(exact, "APPEND demo/train/attempt-001");
        let request = validate(
            "demo",
            "train",
            "attempt-001",
            &body(records, &sha256, &exact),
        );
        assert_eq!(request.unwrap().records, records);
        let wrong = validate(
            "demo",
            "train",
            "attempt-002",
            &body(records, &sha256, &exact),
        );
        assert_eq!(wrong.unwrap_err().code, "INVALID_DERIVED_METRICS_REQUEST");
        let changed = validate("demo", "train", "attempt-001", &body("{}", &sha256, &exact));
        assert_eq!(changed.unwrap_err().code, "DERIVED_METRICS_DIGEST_MISMATCH");
        let large = "x".repeat(DERIVED_MAX_BYTES + 1);
        let digest = format!("{:x}", Sha256::digest(large.as_bytes()));
        let refused = validate(
            "demo",
            "train",
            "attempt-001",
            &body(&large, &digest, &exact),
        );
        assert_eq!(refused.unwrap_err().status, 413);
        assert!(validate("demo", "train", "attempt-001", b"{\"records\":\"\"}").is_err());
    }
}
