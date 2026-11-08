//! `POST /api/runs/P/R/attempts/A/purge-state`: remove one ended Attempt's `state/`
//! on backend storage now, under the same conditions as the automatic purge. The
//! identity and confirmation are checked here, before any state is read.
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
    response::Response,
    routing::post,
};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    confirmation: String,
}

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/runs/{project}/{run}/attempts/{attempt}/purge-state",
            post(purge),
        )
        .layer(DefaultBodyLimit::max(4096))
}

fn invalid() -> Error {
    Error {
        status: 422,
        code: "INVALID_STATE_PURGE_REQUEST",
        detail: "send the exact Attempt and its confirmation: PURGE P/R/A, or PURGE UNFINISHED P/R/A for an Attempt whose worker never reported its end",
    }
}

/// The confirmation a client repeats to purge this exact Attempt's `state/`.
pub(crate) fn confirmation(project: &str, run: &str, attempt: &str) -> String {
    format!("PURGE {project}/{run}/{attempt}")
}
/// The same, accepting that the worker never reported its end.
pub(crate) fn unfinished_confirmation(project: &str, run: &str, attempt: &str) -> String {
    format!("PURGE UNFINISHED {project}/{run}/{attempt}")
}

/// Whether the request accepts an unfinished worker.
fn validate(project: &str, run: &str, attempt: &str, body: &[u8]) -> Result<bool, Error> {
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
    if request.confirmation == confirmation(project, run, attempt) {
        Ok(false)
    } else if request.confirmation == unfinished_confirmation(project, run, attempt) {
        Ok(true)
    } else {
        Err(invalid())
    }
}

fn clock(now: f64) -> Result<String, Error> {
    time::OffsetDateTime::from_unix_timestamp(now as i64)
        .map_err(|_| invalid())?
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| invalid())
}

async fn purge(
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
    let Ok(body) = body else {
        return http_effect::failure(invalid());
    };
    let unfinished = match validate(&project, &run, &attempt, &body) {
        Ok(value) => value,
        Err(error) => return http_effect::failure(error),
    };
    // Repeating it is safe: the plan decides the job, so a second request observes it.
    http_effect::execute_timeout(
        state,
        OnTimeout::Effect,
        std::time::Duration::from_secs(180),
        move |api, now| {
            api.cloud_only()?
                .purge_state(&project, &run, &attempt, unfinished, &clock(now)?)
                .map_err(Into::into)
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_names_its_attempt_and_whether_its_worker_finished() {
        let body =
            |text: &str| serde_json::to_vec(&serde_json::json!({"confirmation":text})).unwrap();
        assert!(
            !validate(
                "demo",
                "train",
                "attempt-001",
                &body("PURGE demo/train/attempt-001")
            )
            .unwrap()
        );
        assert!(
            validate(
                "demo",
                "train",
                "attempt-001",
                &body("PURGE UNFINISHED demo/train/attempt-001")
            )
            .unwrap()
        );
        assert!(
            validate(
                "demo",
                "train",
                "attempt-002",
                &body("PURGE demo/train/attempt-001")
            )
            .is_err()
        );
        assert!(validate("demo", "train", "attempt-001", b"{\"confirmation\":1}").is_err());
        assert!(
            validate(
                "../x",
                "train",
                "attempt-001",
                &body("PURGE ../x/train/attempt-001")
            )
            .is_err()
        );
    }
}
