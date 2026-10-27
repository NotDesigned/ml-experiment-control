//! Authenticated GET adapters. Parsing/planning and the read application shell
//! remain separate; bounded blocking filesystem work cannot invoke a provider.
use crate::{
    AppState,
    read_api::{self, Query, View},
};
use axum::{
    Json, Router,
    extract::{Path, RawQuery, State, rejection::PathRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/projects", get(projects))
        .route("/api/runs/{project}/{run}", get(run))
        .route("/api/runs/{project}/{run}/attempts", get(attempts))
        .route(
            "/api/runs/{project}/{run}/attempts/{attempt}/collection",
            get(collection),
        )
        .route("/api/attempts/{project}/{identity}/{view}", get(attempt))
}
fn failure(error: read_api::Error) -> Response {
    let mut response = (
        StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(json!({"code":error.code,"detail":error.detail})),
    )
        .into_response();
    response.headers_mut().insert(
        "x-ml-expd-error-code",
        axum::http::HeaderValue::from_static(error.code),
    );
    response
}
fn bad_path() -> Response {
    failure(read_api::Error {
        status: 422,
        code: "INVALID_EXPERIMENT_IDENTITY",
        detail: "invalid experiment identity",
    })
}
async fn execute(
    state: Arc<AppState>,
    read: impl FnOnce(&read_api::Api, f64) -> Result<Value, read_api::Error> + Send + 'static,
) -> Response {
    let Some(api) = state.reads.clone() else {
        return failure(read_api::Error {
            status: 501,
            code: "RUST_PORT_NOT_CONFIGURED",
            detail: "local experiment reads are not configured in this preview",
        });
    };
    let Ok(permit) = state.read_slots.clone().try_acquire_owned() else {
        let mut response = failure(read_api::Error {
            status: 429,
            code: "READ_BUSY",
            detail: "local read concurrency limit reached",
        });
        response
            .headers_mut()
            .insert("retry-after", axum::http::HeaderValue::from_static("1"));
        return response;
    };
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| read_api::Error {
                status: 503,
                code: "CLOCK_UNAVAILABLE",
                detail: "observation clock is unavailable",
            })?
            .as_secs_f64();
        read(&api, now)
    });
    match tokio::time::timeout(Duration::from_secs(30), task).await {
        Ok(Ok(Ok(value))) => {
            if read_api::contains_secret(&value, &state.auth.token) {
                return failure(read_api::Error {
                    status: 409,
                    code: "OUTWARD_SECRET_DETECTED",
                    detail: "read evidence could not be exposed safely",
                });
            }
            Json(value).into_response()
        }
        Ok(Ok(Err(error))) => failure(error),
        Ok(Err(_)) => failure(read_api::Error {
            status: 500,
            code: "READ_FAILED",
            detail: "local read failed",
        }),
        Err(_) => failure(read_api::Error {
            status: 504,
            code: "READ_TIMEOUT",
            detail: "local read observation timed out; no provider action was sent",
        }),
    }
}
async fn projects(State(state): State<Arc<AppState>>, RawQuery(raw): RawQuery) -> Response {
    if let Err(e) = read_api::no_parameters(raw.as_deref()) {
        return failure(e);
    }
    execute(state, |api, _| api.projects()).await
}
async fn run(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run))) = path else {
        return bad_path();
    };
    if let Err(e) = read_api::no_parameters(raw.as_deref()) {
        return failure(e);
    }
    execute(state, move |api, now| {
        api.query(
            Query {
                project,
                run,
                view: View::Run,
            },
            now,
        )
    })
    .await
}
async fn attempts(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run))) = path else {
        return bad_path();
    };
    if let Err(e) = read_api::no_parameters(raw.as_deref()) {
        return failure(e);
    }
    execute(state, move |api, now| {
        api.query(
            Query {
                project,
                run,
                view: View::Attempts,
            },
            now,
        )
    })
    .await
}
async fn collection(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run, attempt))) = path else {
        return bad_path();
    };
    if let Err(e) = read_api::no_parameters(raw.as_deref()) {
        return failure(e);
    }
    if ml_exp_core::identity::validate_attempt(&attempt).is_err() {
        return bad_path();
    }
    execute(state, move |api, now| {
        api.query(
            Query {
                project,
                run,
                view: View::Collection { attempt },
            },
            now,
        )
    })
    .await
}
async fn attempt(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, identity, view))) = path else {
        return bad_path();
    };
    let (run, attempt) = match read_api::exact_identity(&identity) {
        Ok(value) => value,
        Err(e) => return failure(e),
    };
    let view = match view.as_str() {
        "show" => {
            if let Err(e) = read_api::no_parameters(raw.as_deref()) {
                return failure(e);
            }
            View::Show { attempt }
        }
        "logs" => match read_api::log_query(raw.as_deref()) {
            Ok(query) => View::Logs { attempt, query },
            Err(e) => return failure(e),
        },
        "metrics" => match read_api::metric_query(raw.as_deref()) {
            Ok(query) => View::Metrics { attempt, query },
            Err(e) => return failure(e),
        },
        _ => {
            return failure(read_api::Error {
                status: 404,
                code: "UNKNOWN_ATTEMPT_VIEW",
                detail: "unknown Attempt view",
            });
        }
    };
    execute(state, move |api, now| {
        api.query(Query { project, run, view }, now)
    })
    .await
}
