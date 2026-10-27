//! Authenticated observation adapters and a shared bounded local effect shell.
//! A timed-out write remains uncertain; waiting does not cancel kernel I/O.
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
pub(crate) fn failure(error: read_api::Error) -> Response {
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
#[derive(Clone, Copy)]
pub(crate) enum Access {
    Read,
    Write,
}
fn uncertain() -> Response {
    failure(read_api::Error {
        status: 500,
        code: "PROJECT_OPERATION_UNCERTAIN",
        detail: "project state may have changed; inspect the lifecycle ledger before retrying",
    })
}
pub(crate) async fn execute(
    state: Arc<AppState>,
    access: Access,
    read: impl FnOnce(&read_api::Api, f64) -> Result<Value, read_api::Error> + Send + 'static,
) -> Response {
    execute_with_timeout(state, access, Duration::from_secs(30), read).await
}
pub(crate) async fn execute_timeout(
    state: Arc<AppState>,
    access: Access,
    timeout: Duration,
    read: impl FnOnce(&read_api::Api, f64) -> Result<Value, read_api::Error> + Send + 'static,
) -> Response {
    execute_with_timeout(state, access, timeout, read).await
}
async fn execute_with_timeout(
    state: Arc<AppState>,
    access: Access,
    timeout: Duration,
    read: impl FnOnce(&read_api::Api, f64) -> Result<Value, read_api::Error> + Send + 'static,
) -> Response {
    let Some(api) = state.reads.clone() else {
        return failure(read_api::Error {
            status: 501,
            code: "RUST_PORT_NOT_CONFIGURED",
            detail: "local experiment services are not configured in this preview",
        });
    };
    let Ok(permit) = state.read_slots.clone().try_acquire_owned() else {
        let mut response = failure(read_api::Error {
            status: 429,
            code: if matches!(access, Access::Write) {
                "LOCAL_OPERATION_BUSY"
            } else {
                "READ_BUSY"
            },
            detail: "local operation concurrency limit reached",
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
    match tokio::time::timeout(timeout, task).await {
        Ok(Ok(Ok(value))) => {
            if crate::read_model::encoded_size(&value).is_err() {
                if matches!(access, Access::Write) {
                    return uncertain();
                }
                return failure(read_api::Error {
                    status: 413,
                    code: "READ_RESPONSE_LIMIT",
                    detail: "local response exceeds the bounded output limit",
                });
            }
            if read_api::contains_secret(&value, &state.auth.token) {
                if matches!(access, Access::Write) {
                    return uncertain();
                }
                return failure(read_api::Error {
                    status: 409,
                    code: "OUTWARD_SECRET_DETECTED",
                    detail: "read evidence could not be exposed safely",
                });
            }
            Json(value).into_response()
        }
        Ok(Ok(Err(error))) => failure(error),
        Ok(Err(_)) if matches!(access, Access::Write) => uncertain(),
        Ok(Err(_)) => failure(read_api::Error {
            status: 500,
            code: "READ_FAILED",
            detail: "local read failed",
        }),
        Err(_) if matches!(access, Access::Write) => uncertain(),
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
    execute(state, Access::Read, |api, now| api.projects(now)).await
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
    execute(state, Access::Read, move |api, now| {
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
    execute(state, Access::Read, move |api, now| {
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
    execute(state, Access::Read, move |api, now| {
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
    execute(state, Access::Read, move |api, now| {
        api.query(Query { project, run, view }, now)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };
    const TOKEN: &str = "synthetic-http-boundary-secret-0000000000000000";
    fn state() -> (tempfile::TempDir, Arc<AppState>) {
        let temp = tempfile::tempdir().unwrap();
        let registry = temp.path().join("registry");
        crate::read_config::initialize(&registry).unwrap();
        let settings = crate::read_config::Settings::from_json(
            &serde_json::to_vec(
                &json!({"schema_version":1,"registry_root":registry,"projects":[]}),
            )
            .unwrap(),
        )
        .unwrap();
        let api = read_api::Api::open(settings).unwrap();
        (
            temp,
            Arc::new(AppState {
                auth: crate::Auth {
                    token: TOKEN.into(),
                },
                reads: Some(Arc::new(api)),
                read_slots: Arc::new(tokio::sync::Semaphore::new(4)),
            }),
        )
    }
    async fn value(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap()).unwrap()
    }
    #[tokio::test]
    async fn timeout_keeps_real_operation_permit_and_writes_remain_uncertain() {
        for access in [Access::Read, Access::Write] {
            let (_temp, state) = state();
            let (release, wait) = mpsc::channel();
            let (started, ready) = mpsc::channel();
            let completed = Arc::new(AtomicUsize::new(0));
            let observed = completed.clone();
            let response = execute_with_timeout(
                state.clone(),
                access,
                Duration::from_millis(10),
                move |_, _| {
                    started.send(()).unwrap();
                    wait.recv().unwrap();
                    observed.fetch_add(1, Ordering::SeqCst);
                    Ok(json!({"done":true}))
                },
            )
            .await;
            assert_eq!(
                response.status(),
                if matches!(access, Access::Write) {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::GATEWAY_TIMEOUT
                }
            );
            let response = value(response).await;
            assert_eq!(
                response["code"],
                if matches!(access, Access::Write) {
                    "PROJECT_OPERATION_UNCERTAIN"
                } else {
                    "READ_TIMEOUT"
                }
            );
            ready.recv_timeout(Duration::from_secs(1)).unwrap();
            assert_eq!(state.read_slots.available_permits(), 3);
            assert_eq!(completed.load(Ordering::SeqCst), 0);
            release.send(()).unwrap();
            let end = std::time::Instant::now() + Duration::from_secs(2);
            while state.read_slots.available_permits() != 4 && std::time::Instant::now() < end {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            assert_eq!(completed.load(Ordering::SeqCst), 1);
            assert_eq!(state.read_slots.available_permits(), 4);
        }
    }
    #[tokio::test]
    async fn post_effect_panics_and_secret_response_failures_do_not_report_absence() {
        let (temp, state) = state();
        let marker = temp.path().join("committed-before-response-failure");
        let effect = marker.clone();
        let response = execute(state.clone(), Access::Write, move |_, _| {
            std::fs::write(effect, b"committed synthetic effect").unwrap();
            panic!("synthetic post-effect worker failure")
        })
        .await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(value(response).await["code"], "PROJECT_OPERATION_UNCERTAIN");
        assert_eq!(
            std::fs::read(marker).unwrap(),
            b"committed synthetic effect"
        );
        for access in [Access::Read, Access::Write] {
            let response = execute(state.clone(), access, |_, _| Ok(json!({"reason":TOKEN}))).await;
            assert_eq!(
                response.status(),
                if matches!(access, Access::Write) {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::CONFLICT
                }
            );
            let response = value(response).await;
            assert_eq!(
                response["code"],
                if matches!(access, Access::Write) {
                    "PROJECT_OPERATION_UNCERTAIN"
                } else {
                    "OUTWARD_SECRET_DETECTED"
                }
            );
            assert!(!response.to_string().contains(TOKEN));
        }
    }
    #[tokio::test]
    async fn oversized_post_response_remains_uncertain_while_read_is_explicitly_limited() {
        let (_temp, state) = state();
        for access in [Access::Read, Access::Write] {
            let response = execute(state.clone(), access, |_, _| {
                Ok(json!({"large":"x".repeat(crate::read_model::MAX_RESPONSE_BYTES)}))
            })
            .await;
            assert_eq!(
                response.status(),
                if matches!(access, Access::Write) {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::PAYLOAD_TOO_LARGE
                }
            );
            assert_eq!(
                value(response).await["code"],
                if matches!(access, Access::Write) {
                    "PROJECT_OPERATION_UNCERTAIN"
                } else {
                    "READ_RESPONSE_LIMIT"
                }
            );
        }
    }
}
