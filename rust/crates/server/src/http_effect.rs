//! The bounded shell around every local effect a handler runs, read or write:
//! permits, a blocking task, a deadline, the response limit and the secret check.
//! A timed-out write remains uncertain; waiting does not cancel kernel I/O.
use crate::{AppState, api, error::Error};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};

pub(crate) fn failure(error: Error) -> Response {
    if error.status >= 500 {
        crate::why::record_status(file!(), line!(), error.status, error.code);
    }
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
/// What the caller may conclude when no result can be delivered: the deadline
/// passed, the blocking task died, or the result cannot leave this process. The
/// task keeps running after a timeout, so only `NoEffect` may report a plain failure.
#[derive(Clone, Copy)]
pub(crate) enum OnTimeout {
    /// Nothing the caller depends on changes; the attempt can simply be repeated.
    NoEffect,
    /// A project lifecycle transition, which its ledger records.
    Project,
    /// Any other local or provider effect.
    Effect,
    /// A Submission operation, which is reconciled and never repeated blindly.
    Submission,
    /// Run metadata creation.
    RunCreate,
}
impl OnTimeout {
    /// The answer when no result can be delivered. `None` for `NoEffect`, which is the
    /// only case allowed to report a plain failure.
    fn uncertainty(self) -> Option<Error> {
        let (code, detail) = match self {
            Self::NoEffect => return None,
            Self::RunCreate => (
                "RUN_CREATION_UNCERTAIN",
                "Run metadata may have been published; query the same Project/Run before retrying; no scheduler job was submitted by creation",
            ),
            Self::Submission => (
                "SUBMISSION_OPERATION_UNCERTAIN",
                "submission state may have changed; query the same Run or Submission and reconcile; do not repeat execution",
            ),
            Self::Project => (
                "PROJECT_OPERATION_UNCERTAIN",
                "project state may have changed; inspect the lifecycle ledger before retrying",
            ),
            Self::Effect => (
                "OPERATION_UNCERTAIN",
                "the operation may have taken effect; query the same resource before retrying",
            ),
        };
        Some(Error {
            status: 500,
            code,
            detail,
        })
    }
}
fn undelivered(on_timeout: OnTimeout, plain: Error) -> Response {
    failure(on_timeout.uncertainty().unwrap_or(plain))
}
pub(crate) async fn execute(
    state: Arc<AppState>,
    on_timeout: OnTimeout,
    effect: impl FnOnce(&api::Api, f64) -> Result<Value, Error> + Send + 'static,
) -> Response {
    execute_timeout(state, on_timeout, Duration::from_secs(30), effect).await
}
pub(crate) async fn execute_timeout(
    state: Arc<AppState>,
    on_timeout: OnTimeout,
    timeout: Duration,
    effect: impl FnOnce(&api::Api, f64) -> Result<Value, Error> + Send + 'static,
) -> Response {
    let Some(api) = state.api.clone() else {
        return failure(Error {
            status: 501,
            code: "LOCAL_SERVICES_NOT_CONFIGURED",
            detail: "local experiment services are not configured",
        });
    };
    let Ok(permit) = state.effect_slots.clone().try_acquire_owned() else {
        let mut response = failure(Error {
            status: 429,
            code: if matches!(on_timeout, OnTimeout::NoEffect) {
                "READ_BUSY"
            } else {
                "LOCAL_OPERATION_BUSY"
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
            .map_err(|_| Error {
                status: 503,
                code: "CLOCK_UNAVAILABLE",
                detail: "observation clock is unavailable",
            })?
            .as_secs_f64();
        effect(&api, now)
    });
    match tokio::time::timeout(timeout, task).await {
        Ok(Ok(Ok(value))) => {
            if crate::read_model::encoded_size(&value).is_err() {
                return undelivered(
                    on_timeout,
                    Error {
                        status: 413,
                        code: "READ_RESPONSE_LIMIT",
                        detail: "local response exceeds the bounded output limit",
                    },
                );
            }
            if api::contains_secret(&value, &state.auth.token) {
                return undelivered(
                    on_timeout,
                    Error {
                        status: 409,
                        code: "OUTWARD_SECRET_DETECTED",
                        detail: "read evidence could not be exposed safely",
                    },
                );
            }
            Json(value).into_response()
        }
        Ok(Ok(Err(error))) => failure(error),
        Ok(Err(_)) => undelivered(
            on_timeout,
            Error {
                status: 500,
                code: "READ_FAILED",
                detail: "local read failed",
            },
        ),
        Err(_) => undelivered(
            on_timeout,
            Error {
                status: 504,
                code: "READ_TIMEOUT",
                detail: "local read observation timed out; no provider action was sent",
            },
        ),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use axum::body::to_bytes;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };
    const TOKEN: &str = "synthetic-http-boundary-secret-0000000000000000";
    pub(crate) fn state() -> (tempfile::TempDir, Arc<AppState>) {
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
        let api = api::Api::open(settings).unwrap();
        (
            temp,
            Arc::new(AppState {
                data_slots: Arc::new(tokio::sync::Semaphore::new(4)),
                download_slots: Arc::new(tokio::sync::Semaphore::new(4)),
                auth: crate::Auth {
                    token: TOKEN.into(),
                },
                api: Some(Arc::new(api)),
                effect_slots: Arc::new(tokio::sync::Semaphore::new(4)),
            }),
        )
    }
    async fn value(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap()).unwrap()
    }
    #[tokio::test]
    async fn timeout_keeps_real_operation_permit_and_writes_remain_uncertain() {
        for on_timeout in [OnTimeout::NoEffect, OnTimeout::Project] {
            let (_temp, state) = state();
            let (release, wait) = mpsc::channel();
            let (started, ready) = mpsc::channel();
            let completed = Arc::new(AtomicUsize::new(0));
            let observed = completed.clone();
            let response = execute_timeout(
                state.clone(),
                on_timeout,
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
                if matches!(on_timeout, OnTimeout::Project) {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::GATEWAY_TIMEOUT
                }
            );
            let response = value(response).await;
            assert_eq!(
                response["code"],
                if matches!(on_timeout, OnTimeout::Project) {
                    "PROJECT_OPERATION_UNCERTAIN"
                } else {
                    "READ_TIMEOUT"
                }
            );
            ready.recv_timeout(Duration::from_secs(1)).unwrap();
            assert_eq!(state.effect_slots.available_permits(), 3);
            assert_eq!(completed.load(Ordering::SeqCst), 0);
            release.send(()).unwrap();
            let end = std::time::Instant::now() + Duration::from_secs(2);
            while state.effect_slots.available_permits() != 4 && std::time::Instant::now() < end {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            assert_eq!(completed.load(Ordering::SeqCst), 1);
            assert_eq!(state.effect_slots.available_permits(), 4);
        }
    }
    #[tokio::test]
    async fn post_effect_panics_and_secret_response_failures_do_not_report_absence() {
        let (temp, state) = state();
        let marker = temp.path().join("committed-before-response-failure");
        let effect = marker.clone();
        let response = execute(state.clone(), OnTimeout::Project, move |_, _| {
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
        for on_timeout in [OnTimeout::NoEffect, OnTimeout::Project] {
            let response = execute(state.clone(), on_timeout, |_, _| {
                Ok(json!({"reason":TOKEN}))
            })
            .await;
            assert_eq!(
                response.status(),
                if matches!(on_timeout, OnTimeout::Project) {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::CONFLICT
                }
            );
            let response = value(response).await;
            assert_eq!(
                response["code"],
                if matches!(on_timeout, OnTimeout::Project) {
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
        for on_timeout in [OnTimeout::NoEffect, OnTimeout::Project] {
            let response = execute(state.clone(), on_timeout, |_, _| {
                Ok(json!({"large":"x".repeat(crate::read_model::MAX_RESPONSE_BYTES)}))
            })
            .await;
            assert_eq!(
                response.status(),
                if matches!(on_timeout, OnTimeout::Project) {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::PAYLOAD_TOO_LARGE
                }
            );
            assert_eq!(
                value(response).await["code"],
                if matches!(on_timeout, OnTimeout::Project) {
                    "PROJECT_OPERATION_UNCERTAIN"
                } else {
                    "READ_RESPONSE_LIMIT"
                }
            );
        }
    }
    #[tokio::test]
    async fn other_effects_stay_uncertain_without_naming_a_ledger_they_do_not_have() {
        let (_temp, state) = state();
        let (release, wait) = mpsc::channel::<()>();
        let timed_out = execute_timeout(
            state.clone(),
            OnTimeout::Effect,
            Duration::from_millis(10),
            move |_, _| {
                let _ = wait.recv();
                Ok(json!({"done":true}))
            },
        )
        .await;
        let panicked = execute(state.clone(), OnTimeout::Effect, |_, _| {
            panic!("synthetic post-effect worker failure")
        })
        .await;
        let secret = execute(state.clone(), OnTimeout::Effect, |_, _| {
            Ok(json!({"reason":TOKEN}))
        })
        .await;
        let oversized = execute(state.clone(), OnTimeout::Effect, |_, _| {
            Ok(json!({"large":"x".repeat(crate::read_model::MAX_RESPONSE_BYTES)}))
        })
        .await;
        for response in [timed_out, panicked, secret, oversized] {
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            let response = value(response).await;
            assert_eq!(response["code"], "OPERATION_UNCERTAIN");
            let detail = response["detail"].as_str().unwrap();
            assert!(detail.contains("may have taken effect"));
            assert!(!detail.contains("lifecycle"));
            assert!(!response.to_string().contains(TOKEN));
        }
        drop(release);
    }
    #[tokio::test]
    async fn interrupted_submission_response_uses_domain_specific_uncertainty() {
        for (on_timeout, code, needle) in [
            (
                OnTimeout::Submission,
                "SUBMISSION_OPERATION_UNCERTAIN",
                "same Run or Submission",
            ),
            (
                OnTimeout::RunCreate,
                "RUN_CREATION_UNCERTAIN",
                "same Project/Run",
            ),
        ] {
            let (_temp, state) = state();
            let response = execute(state, on_timeout, |_, _| {
                panic!("owned failed response after possible claim")
            })
            .await;
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            let value = value(response).await;
            assert_eq!(value["code"], code);
            assert!(value["detail"].as_str().unwrap().contains(needle));
        }
    }
}
