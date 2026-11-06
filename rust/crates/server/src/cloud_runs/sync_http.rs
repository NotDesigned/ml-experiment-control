use crate::{
    AppState, read_api,
    read_http::{self, Access},
};
use axum::{
    Router,
    body::Bytes,
    extract::{Path, RawQuery, State},
    http::Method,
    response::Response,
    routing::{get, post},
};
use std::sync::{Arc, atomic::AtomicBool};
fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_SYNC_REQUEST",
        detail: "invalid original synchronization identity or retry confirmation",
    }
}
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/submissions/{action}/sync/{request}", get(show))
        .route(
            "/api/runs/{project}/{run}/attempts/{attempt}/provider-diagnostics/refresh",
            post(provider_diagnostics),
        )
        .route(
            "/api/projects/{project}/datasets/{executor}/{asset}/publications/{request}/provider-diagnostics/refresh",
            post(data_provider_diagnostics),
        )
        .route(
            "/api/submissions/{action}/sync/{request}/retry",
            post(retry),
        )
        .route(
            "/api/submissions/{action}/sync/{request}/diagnostic",
            get(diagnose_sync).post(diagnose_sync),
        )
        .route(
            "/api/projects/{project}/datasets/{executor}/{asset}/publications/{request}/diagnostic",
            get(diagnose_data).post(diagnose_data),
        )
}
fn diagnostic_lines(query: Option<&str>) -> Option<u32> {
    let Some(query) = query else {
        return Some(200);
    };
    query
        .strip_prefix("lines=")?
        .parse()
        .ok()
        .filter(|lines| (1..=2000).contains(lines))
}
async fn provider_diagnostics(
    State(state): State<Arc<AppState>>,
    Path((project, run, attempt)): Path<(String, String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    let Some(lines) = diagnostic_lines(query.as_deref()) else {
        return read_http::failure(invalid());
    };
    if !empty(&body, true) {
        return read_http::failure(invalid());
    }
    read_http::execute_timeout(
        state,
        Access::Read,
        std::time::Duration::from_secs(120),
        move |api, _| {
            api.cloud()
                .ok_or_else(invalid)?
                .provider_diagnostics(&project, &run, &attempt, lines)
                .map_err(Into::into)
        },
    )
    .await
}
async fn data_provider_diagnostics(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset, request)): Path<(String, String, String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    let Some(lines) = diagnostic_lines(query.as_deref()) else {
        return read_http::failure(invalid());
    };
    if !empty(&body, true) {
        return read_http::failure(invalid());
    }
    read_http::execute_timeout(
        state,
        Access::Read,
        std::time::Duration::from_secs(120),
        move |api, _| {
            api.cloud()
                .ok_or_else(invalid)?
                .data_provider_diagnostics(&project, &executor, &asset, &request, lines)
                .map_err(Into::into)
        },
    )
    .await
}
async fn diagnose_sync(
    State(state): State<Arc<AppState>>,
    Path((action, request)): Path<(String, String)>,
    RawQuery(query): RawQuery,
    method: Method,
    body: Bytes,
) -> Response {
    let execute = method == Method::POST;
    if read_api::no_parameters(query.as_deref()).is_err() || !empty(&body, execute) {
        return read_http::failure(invalid());
    }
    read_http::execute_timeout(
        state,
        if execute {
            Access::Submission
        } else {
            Access::Read
        },
        std::time::Duration::from_secs(120),
        move |api, now| {
            api.cloud()
                .ok_or_else(invalid)?
                .sync_diagnostic(&action, &request, execute, &clock(now)?)
                .map_err(Into::into)
        },
    )
    .await
}
async fn diagnose_data(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset, request)): Path<(String, String, String, String)>,
    RawQuery(query): RawQuery,
    method: Method,
    body: Bytes,
) -> Response {
    let execute = method == Method::POST;
    if read_api::no_parameters(query.as_deref()).is_err() || !empty(&body, execute) {
        return read_http::failure(invalid());
    }
    read_http::execute_timeout(
        state,
        if execute {
            Access::Submission
        } else {
            Access::Read
        },
        std::time::Duration::from_secs(120),
        move |api, now| {
            api.cloud()
                .ok_or_else(invalid)?
                .data_diagnostic(&project, &executor, &asset, &request, execute, &clock(now)?)
                .map_err(Into::into)
        },
    )
    .await
}
fn empty(body: &[u8], execute: bool) -> bool {
    if !execute {
        return body.is_empty();
    }
    ml_exp_core::json::mapping(body).is_ok_and(|m| m.is_empty())
}
fn clock(now: f64) -> Result<String, read_api::Error> {
    time::OffsetDateTime::from_unix_timestamp(now as i64)
        .map_err(|_| invalid())?
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| invalid())
}
async fn show(
    State(state): State<Arc<AppState>>,
    Path((action, request)): Path<(String, String)>,
    RawQuery(query): RawQuery,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let check = ml_exp_core::cloud::exchange::Sync {
        request_id: request.clone(),
        confirmation: format!("SYNC {action}"),
    };
    if check.validate(&action).is_err() {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Read, move |api, now| {
        api.cloud()
            .ok_or_else(invalid)?
            .sync_status(&action, &request, &clock(now)?)
            .map_err(Into::into)
    })
    .await
}
async fn retry(
    State(state): State<Arc<AppState>>,
    Path((action, request)): Path<(String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let retry: ml_exp_core::cloud::exchange::SyncRetry = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| serde_json::from_value(serde_json::Value::Object(m)).ok())
    {
        Some(r) => r,
        None => return read_http::failure(invalid()),
    };
    if retry.validate(&action, &request).is_err() {
        return read_http::failure(invalid());
    }
    read_http::execute_timeout(
        state,
        Access::Submission,
        std::time::Duration::from_secs(120),
        move |api, now| {
            api.cloud()
                .ok_or_else(invalid)?
                .retry_sync(
                    &action,
                    &request,
                    &retry,
                    &clock(now)?,
                    &AtomicBool::new(false),
                )
                .map_err(Into::into)
        },
    )
    .await
}
