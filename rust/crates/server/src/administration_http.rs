use crate::{
    AppState, read_api,
    read_http::{self, Access},
};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, RawQuery, State},
    http::HeaderMap,
    response::Response,
    routing::{get, post},
};
use serde_json::{Value, json};
use std::sync::Arc;
fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_ADMINISTRATIVE_REQUEST",
        detail: "invalid administrative request or exact confirmation",
    }
}
fn service(api: &read_api::Api) -> Result<Arc<crate::administration::Service>, read_api::Error> {
    api.administration().ok_or(read_api::Error {
        status: 501,
        code: "ADMINISTRATION_DISABLED",
        detail: "administrative deletion is not configured",
    })
}
fn now(value: f64) -> Result<String, read_api::Error> {
    time::OffsetDateTime::from_unix_timestamp(value as i64)
        .map_err(|_| invalid())?
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| invalid())
}
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/admin/submissions/{action}/maintenance", post(prepare))
        .route("/api/admin/maintenance/{identity}", get(show))
        .route("/api/admin/maintenance/{identity}/execute", post(execute))
        .route("/api/admin/runs/{project}/{run}/retire", post(retire))
        .route(
            "/api/worker-exchange/{project}/{run}/{attempt}/maintenance",
            post(receive),
        )
        .layer(DefaultBodyLimit::max(262144))
}
async fn prepare(
    State(state): State<Arc<AppState>>,
    Path(action): Path<String>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let request = match ml_exp_core::json::mapping(&body).ok().and_then(|m| {
        serde_json::from_value::<crate::administration::Request>(Value::Object(m)).ok()
    }) {
        Some(r) => r,
        None => return read_http::failure(invalid()),
    };
    read_http::execute_timeout(
        state,
        Access::Write,
        std::time::Duration::from_secs(180),
        move |api, time| service(api)?.prepare(&action, request, &now(time)?),
    )
    .await
}
async fn show(
    State(state): State<Arc<AppState>>,
    Path(identity): Path<String>,
    RawQuery(query): RawQuery,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Read, move |api, _| {
        service(api)?.status(&identity)
    })
    .await
}
async fn execute(
    State(state): State<Arc<AppState>>,
    Path(identity): Path<String>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let confirmation = format!("DELETE {identity}");
    if ml_exp_core::json::mapping(&body).ok().map(Value::Object)
        != Some(json!({"confirmation":confirmation}))
    {
        return read_http::failure(invalid());
    }
    read_http::execute_timeout(
        state,
        Access::Write,
        std::time::Duration::from_secs(930),
        move |api, _| service(api)?.execute(&identity, &confirmation),
    )
    .await
}
async fn retire(
    State(state): State<Arc<AppState>>,
    Path((project, run)): Path<(String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let confirmation = format!("DELETE {project}/{run}");
    if ml_exp_core::json::mapping(&body).ok().map(Value::Object)
        != Some(json!({"confirmation":confirmation}))
    {
        return read_http::failure(invalid());
    }
    read_http::execute_timeout(
        state,
        Access::Write,
        std::time::Duration::from_secs(180),
        move |api, time| service(api)?.retire(&project, &run, &confirmation, &now(time)?),
    )
    .await
}
async fn receive(
    State(state): State<Arc<AppState>>,
    Path((project, run, attempt)): Path<(String, String, String)>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let values: Vec<_> = headers.get_all("authorization").iter().collect();
    let token = if values.len() == 1 {
        values[0]
            .to_str()
            .ok()
            .and_then(|s| s.strip_prefix("Bearer "))
            .map(str::to_owned)
    } else {
        None
    };
    let Some(token) = token else {
        return read_http::failure(read_api::Error {
            status: 401,
            code: "WORKER_AUTH_REQUIRED",
            detail: "Attempt worker authentication required",
        });
    };
    let receipt = match ml_exp_core::json::mapping(&body) {
        Ok(m) => Value::Object(m),
        Err(_) => return read_http::failure(invalid()),
    };
    read_http::execute(state, Access::Write, move |api, _| {
        service(api)?.receive(&project, &run, &attempt, &token, receipt)
    })
    .await
}
