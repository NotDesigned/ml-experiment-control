use crate::{
    AppState, read_api,
    read_http::{self, Access},
};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, RawQuery, State},
    response::Response,
    routing::{get, post},
};
use serde_json::{Value, json};
use std::sync::Arc;
fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_RUNTIME_REQUEST",
        detail: "invalid exact native runtime request",
    }
}
fn service(api: &read_api::Api) -> Result<Arc<super::Service>, read_api::Error> {
    api.cloud()
        .filter(|s| s.has_builds())
        .ok_or(read_api::Error {
            status: 501,
            code: "NATIVE_BUILDER_NOT_CONFIGURED",
            detail: "native runtime builder is not configured",
        })
}
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/projects/{project}/runtimes", post(prepare).get(list))
        .route("/api/projects/{project}/runtimes/{runtime}", get(show))
        .route(
            "/api/projects/{project}/runtimes/{runtime}/{operation}",
            post(change).get(observe),
        )
        .layer(DefaultBodyLimit::max(65536))
}
async fn list(
    State(state): State<Arc<AppState>>,
    Path(project): Path<String>,
    RawQuery(query): RawQuery,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Read, move |api, _| {
        service(api)?.runtimes(&project).map_err(super::failure)
    })
    .await
}
async fn prepare(
    State(state): State<Arc<AppState>>,
    Path(project): Path<String>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if ml_exp_core::identity::validate_id(&project).is_err()
        || read_api::no_parameters(query.as_deref()).is_err()
    {
        return read_http::failure(invalid());
    }
    let request: ml_exp_core::cloud::build::Definition = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|v| serde_json::from_value(Value::Object(v)).ok())
    {
        Some(r) => r,
        None => return read_http::failure(invalid()),
    };
    read_http::execute(state, Access::Write, move |api, _| {
        service(api)?
            .prepare_runtime(&project, request)
            .map_err(super::failure)
    })
    .await
}
async fn show(
    State(state): State<Arc<AppState>>,
    Path((project, runtime)): Path<(String, String)>,
    RawQuery(query): RawQuery,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Read, move |api, _| {
        service(api)?
            .runtime_status(&project, &runtime)
            .map_err(super::failure)
    })
    .await
}
async fn observe(
    State(state): State<Arc<AppState>>,
    Path((project, runtime, operation)): Path<(String, String, String)>,
    RawQuery(query): RawQuery,
) -> Response {
    if !matches!(operation.as_str(), "logs" | "progress")
        || read_api::no_parameters(query.as_deref()).is_err()
    {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Read, move |api, _| {
        service(api)?
            .runtime_logs(&project, &runtime, operation == "progress")
            .map_err(super::failure)
    })
    .await
}
async fn change(
    State(state): State<Arc<AppState>>,
    Path((project, runtime, operation)): Path<(String, String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let request = match ml_exp_core::json::mapping(&body) {
        Ok(m) => Value::Object(m),
        Err(_) => return read_http::failure(invalid()),
    };
    let execute = operation == "execute";
    if !(execute && request == json!({"confirmation":format!("BUILD {runtime}")})
        || matches!(operation.as_str(), "reconcile" | "materialize") && request == json!({}))
    {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Write, move |api, _| {
        let service = service(api)?;
        if execute {
            service.execute_runtime(&project, &runtime, &format!("BUILD {runtime}"))
        } else if operation == "materialize" {
            service.materialize_runtime(&project, &runtime, true)
        } else {
            service.reconcile_runtime(&project, &runtime)
        }
        .map_err(super::failure)
    })
    .await
}
