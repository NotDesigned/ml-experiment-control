use crate::{
    AppState, api,
    error::Error,
    http_effect::{self, OnTimeout},
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
fn invalid() -> Error {
    Error {
        status: 422,
        code: "INVALID_DEBUG_REQUEST",
        detail: "invalid debug identity, resources, SSH public key or confirmation",
    }
}
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/debug", get(list).post(create))
        .route("/api/debug/{identity}", get(show))
        .route("/api/debug/{identity}/{operation}", post(change))
        .layer(DefaultBodyLimit::max(16384))
}
fn service(api: &api::Api) -> Result<Arc<super::Service>, Error> {
    api.cloud().filter(|s| s.has_debug()).ok_or(Error {
        status: 501,
        code: "DEBUG_NOT_CONFIGURED",
        detail: "debug machines are not configured",
    })
}
fn timestamp(now: f64) -> Result<String, Error> {
    time::OffsetDateTime::from_unix_timestamp(now as i64)
        .map_err(|_| invalid())?
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| invalid())
}
async fn list(State(state): State<Arc<AppState>>, RawQuery(query): RawQuery) -> Response {
    if api::no_parameters(query.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
        service(api)?.debug_list().map_err(Into::into)
    })
    .await
}
async fn show(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    RawQuery(query): RawQuery,
) -> Response {
    if api::no_parameters(query.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
        service(api)?.debug_status(&id).map_err(Into::into)
    })
    .await
}
async fn create(
    State(state): State<Arc<AppState>>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if api::no_parameters(query.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    let request = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| {
            serde_json::from_value::<ml_exp_core::cloud::debug::Create>(Value::Object(m)).ok()
        })
        .filter(|v| v.validate().is_ok())
    {
        Some(r) => r,
        None => return http_effect::failure(invalid()),
    };
    http_effect::execute_timeout(
        state,
        OnTimeout::Effect,
        std::time::Duration::from_secs(180),
        move |api, now| {
            service(api)?
                .debug_create(request, &timestamp(now)?)
                .map_err(Into::into)
        },
    )
    .await
}
async fn change(
    State(state): State<Arc<AppState>>,
    Path((id, op)): Path<(String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if api::no_parameters(query.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    let Ok(m) = ml_exp_core::json::mapping(&body) else {
        return http_effect::failure(invalid());
    };
    let expected = match op.as_str() {
        "refresh" => json!({}),
        "stop" => json!({"confirmation":format!("STOP {id}")}),
        "delete" => json!({"confirmation":format!("DELETE {id}")}),
        _ => return http_effect::failure(invalid()),
    };
    if Value::Object(m) != expected {
        return http_effect::failure(invalid());
    }
    http_effect::execute_timeout(
        state,
        OnTimeout::Effect,
        std::time::Duration::from_secs(180),
        move |api, now| {
            let service = service(api)?;
            let now = timestamp(now)?;
            if op == "refresh" {
                service.debug_refresh(&id, &now)
            } else {
                service.debug_stop(
                    &id,
                    expected["confirmation"].as_str().ok_or_else(invalid)?,
                    op == "delete",
                    &now,
                )
            }
            .map_err(Into::into)
        },
    )
    .await
}
