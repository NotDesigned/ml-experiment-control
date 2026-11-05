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
use ml_exp_core::local::{Packet, Registration};
use std::sync::Arc;
fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_LOCAL_INGRESS",
        detail: "invalid local execution or JSONL packet",
    }
}
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/projects/{project}/local-runs", post(register))
        .route("/api/local-runs/{project}/{run}", get(status))
        .route("/api/local-runs/{project}/{run}/metrics", post(ingest))
        .layer(DefaultBodyLimit::max(512 * 1024))
}
async fn register(
    State(state): State<Arc<AppState>>,
    Path(project): Path<String>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() || body.len() > 131072 {
        return read_http::failure(invalid());
    }
    let Some(request) = ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| serde_json::from_value::<Registration>(m.into()).ok())
    else {
        return read_http::failure(invalid());
    };
    read_http::execute(state, Access::RunCreate, move |api, _| {
        let (now, _) = crate::projects::transition_context()?;
        api.register_local(&project, request, &now)
    })
    .await
}
async fn status(
    State(state): State<Arc<AppState>>,
    Path((project, run)): Path<(String, String)>,
    RawQuery(query): RawQuery,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Read, move |api, _| {
        api.local_status(&project, &run)
    })
    .await
}
async fn ingest(
    State(state): State<Arc<AppState>>,
    Path((project, run)): Path<(String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let Some(packet) = ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| serde_json::from_value::<Packet>(m.into()).ok())
    else {
        return read_http::failure(invalid());
    };
    read_http::execute(state, Access::Write, move |api, _| {
        let (now, _) = crate::projects::transition_context()?;
        api.ingest_local(&project, &run, packet, &now)
    })
    .await
}
