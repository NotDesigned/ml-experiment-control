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
use std::sync::Arc;
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/executors", get(list))
        .route("/api/backends/{executor}/capacity", get(show))
        .route("/api/backends/{executor}/capacity/refresh", post(refresh))
        .layer(DefaultBodyLimit::max(16))
}
async fn list(State(state): State<Arc<AppState>>, RawQuery(query): RawQuery) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Read, move |api, _| {
        let mut rows = api.submissions().map(|s| s.catalogue()).unwrap_or_default();
        if let Some(cloud) = api.cloud() {
            rows.extend(cloud.catalogue().map_err(crate::cloud_runs::failure)?);
        }
        let mut names = std::collections::BTreeSet::new();
        if rows
            .iter()
            .any(|r| !names.insert(r["executor_id"].clone().to_string()))
        {
            return Err(invalid());
        }
        Ok(serde_json::json!({"executors":rows}))
    })
    .await
}
fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_CAPACITY_REQUEST",
        detail: "choose a configured backend and an empty refresh request",
    }
}
async fn show(
    State(state): State<Arc<AppState>>,
    Path(executor): Path<String>,
    RawQuery(query): RawQuery,
) -> Response {
    observe(state, executor, query, false).await
}
async fn refresh(
    State(state): State<Arc<AppState>>,
    Path(executor): Path<String>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if body.as_ref() != b"{}" {
        return read_http::failure(invalid());
    }
    observe(state, executor, query, true).await
}
async fn observe(
    state: Arc<AppState>,
    executor: String,
    query: Option<String>,
    refresh: bool,
) -> Response {
    if ml_exp_core::identity::validate_id(&executor).is_err()
        || read_api::no_parameters(query.as_deref()).is_err()
    {
        return read_http::failure(invalid());
    }
    read_http::execute_timeout(
        state,
        Access::Read,
        std::time::Duration::from_secs(120),
        move |api, _| {
            let (now, _) = crate::projects::transition_context()?;
            if let Some(cloud) = api.cloud() {
                if cloud.accepts(&executor) {
                    return cloud
                        .capacity(&executor, refresh, &now)
                        .map_err(crate::cloud_runs::failure);
                }
            }
            if let Some(slurm) = api.submissions() {
                if slurm.accepts(&executor) {
                    return slurm
                        .capacity(&executor, refresh, &now)
                        .map_err(crate::submissions::http::failure);
                }
            }
            Err(read_api::Error {
                status: 404,
                code: "BACKEND_NOT_FOUND",
                detail: "backend is not configured",
            })
        },
    )
    .await
}
