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
        code: "INVALID_EXPERIMENT_PREPARATION",
        detail: "invalid experiment configuration or exact preparation identity",
    }
}
fn service(api: &read_api::Api) -> Result<Arc<crate::cloud_runs::Service>, read_api::Error> {
    api.cloud().ok_or(read_api::Error {
        status: 501,
        code: "EXPERIMENT_PREPARATION_DISABLED",
        detail: "native experiment preparation is not configured",
    })
}
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/projects/{project}/experiment-preparations",
            post(prepare),
        )
        .route(
            "/api/projects/{project}/experiment-preparations/{identity}",
            get(show),
        )
        .route(
            "/api/projects/{project}/experiment-preparations/{identity}/continue",
            post(advance),
        )
        .layer(DefaultBodyLimit::max(128 * 1024))
}
async fn prepare(
    State(state): State<Arc<AppState>>,
    Path(project): Path<String>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let request: ml_exp_core::preparation::Request = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| serde_json::from_value(Value::Object(m)).ok())
    {
        Some(r) => r,
        None => return read_http::failure(invalid()),
    };
    if request.validate(&project).is_err() {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Write, move |api, _| {
        service(api)?
            .experiment_prepare(&project, request, api.submissions().as_deref())
            .map_err(crate::cloud_runs::failure)
    })
    .await
}
async fn show(
    State(state): State<Arc<AppState>>,
    Path((project, id)): Path<(String, String)>,
    RawQuery(query): RawQuery,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Read, move |api, _| {
        service(api)?
            .experiment_status(&project, &id)
            .map_err(crate::cloud_runs::failure)
    })
    .await
}
async fn advance(
    State(state): State<Arc<AppState>>,
    Path((project, id)): Path<(String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let confirmation = format!("PREPARE {id}");
    if ml_exp_core::json::mapping(&body).ok().map(Value::Object)
        != Some(json!({"confirmation":confirmation}))
    {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::RunCreate, move |api, _| {
        let (now, transition) = crate::projects::transition_context()?;
        let timestamp =
            time::OffsetDateTime::parse(&now, &time::format_description::well_known::Rfc3339)
                .map_err(|_| invalid())?;
        let expires = (timestamp + time::Duration::minutes(30))
            .format(&time::format_description::well_known::Rfc3339)
            .map_err(|_| invalid())?;
        service(api)?
            .experiment_continue(
                &project,
                &id,
                &confirmation,
                api.submissions(),
                crate::submissions::Context {
                    now: &now,
                    expires: &expires,
                    transition_id: &transition,
                },
            )
            .map_err(crate::cloud_runs::failure)
    })
    .await
}
