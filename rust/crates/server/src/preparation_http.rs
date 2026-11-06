use crate::{
    AppState,
    http_effect::{self, OnTimeout},
    read_api,
    runs::Owner,
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
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/runs/{project}/{run}/preparation", get(discover))
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
async fn discover(
    State(state): State<Arc<AppState>>,
    Path((project, run)): Path<(String, String)>,
    RawQuery(query): RawQuery,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    http_effect::execute(state, OnTimeout::NoEffect, move |api, now| {
        let preparation = match api.cloud() {
            Some(cloud) => cloud.experiment_for_run(&project, &run)?,
            None => None,
        };
        if let Some(value) = preparation {
            return Ok(value);
        }
        let now = timestamp(now)?;
        match api.run_owner(&project, &run)? {
            Owner::Cloud(cloud) => cloud.for_run(&project, &run, &now).map_err(Into::into),
            Owner::Slurm(slurm) => slurm.for_run(&project, &run, &now).map_err(Into::into),
        }
    })
    .await
}
fn timestamp(now: f64) -> Result<String, read_api::Error> {
    time::OffsetDateTime::from_unix_timestamp(now as i64)
        .map_err(|_| invalid())?
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| invalid())
}
async fn prepare(
    State(state): State<Arc<AppState>>,
    Path(project): Path<String>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    let request: ml_exp_core::preparation::Request = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| serde_json::from_value(Value::Object(m)).ok())
    {
        Some(r) => r,
        None => return http_effect::failure(invalid()),
    };
    if request.validate(&project).is_err() {
        return http_effect::failure(invalid());
    }
    http_effect::execute(state, OnTimeout::Effect, move |api, _| {
        api.cloud_only()?
            .experiment_prepare(&project, request, api.submissions().as_deref())
            .map_err(Into::into)
    })
    .await
}
async fn show(
    State(state): State<Arc<AppState>>,
    Path((project, id)): Path<(String, String)>,
    RawQuery(query): RawQuery,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
        api.cloud_only()?
            .experiment_status(&project, &id)
            .map_err(Into::into)
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
        return http_effect::failure(invalid());
    }
    let confirmation = format!("PREPARE {id}");
    if ml_exp_core::json::mapping(&body).ok().map(Value::Object)
        != Some(json!({"confirmation":confirmation}))
    {
        return http_effect::failure(invalid());
    }
    http_effect::execute(state, OnTimeout::RunCreate, move |api, _| {
        let (now, transition) = crate::projects::transition_context()?;
        let timestamp =
            time::OffsetDateTime::parse(&now, &time::format_description::well_known::Rfc3339)
                .map_err(|_| invalid())?;
        let expires = (timestamp + time::Duration::minutes(30))
            .format(&time::format_description::well_known::Rfc3339)
            .map_err(|_| invalid())?;
        api.cloud_only()?
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
            .map_err(Into::into)
    })
    .await
}
