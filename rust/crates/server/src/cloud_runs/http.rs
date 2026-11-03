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
    routing::post,
};
use ml_exp_core::cloud::exchange::{MAX_BYTES, Packet};
use std::sync::Arc;
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/worker-exchange/{project}/{run}/{attempt}/datasets/{asset}/{part}",
            post(dataset),
        )
        .route(
            "/api/worker-exchange/{project}/{run}/{attempt}",
            post(receive),
        )
        .layer(DefaultBodyLimit::max(MAX_BYTES))
        .route(
            "/api/worker-exchange/{project}/{run}/{attempt}/artifacts",
            post(artifact).layer(DefaultBodyLimit::max(8 * 1024 * 1024)),
        )
}
async fn artifact(
    State(state): State<Arc<AppState>>,
    Path((project, run, attempt)): Path<(String, String, String)>,
    RawQuery(raw): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let invalid = || read_api::Error {
        status: 422,
        code: "INVALID_ARTIFACT_EXCHANGE",
        detail: "invalid cloud artifact request",
    };
    if raw.as_deref().is_some_and(|s| !s.is_empty())
        || ml_exp_core::identity::validate_id(&project)
            .and_then(|_| ml_exp_core::identity::validate_id(&run))
            .and_then(|_| ml_exp_core::identity::validate_attempt(&attempt))
            .is_err()
    {
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
    let request: ml_exp_core::cloud::exchange::ArtifactRequest =
        match ml_exp_core::json::mapping(&body)
            .ok()
            .and_then(|v| serde_json::from_value(serde_json::Value::Object(v)).ok())
        {
            Some(r) => r,
            None => return read_http::failure(invalid()),
        };
    read_http::execute_timeout(
        state,
        Access::Write,
        std::time::Duration::from_secs(3600),
        move |api, now| {
            let cloud = api.cloud().ok_or(read_api::Error {
                status: 501,
                code: "WORKER_EXCHANGE_NOT_CONFIGURED",
                detail: "cloud worker exchange is not configured",
            })?;
            let now = time::OffsetDateTime::from_unix_timestamp(now as i64)
                .map_err(|_| invalid())?
                .format(&time::format_description::well_known::Rfc3339)
                .map_err(|_| invalid())?;
            cloud
                .receive_artifacts(&project, &run, &attempt, &token, request, &now)
                .map_err(super::failure)
        },
    )
    .await
}
async fn receive(
    State(state): State<Arc<AppState>>,
    Path((project, run, attempt)): Path<(String, String, String)>,
    RawQuery(raw): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let invalid = || read_api::Error {
        status: 422,
        code: "INVALID_WORKER_EXCHANGE",
        detail: "invalid bounded worker packet",
    };
    if raw.as_deref().is_some_and(|v| !v.is_empty())
        || ml_exp_core::identity::validate_id(&project)
            .and_then(|_| ml_exp_core::identity::validate_id(&run))
            .and_then(|_| ml_exp_core::identity::validate_attempt(&attempt))
            .is_err()
    {
        return read_http::failure(invalid());
    }
    let values: Vec<_> = headers.get_all("authorization").iter().collect();
    let token = if values.len() == 1 {
        values[0]
            .to_str()
            .ok()
            .and_then(|v| v.strip_prefix("Bearer "))
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
    let packet: Packet = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|v| serde_json::from_value(serde_json::Value::Object(v)).ok())
    {
        Some(p) => p,
        None => return read_http::failure(invalid()),
    };
    read_http::execute(state, Access::Write, move |api, now| {
        let cloud = api.cloud().ok_or(read_api::Error {
            status: 501,
            code: "WORKER_EXCHANGE_NOT_CONFIGURED",
            detail: "cloud worker exchange is not configured",
        })?;
        let now = time::OffsetDateTime::from_unix_timestamp(now as i64)
            .map_err(|_| invalid())?
            .format(&time::format_description::well_known::Rfc3339)
            .map_err(|_| invalid())?;
        let ack = cloud
            .receive(&project, &run, &attempt, &token, &packet, &now)
            .map_err(super::failure)?;
        serde_json::to_value(ack).map_err(|_| invalid())
    })
    .await
}

async fn dataset(
    State(state): State<Arc<AppState>>,
    Path((project, run, attempt, asset, part)): Path<(String, String, String, String, u64)>,
    RawQuery(raw): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let invalid = || read_api::Error {
        status: 422,
        code: "INVALID_DATA_TICKET",
        detail: "invalid exact input data ticket request",
    };
    if read_api::no_parameters(raw.as_deref()).is_err()
        || body.as_ref() != b"{}"
        || !ml_exp_core::datasets::asset_id(&asset)
    {
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
    read_http::execute(state, Access::Read, move |api, _| {
        api.cloud()
            .ok_or_else(invalid)?
            .dataset_ticket(&project, &run, &attempt, &token, &asset, part)
            .map_err(super::failure)
    })
    .await
}
