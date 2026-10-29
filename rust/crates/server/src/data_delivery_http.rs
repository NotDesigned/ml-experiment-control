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
use serde_json::Value;
use std::sync::Arc;
fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_DATA_DELIVERY_REQUEST",
        detail: "invalid data-only delivery request",
    }
}
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/dataset-exchange/{project}/{executor}/{asset}/{operation}",
            post(exchange),
        )
        .route(
            "/api/projects/{project}/datasets/{executor}/{asset}/publish",
            post(publish),
        )
        .route(
            "/api/projects/{project}/datasets/{executor}/{asset}/download-ticket",
            post(download),
        )
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
}
async fn download(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset)): Path<(String, String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        part: u64,
    }
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let request: Request = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| serde_json::from_value(Value::Object(m)).ok())
    {
        Some(r) => r,
        None => return read_http::failure(invalid()),
    };
    read_http::execute(state, Access::Read, move |api, _| {
        api.cloud()
            .ok_or_else(invalid)?
            .dataset_download(&project, &executor, &asset, request.part)
            .map_err(crate::cloud_runs::failure)
    })
    .await
}
async fn exchange(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset, operation)): Path<(String, String, String, String)>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if read_api::no_parameters(query.as_deref()).is_err()
        || ml_exp_core::identity::validate_id(&project)
            .and_then(|_| ml_exp_core::identity::validate_id(&executor))
            .is_err()
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
            code: "DATASET_AUTH_REQUIRED",
            detail: "data-only capability required",
        });
    };
    let body = match ml_exp_core::json::mapping(&body) {
        Ok(m) => Value::Object(m),
        Err(_) => return read_http::failure(invalid()),
    };
    if operation == "ready" {
        let receipt = match serde_json::from_value(body) {
            Ok(r) => r,
            Err(_) => return read_http::failure(invalid()),
        };
        read_http::execute(state, Access::Write, move |api, _| {
            api.cloud()
                .ok_or_else(invalid)?
                .data_ready(&project, &executor, &asset, &token, receipt)
                .map_err(crate::cloud_runs::failure)
        })
        .await
    } else {
        let part = match operation.parse::<u64>() {
            Ok(n) if n.to_string() == operation && body == serde_json::json!({}) => n,
            _ => return read_http::failure(invalid()),
        };
        read_http::execute(state, Access::Read, move |api, _| {
            api.cloud()
                .ok_or_else(invalid)?
                .data_download_ticket(&project, &executor, &asset, &token, part)
                .map_err(crate::cloud_runs::failure)
        })
        .await
    }
}
async fn publish(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset)): Path<(String, String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        request_id: String,
        confirmation: String,
    }
    if read_api::no_parameters(query.as_deref()).is_err() {
        return read_http::failure(invalid());
    }
    let request: Request = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| serde_json::from_value(Value::Object(m)).ok())
    {
        Some(r) => r,
        None => return read_http::failure(invalid()),
    };
    if request.confirmation != format!("PUBLISH {asset}") {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Write, move |api, _| {
        api.cloud()
            .ok_or_else(invalid)?
            .publish_dataset(&project, &executor, &asset, &request.request_id)
            .map_err(crate::cloud_runs::failure)
    })
    .await
}
