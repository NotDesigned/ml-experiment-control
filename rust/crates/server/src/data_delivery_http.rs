use crate::{
    AppState, api,
    error::Error,
    http_effect::{self, OnTimeout},
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
fn invalid() -> Error {
    Error {
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
        .route(
            "/api/projects/{project}/datasets/{executor}/{asset}/upload-ticket",
            post(upload_ticket),
        )
        .route(
            "/api/projects/{project}/datasets/{executor}/{asset}/part-uploaded",
            post(part_uploaded),
        )
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectPart {
    part: u64,
    sha256: String,
}
/// The shape of the request. The service checks it against the asset's definition.
fn direct_part(query: Option<&str>, asset: &str, body: &[u8]) -> Result<DirectPart, Error> {
    api::no_parameters(query).map_err(|_| invalid())?;
    if !ml_exp_core::datasets::asset_id(asset) {
        return Err(invalid());
    }
    ml_exp_core::json::mapping(body)
        .ok()
        .and_then(|m| serde_json::from_value(Value::Object(m)).ok())
        .ok_or_else(invalid)
}
/// Slurm has no object store, and a client takes this refusal to mean "send through
/// the server". An older server has no route at all and no capability to say it does.
fn direct_service(
    api: &api::Api,
    executor: &str,
) -> Result<Arc<crate::backends::cloud::service::Service>, Error> {
    api.cloud()
        .filter(|cloud| cloud.accepts(executor) && cloud.has_datasets())
        .ok_or(Error {
            status: 422,
            code: "DATASET_DIRECT_UPLOAD_UNSUPPORTED",
            detail: "this executor has no object store; send the parts through the server",
        })
}
async fn upload_ticket(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset)): Path<(String, String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    let request = match direct_part(query.as_deref(), &asset, &body) {
        Ok(request) => request,
        Err(error) => return http_effect::failure(error),
    };
    http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
        direct_service(api, &executor)?
            .dataset_upload_ticket(&project, &executor, &asset, request.part, &request.sha256)
            .map_err(Into::into)
    })
    .await
}
async fn part_uploaded(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset)): Path<(String, String, String)>,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Response {
    let request = match direct_part(query.as_deref(), &asset, &body) {
        Ok(request) => request,
        Err(error) => return http_effect::failure(error),
    };
    http_effect::execute(state, OnTimeout::Effect, move |api, _| {
        direct_service(api, &executor)?
            .dataset_part_uploaded(&project, &executor, &asset, request.part, &request.sha256)
            .map_err(Into::into)
    })
    .await
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
    if api::no_parameters(query.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    let request: Request = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| serde_json::from_value(Value::Object(m)).ok())
    {
        Some(r) => r,
        None => return http_effect::failure(invalid()),
    };
    http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
        api.cloud_only()?
            .dataset_download(&project, &executor, &asset, request.part)
            .map_err(Into::into)
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
    if api::no_parameters(query.as_deref()).is_err()
        || ml_exp_core::identity::validate_id(&project)
            .and_then(|_| ml_exp_core::identity::validate_id(&executor))
            .is_err()
        || !ml_exp_core::datasets::asset_id(&asset)
    {
        return http_effect::failure(invalid());
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
        return http_effect::failure(Error {
            status: 401,
            code: "DATASET_AUTH_REQUIRED",
            detail: "data-only capability required",
        });
    };
    let body = match ml_exp_core::json::mapping(&body) {
        Ok(m) => Value::Object(m),
        Err(_) => return http_effect::failure(invalid()),
    };
    if operation == "ready" {
        let receipt = match serde_json::from_value(body) {
            Ok(r) => r,
            Err(_) => return http_effect::failure(invalid()),
        };
        http_effect::execute(state, OnTimeout::Effect, move |api, _| {
            api.cloud_only()?
                .data_ready(&project, &executor, &asset, &token, receipt)
                .map_err(Into::into)
        })
        .await
    } else {
        let part = match operation.parse::<u64>() {
            Ok(n) if n.to_string() == operation && body == serde_json::json!({}) => n,
            _ => return http_effect::failure(invalid()),
        };
        http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
            api.cloud_only()?
                .data_download_ticket(&project, &executor, &asset, &token, part)
                .map_err(Into::into)
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
        #[serde(default)]
        runtime_id: Option<String>,
    }
    if api::no_parameters(query.as_deref()).is_err() {
        return http_effect::failure(invalid());
    }
    let request: Request = match ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|m| serde_json::from_value(Value::Object(m)).ok())
    {
        Some(r) => r,
        None => return http_effect::failure(invalid()),
    };
    if request.confirmation != format!("PUBLISH {asset}") {
        return http_effect::failure(invalid());
    }
    http_effect::execute(state, OnTimeout::Effect, move |api, _| {
        api.cloud_only()?
            .publish_dataset(
                &project,
                &executor,
                &asset,
                &request.request_id,
                request.runtime_id.as_deref(),
            )
            .map_err(Into::into)
    })
    .await
}
