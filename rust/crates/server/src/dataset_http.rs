//! Dataset routes for either backend, chosen by which one accepts the executor.
//! Acquire bounded ingress capacity before reading a dataset part. No payload
//! file is created on the API host; durable acknowledgement belongs to gateway.
use crate::{
    AppState, api,
    error::Error,
    http_effect::{self, OnTimeout},
};
use axum::{
    Router,
    body::to_bytes,
    extract::{Path, RawQuery, Request, State},
    response::Response,
    routing::{get, post, put},
};
use ml_exp_core::datasets::{self, Command, Definition};
use serde_json::{Value, json};
use std::{
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/projects/{project}/datasets/{executor}/import-oss",
            post(import_oss),
        )
        .route(
            "/api/projects/{project}/datasets/{executor}",
            get(list).post(begin),
        )
        .route(
            "/api/projects/{project}/datasets/{executor}/{asset}",
            get(status),
        )
        .route(
            "/api/projects/{project}/datasets/{executor}/{asset}/parts/{number}/{sha256}",
            put(part),
        )
        .route(
            "/api/projects/{project}/datasets/{executor}/{asset}/complete",
            post(complete),
        )
}
async fn import_oss(
    State(state): State<Arc<AppState>>,
    Path((project, executor)): Path<(String, String)>,
    RawQuery(raw): RawQuery,
    request: Request,
) -> Response {
    if let Err(e) = scope(&project, &executor, None, raw.as_deref()) {
        return http_effect::failure(e);
    }
    let (body, permit) = match ingress(&state, request, 8192).await {
        Ok(value) => value,
        Err(e) => return http_effect::failure(e),
    };
    let source = ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|v| serde_json::from_value::<datasets::oss::Import>(Value::Object(v)).ok());
    let Some(source) = source.filter(|v| v.validate().is_ok()) else {
        return http_effect::failure(invalid());
    };
    http_effect::execute_timeout(
        state,
        OnTimeout::Effect,
        Duration::from_secs(30),
        move |api, _| {
            let _permit = permit;
            let cloud = api.cloud().filter(|c| c.accepts(&executor)).ok_or(Error {
                status: 422,
                code: "OSS_DATASET_IMPORT_UNSUPPORTED",
                detail: "this executor does not support configured object-store imports",
            })?;
            cloud
                .import_oss_dataset(&project, &executor, source)
                .map_err(Into::into)
        },
    )
    .await
}
fn invalid() -> Error {
    Error {
        status: 422,
        code: "INVALID_DATASET_REQUEST",
        detail: "invalid dataset scope, archive definition or part",
    }
}
async fn list(
    State(state): State<Arc<AppState>>,
    Path((project, executor)): Path<(String, String)>,
    RawQuery(raw): RawQuery,
) -> Response {
    if let Err(e) = scope(&project, &executor, None, raw.as_deref()) {
        return http_effect::failure(e);
    }
    http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
        if let Some(cloud) = api.cloud().filter(|c| c.accepts(&executor)) {
            return cloud.dataset_list(&project, &executor).map_err(Into::into);
        }
        api.slurm_only()?
            .dataset_list(&project, &executor)
            .map_err(Into::into)
    })
    .await
}
fn scope(
    project: &str,
    executor: &str,
    asset: Option<&str>,
    raw: Option<&str>,
) -> Result<(), Error> {
    ml_exp_core::identity::validate_id(project).map_err(|_| invalid())?;
    ml_exp_core::identity::validate_id(executor).map_err(|_| invalid())?;
    api::no_parameters(raw)?;
    if asset.is_some_and(|v| !datasets::asset_id(v)) {
        return Err(invalid());
    }
    Ok(())
}
async fn status(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset)): Path<(String, String, String)>,
    RawQuery(raw): RawQuery,
) -> Response {
    if let Err(e) = scope(&project, &executor, Some(&asset), raw.as_deref()) {
        return http_effect::failure(e);
    }
    http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
        if let Some(cloud) = api.cloud().filter(|c| c.accepts(&executor)) {
            return cloud
                .dataset_state(&project, &executor, &asset)
                .map_err(Into::into);
        }
        let result = api
            .slurm_only()?
            .dataset_state(&project, &executor, &asset)?;
        Ok(json!(result))
    })
    .await
}
async fn ingress(
    state: &Arc<AppState>,
    request: Request,
    limit: usize,
) -> Result<(Vec<u8>, tokio::sync::OwnedSemaphorePermit), Error> {
    let permit = state
        .data_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| Error {
            status: 429,
            code: "DATASET_TRANSFER_BUSY",
            detail: "dataset transfer buffer is busy; resume the same asset",
        })?;
    let bytes = tokio::time::timeout(
        Duration::from_secs(90),
        to_bytes(request.into_body(), limit),
    )
    .await
    .map_err(|_| Error {
        status: 408,
        code: "DATASET_BODY_TIMEOUT",
        detail: "dataset request body timed out; no gateway request was issued",
    })?
    .map_err(|_| Error {
        status: 413,
        code: "DATASET_BODY_LIMIT",
        detail: "dataset part exceeds its fixed bound or the request body is invalid",
    })?;
    Ok((bytes.to_vec(), permit))
}
async fn transfer(
    state: Arc<AppState>,
    project: String,
    executor: String,
    command: Command,
    payload: Vec<u8>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    http_effect::execute_timeout(
        state,
        OnTimeout::Effect,
        Duration::from_secs(1230),
        move |api, _| {
            let _permit = permit;
            if let Some(cloud) = api.cloud().filter(|c| c.accepts(&executor)) {
                return cloud
                    .dataset_transfer(&project, &executor, command, &payload)
                    .map_err(Into::into);
            }
            let value = api.slurm_only()?.dataset_transfer(
                &project,
                &executor,
                command,
                &payload,
                &AtomicBool::new(false),
            )?;
            Ok(json!(value))
        },
    )
    .await
}
async fn begin(
    State(state): State<Arc<AppState>>,
    Path((project, executor)): Path<(String, String)>,
    RawQuery(raw): RawQuery,
    request: Request,
) -> Response {
    if let Err(e) = scope(&project, &executor, None, raw.as_deref()) {
        return http_effect::failure(e);
    }
    let (body, permit) = match ingress(&state, request, datasets::MAX_METADATA / 2).await {
        Ok(v) => v,
        Err(e) => return http_effect::failure(e),
    };
    let definition = ml_exp_core::json::mapping(&body)
        .ok()
        .and_then(|value| serde_json::from_value::<Definition>(Value::Object(value)).ok());
    let Some(definition) = definition.filter(|d| d.validate().is_ok()) else {
        return http_effect::failure(invalid());
    };
    transfer(
        state,
        project,
        executor,
        Command::Begin { definition },
        Vec::new(),
        permit,
    )
    .await
}
async fn part(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset, number, sha256)): Path<(String, String, String, u64, String)>,
    RawQuery(raw): RawQuery,
    request: Request,
) -> Response {
    if let Err(e) = scope(&project, &executor, Some(&asset), raw.as_deref()) {
        return http_effect::failure(e);
    }
    if number >= ml_exp_core::multipart::MAX_PARTS
        || ml_exp_core::sources::validate_archive_digest(&sha256).is_err()
    {
        return http_effect::failure(invalid());
    }
    let (body, permit) =
        match ingress(&state, request, ml_exp_core::multipart::PART_BYTES as usize).await {
            Ok(v) => v,
            Err(e) => return http_effect::failure(e),
        };
    let command = Command::Part {
        asset_id: asset,
        number,
        sha256,
        bytes: body.len() as u64,
    };
    transfer(state, project, executor, command, body, permit).await
}
async fn complete(
    State(state): State<Arc<AppState>>,
    Path((project, executor, asset)): Path<(String, String, String)>,
    RawQuery(raw): RawQuery,
    request: Request,
) -> Response {
    if let Err(e) = scope(&project, &executor, Some(&asset), raw.as_deref()) {
        return http_effect::failure(e);
    }
    let (body, permit) = match ingress(&state, request, 1).await {
        Ok(v) => v,
        Err(e) => return http_effect::failure(e),
    };
    if !body.is_empty() {
        return http_effect::failure(invalid());
    }
    transfer(
        state,
        project,
        executor,
        Command::Complete { asset_id: asset },
        Vec::new(),
        permit,
    )
    .await
}
