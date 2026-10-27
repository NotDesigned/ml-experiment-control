//! HTTP adapters for bounded source archive publication and exact lookup.
use super::Service;
use crate::{
    read_api,
    read_http::{self, Access},
};
use axum::{
    Router,
    body::Body,
    extract::{Path, RawQuery, State},
    response::Response,
    routing::{get, post},
};
use http_body_util::BodyExt;
use std::{
    io::{Seek, Write},
    sync::Arc,
    time::Duration,
};
use tempfile::NamedTempFile;

const MAX_BODY: usize = 512 * 1024 * 1024;

pub(crate) fn routes() -> Router<Arc<crate::AppState>> {
    Router::new()
        .route("/api/source-imports/archive", post(upload))
        .route("/api/projects/{project}/sources/{source_id}", get(read))
        .route(
            "/api/projects/{project}/source-imports/{archive_sha256}",
            get(lookup),
        )
}

fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_SOURCE_REQUEST",
        detail: "invalid source project, archive digest or query",
    }
}
fn service(api: &read_api::Api) -> Result<Arc<Service>, read_api::Error> {
    api.sources().ok_or(read_api::Error {
        status: 501,
        code: "SOURCE_ARCHIVE_NOT_CONFIGURED",
        detail: "source archive storage is not enabled in this isolated preview",
    })
}
fn query(raw: Option<&str>) -> Result<(String, String), read_api::Error> {
    let values = read_api::parameters(raw)?;
    let mut project = None;
    let mut sha = None;
    for (key, value) in values {
        match key.as_str() {
            "project" if project.is_none() => project = Some(value),
            "sha256" if sha.is_none() => sha = Some(value),
            _ => return Err(invalid()),
        }
    }
    Ok((project.ok_or_else(invalid)?, sha.ok_or_else(invalid)?))
}

async fn upload(
    State(state): State<Arc<crate::AppState>>,
    RawQuery(raw): RawQuery,
    body: Body,
) -> Response {
    let (project, digest) = match query(raw.as_deref()) {
        Ok(value) => value,
        Err(error) => return read_http::failure(error),
    };
    let source_service = match state.reads.clone().and_then(|api| service(&api).ok()) {
        Some(value) => value,
        None => {
            return read_http::failure(read_api::Error {
                status: 501,
                code: "SOURCE_ARCHIVE_NOT_CONFIGURED",
                detail: "source archive storage is not enabled in this isolated preview",
            });
        }
    };
    let limit = source_service.maximum_archive_bytes().min(MAX_BODY as u64);
    if let Err(error) = source_service.authorize_upload(&project, &digest) {
        return read_http::failure(error);
    }
    let mut staged = match NamedTempFile::new() {
        Ok(file) => file,
        Err(_) => {
            return read_http::failure(read_api::Error {
                status: 503,
                code: "SOURCE_ARCHIVE_STAGE_UNAVAILABLE",
                detail: "source archive could not be staged safely",
            });
        }
    };
    let mut received = 0u64;
    let mut body = body;
    while let Some(frame) = body.frame().await {
        let frame = match frame {
            Ok(frame) => frame,
            Err(_) => {
                return read_http::failure(read_api::Error {
                    status: 400,
                    code: "SOURCE_ARCHIVE_BODY_INVALID",
                    detail: "source archive request body could not be read",
                });
            }
        };
        let data = match frame.into_data() {
            Ok(data) => data,
            Err(_) => {
                return read_http::failure(read_api::Error {
                    status: 400,
                    code: "SOURCE_ARCHIVE_BODY_INVALID",
                    detail: "source archive request body contained unsupported trailers",
                });
            }
        };
        received = match received.checked_add(data.len() as u64) {
            Some(value) if value <= limit => value,
            _ => {
                return read_http::failure(read_api::Error {
                    status: 413,
                    code: "SOURCE_ARCHIVE_LIMIT",
                    detail: "source archive exceeds the configured upload limit",
                });
            }
        };
        if staged.write_all(&data).is_err() {
            return read_http::failure(read_api::Error {
                status: 503,
                code: "SOURCE_ARCHIVE_STAGE_UNAVAILABLE",
                detail: "source archive could not be staged safely",
            });
        }
    }
    if staged.as_file_mut().sync_all().is_err() {
        return read_http::failure(read_api::Error {
            status: 503,
            code: "SOURCE_ARCHIVE_STAGE_UNAVAILABLE",
            detail: "source archive could not be made durable",
        });
    }
    if staged.as_file_mut().rewind().is_err() {
        return read_http::failure(read_api::Error {
            status: 503,
            code: "SOURCE_ARCHIVE_STAGE_UNAVAILABLE",
            detail: "source archive staging file could not be rewound",
        });
    }
    read_http::execute_timeout(
        state,
        Access::Write,
        Duration::from_secs(600),
        move |api, now| {
            let service = service(api)?;
            service.publish(
                &project,
                staged.as_file_mut(),
                &digest,
                &format!("{now:.6}"),
            )
        },
    )
    .await
}

async fn read(
    State(state): State<Arc<crate::AppState>>,
    Path((project, source_id)): Path<(String, String)>,
) -> Response {
    read_http::execute(state, Access::Read, move |api, _| {
        service(api)?.read(&project, &source_id)
    })
    .await
}

async fn lookup(
    State(state): State<Arc<crate::AppState>>,
    Path((project, digest)): Path<(String, String)>,
) -> Response {
    read_http::execute(state, Access::Read, move |api, _| {
        service(api)?.lookup(&project, &digest)
    })
    .await
}
