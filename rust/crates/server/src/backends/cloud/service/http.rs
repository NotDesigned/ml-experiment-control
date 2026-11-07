use crate::{
    AppState, api,
    error::Error,
    http_effect::{self, OnTimeout},
};
use axum::{
    Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Path, RawQuery, State},
    http::{HeaderMap, HeaderValue, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use ml_exp_core::cloud::exchange::{MAX_BYTES, Packet};
use std::{
    io,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
/// A worker that stops reading must not hold a transfer slot for ever.
const CODE_TRANSFER: Duration = Duration::from_secs(15 * 60);
type Cloud = crate::backends::cloud::service::Service;
type Chunks<'a> = &'a mut dyn FnMut(Vec<u8>) -> io::Result<()>;
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/worker-exchange/{project}/{run}/{attempt}/status",
            get(worker_read),
        )
        .route(
            "/api/worker-exchange/{project}/{run}/{attempt}/package",
            get(worker_read),
        )
        .route(
            "/api/worker-exchange/{project}/{run}/{attempt}/code",
            get(worker_read),
        )
        .route(
            "/api/worker-exchange/{project}/{run}/{attempt}/code-content",
            get(code_content),
        )
        .route(
            "/api/worker-exchange/{project}/{run}/{attempt}/worker",
            get(worker_binary),
        )
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
async fn worker_read(
    State(state): State<Arc<AppState>>,
    Path((project, run, attempt)): Path<(String, String, String)>,
    RawQuery(raw): RawQuery,
    headers: HeaderMap,
    uri: axum::http::Uri,
) -> Response {
    if api::no_parameters(raw.as_deref()).is_err() {
        return http_effect::failure(Error {
            status: 422,
            code: "INVALID_WORKER_EXCHANGE",
            detail: "worker reads take no query parameters",
        });
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
            code: "WORKER_AUTH_REQUIRED",
            detail: "Attempt worker authentication required",
        });
    };
    http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
        let cloud = api.cloud_only()?;
        if uri.path().ends_with("/package") {
            cloud.worker_package(&project, &run, &attempt, &token)
        } else if uri.path().ends_with("/code") {
            cloud.worker_code(&project, &run, &attempt, &token)
        } else {
            cloud.worker_diagnostic_status(&project, &run, &attempt, &token)
        }
        .map_err(Into::into)
    })
    .await
}
struct Receiver(mpsc::Receiver<io::Result<Bytes>>);
impl futures_core::Stream for Receiver {
    type Item = io::Result<Bytes>;
    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().0.poll_recv(context)
    }
}
fn send_code(
    sender: &mpsc::Sender<io::Result<Bytes>>,
    mut item: io::Result<Bytes>,
    deadline: Instant,
) -> io::Result<()> {
    loop {
        match sender.try_send(item) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(io::Error::other("the worker went away"));
            }
            Err(mpsc::error::TrySendError::Full(back)) => {
                if Instant::now() >= deadline {
                    return Err(io::Error::other("the worker stopped reading"));
                }
                item = back;
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
async fn code_content(
    state: State<Arc<AppState>>,
    path: Path<(String, String, String)>,
    raw: RawQuery,
    headers: HeaderMap,
) -> Response {
    attempt_transfer(
        state,
        path,
        raw,
        headers,
        Transfer {
            busy: "CODE_TRANSFER_BUSY",
            unavailable: "CODE_STREAM_UNAVAILABLE",
            authorize: |cloud, project, run, attempt, token| {
                cloud.worker_code_grant(project, run, attempt, token)
            },
            open: |cloud, grant| cloud.worker_code_content(grant),
            bytes: |content| content.bytes,
            send: |content, send| content.each_chunk(send).map_err(io::Error::other),
        },
    )
    .await
}
async fn worker_binary(
    state: State<Arc<AppState>>,
    path: Path<(String, String, String)>,
    raw: RawQuery,
    headers: HeaderMap,
) -> Response {
    attempt_transfer(
        state,
        path,
        raw,
        headers,
        Transfer {
            busy: "WORKER_TRANSFER_BUSY",
            unavailable: "WORKER_STREAM_UNAVAILABLE",
            authorize: |cloud, project, run, attempt, token| {
                cloud.worker_binary_grant(project, run, attempt, token)
            },
            open: |cloud, grant| cloud.worker_binary(grant),
            bytes: |content| content.bytes,
            send: |content, send| content.each_chunk(send),
        },
    )
    .await
}
/// What an Attempt may read, and how. Only `Cloud` can say what either grant allows.
struct Transfer<G, C> {
    busy: &'static str,
    unavailable: &'static str,
    authorize: fn(&Cloud, &str, &str, &str, &str) -> crate::backends::cloud::Result<G>,
    open: fn(&Cloud, &G) -> crate::backends::cloud::Result<C>,
    bytes: fn(&C) -> u64,
    send: fn(C, Chunks<'_>) -> io::Result<()>,
}
/// Streamed from files, so a large one neither sits in this server's memory nor holds one
/// of the slots ordinary requests wait for. The worker checks every byte.
async fn attempt_transfer<G: Send + 'static, C: Send + 'static>(
    State(state): State<Arc<AppState>>,
    Path((project, run, attempt)): Path<(String, String, String)>,
    RawQuery(raw): RawQuery,
    headers: HeaderMap,
    transfer: Transfer<G, C>,
) -> Response {
    let Transfer {
        busy,
        unavailable,
        authorize,
        open,
        bytes,
        send,
    } = transfer;
    if api::no_parameters(raw.as_deref()).is_err() {
        return http_effect::failure(Error {
            status: 422,
            code: "INVALID_WORKER_EXCHANGE",
            detail: "worker reads take no query parameters",
        });
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
            code: "WORKER_AUTH_REQUIRED",
            detail: "Attempt worker authentication required",
        });
    };
    let Some(api) = state.api.clone() else {
        return http_effect::failure(Error {
            status: 501,
            code: "LOCAL_SERVICES_NOT_CONFIGURED",
            detail: "local experiment services are not configured",
        });
    };
    let cloud = match api.cloud_only() {
        Ok(cloud) => cloud,
        Err(error) => return http_effect::failure(error),
    };
    // The token can only be derived from the saved Action, so anyone can make this read
    // it. It takes a slot as other worker reads do, kept until the read really ends.
    let Ok(authorizing_slot) = state.effect_slots.clone().try_acquire_owned() else {
        let mut response = http_effect::failure(Error {
            status: 429,
            code: "READ_BUSY",
            detail: "local operation concurrency limit reached",
        });
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("1"));
        return response;
    };
    let authorizing = cloud.clone();
    let task = tokio::task::spawn_blocking(move || {
        let _slot = authorizing_slot;
        authorize(&authorizing, &project, &run, &attempt, &token)
    });
    let grant = match tokio::time::timeout(Duration::from_secs(35), task).await {
        Ok(Ok(Ok(grant))) => grant,
        Ok(Ok(Err(error))) => return http_effect::failure(error.into()),
        _ => {
            return http_effect::failure(Error {
                status: 503,
                code: unavailable,
                detail: "the file could not be read yet; try again",
            });
        }
    };
    let Ok(permit) = state.data_slots.clone().try_acquire_owned() else {
        let mut response = http_effect::failure(Error {
            status: 429,
            code: busy,
            detail: "transfer concurrency limit reached; try again",
        });
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("1"));
        return response;
    };
    let task = tokio::task::spawn_blocking(move || open(&cloud, &grant));
    let content = match tokio::time::timeout(Duration::from_secs(35), task).await {
        Ok(Ok(Ok(content))) => content,
        Ok(Ok(Err(error))) => return http_effect::failure(error.into()),
        _ => {
            return http_effect::failure(Error {
                status: 503,
                code: unavailable,
                detail: "the file could not be read yet; try again",
            });
        }
    };
    let length = bytes(&content);
    let (sender, receiver) = mpsc::channel(2);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let deadline = Instant::now() + CODE_TRANSFER;
        let sent = send(content, &mut |chunk| {
            send_code(&sender, Ok(chunk.into()), deadline)
        });
        // Ending early with an error tells the worker it did not receive everything.
        if let Err(error) = sent {
            let _ = send_code(&sender, Err(io::Error::other(error)), deadline);
        }
    });
    let mut response = Body::from_stream(Receiver(receiver)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, HeaderValue::from(length));
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}
async fn artifact(
    State(state): State<Arc<AppState>>,
    Path((project, run, attempt)): Path<(String, String, String)>,
    RawQuery(raw): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let invalid = || Error {
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
            None => return http_effect::failure(invalid()),
        };
    http_effect::execute_timeout(
        state,
        OnTimeout::Effect,
        std::time::Duration::from_secs(3600),
        move |api, now| {
            let cloud = api.cloud_only()?;
            let now = time::OffsetDateTime::from_unix_timestamp(now as i64)
                .map_err(|_| invalid())?
                .format(&time::format_description::well_known::Rfc3339)
                .map_err(|_| invalid())?;
            cloud
                .receive_artifacts(&project, &run, &attempt, &token, request, &now)
                .map_err(Into::into)
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
    let invalid = || Error {
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
        return http_effect::failure(invalid());
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
        return http_effect::failure(Error {
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
        None => return http_effect::failure(invalid()),
    };
    http_effect::execute(state, OnTimeout::Effect, move |api, now| {
        let cloud = api.cloud_only()?;
        let now = time::OffsetDateTime::from_unix_timestamp(now as i64)
            .map_err(|_| invalid())?
            .format(&time::format_description::well_known::Rfc3339)
            .map_err(|_| invalid())?;
        let ack = cloud.receive(&project, &run, &attempt, &token, &packet, &now)?;
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
    let invalid = || Error {
        status: 422,
        code: "INVALID_DATA_TICKET",
        detail: "invalid exact input data ticket request",
    };
    if api::no_parameters(raw.as_deref()).is_err()
        || body.as_ref() != b"{}"
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
            code: "WORKER_AUTH_REQUIRED",
            detail: "Attempt worker authentication required",
        });
    };
    http_effect::execute(state, OnTimeout::NoEffect, move |api, _| {
        api.cloud_only()?
            .dataset_ticket(&project, &run, &attempt, &token, &asset, part)
            .map_err(Into::into)
    })
    .await
}
