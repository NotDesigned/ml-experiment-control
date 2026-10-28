//! The download endpoint is intentionally the one GET that reads backend bytes.
//! It neither publishes metadata nor acknowledges a successful download.
use super::http::{failure, service};
use crate::{AppState, read_api, read_http};
use axum::{
    body::{Body, Bytes},
    extract::{Path, RawQuery, State},
    http::{HeaderValue, header},
    response::{IntoResponse, Response},
};
use std::{
    io,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::mpsc;
struct Receiver(mpsc::Receiver<io::Result<Bytes>>);
impl futures_core::Stream for Receiver {
    type Item = io::Result<Bytes>;
    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().0.poll_recv(context)
    }
}
pub(crate) async fn download(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, axum::extract::rejection::PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((identity, sha256))) = path else {
        return read_http::failure(invalid());
    };
    if ml_exp_core::operations::validate_submission_id(&identity).is_err()
        || ml_exp_core::sources::validate_archive_digest(&sha256).is_err()
    {
        return read_http::failure(invalid());
    }
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(error);
    }
    let Some(api) = state.reads.clone() else {
        return read_http::failure(invalid());
    };
    let service = match service(&api) {
        Ok(s) => s,
        Err(e) => return read_http::failure(e),
    };
    let Ok(permit) = state.read_slots.clone().try_acquire_owned() else {
        return read_http::failure(read_api::Error {
            status: 429,
            code: "DOWNLOAD_BUSY",
            detail: "download concurrency limit reached",
        });
    };
    let task = tokio::task::spawn_blocking(move || {
        let metadata = service.artifacts(&identity)?;
        let bytes = metadata["state"]["bundle"]["archive"]["bytes"]
            .as_u64()
            .ok_or(super::Error::Authority)?;
        let download = service.artifact_download(&identity, &sha256)?;
        Ok::<_, super::Error>((download, bytes, permit))
    });
    let (download, bytes, permit) = match tokio::time::timeout(Duration::from_secs(35), task).await
    {
        Ok(Ok(Ok(v))) => v,
        Ok(Ok(Err(e))) => return read_http::failure(failure(e)),
        _ => {
            return read_http::failure(read_api::Error {
                status: 503,
                code: "ARTIFACT_STREAM_UNAVAILABLE",
                detail: "backend stream could not be opened; no download acknowledgement was sent",
            });
        }
    };
    let (sender, receiver) = mpsc::channel(2);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _ = download.pump(sender);
    });
    let mut response = Body::from_stream(Receiver(receiver)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-tar"),
    );
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, HeaderValue::from(bytes));
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}
fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_ARTIFACT_IDENTITY",
        detail: "invalid exact artifact download identity",
    }
}
