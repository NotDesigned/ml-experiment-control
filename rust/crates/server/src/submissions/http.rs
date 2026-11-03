//! Narrow submission HTTP effects. GET is local-only; reconcile observes the
//! original provider identity, while only execute can submit a new scheduler job.
use super::{Context, Error, Service};
use crate::{
    AppState, read_api,
    read_http::{self, Access},
};
use axum::{
    Router,
    body::Bytes,
    extract::{
        DefaultBodyLimit, Path, RawQuery, State,
        rejection::{BytesRejection, PathRejection},
    },
    response::Response,
    routing::{get, post},
};
use ml_exp_core::{first_submission::native::Approval, operations::validate_submission_id};
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, atomic::AtomicBool};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

const MAX_BODY: usize = 8 * 1024 * 1024;
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/runs/{project}/{run}/submissions", post(prepare))
        .route("/api/runs/{project}/{run}/submission", get(existing))
        .route("/api/projects/{project}/runs", post(create))
        .route("/api/observer", get(observer))
        .route("/api/runs/{project}/{run}/tracking", get(tracking))
        .route("/api/submissions/{identity}/{operation}", post(change))
        .route(
            "/api/runs/{project}/{run}/attempts/{attempt}/checkpoints",
            get(checkpoint_list),
        )
        .route(
            "/api/runs/{project}/{run}/attempts/{attempt}/checkpoints/{checkpoint}",
            get(checkpoint_read),
        )
        .route(
            "/api/submissions/{identity}/artifacts",
            get(artifact_metadata),
        )
        .route(
            "/api/submissions/{identity}/artifacts/{sha256}",
            get(super::artifact_http::download),
        )
        .layer(DefaultBodyLimit::max(MAX_BODY))
}
async fn tracking(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run))) = path else {
        return read_http::failure(invalid());
    };
    if ml_exp_core::identity::validate_id(&project).is_err()
        || ml_exp_core::identity::validate_id(&run).is_err()
    {
        return read_http::failure(invalid());
    }
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(error);
    }
    read_http::execute(state, Access::Read, move |api, _| {
        if let Some(value) = api.historical_tracking(&project, &run)? {
            return Ok(value);
        }
        if let Some(cloud) = api.cloud() {
            if cloud
                .handles_run(&project, &run)
                .map_err(crate::cloud_runs::failure)?
            {
                return cloud
                    .tracking_status(&project, &run)
                    .map_err(crate::cloud_runs::failure);
            }
        }
        service(api)?
            .tracking_status(&project, &run)
            .map_err(failure)
    })
    .await
}
async fn observer(State(state): State<Arc<AppState>>, RawQuery(raw): RawQuery) -> Response {
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(error);
    }
    read_http::execute(state, Access::Read, move |api, _| {
        service(api)?.background_status().map_err(failure)
    })
    .await
}
async fn checkpoint_list(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run, attempt))) = path else {
        return read_http::failure(invalid());
    };
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(error);
    }
    checkpoint_get(state, project, run, attempt, None).await
}
async fn checkpoint_read(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String, String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run, attempt, id))) = path else {
        return read_http::failure(invalid());
    };
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(error);
    }
    checkpoint_get(state, project, run, attempt, Some(id)).await
}
async fn checkpoint_get(
    state: Arc<AppState>,
    project: String,
    run: String,
    attempt: String,
    id: Option<String>,
) -> Response {
    if ml_exp_core::identity::validate_id(&project).is_err()
        || ml_exp_core::identity::validate_id(&run).is_err()
        || ml_exp_core::identity::validate_attempt(&attempt).is_err()
        || id
            .as_ref()
            .is_some_and(|id| ml_exp_core::checkpoints::validate_checkpoint_id(id).is_err())
    {
        return read_http::failure(invalid());
    }
    read_http::execute(state, Access::Read, move |api, _| {
        if let Some(value) = api.historical_checkpoints(&project, &run, &attempt, id.as_deref())? {
            return Ok(value);
        }
        if let Some(cloud) = api.cloud() {
            if cloud
                .handles_run(&project, &run)
                .map_err(crate::cloud_runs::failure)?
            {
                return cloud
                    .checkpoints(&project, &run, &attempt, id.as_deref())
                    .map_err(crate::cloud_runs::failure);
            }
        }
        service(api)?
            .checkpoints(&project, &run, &attempt, id.as_deref())
            .map_err(failure)
    })
    .await
}
fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_SUBMISSION_REQUEST",
        detail: "invalid native submission identity or request",
    }
}
pub(crate) fn failure(error: Error) -> read_api::Error {
    let (status, code, detail) = match error {
        Error::Missing | Error::Action(crate::operations::Error::NotFound) => (
            404,
            "SUBMISSION_NOT_FOUND",
            "the original preparation is unavailable; no new identity was created",
        ),
        Error::ObservationConflict => (
            409,
            "WORKER_OBSERVATION_CONFLICT",
            "remote worker state or accepted metric history changed; previous evidence is retained and no provider mutation was issued",
        ),
        Error::ArtifactsUnavailable => (
            503,
            "ARTIFACTS_UNAVAILABLE",
            "backend archive cannot be collected, verified or streamed; original outputs and checkpoints are retained",
        ),
        Error::ArtifactsReleased => (
            410,
            "ARTIFACTS_RELEASED",
            "the generated archive was released after a verified download; checkpoints and original outputs are retained",
        ),
        Error::CheckpointUnavailable => (
            503,
            "CHECKPOINT_VERIFICATION_UNAVAILABLE",
            "backend checkpoint ready marker or generation could not be verified; no scheduler operation was requested",
        ),
        Error::CheckpointMissing => (
            404,
            "CHECKPOINT_NOT_FOUND",
            "the exact registered checkpoint is unavailable",
        ),
        Error::DatasetMissing => (
            404,
            "DATASET_NOT_FOUND",
            "the exact asset is not registered for this Project and backend",
        ),
        Error::DatasetUnavailable => (
            503,
            "DATASET_TRANSFER_UNCONFIRMED",
            "dataset transfer or verification is unconfirmed; resume the same asset identity; no scheduler operation was issued",
        ),
        Error::CheckpointConflict => (
            409,
            "CHECKPOINT_EVIDENCE_UNAVAILABLE",
            "checkpoint identity or immutable registry evidence cannot be validated",
        ),
        Error::ObservationUnavailable => (
            503,
            "WORKER_OBSERVATION_UNAVAILABLE",
            "worker evidence is unavailable; the last accepted sample is retained and no provider mutation was issued",
        ),
        Error::Busy => (
            503,
            "SUBMISSION_BUSY",
            "preparation is busy; query the same Run before retrying",
        ),
        Error::Disabled => (
            403,
            "SUBMISSION_DISABLED",
            "scheduler execution is disabled by the operator",
        ),
        Error::RunCreationDisabled => (
            403,
            "RUN_CREATION_DISABLED",
            "native Run creation is not enabled for this Project",
        ),
        Error::Admission(cause) => (422, "SUBMISSION_ADMISSION_REJECTED", cause.0),
        Error::Authority | Error::Conflict | Error::Configuration => (
            409,
            "SUBMISSION_EVIDENCE_UNAVAILABLE",
            "frozen submission authority changed or cannot be validated; no replay is permitted",
        ),
        Error::Action(crate::operations::Error::Decision(_))
        | Error::Runner(crate::backends::slurm::action::Error::Action(
            crate::operations::Error::Decision(_),
        )) => (
            409,
            "SUBMISSION_STATE_CONFLICT",
            "the requested transition or confirmation does not match the original Action",
        ),
        _ => (
            500,
            "SUBMISSION_OPERATION_UNCERTAIN",
            "submission state may have changed; query the same Run or Submission and reconcile before deciding any further action",
        ),
    };
    read_api::Error {
        status,
        code,
        detail,
    }
}
pub(super) fn service(api: &read_api::Api) -> Result<Arc<Service>, read_api::Error> {
    api.submissions().ok_or(read_api::Error {
        status: 501,
        code: "SUBMISSION_NOT_CONFIGURED",
        detail: "native submissions are not explicitly configured",
    })
}
fn decode<T: serde::de::DeserializeOwned>(
    body: Result<Bytes, BytesRejection>,
) -> Result<T, read_api::Error> {
    let bytes = body.map_err(|error| read_api::Error {
        status: if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
            413
        } else {
            422
        },
        code: "INVALID_SUBMISSION_BODY",
        detail: "submission body is invalid or exceeds 8 MiB",
    })?;
    let value = crate::experiments::parse_mapping(&bytes, crate::experiments::Format::Json)
        .map_err(|_| invalid())?;
    serde_json::from_value(Value::Object(value)).map_err(|_| invalid())
}
fn clock(now: f64) -> Result<String, read_api::Error> {
    if !now.is_finite() {
        return Err(invalid());
    }
    let value =
        OffsetDateTime::from_unix_timestamp_nanos((now * 1e9) as i128).map_err(|_| invalid())?;
    value.format(&Rfc3339).map_err(|_| invalid())
}
async fn create(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    RawQuery(raw): RawQuery,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let Ok(Path(project)) = path else {
        return read_http::failure(invalid());
    };
    if ml_exp_core::identity::validate_id(&project).is_err() {
        return read_http::failure(invalid());
    }
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(error);
    }
    let request: ml_exp_core::first_submission::native::creation::Request = match decode(body) {
        Ok(value) => value,
        Err(error) => return read_http::failure(error),
    };
    read_http::execute(state, Access::RunCreate, move |api, now| {
        if let Some(cloud) = api.cloud() {
            if cloud.accepts(&request.executor) {
                return cloud
                    .create_run(&project, request, &clock(now)?)
                    .map_err(crate::cloud_runs::failure);
            }
        }
        service(api)?
            .create_run(&project, request, &clock(now)?)
            .map_err(creation_failure)
    })
    .await
}
fn creation_failure(cause: Error) -> read_api::Error {
    let mut error = failure(cause);
    match error.status {
        422 => {
            error.code = "RUN_CREATION_REJECTED";
            error.detail =
                "Run request, resources, source or published runtime fail native admission";
        }
        409 => {
            error.code = "RUN_CREATION_CONFLICT";
            error.detail = "existing Run or approved authority differs or cannot be validated; existing files were not repaired";
        }
        500 => {
            error.code = "RUN_CREATION_UNCERTAIN";
            error.detail = "Run metadata may have been published; query the same Project/Run before deciding whether to repeat the request";
        }
        503 => {
            error.code = "RUN_CREATION_BUSY";
            error.detail =
                "Run metadata creation is busy; query the same Project/Run before retrying";
        }
        _ => {}
    }
    error
}
fn identities(project: &str, run: &str) -> Result<(), read_api::Error> {
    ml_exp_core::identity::validate_id(project)
        .and_then(|_| ml_exp_core::identity::validate_id(run))
        .map(|_| ())
        .map_err(|_| invalid())
}
async fn existing(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run))) = path else {
        return read_http::failure(invalid());
    };
    if let Err(error) = identities(&project, &run) {
        return read_http::failure(error);
    }
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(error);
    }
    read_http::execute(state, Access::Read, move |api, now| {
        if let Some(cloud) = api.cloud() {
            if cloud
                .handles_run(&project, &run)
                .map_err(crate::cloud_runs::failure)?
            {
                return cloud
                    .for_run(&project, &run, &clock(now)?)
                    .map_err(crate::cloud_runs::failure);
            }
        }
        service(api)?
            .for_run(&project, &run, &clock(now)?)
            .map_err(failure)
    })
    .await
}
async fn prepare(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let Ok(Path((project, run))) = path else {
        return read_http::failure(invalid());
    };
    if let Err(error) = identities(&project, &run) {
        return read_http::failure(error);
    }
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(error);
    }
    let approval: Approval = match decode(body) {
        Ok(value) => value,
        Err(error) => return read_http::failure(error),
    };
    read_http::execute(state, Access::Submission, move |api, now| {
        let now = clock(now)?;
        let expires = OffsetDateTime::parse(&now, &Rfc3339)
            .map_err(|_| invalid())?
            .checked_add(time::Duration::hours(1))
            .ok_or_else(invalid)?
            .format(&Rfc3339)
            .map_err(|_| invalid())?;
        let (_, identity) = crate::projects::transition_context().map_err(read_api::Error::from)?;
        if let Some(cloud) = api.cloud() {
            if cloud
                .handles_run(&project, &run)
                .map_err(crate::cloud_runs::failure)?
            {
                return cloud
                    .prepare(&project, &run, approval, &now, &expires, &identity)
                    .map_err(crate::cloud_runs::failure);
            }
        }
        service(api)?
            .prepare(
                &project,
                &run,
                approval,
                Context {
                    now: &now,
                    expires: &expires,
                    transition_id: &identity,
                },
            )
            .map_err(failure)
    })
    .await
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Authorize {
    #[serde(default)]
    reason: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Execute {
    confirmation: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Part {
    part: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Acknowledge {
    proof: ml_exp_core::artifacts::DownloadProof,
    release: bool,
}
enum Operation {
    DownloadTicket(u64),
    Sync(ml_exp_core::cloud::exchange::Sync),
    Authorize(String),
    Renew(ml_exp_core::renewal::Request),
    Execute(String),
    Cancel(String),
    Retry(ml_exp_core::first_submission::native::retry::Request),
    Reconcile,
    Refresh,
    History,
    Checkpoint,
    Collect,
    Acknowledge(Acknowledge),
}
async fn change(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let Ok(Path((identity, operation))) = path else {
        return read_http::failure(invalid());
    };
    if validate_submission_id(&identity).is_err() {
        return read_http::failure(invalid());
    }
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(error);
    }
    let operation = match operation.as_str() {
        "download-ticket" => decode::<Part>(body).map(|r| Operation::DownloadTicket(r.part)),
        "sync" => decode::<ml_exp_core::cloud::exchange::Sync>(body).and_then(|r| {
            r.validate(&identity).map_err(|_| invalid())?;
            Ok(Operation::Sync(r))
        }),
        "retry" => {
            decode::<ml_exp_core::first_submission::native::retry::Request>(body).and_then(|r| {
                r.validate(&identity).map_err(|_| invalid())?;
                Ok(Operation::Retry(r))
            })
        }
        "cancel" => decode::<Execute>(body).and_then(|r| {
            if r.confirmation == format!("CANCEL {identity}") {
                Ok(Operation::Cancel(r.confirmation))
            } else {
                Err(invalid())
            }
        }),
        "authorize" => decode::<Authorize>(body).and_then(|request| {
            if request.reason.chars().count() > 4000 || request.reason.chars().any(char::is_control)
            {
                Err(invalid())
            } else {
                Ok(Operation::Authorize(request.reason))
            }
        }),
        "execute" => decode::<Execute>(body).and_then(|request| {
            if request.confirmation == format!("EXECUTE {identity}") {
                Ok(Operation::Execute(request.confirmation))
            } else {
                Err(invalid())
            }
        }),
        "renew" => decode::<ml_exp_core::renewal::Request>(body).and_then(|request| {
            request.validate(&identity).map_err(|_| invalid())?;
            Ok(Operation::Renew(request))
        }),
        "reconcile" => decode::<Empty>(body).map(|_| Operation::Reconcile),
        "refresh" => decode::<Empty>(body).map(|_| Operation::Refresh),
        "history" => decode::<Empty>(body).map(|_| Operation::History),
        "checkpoint" => decode::<Empty>(body).map(|_| Operation::Checkpoint),
        "collect" => decode::<Empty>(body).map(|_| Operation::Collect),
        "acknowledge" => decode::<Acknowledge>(body).map(Operation::Acknowledge),
        _ => Err(invalid()),
    };
    let operation = match operation {
        Ok(value) => value,
        Err(error) => return read_http::failure(error),
    };
    let access = if matches!(
        &operation,
        Operation::History
            | Operation::DownloadTicket(_)
            | Operation::Refresh
            | Operation::Checkpoint
            | Operation::Collect
            | Operation::Acknowledge(_)
    ) {
        Access::Read
    } else {
        Access::Submission
    };
    read_http::execute_timeout(
        state,
        access,
        std::time::Duration::from_secs(120),
        move |api, now| {
            let now = clock(now)?;
            let (_, transition) =
                crate::projects::transition_context().map_err(read_api::Error::from)?;
            let expires=crate::renewal::expiry(&now)?;
            let context = Context {
                now: &now,
                expires: &expires,
                transition_id: &transition,
            };
            if let Some(cloud)=api.cloud(){if cloud.handles_action(&identity).map_err(crate::cloud_runs::failure)?{
                let cancelled=AtomicBool::new(false);
                return match operation {
                    Operation::Renew(request)=>cloud.renew(&identity,request,context),
                    Operation::Authorize(reason)=>cloud.authorize(&identity,&reason,&now,&transition),
                    Operation::Execute(confirmation)=>cloud.execute(&identity,&confirmation,&now,&transition,&cancelled),
                    Operation::Reconcile=>cloud.reconcile(&identity,&now,&transition,&cancelled),
                    Operation::Refresh=>cloud.refresh(&identity,&now,&cancelled),
                    Operation::Cancel(confirmation)=>cloud.cancel(&identity,&confirmation,&now,&cancelled),
                    Operation::Sync(request)=>cloud.synchronize(&identity,&request,&now,&cancelled),
                    Operation::History=>cloud.history(&identity),
                    Operation::Checkpoint=>cloud.latest_registered_checkpoint(&identity),
                    Operation::Retry(request)=>cloud.retry(&identity,request,&now,&cancelled),
                    Operation::Collect=>cloud.collect_artifacts(&identity,&now,&cancelled),
                    Operation::Acknowledge(request)=>cloud.acknowledge_artifacts(&identity,request.proof,request.release,&now),
                    Operation::DownloadTicket(part)=>cloud.download_ticket(&identity,part,&now),
                }.map_err(crate::cloud_runs::failure);
            }}
            let service = service(api)?;
            let refreshing = matches!(&operation,Operation::History | Operation::Refresh | Operation::Checkpoint | Operation::Collect | Operation::Acknowledge(_));
            match operation {
                Operation::DownloadTicket(_)=>return Err(read_api::Error{status:422,code:"DIRECT_DOWNLOAD_UNAVAILABLE",detail:"this backend streams downloads through its gateway"}),
                Operation::Sync(_)=>return Err(read_api::Error{status:501,code:"CLOUD_SYNC_NOT_CONFIGURED",detail:"CPU cloud synchronization is not configured for this submission"}),
                Operation::Renew(request)=>service.renew(&identity,request,context),
                Operation::Retry(request)=>service.retry(&identity,request,&now,&AtomicBool::new(false)),
                Operation::Cancel(confirmation)=>service.cancel(&identity,&confirmation,&now,&AtomicBool::new(false)),
                Operation::Authorize(reason) => {
                    service.authorize(&identity, "bearer-client", &reason, context)
                }
                Operation::Execute(confirmation) => {
                    service.execute(&identity, &confirmation, context, &AtomicBool::new(false))
                }
                Operation::Reconcile => service.reconcile(&identity, &now, &AtomicBool::new(false)),
                Operation::History=>service.ingest_history(&identity,&AtomicBool::new(false)),
                Operation::Refresh => service.refresh(&identity, &now, &AtomicBool::new(false)),
                Operation::Checkpoint => service.refresh_checkpoint(&identity,&now,&AtomicBool::new(false)),
                Operation::Collect=>service.collect_artifacts(&identity,&AtomicBool::new(false)),
                Operation::Acknowledge(request)=>service.acknowledge_artifacts(&identity,request.proof,request.release,&now,&AtomicBool::new(false)),
            }
            .map_err(|cause| {
                if refreshing && matches!(cause,Error::Uncertain) {
                    read_api::Error { status:503, code:"WORKER_OBSERVATION_CACHE_UNCERTAIN", detail:"the sample may have been saved; query it or repeat refresh; no scheduler effect was requested" }
                } else { failure(cause) }
            })
        },
    )
    .await
}

async fn artifact_metadata(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path(identity)) = path else {
        return read_http::failure(invalid());
    };
    if validate_submission_id(&identity).is_err() {
        return read_http::failure(invalid());
    }
    if let Err(e) = read_api::no_parameters(raw.as_deref()) {
        return read_http::failure(e);
    }
    read_http::execute(state, Access::Read, move |api, _| {
        if let Some(cloud) = api.cloud() {
            if cloud
                .handles_action(&identity)
                .map_err(crate::cloud_runs::failure)?
            {
                return cloud
                    .artifacts(&identity)
                    .map_err(crate::cloud_runs::failure);
            }
        }
        service(api)?.artifacts(&identity).map_err(failure)
    })
    .await
}
