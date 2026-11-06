//! Authenticated observation routes. Every handler runs with `OnTimeout::NoEffect`.
use crate::{
    AppState,
    http_effect::{OnTimeout, execute, failure},
    read_api::{self, Query, View},
};
use axum::{
    Router,
    extract::{Path, RawQuery, State, rejection::PathRejection},
    response::Response,
    routing::get,
};
use std::sync::Arc;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/projects", get(projects))
        .route("/api/runs/{project}/{run}", get(run))
        .route("/api/runs/{project}/{run}/attempts", get(attempts))
        .route("/api/submissions/{identity}", get(submission))
        .route(
            "/api/submissions/{identity}/progress",
            get(submission_progress),
        )
        .route(
            "/api/experiments/{project}/{run}/submissions",
            get(experiment_submissions),
        )
        .route(
            "/api/runs/{project}/{run}/attempts/{attempt}/collection",
            get(collection),
        )
        .route("/api/attempts/{project}/{identity}/{view}", get(attempt))
}
fn bad_path() -> Response {
    failure(read_api::Error {
        status: 422,
        code: "INVALID_EXPERIMENT_IDENTITY",
        detail: "invalid experiment identity",
    })
}
async fn projects(State(state): State<Arc<AppState>>, RawQuery(raw): RawQuery) -> Response {
    if let Err(e) = read_api::no_parameters(raw.as_deref()) {
        return failure(e);
    }
    execute(state, OnTimeout::NoEffect, |api, now| api.projects(now)).await
}
async fn submission(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path(identity)) = path else {
        return bad_path();
    };
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return failure(error);
    }
    execute(state, OnTimeout::NoEffect, move |api, now| {
        api.submission(&identity, now)
    })
    .await
}
async fn submission_progress(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path(identity)) = path else {
        return bad_path();
    };
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return failure(error);
    }
    execute(state, OnTimeout::NoEffect, move |api, now| {
        api.submission_progress(&identity, now)
    })
    .await
}
async fn experiment_submissions(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run))) = path else {
        return bad_path();
    };
    if let Err(error) = read_api::no_parameters(raw.as_deref()) {
        return failure(error);
    }
    execute(state, OnTimeout::NoEffect, move |api, _| {
        api.experiment_submissions(&project, &run)
    })
    .await
}
async fn run(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run))) = path else {
        return bad_path();
    };
    if let Err(e) = read_api::no_parameters(raw.as_deref()) {
        return failure(e);
    }
    execute(state, OnTimeout::NoEffect, move |api, now| {
        api.query(
            Query {
                project,
                run,
                view: View::Run,
            },
            now,
        )
    })
    .await
}
async fn attempts(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run))) = path else {
        return bad_path();
    };
    if let Err(e) = read_api::no_parameters(raw.as_deref()) {
        return failure(e);
    }
    execute(state, OnTimeout::NoEffect, move |api, now| {
        api.query(
            Query {
                project,
                run,
                view: View::Attempts,
            },
            now,
        )
    })
    .await
}
async fn collection(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, run, attempt))) = path else {
        return bad_path();
    };
    if let Err(e) = read_api::no_parameters(raw.as_deref()) {
        return failure(e);
    }
    if ml_exp_core::identity::validate_attempt(&attempt).is_err() {
        return bad_path();
    }
    execute(state, OnTimeout::NoEffect, move |api, now| {
        api.query(
            Query {
                project,
                run,
                view: View::Collection { attempt },
            },
            now,
        )
    })
    .await
}
async fn attempt(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String, String)>, PathRejection>,
    RawQuery(raw): RawQuery,
) -> Response {
    let Ok(Path((project, identity, view))) = path else {
        return bad_path();
    };
    let (run, attempt) = match read_api::exact_identity(&identity) {
        Ok(value) => value,
        Err(e) => return failure(e),
    };
    let view = match view.as_str() {
        "show" => {
            if let Err(e) = read_api::no_parameters(raw.as_deref()) {
                return failure(e);
            }
            View::Show { attempt }
        }
        "logs" => match read_api::log_query(raw.as_deref()) {
            Ok(query) => View::Logs { attempt, query },
            Err(e) => return failure(e),
        },
        "metrics" => match read_api::metric_query(raw.as_deref()) {
            Ok(query) => View::Metrics { attempt, query },
            Err(e) => return failure(e),
        },
        _ => {
            return failure(read_api::Error {
                status: 404,
                code: "UNKNOWN_ATTEMPT_VIEW",
                detail: "unknown Attempt view",
            });
        }
    };
    execute(state, OnTimeout::NoEffect, move |api, now| {
        api.query(Query { project, run, view }, now)
    })
    .await
}
