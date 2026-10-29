//! Narrow lifecycle HTTP operations. Client paths are compared with approved
//! configuration by Service; requests cannot import arbitrary files or code.
use super::{Operation, Service, transition_context};
use crate::{
    AppState,
    experiments::{Format, parse_mapping},
    read_api,
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
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

const MAX_BODY: usize = 64 * 1024;

pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/project-lifecycle", get(list))
        .route("/api/project-lifecycle/register", post(register))
        .route(
            "/api/project-lifecycle/unregister-all",
            post(unregister_all),
        )
        .route("/api/project-lifecycle/{project}/{action}", post(action))
        .layer(DefaultBodyLimit::max(MAX_BODY))
}
fn invalid() -> read_api::Error {
    read_api::Error {
        status: 422,
        code: "INVALID_PROJECT_LIFECYCLE_REQUEST",
        detail: "invalid project lifecycle request",
    }
}
fn service(api: &read_api::Api) -> Result<Arc<Service>, read_api::Error> {
    api.lifecycle().ok_or(read_api::Error {
        status: 501,
        code: "PROJECT_LIFECYCLE_NOT_CONFIGURED",
        detail: "project lifecycle is not enabled in this isolated preview",
    })
}
fn request<T: serde::de::DeserializeOwned>(
    body: Result<Bytes, BytesRejection>,
) -> Result<T, read_api::Error> {
    let bytes = body.map_err(|error| read_api::Error {
        status: if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
            413
        } else {
            422
        },
        code: "INVALID_PROJECT_LIFECYCLE_BODY",
        detail: "project lifecycle body is unavailable or exceeds its limit",
    })?;
    if bytes.len() > MAX_BODY {
        return Err(read_api::Error {
            status: 413,
            code: "PROJECT_LIFECYCLE_BODY_LIMIT",
            detail: "project lifecycle body exceeds 64 KiB",
        });
    }
    let value = parse_mapping(&bytes, Format::Json).map_err(|_| invalid())?;
    serde_json::from_value(Value::Object(value)).map_err(|_| invalid())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LifecycleRequest {
    #[serde(default)]
    reason: String,
    expected_revision: Option<u64>,
}
impl LifecycleRequest {
    fn validate(&self) -> Result<(), read_api::Error> {
        if self.reason.chars().count() > 4000 {
            return Err(invalid());
        }
        Ok(())
    }
}
#[derive(Default, Deserialize)]
enum SourceKind {
    #[default]
    #[serde(rename = "daemon_path")]
    DaemonPath,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    #[serde(default)]
    kind: SourceKind,
    manifest_path: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisterRequest {
    project_file: Option<String>,
    source: Option<Source>,
    expected_revision: Option<u64>,
}
impl RegisterRequest {
    fn path(self) -> Result<(String, Option<u64>), read_api::Error> {
        let path = match (self.project_file, self.source) {
            (Some(path), None) => path,
            (
                None,
                Some(Source {
                    kind: SourceKind::DaemonPath,
                    manifest_path,
                }),
            ) => manifest_path,
            _ => return Err(invalid()),
        };
        if path.is_empty() || path.chars().count() > 4096 || path.contains('\0') {
            return Err(invalid());
        }
        Ok((path, self.expected_revision))
    }
}

async fn list(State(state): State<Arc<AppState>>, RawQuery(query): RawQuery) -> Response {
    if let Err(error) = read_api::no_parameters(query.as_deref()) {
        return read_http::failure(error);
    }
    read_http::execute(state, Access::Read, |api, _| api.project_lifecycle_read()).await
}
async fn register(
    State(state): State<Arc<AppState>>,
    RawQuery(query): RawQuery,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let parsed = (|| {
        read_api::no_parameters(query.as_deref())?;
        request::<RegisterRequest>(body)?.path()
    })();
    let (path, revision) = match parsed {
        Ok(value) => value,
        Err(error) => return read_http::failure(error),
    };
    read_http::execute(state, Access::Write, move |api, _| {
        let service = service(api)?;
        let (now, id) = transition_context().map_err(read_api::Error::from)?;
        service
            .register(&path, &now, &id, revision)
            .map_err(Into::into)
    })
    .await
}
async fn unregister_all(
    State(state): State<Arc<AppState>>,
    RawQuery(query): RawQuery,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let parsed = (|| {
        read_api::no_parameters(query.as_deref())?;
        let request = request::<LifecycleRequest>(body)?;
        request.validate()?;
        Ok::<_, read_api::Error>(request)
    })();
    let request = match parsed {
        Ok(value) => value,
        Err(error) => return read_http::failure(error),
    };
    read_http::execute(state, Access::Write, move |api, _| {
        let service = service(api)?;
        let (now, id) = transition_context().map_err(read_api::Error::from)?;
        service
            .execute(
                Operation::UnregisterAll {
                    reason: request.reason,
                },
                &now,
                &id,
                request.expected_revision,
            )
            .map_err(Into::into)
    })
    .await
}
async fn action(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
    RawQuery(query): RawQuery,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let parsed = (|| {
        let Path((project, action)) = path.map_err(|_| invalid())?;
        ml_exp_core::identity::validate_id(&project).map_err(|_| invalid())?;
        if !matches!(
            action.as_str(),
            "pause" | "resume" | "archive" | "restore" | "unregister"
        ) {
            return Err(invalid());
        }
        read_api::no_parameters(query.as_deref())?;
        let request = request::<LifecycleRequest>(body)?;
        request.validate()?;
        let operation = if action == "unregister" {
            Operation::Unregister {
                project,
                reason: request.reason,
            }
        } else {
            Operation::Action {
                project,
                action,
                reason: request.reason,
            }
        };
        Ok((operation, request.expected_revision))
    })();
    let (operation, revision) = match parsed {
        Ok(value) => value,
        Err(error) => return read_http::failure(error),
    };
    read_http::execute(state, Access::Write, move |api, _| {
        let service = service(api)?;
        let (now, id) = transition_context().map_err(read_api::Error::from)?;
        service
            .execute(operation, &now, &id, revision)
            .map_err(Into::into)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{Method, Request, StatusCode},
    };
    use serde_json::json;
    use std::fs;
    use tower::ServiceExt;
    const TOKEN: &str = "synthetic-lifecycle-http-secret-0000000000000000";
    fn body(value: Value) -> Result<Bytes, BytesRejection> {
        Ok(Bytes::from(serde_json::to_vec(&value).unwrap()))
    }
    #[test]
    fn lifecycle_body_validation_is_strict_and_never_echoes_payloads() {
        for value in [
            json!({}),
            json!({"project_file":"/approved","source":{"manifest_path":"/other"}}),
            json!({"source":{"kind":"url","manifest_path":"/private"}}),
            json!({"project_file":"","expected_revision":0}),
            json!({"project_file":"/private","expected_revision":true}),
            json!({"project_file":"/private","expected_revision":1.0}),
            json!({"project_file":"/private","token":"must-never-echo"}),
        ] {
            let result = request::<RegisterRequest>(body(value)).and_then(RegisterRequest::path);
            assert_eq!(
                result.err().unwrap().code,
                "INVALID_PROJECT_LIFECYCLE_REQUEST"
            );
        }
        for value in [
            json!({"project_file":"/approved","expected_revision":0}),
            json!({"source":{"kind":"daemon_path","manifest_path":"/approved"}}),
            json!({"source":{"manifest_path":"/approved"}}),
        ] {
            assert_eq!(
                request::<RegisterRequest>(body(value))
                    .unwrap()
                    .path()
                    .unwrap()
                    .0,
                "/approved"
            );
        }
        assert!(
            request::<LifecycleRequest>(body(json!({"reason":"x".repeat(4001)})))
                .unwrap()
                .validate()
                .is_err()
        );
        assert!(
            request::<LifecycleRequest>(body(
                json!({"reason":"汉".repeat(4000),"expected_revision":1})
            ))
            .unwrap()
            .validate()
            .is_ok()
        );
        assert!(
            request::<LifecycleRequest>(Ok(Bytes::from_static(
                b"{\"reason\":\"one\",\"reason\":\"two\"}"
            )))
            .is_err()
        );
        assert!(request::<LifecycleRequest>(Ok(Bytes::from(vec![b'x'; MAX_BODY + 1]))).is_err());
    }

    struct Fixture {
        temp: tempfile::TempDir,
        service: Arc<Service>,
        router: Router,
        project: String,
    }
    impl Fixture {
        fn new() -> Self {
            let temp = tempfile::tempdir().unwrap();
            let registry = temp.path().join("registry");
            crate::read_config::initialize(&registry).unwrap();
            let project = temp.path().join("demo/research_project.yaml");
            fs::create_dir_all(temp.path().join("demo/runs")).unwrap();
            ml_exp_io::write_atomic(&project,&serde_json::to_vec(&json!({"schema_version":1,"project":"demo","title":"Synthetic lifecycle","run_roots":["runs"],"campaigns":[]})).unwrap(),true).unwrap();
            let settings=crate::read_config::Settings::from_json(&serde_json::to_vec(&json!({"schema_version":1,"registry_root":registry,"projects":[{"project":"demo","project_file":project,"run_roots":[temp.path().join("demo/runs")]}]})).unwrap()).unwrap();
            Service::initialize(&settings, "2026-10-27T00:00:00Z", &"1".repeat(32)).unwrap();
            let service = Arc::new(Service::open(&settings).unwrap());
            let api = crate::read_api::Api::open(settings)
                .unwrap()
                .with_lifecycle(service.clone())
                .unwrap();
            let router = crate::app(
                crate::Auth {
                    token: TOKEN.into(),
                },
                Some(Arc::new(api)),
            );
            Self {
                temp,
                service,
                router,
                project: project.to_str().unwrap().into(),
            }
        }
        async fn raw(
            &self,
            method: Method,
            path: &str,
            bytes: Vec<u8>,
            auth: bool,
            protocol: bool,
        ) -> (StatusCode, Value) {
            let mut request = Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json");
            if auth {
                request = request.header("authorization", format!("Bearer {TOKEN}"));
            }
            if protocol {
                request = request.header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2");
            }
            let response = self
                .router
                .clone()
                .oneshot(request.body(Body::from(bytes)).unwrap())
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), crate::read_model::MAX_RESPONSE_BYTES)
                .await
                .unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert!(!value.to_string().contains(TOKEN));
            (status, value)
        }
        async fn post(&self, path: &str, value: Value) -> (StatusCode, Value) {
            self.raw(
                Method::POST,
                path,
                serde_json::to_vec(&value).unwrap(),
                true,
                true,
            )
            .await
        }
        async fn get(&self, path: &str) -> Value {
            let (status, value) = self.raw(Method::GET, path, Vec::new(), true, true).await;
            assert_eq!(status, StatusCode::OK, "{value}");
            value
        }
        fn files(&self) -> (Vec<u8>, Vec<u8>) {
            (
                fs::read(self.temp.path().join("registry/projects/registry.json")).unwrap(),
                fs::read(self.temp.path().join("registry/projects/events.jsonl")).unwrap(),
            )
        }
    }
    #[tokio::test]
    async fn native_lifecycle_routes_preserve_cas_authority_and_active_health_without_jobs() {
        let fixture = Fixture::new();
        let initial = fixture.files();
        let health = fixture.get("/api/health").await;
        assert_eq!(health["projects"], 1);
        assert_eq!(health["project_writes"], false);
        assert_eq!(health["scheduler_mutations"], false);
        assert!(
            health["capabilities"]
                .as_array()
                .unwrap()
                .contains(&json!("project-lifecycle.v1"))
        );
        let listed = fixture.get("/api/project-lifecycle").await;
        assert_eq!(listed["durability"]["revision"], 1);
        assert_eq!(fixture.files(), initial); // ordinary GET never repairs/writes
        let (status, value) = fixture
            .post(
                "/api/project-lifecycle/demo/pause",
                json!({"expected_revision":1}),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{value}");
        assert_eq!(value["active"], false);
        assert_eq!(fixture.get("/api/health").await["projects"], 0);
        assert_eq!(fixture.get("/api/projects").await, json!([]));
        let paused = fixture.files();
        assert_eq!(
            fixture
                .post(
                    "/api/project-lifecycle/demo/resume",
                    json!({"expected_revision":1})
                )
                .await
                .0,
            StatusCode::CONFLICT
        );
        assert_eq!(fixture.files(), paused);
        for (action, reason, revision) in [
            ("resume", "", 2),
            ("archive", "completed", 3),
            ("restore", "", 4),
            ("resume", "", 5),
            ("unregister", "retire tracking", 6),
        ] {
            let (status, value) = fixture
                .post(
                    &format!("/api/project-lifecycle/demo/{action}"),
                    json!({"reason":reason,"expected_revision":revision}),
                )
                .await;
            assert_eq!(status, StatusCode::OK, "{action}: {value}");
        }
        assert_eq!(fixture.get("/api/health").await["projects"], 0);
        let (status,value)=fixture.post("/api/project-lifecycle/register",json!({"source":{"kind":"daemon_path","manifest_path":fixture.project},"expected_revision":7})).await;
        assert_eq!(status, StatusCode::OK, "{value}");
        assert_eq!(fixture.get("/api/health").await["projects"], 1);
        assert_eq!(
            fixture
                .post(
                    "/api/project-lifecycle/unregister-all",
                    json!({"expected_revision":8})
                )
                .await
                .0,
            StatusCode::OK
        );
        assert!(fixture.service.snapshot().unwrap().projects.is_empty());
        assert_eq!(fixture.get("/api/health").await["projects"], 0);
    }
    #[tokio::test]
    async fn rejected_http_bodies_auth_queries_and_paths_do_not_mutate_or_echo() {
        let fixture = Fixture::new();
        let initial = fixture.files();
        for (auth, protocol, expected) in [
            (false, true, StatusCode::UNAUTHORIZED),
            (true, false, StatusCode::UPGRADE_REQUIRED),
        ] {
            let response = fixture
                .raw(
                    Method::POST,
                    "/api/project-lifecycle/demo/pause",
                    b"{}".to_vec(),
                    auth,
                    protocol,
                )
                .await;
            assert_eq!(response.0, expected);
        }
        for (path, value, expected) in [
            (
                "/api/project-lifecycle/demo/pause",
                json!({"reason":TOKEN,"unknown":"discard"}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/api/project-lifecycle/demo/pause?token=not-accepted",
                json!({}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/api/project-lifecycle/demo/remove-database",
                json!({}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/api/project-lifecycle/register",
                json!({"project_file":"/unapproved/private/credential-path"}),
                StatusCode::CONFLICT,
            ),
            (
                "/api/project-lifecycle/unknown/pause",
                json!({}),
                StatusCode::NOT_FOUND,
            ),
        ] {
            let (status, value) = fixture.post(path, value).await;
            assert_eq!(status, expected, "{path}: {value}");
            assert!(
                !value
                    .to_string()
                    .contains("/unapproved/private/credential-path")
            );
        }
        assert_eq!(
            fixture
                .raw(
                    Method::POST,
                    "/api/project-lifecycle/demo/pause",
                    vec![b'x'; MAX_BODY + 1],
                    true,
                    true
                )
                .await
                .0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
        assert_eq!(
            fixture
                .raw(
                    Method::POST,
                    "/api/project-lifecycle/demo/pause",
                    b"{\"reason\":\"one\",\"reason\":\"two\"}".to_vec(),
                    true,
                    true
                )
                .await
                .0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert_eq!(fixture.files(), initial);
    }
}
