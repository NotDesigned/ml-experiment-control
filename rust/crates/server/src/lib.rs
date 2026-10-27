//! Isolated HTTP boundary preview. No production workspace or provider access.
pub mod backends;
pub mod deletion;
pub mod experiments;
pub mod index;
pub mod operations;
pub mod read_api;
pub mod read_config;
mod read_http;
pub mod read_model;
pub mod records;
pub mod storage;
use axum::{
    Json, Router,
    extract::{Request, State},
    http::{HeaderMap, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use ml_exp_core::{API_PROTOCOL_VERSION, CLIENT_PROTOCOL_HEADER};
use ml_exp_io::read_private;
use serde_json::json;
use std::{path::Path, sync::Arc};
use subtle::ConstantTimeEq;

// No Debug implementation: credentials must not reach logs or error formatting.
pub struct Auth {
    token: String,
}
impl Auth {
    pub fn load(path: &Path) -> Result<Self, &'static str> {
        let bytes =
            read_private(path, 4096, true).map_err(|_| "invalid private daemon token file")?;
        let token = String::from_utf8(bytes)
            .map_err(|_| "invalid daemon token encoding")?
            .trim()
            .to_owned();
        ml_exp_core::auth::validate_bearer(&token, 32, 4096)
            .map_err(|_| "daemon token needs 32..4096 header-safe ASCII characters")?;
        Ok(Self { token })
    }
}

/// Pure HTTP-boundary policy. It never reads configuration or dispatches a request.
fn access_policy(
    token: &str,
    headers: &HeaderMap,
    method: &Method,
    path: &str,
) -> Result<(), StatusCode> {
    let authorization: Vec<_> = headers.get_all("authorization").iter().collect();
    let supplied = if authorization.len() == 1 {
        authorization[0]
            .to_str()
            .ok()
            .and_then(|s| s.strip_prefix("Bearer "))
    } else {
        None
    };
    if !supplied.is_some_and(|supplied| bool::from(supplied.as_bytes().ct_eq(token.as_bytes()))) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    let protocols: Vec<_> = headers.get_all(CLIENT_PROTOCOL_HEADER).iter().collect();
    let bootstrap = protocols.is_empty() && method == Method::GET && path == "/api/health";
    if !(bootstrap || protocols.len() == 1 && protocols[0] == "2") {
        return Err(StatusCode::UPGRADE_REQUIRED);
    }
    Ok(())
}

pub(crate) struct AppState {
    auth: Auth,
    reads: Option<Arc<read_api::Api>>,
    read_slots: Arc<tokio::sync::Semaphore>,
}

async fn guard(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    if let Err(status) = access_policy(
        &state.auth.token,
        request.headers(),
        request.method(),
        request.uri().path(),
    ) {
        let message = if status == StatusCode::UNAUTHORIZED {
            "authentication required"
        } else {
            "client protocol 2 required"
        };
        return (status, Json(json!({"detail":message}))).into_response();
    }
    next.run(request).await
}

fn health_document(projects: usize, workspace_id: &str, reads: bool) -> serde_json::Value {
    let mut capabilities = vec!["bearer-auth.v1", "deployment-readiness.v1"];
    if reads {
        capabilities.extend(["metric-records.v1", "result-collection-read.v1"]);
    }
    json!({
        "status":"ok", "server_version":env!("CARGO_PKG_VERSION"),
        "api_protocol_version":API_PROTOCOL_VERSION, "min_client_protocol_version":2,
        "openapi_path":"/api/v2/openapi.json", "capabilities":capabilities,
        "supported_capabilities":capabilities,
        "authentication":"bearer", "transport_security":"http", "projects":projects,
        "workspace_id":workspace_id, "local_reads_configured":reads, "collector_enabled":false,
        "collector_requested":false, "collector_error":null, "storage_maintenance_error":null,
        "project_write_recovery_errors":[], "project_writes":false, "source_imports":false,
        "scheduler_mutations":false, "local_evidence_rebuild":false, "telemetry_enabled":false,
        "deployment":{"api":"READY", "training":"INCOMPLETE", "migration":"IN_PROGRESS"}
    })
}

async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let reads = state.reads.as_deref();
    Json(health_document(
        reads.map_or(0, read_api::Api::project_count),
        reads.map_or("rust-isolated-preview", read_api::Api::workspace_id),
        reads.is_some(),
    ))
}
async fn readiness() -> Json<serde_json::Value> {
    Json(
        json!({"api":"READY", "training":"INCOMPLETE", "migration":"IN_PROGRESS",
        "scheduler_mutations":false,"providers_configured":false}),
    )
}
fn schema_document() -> serde_json::Value {
    let parameter = |name: &str| json!({"$ref":format!("#/components/parameters/{name}")});
    let operation = |description: &str, parameters: Vec<serde_json::Value>| {
        json!({"get":{"description":description,"parameters":parameters,"responses":{
            "200":{"description":"Bounded read-only observation; diagnostics distinguish missing and unavailable evidence"},
            "401":{"description":"Bearer authentication required"},
            "404":{"description":"Exact configured identity is unavailable"},
            "409":{"description":"Evidence, deletion fence or journal identity cannot be validated"},
            "413":{"description":"Evidence history or response exceeds its explicit read budget"},
            "422":{"description":"Invalid identity or query"},
            "426":{"description":"Client protocol 2 required"},
            "429":{"description":"Four read operations are already in flight; retry later"},
            "501":{"description":"Isolated read configuration is not enabled"},
            "504":{"description":"Observation deadline elapsed; blocking filesystem I/O may still finish"}
        }}})
    };
    let exact = || {
        vec![
            parameter("protocol"),
            parameter("project"),
            parameter("identity"),
        ]
    };
    let run = || {
        vec![
            parameter("protocol"),
            parameter("project"),
            parameter("run"),
        ]
    };
    let query = |name: &str, schema: serde_json::Value, description: &str| json!({"in":"query","name":name,"schema":schema,"description":description});
    let mut logs = exact();
    logs.extend([
        query(
            "stream",
            json!({"type":"string","enum":["both","stdout","stderr"],"default":"both"}),
            "Select captured log streams",
        ),
        query(
            "lines",
            json!({"type":"integer","minimum":1,"maximum":10000,"default":80}),
            "Maximum returned lines per stream; bounded source tails report truncation",
        ),
    ]);
    let mut metrics = exact();
    metrics.extend([
        query("format", json!({"type":"string","enum":["legacy","records"],"default":"legacy"}), "Use records to preserve metric definitions, units, variants and protocol context"),
        query("max_points", json!({"type":"integer","minimum":1,"default":2000}), "Maximum downsampled output points; validation precedes filtering"),
        query("keys", json!({"type":"string"}), "Comma-separated legacy metric keys"),
        json!({"in":"query","name":"name","style":"form","explode":true,"schema":{"type":"array","items":{"type":"string"}},"description":"Repeat to select exact record metric names"})
    ]);
    let mut collection = run();
    collection.push(parameter("attempt"));
    json!({"openapi":"3.1.0", "info":{"title":"ML-Expd Rust read-only preview", "version":env!("CARGO_PKG_VERSION"),
    "description":"Incomplete native migration. Only explicitly configured isolated files are observed; no provider refresh, submission, collection, repair or acknowledgement is performed. ArtifactStore publication receipts are not migrated, so collection observations never assert verified download availability."},
    "components":{"securitySchemes":{"bearer":{"type":"http","scheme":"bearer"}},"parameters":{
        "protocol":{"in":"header","name":CLIENT_PROTOCOL_HEADER,"required":true,"schema":{"type":"integer","const":2}},
        "project":{"in":"path","name":"project","required":true,"schema":{"type":"string"}},
        "run":{"in":"path","name":"run","required":true,"schema":{"type":"string"}},
        "attempt":{"in":"path","name":"attempt","required":true,"schema":{"type":"string","pattern":"^attempt-[0-9]{3,}$"}},
        "identity":{"in":"path","name":"identity","required":true,"schema":{"type":"string"},"description":"Exact Run::Attempt identity, e.g. train::attempt-001"}
    }},"security":[{"bearer":[]}],"paths":{
        "/api/health":{"get":{"description":"Authenticated bootstrap; protocol header may be omitted only here","responses":{"200":{"description":"Capabilities and incomplete training readiness"},"401":{"description":"Authentication required"}}}},
        "/api/readiness":operation("Preview readiness; training remains incomplete",vec![parameter("protocol")]),
        "/api/v2/openapi.json":operation("This static preview contract",vec![parameter("protocol")]),
        "/api/projects":operation("List only configured projects, without filesystem paths or private configuration",vec![parameter("protocol")]),
        "/api/runs/{project}/{run}":operation("Current explicitly bound Attempt and independent evidence layers; missing scheduler evidence is unknown",run()),
        "/api/runs/{project}/{run}/attempts":operation("Verified Attempt manifests; no latest-attempt guessing",run()),
        "/api/runs/{project}/{run}/attempts/{attempt}/collection":operation("Read process completion separately from unobserved artifact publication; never starts collection",collection),
        "/api/attempts/{project}/{identity}/show":operation("Exact Attempt metadata and evidence provenance",exact()),
        "/api/attempts/{project}/{identity}/logs":operation("Exact Attempt bounded captured log tails with source diagnostics",logs),
        "/api/attempts/{project}/{identity}/metrics":operation("Exact Attempt scientific metrics; truncated raw history is PARTIAL, oversized canonical journal is refused",metrics)
    }})
}
async fn schema() -> Json<serde_json::Value> {
    Json(schema_document())
}
async fn unavailable(request: Request) -> Response {
    if request.uri().path().starts_with("/api/") {
        (StatusCode::NOT_IMPLEMENTED, Json(json!({"code":"RUST_PORT_NOT_IMPLEMENTED", "detail":"endpoint has not passed Rust migration acceptance"}))).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

pub fn app(auth: Auth, reads: Option<Arc<read_api::Api>>) -> Router {
    let state = Arc::new(AppState {
        auth,
        reads,
        read_slots: Arc::new(tokio::sync::Semaphore::new(4)),
    });
    Router::new()
        .route("/api/health", get(health))
        .route("/api/readiness", get(readiness))
        .route("/api/v2/openapi.json", get(schema))
        .merge(read_http::routes())
        .fallback(unavailable)
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;
    const TOKEN: &str = "native-test-token-000000000000000000000000";
    #[tokio::test]
    async fn boundary_requires_auth_and_protocol_and_never_claims_training() {
        let make = || {
            app(
                Auth {
                    token: TOKEN.into(),
                },
                None,
            )
        };
        let response = make()
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let response = make()
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .header("authorization", format!("Bearer {TOKEN}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let data: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(data["scheduler_mutations"], false);
        assert_eq!(data["api_protocol_version"], 2);
        for (path, protocol, status) in [
            ("/api/readiness", "", 426),
            ("/api/readiness", "1", 426),
            ("/api/readiness", "2", 200),
            ("/api/actions/execute", "2", 501),
        ] {
            let response = make()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .header("authorization", format!("Bearer {TOKEN}"))
                        .header(CLIENT_PROTOCOL_HEADER, protocol)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), status);
        }
    }
    #[tokio::test]
    async fn duplicate_auth_and_protocol_headers_are_rejected() {
        for duplicate in ["authorization", CLIENT_PROTOCOL_HEADER] {
            let mut request = Request::builder()
                .uri("/api/readiness")
                .header("authorization", format!("Bearer {TOKEN}"))
                .header(CLIENT_PROTOCOL_HEADER, "2")
                .body(Body::empty())
                .unwrap();
            let value = request.headers().get(duplicate).unwrap().clone();
            request.headers_mut().append(duplicate, value);
            let response = app(
                Auth {
                    token: TOKEN.into(),
                },
                None,
            )
            .oneshot(request)
            .await
            .unwrap();
            assert_eq!(
                response.status(),
                if duplicate == "authorization" {
                    StatusCode::UNAUTHORIZED
                } else {
                    StatusCode::UPGRADE_REQUIRED
                }
            );
        }
    }
    #[test]
    fn token_file_validation_is_safe() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("token");
        ml_exp_io::write_atomic(&path, TOKEN.as_bytes(), true).unwrap();
        assert!(Auth::load(&path).is_ok());
        ml_exp_io::write_atomic(&path, b"short", false).unwrap();
        assert!(Auth::load(&path).is_err());
        for token in [
            format!("{}\0", TOKEN),
            format!("{}密码", TOKEN),
            format!("{}\u{7}", TOKEN),
        ] {
            ml_exp_io::write_atomic(&path, token.as_bytes(), false).unwrap();
            assert!(Auth::load(&path).is_err());
        }
    }
}

#[cfg(test)]
mod schema_tests {
    #[test]
    fn native_read_contract_contains_only_gets_and_declares_exact_identity() {
        let schema = super::schema_document();
        let paths = schema["paths"].as_object().unwrap();
        assert_eq!(paths.len(), 10);
        for operations in paths.values() {
            assert_eq!(operations.as_object().unwrap().len(), 1);
            assert!(operations.get("get").is_some());
        }
        assert!(paths.contains_key("/api/attempts/{project}/{identity}/metrics"));
        assert!(paths.contains_key("/api/runs/{project}/{run}/attempts/{attempt}/collection"));
        assert_eq!(
            schema["components"]["parameters"]["protocol"]["schema"]["const"],
            2
        );
        let health = super::health_document(1, "isolated", true);
        assert!(
            health["capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "result-collection-read.v1")
        );
        assert!(
            !health["capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "result-collection.v1")
        );
    }
}
