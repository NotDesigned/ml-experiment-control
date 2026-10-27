//! Isolated HTTP boundary preview. No production workspace or provider access.
pub mod deletion;
pub mod index;
pub mod operations;
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

async fn guard(State(auth): State<Arc<Auth>>, request: Request, next: Next) -> Response {
    if let Err(status) = access_policy(
        &auth.token,
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

fn health_document() -> serde_json::Value {
    json!({
        "status":"ok", "server_version":env!("CARGO_PKG_VERSION"),
        "api_protocol_version":API_PROTOCOL_VERSION, "min_client_protocol_version":2,
        "openapi_path":"/api/v2/openapi.json", "capabilities":["bearer-auth.v1"],
        "supported_capabilities":["bearer-auth.v1"],
        "authentication":"bearer", "transport_security":"http", "projects":0,
        "workspace_id":"rust-isolated-preview", "collector_enabled":false,
        "collector_requested":false, "collector_error":null, "storage_maintenance_error":null,
        "project_write_recovery_errors":[], "project_writes":false, "source_imports":false,
        "scheduler_mutations":false, "local_evidence_rebuild":false, "telemetry_enabled":false,
        "deployment":{"api":"READY", "training":"INCOMPLETE", "migration":"IN_PROGRESS"}
    })
}

async fn health() -> Json<serde_json::Value> {
    Json(health_document())
}
async fn readiness() -> Json<serde_json::Value> {
    Json(
        json!({"api":"READY", "training":"INCOMPLETE", "migration":"IN_PROGRESS",
        "scheduler_mutations":false,"providers_configured":false}),
    )
}
async fn schema() -> Json<serde_json::Value> {
    Json(
        json!({"openapi":"3.1.0", "info":{"title":"ML-Expd Rust boundary preview", "version":env!("CARGO_PKG_VERSION")},
        "components":{"securitySchemes":{"bearer":{"type":"http","scheme":"bearer"}}},
        "security":[{"bearer":[]}],
        "paths":{
            "/api/health":{"get":{"responses":{"200":{"description":"Authenticated protocol handshake"}}}},
            "/api/readiness":{"get":{"parameters":[{"in":"header","name":CLIENT_PROTOCOL_HEADER,"required":true,"schema":{"type":"integer","const":2}}],"responses":{"200":{"description":"Preview readiness; training unavailable"}}}}
        }}),
    )
}
async fn unavailable(request: Request) -> Response {
    if request.uri().path().starts_with("/api/") {
        (StatusCode::NOT_IMPLEMENTED, Json(json!({"code":"RUST_PORT_NOT_IMPLEMENTED", "detail":"endpoint has not passed Rust migration acceptance"}))).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

pub fn app(auth: Auth) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/readiness", get(readiness))
        .route("/api/v2/openapi.json", get(schema))
        .fallback(unavailable)
        .layer(middleware::from_fn_with_state(Arc::new(auth), guard))
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
            app(Auth {
                token: TOKEN.into(),
            })
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
            let response = app(Auth {
                token: TOKEN.into(),
            })
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
