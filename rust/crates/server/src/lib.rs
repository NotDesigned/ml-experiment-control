//! Isolated HTTP boundary preview. No production workspace or provider access.
pub mod backends;
pub mod background;
pub mod checkpoints;
pub mod cloud_runs;
pub mod deletion;
pub mod experiments;
pub mod index;
pub mod operations;
pub mod projects;
pub mod read_api;
pub mod read_config;
mod read_http;
pub mod read_model;
pub mod records;
pub mod sources;
pub mod storage;
pub mod submissions;
pub mod tracking;
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
    data_slots: Arc<tokio::sync::Semaphore>,
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

fn health_document(
    projects: usize,
    workspace_id: &str,
    reads: bool,
    lifecycle: bool,
    sources: Option<&sources::Service>,
    submissions: Option<&submissions::Service>,
) -> serde_json::Value {
    let mut capabilities = vec!["bearer-auth.v1", "deployment-readiness.v1"];
    if reads {
        capabilities.extend(["metric-records.v1", "result-collection-read.v1"]);
    }
    if lifecycle {
        capabilities.push("project-lifecycle.v1");
    }
    if sources.is_some() {
        capabilities.push("source-archive.v1");
    }
    if submissions.is_some() {
        capabilities.push("native-submissions.v1");
        capabilities.push("native-slurm-submissions.v1");
        capabilities.push("native-worker-observation.v1");
        capabilities.push("native-checkpoints.v1");
        capabilities.push("native-artifacts.v1");
        capabilities.push("native-cancellation.v1");
        capabilities.push("native-metric-history.v1");
        capabilities.push("native-tracking-status.v1");
        if submissions.is_some_and(submissions::Service::can_create_runs) {
            capabilities.push("native-attempt-retry.v1");
        }
    }
    if submissions.is_some_and(submissions::Service::can_create_runs) {
        capabilities.push("native-run-creation.v1");
    }
    if submissions.is_some_and(submissions::Service::has_datasets) {
        capabilities.push("native-datasets.v1");
    }
    json!({
        "status":"ok", "server_version":env!("CARGO_PKG_VERSION"),
        "api_protocol_version":API_PROTOCOL_VERSION, "min_client_protocol_version":2,
        "openapi_path":"/api/v2/openapi.json", "capabilities":capabilities,
        "supported_capabilities":capabilities,
        "authentication":"bearer", "transport_security":"http", "projects":projects,
        "workspace_id":workspace_id, "local_reads_configured":reads, "collector_enabled":false,
        "collector_requested":false, "collector_error":null, "storage_maintenance_error":null,
        "project_write_recovery_errors":[], "project_writes":false, "source_imports":sources.is_some(),
        "source_archive_limits":sources.map(sources::Service::limits),
        "scheduler_mutations":submissions.is_some_and(submissions::Service::enabled), "run_writes":submissions.is_some_and(submissions::Service::can_create_runs), "local_evidence_rebuild":false, "telemetry_enabled":false,
        "deployment":{"api":"READY", "training":"INCOMPLETE", "migration":"IN_PROGRESS"}
    })
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    if state.reads.is_none() {
        return Json(health_document(
            0,
            "rust-isolated-preview",
            false,
            false,
            None,
            None,
        ))
        .into_response();
    }
    read_http::execute(state, read_http::Access::Read, |api, _| {
        let mut document = health_document(
            api.active_project_count()?,
            api.workspace_id(),
            true,
            api.lifecycle().is_some(),
            api.sources().as_deref(),
            api.submissions().as_deref(),
        );
        if let Some(cloud) = api.cloud() {
            for field in ["capabilities", "supported_capabilities"] {
                let values = document[field].as_array_mut().expect("capabilities array");
                for capability in [
                    "native-cloud-submissions.v1",
                    "native-submissions.v1",
                    "native-cancellation.v1",
                    "native-scheduler-observation.v1",
                ] {
                    if !values.iter().any(|v| v == capability) {
                        values.push(json!(capability));
                    }
                }
                if cloud.can_create_runs() && !values.iter().any(|v| v == "native-run-creation.v1")
                {
                    values.push(json!("native-run-creation.v1"));
                }
            }
            document["cloud_scheduler_mutations"] = json!(cloud.enabled());
            document["cloud_run_creation"] = json!(cloud.can_create_runs());
            document["scheduler_mutations"] =
                json!(cloud.enabled() || document["scheduler_mutations"] == true);
            document["run_writes"] =
                json!(cloud.can_create_runs() || document["run_writes"] == true);
        }
        Ok(document)
    })
    .await
}
async fn readiness(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let submission = state.reads.as_ref().and_then(|api| api.submissions());
    let cloud = state.reads.as_ref().and_then(|api| api.cloud());
    Json(
        json!({"api":"READY", "training":"INCOMPLETE", "migration":"IN_PROGRESS",
        "scheduler_mutations":submission.as_ref().is_some_and(|service|service.enabled())||cloud.as_ref().is_some_and(|service|service.enabled()),"providers_configured":submission.is_some()||cloud.is_some()}),
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
    let mut document = json!({"openapi":"3.1.0", "info":{"title":"ML-Expd Rust isolated preview", "version":env!("CARGO_PKG_VERSION"),
    "description":"Incomplete native migration. Experiment GETs only observe approved isolated evidence. Explicit project lifecycle POSTs update their dedicated registry ledger. Separately configured native Slurm submissions can prepare, authorize, execute once and reconcile the original owned job. The optional native Slurm artifact path explicitly collects backend results, streams exact archives and accepts verified ACKs; original outputs and checkpoints remain. Legacy collection JSON alone never establishes download availability."},
    "components":{"securitySchemes":{"bearer":{"type":"http","scheme":"bearer"}},"parameters":{
        "protocol":{"in":"header","name":CLIENT_PROTOCOL_HEADER,"required":true,"schema":{"type":"integer","const":2}},
        "project":{"in":"path","name":"project","required":true,"schema":{"type":"string"}},
        "run":{"in":"path","name":"run","required":true,"schema":{"type":"string"}},
        "attempt":{"in":"path","name":"attempt","required":true,"schema":{"type":"string","pattern":"^attempt-[0-9]{3,}$"}},
        "submission":{"in":"path","name":"identity","required":true,"schema":{"type":"string","pattern":"^action-[a-f0-9]{16}$"}},
        "identity":{"in":"path","name":"identity","required":true,"schema":{"type":"string"},"description":"Exact Run::Attempt identity, e.g. train::attempt-001"}
    }},"security":[{"bearer":[]}],"paths":{
        "/api/health":{"get":{"description":"Authenticated bootstrap with active project count; protocol header may be omitted only here","responses":{"200":{"description":"Enabled capabilities and incomplete training readiness"},"401":{"description":"Authentication required"},"409":{"description":"Project lifecycle evidence is unavailable"},"429":{"description":"Local observation concurrency is full"},"504":{"description":"Observation deadline elapsed; underlying I/O can continue"}}}},
        "/api/readiness":operation("Preview readiness; training remains incomplete",vec![parameter("protocol")]),
        "/api/observer":operation("Local background observer progress and sanitized failure; never performs a provider query",vec![parameter("protocol")]),
        "/api/runs/{project}/{run}/tracking":operation("Frozen W&B destination and local publication cursors; never queries cloud or creates a binding",run()),
        "/api/v2/openapi.json":operation("This static preview contract",vec![parameter("protocol")]),
        "/api/projects":operation("List only configured projects, without filesystem paths or private configuration",vec![parameter("protocol")]),
        "/api/submissions/{identity}":operation("Read one committed Submission from existing isolated Action storage; never repairs or submits",vec![parameter("protocol"),parameter("submission")]),
        "/api/runs/{project}/{run}":operation("Current explicitly bound Attempt and independent evidence layers; missing scheduler evidence is unknown",run()),
        "/api/runs/{project}/{run}/attempts":operation("Verified Attempt manifests; no latest-attempt guessing",run()),
        "/api/runs/{project}/{run}/attempts/{attempt}/collection":operation("Read process completion separately from unobserved artifact publication; never starts collection",collection),
        "/api/attempts/{project}/{identity}/show":operation("Exact Attempt metadata and evidence provenance",exact()),
        "/api/attempts/{project}/{identity}/logs":operation("Exact Attempt bounded captured log tails with source diagnostics",logs),
        "/api/attempts/{project}/{identity}/metrics":operation("Exact Attempt scientific metrics; truncated raw history is PARTIAL, oversized canonical journal is refused",metrics)
    }});
    let revision = json!({"type":["integer","null"],"minimum":0,"maximum":u64::MAX,"description":"Optional exact registry revision for compare-and-set; null omits the precondition"});
    let lifecycle_body = json!({"type":"object","additionalProperties":false,"properties":{"reason":{"type":"string","maxLength":4000,"default":""},"expected_revision":revision}});
    let source = json!({"type":"object","additionalProperties":false,"required":["manifest_path"],"properties":{"kind":{"type":"string","enum":["daemon_path"],"default":"daemon_path"},"manifest_path":{"type":"string","minLength":1,"maxLength":4096}}});
    let register_body = json!({"type":"object","additionalProperties":false,"properties":{
        "project_file":{"type":["string","null"],"minLength":1,"maxLength":4096},
        "source":{"anyOf":[source,{"type":"null"}]},"expected_revision":revision},
        "oneOf":[{"required":["project_file"],"properties":{"project_file":{"type":"string"}}},{"required":["source"],"properties":{"source":{"type":"object"}}}]});
    let post = |description: &str, body: serde_json::Value, parameters: Vec<serde_json::Value>| {
        json!({"post":{
        "description":description,"parameters":parameters,"requestBody":{"required":true,"description":"Strict JSON mapping; at most 64 KiB","content":{"application/json":{"schema":body}}},
        "responses":{"200":{"description":"Durable operation result with the original identity and state"},"401":{"description":"Bearer authentication required"},"403":{"description":"Operation is disabled by the operator"},"404":{"description":"Exact configured identity is unavailable"},"409":{"description":"Operation state conflict, binding mismatch or invalid durable evidence"},"413":{"description":"Request exceeds the 64 KiB body limit"},"422":{"description":"Invalid or unsupported request"},"426":{"description":"Client protocol 2 required"},"429":{"description":"Local operation concurrency is full; no operation admitted"},"500":{"description":"Operation may have changed state, including after timeout or response failure; query the original identity before deciding whether to retry"},"501":{"description":"Operation was not explicitly configured"},"503":{"description":"Operation lock or transition context is unavailable"}}}})
    };
    let paths = document["paths"]
        .as_object_mut()
        .expect("static OpenAPI paths object");
    let checkpoint_parameters = vec![
        parameter("protocol"),
        parameter("project"),
        parameter("run"),
        json!({"name":"attempt","in":"path","required":true,"schema":{"type":"string","pattern":"^attempt-[0-9]{3,}$"}}),
    ];
    paths.insert(
        "/api/runs/{project}/{run}/attempts/{attempt}/checkpoints".into(),
        operation(
            "Read registered backend checkpoint metadata; no remote file access",
            checkpoint_parameters.clone(),
        ),
    );
    let mut checkpoint = checkpoint_parameters;
    checkpoint.push(json!({"name":"checkpoint","in":"path","required":true,"schema":{"type":"string","pattern":"^checkpoint[.][0-9a-f]{64}$"}}));
    paths.insert(
        "/api/runs/{project}/{run}/attempts/{attempt}/checkpoints/{checkpoint}".into(),
        operation(
            "Read one exact registered checkpoint and provenance",
            checkpoint,
        ),
    );
    paths.insert(
        "/api/submissions/{identity}/artifacts".into(),
        operation(
            "Read exact native bundle and retention metadata; no backend payload read",
            vec![parameter("protocol"), parameter("submission")],
        ),
    );
    let mut stream = operation(
        "Stream the exact backend tar with bounded memory and no API disk spool; this GET sends no ACK",
        vec![
            parameter("protocol"),
            parameter("submission"),
            json!({"in":"path","name":"sha256","required":true,"schema":{"type":"string","pattern":"^[a-f0-9]{64}$"}}),
        ],
    );
    stream["get"]["responses"]["200"] = json!({"description":"Exact immutable archive; a truncated transfer is not verified","content":{"application/x-tar":{"schema":{"type":"string","format":"binary"}}}});
    stream["get"]["responses"]["410"] = json!({"description":"Generated archive released after verified ACK; original outputs and checkpoints remain"});
    stream["get"]["responses"]["503"] = json!({"description":"Backend stream unavailable or archive release still pending; no acknowledgement is inferred"});
    paths.insert(
        "/api/submissions/{identity}/artifacts/{sha256}".into(),
        stream,
    );
    paths.insert("/api/project-lifecycle".into(),operation("Observe the existing lifecycle registry and audit tail without repairing its pending journal",vec![parameter("protocol")]));
    paths.insert("/api/project-lifecycle/register".into(),post("Register one configured approved daemon manifest path; this does not discover arbitrary client paths or import a source tree",register_body,vec![parameter("protocol")]));
    paths.insert(
        "/api/project-lifecycle/unregister-all".into(),
        post(
            "Remove all daemon tracking registrations; research files, artifacts and jobs remain",
            lifecycle_body.clone(),
            vec![parameter("protocol")],
        ),
    );
    for action in ["pause", "resume", "archive", "restore", "unregister"] {
        paths.insert(
            format!("/api/project-lifecycle/{{project}}/{action}"),
            post(
                "Change daemon project tracking only; archive requires a nonempty reason",
                lifecycle_body.clone(),
                vec![parameter("protocol"), parameter("project")],
            ),
        );
    }
    paths.insert("/api/runs/{project}/{run}/submission".into(),operation("Read the original native preparation slot after response loss; never creates an identity",run()));
    let data_parameters = || {
        vec![
            parameter("protocol"),
            parameter("project"),
            json!({"in":"path","name":"executor","required":true,"schema":{"type":"string"}}),
        ]
    };
    let data_asset_parameters = || {
        let mut p = data_parameters();
        p.push(json!({"in":"path","name":"asset","required":true,"schema":{"type":"string","pattern":"^asset[.][a-f0-9]{64}$"}}));
        p
    };
    paths.insert("/api/projects/{project}/datasets/{executor}".into(),post("Create or reconcile the same immutable dataset on an explicitly configured gateway; no scheduler operation",
        json!({"type":"object","additionalProperties":false,"required":["sha256","archive_bytes","files"],"properties":{"sha256":{"type":"string"},"archive_bytes":{"type":"integer","minimum":1},"files":{"type":"array","items":{"type":"object","additionalProperties":false,"required":["path","bytes","sha256"],"properties":{"path":{"type":"string"},"bytes":{"type":"integer","minimum":0},"sha256":{"type":"string"}}}}}}),data_parameters()));
    paths.insert(
        "/api/projects/{project}/datasets/{executor}/{asset}".into(),
        operation(
            "Read cached durable dataset receipts; does not reverify backend bytes",
            data_asset_parameters(),
        ),
    );
    paths.insert("/api/projects/{project}/datasets/{executor}/{asset}/complete".into(),post("Verify and publish the existing backend dataset, then remove owned parts and archive; send an empty request body",json!({"type":"object","maxProperties":0}),data_asset_parameters()));
    let mut data_part = post(
        "Persist one exact zero-based part on the gateway; fixed 16 MiB maximum; no API disk spool",
        json!({"type":"string","format":"binary","maxLength":16777216}),
        {
            let mut p = data_asset_parameters();
            p.push(json!({"in":"path","name":"number","required":true,"schema":{"type":"integer","minimum":0,"maximum":65535}}));
            p.push(json!({"in":"path","name":"sha256","required":true,"schema":{"type":"string"}}));
            p
        },
    );
    let mut part_operation = data_part
        .as_object_mut()
        .expect("operation object")
        .remove("post")
        .expect("post operation");
    part_operation["requestBody"]["content"] = json!({"application/octet-stream":{"schema":{"type":"string","format":"binary","maxLength":16777216}}});
    data_part["put"] = part_operation;
    paths.insert(
        "/api/projects/{project}/datasets/{executor}/{asset}/parts/{number}/{sha256}".into(),
        data_part,
    );
    paths.insert("/api/projects/{project}/runs".into(),post("Create one immutable standalone Run and first Attempt in an explicitly approved canonical root; never submits a scheduler job",
        json!({"type":"object","additionalProperties":false,"required":["run_id","runtime_id","executor","command","resources","walltime_seconds","seed"],"properties":{
            "run_id":{"type":"string"},"runtime_id":{"type":"string"},"executor":{"type":"string"},"command":{"type":"array","minItems":1,"maxItems":128,"items":{"type":"string"}},
            "resources":{"type":"object","additionalProperties":false,"required":["gpus","memory_gb"],"properties":{"gpus":{"type":"integer","minimum":1},"cpus":{"type":"integer","minimum":1,"default":8},"nodes":{"type":"integer","const":1,"default":1},"memory_gb":{"type":"integer","minimum":1}}},
            "walltime_seconds":{"type":"integer","minimum":1,"maximum":3600000},"seed":{"type":"integer"},"wandb":{"type":"object","additionalProperties":false,"properties":{"enabled":{"type":"boolean","default":true},"preparation_enabled":{"type":"boolean","default":false},"entity":{"type":["string","null"]},"project":{"type":["string","null"]}}},"data":{"type":"array","maxItems":32,"items":{"type":"object","additionalProperties":false,"required":["name","asset_id"],"properties":{"name":{"type":"string"},"asset_id":{"type":"string"}}}},"parameters":{"type":"object"},"evaluation":{"type":"object"},"metrics_schema":{"type":"object"},"artifacts":{"type":"object","additionalProperties":false,"properties":{"required_files":{"type":"array","items":{"type":"string"},"maxItems":20000}}}}}),vec![parameter("protocol"),parameter("project")]));
    paths.insert("/api/runs/{project}/{run}/submissions".into(),post("Prepare the first authored Run/Attempt with trusted source and materialized-image receipts; never submits a scheduler job",json!({"type":"object","additionalProperties":false,"required":["max_gpu_hours"],"properties":{"max_gpu_hours":{"type":"number","exclusiveMinimum":0},"reason":{"type":"string","maxLength":4000,"default":""}}}),run()));
    for (name, description, body) in [
        (
            "history",
            "Ingest one bounded metrics page into the canonical local journal; persist events before advancing the exact Attempt cursor; never submits",
            json!({"type":"object","additionalProperties":false}),
        ),
        (
            "retry",
            "Append a new Attempt from an explicit parent checkpoint after exact terminal scheduler evidence; reserves cumulative GPU budget and never submits",
            json!({"type":"object","additionalProperties":false,"required":["confirmation","resume_from","max_gpu_hours"],"properties":{"confirmation":{"type":"string","pattern":"^RETRY action-[a-f0-9]{16}$"},"max_gpu_hours":{"type":"number","exclusiveMinimum":0},"reason":{"type":"string","maxLength":4000},"resume_from":{"type":"object","additionalProperties":false,"required":["run_id","attempt_id","checkpoint_id"],"properties":{"run_id":{"type":"string"},"attempt_id":{"type":"string"},"checkpoint_id":{"type":"string"}}}}}),
        ),
        (
            "cancel",
            "Persist an exact owned-job cancellation intent and reconcile the helper cancel outbox; terminal observation is required for termination verification",
            json!({"type":"object","additionalProperties":false,"required":["confirmation"],"properties":{"confirmation":{"type":"string","pattern":"^CANCEL action-[a-f0-9]{16}$"}}}),
        ),
        (
            "authorize",
            "Authorize the original immutable preparation without submitting",
            json!({"type":"object","additionalProperties":false,"properties":{"reason":{"type":"string","maxLength":4000,"default":""}}}),
        ),
        (
            "execute",
            "Execute a previously authorized native Slurm Action once; response loss requires same-identity reconciliation",
            json!({"type":"object","additionalProperties":false,"required":["confirmation"],"properties":{"confirmation":{"type":"string","pattern":"^EXECUTE action-[a-f0-9]{16}$"}}}),
        ),
        (
            "reconcile",
            "Inspect the original helper ledger and owned scheduler evidence; never submits or replays",
            json!({"type":"object","additionalProperties":false}),
        ),
        (
            "refresh",
            "Read bounded exact-worker logs, metrics and process evidence into the local observation cache; never changes a Submission or provider job",
            json!({"type":"object","additionalProperties":false}),
        ),
        (
            "collect",
            "Collect completed process outputs on backend shared storage; never starts a training job",
            json!({"type":"object","additionalProperties":false}),
        ),
        (
            "acknowledge",
            "Acknowledge an exact fully verified download; optionally release only the generated backend archive",
            json!({"type":"object","required":["proof","release"],"properties":{"proof":{"type":"object"},"release":{"type":"boolean"}},"additionalProperties":false}),
        ),
        (
            "checkpoint",
            "Verify and seal the named complete generation on backend storage and register metadata only; never resubmits training",
            json!({"type":"object","additionalProperties":false}),
        ),
    ] {
        let mut operation = post(
            description,
            body,
            vec![parameter("protocol"), parameter("submission")],
        );
        operation["post"]["requestBody"]["description"] =
            json!("Strict JSON mapping; at most 8 MiB including a complete download proof");
        operation["post"]["responses"]["413"]["description"] =
            json!("Request exceeds the 8 MiB body limit");
        operation["post"]["responses"]["200"]["description"] = json!(
            "Original native Submission, exact Run/Attempt, durable execution revision and retry advice"
        );
        operation["post"]["responses"]["500"]["description"] = json!(
            "Submission outcome is uncertain; query the original Run preparation or Submission before reconciliation; no automatic retry"
        );
        operation["post"]["responses"]["403"] =
            json!({"description":"Native scheduler execution is disabled by the operator"});
        if matches!(name, "refresh" | "checkpoint" | "collect" | "acknowledge") {
            operation["post"]["responses"]["200"]["description"] = json!(
                "Original Submission plus sampled worker phase and receive time; no Action or scheduler mutation"
            );
            operation["post"]["responses"]["500"]["description"] =
                json!("Observation response failed; no scheduler effect was requested");
            operation["post"]["responses"]["503"] = json!({"description":"Sample unavailable or cache response uncertain; accepted evidence retained; safe to query or repeat refresh"});
            operation["post"]["responses"]
                .as_object_mut()
                .expect("response mapping")
                .remove("403");
        }
        if name == "checkpoint" {
            operation["post"]["responses"]["200"]["description"] = json!(
                "Original Submission identity and registered checkpoint metadata; payload remains on backend storage; no scheduler operation"
            );
            operation["post"]["responses"]["503"] = json!({"description":"Ready marker or generation verification unavailable; prior registrations retained; query the checkpoint list before repeating registration"});
        }
        paths.insert(format!("/api/submissions/{{identity}}/{name}"), operation);
    }
    document
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
        data_slots: Arc::new(tokio::sync::Semaphore::new(4)),
    });
    Router::new()
        .route("/api/health", get(health))
        .route("/api/readiness", get(readiness))
        .route("/api/v2/openapi.json", get(schema))
        .merge(read_http::routes())
        .merge(projects::http::routes())
        .merge(sources::http::routes())
        .merge(submissions::http::routes())
        .merge(submissions::dataset_http::routes())
        .method_not_allowed_fallback(|| async {
            (StatusCode::METHOD_NOT_ALLOWED,Json(json!({"code":"METHOD_NOT_ALLOWED","detail":"HTTP method is unavailable for this endpoint"})))
        })
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
    fn native_contract_declares_narrow_lifecycle_posts_and_exact_read_identity() {
        let schema = super::schema_document();
        let paths = schema["paths"].as_object().unwrap();
        assert_eq!(paths.len(), 42);
        for (path, operations) in paths {
            assert_eq!(operations.as_object().unwrap().len(), 1);
            if path.contains("/datasets/") && path.contains("/parts/") {
                assert!(operations.get("put").is_some());
            } else if path.starts_with("/api/project-lifecycle/")
                || path.ends_with("/submissions")
                || path == "/api/projects/{project}/datasets/{executor}"
                || path.ends_with("/datasets/{executor}/{asset}/complete")
                || path == "/api/projects/{project}/runs"
                || path.starts_with("/api/submissions/")
                    && path != "/api/submissions/{identity}"
                    && !path.contains("/artifacts")
            {
                assert!(operations.get("post").is_some());
            } else {
                assert!(operations.get("get").is_some());
            }
        }
        assert!(paths.contains_key("/api/attempts/{project}/{identity}/metrics"));
        assert!(paths.contains_key("/api/runs/{project}/{run}/attempts/{attempt}/collection"));
        assert_eq!(
            schema["components"]["parameters"]["protocol"]["schema"]["const"],
            2
        );
        let health = super::health_document(1, "isolated", true, false, None, None);
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
        assert!(
            !health["capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "project-lifecycle.v1")
        );
        let lifecycle = super::health_document(1, "isolated", true, true, None, None);
        assert!(
            lifecycle["capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "project-lifecycle.v1")
        );
        assert_eq!(lifecycle["project_writes"], false);
        assert_eq!(lifecycle["scheduler_mutations"], false);
    }
}
