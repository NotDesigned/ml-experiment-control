//! Native HTTP boundary with isolated storage and explicitly configured effects.
#[macro_use]
mod why;

pub mod administration;
mod administration_http;
pub mod api;
pub mod backends;
pub mod background;
mod capacity_http;
pub mod checkpoints;
mod data_delivery_http;
mod dataset_http;
pub mod deletion;
pub mod deployment;
pub mod error;
pub mod experiments;
mod http_effect;
mod local_http;
mod native_contract;
pub mod objects;
pub mod operations;
mod preparation_http;
pub mod projects;
pub mod read_config;
mod read_http;
pub mod read_model;
mod readiness;
pub mod records;
mod renewal;
pub mod runs;
pub mod sources;
pub mod storage;
mod submission_http;
pub mod tracking;
use axum::{
    Json, Router,
    extract::{Request, State},
    http::{HeaderMap, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
// For two sealed test files only, `tests/cloud_runs.rs` and `submission_tests.rs`,
// which name these paths. All other code names the real ones.
#[doc(hidden)]
pub use api as read_api;
#[doc(hidden)]
pub use backends::{cloud::service as cloud_runs, slurm::service as submissions};
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
        let bytes = read_private(path, 4096, true).map_err(why!(
            "invalid private daemon token file",
            "daemon token file unreadable"
        ))?;
        let token = String::from_utf8(bytes)
            .map_err(why!(
                "invalid daemon token encoding",
                "daemon token encoding invalid"
            ))?
            .trim()
            .to_owned();
        ml_exp_core::auth::validate_bearer(&token, 32, 4096).map_err(why!(
            "daemon token needs 32..4096 header-safe ASCII characters",
            "daemon token format invalid"
        ))?;
        Ok(Self { token })
    }
}

/// Pure HTTP-boundary policy. It never reads configuration or dispatches a request.
/// Exactly one Authorization header, compared in constant time.
fn authorized(token: &str, headers: &HeaderMap) -> bool {
    let authorization: Vec<_> = headers.get_all("authorization").iter().collect();
    let supplied = if authorization.len() == 1 {
        authorization[0]
            .to_str()
            .ok()
            .and_then(|s| s.strip_prefix("Bearer "))
    } else {
        None
    };
    supplied.is_some_and(|supplied| bool::from(supplied.as_bytes().ct_eq(token.as_bytes())))
}

pub(crate) struct AppState {
    auth: Auth,
    api: Option<Arc<api::Api>>,
    effect_slots: Arc<tokio::sync::Semaphore>,
    data_slots: Arc<tokio::sync::Semaphore>,
    /// Jobs fetching their worker and code, apart from dataset transfers. Each is streamed
    /// from a file through a small buffer, so even all of these cost the server little.
    download_slots: Arc<tokio::sync::Semaphore>,
}

async fn guard(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    // This route has separate, Attempt-scoped authentication in its handler.
    // A worker credential is never accepted by any ordinary API route.
    if request.method() == Method::POST
        && (request.uri().path().starts_with("/api/worker-exchange/")
            || request.uri().path().starts_with("/api/dataset-exchange/"))
        || request.method() == Method::GET && {
            let parts: Vec<_> = request.uri().path().split('/').collect();
            parts.len() == 7
                && parts[1] == "api"
                && parts[2] == "worker-exchange"
                && matches!(
                    parts[6],
                    "status" | "package" | "code" | "code-content" | "worker"
                )
        }
    {
        return next.run(request).await;
    }
    if !authorized(&state.auth.token, request.headers()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"detail":"authentication required"})),
        )
            .into_response();
    }
    next.run(request).await
}

fn health_document(
    projects: usize,
    workspace_id: &str,
    reads: bool,
    lifecycle: bool,
    sources: Option<&sources::Service>,
    submissions: Option<&backends::slurm::service::Service>,
) -> serde_json::Value {
    let mut capabilities = vec!["bearer-auth.v1", "deployment-readiness.v1"];
    if lifecycle {
        capabilities.push("project-lifecycle.v1");
    }
    if sources.is_some() {
        capabilities.push("source-archive.v1");
    }
    if submissions.is_some() {
        capabilities.push("native-submissions.v1");
        capabilities.push("native-worker-observation.v1");
        capabilities.push("execution-progress.v1");
        capabilities.push("native-checkpoints.v1");
        capabilities.push("checkpoint-record-read.v1");
        capabilities.push("native-artifacts.v1");
        capabilities.push("native-cancellation.v1");
        capabilities.push("native-metric-history.v1");
        capabilities.push("native-tracking-status.v1");
        if submissions.is_some_and(backends::slurm::service::Service::can_create_runs) {
            capabilities.push("native-attempt-retry.v1");
        }
    }
    if submissions.is_some_and(backends::slurm::service::Service::can_create_runs) {
        capabilities.push("native-run-creation.v1");
    }
    if submissions.is_some_and(backends::slurm::service::Service::has_datasets) {
        capabilities.push("native-datasets.v1");
    }
    json!({
        "status":"ok", "server_version":env!("CARGO_PKG_VERSION"),
        "openapi_path":"/api/v2/openapi.json", "capabilities":capabilities,
        "projects":projects, "workspace_id":workspace_id, "local_reads_configured":reads,
        "source_archive_limits":sources.map(sources::Service::limits),
        "scheduler_mutations":submissions.is_some_and(backends::slurm::service::Service::enabled),
        "run_writes":submissions.is_some_and(backends::slurm::service::Service::can_create_runs),
        "deployment":readiness::observe(None)
    })
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    if state.api.is_none() {
        return Json(health_document(0, "unconfigured", false, false, None, None)).into_response();
    }
    http_effect::execute(state, http_effect::OnTimeout::NoEffect, |api, _| {
        let mut document = health_document(
            api.active_project_count()?,
            api.workspace_id(),
            true,
            api.lifecycle().is_some(),
            api.sources().as_deref(),
            api.submissions().as_deref(),
        );
        document["deployment"] = readiness::observe(Some(api));
        let mut enabled = vec!["backend-catalogue.v1"];
        if api.submissions().is_some() || api.cloud().is_some() {
            enabled.push("backend-capacity.v1");
        }
        if api.administration().is_some() {
            enabled.push("native-administration.v1");
        }
        if let Some(cloud) = api.cloud() {
            enabled.extend([
                "native-experiment-preparations.v1",
                "native-cloud-submissions.v1",
                "native-submissions.v1",
                "native-cancellation.v1",
                "native-scheduler-observation.v1",
                "checkpoint-record-read.v1",
            ]);
            if cloud.can_create_runs() {
                enabled.extend(["native-run-creation.v1", "native-attempt-retry.v1"]);
            }
            if cloud.has_debug() {
                enabled.push("native-debug-machines.v1");
            }
            if cloud.has_builds() {
                enabled.push("native-runtime-builds.v1");
            }
            if cloud.has_datasets() {
                enabled.extend([
                    "native-datasets.v1",
                    "native-object-datasets.v1",
                    "native-oss-dataset-import.v1",
                    "native-dataset-direct-upload.v1",
                ]);
            }
            if cloud.has_exchange() {
                enabled.extend([
                    "native-worker-observation.v1",
                    "native-metric-history.v1",
                    "native-checkpoints.v1",
                    "native-tracking-status.v1",
                ]);
            }
            if cloud.has_collectors() {
                enabled.push("native-cloud-sync.v1");
            }
            if cloud.has_artifacts() {
                enabled.push("native-artifacts.v1");
            }
            document["cloud_scheduler_mutations"] = json!(cloud.enabled());
            document["cloud_run_creation"] = json!(cloud.can_create_runs());
            document["scheduler_mutations"] =
                json!(cloud.enabled() || document["scheduler_mutations"] == true);
            document["run_writes"] =
                json!(cloud.can_create_runs() || document["run_writes"] == true);
        }
        if api.local_enabled() {
            enabled.extend([
                "native-local-metrics.v1",
                "native-metric-history.v1",
                "native-tracking-status.v1",
            ]);
            document["local_run_creation"] = json!(true);
        }
        let capabilities = document["capabilities"]
            .as_array_mut()
            .expect("capabilities");
        for capability in enabled {
            if !capabilities.iter().any(|value| value == capability) {
                capabilities.push(json!(capability));
            }
        }
        Ok(document)
    })
    .await
}
async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    if state.api.is_none() {
        return Json(readiness::observe(None)).into_response();
    }
    http_effect::execute(state, http_effect::OnTimeout::NoEffect, |api, _| {
        Ok(readiness::observe(Some(api)))
    })
    .await
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
            "429":{"description":"Four read operations are already in flight; retry later"},
            "501":{"description":"Isolated read configuration is not enabled"},
            "504":{"description":"Observation deadline elapsed; blocking filesystem I/O may still finish"}
        }}})
    };
    let exact = || vec![parameter("project"), parameter("identity")];
    let run = || vec![parameter("project"), parameter("run")];
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
        query("max_points", json!({"type":"integer","minimum":1,"default":2000}), "Maximum downsampled output points; validation precedes filtering"),
        json!({"in":"query","name":"name","style":"form","explode":true,"schema":{"type":"array","items":{"type":"string"}},"description":"Repeat to select exact record metric names"})
    ]);
    let mut collection = run();
    collection.push(parameter("attempt"));
    let mut document = json!({"openapi":"3.1.0", "info":{"title":"ML-Expd native API", "version":env!("CARGO_PKG_VERSION"),
    "description":"Experiment GETs only observe approved evidence. Explicit project lifecycle POSTs update their dedicated registry ledger. Separately configured native submissions can prepare, execute once and reconcile the original owned job. The optional artifact path explicitly collects backend results, streams exact archives and accepts verified ACKs; original outputs and checkpoints remain. Collection JSON alone never establishes download availability."},
    "components":{"securitySchemes":{"bearer":{"type":"http","scheme":"bearer"}},"parameters":{
        "project":{"in":"path","name":"project","required":true,"schema":{"type":"string"}},
        "run":{"in":"path","name":"run","required":true,"schema":{"type":"string"}},
        "attempt":{"in":"path","name":"attempt","required":true,"schema":{"type":"string","pattern":"^attempt-[0-9]{3,}$"}},
        "submission":{"in":"path","name":"identity","required":true,"schema":{"type":"string","pattern":"^action-[a-f0-9]{16}$"}},
        "identity":{"in":"path","name":"identity","required":true,"schema":{"type":"string"},"description":"Exact Run::Attempt identity, e.g. train::attempt-001"}
    }},"security":[{"bearer":[]}],"paths":{
        "/api/health":{"get":{"description":"Authenticated health with active project count","responses":{"200":{"description":"Enabled capabilities and incomplete training readiness"},"401":{"description":"Authentication required"},"409":{"description":"Project lifecycle evidence is unavailable"},"429":{"description":"Local observation concurrency is full"},"504":{"description":"Observation deadline elapsed; underlying I/O can continue"}}}},
        "/api/readiness":operation("Readiness; training stays INCOMPLETE until acceptance evidence exists",vec![]),
        "/api/observer":operation("Local background observer progress and sanitized failure; never performs a provider query",vec![]),
        "/api/runs/{project}/{run}/tracking":operation("Frozen W&B destination and local publication cursors; never queries cloud or creates a binding",run()),
        "/api/v2/openapi.json":operation("This static contract",vec![]),
        "/api/projects":operation("List only configured projects, without filesystem paths or private configuration",vec![]),
        "/api/submissions/{identity}":operation("Read one committed Submission from existing Action storage; never repairs or submits",vec![parameter("submission")]),
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
        "responses":{"200":{"description":"Durable operation result with the original identity and state"},"401":{"description":"Bearer authentication required"},"403":{"description":"Operation is disabled by the operator"},"404":{"description":"Exact configured identity is unavailable"},"409":{"description":"Operation state conflict, binding mismatch or invalid durable evidence"},"413":{"description":"Request exceeds the 64 KiB body limit"},"422":{"description":"Invalid or unsupported request"},"429":{"description":"Local operation concurrency is full; no operation admitted"},"500":{"description":"Operation may have changed state, including after timeout or response failure; query the original identity before deciding whether to retry"},"501":{"description":"Operation was not explicitly configured"},"503":{"description":"Operation lock or transition context is unavailable"}}}})
    };
    let paths = document["paths"]
        .as_object_mut()
        .expect("static OpenAPI paths object");
    let mut register_local = post(
        "Register an explicit local execution; identical definition reuses the Run, no process or scheduler effects",
        json!({"type":"object","additionalProperties":false,"required":["run_id","execution_id"],"properties":{
            "run_id":{"type":"string"},"execution_id":{"type":"string","pattern":"^[0-9a-f]{32}$"},
            "metrics_schema":{"type":"object"},"evaluation":{"type":"object"},"parameters":{"type":"object"},
            "seed":{"type":"integer","minimum":0},"wandb":{"type":["object","null"]}}}),
        vec![parameter("project")],
    );
    register_local["post"]["requestBody"]["description"] =
        json!("Strict local definition, normalized at most 64 KiB; HTTP body at most 128 KiB");
    paths.insert("/api/projects/{project}/local-runs".into(), register_local);
    paths.insert(
        "/api/local-runs/{project}/{run}".into(),
        operation(
            "Read durable local cursor and W&B progress, process liveness remains unknown",
            run(),
        ),
    );
    let mut local_metrics = post(
        "Append the next canonical JSONL packet; exact replay returns the original receipt without duplicate events",
        json!({"type":"object","additionalProperties":false,"required":["execution_id","offset","previous_chain_sha256","bytes"],"properties":{
            "execution_id":{"type":"string","pattern":"^[0-9a-f]{32}$"},"offset":{"type":"integer","minimum":0},
            "previous_chain_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},
            "bytes":{"type":"array","maxItems":65536,"items":{"type":"integer","minimum":0,"maximum":255}},
            "finish":{"enum":[null,"SUCCEEDED","FAILED","CANCELLED"]}}}),
        run(),
    );
    local_metrics["post"]["requestBody"]["description"] =
        json!("At most 64 KiB of raw JSONL input; encoded HTTP body at most 512 KiB");
    local_metrics["post"]["responses"]["413"] =
        json!({"description":"Encoded request body exceeds 512 KiB"});
    paths.insert(
        "/api/local-runs/{project}/{run}/metrics".into(),
        local_metrics,
    );

    let checkpoint_parameters = vec![
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
            vec![parameter("submission")],
        ),
    );
    let mut stream = operation(
        "Stream the exact backend tar with bounded memory and no API disk spool; this GET sends no ACK",
        vec![
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
    paths.insert("/api/project-lifecycle".into(),operation("Observe the existing lifecycle registry and audit tail without repairing its pending journal",vec![]));
    paths.insert("/api/project-lifecycle/register".into(),post("Register one configured approved daemon manifest path; this does not discover arbitrary client paths or import a source tree",register_body,vec![]));
    paths.insert(
        "/api/project-lifecycle/unregister-all".into(),
        post(
            "Remove all daemon tracking registrations; research files, artifacts and jobs remain",
            lifecycle_body.clone(),
            vec![],
        ),
    );
    for action in ["pause", "resume", "archive", "restore", "unregister"] {
        paths.insert(
            format!("/api/project-lifecycle/{{project}}/{action}"),
            post(
                "Change daemon project tracking only; archive requires a nonempty reason",
                lifecycle_body.clone(),
                vec![parameter("project")],
            ),
        );
    }
    paths.insert("/api/runs/{project}/{run}/submission".into(),operation("Read the original native preparation slot after response loss; never creates an identity",run()));
    let data_parameters = || {
        vec![
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
    paths.insert("/api/projects/{project}/datasets/{executor}/import-oss".into(),post("Verify an existing approved OSS tar and import through bounded immutable parts; no scheduler operation or local disk spool",
        json!({"type":"object","additionalProperties":false,"required":["object_key","sha256","archive_bytes"],"properties":{"object_key":{"type":"string","maxLength":1024},"sha256":{"type":"string","pattern":"^[a-f0-9]{64}$"},"archive_bytes":{"type":"integer","minimum":1}}}),data_parameters()));
    let mut data_part = post(
        "Persist one exact zero-based part on the gateway; fixed 16 MiB maximum; no API disk spool",
        json!({"type":"string","format":"binary","maxLength":ml_exp_core::multipart::PART_BYTES}),
        {
            let mut p = data_asset_parameters();
            p.push(json!({"in":"path","name":"number","required":true,"schema":{"type":"integer","minimum":0,"maximum":ml_exp_core::multipart::MAX_PARTS - 1}}));
            p.push(json!({"in":"path","name":"sha256","required":true,"schema":{"type":"string"}}));
            p
        },
    );
    let mut part_operation = data_part
        .as_object_mut()
        .expect("operation object")
        .remove("post")
        .expect("post operation");
    part_operation["requestBody"]["content"] = json!({"application/octet-stream":{"schema":{"type":"string","format":"binary","maxLength":ml_exp_core::multipart::PART_BYTES}}});
    data_part["put"] = part_operation;
    paths.insert(
        "/api/projects/{project}/datasets/{executor}/{asset}/parts/{number}/{sha256}".into(),
        data_part,
    );
    paths.insert("/api/projects/{project}/runs".into(),post("Create one immutable standalone Run and first Attempt in an explicitly approved canonical root; never submits a scheduler job",
        json!({"type":"object","additionalProperties":false,"required":["run_id","runtime_id","executor","command","resources","walltime_seconds","seed"],"properties":{
            "run_id":{"type":"string"},"runtime_id":{"type":"string"},"executor":{"type":"string"},"command":{"type":"array","minItems":1,"maxItems":128,"items":{"type":"string"}},
            "resources":{"type":"object","additionalProperties":false,"required":["gpus","memory_gb"],"properties":{"gpus":{"type":"integer","minimum":1},"cpus":{"type":"integer","minimum":1,"default":8},"nodes":{"type":"integer","const":1,"default":1},"memory_gb":{"type":"integer","minimum":1}}},
            "walltime_seconds":{"type":"integer","minimum":1,"maximum":3600000},"seed":{"type":"integer"},"wandb":{"type":"object","additionalProperties":false,"properties":{"enabled":{"type":"boolean","default":true},"preparation_enabled":{"type":"boolean","default":false},"entity":{"type":["string","null"]},"project":{"type":["string","null"]}}},"data":{"type":"array","maxItems":32,"items":{"type":"object","additionalProperties":false,"required":["name","asset_id"],"properties":{"name":{"type":"string"},"asset_id":{"type":"string"}}}},"parameters":{"type":"object"},"evaluation":{"type":"object"},"metrics_schema":{"type":"object"},"artifacts":{"type":"object","additionalProperties":false,"properties":{"required_files":{"type":"array","items":{"type":"string"},"maxItems":20000}}}}}),vec![parameter("project")]));
    paths.insert("/api/runs/{project}/{run}/submissions".into(),post("Prepare the first authored Run/Attempt with trusted source and materialized-image receipts; never submits a scheduler job",json!({"type":"object","additionalProperties":false,"properties":{"max_gpu_hours":{"type":"number","default":0},"reason":{"type":"string","default":""}}}),run()));
    for (name, description, body) in [
        (
            "history",
            "Ingest one bounded metrics page into the canonical local journal; persist events before advancing the exact Attempt cursor; never submits",
            json!({"type":"object","additionalProperties":false}),
        ),
        (
            "retry",
            "Append a new Attempt from an explicit checkpoint or full restart after exact terminal scheduler evidence; never submits",
            json!({"type":"object","additionalProperties":false,"oneOf":[{"required":["resume_from"]},{"required":["restart_from"]}],"properties":{"confirmation":{"type":"string","default":""},"max_gpu_hours":{"type":"number","default":0},"reason":{"type":"string"},"restart_from":{"type":"object","additionalProperties":false,"required":["run_id","attempt_id"],"properties":{"run_id":{"type":"string"},"attempt_id":{"type":"string"}}},"resume_from":{"type":"object","additionalProperties":false,"required":["run_id","attempt_id","checkpoint_id"],"properties":{"run_id":{"type":"string"},"attempt_id":{"type":"string"},"checkpoint_id":{"type":"string"}}}}}),
        ),
        (
            "cancel",
            "Persist an exact owned-job cancellation intent and reconcile the helper cancel outbox; terminal observation is required for termination verification",
            json!({"type":"object","additionalProperties":false,"required":["confirmation"],"properties":{"confirmation":{"type":"string","pattern":"^CANCEL action-[a-f0-9]{16}$"}}}),
        ),
        (
            "execute",
            "Execute a ready prepared native Action once, recording an optional reason; response loss requires same-identity reconciliation",
            json!({"type":"object","additionalProperties":false,"required":["confirmation"],"properties":{"confirmation":{"type":"string","pattern":"^EXECUTE action-[a-f0-9]{16}$"},"reason":{"type":"string","maxLength":4000,"default":""}}}),
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
        let mut operation = post(description, body, vec![parameter("submission")]);
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
        if matches!(name, "collect" | "acknowledge") {
            operation["post"]["responses"]["200"]["description"] = json!(
                "Original Submission and its result state; cloud collection may have submitted a synchronization job, and an acknowledgement with release may have deleted the generated archive"
            );
            operation["post"]["responses"]["500"]["description"] = json!(
                "Outcome is uncertain: a synchronization job or an archive release may have happened; query the original Submission and reconcile; do not repeat execution"
            );
            operation["post"]["responses"]["503"] = json!({"description":"Result evidence is unavailable or the response is uncertain; original outputs and checkpoints are retained; query the original Submission before repeating"});
        }
        if matches!(name, "refresh" | "checkpoint") {
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
    native_contract::extend(&mut document);
    document
}
async fn schema() -> Json<serde_json::Value> {
    Json(schema_document())
}
async fn unavailable(request: Request) -> Response {
    if request.uri().path().starts_with("/api/") {
        (StatusCode::NOT_IMPLEMENTED, Json(json!({"code":"ENDPOINT_NOT_IMPLEMENTED", "detail":"this server does not implement the endpoint"}))).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

pub fn app(auth: Auth, api: Option<Arc<api::Api>>) -> Router {
    let state = Arc::new(AppState {
        auth,
        api,
        effect_slots: Arc::new(tokio::sync::Semaphore::new(4)),
        data_slots: Arc::new(tokio::sync::Semaphore::new(4)),
        download_slots: Arc::new(tokio::sync::Semaphore::new(16)),
    });
    Router::new()
        .route("/api/health", get(health))
        .route("/api/readiness", get(readiness))
        .route("/api/v2/openapi.json", get(schema))
        .merge(read_http::routes())
        .merge(projects::http::routes())
        .merge(sources::http::routes())
        .merge(submission_http::routes())
        .merge(dataset_http::routes())
        .merge(backends::cloud::service::routes())
        .merge(backends::cloud::service::build_routes())
        .merge(backends::cloud::service::debug_routes())
        .merge(administration_http::routes())
        .merge(preparation_http::routes())
        .merge(local_http::routes())
        .merge(capacity_http::routes())
        .merge(data_delivery_http::routes())
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
    async fn boundary_requires_auth_and_never_claims_training() {
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
        assert!(data["capabilities"].is_array());
        for (path, status) in [("/api/readiness", 200), ("/api/actions/execute", 501)] {
            let response = make()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .header("authorization", format!("Bearer {TOKEN}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), status);
        }
    }
    /// A server with no object store behind an executor says so in the one code a
    /// client takes to mean "send through the server", and only to an authorized caller.
    #[tokio::test]
    async fn direct_upload_routes_are_authorized_and_refuse_an_executor_with_no_object_store() {
        let (_temp, state) = http_effect::tests::state();
        let asset = format!("asset.{}", "a".repeat(64));
        for route in ["upload-ticket", "part-uploaded"] {
            let uri = format!("/api/projects/demo/datasets/slurm/{asset}/{route}");
            let body = json!({"part":0,"sha256":"b".repeat(64)}).to_string();
            for (token, status, code) in [
                (None, 401, None),
                (Some("wrong"), 401, None),
                (Some(TOKEN), 422, Some("DATASET_DIRECT_UPLOAD_UNSUPPORTED")),
            ] {
                let mut request = Request::builder().method(Method::POST).uri(&uri);
                if let Some(token) = token {
                    request = request.header("authorization", format!("Bearer {token}"));
                }
                let response = app(
                    Auth {
                        token: TOKEN.into(),
                    },
                    state.api.clone(),
                )
                .oneshot(request.body(Body::from(body.clone())).unwrap())
                .await
                .unwrap();
                assert_eq!(response.status().as_u16(), status, "{route} {token:?}");
                if let Some(code) = code {
                    assert_eq!(response.headers()["x-ml-expd-error-code"], code);
                }
            }
        }
    }
    #[tokio::test]
    async fn reads_are_authenticated_and_other_methods_are_refused() {
        for (method, authorization, expected) in [
            (Method::GET, false, StatusCode::UNAUTHORIZED),
            (Method::GET, true, StatusCode::OK),
            (Method::POST, true, StatusCode::METHOD_NOT_ALLOWED),
        ] {
            let mut request = Request::builder().method(method).uri("/api/readiness");
            if authorization {
                request = request.header("authorization", format!("Bearer {TOKEN}"));
            }
            let response = app(
                Auth {
                    token: TOKEN.into(),
                },
                None,
            )
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap();
            assert_eq!(response.status(), expected);
        }
    }
    #[tokio::test]
    async fn duplicate_authorization_headers_are_rejected() {
        let mut request = Request::builder()
            .uri("/api/readiness")
            .header("authorization", format!("Bearer {TOKEN}"))
            .body(Body::empty())
            .unwrap();
        let value = request.headers().get("authorization").unwrap().clone();
        request.headers_mut().append("authorization", value);
        let response = app(
            Auth {
                token: TOKEN.into(),
            },
            None,
        )
        .oneshot(request)
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
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
    fn only_observations_promise_that_a_failed_response_had_no_effect() {
        let schema = super::schema_document();
        let responses = |name: &str| {
            schema["paths"][format!("/api/submissions/{{identity}}/{name}")]["post"]["responses"]
                .clone()
        };
        for name in ["collect", "acknowledge"] {
            let text = responses(name).to_string();
            assert!(text.contains("do not repeat execution"), "{name}");
            assert!(!text.contains("no scheduler effect"), "{name}");
            assert!(!text.contains("no Action or scheduler mutation"), "{name}");
        }
        for name in ["refresh", "checkpoint"] {
            assert_eq!(
                responses(name)["500"]["description"],
                "Observation response failed; no scheduler effect was requested"
            );
        }
    }
    #[test]
    fn native_contract_declares_narrow_lifecycle_posts_and_exact_read_identity() {
        let schema = super::schema_document();
        let paths = schema["paths"].as_object().unwrap();
        for (path, operations) in paths {
            if matches!(
                path.as_str(),
                "/api/debug"
                    | "/api/projects/{project}/runtimes"
                    | "/api/projects/{project}/datasets/{executor}"
            ) || path.ends_with("/diagnostic")
            {
                assert_eq!(operations.as_object().unwrap().len(), 2);
                assert!(operations.get("get").is_some());
                assert!(operations.get("post").is_some());
            } else if path.contains("/datasets/") && path.contains("/parts/") {
                assert_eq!(operations.as_object().unwrap().len(), 1);
                assert!(operations.get("put").is_some());
            } else if path.starts_with("/api/project-lifecycle/")
                || path.starts_with("/api/admin/") && !path.ends_with("/maintenance/{identity}")
                || path.starts_with("/api/worker-exchange/")
                    && !path.ends_with("/status")
                    && !path.ends_with("/package")
                    && !path.ends_with("/code")
                    && !path.ends_with("/code-content")
                    && !path.ends_with("/worker")
                || path.starts_with("/api/dataset-exchange/")
                || path.ends_with("/refresh")
                || path == "/api/debug/{identity}/stop"
                || path == "/api/debug/{identity}/delete"
                || path.ends_with("/experiment-preparations")
                || path.ends_with("/continue")
                || path.ends_with("/publish")
                || path.ends_with("/download-ticket")
                || path.ends_with("/upload-ticket")
                || path.ends_with("/part-uploaded")
                || path.ends_with("/import-oss")
                || path.contains("/runtimes/")
                    && (path.ends_with("/execute")
                        || path.ends_with("/reconcile")
                        || path.ends_with("/materialize"))
                || path.ends_with("/submissions") && !path.starts_with("/api/experiments/")
                || path.ends_with("/datasets/{executor}/{asset}/complete")
                || path == "/api/projects/{project}/runs"
                || path == "/api/projects/{project}/local-runs"
                || path == "/api/local-runs/{project}/{run}/metrics"
                || path.starts_with("/api/submissions/")
                    && path != "/api/submissions/{identity}"
                    && path != "/api/submissions/{identity}/progress"
                    && !path.contains("/artifacts")
                    && path != "/api/submissions/{action}/sync/{request}"
            {
                assert_eq!(operations.as_object().unwrap().len(), 1);
                assert!(operations.get("post").is_some());
            } else {
                assert_eq!(operations.as_object().unwrap().len(), 1);
                assert!(operations.get("get").is_some());
            }
            if path.starts_with("/api/worker-exchange/") {
                assert_eq!(
                    operations[if path.ends_with("/status")
                        || path.ends_with("/package")
                        || path.ends_with("/code")
                        || path.ends_with("/code-content")
                        || path.ends_with("/worker")
                    {
                        "get"
                    } else {
                        "post"
                    }]["security"],
                    serde_json::json!([{"attemptCapability":[]}])
                );
            } else if path.starts_with("/api/dataset-exchange/") {
                assert_eq!(
                    operations["post"]["security"],
                    serde_json::json!([{"datasetCapability":[]}])
                );
            }
        }
        for path in [
            "/api/executors",
            "/api/backends/{executor}/capacity",
            "/api/projects/{project}/experiment-preparations",
            "/api/projects/{project}/runtimes",
            "/api/debug",
            "/api/admin/maintenance/{identity}",
            "/api/worker-exchange/{project}/{run}/{attempt}",
            "/api/projects/{project}/local-runs",
            "/api/local-runs/{project}/{run}",
            "/api/local-runs/{project}/{run}/metrics",
        ] {
            assert!(paths.contains_key(path), "missing native contract: {path}");
        }
        assert!(paths.contains_key("/api/attempts/{project}/{identity}/metrics"));
        assert!(paths.contains_key("/api/runs/{project}/{run}/attempts/{attempt}/collection"));
        // A reference to a removed component would leave the contract unresolvable.
        assert!(!schema.to_string().contains("parameters/protocol"));
        let health = super::health_document(1, "isolated", true, false, None, None);
        assert!(
            !health["capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "project-lifecycle.v1")
        );
        // Checkpoint GETs need a submission or cloud service behind them.
        assert!(
            !health["capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "checkpoint-record-read.v1")
        );
        let lifecycle = super::health_document(1, "isolated", true, true, None, None);
        assert!(
            lifecycle["capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "project-lifecycle.v1")
        );
        assert_eq!(lifecycle["scheduler_mutations"], false);
    }
}
