//! Documentation of the new native boundaries, kept beside the HTTP shells.
use serde_json::{Value, json};
fn object(required: &[&str], properties: Value) -> Value {
    json!({"type":"object","additionalProperties":false,"required":required,"properties":properties})
}
fn parameters(path: &str) -> Vec<Value> {
    let mut result = Vec::new();
    for name in path
        .split('/')
        .filter_map(|s| s.strip_prefix('{').and_then(|s| s.strip_suffix('}')))
    {
        result.push(json!({"in":"path","name":name,"required":true,"schema":{"type":"string"}}));
    }
    result
}
fn operation(path: &str, description: &str, body: Option<Value>) -> Value {
    let mut value = json!({"description":description,"parameters":parameters(path),"responses":{
        "200":{"description":"Bounded result for the exact identity; availability and diagnostic fields remain independent"},
        "401":{"description":"Authentication required"},"403":{"description":"Disabled by operator"},
        "404":{"description":"Identity not found"},"409":{"description":"Frozen identity, state or effect cannot be confirmed; inspect the original record"},
        "410":{"description":"Object released or administratively retired"},"422":{"description":"Invalid request"},
        "429":{"description":"Local capacity full"},
        "501":{"description":"Capability is not configured"},"503":{"description":"Effect or provider observation unavailable; original evidence retained"}}});
    if let Some(schema) = body {
        value["requestBody"] =
            json!({"required":true,"content":{"application/json":{"schema":schema}}});
    }
    if path.starts_with("/api/worker-exchange/") {
        value["security"] = json!([{"attemptCapability":[]}]);
    } else if path.starts_with("/api/dataset-exchange/") {
        value["security"] = json!([{"datasetCapability":[]}]);
    }
    value
}
pub(crate) fn extend(document: &mut Value) {
    document["components"]["securitySchemes"]["attemptCapability"] = json!({"type":"http","scheme":"bearer","description":"Exact Attempt capability; cannot authenticate ordinary client or administrative routes"});
    document["components"]["securitySchemes"]["datasetCapability"] = json!({"type":"http","scheme":"bearer","description":"Exact Project/executor/asset and download-or-READY capability; never the API token"});
    let scheduling = object(
        &[],
        json!({"allocation":{"type":"string","enum":["guaranteed","allow_idle","idle_only"],"default":"guaranteed"},"priority":{"type":"integer","minimum":1,"maximum":9,"default":5}}),
    );
    let mut run=document["paths"]["/api/projects/{project}/runs"]["post"]["requestBody"]["content"]["application/json"]["schema"].clone();
    run["properties"]["scheduling"] = scheduling;
    run["properties"]["env"] = json!({"type":"object","additionalProperties":{"type":"string"},"description":"Frozen per-Run environment. Platform directory, identity and ML_EXPD_ variables are reserved. Store credentials separately."});
    run["properties"]["code_source_id"] = json!({"type":"string","description":"Optional, on a cloud executor. Code delivered to each Attempt and used as its working directory, instead of the code in the image. At most 64 MiB; not with data_preparation."});
    run["properties"]["resume_from"] = object(
        &["run_id", "attempt_id", "checkpoint_id"],
        json!({"run_id":{"type":"string"},"attempt_id":{"type":"string"},"checkpoint_id":{"type":"string"}}),
    );
    document["paths"]["/api/projects/{project}/runs"]["post"]["requestBody"]["content"]["application/json"]
        ["schema"] = run.clone();
    // Inline runtime definitions supply runtime_id during resolution; the
    // ordinary standalone Run endpoint still requires it explicitly.
    run["required"]
        .as_array_mut()
        .expect("Run required fields")
        .retain(|v| v != "runtime_id");
    let definition = object(
        &["source_id", "executor", "base_image", "dockerfile"],
        json!({
        "source_id":{"type":"string"},"executor":{"type":"string"},"base_image":{"type":"string","description":"Digest-pinned base"},"dockerfile":{"type":"string"}}),
    );
    let confirmation = object(&["confirmation"], json!({"confirmation":{"type":"string"}}));
    let empty = object(&[], json!({}));
    let approval = object(
        &[],
        json!({"max_gpu_hours":{"type":"number","default":0,"description":"Optional cost metadata; not a submission limit"},"reason":{"type":"string","default":""}}),
    );
    let paths = document["paths"].as_object_mut().expect("OpenAPI paths");
    let mut insert = |path: &str, method: &str, description: &str, body: Option<Value>| {
        let row = paths.entry(path.to_owned()).or_insert_with(|| json!({}));
        row[method] = operation(path, description, body);
    };
    insert(
        "/api/submissions/{identity}/progress",
        "get",
        "Read retained worker phase and input delivery evidence for one Submission; no provider query. A Scheduler state or absent report does not prove training started.",
        None,
    );
    insert(
        "/api/experiments/{project}/{run}/submissions",
        "get",
        "List the current Attempt's Submission for clients without saved state; no provider query.",
        None,
    );
    insert(
        "/api/worker-exchange/{project}/{run}/{attempt}/package",
        "get",
        "Read the same frozen launch JSON using its existing exact Attempt capability; no provider query or state mutation.",
        None,
    );
    insert(
        "/api/worker-exchange/{project}/{run}/{attempt}/code",
        "get",
        "List the code source frozen in this Attempt, with each file's size and SHA256, using its exact Attempt capability; no state mutation.",
        None,
    );
    insert(
        "/api/worker-exchange/{project}/{run}/{attempt}/code-content",
        "get",
        "Read the files of the code source frozen in this Attempt, one after another in the order of its code listing, using its exact Attempt capability; no state mutation.",
        None,
    );
    insert(
        "/api/worker-exchange/{project}/{run}/{attempt}/worker",
        "get",
        "Read the worker program frozen in this Attempt, whose hash its package names, using its exact Attempt capability; no state mutation.",
        None,
    );
    insert(
        "/api/worker-exchange/{project}/{run}/{attempt}/status",
        "get",
        "Read only accepted exchange/artifact stage booleans using this exact Attempt capability; never refresh or write.",
        None,
    );
    for path in [
        "/api/submissions/{action}/sync/{request}/diagnostic",
        "/api/projects/{project}/datasets/{executor}/{asset}/publications/{request}/diagnostic",
    ] {
        insert(
            path,
            "get",
            "Read retained diagnostic evidence only; no provider query or job creation.",
            None,
        );
        insert(
            path,
            "post",
            "Create or observe one persisted 0GPU/2CPU/4GiB/300s diagnostic under the original lineage, CPFS mounted read-only. No worker replay, archive publication or automatic recovery.",
            Some(empty.clone()),
        );
    }
    insert(
        "/api/runs/{project}/{run}/attempts/{attempt}/provider-diagnostics/refresh",
        "post",
        "Explicitly ask the provider for one Attempt's reason, pods, events and bounded log tail (optional lines=1..2000). Read-only at the provider; nothing is stored.",
        Some(empty.clone()),
    );
    insert(
        "/api/projects/{project}/datasets/{executor}/{asset}/publications/{request}/provider-diagnostics/refresh",
        "post",
        "Explicitly ask the provider for the reason, pods, events and bounded log tail (optional lines=1..2000) of one data copy and its read-only diagnostic. Read-only at the provider; nothing is stored.",
        Some(empty.clone()),
    );
    insert(
        "/api/submissions/{identity}/renew",
        "post",
        "Explicitly replace an expired never-started gate under one durable request ID. Preserve Run/Attempt, resources, budget and scheduler token; return a PREPARED replacement with a fresh gate. Possible prior effects block renewal.",
        Some(object(
            &["request_id", "confirmation"],
            json!({"request_id":{"type":"string"},"confirmation":{"type":"string"}}),
        )),
    );
    insert(
        "/api/projects/{project}/datasets/{executor}",
        "get",
        "List retained immutable dataset identities and upload/cache readiness; no payload read or provider request. ARCHIVED means uploaded bytes are verified, not that the backend cache is READY.",
        None,
    );
    insert(
        "/api/executors",
        "get",
        "Configured backend capabilities; no credentials, private paths or implicit provider query",
        None,
    );
    insert(
        "/api/backends/{executor}/capacity",
        "get",
        "Cached capacity with original observation time; UNKNOWN or STALE is not a free-GPU guarantee",
        None,
    );
    insert(
        "/api/backends/{executor}/capacity/refresh",
        "post",
        "Read provider allocation/quota evidence into the cache; no scheduler mutation",
        Some(empty.clone()),
    );
    insert(
        "/api/projects/{project}/runtimes",
        "get",
        "List native Runtime records belonging to this Project",
        None,
    );
    insert(
        "/api/projects/{project}/runtimes",
        "post",
        "Freeze a native Dockerfile recipe; does not start a build",
        Some(definition.clone()),
    );
    insert(
        "/api/projects/{project}/runtimes/{runtime}",
        "get",
        "Read immutable recipe and retained build state",
        None,
    );
    for action in ["execute", "reconcile", "materialize"] {
        insert(
            &format!("/api/projects/{{project}}/runtimes/{{runtime}}/{action}"),
            "post",
            "Explicit native image operation; reconcile observes the original publication without a second builder",
            Some(if action == "execute" {
                confirmation.clone()
            } else {
                empty.clone()
            }),
        );
    }
    for view in ["logs", "progress"] {
        insert(
            &format!("/api/projects/{{project}}/runtimes/{{runtime}}/{view}"),
            "get",
            "Bounded redacted build evidence",
            None,
        );
    }
    insert(
        "/api/projects/{project}/experiment-preparations",
        "post",
        "Freeze one experiment configuration and deterministic recovery slot; no training authorization",
        Some(object(
            &["request_id", "run"],
            json!({"request_id":{"type":"string"},"run":run,"runtime":{"anyOf":[definition,{"type":"null"}]},"approval":approval}),
        )),
    );
    insert(
        "/api/projects/{project}/experiment-preparations/{identity}",
        "get",
        "Read retained preparation evidence; does not advance preparation",
        None,
    );
    insert(
        "/api/projects/{project}/experiment-preparations/{identity}/continue",
        "post",
        "Advance original environment/data preparation and prepare an ordinary Submission; does not authorize training",
        Some(confirmation.clone()),
    );
    insert(
        "/api/projects/{project}/datasets/{executor}/{asset}/download-ticket",
        "post",
        "Obtain a short-lived direct object part URL; full verification belongs to the client; no ACK or deletion",
        Some(object(
            &["part"],
            json!({"part":{"type":"integer","minimum":0}}),
        )),
    );
    let direct_part = object(
        &["part", "sha256"],
        json!({"part":{"type":"integer","minimum":0},"sha256":{"type":"string"}}),
    );
    insert(
        "/api/projects/{project}/datasets/{executor}/{asset}/upload-ticket",
        "post",
        "Obtain a short-lived signed PUT for one part, which cannot overwrite an object. 422 DATASET_DIRECT_UPLOAD_UNSUPPORTED means this executor has no object store.",
        Some(direct_part.clone()),
    );
    insert(
        "/api/projects/{project}/datasets/{executor}/{asset}/part-uploaded",
        "post",
        "Record a part sent directly to the object store once it exists at its exact size; complete proves its bytes",
        Some(direct_part),
    );
    insert(
        "/api/runs/{project}/{run}/preparation",
        "get",
        "Discover the retained experiment preparation before a Run exists; fall back to an existing submission. Multiple preparations require the saved exact ID. No provider or write operation.",
        None,
    );
    insert(
        "/api/submissions/{action}/sync/{request}",
        "get",
        "Read retained original/recovery CPU synchronization evidence only; no job creation, refresh or replay.",
        None,
    );
    insert(
        "/api/submissions/{action}/sync/{request}/retry",
        "post",
        "One explicit deterministic CPU recovery child under the original sync request; requires exact FAILED original job and terminal training evidence, retains all original ledgers. Repeated or uncertain replies observe the same child.",
        Some(object(
            &["expected_failed_job_id", "confirmation"],
            json!({"expected_failed_job_id":{"type":"string"},"confirmation":{"type":"string"}}),
        )),
    );
    insert(
        "/api/projects/{project}/datasets/{executor}/{asset}/publish",
        "post",
        "Prepare data on backend storage before training. PAI uses an existing READY runtime_id, 0GPU/2CPU/4GiB/3600s, verified OSS parts and personal CPFS; SenseCore retains its data-image/debug CPU copy. Never reports READY without a full verified receipt.",
        Some(object(
            &["request_id", "confirmation"],
            json!({"request_id":{"type":"string"},"confirmation":{"type":"string"},"runtime_id":{"type":["string","null"]}}),
        )),
    );
    insert(
        "/api/debug",
        "get",
        "Configured DSW/CCI profiles and retained machine state; CCI supports create and stop only",
        None,
    );
    insert(
        "/api/debug",
        "post",
        "Create one owned SSH test machine with explicit bounded resources and lease",
        Some(object(
            &[
                "profile",
                "request_id",
                "ssh_public_key",
                "lease_seconds",
                "gpus",
                "cpus",
                "memory_gb",
            ],
            json!({
        "profile":{"type":"string"},"request_id":{"type":"string"},"ssh_public_key":{"type":"string","maxLength":8192},"lease_seconds":{"type":"integer","minimum":1,"description":"DSW maximum 14400 seconds; other profiles use provider limits"},"gpus":{"type":"integer","minimum":0},"cpus":{"type":"integer","minimum":1},"memory_gb":{"type":"integer","minimum":1}}),
        )),
    );
    insert(
        "/api/debug/{identity}",
        "get",
        "Retained machine and connection evidence; no provider operation",
        None,
    );
    for action in ["refresh", "stop", "delete"] {
        insert(
            &format!("/api/debug/{{identity}}/{action}"),
            "post",
            "Observe or stop the original owned machine; delete is DSW-only",
            Some(if action == "refresh" {
                empty.clone()
            } else {
                confirmation.clone()
            }),
        );
    }
    insert(
        "/api/admin/submissions/{action}/maintenance",
        "post",
        "Prepare scoped deletion after terminal scheduler and checkpoint dependency checks; does not remove payloads",
        Some(object(
            &["request_id", "selection"],
            json!({"request_id":{"type":"string"},"selection":{"oneOf":[object(&["kind"],json!({"kind":{"const":"outputs"}})),object(&["kind"],json!({"kind":{"const":"state"}})),object(&["kind","checkpoint_id"],json!({"kind":{"const":"checkpoint"},"checkpoint_id":{"type":"string"}}))]}}),
        )),
    );
    insert(
        "/api/admin/maintenance/{identity}",
        "get",
        "Read deletion plan, receipt and durable object release cursor",
        None,
    );
    insert(
        "/api/admin/maintenance/{identity}/execute",
        "post",
        "Execute original reviewed plan; never treats administrative deletion as download verification",
        Some(confirmation.clone()),
    );
    insert(
        "/api/admin/runs/{project}/{run}/retire",
        "post",
        "Retire a terminal Run and keep small audit evidence. Removes no bytes and later plans are refused, so run scoped cleanup first",
        Some(confirmation),
    );
    insert(
        "/api/worker-exchange/{project}/{run}/{attempt}",
        "post",
        "Commit a bounded, ordered worker packet before ACK; repeated packets are idempotent and a conflicting sequence is rejected",
        Some(object(
            &[
                "contract",
                "sequence",
                "previous_sha256",
                "package_sha256",
                "evidence",
                "history",
                "checkpoint",
            ],
            json!({
                "contract":{"const":"native-cloud-exchange.v1"},"sequence":{"type":"integer","minimum":1},
                "previous_sha256":{"type":["string","null"]},"package_sha256":{"type":"string"},
                "evidence":{"type":"object"},"history":{"type":["object","null"]},"checkpoint":{"type":["object","null"]},
                "inputs":{"type":"array","items":{"type":"object"}},"checkpoint_error":{"enum":["CHECKPOINT_VERIFICATION_FAILED",null]}
            }),
        )),
    );
    insert(
        "/api/worker-exchange/{project}/{run}/{attempt}/datasets/{asset}/{part}",
        "post",
        "Return a short-lived URL for one frozen input part; no API payload spooling",
        Some(empty.clone()),
    );
    insert(
        "/api/worker-exchange/{project}/{run}/{attempt}/artifacts",
        "post",
        "Prepare verified result metadata, obtain a part upload ticket or verify complete publication; these are independent from training completion",
        Some(object(
            &["package_sha256", "operation"],
            json!({"package_sha256":{"type":"string"},"operation":{"oneOf":[
                object(&["kind","bundle"],json!({"kind":{"const":"prepare"},"bundle":{"type":"object"}})),
                object(&["kind","part"],json!({"kind":{"const":"ticket"},"part":{"type":"integer","minimum":0}})),
                object(&["kind"],json!({"kind":{"const":"complete"}}))
            ]}}),
        )),
    );
    insert(
        "/api/worker-exchange/{project}/{run}/{attempt}/maintenance",
        "post",
        "Receive the exact reviewed maintenance receipt; it cannot authorize another deletion plan",
        Some(object(
            &[
                "contract",
                "plan_sha256",
                "plan_id",
                "identity",
                "paths",
                "status",
            ],
            json!({
                "contract":{"const":"native-storage-maintenance.v1"},"plan_sha256":{"type":"string"},"plan_id":{"type":"string"},
                "identity":{"type":"object"},"paths":{"type":"array","items":{"type":"string"}},"status":{"const":"REMOVED"}
            }),
        )),
    );
    insert(
        "/api/dataset-exchange/{project}/{executor}/{asset}/{operation}",
        "post",
        "Data-only capability: operation is a canonical zero-based part number with an empty body, or ready with the complete verified dataset receipt",
        Some(json!({"type":"object"})),
    );
}
