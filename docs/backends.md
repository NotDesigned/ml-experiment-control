# Executor capabilities and server preparation

A client uploads code/data and describes a Run. The server matches an executor,
prepares its immutable Runtime and required backend data, freezes the Run and
prepares submission checks. The client needs no Docker, SSH or cloud credentials.
GPU execution still requires an explicit submission operation. A GPU-hour budget
can be supplied when preparing a submission.

## What executors declare

`GET /api/executors` includes a nested `capabilities` contract:

| Group | Fields | Meaning |
|---|---|---|
| runtime | platforms, materialization | Supported target platform and OCI/OCI-to-SIF path |
| resources | allocation_mode, inventory_query | Flexible requests or a fixed worker spec; whether this contract implements live inventory |
| storage | persistent, storage_id, checkpoint_registration | Configured shared filesystem and metadata registration |
| data | assets, asset_preparation, download_script, preparation | Asset delivery before/inside a job; optional preparation script before/inside a job |
| observability | logs, queue_reason, exit_code, preemption_reason | Evidence the adapter can actually retrieve |
| jobs | exact_submission_lookup, walltime_enforcement | Exact request reconciliation and which layer enforces duration |
| outputs | transfer, offline_export | Adapter result transfer and a generic adapter offline-export hook |

This is a configured contract, not a health probe or a promise of free GPUs.
`storage_id` fingerprints the configured namespace; it does not independently
prove two mounts share a filesystem. Checkpoint matching additionally checks the
registered project's storage scope. A Runtime image keeps the worker it was built
with, so a server upgrade does not change what it does.

PAI DLC is an opt-in single-node adapter; see [PAI configuration and limits](pai.md).
It supports quota queries, approved persistent Dataset mounts and worker
transfers. Uploaded assets require configured personal OSS storage plus CPFS
caching; an optional CPU collector supports data preparation and result sync.
Native OSS filesystem mounts
are provider-blocked and stay disabled.

The existing adapters declare:

| Detail | Slurm | PAI DLC |
|---|---|---|
| Runtime | OCI → Apptainer SIF | OCI |
| Allocation | Flexible, bounded by configured capacity | Flexible within configured quota/capacity; single node, 1–8 GPUs |
| Uploaded input | Parts over SSH to shared storage | Approved OSS asset → zero-GPU CPU preparation or inside-job CPFS cache |
| Script download | Gateway before allocation when `script_preparation` is on | Inside allocated job |
| Queue/exit/preemption reasons | Supported | Queue blockers when available; exit/preemption not guaranteed |
| Walltime | Scheduler and worker | Scheduler and worker; 24-hour cap |
| Server resource status | Slurm partition allocation | Quota allocation snapshot |
| Generic adapter offline-export hook | Not implemented | Not implemented |

SenseCore support was removed. A cloud config that still lists a SenseCore backend,
collector or debug profile is refused at load, and the deployment check reports
`cloud_backend_retired`. Runs created on SenseCore stay readable from the registry,
like any Run whose executor is no longer configured.

The server separately collects results to recover outputs from an exact original
Attempt. It publishes Slurm shared-directory outputs or
uses a bounded zero-GPU PAI CPU job to read CPFS and resume the original
archive transfer. This server operation does not require the generic adapter
`outputs.offline_export` hook. Query or start it with `ml-exp run collect`; see
[result recovery](recovery.md). It preserves the original scheduler outcome and
does not rerun training or prepare input data.

Native cloud Runs use `ml-exp run sync` with the original Submission and a saved
request ID to deliver retained logs, metrics, checkpoints and results. The
optional PAI collector reads the same personal CPFS outputs with no GPUs. An
explicit, bounded retry can recover a confirmed failed CPU sync without running
training; see [CPU sync recovery](recovery.md#native-cpu-sync-recovery).

A separate resource endpoint may expose backend quota/catalogue data. That is
not an admission guarantee. Queue ETA remains null without reliable backend data.

### Slurm allocation status

`GET /api/backends/{executor}/capacity` reports a cached partition snapshot for Slurm;
`POST .../capacity/refresh` updates it.
`resources` contains `gpus`, `cpus`, and `memory_bytes`, each with `total`,
`allocated`, and `unallocated`. Memory uses Slurm MiB converted to bytes.
`nodes` includes each unique node's state and allocation; normal/debug partitions
sharing nodes are separate views of the same resources, and must not be added together.
`pending_queue_entries` counts visible pending `squeue` rows (an array may be
compressed into one row); it exposes no users, job names or commands.

The server advertises `resources.inventory_query=true` for Slurm. This server
endpoint does not change the generic core adapter's capabilities or executor
matching: configured limits remain distinct from observed allocations.
Snapshots are cached for 25 seconds with one bounded SSH observation at a time.
`observed_at` is the last successful sample and `checked_at` the last completed
query. On failure, prior good data is explicitly `STALE`; without prior evidence
the response is `UNKNOWN` and resource/queue counts are null. A concurrent query
can return `UNKNOWN` / `QUERY_IN_PROGRESS`. Unsupported backends return
`UNSUPPORTED`; unknown executor IDs return 404. Authentication is unchanged.

`unallocated` is arithmetic scheduler allocation headroom, including down/drained
nodes; it is not physical GPU utilization or immediately usable capacity.
Reservations, priorities, node flags and resource combinations can prevent
allocation. `admission_guaranteed` is always false; no ETA, submission or provider
mutation is performed by this GET.

## Matching

The experiment file names an explicit `executor` from `ml-exp backend`. The native
API has no selector or matching route. It never silently switches clouds or ranks
unknown free capacity.

## Server-owned preparation

`POST /api/projects/{project}/experiment-preparations` accepts:

```json
{
  "request_id": "32_HEX_REQUEST_ID",
  "runtime": {"source_id": "source.64_HEX", "executor": "slurm-gpu", "base_image": "registry.example/base@sha256:64_HEX", "dockerfile": "Dockerfile"},
  "run": {
    "run_id": "trial-001", "executor": "slurm-gpu", "command": ["python3", "train.py"],
    "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32}, "walltime_seconds": 600, "seed": 42
  },
  "approval": {"max_gpu_hours": 0.2, "reason": "bounded validation"}
}
```

Replace placeholder identities with API responses. Instead of `runtime`, supply
`run.runtime_id` for an existing image. Uploaded assets use `run.inputs`; selection
and requirements are sibling fields. The API returns 202 and a durable preparation
ID. GET the same resource or its `/progress` endpoint to observe:
`ACCEPTED → BUILDING_RUNTIME → PREPARING_DATA → FREEZING_RUN → PREPARING_DATA_SCRIPT → CHECKING_SUBMISSION`.
The script stage is optional; progress includes its exact operation/job identity and safe diagnostic.
A completed preparation returns the frozen Run and submission ID. `READY` denotes
completed preparation; check `submission.ready` before authorizing GPU execution.

One project/run ID binds one request hash. Repeating an identical POST observes
the original operation; a changed intent returns 409. Concurrent preparations
share a Runtime build and data receipt. Effect markers are persisted before each
request. An interruption or uncertain dependency becomes `RECONCILE_REQUIRED`;
restarting the daemon never replays a build, copy or scheduler request.

Inspect saved dependency IDs first. Once they are independently READY, an explicit
POST to `/experiment-preparations/{id}/continue` can advance the original request.
It retains all effect markers and never retries an uncertain effect. Client:

```bash
ml-exp run continue PREPARATION_ID --project my-study --confirmation 'PREPARE PREPARATION_ID'
```

Low-level source, Runtime, Run and submission APIs remain available.

## Data delivery diagnosis

For a cloud Attempt, `GET /api/submissions/{id}/progress` includes `input_delivery`
from bounded, exact-Attempt worker logs. When a job failed before any worker report,
use `ml-exp run diagnose` to read the provider's reason, events and logs. If the cloud
worker's launch itself failed, it sent the server its error line; progress carries it as
`startup_failure`, and `ml-exp status` and `ml-exp run diagnose` print it as
`Startup failure: ...` even when the provider kept no log. The worker reports
waiting for the cache lock, downloading, extracting, verifying, publishing,
mounting and READY/FAILED. Evidence includes HTTP status, received/expected bytes,
elapsed time, observed throughput, timestamp, failed phase and a safe error code.
This is observed log evidence, not a scheduler heartbeat or execution authority.
Logs are collected on polling; timestamps are not refreshed merely by API reads.

URLs, tokens, response bodies, exception messages and private paths are excluded.
Examples distinguish timeout, TLS, HTTP/length rejection, truncated download,
archive/file SHA256 mismatch, archive-manifest mismatch and storage/network errors
(with errno when available). Training does not start after delivery failure.

The server cannot reconstruct an exception that was never recorded, and it does not
resubmit a failed GPU job automatically.

### Slurm shared-data staging

The API holds each upload part in memory while it forwards it to a pinned Slurm
receiver over SSH, and writes no part to its own disk. It handles four dataset
requests at once, each at most 16 MiB; a further one gets 429
`DATASET_TRANSFER_BUSY`. The whole archive and expanded files live only on shared
storage.
The backend validates archive/file hashes and atomically publishes the cache.
A bounded read-only check permits reuse before GPU submission. GPU nodes need no
external data access. New frozen inputs require the verified cache and fail closed
if it disappears. Upload data to a configured backend.

### Slurm compute nodes without Internet access

Slurm compute nodes need no route to the API or the Internet, so there is no relay to
configure. The worker writes metrics, checkpoints and results to shared storage, and
the server reads them over SSH through the helper on the gateway.

Set `script_preparation` on an executor to run download scripts in the same SIF on the
SSH gateway, with no GPU allocation, during submission staging. An executor without it
refuses a Run that has a download script. Script/parameter identity, SHA256 file
inventory, cache lock and atomic READY receipt are the same as elsewhere.
The allocated worker only verifies the sealed cache; missing or corrupt data
fails before training rather than downloading on an offline GPU node. The script
retains its own timeout; a 60-second launcher allowance covers container startup
and teardown. Staging hashes the cache once. Before scheduler submission the helper
checks only the sealed file set and sizes, and the allocated worker's own hash proves
the content. An uncertain stage remains subject to the existing explicit
reconciliation rules.


## Internal boundary for another backend

There is no plugin mechanism or backend registry. Slurm lives in the server's
`backends/slurm`. PAI lives in `backends/cloud`: the `Profile` enum in
`backends/cloud/adapter.rs` selects the provider, and its request compiler lives in
`core/src/cloud/`. Neither tree decides ownership: `runs/` chooses the backend for a Run
or Action and calls only its service, and the handlers that serve either backend
(`submission_http.rs`, `dataset_http.rs`, `preparation_http.rs`, `capacity_http.rs`) sit
above both trees. Data preparation
(`backends/cloud/service/engine/data_delivery.rs`, `materialization.rs`) and independent result
collection (`synchronization.rs`) are separate operations from training. Another cloud
provider means a new `Profile` variant and compiler, written against those modules.
