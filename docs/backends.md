# Executor capabilities and server preparation

A client uploads code/data and describes a Run. The server matches an executor,
prepares its immutable Runtime and required backend data, freezes the Run and
prepares submission checks. The client needs no Docker, SSH or cloud credentials.
GPU execution still requires explicit `--execute` and a GPU-hour budget.

## What executors declare

`GET /api/executors` includes a nested `capabilities` contract:

| Group | Fields | Meaning |
|---|---|---|
| runtime | platforms, materialization | Supported target platform and OCI/OCI-to-SIF path |
| resources | allocation_mode, inventory_query | Flexible requests or a fixed worker spec; whether this contract implements live inventory |
| network | compute_internet, api_transport | Configured compute egress and direct HTTPS/private TCP relay |
| storage | persistent, storage_id, checkpoint_registration | Configured shared filesystem and metadata registration |
| data | assets, asset_preparation, download_script | Asset delivery before/inside a job; optional preparation script before/inside a job |
| observability | logs, queue_reason, exit_code, preemption_reason | Evidence the adapter can actually retrieve |
| jobs | exact_submission_lookup, walltime_enforcement | Exact request reconciliation and which layer enforces duration |
| outputs | transfer, offline_export | Adapter result transfer and a generic adapter offline-export hook |

This is a configured contract, not a health probe or a promise of free GPUs.
`storage_id` fingerprints the configured namespace; it does not independently
prove two mounts share a filesystem. Checkpoint matching additionally checks the
registered project's storage scope. Runtime worker features are validated
separately: old images do not gain new behavior from a server upgrade.

Current adapters declare:

| Detail | WYD Slurm | SenseCore ACP |
|---|---|---|
| Runtime | OCI → Apptainer SIF | OCI |
| Allocation | Flexible, bounded by configured catalogue capacity | Exact GPU count of fixed worker spec |
| Uploaded desktop asset | Stream by SSH/verify on datapool before the job | Bounded zero-GPU ACP copy into NAS before training |
| Script download | Before allocation on the connected gateway when relay is configured | Inside the allocated job |
| Queue/exit/preemption reasons | Supported | Not declared as reliable structured fields |
| Walltime | Scheduler and worker | Worker |
| Live resource inventory | Not implemented by this capability contract | Not implemented by this capability contract |
| Generic adapter offline-export hook | Not implemented | Not implemented |

The server separately advertises `result-collection.v1` for recovering outputs
from an exact original Attempt. It publishes WYD shared-directory outputs or
uses a bounded zero-GPU SenseCore CPU job to read NAS and resume the original
archive transfer. This server operation does not require the generic adapter
`outputs.offline_export` hook. Query or start it with `ml-exp collect`; see
[result recovery](recovery.md). It preserves the original scheduler outcome and
does not rerun training or prepare input data.

A separate resource endpoint may expose backend quota/catalogue data. That is
not an admission guarantee. Queue ETA remains null without reliable backend data.

## Matching

Use either an explicit `executor`, or an `executor_selector`:

```json
{
  "executor_selector": {"backend": "slurm", "candidates": ["wyd-h100", "wyd-l40s"]},
  "requirements": {"persistent_checkpoints": true, "exit_code": true}
}
```

Candidate order is preference. With only a backend, matching chooses the smallest
configured allocation that fits, then a stable ID. It never silently switches
clouds or ranks unknown free capacity. `POST /api/executors/match` validates before
uploads/builds and returns selected/compatible executors, rejection fields and
`live_availability_checked=false`. Unsupported requirements fail before work.

## Server-owned preparation

`POST /api/projects/{project}/experiment-preparations` accepts:

```json
{
  "runtime": {"source_id": "source.64_HEX", "dockerfile": "Dockerfile", "entrypoint": ["python3", "train.py"]},
  "run": {
    "run_id": "trial-001", "executor": "wyd-l40s",
    "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "max_time": "00:10:00"}
  },
  "max_gpu_hours": 0.2
}
```

Replace placeholder identities with API responses. Instead of `runtime`, supply
`run.runtime_id` for an existing image. Uploaded assets use `run.inputs`; selection
and requirements are sibling fields. The API returns 202 and a durable preparation
ID. GET the same resource or its `/progress` endpoint to observe:
`ACCEPTED → BUILDING_RUNTIME → PREPARING_DATA → FREEZING_RUN → CHECKING_SUBMISSION`.
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
ml-exp experiment experiment.json --state trial.state.json --resume --continue-preparation
```

Low-level source, Runtime, Run and submission APIs remain available. The new
one-config workflow requires `server-experiment-preparation.v1`; older clients
continue to use the existing low-level protocol-2 endpoints.

## Data delivery diagnosis

`GET /api/submissions/{id}/progress` includes `input_delivery` from bounded,
exact-Attempt worker logs. New images with `input-delivery-progress.v1` report
waiting for the cache lock, downloading, extracting, verifying, publishing,
mounting and READY/FAILED. Evidence includes HTTP status, received/expected bytes,
elapsed time, observed throughput, timestamp, failed phase and a safe error code.
This is observed log evidence, not a scheduler heartbeat or execution authority.
Logs are collected on polling; timestamps are not refreshed merely by API reads.

URLs, tokens, response bodies, exception messages and private paths are excluded.
Examples distinguish timeout, TLS, HTTP/length rejection, truncated download,
archive/file SHA256 mismatch, archive-manifest mismatch and storage/network errors
(with errno when available). Training does not start after delivery failure.

Historical images retain their old worker. A generic failure produces
`INPUT_DELIVERY_FAILED_DETAILS_UNAVAILABLE`; the server cannot reconstruct an
exception that was never recorded. Rebuild a Runtime for detailed diagnostics;
do not resubmit historical GPU jobs automatically.

### WYD shared-data staging

The server-owned workflow streams desktop-backed uploaded assets through bounded
API memory into `/datapool` over the configured WYD SSH connection. Archive and
expanded files are staged only on backend storage; GPU nodes need no external
HTTPS data access. The backend checks archive/file hashes and atomically publishes
the shared cache. A bounded read-only cache check permits reuse. Interrupted or
unacknowledged transfers require inspection; continuation only accepts a verified
cache and never repeats an uncertain transfer. New frozen inputs require the
cache and fail closed if it disappears. Old low-level Runs retain their original
worker download behavior.

### WYD compute nodes without Internet access

Configure `backend.api_relay` on WYD executors whose workers must use the L40S
private-network gateway. The gateway uses systemd's existing
`systemd-socket-proxyd` with one fixed API destination; example user units are in
`server/examples/wyd-api-relay/`. Bind the socket to the gateway's private IP,
install both units in its `~/.config/systemd/user/`, and enable the socket:
`systemctl --user enable --now ml-expd-api-relay.socket`. The user manager must
remain running after SSH disconnect (linger).

```yaml
compute_internet: false
backend:
  # Existing Slurm settings remain here.
  api_relay:
    endpoint: tcp://172.16.78.148:18443
    origin: https://api.example.org
```

The origin must match the configured artifact API host and port. The worker
connects TCP to the gateway while retaining origin TLS certificate verification,
SNI, Host and exact-Attempt authorization. No general HTTP/SOCKS proxy, TLS
termination or public listener is needed. Bootstrap, data transfers, checkpoint
registration and multipart result uploads share this transport. Worker requests
to other origins fail closed. Network declarations expose configured
`compute_internet` (null if unknown) and `api_transport`; these are not live
reachability claims and do not expose the private endpoint.

A relay-enabled executor requires a newly built Runtime declaring
`api-tcp-relay.v1`; old immutable images and Runs remain unchanged. Stage and
submission preflight verify the gateway's origin TLS path through the configured
SSH gateway before scheduler submission. Unreachable relay errors carry the
fixed `API_RELAY_UNREACHABLE` code; origin mismatch is
`API_RELAY_TARGET_MISMATCH`. TLS and fixed-destination upstream failures can be
investigated with the gateway user service journal.

For these executors, download scripts run in the same SIF on the SSH gateway,
with no GPU allocation, during submission staging. Existing script/parameter
identity, SHA256 file inventory, cache lock and atomic READY receipt are reused.
The allocated worker only verifies the sealed cache; missing or corrupt data
fails before training rather than downloading on an offline GPU node. The script
retains its own timeout; a 60-second launcher allowance covers container startup
and teardown. A cache verification also runs before scheduler submission. An
uncertain stage remains subject to the existing explicit reconciliation rules.
