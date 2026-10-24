# API resource and diagnostic reference

For a complete install-to-results example use the [client quickstart](api-quickstart.md).
This page is a lookup table for custom clients, Runtime reuse and diagnosis;
request/response bodies are defined by the installed `GET /api/v2/openapi.json`.
Paths are relative to the API base. `P`, `R`, `A`, `S` denote returned project,
Run, Attempt and Submission IDs, not filesystem paths.

Send the API Bearer credential and `X-ML-Expd-Client-Protocol: 2` on control
requests. `GET /api/health` is the protocol-discovery exception; see the
[HTTP contract](http_contract.md) for auth, errors and asynchronous capacity.

## Discover, import and build

| Method/path | Resource or purpose |
|---|---|
| `GET /api/health` | Installed version, protocol, capabilities and policy |
| `GET /api/executors` | Configured resource/backend choices |
| `GET /api/executors/EXECUTOR_ID/status` | Read-only resource evidence when supported |
| `GET /api/environments` | Approved digest-pinned base images |
| `GET /api/storage-limits` | Transfer/part/storage limits |
| `POST /api/source-imports/archive?project=P&sha256=HASH` | Raw source archive |
| `POST /api/source-imports/git` | HTTPS repository and exact commit |
| `GET /api/projects/P/sources/SOURCE_ID` | Immutable source identity |
| `POST /api/projects/P/runtimes/prepare` | Freeze source, Dockerfile and entrypoint |
| `POST /api/projects/P/runtimes/RUNTIME_ID/execute` | Build using returned confirmation |
| `GET /api/projects/P/runtimes/RUNTIME_ID` | Runtime state/receipt |
| `GET …/runtimes/RUNTIME_ID/logs` / `/progress` | Build diagnostics |
| `POST …/runtimes/RUNTIME_ID/reconcile` | Observe publication without rebuilding |

New builds use Dockerfile only, with pinned external bases. Old READY images
retain their workers; an obsolete unbuilt recipe requires new preparation.
Source admission rejects links, unsafe paths and credential-looking filenames;
it is not proof that file contents contain no secrets. See [builds](builds.md).

## Data and server preparation

| Method/path | Resource or purpose |
|---|---|
| `POST /api/projects/P/asset-uploads` | Target executor, archive hash/size and inventory |
| `GET /api/projects/P/asset-uploads/UPLOAD_ID` | Saved upload session |
| `GET /api/projects/P/assets/ASSET_ID` | Verified asset metadata |
| `POST /api/projects/P/experiment-preparations` | Combined Runtime/data/Run/Submission preparation |
| `GET /api/projects/P/experiment-preparations/PREPARATION_ID` | Dependencies, phase and next action |
| `POST …/experiment-preparations/PREPARATION_ID/continue` | Explicitly advance inspected dependencies |

Use the exact [multipart paths and resume rules](data.md#multipart-http-contract),
including zero-based part numbers. Completion establishes destination durability,
not GPU training. SenseCore data-image publication and NAS delivery are distinct.
Slurm/SenseCore target-backed assets do not provide client archive downloads;
approved OSS datasets have a separate signed-download route. Do not use output
routes to retrieve datasets. Legacy `POST /api/assets/archive` is removed.

## Freeze, review and submit

| Method/path | Resource or purpose |
|---|---|
| `POST /api/projects/P/runs` | Freeze a Run referencing a READY Runtime |
| `POST /api/experiments/P/R/submissions/prepare` | Preflight and GPU-hour budget review |
| `GET /api/experiments/P/R/submissions` | Find the original Submission after a lost reply |
| `POST /api/submissions/S/authorize` | Authorize current prepared gates |
| `POST /api/submissions/S/execute` | Submit using returned confirmation |
| `GET /api/submissions/S` / `/progress` | Durable state and next action |
| `POST /api/submissions/S/reconcile` | Resolve an uncertain original submission |
| `GET /api/operations` | Available generic operations and parameter schema |
| `POST /api/operations/direct` | Prepare a reviewed cancel/retry or other Action |

Prepare/build/execute may return HTTP 202: poll the returned identity. An expired
review requires reprepare, not bypassing gates. Unknown effects require
reconciliation, not resubmission. Retry creates a new Attempt of the same Run;
changing source, parameters or restore reference requires a new Run.

## Read evidence and retrieve results

| Method/path | Resource or purpose |
|---|---|
| `GET /api/runs/P/R` / `/attempts` | Run evidence and exact executions |
| `GET /api/attempts/P/R::A/logs` | Stored bounded stdout/stderr |
| `GET /api/attempts/P/R::A/metrics?format=records&name=loss` | Compact normalized [metrics](metrics.md); repeat `name` |
| `GET /api/runs/P/R/attempts/A/diagnostics` | Persisted provider evidence; `refresh=true` requests observation |
| `GET …/attempts/A/checkpoints` / `/checkpoints/CHECKPOINT_ID` | Registered checkpoint metadata |
| `GET …/attempts/A/collection` | Training, collection and verification states |
| `POST …/attempts/A/collection` | Explicit recovery; backend may use a bounded zero-GPU job |
| `GET …/attempts/A/files` | File inventory |
| `GET …/attempts/A/artifacts/download` | Signed archive ticket and verified inventory |
| `GET …/attempts/A/artifacts/archive` | Authenticated proxy archive stream |
| `POST …/attempts/A/artifacts/ack` | Confirm verified local files and choose archive retention |
| `GET …/attempts/A/files/outputs/PATH` | Exact file, with ranges and ETags |
| `GET /api/runs/P/R/tracking` | Frozen [W&B](wandb.md) destination and publication status |

The signed object GET carries no API token. Validate archive and file hashes,
then follow the ticket's acknowledgement contract. Signed URLs are temporary;
do not treat them as artifact identities. Download ACK can release the server
archive; it does not delete checkpoints or the metric journal.

A scheduler terminal state does not prove training/result success. Empty delayed
logs, missing files or a 404 are not proof that backend storage is empty.
Provider diagnostics are bound to the original job; refreshing submits no training.
For incomplete archives, repair and other recovery options see [recovery](recovery.md).
Internal launch/data/checkpoint/record transfer routes use narrow worker
capabilities, not the administrative client token. Scientific evaluation uses an
ordinary Run with checkpoint/data/protocol inputs, not a project-specific metric
or research-question prerequisite.
